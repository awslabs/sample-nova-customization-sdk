# Copyright Amazon.com, Inc. or its affiliates

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Glue Runtime Manager for data preparation pipelines on AWS Glue for Ray."""

from __future__ import annotations

import hashlib
import io
import json
import os
import textwrap
import time
import zipfile
from importlib import resources
from typing import Any, Dict, List, Optional, Sequence

import boto3
from botocore.exceptions import ClientError

from amzn_nova_forge.core.enums import Platform
from amzn_nova_forge.iam import create_sagemaker_invoke_role
from amzn_nova_forge.manager.runtime_manager import (
    DataPrepJobConfig,
    JobConfig,
    RuntimeManager,
)
from amzn_nova_forge.util.aws_utils import get_caller_account_id
from amzn_nova_forge.util.logging import logger
from amzn_nova_forge.util.s3_utils import (
    GLUE_ARTIFACT_PREFIX,
    ensure_bucket_exists,
    get_dataprep_bucket_name,
)

GLUE_IAM_ROLE_NAME = "GlueDataPrepExecutionRole"
_WHL_FILENAME = "agi_data_curator-1.0.0-py3-none-any.whl"
_WHL_SHA256 = "18d631f2a367dc744d34ac3b3e1cd8718a4fe3a859a79d63c29472169e7abba1"
_TERMINAL_STATES = {"SUCCEEDED", "FAILED", "STOPPED", "TIMEOUT", "ERROR"}

# Glue entry-point script, uploaded to S3 and executed inside the Glue Ray worker.
# Reads job parameters from environment variables and runs the requested
# ForgeWorkflows pipeline.
_GLUE_ENTRY_SCRIPT = textwrap.dedent("""\
    from __future__ import annotations

    import json
    import logging
    import os

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("invoke_glue_pipeline")


    def main() -> None:
        pipeline_id = os.environ.get("pipeline_id", "")
        input_path = os.environ.get("input_path", "")
        output_path = os.environ.get("output_path", "")
        input_format = os.environ.get("input_format", "parquet")
        output_format = os.environ.get("output_format", "parquet")
        text_field = os.environ.get("text_field", "text")
        extra_args_raw = os.environ.get("extra_args", "{}")

        try:
            extra_kwargs = json.loads(extra_args_raw)
        except json.JSONDecodeError:
            logger.warning("Could not parse extra_args as JSON, using empty dict")
            extra_kwargs = {}

        logger.info(
            "Glue pipeline invocation:\\n"
            "  pipeline_id:   %s\\n"
            "  input_path:    %s\\n"
            "  output_path:   %s\\n"
            "  input_format:  %s\\n"
            "  output_format: %s\\n"
            "  text_field:    %s\\n"
            "  extra_kwargs:  %s",
            pipeline_id, input_path, output_path,
            input_format, output_format, text_field, extra_kwargs,
        )

        from agi_data_curator.workflows.forge_workflows import ForgeWorkflows

        forge = ForgeWorkflows(
            input_path=input_path,
            output_path=output_path,
            input_format=input_format,
            output_format=output_format,
        )

        merged_kwargs = {"text_field": text_field, **extra_kwargs}
        metrics = forge.execute(pipeline_id, **merged_kwargs)

        logger.info(
            "Pipeline %r finished [%s] in %.1fs (%.1f min)",
            metrics["identifier"],
            metrics["status"],
            metrics["elapsed_seconds"],
            metrics["elapsed_minutes"],
        )

        if metrics["status"] == "error":
            error_msg = metrics.get("error", "unknown error")
            logger.error("Pipeline failed: %s", error_msg)
            raise RuntimeError(f"Pipeline {pipeline_id!r} failed: {error_msg}")

        result = metrics.get("result")
        if isinstance(result, dict):
            import boto3 as _boto3

            summary_key = output_path.rstrip("/") + "/_summary.json"
            _bucket, _key = summary_key[len("s3://"):].split("/", 1)
            try:
                _boto3.client("s3").put_object(
                    Bucket=_bucket,
                    Key=_key,
                    Body=json.dumps(result).encode("utf-8"),
                    ContentType="application/json",
                )
                logger.info("Wrote summary to %s", summary_key)
            except Exception:
                logger.warning("Failed to write summary to %s", summary_key, exc_info=True)


    if __name__ == "__main__":
        main()
""")


def _ensure_artifact_bucket(
    region: str,
    bucket: Optional[str] = None,
) -> str:
    """Return an S3 artifact bucket, creating a default one if needed."""
    if bucket:
        return bucket

    default_bucket = get_dataprep_bucket_name(region=region)
    ensure_bucket_exists(default_bucket, region=region)
    return default_bucket


class GlueRuntimeManager(RuntimeManager):
    """Runtime manager for AWS Glue for Ray data preparation jobs.

    Manages the lifecycle of Glue jobs: uploading artifacts (script + wheel)
    to S3, creating/updating Glue jobs, starting runs, polling for completion,
    and cleanup.

    Args:
        glue_role_name: IAM role name for the Glue job.
        s3_artifact_bucket: S3 bucket for script and wheel uploads.
            When ``None``, a default bucket is auto-created.
        s3_artifact_prefix: S3 key prefix for uploaded artifacts.
        worker_type: Glue worker type (e.g. ``"Z.2X"``).
        num_workers: Number of Glue workers.
        glue_version: Glue version string.
        region: AWS region. Defaults to session region.
        poll_interval: Seconds between status polls.
        max_wait_time: Maximum seconds to wait for a job to complete (default 3600).
        kms_key_id: Optional KMS key for S3 encryption.
        extra_py_modules: Optional list of local file paths to ``.whl`` or
            ``.zip`` files to upload and include as ``--s3-py-modules`` in the
            Glue job. When ``None`` (the default), the bundled
            ``agi_data_curator`` wheel is used. Pass an explicit list to
            override — an empty list means no extra modules are uploaded.
    """

    def __init__(
        self,
        glue_role_name: str = GLUE_IAM_ROLE_NAME,
        s3_artifact_bucket: Optional[str] = None,
        s3_artifact_prefix: str = GLUE_ARTIFACT_PREFIX,
        worker_type: str = "Z.2X",
        num_workers: int = 2,
        glue_version: str = "4.0",
        region: Optional[str] = None,
        poll_interval: int = 30,
        max_wait_time: int = 3600,
        kms_key_id: Optional[str] = None,
        extra_py_modules: Optional[Sequence[str]] = None,
    ):
        self.glue_role_name = glue_role_name
        self.s3_artifact_bucket = s3_artifact_bucket
        self.s3_artifact_prefix = s3_artifact_prefix.rstrip("/")
        self.worker_type = worker_type
        self.num_workers = num_workers
        self.glue_version = glue_version
        self._region = region
        self.poll_interval = poll_interval
        self.max_wait_time = max_wait_time
        self._extra_py_modules = extra_py_modules

        # RuntimeManager base: instance_type/count not applicable for Glue
        super().__init__(
            instance_type=None,
            instance_count=None,
            kms_key_id=kms_key_id,
        )
        self.setup()

    @property
    def platform(self) -> Platform:
        return Platform.GLUE

    @property
    def runtime_name(self) -> str:
        return "AWS Glue"

    @property
    def runtime_config(self) -> str:
        return (
            f"region={getattr(self, 'region', 'unknown')}, "
            f"worker_type={self.worker_type}, "
            f"num_workers={self.num_workers}"
        )

    def setup(self) -> None:
        """Initialize boto3 clients, resolve region, artifact bucket, and upload artifacts."""
        session = boto3.session.Session()
        self.region = self._region or session.region_name or "us-east-1"

        self.s3_client = boto3.client("s3", region_name=self.region)
        self.glue_client = boto3.client("glue", region_name=self.region)
        self.iam_client = boto3.client("iam", region_name=self.region)

        # Resolve artifact bucket (auto-create if needed)
        self.s3_artifact_bucket = _ensure_artifact_bucket(self.region, self.s3_artifact_bucket)

        # Upload script and wheel to S3
        self._upload_artifacts()

    def _upload_artifacts(self) -> None:
        """Upload the Glue entry script and dependency modules to S3."""
        # Upload entry script (embedded as _GLUE_ENTRY_SCRIPT constant)
        script_key = f"{self.s3_artifact_prefix}/scripts/invoke_glue_pipeline.py"
        self.s3_client.put_object(
            Bucket=self.s3_artifact_bucket,
            Key=script_key,
            Body=_GLUE_ENTRY_SCRIPT.encode("utf-8"),
        )
        self.script_s3_path = f"s3://{self.s3_artifact_bucket}/{script_key}"
        logger.debug("Uploaded Glue script to %s", self.script_s3_path)

        # Upload dependency modules as zip (Glue Ray requires .zip for --s3-py-modules)
        module_s3_paths: List[str] = []

        if self._extra_py_modules is not None:
            # Caller-supplied modules
            for module_path in self._extra_py_modules:
                s3_path = self._upload_module(module_path)
                module_s3_paths.append(s3_path)
        else:
            # Default: bundled agi_data_curator wheel
            s3_path = self._upload_bundled_module()
            module_s3_paths.append(s3_path)

        self.module_s3_paths = module_s3_paths
        # Keep whl_s3_path for backward compatibility
        self.whl_s3_path = ",".join(module_s3_paths)

    def _upload_bundled_module(self) -> str:
        """Upload the bundled agi_data_curator wheel and return its S3 path."""
        whl_ref = resources.files("amzn_nova_forge.dataset.bundled").joinpath(_WHL_FILENAME)
        whl_bytes = whl_ref.read_bytes()
        actual_sha = hashlib.sha256(whl_bytes).hexdigest()
        if actual_sha != _WHL_SHA256:
            raise RuntimeError(
                f"Integrity check failed for bundled wheel {_WHL_FILENAME}: "
                f"expected sha256 {_WHL_SHA256}, got {actual_sha}"
            )
        return self._upload_whl_as_zip(_WHL_FILENAME, whl_bytes)

    def _upload_module(self, module_path: str) -> str:
        """Upload a local .whl or .zip file and return its S3 path."""
        filename = os.path.basename(module_path)
        with open(module_path, "rb") as f:
            file_bytes = f.read()

        if filename.endswith(".whl"):
            return self._upload_whl_as_zip(filename, file_bytes)

        # Already a .zip — upload directly
        module_key = f"{self.s3_artifact_prefix}/modules/{filename}"
        self.s3_client.put_object(Bucket=self.s3_artifact_bucket, Key=module_key, Body=file_bytes)
        s3_path = f"s3://{self.s3_artifact_bucket}/{module_key}"
        logger.debug("Uploaded module to %s", s3_path)
        return s3_path

    def _upload_whl_as_zip(self, whl_filename: str, whl_bytes: bytes) -> str:
        """Convert a .whl to .zip and upload to S3, returning the S3 path."""
        zip_name = whl_filename.replace(".whl", ".zip")
        buf = io.BytesIO()
        with zipfile.ZipFile(io.BytesIO(whl_bytes), "r") as whl_zf:
            with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as out_zf:
                for item in whl_zf.infolist():
                    out_zf.writestr(item, whl_zf.read(item.filename))
        buf.seek(0)

        module_key = f"{self.s3_artifact_prefix}/modules/{zip_name}"
        self.s3_client.put_object(Bucket=self.s3_artifact_bucket, Key=module_key, Body=buf.read())
        s3_path = f"s3://{self.s3_artifact_bucket}/{module_key}"
        logger.debug("Uploaded module zip to %s", s3_path)
        return s3_path

    def execute(self, job_config: JobConfig) -> str:
        """Create/update a Glue job, start a run, poll until completion, and return the job run ID.

        Args:
            job_config: Must be a ``DataPrepJobConfig`` instance with
                data_s3_path (input) and output_s3_path (output) set.
                ``extra_args["pipeline_id"]`` identifies the pipeline to run.

        Returns:
            The Glue job run ID.

        Raises:
            TypeError: If job_config is not a DataPrepJobConfig.
            ValueError: If required fields are missing.
            RuntimeError: If the Glue job run ends in a non-SUCCEEDED state.
        """
        if not isinstance(job_config, DataPrepJobConfig):
            raise TypeError(
                f"GlueRuntimeManager.execute() requires a DataPrepJobConfig, "
                f"got {type(job_config).__name__}"
            )

        pipeline_id = job_config.extra_args.get("pipeline_id", "")
        if not pipeline_id:
            raise ValueError("pipeline_id is required in extra_args")
        if not job_config.data_s3_path:
            raise ValueError("data_s3_path (input_path) is required in DataPrepJobConfig")
        if not job_config.output_s3_path:
            raise ValueError("output_s3_path (output_path) is required in DataPrepJobConfig")

        job_name = job_config.job_name or f"nova-forge-{pipeline_id}"
        start = time.time()

        # Resolve IAM role ARN, creating the role if it doesn't exist
        if self.glue_role_name.startswith("arn:"):
            role_arn = self.glue_role_name
        else:
            try:
                role_arn = self.iam_client.get_role(RoleName=self.glue_role_name)["Role"]["Arn"]
            except self.iam_client.exceptions.NoSuchEntityException:
                logger.info(
                    "IAM role %r not found — auto-creating it in your account for Glue data preparation.",
                    self.glue_role_name,
                )
                assert self.s3_artifact_bucket is not None
                role_response = create_sagemaker_invoke_role(
                    self.iam_client,
                    role_name=self.glue_role_name,
                    s3_resource=self.s3_artifact_bucket,
                )
                role_arn = role_response["Role"]["Arn"]
                logger.info("Created IAM role: %s", role_arn)

        # Build --pip-install list: baseline deps + any stage-specific deps
        # declared by this job. Keep the order stable and deduplicated so
        # job-config hashes don't flip on every run.
        _BASELINE_PIP = ["s3fs", "loguru"]
        extra_pip = list(job_config.extra_pip_packages or [])
        seen: set[str] = set()
        ordered_pip: List[str] = []
        for pkg in _BASELINE_PIP + extra_pip:
            if pkg not in seen:
                seen.add(pkg)
                ordered_pip.append(pkg)
        pip_install_list = ",".join(ordered_pip)

        # Create or update Glue job
        job_params: Dict[str, Any] = {
            "Name": job_name,
            "Role": role_arn,
            "Command": {
                "Name": "glueray",
                "ScriptLocation": self.script_s3_path,
                "PythonVersion": "3.9",
                "Runtime": "Ray2.4",
            },
            "GlueVersion": self.glue_version,
            "WorkerType": self.worker_type,
            "NumberOfWorkers": self.num_workers,
            "DefaultArguments": {
                "--s3-py-modules": self.whl_s3_path,
                # ``--pip-install`` installs PyPI packages on Ray workers
                # at startup. The baseline deps are used by every curator
                # pipeline. Stage-specific deps (e.g. ``fasttext-wheel``
                # for language detection) come from
                # ``job_config.extra_pip_packages`` and are appended per
                # run below, so unrelated jobs don't pay the install cost.
                "--pip-install": pip_install_list,
            },
        }
        try:
            self.glue_client.create_job(**job_params)
            logger.debug("Created Glue job %r", job_name)
        except (
            self.glue_client.exceptions.AlreadyExistsException,
            self.glue_client.exceptions.IdempotentParameterMismatchException,
        ):
            update_params = {k: v for k, v in job_params.items() if k != "Name"}
            self.glue_client.update_job(JobName=job_name, JobUpdate=update_params)
            logger.debug("Updated existing Glue job %r", job_name)
        except ClientError as exc:
            error_code = exc.response["Error"].get("Code", "")
            if error_code in ("AccessDeniedException", "AccessDenied"):
                raise PermissionError(
                    f"Permission denied when creating Glue job '{job_name}'. "
                    "Your IAM identity needs glue:CreateJob permission. "
                    "Ensure your calling role has the required Glue permissions."
                ) from exc
            raise

        # Build run arguments
        run_args: Dict[str, str] = {
            "--pipeline_id": pipeline_id,
            "--input_path": job_config.data_s3_path,
            "--output_path": job_config.output_s3_path,
            "--input_format": job_config.input_format,
            "--output_format": job_config.output_format,
            "--text_field": job_config.text_field,
        }
        # Forward extra_args (minus pipeline_id) to the Glue script
        forwarded_args = {k: v for k, v in job_config.extra_args.items() if k != "pipeline_id"}
        if forwarded_args:
            run_args["--extra_args"] = json.dumps(forwarded_args)

        # Start job run
        job_run_id = self.glue_client.start_job_run(JobName=job_name, Arguments=run_args)[
            "JobRunId"
        ]
        logger.info("Started Glue job run %s for %r", job_run_id, job_name)
        logger.info(
            "  Console: https://%s.console.aws.amazon.com/gluestudio/home?region=%s#/job/%s/run/%s",
            self.region,
            self.region,
            job_name,
            job_run_id,
        )

        # Store for cleanup
        self._last_job_name = job_name

        # Poll until terminal state
        status = self._poll_until_terminal(job_name, job_run_id)

        elapsed = round(time.time() - start, 2)
        logger.info(
            "Glue job run %s completed in %.1fs with status %s",
            job_run_id,
            elapsed,
            status,
        )

        if status != "SUCCEEDED":
            resp = self.glue_client.get_job_run(JobName=job_name, RunId=job_run_id)
            error_msg = resp["JobRun"].get("ErrorMessage", "")
            raise RuntimeError(
                f"Glue job {job_name!r} run {job_run_id} failed with status {status}: {error_msg}"
            )

        return job_run_id

    def _poll_until_terminal(self, job_name: str, job_run_id: str) -> str:
        """Poll Glue job run until it reaches a terminal state."""
        deadline = time.monotonic() + self.max_wait_time
        while True:
            resp = self.glue_client.get_job_run(JobName=job_name, RunId=job_run_id)
            status = resp["JobRun"]["JobRunState"]
            logger.info("Glue job run %s: %s", job_run_id, status)
            if status in _TERMINAL_STATES:
                return status
            if time.monotonic() > deadline:
                raise TimeoutError(
                    f"Glue job run {job_run_id} did not reach a terminal state "
                    f"within {self.max_wait_time}s (last status: {status})"
                )
            time.sleep(self.poll_interval)

    def cleanup(self, job_id: str) -> None:
        """Stop a running Glue job run.

        Args:
            job_id: The Glue job run ID to stop.
        """
        job_name = getattr(self, "_last_job_name", None)
        if not job_name:
            raise ValueError(
                "Cannot cleanup: no job_name available. cleanup() must be called after execute()."
            )

        try:
            self.glue_client.batch_stop_job_run(JobName=job_name, JobRunIds=[job_id])
            logger.info("Stopped Glue job run %s for job %r", job_id, job_name)
        except Exception as e:
            logger.error("Failed to stop Glue job run %s: %s", job_id, e)
            raise

    @classmethod
    def required_calling_role_permissions(cls, data_s3_path=None, output_s3_path=None) -> List:
        """Required IAM permissions for Glue operations."""
        permissions = super().required_calling_role_permissions(data_s3_path, output_s3_path)

        permissions.extend(
            [
                (
                    "glue:CreateJob",
                    lambda infra: (
                        f"arn:aws:glue:{infra.region}:{get_caller_account_id(infra.region)}:job/*"
                    ),
                ),
                (
                    "glue:UpdateJob",
                    lambda infra: (
                        f"arn:aws:glue:{infra.region}:{get_caller_account_id(infra.region)}:job/*"
                    ),
                ),
                (
                    "glue:StartJobRun",
                    lambda infra: (
                        f"arn:aws:glue:{infra.region}:{get_caller_account_id(infra.region)}:job/*"
                    ),
                ),
                (
                    "glue:GetJobRun",
                    lambda infra: (
                        f"arn:aws:glue:{infra.region}:{get_caller_account_id(infra.region)}:job/*"
                    ),
                ),
                (
                    "glue:BatchStopJobRun",
                    lambda infra: (
                        f"arn:aws:glue:{infra.region}:{get_caller_account_id(infra.region)}:job/*"
                    ),
                ),
                (
                    "iam:GetRole",
                    lambda infra: f"arn:aws:iam::{get_caller_account_id(infra.region)}:role/*",
                ),
                (
                    "iam:PassRole",
                    lambda infra: f"arn:aws:iam::{get_caller_account_id(infra.region)}:role/*",
                ),
            ]
        )

        return permissions
