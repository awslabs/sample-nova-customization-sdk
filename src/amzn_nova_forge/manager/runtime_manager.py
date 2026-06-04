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
import enum
import hashlib
import io
import json
import os
import re
import subprocess
import textwrap
import time
import zipfile
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse

import boto3
import yaml
from botocore.exceptions import ClientError, NoRegionError
from sagemaker.core.helper.session_helper import Session, get_execution_role
from sagemaker.core.shapes import (
    ModelPackageConfig,
    OutputDataConfig,
    S3DataSource,
    TensorBoardOutputConfig,
)
from sagemaker.core.training.configs import (
    Compute,
    InputData,
    Networking,
    StoppingCondition,
)
from sagemaker.train.model_trainer import ModelTrainer

from amzn_nova_forge.core.constants import (
    BYOD_AVAILABLE_EVAL_TASKS,
    HYPERPOD_RECIPE_PATH,
    SERVERLESS_CUSTOM_SCORER_EVAL_TASKS,
)
from amzn_nova_forge.core.enums import Model, Platform, TrainingMethod
from amzn_nova_forge.core.runtime import RuntimeManager as RuntimeManagerBase
from amzn_nova_forge.core.types import JobConfig
from amzn_nova_forge.core.validation_patterns import MODEL_PACKAGE_ARN_REGEX
from amzn_nova_forge.manager.mtrl_manager import MTRLOperations
from amzn_nova_forge.telemetry import Feature, _telemetry_emitter
from amzn_nova_forge.util.bedrock import (
    get_customization_type,
    parse_bedrock_recipe_config,
    resolve_base_model_identifier,
)
from amzn_nova_forge.util.hub_util import get_hub_content
from amzn_nova_forge.util.logging import logger
from amzn_nova_forge.util.reward_verifier import verify_reward_function
from amzn_nova_forge.util.sagemaker import (
    extract_lambda_arn_from_hub_content,
    register_lambda_as_hub_content,
)
from amzn_nova_forge.util.subprocess_utils import _check_hyperpod_stderr
from amzn_nova_forge.validation.endpoint_validator import is_sagemaker_arn

# Maps TrainingMethod to (CustomizationTechnique, Peft|None) for ServerlessJobConfig
_METHOD_TO_SERVERLESS_CONFIG: Dict[TrainingMethod, tuple[str, Optional[str]]] = {
    TrainingMethod.SFT_LORA: ("SFT", "LORA"),
    TrainingMethod.SFT_FULL: ("SFT", None),
    TrainingMethod.DPO_LORA: ("DPO", "LORA"),
    TrainingMethod.DPO_FULL: ("DPO", None),
    TrainingMethod.RFT_LORA: ("RLVR", "LORA"),
    TrainingMethod.RFT_MULTITURN_LORA: ("RLVR", "LORA"),
    # TrainingMethod.RFT_FULL: ("RLVR", None),  # TODO: Add RLVR full support
}
DEFAULT_SMTJ_JOB_MAX_RUNTIME = 86400  # 1 day
DEFAULT_SMTJ_JOB_SUBMIT_POLL_TIMEOUT = 30  # seconds to poll for job after submission

_BUNDLED_WHL_SHA256 = {
    "agi_data_curator-1.0.0-py3-none-any.whl": "18d631f2a367dc744d34ac3b3e1cd8718a4fe3a859a79d63c29472169e7abba1",
}

DEFAULT_DATAPREP_MAX_JOB_RUNTIME = 3600
DEFAULT_VOLUME_SIZE_GB = 100

# Defaults for the AWS-managed Deep Learning Container used when no image_uri
# is supplied. The actual URI is resolved at runtime via
# ``sagemaker.core.image_uris.retrieve`` so the registry account is correct for
# the caller's region (commercial, GovCloud, China, opt-in regions differ).
DEFAULT_DLC_FRAMEWORK = "pytorch"
DEFAULT_DLC_FRAMEWORK_VERSION = "2.8"
DEFAULT_DLC_PY_VERSION = "py312"
# Packages auto-installed for data-prep jobs. The stock PyTorch DLC doesn't
# ship ray (needed for cluster bootstrap), and the bundled agi_data_curator
# wheel still imports pandas.core.common.SettingWithCopyWarning, which was
# removed in pandas 2.2 — so we pin pandas below that. The wheel also imports
# loguru and uses s3fs for S3 IO, neither of which ship in the DLC. Pins are
# skipped when the caller has already supplied their own version of the same
# package.
_DLC_REQUIRED_PACKAGES = ["ray[default]==2.35.0", "pandas<2.2", "loguru", "s3fs"]
# Fallback passed to image_uris.retrieve when no instance_type is set. GPU
# data-prep is not yet supported, so we always coerce to the CPU image below —
# when GPU is added, wire a flag and forward the real instance_type instead.
_DLC_DEFAULT_RESOLVER_INSTANCE = "ml.m5.xlarge"


class SMTJRuntimeMode(enum.Enum):
    """Mode selector for :class:`SMTJRuntimeManager`.

    TRAINING is the default; data-prep pipelines (filter operations) call
    ``set_mode(SMTJRuntimeMode.DATA_PREP)`` to route ``execute()`` /
    ``cleanup()`` through the internal data-prep delegate.
    """

    TRAINING = "training"
    DATA_PREP = "data_prep"


def _is_hub_content_arn(arn: Optional[str]) -> bool:
    """Return True if the ARN is a SageMaker hub-content ARN."""
    from amzn_nova_forge.validation.validator import (
        is_hub_content_arn,  # lazy — avoids circular import at module level
    )

    return is_hub_content_arn(arn)


@dataclass
class DataPrepJobConfig(JobConfig):
    """Job configuration for data preparation pipelines.

    Extends JobConfig, mapping:
        - job_name -> Glue job name
        - data_s3_path -> input_path
        - output_s3_path -> output_path

    Additional data-prep-specific fields:
        - input_format / output_format: "parquet" or "jsonl"
        - text_field: Column name containing the text to process
        - extra_args: Additional kwargs forwarded to the pipeline builder.
            Operation-specific keys (e.g. ``pipeline_id``) are passed here.
        - extra_pip_packages: Optional extra PyPI packages to install on
            the runtime workers. Appended to the runtime's base install
            list (e.g. Glue's ``--pip-install``). Use this for stage-level
            dependencies that aren't needed by every filter (e.g.
            ``fasttext-wheel`` for language detection).
    """

    input_format: str = "parquet"
    output_format: str = "parquet"
    text_field: str = "text"
    extra_args: Dict[str, Any] = field(default_factory=dict)
    extra_pip_packages: List[str] = field(default_factory=list)


_account_id_cache: Optional[str] = None


def _get_caller_account_id(region: str = "us-east-1") -> str:
    """Return the AWS account ID of the caller, cached to avoid redundant STS calls.

    Only caches successful results — transient STS failures return "*" without poisoning
    the cache, so subsequent calls will retry.
    """
    global _account_id_cache
    if _account_id_cache is None:
        try:
            _account_id_cache = boto3.client("sts", region_name=region).get_caller_identity()[
                "Account"
            ]
        except Exception:
            logger.warning(
                "Failed to retrieve caller account ID via STS in region %s; "
                "falling back to wildcard '*'",
                region,
                exc_info=True,
            )
            return "*"
    return _account_id_cache


def _poll_for_training_job(sagemaker_client, job_name: str, timeout: int) -> str:
    """Poll ``list_training_jobs`` until the submitted job appears.

    Uses exponential backoff (2, 4, 8, … seconds) up to *timeout* seconds total.

    Args:
        sagemaker_client: A ``boto3`` SageMaker client.
        job_name: The base job name passed to ``ModelTrainer``.
        timeout: Maximum seconds to spend polling.

    Returns:
        The unique ``TrainingJobName`` assigned by SageMaker.

    Raises:
        ValueError: If the job does not appear within *timeout* seconds.
    """
    delay = 2
    elapsed = 0
    while True:
        time.sleep(delay)
        elapsed += delay
        response = sagemaker_client.list_training_jobs(
            NameContains=job_name,
            SortBy="CreationTime",
            SortOrder="Descending",
            MaxResults=1,
        )
        summaries = response["TrainingJobSummaries"]
        if summaries:
            logger.info(f"Job lookup for '{job_name}': found after {elapsed}s")
            return summaries[0]["TrainingJobName"]
        if elapsed >= timeout:
            raise ValueError(f"{job_name} failed to submit (not found after {timeout}s)")
        delay = min(delay * 2, timeout - elapsed)


class RuntimeManager(RuntimeManagerBase):
    """Extends core ABC with ARN validation and concrete methods.

    The ``rft_lambda`` setter adds validation via a lazy import from
    ``validation.validator``.  Concrete helper methods (deploy_lambda,
    validate_lambda) also live here because they depend on modules
    outside ``core/``.
    """

    # Override the parent's rft_lambda setter to add ARN validation.
    # This replaces only the setter while inheriting the getter from
    # the base RuntimeManager.  Python requires referencing the parent
    # property explicitly — using @rft_lambda.setter here would fail
    # because rft_lambda isn't defined on this class.
    @RuntimeManagerBase.rft_lambda.setter  # type: ignore[attr-defined]
    def rft_lambda(self, value: Optional[str]) -> None:
        self._rft_lambda = value
        # Keep the resolved ARN in sync: set immediately when value is already an ARN,
        # clear it when switching to a file path or None so stale ARNs aren't reused.
        from amzn_nova_forge.validation.validator import (
            is_lambda_arn,  # avoid circular import
        )

        if value and is_lambda_arn(value):
            self._rft_lambda_arn = value
        elif value and _is_hub_content_arn(value):
            self._rft_lambda_arn = value
        else:
            self._rft_lambda_arn = None

    @_telemetry_emitter(Feature.INFRA, "deploy_lambda")
    def deploy_lambda(
        self,
        lambda_name: Optional[str] = None,
        execution_role_arn: Optional[str] = None,
    ) -> str:
        """
        Deploy the RFT reward lambda from a local Python file set on this manager.

        Uses self.rft_lambda as the source file. If self.rft_lambda is a Lambda ARN
        rather than a file path, raises ValueError — the lambda is already deployed.

        Args:
            lambda_name: Name for the Lambda function. Defaults to the source filename stem.
            execution_role_arn: IAM role ARN for the Lambda. If not provided, falls
                back to the runtime manager's execution_role attribute.

        Returns:
            The deployed Lambda function ARN.

        Raises:
            ValueError: If rft_lambda is not set, is already an ARN, file is not found,
                or if no execution role can be resolved.
        """
        if not self.rft_lambda:
            raise ValueError(
                "rft_lambda must be set on the runtime manager to a local .py file path before calling deploy_lambda()."
            )
        from amzn_nova_forge.validation.validator import (
            is_lambda_arn,
            validate_rft_lambda_name,  # to avoid circular import error
        )

        if is_lambda_arn(self.rft_lambda):
            raise ValueError(
                f"rft_lambda is already a deployed Lambda ARN ('{self.rft_lambda}'). "
                "deploy_lambda() deploys from a local .py file — there is nothing to deploy."
            )

        lambda_source = self.rft_lambda

        if lambda_name is None:
            lambda_name = os.path.splitext(os.path.basename(lambda_source))[0].replace("_", "-")

        platform = self.platform

        validate_rft_lambda_name(lambda_name, platform)

        role_arn = execution_role_arn or getattr(self, "execution_role", None)
        if not role_arn:
            raise ValueError(
                "An execution_role_arn must be provided to deploy_lambda(), or set as "
                "execution_role on the runtime manager."
            )

        if not os.path.isfile(lambda_source):
            raise ValueError(f"lambda_source file not found: {lambda_source}")

        # Package the .py file into a zip in memory
        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zf:
            zf.write(lambda_source, arcname="lambda_function.py")
        zip_bytes = zip_buffer.getvalue()

        region = getattr(self, "region", None)
        if not region:
            logger.warning(
                "region is not set on the runtime manager; falling back to 'us-east-1'. "
                "This may deploy the Lambda to the wrong region."
            )
            region = "us-east-1"
        lambda_client = boto3.client("lambda", region_name=region)

        try:
            lambda_client.get_function(FunctionName=lambda_name)
            logger.info(f"Lambda '{lambda_name}' already exists — updating function code.")
            response = lambda_client.update_function_code(
                FunctionName=lambda_name,
                ZipFile=zip_bytes,
            )
            lambda_arn = response["FunctionArn"]
        except lambda_client.exceptions.ResourceNotFoundException:
            logger.info(f"Creating Lambda function '{lambda_name}'...")
            response = lambda_client.create_function(
                FunctionName=lambda_name,
                Runtime="python3.12",
                Role=role_arn,
                Handler="lambda_function.lambda_handler",
                Code={"ZipFile": zip_bytes},
                Timeout=300,
            )
            lambda_arn = response["FunctionArn"]

        waiter = lambda_client.get_waiter("function_active_v2")
        waiter.wait(FunctionName=lambda_name)
        logger.info(f"Lambda '{lambda_name}' is active. ARN: {lambda_arn}")
        self.rft_lambda_arn = lambda_arn

        return lambda_arn

    @_telemetry_emitter(Feature.INFRA, "validate_lambda")
    def validate_lambda(
        self,
        data_s3_path: str,
        validation_samples: int = 10,
    ) -> None:
        """
        Validate the RFT reward lambda with sample data.

        Resolves the lambda to validate from self.rft_lambda / self.rft_lambda_arn:
        - If self.rft_lambda is an ARN (or self.rft_lambda_arn is set), invokes the
          deployed lambda with samples from data_s3_path.
        - If self.rft_lambda is a local .py path, validates by executing lambda_handler
          directly without deploying.

        Args:
            data_s3_path: S3 path to the training dataset for pulling sample data.
            validation_samples: Number of samples to pull from data_s3_path (default: 10).

        Raises:
            ValueError: If rft_lambda is not set, or if validation fails.
        """
        lambda_arn = self.rft_lambda_arn

        # rft_lambda is a local file only if it's not any kind of ARN
        lambda_source = (
            self.rft_lambda
            if self.rft_lambda and not (self.rft_lambda.startswith("arn:"))
            else None
        )

        if not lambda_arn and not lambda_source:
            raise ValueError(
                "Either lambda_arn or lambda_source must be provided to validate_lambda()."
            )

        region = getattr(self, "region", None)
        if not region:
            logger.warning(
                "region is not set on the runtime manager; falling back to 'us-east-1'. "
                "This may validate the Lambda in the wrong region."
            )
            region = "us-east-1"
        platform = self.platform

        if lambda_arn:
            # Extract function name from ARN and validate naming requirements early
            function_name = lambda_arn.split(":")[-1]
            from amzn_nova_forge.validation.validator import (
                validate_rft_lambda_name,
                verify_rft_lambda,
            )

            validate_rft_lambda_name(function_name, platform)
            verify_rft_lambda(
                lambda_arn=lambda_arn,
                sample_count=validation_samples,
                data_s3_path=data_s3_path,
                region=region,
                platform=platform,
            )
        else:
            # Validate local file without deploying — executes lambda_handler directly
            logger.info(f"Validating local lambda source '{lambda_source}' without deployment...")
            sample_data = []
            if data_s3_path:
                from amzn_nova_forge.validation.validator import _parse_s3_uri

                s3_parts = _parse_s3_uri(data_s3_path)
                if not s3_parts:
                    raise ValueError(
                        f"Invalid S3 path: {data_s3_path}. Expected format: s3://bucket/key"
                    )
                bucket, key = s3_parts
                logger.info(f"Loading up to {validation_samples} sample(s) from {data_s3_path}")
                try:
                    s3_client = boto3.client("s3", region_name=region)
                    response = s3_client.get_object(Bucket=bucket, Key=key)
                    for i, line in enumerate(response["Body"].iter_lines()):
                        if i >= validation_samples:
                            break
                        try:
                            sample_data.append(json.loads(line.decode("utf-8")))
                        except json.JSONDecodeError as e:
                            logger.warning(f"Skipping malformed JSON on line {i + 1}: {e}")
                    logger.info(f"Loaded {len(sample_data)} sample(s) from S3")
                except Exception as e:
                    raise ValueError(f"Failed to read samples from {data_s3_path}: {e}") from e

            verify_reward_function(
                reward_function=lambda_source or "",
                sample_data=sample_data,
                validate_format=len(sample_data) > 0,
            )
            logger.info("Local lambda source validation passed.")


class SMTJRuntimeManager(RuntimeManager):
    """Runtime manager for SageMaker Training Jobs (training and data prep).

    Defaults to training mode: behaves as a standard SMTJ training manager
    using the SageMaker SDK ``ModelTrainer``. To run data-prep pipelines,
    call :meth:`set_mode` with :class:`SMTJRuntimeMode.DATA_PREP` — filter
    operations do this automatically when a manager of this type is passed
    as ``runtime_manager``.

    Data-prep-mode constructor kwargs (``image_uri``, ``poll_interval``,
    ``s3_artifact_bucket``, etc.) are stashed at construction and used
    when :meth:`set_mode` flips the manager into data-prep mode.

    Args:
        instance_type: EC2 instance type.
        instance_count: Number of instances.
        execution_role: IAM role ARN.  Auto-resolved when ``None``.
        kms_key_id: Optional KMS key for encryption.
        encrypt_inter_container_traffic: Encrypt traffic between containers.
        subnets: Optional VPC subnets.
        security_group_ids: Optional VPC security group IDs.
        max_job_runtime: Maximum runtime in seconds.
        job_submit_poll_timeout: Seconds to poll for the job after submission.
        rft_lambda: Optional RFT reward Lambda ARN or file path.
        image_uri: Docker image URI for data prep containers. Defaults to the
            AWS-managed PyTorch DLC when empty.
        poll_interval: Seconds between status polls for data prep jobs.
        max_wait_time: Maximum seconds to wait for data prep completion.
        s3_artifact_bucket: S3 bucket for data prep artifacts.
        s3_artifact_prefix: S3 key prefix for data prep artifacts.
        volume_size_gb: EBS volume size in GB for data prep jobs.
        keep_alive_seconds: Warm pool keep-alive period for data prep.
        region: AWS region override for data prep. Defaults to session region.
    """

    def __init__(
        self,
        instance_type: str,
        instance_count: int,
        execution_role: Optional[str] = None,
        kms_key_id: Optional[str] = None,
        encrypt_inter_container_traffic: bool = False,
        subnets: Optional[list[str]] = None,
        security_group_ids: Optional[list[str]] = None,
        max_job_runtime: Optional[int] = None,
        job_submit_poll_timeout: int = DEFAULT_SMTJ_JOB_SUBMIT_POLL_TIMEOUT,
        rft_lambda: Optional[str] = None,
        # --- Data-prep-mode kwargs (used after set_mode(DATA_PREP)) ---
        image_uri: str = "",
        poll_interval: int = 30,
        max_wait_time: int = 3600,
        s3_artifact_bucket: Optional[str] = None,
        s3_artifact_prefix: Optional[str] = None,
        volume_size_gb: int = DEFAULT_VOLUME_SIZE_GB,
        keep_alive_seconds: int = 0,
        region: Optional[str] = None,
    ):
        # NOTE: Not setting execution_role directly due to issues with mypy type inference
        self._execution_role = execution_role
        # Preserve the caller's original input (None, a role name, or an ARN)
        # so the data-prep delegate can apply its own default (None → auto-
        # create SmtjDataPrepExecutionRole) rather than inheriting whatever
        # training-mode setup() resolved to.
        self._caller_execution_role = execution_role

        self.subnets = subnets
        self.security_group_ids = security_group_ids
        self.encrypt_inter_container_traffic = encrypt_inter_container_traffic
        # Default training runtime. When set_mode(DATA_PREP) is called and no
        # caller override was supplied, the delegate is built with the
        # data-prep default (see _build_data_prep_delegate).
        self.max_job_runtime = max_job_runtime or DEFAULT_SMTJ_JOB_MAX_RUNTIME
        self._max_job_runtime_override = max_job_runtime
        self.job_submit_poll_timeout = job_submit_poll_timeout

        # Stash data-prep kwargs; consumed by set_mode(DATA_PREP).
        self._dataprep_image_uri = image_uri
        self._dataprep_poll_interval = poll_interval
        self._dataprep_max_wait_time = max_wait_time
        self._dataprep_s3_artifact_bucket = s3_artifact_bucket
        self._dataprep_s3_artifact_prefix = s3_artifact_prefix
        self._dataprep_volume_size_gb = volume_size_gb
        self._dataprep_keep_alive_seconds = keep_alive_seconds
        self._dataprep_region = region

        self._mode: SMTJRuntimeMode = SMTJRuntimeMode.TRAINING
        self._data_prep_delegate: Optional["SMTJDataPrepRuntimeManager"] = None

        super().__init__(instance_type, instance_count, kms_key_id, rft_lambda)
        self.setup()

    def set_mode(self, mode: SMTJRuntimeMode) -> None:
        """Switch the manager between training and data-prep modes.

        Idempotent when called with the current mode. The only supported
        transition is TRAINING → DATA_PREP; once a manager is in data-prep
        mode it cannot flip back, since the training setup state has been
        discarded.

        Called automatically by filter operations when an ``SMTJRuntimeManager``
        is used as their ``runtime_manager``.

        Args:
            mode: Target mode.

        Raises:
            TypeError: If ``mode`` is not an :class:`SMTJRuntimeMode`.
            RuntimeError: On an unsupported DATA_PREP → TRAINING flip.
        """
        if not isinstance(mode, SMTJRuntimeMode):
            raise TypeError(f"set_mode() expects an SMTJRuntimeMode, got {type(mode).__name__}")

        if mode == self._mode:
            return

        if mode == SMTJRuntimeMode.TRAINING:
            # We've already switched into data-prep mode and discarded the
            # training setup state; flipping back would need a fresh manager.
            raise RuntimeError(
                "Cannot switch an SMTJRuntimeManager from DATA_PREP back to "
                "TRAINING. Construct a new SMTJRuntimeManager for training."
            )

        self._build_data_prep_delegate()
        self._mode = SMTJRuntimeMode.DATA_PREP

    def _build_data_prep_delegate(self) -> None:
        from amzn_nova_forge.util.s3_utils import SMTJ_DATAPREP_ARTIFACT_PREFIX

        dataprep_max_runtime = self._max_job_runtime_override or DEFAULT_DATAPREP_MAX_JOB_RUNTIME
        self.max_job_runtime = dataprep_max_runtime

        # Narrow the Optional types inherited from the base class — the
        # SMTJRuntimeManager constructor requires both to be non-None.
        assert self.instance_type is not None
        assert self.instance_count is not None

        self._data_prep_delegate = SMTJDataPrepRuntimeManager(
            instance_type=self.instance_type,
            instance_count=self.instance_count,
            image_uri=self._dataprep_image_uri,
            # Forward the *original* caller input (None / role-name / ARN)
            # so the delegate falls back to its own default
            # (SmtjDataPrepExecutionRole) when the caller passed nothing.
            # Don't forward the training-mode resolved execution_role — it
            # typically resolves to the caller's own identity, which doesn't
            # trust sagemaker.amazonaws.com.
            execution_role_name=self._caller_execution_role,
            kms_key_id=self.kms_key_id,
            subnets=self.subnets,
            security_group_ids=self.security_group_ids,
            max_job_runtime=dataprep_max_runtime,
            poll_interval=self._dataprep_poll_interval,
            max_wait_time=self._dataprep_max_wait_time,
            s3_artifact_bucket=self._dataprep_s3_artifact_bucket,
            s3_artifact_prefix=(self._dataprep_s3_artifact_prefix or SMTJ_DATAPREP_ARTIFACT_PREFIX),
            volume_size_gb=self._dataprep_volume_size_gb,
            keep_alive_seconds=self._dataprep_keep_alive_seconds,
            region=self._dataprep_region,
        )
        # Drop training-setup state — unused in data-prep mode and confusing
        # if it lingers.
        for attr in (
            "execution_role",
            "sagemaker_client",
            "sagemaker_session",
            "_execution_role",
        ):
            if hasattr(self, attr):
                delattr(self, attr)

    @classmethod
    def required_calling_role_permissions(cls, data_s3_path=None, output_s3_path=None):
        """Required permissions for SMTJ calling role operations and execution role validation."""
        # Start with base S3 permissions
        permissions = super().required_calling_role_permissions(data_s3_path, output_s3_path)

        # Add SMTJ-specific permissions
        permissions.extend(
            [
                ("sagemaker:CreateTrainingJob", "*"),
                ("sagemaker:DescribeTrainingJob", "*"),
                ("sagemaker:StopTrainingJob", "*"),
                "iam:GetRole",
                "iam:PassRole",
                "iam:GetPolicy",
                "iam:GetPolicyVersion",
                "iam:ListRolePolicies",
                "iam:GetRolePolicy",
                "iam:ListAttachedRolePolicies",
            ]
        )

        return permissions

    @property
    def platform(self) -> Platform:
        return Platform.SMTJ

    @property
    def runtime_name(self) -> str:
        return "SageMaker Training Job"

    def setup(self) -> None:
        boto_session = boto3.session.Session()
        self.region = boto_session.region_name or "us-east-1"
        self.sagemaker_client = boto3.client("sagemaker", region_name=self.region)
        self.sagemaker_session = Session(
            boto_session=boto_session, sagemaker_client=self.sagemaker_client
        )

        if self._execution_role is None:
            self.execution_role = get_execution_role(use_default=True)
        else:
            self.execution_role = self._execution_role
        # Delete temporary attribute so customers don't confuse it with the actual attribute
        del self._execution_role

    def execute(self, job_config: JobConfig) -> str:
        if self._mode == SMTJRuntimeMode.DATA_PREP:
            assert self._data_prep_delegate is not None
            return self._data_prep_delegate.execute(job_config)

        if isinstance(job_config, DataPrepJobConfig):
            raise ValueError(
                "This SMTJRuntimeManager is in training mode but received a "
                "DataPrepJobConfig. Call set_mode(SMTJRuntimeMode.DATA_PREP) "
                "before running data-prep pipelines (filter operations do "
                "this automatically)."
            )

        from amzn_nova_forge.validation.validator import Validator

        Validator.validate_job_name(job_name=job_config.job_name)

        try:
            assert job_config.output_s3_path is not None

            tensorboard_output_config = TensorBoardOutputConfig(
                s3_output_path=job_config.output_s3_path,
            )

            compute = Compute(
                instance_count=self.instance_count,
                instance_type=self.instance_type,
            )
            output_data_config = OutputDataConfig(
                s3_output_path=job_config.output_s3_path,
                kms_key_id=self.kms_key_id,
            )
            networking = Networking(
                subnets=self.subnets,
                security_group_ids=self.security_group_ids,
            )
            stopping_condition = StoppingCondition(max_runtime_in_seconds=self.max_job_runtime)

            trainer_config = {
                "training_recipe": job_config.recipe_path,
                "compute": compute,
                "networking": networking,
                "stopping_condition": stopping_condition,
                "output_data_config": output_data_config,
                "base_job_name": job_config.job_name,
                "role": self.execution_role,
                "sagemaker_session": self.sagemaker_session,
                "training_image": job_config.image_uri,
            }
            if job_config.environment:
                # Pass arbitrary environment variables to the training container
                # (used by InspectLens to inject HF_TOKEN, timeouts, etc.)
                trainer_config["environment"] = job_config.environment
            # For eval job, the input could be none
            # https://docs.aws.amazon.com/sagemaker/latest/dg/nova-model-evaluation.html#nova-model-evaluation-notebook
            if job_config.data_s3_path:
                input_data = InputData(
                    channel_name="train",
                    data_source=S3DataSource(
                        s3_uri=job_config.data_s3_path,
                        s3_data_type=job_config.input_s3_data_type,
                        s3_data_distribution_type="FullyReplicated",
                    ),
                )
                input_data_config = [input_data]
                if job_config.validation_data_s3_path:
                    validation_input = InputData(
                        channel_name="validation",
                        data_source=S3DataSource(
                            s3_uri=job_config.validation_data_s3_path,
                            s3_data_type=job_config.input_s3_data_type,
                            s3_data_distribution_type="FullyReplicated",
                        ),
                    )
                    input_data_config.append(validation_input)
                trainer_config["input_data_config"] = input_data_config

            if job_config.trainer_config_hyperparameters:
                trainer_config["hyperparameters"] = job_config.trainer_config_hyperparameters

            # If the model_name_or_path is a model package ARN,
            # ModelTrainer.from_recipe() requires a ModelPackageConfig with model_package_group_arn.
            model_package_config = None
            if (
                job_config.method == TrainingMethod.EVALUATION
                and job_config.model_name_or_path
                and MODEL_PACKAGE_ARN_REGEX.match(job_config.model_name_or_path)
            ):
                # Extract model package group name from the ARN
                _parts = job_config.model_name_or_path.split("/")
                if len(_parts) < 2:
                    raise ValueError(
                        f"Could not parse model package group name from ARN: '{job_config.model_name_or_path}'. "
                        f"Expected format: arn:aws:sagemaker:region:account:model-package/group-name/version"
                    )
                _group_name = _parts[1]
                # Look up the existing model package group ARN
                try:
                    _resp = self.sagemaker_client.describe_model_package_group(
                        ModelPackageGroupName=_group_name
                    )
                    _group_arn = _resp["ModelPackageGroupArn"]
                except ClientError as e:
                    if e.response["Error"]["Code"] in (
                        "ValidationException",
                        "ResourceNotFoundException",
                    ):
                        raise ValueError(
                            f"Model package group '{_group_name}' does not exist. "
                            f"Ensure the model package ARN '{job_config.model_name_or_path}' refers to an existing group."
                        ) from e
                    raise  # Re-raise unexpected errors
                model_package_config = ModelPackageConfig(
                    model_package_group_arn=_group_arn,
                )

            if model_package_config:
                trainer_config["model_package_config"] = model_package_config

            model_trainer = ModelTrainer.from_recipe(
                **trainer_config
            ).with_tensorboard_output_config(tensorboard_output_config)
            model_trainer.train(wait=False, logs=False)
            logger.info(f"Training job submitted with base_job_name='{job_config.job_name}'")

            unique_job_name = _poll_for_training_job(
                self.sagemaker_client, job_config.job_name, self.job_submit_poll_timeout
            )
            return unique_job_name

        except Exception as e:
            logger.error(f"Failed to start training job: {str(e)}")
            raise

    def cleanup(self, job_name: str) -> None:
        if self._mode == SMTJRuntimeMode.DATA_PREP:
            assert self._data_prep_delegate is not None
            return self._data_prep_delegate.cleanup(job_name)

        try:
            self.sagemaker_client.stop_training_job(TrainingJobName=job_name)
            self.sagemaker_client.close()
        except Exception as e:
            logger.error(f"Failed to cleanup job {job_name}: {str(e)}")
            raise


class SMTJDataPrepRuntimeManager(RuntimeManager):
    """Runtime manager for data preparation pipelines on SageMaker Training Jobs with Ray.

    Launches a Ray cluster on SMTJ instances (including GPU types like
    ml.g5, ml.p4d, ml.p5), uploads an entry script and bundled ``.whl``
    dependencies to S3, and runs ForgeWorkflows pipelines on the cluster.
    SageMaker manages instance provisioning, networking, and teardown.

    Typically used via ``SMTJRuntimeManager(data_prep=True, ...)`` rather
    than instantiated directly.
    """

    SMTJ_DATAPREP_ROLE_NAME = "SmtjDataPrepExecutionRole"
    _TERMINAL_STATES = {"Completed", "Failed", "Stopped"}

    # Entry-point script uploaded to S3 and executed inside the SMTJ container.
    # Bootstraps a Ray cluster using SageMaker multi-node env vars, then runs the
    # requested ForgeWorkflows pipeline on the head node.
    _SMTJ_ENTRY_SCRIPT = textwrap.dedent("""\
        from __future__ import annotations

        import glob
        import json
        import logging
        import os
        import signal
        import socket
        import subprocess
        import sys
        import time

        logging.basicConfig(level=logging.INFO)
        logger = logging.getLogger("smtj_ray_pipeline")


        def _parse_sm_hosts(hosts_raw):
            \"\"\"Parse SM_HOSTS env var into a list of hostnames.\"\"\"
            return (
                json.loads(hosts_raw) if hosts_raw.startswith("[")
                else (hosts_raw.split(",") if hosts_raw else [])
            )


        def _install_deps(pip_install):
            \"\"\"Install bundled .whl files and any caller-supplied pip packages.

            Bundled wheels are extracted directly into site-packages (no pip)
            because at least one shipped wheel has a mismatched dist-info name
            that pip rejects with "invalid wheel, .dist-info directory does not
            start with ...". Extraction bypasses that validation. PyPI packages
            supplied via ``pip_install`` still go through pip so transitive deps
            resolve correctly.
            \"\"\"
            import site
            import zipfile

            deps_dir = "/opt/ml/input/data/deps"
            whls = glob.glob(os.path.join(deps_dir, "*.whl")) if os.path.isdir(deps_dir) else []
            if whls:
                site_dirs = site.getsitepackages()
                target = site_dirs[0] if site_dirs else site.getusersitepackages()
                logger.info("Extracting bundled wheels to %s: %s", target, whls)
                target_abs = os.path.realpath(target)
                for whl_path in whls:
                    with zipfile.ZipFile(whl_path, "r") as zf:
                        for name in zf.namelist():
                            # Reject absolute paths and path traversal
                            if os.path.isabs(name) or name.startswith("/"):
                                raise RuntimeError(f"Unsafe absolute path: {name!r} in {whl_path}")
                            resolved = os.path.realpath(os.path.join(target_abs, name))
                            if (
                                resolved != target_abs
                                and not resolved.startswith(target_abs + os.sep)
                            ):
                                raise RuntimeError(f"Unsafe path traversal: {name!r} in {whl_path}")
                        zf.extractall(target_abs)

            if pip_install:
                logger.info("Installing additional packages: %s", pip_install)
                subprocess.run(
                    [sys.executable, "-m", "pip", "install", *pip_install],
                    check=True,
                )


        def _start_ray_cluster():
            \"\"\"Initialize Ray cluster using SageMaker multi-node environment variables.\"\"\"
            hosts_raw = os.environ.get("SM_HOSTS", "")
            current_host = os.environ.get("SM_CURRENT_HOST", "")

            if not hosts_raw or not current_host:
                import ray
                ray.init()
                logger.info("Single-node mode: Ray initialized locally")
                return

            hosts = _parse_sm_hosts(hosts_raw)
            head_host = hosts[0]
            port = 6379

            if current_host == head_host:
                subprocess.run(
                    ["ray", "start", "--head", "--port", str(port)],
                    check=True,
                )
                logger.info("Ray head started on %s:%d", current_host, port)
            else:
                head_ip = socket.gethostbyname(head_host)
                # Wait for head node to become reachable
                for attempt in range(60):
                    try:
                        with socket.socket() as s:
                            s.settimeout(2)
                            s.connect((head_ip, port))
                        break
                    except Exception:
                        if attempt % 6 == 0:
                            logger.info("Waiting for Ray head at %s:%d ...", head_ip, port)
                        time.sleep(5)
                else:
                    raise RuntimeError(f"Ray head not reachable at {head_ip}:{port} after 300s")

                subprocess.run(
                    ["ray", "start", "--address", f"{head_ip}:{port}"],
                    check=True,
                )
                logger.info("Ray worker started, connected to %s:%d", head_ip, port)

            if current_host == head_host:
                import ray
                ray.init(address="auto")
                logger.info("Ray head connected to cluster")


        def main():
            # SageMaker writes hyperparameters to this JSON file
            hp_path = "/opt/ml/input/config/hyperparameters.json"
            if os.path.exists(hp_path):
                with open(hp_path) as f:
                    hp = json.load(f)
            else:
                hp = {}

            try:
                pip_install = json.loads(hp.get("pip_install", "[]"))
            except json.JSONDecodeError:
                logger.warning("Could not parse pip_install as JSON, skipping")
                pip_install = []
            _install_deps(pip_install)

            pipeline_id = hp.get("pipeline_id", "")
            input_path = hp.get("input_path", "")
            output_path = hp.get("output_path", "")
            input_format = hp.get("input_format", "parquet")
            output_format = hp.get("output_format", "parquet")
            text_field = hp.get("text_field", "text")
            extra_args_raw = hp.get("extra_args", "{}")

            try:
                extra_kwargs = json.loads(extra_args_raw)
            except json.JSONDecodeError:
                logger.warning("Could not parse extra_args as JSON, using empty dict")
                extra_kwargs = {}

            logger.info(
                "SMTJ Ray pipeline invocation:\\n"
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

            hosts_raw = os.environ.get("SM_HOSTS", "")
            current_host = os.environ.get("SM_CURRENT_HOST", "")
            hosts = _parse_sm_hosts(hosts_raw)

            # Initialize Ray cluster
            _start_ray_cluster()

            # Only the head node runs the pipeline; workers block until SageMaker terminates.
            if not hosts or current_host == hosts[0]:
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

            else:
                # Worker nodes: keep alive until SageMaker terminates all containers
                logger.info("Worker node %s blocking until job completion", current_host)
                signal.pause()


        if __name__ == "__main__":
            main()
    """)

    def __init__(
        self,
        instance_type: str,
        instance_count: int = 1,
        image_uri: str = "",
        execution_role_name: Optional[str] = None,
        kms_key_id: Optional[str] = None,
        subnets: Optional[List[str]] = None,
        security_group_ids: Optional[List[str]] = None,
        max_job_runtime: int = DEFAULT_DATAPREP_MAX_JOB_RUNTIME,
        poll_interval: int = 30,
        max_wait_time: int = 3600,
        s3_artifact_bucket: Optional[str] = None,
        s3_artifact_prefix: Optional[str] = None,
        volume_size_gb: int = DEFAULT_VOLUME_SIZE_GB,
        keep_alive_seconds: int = 0,
        region: Optional[str] = None,
    ):
        from amzn_nova_forge.util.s3_utils import SMTJ_DATAPREP_ARTIFACT_PREFIX

        self._execution_role_name = execution_role_name
        self.image_uri = image_uri
        self.subnets = subnets
        self.security_group_ids = security_group_ids
        self.max_job_runtime = max_job_runtime
        self.poll_interval = poll_interval
        self.max_wait_time = max_wait_time
        self.s3_artifact_bucket = s3_artifact_bucket
        self.s3_artifact_prefix = (s3_artifact_prefix or SMTJ_DATAPREP_ARTIFACT_PREFIX).rstrip("/")
        self.volume_size_gb = volume_size_gb
        self.keep_alive_seconds = keep_alive_seconds
        self._region = region

        super().__init__(
            instance_type=instance_type,
            instance_count=instance_count,
            kms_key_id=kms_key_id,
        )
        self.setup()

    @property
    def platform(self) -> Platform:
        return Platform.SMTJ

    @property
    def runtime_name(self) -> str:
        return "SageMaker Training Job"

    @property
    def runtime_config(self) -> str:
        return (
            f"region={getattr(self, 'region', 'unknown')}, "
            f"instance_type={self.instance_type}, "
            f"instance_count={self.instance_count}"
        )

    def setup(self) -> None:
        boto_session = boto3.session.Session()
        self.region = self._region or boto_session.region_name or "us-east-1"

        if not self.image_uri:
            from sagemaker.core.image_uris import retrieve

            # The CPU vs. GPU PyTorch DLC is selected automatically by SageMaker
            # based on the instance type passed to retrieve(): CPU families
            # (ml.m*, ml.c*, ...) resolve to the CPU image, and GPU/accelerator
            # families (ml.g*, ml.p*, ml.trn*) resolve to the matching GPU image.
            # Data-prep pipelines don't leverage GPU compute today, but running
            # on a GPU instance still works — the job just won't use the GPU.
            # When no instance_type is set we fall back to a CPU default so the
            # resolver has a concrete family to pick from.
            resolver_instance = self.instance_type or _DLC_DEFAULT_RESOLVER_INSTANCE

            self.image_uri = retrieve(
                framework=DEFAULT_DLC_FRAMEWORK,
                region=self.region,
                version=DEFAULT_DLC_FRAMEWORK_VERSION,
                py_version=DEFAULT_DLC_PY_VERSION,
                image_scope="training",
                instance_type=resolver_instance,
            )
            logger.info("No image_uri supplied — defaulting to AWS DLC %s", self.image_uri)

        self.sagemaker_client = boto3.client("sagemaker", region_name=self.region)
        self.s3_client = boto3.client("s3", region_name=self.region)

        role_ref = self._execution_role_name or self.SMTJ_DATAPREP_ROLE_NAME
        if role_ref.startswith("arn:"):
            # Already a full ARN — use as-is
            self.execution_role = role_ref
        else:
            # Treat as a role name — look up or create
            iam_client = boto3.client("iam", region_name=self.region)
            try:
                self.execution_role = iam_client.get_role(RoleName=role_ref)["Role"]["Arn"]
            except iam_client.exceptions.NoSuchEntityException:
                logger.info(
                    "IAM role %r not found — auto-creating it in your account for SMTJ data preparation.",
                    role_ref,
                )
                from amzn_nova_forge.iam.iam_role_creator import (
                    create_smtj_dataprep_execution_role,
                )

                self.execution_role = create_smtj_dataprep_execution_role(
                    iam_client=iam_client,
                    role_name=role_ref,
                )["Role"]["Arn"]
                logger.info("Created IAM role: %s", self.execution_role)

        # Resolve artifact bucket and upload entry script + dependencies
        self.s3_artifact_bucket = self._ensure_artifact_bucket()
        self._upload_entry_script()
        self._upload_bundled_whls()

    def _upload_entry_script(self) -> None:
        """Upload the Ray bootstrap + ForgeWorkflows entry script to S3."""
        script_key = f"{self.s3_artifact_prefix}/scripts/invoke_smtj_pipeline.py"
        self.s3_client.put_object(
            Bucket=self.s3_artifact_bucket,
            Key=script_key,
            Body=self._SMTJ_ENTRY_SCRIPT.encode("utf-8"),
        )
        self.script_s3_path = f"s3://{self.s3_artifact_bucket}/{script_key}"
        logger.info("Uploaded SMTJ entry script to %s", self.script_s3_path)

    def _upload_bundled_whls(self) -> None:
        """Upload all bundled .whl files from the dataset/bundled package to S3."""
        from importlib import resources

        bundled = resources.files("amzn_nova_forge.dataset.bundled")
        whl_names = [item.name for item in bundled.iterdir() if item.name.endswith(".whl")]
        if not whl_names:
            raise FileNotFoundError("No .whl files found in amzn_nova_forge.dataset.bundled")

        deps_prefix = f"{self.s3_artifact_prefix}/deps"
        for whl_name in whl_names:
            whl_bytes = bundled.joinpath(whl_name).read_bytes()
            expected_sha = _BUNDLED_WHL_SHA256.get(whl_name)
            if expected_sha:
                actual_sha = hashlib.sha256(whl_bytes).hexdigest()
                if actual_sha != expected_sha:
                    raise RuntimeError(
                        f"Integrity check failed for bundled wheel {whl_name}: "
                        f"expected sha256 {expected_sha}, got {actual_sha}"
                    )
            whl_key = f"{deps_prefix}/{whl_name}"
            self.s3_client.put_object(
                Bucket=self.s3_artifact_bucket,
                Key=whl_key,
                Body=whl_bytes,
            )
            logger.info("Uploaded bundled whl to s3://%s/%s", self.s3_artifact_bucket, whl_key)

        self.deps_s3_prefix = f"s3://{self.s3_artifact_bucket}/{deps_prefix}"
        logger.info("All bundled whls uploaded to %s", self.deps_s3_prefix)

    def _ensure_artifact_bucket(self) -> str:
        """Return an S3 artifact bucket, creating a default one if needed."""
        if self.s3_artifact_bucket:
            return self.s3_artifact_bucket

        from amzn_nova_forge.util.s3_utils import (
            ensure_bucket_exists,
            get_dataprep_bucket_name,
        )

        default_bucket = get_dataprep_bucket_name(region=self.region)
        ensure_bucket_exists(default_bucket, region=self.region)
        return default_bucket

    @_telemetry_emitter(Feature.DATA_PREP, "smtj_dataprep_execute")
    def execute(self, job_config: JobConfig) -> str:
        """Create a SageMaker Training Job for data prep, poll until completion.

        Args:
            job_config: A ``DataPrepJobConfig`` with ``data_s3_path``,
                ``output_s3_path``, and ``extra_args["pipeline_id"]`` set.

        Returns:
            The SageMaker Training Job name.

        Raises:
            TypeError: If job_config is not a DataPrepJobConfig.
            ValueError: If required fields are missing.
            RuntimeError: If the training job fails.
            TimeoutError: If the job does not complete within max_wait_time.
        """
        dataprep_config = self._validate_dataprep_config(job_config)
        pipeline_id = dataprep_config.extra_args["pipeline_id"]

        job_name = dataprep_config.job_name or self._build_job_name(pipeline_id)
        start = time.time()

        hyperparameters = self._build_hyperparameters(dataprep_config, pipeline_id)
        create_params = self._build_create_training_job_params(
            dataprep_config, job_name, hyperparameters
        )

        self.sagemaker_client.create_training_job(**create_params)
        logger.info("Created SMTJ data prep job %r", job_name)
        logger.info(
            "  Console: https://%s.console.aws.amazon.com/sagemaker/home?region=%s#/jobs/%s",
            self.region,
            self.region,
            job_name,
        )

        status = self._poll_until_terminal(job_name)

        elapsed = round(time.time() - start, 2)
        logger.info(
            "SMTJ data prep job %r completed in %.1fs with status %s",
            job_name,
            elapsed,
            status,
        )

        self._raise_if_not_completed(job_name, status)
        return job_name

    def _validate_dataprep_config(self, job_config: JobConfig) -> "DataPrepJobConfig":
        """Validate that ``job_config`` is a well-formed DataPrepJobConfig.

        Returns the same object narrowed to ``DataPrepJobConfig`` and with
        ``extra_args["pipeline_id"]`` guaranteed to be non-empty.
        """
        if not isinstance(job_config, DataPrepJobConfig):
            raise TypeError(
                f"SMTJDataPrepRuntimeManager.execute() requires a DataPrepJobConfig, "
                f"got {type(job_config).__name__}"
            )

        if not job_config.extra_args.get("pipeline_id", ""):
            raise ValueError("pipeline_id is required in extra_args")
        if not job_config.data_s3_path:
            raise ValueError("data_s3_path (input_path) is required in DataPrepJobConfig")
        if not job_config.output_s3_path:
            raise ValueError("output_s3_path (output_path) is required in DataPrepJobConfig")

        return job_config

    @staticmethod
    def _build_job_name(pipeline_id: str) -> str:
        """Build a SageMaker-compliant training job name from the pipeline id.

        SageMaker job names must match ``[a-zA-Z0-9](-*[a-zA-Z0-9]){0,62}``.
        Prefix ``nova-forge-dataprep-`` (20) + ``-`` + 10-digit epoch = 31
        chars, leaving 32 chars for the sanitized pipeline id.
        """
        sanitized_id = pipeline_id.replace("_", "-")[:32].rstrip("-")
        return f"nova-forge-dataprep-{sanitized_id}-{int(time.time())}"

    def _build_hyperparameters(
        self, job_config: "DataPrepJobConfig", pipeline_id: str
    ) -> Dict[str, str]:
        """Build the HyperParameters dict forwarded to the training job.

        SageMaker requires all hyperparameter values to be strings, so
        ``extra_args`` and the merged pip-install list are JSON-encoded.
        """
        # Narrow the Optional[str] fields from the base JobConfig —
        # _validate_dataprep_config already guaranteed these are non-empty.
        assert job_config.data_s3_path is not None
        assert job_config.output_s3_path is not None

        forwarded_args = {k: v for k, v in job_config.extra_args.items() if k != "pipeline_id"}
        hyperparameters: Dict[str, str] = {
            "pipeline_id": pipeline_id,
            "input_path": job_config.data_s3_path,
            "output_path": job_config.output_s3_path,
            "input_format": job_config.input_format,
            "output_format": job_config.output_format,
            "text_field": job_config.text_field,
        }
        if forwarded_args:
            hyperparameters["extra_args"] = json.dumps(forwarded_args)

        merged_pip = self._merge_pip_install(job_config.extra_pip_packages)
        if merged_pip:
            hyperparameters["pip_install"] = json.dumps(merged_pip)

        return hyperparameters

    @staticmethod
    def _merge_pip_install(caller_extras: Optional[List[str]]) -> List[str]:
        """Merge caller pip extras with the DLC baseline, deduped by base name.

        Caller entries win on base-name collision so a caller
        ``"ray==2.40.0"`` pin isn't shadowed by our default
        ``"ray[default]==2.35.0"``.
        """

        def _base(pkg: str) -> str:
            return pkg.split("[")[0].split("=")[0].split("<")[0].split(">")[0]

        caller_pip = list(caller_extras or [])
        caller_bases = {_base(p) for p in caller_pip}
        merged: List[str] = list(caller_pip)
        for pkg in _DLC_REQUIRED_PACKAGES:
            if _base(pkg) not in caller_bases:
                merged.append(pkg)
        return merged

    def _build_create_training_job_params(
        self,
        job_config: "DataPrepJobConfig",
        job_name: str,
        hyperparameters: Dict[str, str],
    ) -> Dict[str, Any]:
        """Assemble the kwargs for ``sagemaker.create_training_job``."""
        resource_config: Dict[str, Any] = {
            "InstanceType": self.instance_type,
            "InstanceCount": self.instance_count,
            "VolumeSizeInGB": self.volume_size_gb,
        }
        if self.keep_alive_seconds > 0:
            resource_config["KeepAlivePeriodInSeconds"] = self.keep_alive_seconds

        create_params: Dict[str, Any] = {
            "TrainingJobName": job_name,
            "RoleArn": self.execution_role,
            "AlgorithmSpecification": {
                "TrainingImage": self.image_uri,
                "TrainingInputMode": "File",
                "ContainerEntrypoint": [
                    "python",
                    "/opt/ml/input/data/code/invoke_smtj_pipeline.py",
                ],
            },
            "ResourceConfig": resource_config,
            "OutputDataConfig": {
                "S3OutputPath": job_config.output_s3_path,
            },
            "StoppingCondition": {
                "MaxRuntimeInSeconds": self.max_job_runtime,
            },
            "HyperParameters": hyperparameters,
            "InputDataConfig": [
                {
                    "ChannelName": "code",
                    "DataSource": {
                        "S3DataSource": {
                            "S3DataType": "S3Prefix",
                            "S3Uri": self.script_s3_path,
                            "S3DataDistributionType": "FullyReplicated",
                        }
                    },
                },
                {
                    "ChannelName": "deps",
                    "DataSource": {
                        "S3DataSource": {
                            "S3DataType": "S3Prefix",
                            "S3Uri": self.deps_s3_prefix,
                            "S3DataDistributionType": "FullyReplicated",
                        }
                    },
                },
            ],
        }

        if self.kms_key_id:
            create_params["OutputDataConfig"]["KmsKeyId"] = self.kms_key_id
            resource_config["VolumeKmsKeyId"] = self.kms_key_id

        vpc_config = self._build_vpc_config()
        if vpc_config:
            create_params["VpcConfig"] = vpc_config
            if not self._is_standard_ecr_image(self.image_uri):
                create_params["AlgorithmSpecification"]["TrainingImageConfig"] = {
                    "TrainingRepositoryAccessMode": "Vpc",
                }

        return create_params

    def _build_vpc_config(self) -> Dict[str, Any]:
        """Return the VpcConfig dict for the training job, or ``{}`` if no VPC is set."""
        vpc_config: Dict[str, Any] = {}
        if self.subnets:
            vpc_config["Subnets"] = self.subnets
        if self.security_group_ids:
            vpc_config["SecurityGroupIds"] = self.security_group_ids
        return vpc_config

    @staticmethod
    def _is_standard_ecr_image(image_uri: str) -> bool:
        """Return True if the image is a standard private ECR image (account.dkr.ecr.region.amazonaws.com/...)."""
        parsed = urlparse(image_uri)
        host = parsed.hostname
        # Handle schemeless image URIs such as:
        # 123456789012.dkr.ecr.us-west-2.amazonaws.com/repo:tag
        if not host:
            host = parsed.path.split("/", 1)[0]
        if not host:
            return False
        return bool(
            re.fullmatch(
                r"\d{12}\.dkr\.ecr\.[a-z0-9-]+\.amazonaws\.com(?:\.cn)?",
                host,
            )
        )

    def _raise_if_not_completed(self, job_name: str, status: str) -> None:
        """Raise ``RuntimeError`` with the FailureReason if the job didn't succeed."""
        if status == "Completed":
            return
        resp = self.sagemaker_client.describe_training_job(TrainingJobName=job_name)
        failure_reason = resp.get("FailureReason", "")
        raise RuntimeError(
            f"SMTJ data prep job {job_name!r} failed with status {status}: {failure_reason}"
        )

    def _poll_until_terminal(self, job_name: str) -> str:
        """Wait for SageMaker training job to reach a terminal state using a waiter."""
        from botocore.exceptions import WaiterError

        waiter = self.sagemaker_client.get_waiter("training_job_completed_or_stopped")
        max_attempts = max(1, self.max_wait_time // self.poll_interval)

        logger.info(
            "Waiting for training job to complete (polling every %ds, max %ds)...",
            self.poll_interval,
            self.max_wait_time,
        )
        try:
            waiter.wait(
                TrainingJobName=job_name,
                WaiterConfig={
                    "Delay": self.poll_interval,
                    "MaxAttempts": max_attempts,
                },
            )
        except WaiterError:
            # Waiter exceeded max attempts — check if the job actually
            # reached a terminal state (e.g. Failed) that the built-in
            # waiter doesn't recognise as an acceptor.
            resp = self.sagemaker_client.describe_training_job(TrainingJobName=job_name)
            status = resp["TrainingJobStatus"]
            if status in self._TERMINAL_STATES:
                return status
            raise TimeoutError(
                f"SMTJ data prep job {job_name!r} did not reach a terminal state "
                f"within {self.max_wait_time}s (last status: {status})"
            )

        # Waiter succeeded — job is Completed or Stopped.
        resp = self.sagemaker_client.describe_training_job(TrainingJobName=job_name)
        return resp["TrainingJobStatus"]

    @_telemetry_emitter(Feature.DATA_PREP, "smtj_dataprep_cleanup")
    def cleanup(self, job_name: str) -> None:
        """Stop a running SageMaker training job."""
        try:
            self.sagemaker_client.stop_training_job(TrainingJobName=job_name)
            logger.info("Stopped SMTJ data prep job %r", job_name)
        except Exception as e:
            logger.error("Failed to stop SMTJ data prep job %r: %s", job_name, e)
            raise

    @classmethod
    def required_calling_role_permissions(cls, data_s3_path=None, output_s3_path=None) -> List:
        """Required IAM permissions for SMTJ data prep operations."""
        permissions = super().required_calling_role_permissions(data_s3_path, output_s3_path)

        permissions.extend(
            [
                ("sagemaker:CreateTrainingJob", "*"),
                ("sagemaker:DescribeTrainingJob", "*"),
                ("sagemaker:StopTrainingJob", "*"),
                ("iam:GetRole", "*"),
                ("iam:PassRole", "*"),
                # Artifact bucket: auto-create, check existence, upload script + .whl
                ("s3:CreateBucket", "*"),
                ("s3:HeadBucket", "*"),
                ("s3:PutObject", "*"),
            ]
        )

        return permissions


# TODO: Might need to take RIG as input in case of multiple RIGs
class SMHPRuntimeManager(RuntimeManager):
    def __init__(
        self,
        instance_type: str,
        instance_count: int,
        cluster_name: str,
        namespace: str,
        kms_key_id: Optional[str] = None,
        rft_lambda: Optional[str] = None,
    ):
        from amzn_nova_forge.validation.validator import Validator

        Validator.validate_cluster_name(cluster_name=cluster_name)
        Validator.validate_namespace(namespace=namespace)

        self.cluster_name = cluster_name
        self.namespace = namespace
        self.execution_role = None
        super().__init__(instance_type, instance_count, kms_key_id, rft_lambda)
        self.setup()

    @classmethod
    def required_calling_role_permissions(cls, data_s3_path=None, output_s3_path=None):
        """Required permissions for HyperPod operations."""
        # Start with base S3 permissions
        permissions = super().required_calling_role_permissions(data_s3_path, output_s3_path)

        # Add SMHP-specific permissions
        permissions.extend(
            [
                (
                    "sagemaker:DescribeCluster",
                    lambda infra: (
                        f"arn:aws:sagemaker:{infra.region}:{_get_caller_account_id(infra.region)}:cluster/{infra.cluster_name}"
                    ),
                ),
                (
                    "eks:DescribeCluster",
                    lambda infra: (
                        f"arn:aws:eks:{infra.region}:{_get_caller_account_id(infra.region)}:cluster/*"
                    ),
                ),
                (
                    "eks:ListAddons",
                    lambda infra: (
                        f"arn:aws:eks:{infra.region}:{_get_caller_account_id(infra.region)}:cluster/{infra.cluster_name}"
                    ),
                ),
                ("sagemaker:ListClusters", "*"),
            ]
        )

        return permissions

    @property
    def platform(self) -> Platform:
        return Platform.SMHP

    @property
    def runtime_name(self) -> str:
        return "SageMaker HyperPod"

    def setup(self) -> None:
        boto_session = boto3.session.Session()
        self.region = boto_session.region_name or "us-east-1"

        response = subprocess.run(
            [
                "hyperpod",
                "connect-cluster",
                "--cluster-name",
                self.cluster_name,
                "--namespace",
                self.namespace,
            ],
            capture_output=True,
            text=True,
            check=True,
        )

        _check_hyperpod_stderr(response.stderr)

        logger.info(
            f"Successfully connected to HyperPod cluster '{self.cluster_name}' in namespace '{self.namespace}'."
        )

    def execute(self, job_config: JobConfig) -> str:
        try:
            # Scrub recipe path so that it will be recognized by the HyperPod CLI
            recipe_path = (
                job_config.recipe_path.split(HYPERPOD_RECIPE_PATH, 1)[1]
                .lstrip("/")
                .lstrip("\\")
                .removesuffix(".yaml")
            )

            override_parameters = json.dumps(
                {
                    "instance_type": self.instance_type,
                    "container": job_config.image_uri,
                }
            )
            response = subprocess.run(
                [
                    "hyperpod",
                    "start-job",
                    "--namespace",
                    self.namespace,
                    "--recipe",
                    recipe_path,
                    "--override-parameters",
                    override_parameters,
                ],
                capture_output=True,
                text=True,
                check=True,
            )

            if matched_job_name := re.search(r"NAME: (\S+)", response.stdout):
                return matched_job_name.group(1)
            raise ValueError(
                f"Could not find job name in output. There may be an issue with the helm installation, "
                f"assumed role permissions to trigger jobs on the cluster, or job specification. Output: {response.stdout}"
            )
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to start HyperPod job: {e.stderr}")
            raise

    def cleanup(self, job_name: str) -> None:
        from amzn_nova_forge.validation.validator import Validator

        Validator.validate_job_name(job_name=job_name)

        try:
            response = subprocess.run(
                [
                    "hyperpod",
                    "cancel-job",
                    "--job-name",
                    job_name,
                    "--namespace",
                    self.namespace,
                ],
                capture_output=True,
                text=True,
                check=True,
            )

            try:
                _check_hyperpod_stderr(response.stderr)
            except RuntimeError as e:
                logger.error(f"Failed to cleanup HyperPod job '{job_name}': {e}")

        except Exception as e:
            logger.error(f"Failed to cleanup HyperPod job '{job_name}': {str(e)}")
            raise

    def scale_cluster(
        self,
        instance_group_name: str,
        target_instance_count: int,
    ) -> Dict[str, Any]:
        """
        Scale a HyperPod cluster RIG up or down. The scaling is asynchronous.
        The cluster status will change to 'Updating' while scaling, and 'InService' when ready.

        Args:
            instance_group_name: Name of the instance group to scale (e.g., 'worker-group')
            target_instance_count: Desired number of instances for the group

        Returns:
            dict: Response containing:
                - ClusterArn: ARN of the updated cluster
                - InstanceGroupName: Name of the scaled instance group
                - InstanceType: Instance type being scaled
                - PreviousCount: Current instance count
                - TargetCount: Target instance count

        Raises:
            ValueError: If target_instance_count is negative
            ClientError: If scaling fails due to insufficient quota, capacity, or cluster issues.
        """
        if target_instance_count < 0:
            raise ValueError(
                f"target_instance_count must be non-negative, got {target_instance_count}"
            )

        sagemaker_client = boto3.client("sagemaker", region_name=self.region)

        # Get current cluster configuration
        try:
            describe_response = sagemaker_client.describe_cluster(ClusterName=self.cluster_name)
        except ClientError as e:
            logger.error(f"Failed to describe cluster '{self.cluster_name}': {e}")
            raise

        # Check if cluster is in InService state
        cluster_status = describe_response.get("ClusterStatus")
        if cluster_status != "InService":
            error_msg = (
                f"Cluster '{self.cluster_name}' is in '{cluster_status}' state. "
                f"Scaling is only allowed when cluster is in 'InService' state."
            )
            logger.error(error_msg)
            raise ValueError(error_msg)

        # Find the Restricted Instance Group (RIG)
        restricted_instance_groups = describe_response.get("RestrictedInstanceGroups", [])
        target_group = None

        for group in restricted_instance_groups:
            if group["InstanceGroupName"] == instance_group_name:
                target_group = group
                break

        if not target_group:
            error_msg = (
                f"Instance group '{instance_group_name}' not found in cluster '{self.cluster_name}'. "
                f"Available groups: {[g['InstanceGroupName'] for g in restricted_instance_groups]}"
            )
            logger.error(error_msg)
            raise ValueError(error_msg)

        instance_type = target_group["InstanceType"]
        current_count = target_group["CurrentCount"]

        logger.info(
            f"Scaling instance group '{instance_group_name}' "
            f"({instance_type}) from {current_count} to {target_instance_count} instances"
        )

        # Update the cluster with new instance count.
        try:
            # Optional fields to preserve from describe_cluster response
            optional_fields = [
                "ThreadsPerCore",
                "InstanceStorageConfigs",
                "OnStartDeepHealthChecks",
                "TrainingPlanArn",
                "OverrideVpcConfig",
                "ScheduledUpdateConfig",
                "EnvironmentConfig",
            ]

            all_groups = []
            for group in restricted_instance_groups:
                group_params = {
                    "InstanceCount": (
                        target_instance_count
                        if group["InstanceGroupName"] == instance_group_name
                        else group["CurrentCount"]
                    ),
                    "InstanceGroupName": group["InstanceGroupName"],
                    "InstanceType": group["InstanceType"],
                    "ExecutionRole": group["ExecutionRole"],
                }

                # Copy optional fields if present
                for field in optional_fields:
                    if field in group:
                        if field == "EnvironmentConfig":
                            env_config = {}
                            # Only pass FSxLustreConfig which is accepted by the UpdateCluster API
                            if "FSxLustreConfig" in group["EnvironmentConfig"]:
                                env_config["FSxLustreConfig"] = group["EnvironmentConfig"][
                                    "FSxLustreConfig"
                                ]
                            group_params[field] = env_config
                        else:
                            group_params[field] = group[field]

                all_groups.append(group_params)

            update_response = sagemaker_client.update_cluster(
                ClusterName=self.cluster_name,
                RestrictedInstanceGroups=all_groups,
            )

            logger.info(f"Successfully initiated scaling for cluster '{self.cluster_name}'. ")

            return {
                "ClusterArn": update_response["ClusterArn"],
                "InstanceGroupName": instance_group_name,
                "InstanceType": instance_type,
                "PreviousCount": current_count,
                "TargetCount": target_instance_count,
            }
        except ClientError as e:
            error_code = e.response.get("Error", {}).get("Code", "Unknown")
            error_message = e.response.get("Error", {}).get("Message", str(e))
            logger.error(f"Failed to scale cluster: {error_code} - {error_message}")
            raise

    def get_instance_groups(self) -> List[Dict[str, Any]]:
        """
        Get a list of available Restricted Instance Groups (RIGs) in the cluster.

        Returns:
            list: List of dicts containing instance group information:
                - InstanceGroupName: Name of the instance group
                - InstanceType: EC2 instance type (e.g., 'ml.p5.48xlarge')
                - CurrentCount: Current number of instances in the group

        Raises:
            ClientError: If unable to describe the cluster
        """
        sagemaker_client = boto3.client("sagemaker", region_name=self.region)

        # Get current cluster configuration
        try:
            describe_response = sagemaker_client.describe_cluster(ClusterName=self.cluster_name)
        except ClientError as e:
            logger.error(f"Failed to describe cluster '{self.cluster_name}': {e}")
            raise

        # Get the RIGs and extract necessary information
        instance_groups = describe_response.get("RestrictedInstanceGroups", [])

        rig_output = [
            {
                "InstanceGroupName": group["InstanceGroupName"],
                "InstanceType": group["InstanceType"],
                "CurrentCount": group["CurrentCount"],
            }
            for group in instance_groups
        ]

        # Log output to terminal
        logger.info(f"Found {len(rig_output)} instance group(s) in cluster '{self.cluster_name}':")
        for group in rig_output:
            logger.info(
                f"  - {group['InstanceGroupName']}: {group['InstanceType']} "
                f"(Current: {group['CurrentCount']} instances)"
            )

        return rig_output


class BedrockRuntimeManager(RuntimeManager):
    """
    Runtime manager for AWS Bedrock model customization jobs.
    """

    def __init__(
        self,
        execution_role: str,
        base_model_identifier: Optional[str] = None,
        kms_key_id: Optional[str] = None,
        vpc_config: Optional[Dict[str, list[str]]] = None,
        rft_lambda: Optional[str] = None,
    ):
        # Store Bedrock-specific configuration
        self.execution_role = execution_role
        self.base_model_identifier = base_model_identifier
        self.vpc_config = vpc_config

        # Calls constructor with None for instance_type and instance_count
        # since Bedrock manages compute resources automatically
        super().__init__(
            instance_type=None,
            instance_count=None,
            kms_key_id=kms_key_id,
            rft_lambda=rft_lambda,
        )

        self.setup()

    @property
    def platform(self) -> Platform:
        return Platform.BEDROCK

    @property
    def runtime_name(self) -> str:
        return "Bedrock"

    def setup(self) -> None:
        """Initialize Bedrock client and session.

        Creates a boto3 session, extracts the region, and initializes the Bedrock client.
        Falls back to us-east-1 if no region is configured.

        Raises:
            Exception: If Bedrock client initialization fails
        """
        try:
            boto_session = boto3.session.Session()
            self.region = boto_session.region_name or "us-east-1"
            self.bedrock_client = boto3.client("bedrock", region_name=self.region)
            logger.info(f"Successfully initialized Bedrock client in region {self.region}")
        except Exception as e:
            logger.error(f"Failed to initialize Bedrock client: {e}")
            raise

    def execute(self, job_config: JobConfig) -> str:
        try:
            # Validate job name
            from amzn_nova_forge.validation.validator import Validator

            Validator.validate_job_name(job_name=job_config.job_name)
            logger.info(f"Starting Bedrock customization job: {job_config.job_name}")

            # Get training method from job_config (passed via nova_model_customizer)
            # For Bedrock, we need the method to determine customization type
            if job_config.method is None:
                raise ValueError(
                    "Training method must be provided in job_config for Bedrock jobs. "
                    "This should be set automatically by NovaModelCustomizer."
                )
            method = job_config.method

            # Extract hyperparameters from recipe if provided
            hyperparameters: Dict[str, str] = {}
            rft_hyperparameters: Dict[
                str, Any
            ] = {}  # RFT hyperparameters use native types (int, float, str)
            if job_config.recipe_path is not None:
                recipe_data = parse_bedrock_recipe_config(job_config.recipe_path, method)
                hyperparameters = recipe_data["hyperparameters"]
                rft_hyperparameters = recipe_data["rft_hyperparameters"]
            else:
                logger.info("No recipe provided - Bedrock will use default hyperparameters")

            # Get customization type from training method
            customization_type = get_customization_type(method)

            # Resolve base model identifier
            if job_config.recipe_path is not None:
                base_model_identifier = resolve_base_model_identifier(
                    job_config.recipe_path, self.base_model_identifier, self.region
                )
            elif self.base_model_identifier is not None:
                base_model_identifier = self.base_model_identifier
            else:
                raise ValueError(
                    "base_model_identifier must be provided either in BedrockRuntimeManager "
                    "constructor or via recipe_path. Cannot determine which model to use."
                )

            # Use job_name as custom_model_name
            custom_model_name = job_config.job_name

            # Build API request dictionary with required fields
            api_request: Dict[str, Any] = {
                "jobName": job_config.job_name,
                "customModelName": custom_model_name,
                "roleArn": self.execution_role,
                "baseModelIdentifier": base_model_identifier,
                "customizationType": customization_type,
            }

            # Add hyperParameters only if provided (optional parameter)
            if hyperparameters:
                api_request["hyperParameters"] = hyperparameters

            # Add trainingDataConfig (REQUIRED by Bedrock API)
            if job_config.data_s3_path is None:
                raise ValueError(
                    "data_s3_path is required for Bedrock customization jobs. "
                    "Bedrock API requires trainingDataConfig.s3Uri to be provided."
                )

            training_data_config: Dict[str, str] = {"s3Uri": job_config.data_s3_path}
            api_request["trainingDataConfig"] = training_data_config

            # Add outputDataConfig with output_s3_path (required)
            if job_config.output_s3_path is None:
                raise ValueError("output_s3_path is required for Bedrock customization jobs")

            output_data_config: Dict[str, str] = {"s3Uri": job_config.output_s3_path}

            # Add KMS key to outputDataConfig only if provided (optional)
            if self.kms_key_id is not None:
                output_data_config["kmsKeyId"] = self.kms_key_id

            api_request["outputDataConfig"] = output_data_config

            # Add validationDataConfig only if validation_data_s3_path is provided (optional)
            if job_config.validation_data_s3_path is not None:
                # Warn that Nova Lite 2 doesn't support validation datasets
                if "nova-2-lite" in base_model_identifier or "nova-lite-2" in base_model_identifier:
                    logger.warning(
                        "Validation datasets are not supported for Nova Lite 2 models on Bedrock. "
                        "The validation_data_s3_path parameter will be ignored. "
                        "To use validation datasets, please use a different model (Nova Micro, Nova Lite, or Nova Pro)."
                    )
                else:
                    validation_data_config: Dict[str, List[Dict[str, str]]] = {
                        "validators": [{"s3Uri": job_config.validation_data_s3_path}]
                    }
                    api_request["validationDataConfig"] = validation_data_config

            # Add vpcConfig only if provided (optional)
            if self.vpc_config is not None:
                vpc_config: Dict[str, List[str]] = {}
                if "subnet_ids" in self.vpc_config:
                    vpc_config["subnetIds"] = self.vpc_config["subnet_ids"]
                if "security_group_ids" in self.vpc_config:
                    vpc_config["securityGroupIds"] = self.vpc_config["security_group_ids"]

                if vpc_config:  # Only add if we have at least one field
                    api_request["vpcConfig"] = vpc_config

            # Add customizationConfig for RFT jobs only if Lambda ARN provided (optional but required for RFT)
            if customization_type == "REINFORCEMENT_FINE_TUNING":
                if not job_config.rft_lambda_arn:
                    raise ValueError(
                        "rft_lambda_arn is required for RFT (Reinforcement Fine-Tuning) jobs. "
                        "Please provide the Lambda ARN in train() method."
                    )

                rft_config: Dict[str, Any] = {
                    "graderConfig": {"lambdaGrader": {"lambdaArn": job_config.rft_lambda_arn}}
                }

                # Add RFT hyperparameters if provided (native types: int, float, string)
                if rft_hyperparameters:
                    rft_config["hyperParameters"] = rft_hyperparameters

                customization_config: Dict[str, Any] = {"rftConfig": rft_config}
                api_request["customizationConfig"] = customization_config

            # Call Bedrock API to create customization job
            response = self.bedrock_client.create_model_customization_job(**api_request)

            # Extract and return job ARN
            job_arn = response["jobArn"]

            return job_arn

        except ClientError as e:
            error_code = e.response["Error"]["Code"]
            error_message = e.response["Error"]["Message"]
            logger.error(f"Bedrock API error ({error_code}): {error_message}")

            if error_code == "ValidationException":
                raise ValueError(f"Invalid Bedrock job configuration: {error_message}") from e
            elif error_code == "AccessDeniedException":
                raise PermissionError(f"Bedrock access denied: {error_message}") from e
            elif error_code == "ResourceNotFoundException":
                raise ValueError(f"Bedrock resource not found: {error_message}") from e
            elif error_code == "ServiceQuotaExceededException":
                raise RuntimeError(f"Bedrock service quota exceeded: {error_message}") from e
            elif error_code == "ThrottlingException":
                raise RuntimeError(f"Bedrock API throttled: {error_message}") from e
            else:
                raise RuntimeError(f"Bedrock API error: {error_message}") from e

        except ValueError as e:
            # Re-raise ValueError (from validation, unsupported methods, etc.)
            logger.error(f"Validation error: {e}")
            raise

        except FileNotFoundError as e:
            # Re-raise FileNotFoundError (from recipe parsing)
            logger.error(f"Recipe file not found: {e}")
            raise

        except Exception as e:
            # Catch-all for unexpected errors
            logger.error(f"Unexpected error creating Bedrock customization job: {e}")
            raise RuntimeError(f"Failed to create Bedrock customization job: {e}") from e

    def cleanup(self, job_id: str) -> None:
        try:
            logger.info(f"Stopping Bedrock customization job: {job_id}")

            # Validate job_id format (should be a Bedrock job ARN)
            if not job_id or not isinstance(job_id, str):
                raise ValueError(f"Invalid job_id: must be a non-empty string, got {type(job_id)}")

            if not job_id.startswith("arn:aws:bedrock:"):
                logger.warning(
                    f"job_id does not appear to be a valid Bedrock ARN: {job_id}. "
                    "Expected format: arn:aws:bedrock:region:account:model-customization-job/job-id"
                )

            # Call Bedrock API to stop the customization job
            self.bedrock_client.stop_model_customization_job(jobIdentifier=job_id)
            logger.info(f"Successfully stopped Bedrock customization job: {job_id}")

        except ClientError as e:
            error_code = e.response["Error"]["Code"]
            error_message = e.response["Error"]["Message"]
            logger.error(f"Bedrock API error while stopping job ({error_code}): {error_message}")

            if error_code == "ValidationException":
                raise ValueError(f"Invalid job ARN: {error_message}") from e
            elif error_code == "ResourceNotFoundException":
                raise ValueError(f"Job not found: {error_message}") from e
            elif error_code == "AccessDeniedException":
                raise PermissionError(
                    f"Insufficient permissions to stop job: {error_message}"
                ) from e
            elif error_code == "ConflictException":
                # Job may already be stopped or in a terminal state
                logger.warning(f"Job may already be stopped: {error_message}")
            else:
                raise RuntimeError(f"Failed to stop Bedrock job: {error_message}") from e

        except Exception as e:
            logger.error(f"Unexpected error stopping Bedrock job: {e}")
            raise RuntimeError(f"Failed to stop Bedrock customization job: {e}") from e

        finally:
            # Close bedrock_client connection
            try:
                if hasattr(self, "bedrock_client") and self.bedrock_client is not None:
                    self.bedrock_client.close()
                    logger.info("Closed Bedrock client connection")
            except Exception as e:
                logger.warning(f"Error closing Bedrock client: {e}")

    @classmethod
    def required_calling_role_permissions(cls, data_s3_path=None, output_s3_path=None):
        """Return required IAM permissions for Bedrock operations."""
        # Start with base S3 permissions
        permissions = super().required_calling_role_permissions(data_s3_path, output_s3_path)

        # Add Bedrock-specific permissions
        permissions.extend(
            [
                ("bedrock:CreateModelCustomizationJob", "*"),
                ("bedrock:StopModelCustomizationJob", "*"),
                ("bedrock:GetModelCustomizationJob", "*"),
                "iam:PassRole",
            ]
        )

        return permissions


class SMTJServerlessRuntimeManager(MTRLOperations, RuntimeManager):
    def __init__(
        self,
        model_package_group_name: str,
        execution_role: Optional[str] = None,
        kms_key_id: Optional[str] = None,
        encrypt_inter_container_traffic: bool = False,
        subnets: Optional[list[str]] = None,
        security_group_ids: Optional[list[str]] = None,
        max_job_runtime: Optional[int] = DEFAULT_SMTJ_JOB_MAX_RUNTIME,  # 1 day
        rft_lambda: Optional[str] = None,
        agent_core_arn: Optional[str] = None,
        evaluator_name: Optional[str] = None,
        hub_content_version: Optional[str] = None,
        intermediate_model_package_group_name: Optional[str] = None,
    ):
        # NOTE: Not setting execution_role directly due to issues with mypy type inference
        self._execution_role = execution_role
        self.model_package_group_name = model_package_group_name
        self.intermediate_model_package_group_name = intermediate_model_package_group_name
        self.agent_core_arn = agent_core_arn
        self.evaluator_name = evaluator_name
        self.hub_content_version: Optional[str] = hub_content_version
        self.subnets = subnets
        self.security_group_ids = security_group_ids
        self.encrypt_inter_container_traffic = encrypt_inter_container_traffic
        self.max_job_runtime = max_job_runtime

        super().__init__(None, None, kms_key_id=kms_key_id, rft_lambda=rft_lambda)
        self.setup()

    @classmethod
    def required_calling_role_permissions(cls, data_s3_path=None, output_s3_path=None):
        """Required permissions for SMTJ calling role operations and execution role validation."""
        # Start with base S3 permissions
        permissions = super().required_calling_role_permissions(data_s3_path, output_s3_path)

        # Add SMTJ-specific permissions
        permissions.extend(
            [
                ("sagemaker:CreateTrainingJob", "*"),
                ("sagemaker:DescribeTrainingJob", "*"),
                "iam:GetRole",
                "iam:PassRole",
                "iam:GetPolicy",
                "iam:GetPolicyVersion",
                "iam:ListRolePolicies",
                "iam:GetRolePolicy",
                "iam:ListAttachedRolePolicies",
            ]
        )

        return permissions

    @property
    def platform(self) -> Platform:
        return Platform.SMTJServerless

    @property
    def runtime_name(self) -> str:
        return "SageMaker Serverless"

    def setup(self) -> None:
        boto_session = boto3.session.Session()
        self.region = boto_session.region_name or "us-east-1"
        self.sagemaker_client = boto3.client("sagemaker", region_name=self.region)
        self.sagemaker_session = Session(
            boto_session=boto_session, sagemaker_client=self.sagemaker_client
        )

        if self._execution_role is None:
            self.execution_role = get_execution_role(use_default=True)
        else:
            self.execution_role = self._execution_role
        # Delete temporary attribute so customers don't confuse it with the actual attribute
        del self._execution_role

        # Create model package group if it doesn't exist
        try:
            resp = self.sagemaker_client.describe_model_package_group(
                ModelPackageGroupName=self.model_package_group_name
            )
            self.model_package_group_arn = resp["ModelPackageGroupArn"]
        except self.sagemaker_client.exceptions.ClientError:
            resp = self.sagemaker_client.create_model_package_group(
                ModelPackageGroupName=self.model_package_group_name
            )
            self.model_package_group_arn = resp["ModelPackageGroupArn"]

    def _get_or_create_checkpoint_model_package_group_arn(self) -> str:
        """Get or create a separate model package group for intermediate checkpoints."""
        checkpoint_group_name = (
            self.intermediate_model_package_group_name
            or f"{self.model_package_group_name}-checkpoints"[:63]
        )
        try:
            resp = self.sagemaker_client.describe_model_package_group(
                ModelPackageGroupName=checkpoint_group_name
            )
            return resp["ModelPackageGroupArn"]
        except self.sagemaker_client.exceptions.ClientError:
            resp = self.sagemaker_client.create_model_package_group(
                ModelPackageGroupName=checkpoint_group_name
            )
            return resp["ModelPackageGroupArn"]

    def _resolve_base_model_arn(self, model: Model) -> str:
        """Resolve the BaseModelArn from SageMaker Hub for the given model."""
        hub_content = get_hub_content(
            hub_name="SageMakerPublicHub",
            hub_content_name=model.hub_content_name,
            hub_content_type="Model",
            region=self.region,
            hub_content_version=self.hub_content_version,
        )
        return hub_content["HubContentArn"]

    def _register_lambda_as_hub_content(self, lambda_arn: str) -> str:
        """Register a Lambda ARN as a JsonDoc hub-content and return the hub-content ARN."""
        hub_name = re.sub(r"[^a-zA-Z0-9-]", "-", self.model_package_group_name)[:63]
        return register_lambda_as_hub_content(
            lambda_arn=lambda_arn,
            hub_name=hub_name,
            sagemaker_client=self.sagemaker_client,
            evaluator_name=self.evaluator_name,
        )

    @_telemetry_emitter(Feature.INFRA, "validate_lambda")
    def validate_lambda(
        self,
        data_s3_path: str,
        validation_samples: int = 10,
    ) -> None:
        """Override to support hub-content ARNs by extracting the Lambda ARN first."""
        lambda_arn = self.rft_lambda_arn
        if lambda_arn and _is_hub_content_arn(lambda_arn):
            extracted = extract_lambda_arn_from_hub_content(lambda_arn, self.sagemaker_client)
            if not extracted:
                logger.warning(
                    f"Hub-content document does not contain a valid Lambda ARN. "
                    "Skipping validation."
                )
                return
            logger.info(f"Extracted Lambda ARN from hub-content: {extracted}")
            # Temporarily set rft_lambda_arn to the extracted Lambda ARN for validation
            original_arn = self._rft_lambda_arn
            self._rft_lambda_arn = extracted
            try:
                super().validate_lambda(data_s3_path, validation_samples)
            finally:
                self._rft_lambda_arn = original_arn
        else:
            super().validate_lambda(data_s3_path, validation_samples)

    def _extract_hyperparameters(
        self,
        recipe: Dict[str, Any],
        method: Optional[TrainingMethod] = None,
        data_mixing_config: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, str]:
        """Extract hyperparameters from recipe as string key-value pairs.

        Args:
            recipe: The rendered recipe dict.
            method: Training method — determines the HyperParameter key for max_length.
                    RFT uses "max_length"; all others use "max_context_length".
        """
        is_rft = method in (TrainingMethod.RFT_LORA, TrainingMethod.RFT_FULL)
        max_length_hp_key = "max_length" if is_rft else "max_context_length"
        # Maps recipe key → HyperParameter key sent to the serverless API.
        hyperparameter_map = {
            # --- Shared across all models/methods ---
            "global_batch_size": "global_batch_size",
            "warmup_steps": "warmup_steps",
            "reasoning_effort": "reasoning_effort",
            "top_logprobs": "top_logprobs",
            "name": "name",
            # --- Learning rate: all recipes use "lr" as the recipe key ---
            "lr": "learning_rate",
            # --- LoRA ratio: v1 uses "loraplus_lr_ratio", v2 uses "lora_plus_lr_ratio" ---
            "loraplus_lr_ratio": "learning_rate_ratio",  # v1 (MICRO, LITE, PRO) recipe key
            "lora_plus_lr_ratio": "learning_rate_ratio",  # v2 (LITE_2) SFT/RFT recipe key
            # --- LoRA alpha: all recipes use "alpha" as the recipe key ---
            "alpha": "lora_alpha",
            # --- Sequence length: recipe key is "max_length" for all methods.
            #     HyperParameter key differs: RFT uses "max_length", SFT/DPO use "max_context_length" ---
            "max_length": max_length_hp_key,
            # --- Training duration: v1 uses "max_epochs", v2 uses "max_steps" ---
            "max_epochs": "max_epochs",  # v1 SFT/DPO
            "max_steps": "max_steps",  # v2 SFT/RFT
            "save_steps": "save_steps",  # v2 SFT/RFT
            # --- v2 SFT only ---
            "reasoning_enabled": "reasoning_enabled",
            # --- DPO only: recipe key is "beta" (under dpo_cfg.beta) ---
            "beta": "adam_beta",
            # --- Validation interval ---
            "val_check_interval": "val_check_interval",
            "fine_tuned_model": "fine_tuned_model",  # v2 SFT/RFT
        }

        result: Dict[str, str] = {}

        def _extract(obj: Dict[str, Any]) -> None:
            for k, v in obj.items():
                if k in hyperparameter_map and v is not None:
                    result[hyperparameter_map[k]] = str(v)
                elif isinstance(v, dict):
                    _extract(v)

        _extract(recipe)

        # Datamix: inject percent fields directly from DataMixing.get_config() (full field names
        # like customer_data_percent, nova_agents_percent). The recipe structure nests these under
        # data_mixing.sources.* with short keys, so we use the config dict as the source of truth.
        if data_mixing_config:
            for k, v in data_mixing_config.items():
                result[k] = str(v)

        logger.info(f"HyperParameters used for the job: {result}")
        return result

    def _build_serverless_job_config(
        self,
        method: TrainingMethod,
        base_model_arn: str,
        eval_task: Optional[str] = None,
        evaluator_arn: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Build ServerlessJobConfig for create_training_job."""
        logger.warning("Accepting End-User License Agreement by using SMTJ Serverless")
        config: Dict[str, Any] = {
            "BaseModelArn": base_model_arn,
            "AcceptEula": True,
        }
        if method == TrainingMethod.EVALUATION:
            config["JobType"] = "Evaluation"
            # Map eval task to API EvaluationType enum
            # Mapping based on serverless API contract:
            # BenchmarkEvaluation: built-in benchmarks + llm_judge/rubric_llm_judge
            if eval_task in SERVERLESS_CUSTOM_SCORER_EVAL_TASKS:
                config["EvaluationType"] = "CustomScorerEvaluation"
            else:
                config["EvaluationType"] = "BenchmarkEvaluation"
            # EvaluatorArn is NOT used for eval — lambda goes in HyperParameters instead
        else:
            if method not in _METHOD_TO_SERVERLESS_CONFIG:
                raise ValueError(
                    f"{method.value} is not supported on SMTJServerless. "
                    + (
                        "Use RFT_LORA instead."
                        if method == TrainingMethod.RFT_FULL
                        else f"Supported methods: {list(_METHOD_TO_SERVERLESS_CONFIG.keys())}"
                    )
                )
            technique, peft = _METHOD_TO_SERVERLESS_CONFIG[method]
            config["JobType"] = "FineTuning"
            config["CustomizationTechnique"] = technique
            if peft:
                config["Peft"] = peft
            # EvaluatorArn: reward function hub-content ARN for RLVR training,
            # or agent runtime ARN for multiturn RLVR.
            # Lambda ARNs are passed via HyperParameters (reward_lambda_arn) instead.
            if evaluator_arn and (
                _is_hub_content_arn(evaluator_arn)
                or method in (TrainingMethod.RFT_MULTITURN_LORA, TrainingMethod.RFT_MULTITURN_FULL)
            ):
                config["EvaluatorArn"] = evaluator_arn

        return config

    def execute(self, job_config: JobConfig) -> str:
        from amzn_nova_forge.validation.validator import Validator

        Validator.validate_job_name(job_name=job_config.job_name)

        try:
            assert job_config.output_s3_path is not None
            assert job_config.method is not None

            with open(job_config.recipe_path) as f:
                recipe = yaml.safe_load(f)

            model_type = recipe["run"]["model_type"]
            model = Model.from_model_type(model_type)

            # Resolve base model ARN from SageMaker Hub
            base_model_arn = self._resolve_base_model_arn(model)

            # Resolve the reward ARN only for RFT and eval jobs — not SFT/DPO.
            # For training (RLVR): auto-register Lambda ARN as hub-content for EvaluatorArn.
            # For eval: Lambda ARN goes directly into HyperParameters — no hub-content needed.
            _is_rft_or_eval = job_config.method in (
                TrainingMethod.RFT_LORA,
                TrainingMethod.RFT_FULL,
                TrainingMethod.EVALUATION,
            )
            evaluator_arn = (
                (
                    recipe.get("rl_env", {}).get("reward_lambda_arn")
                    or recipe.get("run", {}).get("reward_lambda_arn")
                    or self.rft_lambda_arn
                )
                if _is_rft_or_eval
                else None
            )
            from amzn_nova_forge.validation.validator import is_lambda_arn

            if (
                evaluator_arn
                and is_lambda_arn(evaluator_arn)
                and job_config.method != TrainingMethod.EVALUATION
            ):
                evaluator_arn = self._register_lambda_as_hub_content(evaluator_arn)

            hyperparams = self._extract_hyperparameters(
                recipe, method=job_config.method, data_mixing_config=job_config.data_mixing_config
            )
            if job_config.trainer_config_hyperparameters:
                hyperparams.update(job_config.trainer_config_hyperparameters)
            # For eval jobs with a reward lambda, the API requires lambda_arn + lambda_type
            # in HyperParameters. EvaluatorArn is NOT used for eval jobs.
            if job_config.method == TrainingMethod.EVALUATION and evaluator_arn:
                # Resolve the actual Lambda ARN — evaluator_arn may be a hub-content ARN
                lambda_arn_for_params = evaluator_arn
                if not is_lambda_arn(evaluator_arn):
                    extracted = extract_lambda_arn_from_hub_content(
                        evaluator_arn, self.sagemaker_client
                    )
                    if extracted:
                        lambda_arn_for_params = extracted
                    else:
                        logger.warning(
                            "Could not extract Lambda ARN from hub-content for eval. "
                            "Skipping lambda_arn in HyperParameters."
                        )
                        lambda_arn_for_params = None
                if lambda_arn_for_params:
                    hyperparams["lambda_arn"] = lambda_arn_for_params
                    hyperparams["lambda_type"] = "rft"

            # Build create_training_job request
            create_params: Dict[str, Any] = {
                "TrainingJobName": job_config.job_name,
                "RoleArn": self.execution_role,
                "OutputDataConfig": {
                    "S3OutputPath": job_config.output_s3_path,
                    "CompressionType": "NONE",
                },
                "StoppingCondition": {
                    "MaxRuntimeInSeconds": self.max_job_runtime,
                },
                "HyperParameters": hyperparams,
                "ServerlessJobConfig": self._build_serverless_job_config(
                    job_config.method,
                    base_model_arn,
                    eval_task=recipe.get("evaluation", {}).get("task"),
                    evaluator_arn=evaluator_arn,
                ),
            }

            # ModelPackageConfig:
            # - Training jobs: always include ModelPackageGroupArn.
            # - Eval jobs: only include ModelPackageConfig when model_name_or_path is a
            #   SageMaker ARN (fine-tuned model eval). Base model evals omit it entirely.
            model_name_or_path = recipe.get("run", {}).get("model_name_or_path", "")
            if job_config.method == TrainingMethod.EVALUATION:
                if is_sagemaker_arn(model_name_or_path):
                    model_package_config = {
                        "ModelPackageGroupArn": self.model_package_group_arn,
                        "SourceModelPackageArn": model_name_or_path,
                    }
                else:
                    model_package_config = {}
            else:
                model_package_config = {
                    "ModelPackageGroupArn": self.model_package_group_arn,
                    **(
                        {"SourceModelPackageArn": model_name_or_path}
                        if is_sagemaker_arn(model_name_or_path)
                        else {}
                    ),
                }
            if model_package_config:
                create_params["ModelPackageConfig"] = model_package_config

            if self.kms_key_id:
                create_params["OutputDataConfig"]["KmsKeyId"] = self.kms_key_id

            # Input data via DataSet
            # Tasks in BYOD_AVAILABLE_EVAL_TASKS require input data; built-in benchmarks don't
            is_no_input_eval = (
                job_config.method == TrainingMethod.EVALUATION
                and recipe.get("evaluation", {}).get("task") not in BYOD_AVAILABLE_EVAL_TASKS
            )
            if job_config.data_s3_path and not is_no_input_eval:
                from sagemaker.ai_registry.dataset import (
                    DataSet,  # Import DataSet at call site since it crashes without AWS_DEFAULT_REGION
                )

                dataset_name = f"{job_config.job_name[:51]}-train-input"
                input_dataset = DataSet.create(dataset_name, job_config.data_s3_path)
                create_params["InputDataConfig"] = [
                    {
                        "ChannelName": "train",
                        "DataSource": {
                            "DatasetSource": {"DatasetArn": input_dataset.arn},
                        },
                        "CompressionType": "None",
                        "RecordWrapperType": "None",
                    }
                ]
                if job_config.validation_data_s3_path:
                    validation_dataset_name = f"{job_config.job_name[:46]}-val-input"
                    validation_dataset = DataSet.create(
                        validation_dataset_name, job_config.validation_data_s3_path
                    )
                    create_params["InputDataConfig"].append(
                        {
                            "ChannelName": "validation",
                            "DataSource": {
                                "DatasetSource": {"DatasetArn": validation_dataset.arn},
                            },
                            "CompressionType": "None",
                            "RecordWrapperType": "None",
                        }
                    )
            if self.subnets or self.security_group_ids:
                create_params["VpcConfig"] = {
                    "SecurityGroupIds": self.security_group_ids or [],
                    "Subnets": self.subnets or [],
                }
            # MLflow config from recipe
            run_section = recipe.get("run", {})
            mlflow_uri = run_section.get("mlflow_tracking_uri")
            if mlflow_uri:
                create_params["MlflowConfig"] = {
                    "MlflowResourceArn": mlflow_uri,
                    "MlflowExperimentName": run_section.get("mlflow_experiment_name", ""),
                    "MlflowRunName": run_section.get("mlflow_run_name", ""),
                }

            result = self.sagemaker_client.create_training_job(**create_params)
            job_arn = result["TrainingJobArn"]
            return job_arn.split("/")[-1]

        except Exception as e:
            if isinstance(e, NoRegionError):
                logger.error(
                    "Could not connect to SageMaker. Please set a region with `export AWS_DEFAULT_REGION=<your-region>`"
                )
            logger.error(f"Failed to start training job: {str(e)}")
            raise

    def cleanup(self, job_name: str, is_mtrl: bool = False) -> None:
        try:
            if is_mtrl:
                self._cleanup_mtrl(job_name)
            else:
                self.sagemaker_client.stop_training_job(TrainingJobName=job_name)
                self.sagemaker_client.close()
        except Exception as e:
            logger.error(f"Failed to cleanup job {job_name}: {str(e)}")
            raise
