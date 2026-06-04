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
import importlib
import json
import logging
import subprocess
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, ClassVar, Dict, Optional

import boto3

from amzn_nova_forge.core.constants import MTRL_PIPELINE_EXECUTION_RE
from amzn_nova_forge.core.enums import Platform
from amzn_nova_forge.core.validation_patterns import (
    validate_cluster_name,
    validate_job_name,
    validate_namespace,
)
from amzn_nova_forge.util.subprocess_utils import _check_hyperpod_stderr

logger = logging.getLogger("nova_forge_sdk")


class JobStatus(enum.Enum):
    IN_PROGRESS = "InProgress"
    COMPLETED = "Completed"
    FAILED = "Failed"

    @classmethod
    def _missing_(cls, value: object):
        # Handle aliases
        aliases = {
            "Created": cls.IN_PROGRESS,
            "Running": cls.IN_PROGRESS,
            "Succeeded": cls.COMPLETED,
        }
        if isinstance(value, str) and value in aliases:
            return aliases[value]
        # Treat all other case as FAILED
        return cls.FAILED


class JobStatusManager(ABC):
    def __init__(self):
        self._job_status = JobStatus.IN_PROGRESS
        self._raw_status: str = JobStatus.IN_PROGRESS.value

    @abstractmethod
    def get_job_status(self, job_id: str) -> tuple[JobStatus, str]:
        """
        Get the status of the job

        Returns:
            str: JobStatus, raw status from the job platform
        """
        pass

    def resolve_start_time(self, job_id: str) -> datetime:
        """
        Resolve the start time of a job from the platform API.

        Returns:
            datetime: The job's start time

        Raises:
            ValueError: If start time cannot be resolved
        """
        raise ValueError(f"Cannot resolve start time for job {job_id} on {self.__class__.__name__}")


class SMTJStatusManager(JobStatusManager):
    def __init__(self, sagemaker_client=None, region: Optional[str] = None):
        super().__init__()
        self._sagemaker_client = sagemaker_client or boto3.client("sagemaker", region_name=region)

    def get_job_status(self, job_id: str) -> tuple[JobStatus, str]:
        if self._job_status == JobStatus.COMPLETED or self._job_status == JobStatus.FAILED:
            return self._job_status, self._raw_status

        # Call sagemaker api to get job status
        response = self._sagemaker_client.describe_training_job(TrainingJobName=job_id)
        raw_status = response["TrainingJobStatus"]
        job_status = JobStatus(raw_status)

        # Cache job status
        self._job_status = job_status
        self._raw_status = raw_status

        return job_status, raw_status

    def resolve_start_time(self, job_id: str) -> datetime:
        response = self._sagemaker_client.describe_training_job(TrainingJobName=job_id)
        start_time = response.get("TrainingStartTime") or response.get("CreationTime")
        if start_time:
            return (
                start_time
                if isinstance(start_time, datetime)
                else datetime.fromisoformat(str(start_time))
            )
        raise ValueError(f"Cannot resolve start time for SMTJ job {job_id}")


class SMHPStatusManager(JobStatusManager):
    def __init__(self, cluster_name: str, namespace: str):
        super().__init__()
        validate_cluster_name(cluster_name)
        validate_namespace(namespace)

        self.cluster_name = cluster_name
        self.namespace = namespace

    def _connect_cluster(self):
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

    def get_job_status(self, job_id: str) -> tuple[JobStatus, str]:
        if self._job_status == JobStatus.COMPLETED or self._job_status == JobStatus.FAILED:
            return self._job_status, self._raw_status

        validate_job_name(job_id)

        try:
            # Connect cluster before making call
            self._connect_cluster()
            # Call hyperpod CLI to get job status
            result = subprocess.run(
                ["hyperpod", "get-job", "--job-name", job_id],
                capture_output=True,
                text=True,
                check=True,
            )

            response = json.loads(result.stdout)
            status = response.get("Status")

            if status is None:
                # Status is null, job is still pending
                raw_status = "Pending"
                job_status = JobStatus.IN_PROGRESS
            else:
                conditions = status.get("conditions", [])
                if conditions:
                    # Get the last condition (most recent)
                    latest_condition = conditions[-1]
                    raw_status = latest_condition.get("type", "Unknown")
                    job_status = JobStatus(raw_status)
                else:
                    # No conditions but status exists, still in progress
                    raw_status = "Pending"
                    job_status = JobStatus.IN_PROGRESS

            # Cache job status
            self._job_status = job_status
            self._raw_status = raw_status

            return job_status, raw_status

        except (subprocess.CalledProcessError, json.JSONDecodeError, KeyError) as e:
            stderr_info = getattr(e, "stderr", None)
            if isinstance(e, subprocess.CalledProcessError):
                hint = (
                    "This may be due to insufficient permissions "
                    "(e.g., missing EKS access entry) or cluster connectivity issues."
                )
            else:
                hint = "The command output was not in the expected format."

            error_msg = f"Failed to get job status for {job_id}: {e}\n{hint}"

            if stderr_info:
                error_msg += f"\nDetails: {stderr_info}"

            logger.error(error_msg)
            raise RuntimeError(error_msg) from e

    def resolve_start_time(self, job_id: str) -> datetime:
        validate_job_name(job_id)

        try:
            self._connect_cluster()
            result = subprocess.run(
                ["hyperpod", "get-job", "--job-name", job_id, "--verbose"],
                capture_output=True,
                text=True,
                check=True,
            )
            response = json.loads(result.stdout)

            # Try Status.startTime, then Metadata.CreationTimestamp
            start_time_str = response.get("Status", {}).get("startTime") or response.get(
                "Metadata", {}
            ).get("CreationTimestamp")
            if start_time_str:
                return datetime.fromisoformat(start_time_str.replace("Z", "+00:00"))
        except Exception as e:
            logger.error(f"Failed to resolve start time for SMHP job {job_id}: {e}")

        raise ValueError(f"Cannot resolve start time for SMHP job {job_id}")


class MTRLStatusManager(JobStatusManager):
    """Status manager for MTRL jobs (training via AgentRFT Job API, eval via Pipeline Execution)."""

    def __init__(self, region: Optional[str] = None):
        super().__init__()
        self._region = region
        self._session = boto3.Session(region_name=region) if region else None

    def _is_pipeline_execution(self, job_id: str) -> bool:
        return bool(MTRL_PIPELINE_EXECUTION_RE.match(job_id))

    def _get_pipeline_client(self, job_id: str):
        region = job_id.split(":")[3] if ":sagemaker:" in job_id else self._region
        return boto3.client("sagemaker", region_name=region)

    def get_job_status(self, job_id: str) -> tuple[JobStatus, str]:
        if self._job_status == JobStatus.COMPLETED or self._job_status == JobStatus.FAILED:
            return self._job_status, self._raw_status

        if self._is_pipeline_execution(job_id):
            return self._get_pipeline_status(job_id)
        return self._get_job_status(job_id)

    def _get_job_status(self, job_id: str) -> tuple[JobStatus, str]:
        from sagemaker.train.agent_rft_job import AgentRFTJob

        rft_job = AgentRFTJob.get(job_id, session=self._session)
        raw_status = rft_job.job_status

        status_mapping = {
            "InProgress": JobStatus.IN_PROGRESS,
            "Completed": JobStatus.COMPLETED,
            "Failed": JobStatus.FAILED,
            "Stopping": JobStatus.FAILED,
            "Stopped": JobStatus.FAILED,
        }
        job_status = status_mapping.get(raw_status, JobStatus.IN_PROGRESS)

        self._job_status = job_status
        self._raw_status = raw_status
        return job_status, raw_status

    def _get_pipeline_status(self, job_id: str) -> tuple[JobStatus, str]:
        client = self._get_pipeline_client(job_id)
        resp = client.describe_pipeline_execution(PipelineExecutionArn=job_id)
        raw_status = resp["PipelineExecutionStatus"]

        status_mapping = {
            "Executing": JobStatus.IN_PROGRESS,
            "Starting": JobStatus.IN_PROGRESS,
            "Succeeded": JobStatus.COMPLETED,
            "Failed": JobStatus.FAILED,
            "Stopping": JobStatus.FAILED,
            "Stopped": JobStatus.FAILED,
        }
        job_status = status_mapping.get(raw_status, JobStatus.IN_PROGRESS)

        self._job_status = job_status
        self._raw_status = raw_status
        return job_status, raw_status

    def resolve_start_time(self, job_id: str) -> datetime:
        if self._is_pipeline_execution(job_id):
            client = self._get_pipeline_client(job_id)
            resp = client.describe_pipeline_execution(PipelineExecutionArn=job_id)
            start_time = resp.get("CreationTime")
            if start_time:
                return (
                    start_time
                    if isinstance(start_time, datetime)
                    else datetime.fromisoformat(str(start_time))
                )
            raise ValueError(f"Cannot resolve start time for MTRL eval pipeline {job_id}")

        from sagemaker.train.agent_rft_job import AgentRFTJob

        rft_job = AgentRFTJob.get(job_id, session=self._session)
        if rft_job.creation_time:
            return rft_job.creation_time
        raise ValueError(f"Cannot resolve start time for MTRL job {job_id}")


class BedrockStatusManager(JobStatusManager):
    # Injected by util/bedrock.py at import time so core/ has zero internal imports.
    _get_job_details: Optional[Callable] = None
    _log_job_status: Optional[Callable] = None

    @classmethod
    def _register_bedrock_helpers(
        cls,
        get_job_details: Callable,
        log_job_status: Callable,
    ) -> None:
        """Register bedrock helper functions.

        Called by util/bedrock.py at import time to wire up status checking
        without core/ needing to import util/.
        """
        cls._get_job_details = staticmethod(get_job_details)
        cls._log_job_status = staticmethod(log_job_status)

    def __init__(self, bedrock_client=None, region: Optional[str] = None):
        super().__init__()
        self._bedrock_client = bedrock_client or boto3.client("bedrock", region_name=region)

    def get_job_status(self, job_id: str) -> tuple[JobStatus, str]:
        if self._job_status == JobStatus.COMPLETED or self._job_status == JobStatus.FAILED:
            return self._job_status, self._raw_status

        try:
            if (
                BedrockStatusManager._get_job_details is None
                or BedrockStatusManager._log_job_status is None
            ):
                try:
                    import amzn_nova_forge.util.bedrock  # noqa: F401 — triggers registration
                except ImportError:
                    pass
            if (
                BedrockStatusManager._get_job_details is None
                or BedrockStatusManager._log_job_status is None
            ):
                raise RuntimeError(
                    "Bedrock helpers not registered. "
                    "Ensure amzn_nova_forge.util.bedrock is imported."
                )

            # Get detailed job information
            response = BedrockStatusManager._get_job_details(self._bedrock_client, job_id)

            # Log detailed status information
            BedrockStatusManager._log_job_status(response)

            raw_status = response["status"]

            # Map Bedrock status to JobStatus
            # Bedrock statuses: InProgress, Completed, Failed, Stopping, Stopped
            # Note: Stopping/Stopped are mapped to FAILED because they represent
            # user-initiated cancellations or system-stopped jobs, which should be
            # treated as unsuccessful completions for the SDK's job lifecycle.
            status_mapping = {
                "InProgress": JobStatus.IN_PROGRESS,
                "Completed": JobStatus.COMPLETED,
                "Failed": JobStatus.FAILED,
                "Stopping": JobStatus.FAILED,
                "Stopped": JobStatus.FAILED,
            }

            job_status = status_mapping.get(raw_status, JobStatus.FAILED)

            # Cache job status
            self._job_status = job_status
            self._raw_status = raw_status

            return job_status, raw_status

        except Exception as e:
            logger.error(f"Failed to get Bedrock job status for {job_id}: {e}")
            return JobStatus.IN_PROGRESS, "Unknown"


@dataclass
class BaseJobResult(ABC):
    job_id: str
    started_time: datetime

    # Injected by notifications/__init__.py at import time so core/ has zero
    # internal imports.  Signature: (platform, region, **kwargs) -> manager
    _notification_manager_factory: ClassVar[Optional[Callable]] = None

    @classmethod
    def _register_notification_factory(cls, factory: Callable) -> None:
        """Register the notification manager factory.

        Called by notifications/__init__.py at import time to wire up
        enable_job_notifications without core/ needing to import notifications/.
        """
        cls._notification_manager_factory = staticmethod(factory)

    def __init__(self, job_id: str, started_time: Optional[datetime] = None):
        self.job_id = job_id
        self._status_manager: JobStatusManager = self._create_status_manager()
        self._platform = (
            Platform.SMTJ
            if isinstance(self._status_manager, SMTJStatusManager)
            else (
                Platform.SMTJServerless
                if isinstance(self._status_manager, MTRLStatusManager)
                else (
                    Platform.BEDROCK
                    if isinstance(self._status_manager, BedrockStatusManager)
                    else Platform.SMHP
                )
            )
        )

        if started_time is not None:
            self.started_time = started_time
        else:
            logger.info(f"No started_time provided for job {job_id}, resolving from platform...")
            self.started_time = self._status_manager.resolve_start_time(job_id)

    @property
    def status_manager(self):
        return self._status_manager

    @property
    def platform(self):
        return self._platform

    @abstractmethod
    def _create_status_manager(self) -> JobStatusManager:
        """
        Create status manager for this job
        :return:
        """
        pass

    def get_job_status(self) -> tuple[JobStatus, str]:
        """
        Get the status of the job

        Returns:
            str: Job status
        """
        return self._status_manager.get_job_status(self.job_id)

    @abstractmethod
    def get(self) -> Dict:
        """
        Get the job result as dict
        :return: job result dict
        """
        pass

    @abstractmethod
    def show(self):
        """
        Print the job result
        """
        pass

    def enable_job_notifications(
        self,
        emails: list[str],
        output_s3_path: Optional[str] = None,
        region: Optional[str] = "us-east-1",
        **platform_kwargs,  # Platform-specific optional parameters
    ) -> None:
        """
        Enable email notifications for this job when the job reaches a terminal state:
            - Completed, Stopped, Failed

        Args:
            emails: List of email addresses to notify
            output_s3_path: S3 path where job outputs are stored. If not provided,
                it will attempt to extract from model_artifacts. Required for manifest
                validation when the job completes.
            region: AWS region (defaults to us-east-1)
            **platform_kwargs: Platform-specific parameters:
                - For SMTJ: kms_key_id (optional KMS key for SNS encryption)
                - For SMHP: eks_cluster_arn, vpc_id, subnet_ids, security_group_id (required),
                           namespace, kubectl_layer_arn, kms_key_id (optional)

        Raises:
            ValueError: If inputs are invalid or output_s3_path cannot be determined
            NotificationManagerInfraError: If infrastructure setup fails
        """
        if BaseJobResult._notification_manager_factory is None:
            try:
                import amzn_nova_forge.notifications  # noqa: F401 — triggers registration
            except ImportError:
                pass
        if BaseJobResult._notification_manager_factory is None:
            raise RuntimeError(
                "Notification factory not registered. "
                "Ensure amzn_nova_forge.notifications is imported."
            )

        # Use default region if not provided
        resolved_region = region if region is not None else "us-east-1"

        # Determine output_s3_path
        resolved_output_s3_path = output_s3_path

        if resolved_output_s3_path is None:
            # Try to extract from model_artifacts (for TrainingResult)
            if hasattr(self, "model_artifacts") and self.model_artifacts:
                resolved_output_s3_path = self.model_artifacts.output_s3_path
            # Try to extract from eval_output_path (for EvaluationResult)
            elif hasattr(self, "eval_output_path") and self.eval_output_path:
                resolved_output_s3_path = self.eval_output_path
            else:
                raise ValueError(
                    "Cannot enable notifications: output_s3_path is required but can't be found.\n"
                    "Please provide output_s3_path explicitly:\n"
                    f"  - result.enable_job_notifications(emails=[], output_s3_path='s3://path')"
                )

            if not resolved_output_s3_path:
                raise ValueError(
                    "Cannot enable notifications: output_s3_path is required but not set.\n"
                    "Please provide output_s3_path explicitly:\n "
                    f"  - result.enable_job_notifications(emails=[], output_s3_path='s3://path')\n"
                )

        # Build kwargs for factory
        factory_kwargs: Dict[str, Any] = {}
        if self._platform == Platform.SMHP:
            if not hasattr(self._status_manager, "cluster_name"):
                raise ValueError(
                    "Cannot enable SMHP notifications: cluster_name not found in status manager"
                )
            factory_kwargs["cluster_name"] = self._status_manager.cluster_name

        manager = BaseJobResult._notification_manager_factory(
            platform=self._platform,
            region=resolved_region,
            **factory_kwargs,
        )

        # Enable notifications with platform-specific parameters
        manager.enable_notifications(
            job_name=self.job_id,
            emails=emails,
            output_s3_path=resolved_output_s3_path,
            **platform_kwargs,
        )

    def _to_dict(self):
        """
        Convert the job result to dict
        :return: object as dict
        """
        return asdict(self)

    @classmethod
    def _from_dict(cls, data) -> "BaseJobResult":
        """
        Load the job result from json
        :return: object as dict
        """
        return cls(**data)

    def dump(self, file_path: Optional[str] = None, file_name: Optional[str] = None) -> Path:
        """
        Save the job result to file_path path
        :param file_path: Directory path to save the result. Saves to current directory if not provided
        :param file_name: The file name of the result. Default to <job_id>_<platform>.json if not provided
        :return: The full result file path
        """
        if not file_name:
            # MTRL Pipeline execution ARNs contain "/" which is invalid in filenames
            if "/" in self.job_id:
                exec_id = self.job_id.rsplit("/", 1)[-1]
                job_name = getattr(self, "_job_name", None)
                name = f"{job_name}-{exec_id}" if job_name else exec_id
            else:
                name = self.job_id
            file_name = f"{name}_{self._platform.value}.json"

        if file_path is None:
            full_path = Path(file_name)
        else:
            full_path = Path(file_path) / file_name

        data = self._to_dict()
        data["__class_name__"] = self.__class__.__name__

        # Include job cache hash if it exists
        if hasattr(self, "_job_cache_hash"):
            data["_job_cache_hash"] = self._job_cache_hash

        with open(full_path, "w") as f:
            json.dump(data, f, default=str)
        logger.info(f"Job result saved to {full_path}")
        print(f"Job result saved to {full_path}")
        return full_path

    @classmethod
    def load(cls, file_path: str) -> "BaseJobResult":
        """
        Load the job result from file_path path
        :param file_path: file_path to load the result
        :return: The Job result object
        """
        with open(file_path, "r") as f:
            data = json.load(f)

        class_name = data.pop("__class_name__", None)
        # Extract job cache hash before creating the object
        job_cache_hash = data.pop("_job_cache_hash", None)

        if class_name:
            try:
                module = importlib.import_module("amzn_nova_forge.core.result")
                target_class = getattr(module, class_name, None)
                if target_class and issubclass(target_class, BaseJobResult):
                    result = target_class._from_dict(data)
                    # Restore the job cache hash if it existed
                    if job_cache_hash:
                        result._job_cache_hash = job_cache_hash
                    return result
                else:
                    raise ValueError(
                        f"Class {class_name} not found or not a subclass of BaseJobResult"
                    )
            except (ImportError, AttributeError, TypeError) as e:
                logger.error(f"Failed to load job result from {file_path}, due to: {e}")
                raise ValueError(f"Unable to load job result from {file_path}, due to {e}") from e

        raise ValueError(f"Unable to load job result from {file_path}, no class name found")
