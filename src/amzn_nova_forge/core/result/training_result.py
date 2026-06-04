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
from abc import ABC
from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Dict, Optional

import boto3

from amzn_nova_forge.core.enums import Model, TrainingMethod
from amzn_nova_forge.core.result.job_result import (
    BaseJobResult,
    BedrockStatusManager,
    JobStatusManager,
    MTRLStatusManager,
    SMHPStatusManager,
    SMTJStatusManager,
)
from amzn_nova_forge.core.types import ModelArtifacts


@dataclass
class TrainingResult(BaseJobResult, ABC):
    method: TrainingMethod
    model_artifacts: ModelArtifacts
    model_type: Model
    # metrics: Dict[str, float] # TODO: Implement metrics

    def __init__(
        self,
        job_id: str,
        started_time: datetime,
        method: TrainingMethod,
        model_artifacts: ModelArtifacts,
        model_type: Model,
    ):
        self.method = method
        self.model_artifacts = model_artifacts
        self.model_type = model_type
        super().__init__(job_id, started_time)

    def get(self) -> Dict:
        # TODO: Implement getting detailed train result from s3 output path
        return self._to_dict()

    def show(self):
        # TODO: Implement showing train metrics from train result in s3 output path
        result = self.get()
        if result:
            print(result)


@dataclass
class SMTJTrainingResult(TrainingResult):
    _MTRL_METHODS = (TrainingMethod.RFT_MULTITURN_LORA,)

    def __init__(
        self,
        job_id: str,
        started_time: datetime,
        method: TrainingMethod,
        model_artifacts: ModelArtifacts,
        model_type: Model,
        sagemaker_client=None,
        region: Optional[str] = None,
    ):
        self._region = region
        self._sagemaker_client = sagemaker_client or boto3.client("sagemaker", region_name=region)
        self._is_mtrl = method in self._MTRL_METHODS
        self._rft_job = None
        super().__init__(job_id, started_time, method, model_artifacts, model_type)

    def _create_status_manager(self) -> JobStatusManager:
        if self._is_mtrl:
            return MTRLStatusManager(region=self._region)
        return SMTJStatusManager(self._sagemaker_client, region=self._region)

    def _get_rft_job(self):
        """Lazily fetch the AgentRFTJob for MTRL jobs."""
        if self._rft_job is None:
            from sagemaker.train.agent_rft_job import AgentRFTJob

            session = boto3.Session(region_name=self._region) if self._region else None
            self._rft_job = AgentRFTJob.get(self.job_id, session=session)
            # Populate output_model_arn if available and not already set
            if not self.model_artifacts.output_model_arn:
                arn = self._rft_job.output_model_package_arn
                if arn:
                    self.model_artifacts.output_model_arn = arn
        return self._rft_job

    def wait(self, poll: int = 30, timeout: int = 3600) -> None:
        """Wait for an MTRL job to complete.

        Delegates to the AgentRFTJob.wait() which displays a rich
        progress panel showing job status, metrics, and links.

        Args:
            poll: Seconds between status polls (default 30).
            timeout: Maximum seconds to wait (default 3600).
        """
        if not self._is_mtrl:
            raise NotImplementedError(
                "wait() is only supported for MTRL jobs. "
                "For standard SMTJ jobs, use get_job_status() to poll."
            )
        rft_job = self._get_rft_job()
        rft_job.wait(poll=poll, timeout=timeout)
        # Populate model_artifacts.output_model_arn after completion
        rft_job.refresh()
        if rft_job.output_model_package_arn:
            self.model_artifacts.output_model_arn = rft_job.output_model_package_arn

    def get_training_metrics(self) -> list:
        """Fetch per-step training metrics from MLflow for MTRL jobs.

        Delegates to AgentRFTJob.get_training_metrics() which
        retrieves reward/mean, turns/mean, total_tokens, and
        num_trajectories for each training step.

        Returns:
            List of dicts with per-step metrics.
        """
        if not self._is_mtrl:
            raise NotImplementedError("get_training_metrics() is only supported for MTRL jobs.")
        rft_job = self._get_rft_job()
        return rft_job.get_training_metrics()

    def _to_dict(self):
        return {
            "job_id": self.job_id,
            "started_time": self.started_time.isoformat(),
            "method": self.method.value,
            "model_artifacts": asdict(self.model_artifacts),
            "model_type": self.model_type.name,
        }

    @classmethod
    def _from_dict(cls, data) -> "SMTJTrainingResult":
        return cls(
            job_id=data["job_id"],
            started_time=datetime.fromisoformat(data["started_time"]),
            method=TrainingMethod(data["method"]),
            model_artifacts=ModelArtifacts(**data["model_artifacts"]),
            model_type=Model.from_model_name(data["model_type"]),
        )


@dataclass
class SMHPTrainingResult(TrainingResult):
    cluster_name: str
    namespace: str

    def __init__(
        self,
        job_id: str,
        started_time: datetime,
        method: TrainingMethod,
        model_artifacts: ModelArtifacts,
        cluster_name: str,
        model_type: Model,
        namespace: str = "kubeflow",
        region: Optional[str] = None,
    ):
        self._region = region
        self.cluster_name = cluster_name
        self.namespace = namespace
        super().__init__(job_id, started_time, method, model_artifacts, model_type)

    def _create_status_manager(self) -> JobStatusManager:
        return SMHPStatusManager(self.cluster_name, self.namespace)

    def _to_dict(self):
        return {
            "job_id": self.job_id,
            "started_time": self.started_time.isoformat(),
            "method": self.method.value,
            "model_artifacts": asdict(self.model_artifacts),
            "cluster_name": self.cluster_name,
            "namespace": self.namespace,
            "model_type": self.model_type.name,
        }

    @classmethod
    def _from_dict(cls, data) -> "SMHPTrainingResult":
        return cls(
            job_id=data["job_id"],
            started_time=datetime.fromisoformat(data["started_time"]),
            method=TrainingMethod(data["method"]),
            model_artifacts=ModelArtifacts(**data["model_artifacts"]),
            cluster_name=data["cluster_name"],
            model_type=Model.from_model_name(data["model_type"]),
            namespace=data["namespace"],
        )


@dataclass
class BedrockTrainingResult(TrainingResult):
    def __init__(
        self,
        job_id: str,
        started_time: datetime,
        method: TrainingMethod,
        model_artifacts: ModelArtifacts,
        model_type: Model,
        bedrock_client=None,
        region: Optional[str] = None,
    ):
        self._region = region
        self._bedrock_client = bedrock_client or boto3.client("bedrock", region_name=region)
        super().__init__(job_id, started_time, method, model_artifacts, model_type)

    def _create_status_manager(self) -> JobStatusManager:
        return BedrockStatusManager(self._bedrock_client, region=self._region)

    def _to_dict(self):
        return {
            "job_id": self.job_id,
            "started_time": self.started_time.isoformat(),
            "method": self.method.value,
            "model_artifacts": asdict(self.model_artifacts),
            "model_type": self.model_type.name,
        }

    @classmethod
    def _from_dict(cls, data) -> "BedrockTrainingResult":
        return cls(
            job_id=data["job_id"],
            started_time=datetime.fromisoformat(data["started_time"]),
            method=TrainingMethod(data["method"]),
            model_artifacts=ModelArtifacts(**data["model_artifacts"]),
            model_type=Model.from_model_name(data["model_type"]),
        )
