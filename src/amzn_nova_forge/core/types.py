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
"""
Shared data classes and type definitions for Nova Forge SDK.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING, Any, Callable, ClassVar, Dict, List, Optional, TypedDict

from pydantic import BaseModel

from amzn_nova_forge.core.constants import (
    DEFAULT_JOB_CACHE_DIR,
    REGION_TO_ESCROW_ACCOUNT_MAPPING,
)
from amzn_nova_forge.core.enums import DeployPlatform, Model, Platform, TrainingMethod

if TYPE_CHECKING:
    from amzn_nova_forge.core.job_cache import JobCachingConfig
    from amzn_nova_forge.monitor.mlflow_monitor import MLflowMonitor


class ValidationConfig(BaseModel):
    """Configuration controlling which pre-flight validation checks to run."""

    iam: bool = True
    infra: bool = True
    recipe: bool = True


@dataclass
class ForgeConfig:
    """Shared configuration for service classes.

    Holds optional settings that cut across training, evaluation, deployment,
    and inference workflows (e.g. KMS encryption, output paths, validation).

    ``mlflow_monitor`` is passed through to RecipeBuilder for experiment tracking.
    ``kms_key_id`` is used by ForgeDeployer for Bedrock model encryption (not
    sourced from RuntimeManager — ForgeDeployer does not use RuntimeManager).
    """

    kms_key_id: Optional[str] = None
    output_s3_path: Optional[str] = None
    generated_recipe_dir: Optional[str] = None
    validation_config: Optional[ValidationConfig] = None
    image_uri: Optional[str] = None
    mlflow_monitor: Optional[MLflowMonitor] = None
    enable_job_caching: bool = False
    job_cache_dir: str = DEFAULT_JOB_CACHE_DIR
    job_caching_config: Optional[JobCachingConfig] = None


class ModelConfigDict(TypedDict):
    type: str
    path: str


@dataclass
class ModelArtifacts:
    checkpoint_s3_path: Optional[str] = None
    output_s3_path: Optional[str] = None
    output_model_arn: Optional[str] = None  # Model package ARN for SMTJServerless jobs


@dataclass
class EndpointInfo:
    platform: DeployPlatform
    endpoint_name: str
    uri: str
    model_artifact_path: str
    region: Optional[str] = None


@dataclass
class DeploymentResult:
    endpoint: EndpointInfo
    created_at: datetime
    model_publish: Optional[Any] = None  # ModelDeployResult, Optional to avoid circular import

    @property
    def escrow_uri(self) -> Optional[str]:
        """Convenience: delegates to model_publish.escrow_uri."""
        return self.model_publish.escrow_uri if self.model_publish else None

    _status_checker: ClassVar[Optional[Callable]] = None
    _sagemaker_status_checker: ClassVar[Optional[Callable]] = None

    @classmethod
    def _register_status_checker(cls, checker: Callable) -> None:
        """Register the function used to check deployment status.

        Called by util/bedrock.py at import time to wire up the status
        property without core/ needing to import util/.
        """
        cls._status_checker = checker

    @classmethod
    def _register_sagemaker_status_checker(cls, checker: Callable) -> None:
        """Register the function used to check SageMaker deployment status.

        Called by util/sagemaker.py at import time to wire up the status
        property without core/ needing to import util/.
        """
        cls._sagemaker_status_checker = checker

    @property
    def status(self):
        if self.endpoint.platform == DeployPlatform.SAGEMAKER:
            if DeploymentResult._sagemaker_status_checker is None:
                try:
                    import amzn_nova_forge.util.sagemaker  # noqa: F401
                except ImportError:
                    pass
            if DeploymentResult._sagemaker_status_checker is None:
                raise RuntimeError(
                    "SageMaker status checker not available. "
                    "Ensure amzn_nova_forge.util.sagemaker is imported."
                )
            return DeploymentResult._sagemaker_status_checker(
                self.endpoint.uri, region=self.endpoint.region
            )

        if DeploymentResult._status_checker is None:
            # Runtime fallback only — core/types.py does NOT import util.bedrock
            # at module load time.  This triggers registration if the caller
            # forgot to import util.bedrock before accessing .status.
            try:
                import amzn_nova_forge.util.bedrock  # noqa: F401
            except ImportError:
                pass
        if DeploymentResult._status_checker is None:
            raise RuntimeError(
                "Status checker not available. Ensure amzn_nova_forge.util.bedrock is imported."
            )
        return DeploymentResult._status_checker(
            self.endpoint.uri, self.endpoint.platform, self.endpoint.region
        )


def validate_region(region: str) -> None:
    """Validate that the given AWS region is supported by Forge."""
    if region not in REGION_TO_ESCROW_ACCOUNT_MAPPING:
        raise ValueError(
            f"Region '{region}' is not supported. "
            f"Supported regions are: {list(REGION_TO_ESCROW_ACCOUNT_MAPPING.keys())}"
        )


@dataclass
class InferenceComponentConfig:
    """Configuration for creating an inference component on a SageMaker endpoint.

    When passed to create_sagemaker_endpoint, the endpoint is created without a
    ModelName in ProductionVariants (IC-compatible mode) and an inference component
    is created after the endpoint reaches InService. The IC references the SageMaker
    model (created during deploy) via ModelName.

    Args:
        inference_component_name: Unique name for the inference component.
        num_cpus: Number of vCPUs to allocate.
        num_accelerators: Number of accelerators (GPUs) to allocate.
        min_memory_in_mb: Minimum memory in MB to allocate.
        copy_count: Number of model copies to deploy. Default: 1.
        variant_name: Production variant name. Default: "primary".
    """

    inference_component_name: str
    num_cpus: int
    num_accelerators: int
    min_memory_in_mb: int
    copy_count: int = 1
    variant_name: str = "primary"


# Minimum compute resource requirements for inference components per model.
# Format: {Model: (min_cpus, min_memory_mb, min_gpus)}
_IC_MIN_COMPUTE_REQUIREMENTS: Dict[Model, tuple] = {
    Model.NOVA_MICRO: (15, 25000, 4),
    Model.NOVA_LITE: (20, 35000, 4),
    Model.NOVA_LITE_2: (20, 100000, 4),
}


def validate_inference_component_resources(config: InferenceComponentConfig, model: Model) -> None:
    """Validate that inference component compute resources meet minimum requirements.

    Args:
        config: The inference component configuration to validate.
        model: The Nova model being deployed.

    Raises:
        ValueError: If any resource is below the minimum for the given model.
    """
    requirements = _IC_MIN_COMPUTE_REQUIREMENTS.get(model)
    if requirements is None:
        return  # No known requirements for this model, skip validation

    min_cpus, min_memory_mb, min_gpus = requirements
    errors = []

    if config.num_cpus < min_cpus:
        errors.append(f"num_cpus={config.num_cpus} is below minimum {min_cpus} for {model.value}")
    if config.min_memory_in_mb < min_memory_mb:
        errors.append(
            f"min_memory_in_mb={config.min_memory_in_mb} is below minimum {min_memory_mb} for {model.value}"
        )
    if config.num_accelerators < min_gpus:
        errors.append(
            f"num_accelerators={config.num_accelerators} is below minimum {min_gpus} for {model.value}"
        )

    if errors:
        raise ValueError(
            f"Inference component resources do not meet minimum requirements for {model.value}: "
            + "; ".join(errors)
        )


@dataclass
class JobConfig:
    job_name: str
    image_uri: str
    recipe_path: str
    output_s3_path: Optional[str] = None
    data_s3_path: Optional[str] = None
    input_s3_data_type: Optional[str] = None
    validation_data_s3_path: Optional[str] = (
        None  # Validation data S3 path (for CPT, SFT, and Bedrock)
    )
    trainer_config_hyperparameters: Optional[Dict[str, str]] = (
        None  # Extra hyperparameters passed to the training job (e.g., val_check_interval)
    )
    rft_lambda_arn: Optional[str] = None  # RFT Lambda ARN (for RFT jobs on Bedrock)
    mlflow_tracking_uri: Optional[str] = None  # MLflow tracking server ARN
    mlflow_experiment_name: Optional[str] = None
    mlflow_run_name: Optional[str] = None
    method: Optional[TrainingMethod] = None  # Training method (required for Bedrock)
    data_mixing_config: Optional[Dict[str, Any]] = None  # Datamix percent fields (SMTJServerless)
    environment: Optional[Dict[str, str]] = None  # Environment variables for the training container
    model_name_or_path: Optional[str] = None  # Model path or model package ARN
    # TODO: The mlflow config is populated in recipe for both SMTJ and SMHP but will only work for SMHP as SMTJ support for mlflow is only through boto3, fix this with sagemaker 3 update


@dataclass(frozen=True)
class ConfigParameter:
    """A single overridable parameter in a training recipe."""

    name: str
    type: str
    default: Any
    description: Optional[str] = None
    min: Optional[float] = None
    max: Optional[float] = None
    enum: Optional[tuple] = None
    required: bool = False


@dataclass(frozen=True)
class RecipeConfig:
    """Frozen snapshot of the overridable configuration for a training recipe."""

    model: Model
    method: TrainingMethod
    platform: Platform
    parameters: tuple[ConfigParameter, ...]

    def to_dict(self) -> Dict[str, Any]:
        return {p.name: p.default for p in self.parameters}

    def __repr__(self) -> str:
        lines = [
            f"RecipeConfig(model={self.model.name}, method={self.method.name}, platform={self.platform.name})"
        ]
        for p in self.parameters:
            constraint = ""
            if p.min is not None and p.max is not None:
                constraint = f" [{p.min}..{p.max}]"
            elif p.min is not None:
                constraint = f" [>={p.min}]"
            elif p.max is not None:
                constraint = f" [<={p.max}]"
            if p.enum is not None:
                constraint = f" enum={list(p.enum)}"
            lines.append(f"  {p.name}: {p.type} = {p.default}{constraint}")
        return "\n".join(lines)
