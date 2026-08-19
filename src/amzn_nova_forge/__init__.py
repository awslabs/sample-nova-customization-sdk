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
import warnings

warnings.warn(
    "amzn-nova-forge is deprecated. Please migrate to the SageMaker Python SDK V3 "
    "(pip install 'sagemaker>=3.19.0').",
    DeprecationWarning,
    stacklevel=2,
)

from .core.data_mixing_config import DataMixingConfig
from .core.enums import (
    DeploymentMode,
    DeployPlatform,
    EvaluationTask,
    Model,
    Platform,
    TrainingMethod,
)
from .core.result import (
    EvaluationResult,
    SMTJBatchInferenceResult,
    TrainingResult,
)
from .core.result.inference_result import InferenceResult
from .core.result.job_result import (
    BaseJobResult,
    JobStatus,
)
from .core.training_overrides import TrainingOverrides
from .core.types import (
    ConfigParameter,
    DeploymentResult,
    EndpointInfo,
    ForgeConfig,
    ModelArtifacts,
    RecipeConfig,
)
from .dataset import (
    ArrowDatasetLoader,
    CSVDatasetLoader,
    JSONDatasetLoader,
    JSONLDatasetLoader,
    ParquetDatasetLoader,
)
from .dataset.operations.filter_operation import FilterMethod
from .dataset.operations.transform_operation import TransformMethod
from .dataset.operations.validate_operation import ValidateMethod
from .deployer import ForgeDeployer
from .evaluator import EvalTaskConfig, ForgeEvaluator, InspectLensConfig
from .inference import ForgeInference
from .manager import (
    BedrockRuntimeManager,
    SMHPRuntimeManager,
    SMTJRuntimeManager,
    SMTJServerlessRuntimeManager,
)
from .model import (
    NovaModelCustomizer,
)
from .monitor import (
    CloudWatchLogMonitor,
    MLflowMonitor,
)
from .notifications import (
    NotificationManager,
    NotificationManagerInfraError,
    SMHPNotificationManager,
    SMTJNotificationManager,
)
from .rft_multiturn import (
    CustomEnvironment,
    EnvType,
    RFTMultiturnInfrastructure,
    VFEnvId,
    create_rft_execution_role,
    list_rft_stacks,
)
from .trainer import ForgeTrainer
from .util.reward_verifier import (
    verify_reward_function,
)
from .validation.endpoint_validator import SageMakerEndpointEnvironment

_LAZY_IMPORTS = {
    "CloudWatchDatasetLoader": ".dataset.cloudwatch_dataset_loader",
    "DefaultTextFilterOperation": ".dataset.operations.default_text_filter_operation",
    "ExactDedupFilterOperation": ".dataset.operations.exact_dedup_filter_operation",
    "FuzzyDedupFilterOperation": ".dataset.operations.fuzzy_dedup_filter_operation",
    "InvalidRecordsFilterOperation": ".dataset.operations.invalid_records_filter_operation",
    "HuggingFaceDatasetLoader": ".dataset.huggingface_dataset_loader",
}


def __getattr__(name: str):
    if name in _LAZY_IMPORTS:
        import importlib

        module = importlib.import_module(_LAZY_IMPORTS[name], __name__)
        value = getattr(module, name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "ArrowDatasetLoader",
    "CloudWatchDatasetLoader",
    "CSVDatasetLoader",
    "HuggingFaceDatasetLoader",
    "JSONDatasetLoader",
    "JSONLDatasetLoader",
    "ParquetDatasetLoader",
    "FilterMethod",
    "TransformMethod",
    "ValidateMethod",
    "DefaultTextFilterOperation",
    "ExactDedupFilterOperation",
    "BedrockRuntimeManager",
    "SMHPRuntimeManager",
    "SMTJRuntimeManager",
    "SMTJServerlessRuntimeManager",
    
    "Model",
    "TrainingMethod",
    "DeploymentMode",
    "DeployPlatform",
    "Platform",
    "NovaModelCustomizer",
    "ConfigParameter",
    "DataMixingConfig",
    "RecipeConfig",
    "TrainingOverrides",
    "ForgeConfig",
    "ForgeTrainer",
    "ForgeEvaluator",
    "EvalTaskConfig",
    "ForgeDeployer",
    "ForgeInference",
    "BaseJobResult",
    "JobStatus",
    "TrainingResult",
    "EvaluationResult",
    "DeploymentResult",
    "EndpointInfo",
    "InferenceResult",
    "SMTJBatchInferenceResult",
    "ModelArtifacts",
    "MLflowMonitor",
    "CloudWatchLogMonitor",
    "NotificationManager",
    "NotificationManagerInfraError",
    "SMHPNotificationManager",
    "SMTJNotificationManager",
    "EvaluationTask",
    "RFTMultiturnInfrastructure",
    "EnvType",
    "VFEnvId",
    "CustomEnvironment",
    "list_rft_stacks",
    "create_rft_execution_role",
    "verify_reward_function",
    "SageMakerEndpointEnvironment",
]
