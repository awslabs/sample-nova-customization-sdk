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
Shared core module for Nova Forge SDK.

Contains all enums, data classes, result classes, and constants shared across modules.
Rule: core/ imports nothing internal; all other modules import from core/.
"""

from amzn_nova_forge.core.constants import (
    BYOD_AVAILABLE_EVAL_TASKS,
    DEFAULT_JOB_CACHE_DIR,
    DEFAULT_REGION,
    EVAL_AVAILABLE_SUBTASKS,
    EVAL_TASK_METRIC_MAP,
    EVAL_TASK_STRATEGY_MAP,
    HYPERPOD_RECIPE_PATH,
    REGION_TO_ESCROW_ACCOUNT_MAPPING,
    SUPPORTED_DATAMIXING_METHODS,
    SUPPORTED_SMI_CONFIGS,
    get_available_subtasks,
)
from amzn_nova_forge.core.enums import (
    DeploymentMode,
    DeployPlatform,
    EvaluationMetric,
    EvaluationStrategy,
    EvaluationTask,
    FilterMethod,
    Model,
    Platform,
    TrainingMethod,
    Version,
)
from amzn_nova_forge.core.job_cache import (
    JobCacheContext,
    JobCachingConfig,
    build_cache_context,
    load_existing_result,
    persist_result,
)
from amzn_nova_forge.core.runtime import RuntimeManager
from amzn_nova_forge.core.types import (
    ConfigParameter,
    DeploymentResult,
    EndpointInfo,
    ForgeConfig,
    JobConfig,
    ModelArtifacts,
    ModelConfigDict,
    RecipeConfig,
    ValidationConfig,
    validate_region,
)

__all__ = [
    # constants
    "BYOD_AVAILABLE_EVAL_TASKS",
    "DEFAULT_JOB_CACHE_DIR",
    "DEFAULT_REGION",
    "EVAL_AVAILABLE_SUBTASKS",
    "EVAL_TASK_METRIC_MAP",
    "EVAL_TASK_STRATEGY_MAP",
    "HYPERPOD_RECIPE_PATH",
    "REGION_TO_ESCROW_ACCOUNT_MAPPING",
    "SUPPORTED_DATAMIXING_METHODS",
    "SUPPORTED_SMI_CONFIGS",
    "get_available_subtasks",
    # enums
    "DeploymentMode",
    "DeployPlatform",
    "EvaluationMetric",
    "EvaluationStrategy",
    "EvaluationTask",
    "FilterMethod",
    "Model",
    "Platform",
    "TrainingMethod",
    "Version",
    # job_cache
    "JobCacheContext",
    "JobCachingConfig",
    "build_cache_context",
    "load_existing_result",
    "persist_result",
    # runtime
    "RuntimeManager",
    # types
    "ConfigParameter",
    "DeploymentResult",
    "EndpointInfo",
    "ForgeConfig",
    "JobConfig",
    "ModelArtifacts",
    "ModelConfigDict",
    "RecipeConfig",
    # result classes (lazy-loaded)
    "BaseJobResult",
    "BedrockStatusManager",
    "JobStatus",
    "JobStatusManager",
    "SMHPStatusManager",
    "SMTJStatusManager",
    "BedrockTrainingResult",
    "SMHPTrainingResult",
    "SMTJTrainingResult",
    "TrainingResult",
    "BedrockEvaluationResult",
    "EvaluationResult",
    "SMHPEvaluationResult",
    "SMTJEvaluationResult",
    "InferenceResult",
    "SingleInferenceResult",
    "SMTJBatchInferenceResult",
]

# Lazy imports for result classes to avoid pulling in boto3 and heavy
# infrastructure at module load time.  Importing `from amzn_nova_forge.core
# import Model` does NOT trigger result class loading.  Importing directly
# from `amzn_nova_forge.core.result` DOES eagerly load all result classes
# (acceptable — callers who reach into the subpackage expect the full set).
_RESULT_IMPORTS = {
    "BaseJobResult": "amzn_nova_forge.core.result.job_result",
    "BedrockStatusManager": "amzn_nova_forge.core.result.job_result",
    "JobStatus": "amzn_nova_forge.core.result.job_result",
    "JobStatusManager": "amzn_nova_forge.core.result.job_result",
    "SMHPStatusManager": "amzn_nova_forge.core.result.job_result",
    "SMTJStatusManager": "amzn_nova_forge.core.result.job_result",
    "BedrockTrainingResult": "amzn_nova_forge.core.result.training_result",
    "SMHPTrainingResult": "amzn_nova_forge.core.result.training_result",
    "SMTJTrainingResult": "amzn_nova_forge.core.result.training_result",
    "TrainingResult": "amzn_nova_forge.core.result.training_result",
    "BedrockEvaluationResult": "amzn_nova_forge.core.result.eval_result",
    "EvaluationResult": "amzn_nova_forge.core.result.eval_result",
    "SMHPEvaluationResult": "amzn_nova_forge.core.result.eval_result",
    "SMTJEvaluationResult": "amzn_nova_forge.core.result.eval_result",
    "InferenceResult": "amzn_nova_forge.core.result.inference_result",
    "SingleInferenceResult": "amzn_nova_forge.core.result.inference_result",
    "SMTJBatchInferenceResult": "amzn_nova_forge.core.result.inference_result",
}


def __getattr__(name: str):
    if name in _RESULT_IMPORTS:
        import importlib

        module = importlib.import_module(_RESULT_IMPORTS[name])
        return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
