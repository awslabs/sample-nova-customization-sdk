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
"""Result classes for Nova Forge SDK job lifecycle (training, evaluation, inference)."""

from amzn_nova_forge.core.result.eval_result import (
    BedrockEvaluationResult as BedrockEvaluationResult,
)
from amzn_nova_forge.core.result.eval_result import (
    EvaluationResult as EvaluationResult,
)
from amzn_nova_forge.core.result.eval_result import (
    SMHPEvaluationResult as SMHPEvaluationResult,
)
from amzn_nova_forge.core.result.eval_result import (
    SMTJEvaluationResult as SMTJEvaluationResult,
)
from amzn_nova_forge.core.result.inference_result import (
    InferenceResult as InferenceResult,
)
from amzn_nova_forge.core.result.inference_result import (
    SingleInferenceResult as SingleInferenceResult,
)
from amzn_nova_forge.core.result.inference_result import (
    SMTJBatchInferenceResult as SMTJBatchInferenceResult,
)
from amzn_nova_forge.core.result.job_result import (
    BaseJobResult as BaseJobResult,
)
from amzn_nova_forge.core.result.job_result import (
    BedrockStatusManager as BedrockStatusManager,
)
from amzn_nova_forge.core.result.job_result import (
    JobStatus as JobStatus,
)
from amzn_nova_forge.core.result.job_result import (
    JobStatusManager as JobStatusManager,
)
from amzn_nova_forge.core.result.job_result import (
    MTRLStatusManager as MTRLStatusManager,
)
from amzn_nova_forge.core.result.job_result import (
    SMHPStatusManager as SMHPStatusManager,
)
from amzn_nova_forge.core.result.job_result import (
    SMTJStatusManager as SMTJStatusManager,
)
from amzn_nova_forge.core.result.training_result import (
    BedrockTrainingResult as BedrockTrainingResult,
)
from amzn_nova_forge.core.result.training_result import (
    SMHPTrainingResult as SMHPTrainingResult,
)
from amzn_nova_forge.core.result.training_result import (
    SMTJTrainingResult as SMTJTrainingResult,
)
from amzn_nova_forge.core.result.training_result import (
    TrainingResult as TrainingResult,
)
