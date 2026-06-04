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
Shared constants for Nova Forge SDK.
"""

import os
import re
from typing import Dict, List

from amzn_nova_forge.core.enums import (
    EvaluationMetric,
    EvaluationStrategy,
    EvaluationTask,
    Model,
    TrainingMethod,
)

DEFAULT_REGION = "us-east-1"

# MTRL CloudWatch log groups
MTRL_TRAIN_LOG_GROUP = "/aws/sagemaker/Job/AgentRFT"
MTRL_EVAL_LOG_GROUP = "/aws/sagemaker/Job/AgentRFTEvaluation"

# MTRL pipeline execution ARN pattern
MTRL_PIPELINE_EXECUTION_RE = re.compile(
    r"^arn:aws:sagemaker:[a-z0-9-]+:\d{12}:pipeline/.+/execution/.+"
)
DEFAULT_JOB_CACHE_DIR = "~/.nova-forge/cache"
DEFAULT_BATCH_TRACE_CACHE_DIR = "~/.nova-forge/batch_trace_cache"
BATCH_TRACE_LOG_SUBDIR = "batch_tracing"

# Data mixing field name patterns
DATAMIX_NOVA_PREFIX = "nova_"
DATAMIX_PERCENT_SUFFIX = "_percent"
DATAMIX_CUSTOMER_DATA_FIELD = "customer_data_percent"
DATAMIX_DATASET_CATALOG_FIELD = "dataset_catalog"

REGION_TO_ESCROW_ACCOUNT_MAPPING = {
    "us-east-1": "708977205387",
    "us-west-2": "176779409107",
    "eu-west-2": "470633809225",
}

# Supported SageMaker Inference configurations per (Model, instance_type).
# Each entry is a list of (max_context_length, max_concurrency) tiers, sorted by context length.
# Source: https://docs.aws.amazon.com/nova/latest/nova2-userguide/nova-model-sagemaker-inference.html
SUPPORTED_SMI_CONFIGS = {
    # NOVA_MICRO
    (Model.NOVA_MICRO, "ml.g5.12xlarge"): [(4000, 12), (8000, 6)],
    (Model.NOVA_MICRO, "ml.g5.24xlarge"): [(8000, 8)],
    (Model.NOVA_MICRO, "ml.g6e.xlarge"): [(8000, 2)],
    (Model.NOVA_MICRO, "ml.g6e.2xlarge"): [(8000, 2)],
    (Model.NOVA_MICRO, "ml.g6e.4xlarge"): [(8000, 4)],
    (Model.NOVA_MICRO, "ml.g6.12xlarge"): [(4000, 12), (8000, 6)],
    (Model.NOVA_MICRO, "ml.g6.24xlarge"): [(8000, 8)],
    (Model.NOVA_MICRO, "ml.g6.48xlarge"): [(8000, 12)],
    (Model.NOVA_MICRO, "ml.p5.48xlarge"): [(16000, 128), (64000, 32), (128000, 8)],
    # NOVA_LITE
    (Model.NOVA_LITE, "ml.g6.12xlarge"): [(8000, 2)],
    (Model.NOVA_LITE, "ml.g6.24xlarge"): [(8000, 4)],
    (Model.NOVA_LITE, "ml.g6.48xlarge"): [(4000, 16), (8000, 8)],
    (Model.NOVA_LITE, "ml.p5.48xlarge"): [(16000, 128), (60000, 8)],
    # NOVA_PRO
    (Model.NOVA_PRO, "ml.p5.48xlarge"): [(8000, 8), (16000, 2), (24000, 1)],
    # NOVA_LITE_2
    (Model.NOVA_LITE_2, "ml.g6.48xlarge"): [(8000, 8)],
    (Model.NOVA_LITE_2, "ml.p5.48xlarge"): [(16000, 128), (64000, 32), (128000, 8), (256000, 2)],
}

SUPPORTED_DATAMIXING_METHODS = [
    TrainingMethod.CPT,
    TrainingMethod.SFT_FULL,
    TrainingMethod.SFT_LORA,
]

HYPERPOD_RECIPE_PATH = os.path.join(
    "sagemaker_hyperpod_recipes",
    "recipes_collection",
    "recipes",
)

EVAL_TASK_STRATEGY_MAP: Dict[EvaluationTask, EvaluationStrategy] = {
    EvaluationTask.MMLU: EvaluationStrategy.ZERO_SHOT_COT,
    EvaluationTask.MMLU_PRO: EvaluationStrategy.ZERO_SHOT_COT,
    EvaluationTask.BBH: EvaluationStrategy.FEW_SHOT_COT,
    EvaluationTask.GPQA: EvaluationStrategy.ZERO_SHOT_COT,
    EvaluationTask.MATH: EvaluationStrategy.ZERO_SHOT_COT,
    EvaluationTask.STRONG_REJECT: EvaluationStrategy.ZERO_SHOT,
    EvaluationTask.IFEVAL: EvaluationStrategy.ZERO_SHOT,
    EvaluationTask.GEN_QA: EvaluationStrategy.GEN_QA,
    EvaluationTask.MMMU: EvaluationStrategy.ZERO_SHOT_COT,
    EvaluationTask.LLM_JUDGE: EvaluationStrategy.JUDGE,
    EvaluationTask.RUBRIC_LLM_JUDGE: EvaluationStrategy.JUDGE,
    EvaluationTask.RFT_EVAL: EvaluationStrategy.RFT_EVAL,
    EvaluationTask.RFT_MULTITURN_EVAL: EvaluationStrategy.RFT_MULTITURN_EVAL,
}

EVAL_TASK_METRIC_MAP: Dict[EvaluationTask, EvaluationMetric] = {
    EvaluationTask.MMLU: EvaluationMetric.ACCURACY,
    EvaluationTask.MMLU_PRO: EvaluationMetric.ACCURACY,
    EvaluationTask.BBH: EvaluationMetric.ACCURACY,
    EvaluationTask.GPQA: EvaluationMetric.ACCURACY,
    EvaluationTask.MATH: EvaluationMetric.EXACT_MATCH,
    EvaluationTask.STRONG_REJECT: EvaluationMetric.DEFLECTION,
    EvaluationTask.IFEVAL: EvaluationMetric.ACCURACY,
    EvaluationTask.GEN_QA: EvaluationMetric.ALL,
    EvaluationTask.MMMU: EvaluationMetric.ACCURACY,
    EvaluationTask.LLM_JUDGE: EvaluationMetric.ALL,
    EvaluationTask.RUBRIC_LLM_JUDGE: EvaluationMetric.ALL,
    EvaluationTask.RFT_EVAL: EvaluationMetric.ALL,
    EvaluationTask.RFT_MULTITURN_EVAL: EvaluationMetric.ALL,
}

# Available subtasks for each evaluation task (from nova_evaluator.py)
EVAL_AVAILABLE_SUBTASKS: Dict[EvaluationTask, List[str]] = {
    EvaluationTask.MMLU: [
        "abstract_algebra",
        "anatomy",
        "astronomy",
        "business_ethics",
        "clinical_knowledge",
        "college_biology",
        "college_chemistry",
        "college_computer_science",
        "college_mathematics",
        "college_medicine",
        "college_physics",
        "computer_security",
        "conceptual_physics",
        "econometrics",
        "electrical_engineering",
        "elementary_mathematics",
        "formal_logic",
        "global_facts",
        "high_school_biology",
        "high_school_chemistry",
        "high_school_computer_science",
        "high_school_european_history",
        "high_school_geography",
        "high_school_government_and_politics",
        "high_school_macroeconomics",
        "high_school_mathematics",
        "high_school_microeconomics",
        "high_school_physics",
        "high_school_psychology",
        "high_school_statistics",
        "high_school_us_history",
        "high_school_world_history",
        "human_aging",
        "human_sexuality",
        "international_law",
        "jurisprudence",
        "logical_fallacies",
        "machine_learning",
        "management",
        "marketing",
        "medical_genetics",
        "miscellaneous",
        "moral_disputes",
        "moral_scenarios",
        "nutrition",
        "philosophy",
        "prehistory",
        "professional_accounting",
        "professional_law",
        "professional_medicine",
        "professional_psychology",
        "public_relations",
        "security_studies",
        "sociology",
        "us_foreign_policy",
        "virology",
        "world_religions",
    ],
    EvaluationTask.BBH: [
        "boolean_expressions",
        "causal_judgement",
        "date_understanding",
        "disambiguation_qa",
        "dyck_languages",
        "formal_fallacies",
        "geometric_shapes",
        "hyperbaton",
        "logical_deduction_five_objects",
        "logical_deduction_seven_objects",
        "logical_deduction_three_objects",
        "movie_recommendation",
        "multistep_arithmetic_two",
        "navigate",
        "object_counting",
        "penguins_in_a_table",
        "reasoning_about_colored_objects",
        "ruin_names",
        "salient_translation_error_detection",
        "snarks",
        "sports_understanding",
        "temporal_sequences",
        "tracking_shuffled_objects_five_objects",
        "tracking_shuffled_objects_seven_objects",
        "tracking_shuffled_objects_three_objects",
        "web_of_lies",
        "word_sorting",
    ],
    EvaluationTask.MATH: [
        "algebra",
        "counting_and_probability",
        "geometry",
        "intermediate_algebra",
        "number_theory",
        "prealgebra",
        "precalculus",
    ],
    EvaluationTask.STRONG_REJECT: [
        "dangerous_or_sensitive_topics",
        "offensive_language",
    ],
    EvaluationTask.MMMU: [
        "accounting",
        "agriculture",
        "architecture_and_engineering",
        "art",
        "art_theory",
        "basic_medical_science",
        "biology",
        "chemistry",
        "clinical_medicine",
        "computer_science",
        "design",
        "economics",
        "electronics",
        "energy_and_power",
        "finance",
        "geography",
        "history",
        "literature",
        "manage",
        "marketing",
        "materials",
        "math",
        "mechanical_engineering",
        "music",
        "pharmacy",
        "physics",
        "psychology",
        "public_health",
        "sociology",
    ],
}


BYOD_AVAILABLE_EVAL_TASKS: List[str] = [
    EvaluationTask.GEN_QA.value,
    EvaluationTask.LLM_JUDGE.value,
    EvaluationTask.RUBRIC_LLM_JUDGE.value,
    EvaluationTask.RFT_EVAL.value,
    EvaluationTask.RFT_MULTITURN_EVAL.value,
]


def get_available_subtasks(task: EvaluationTask) -> List[str]:
    """
    Get available subtasks for a given evaluation task.

    Args:
        task: The evaluation task

    Returns:
        List of available subtasks for the task
    """
    return EVAL_AVAILABLE_SUBTASKS.get(task, [])


# Subset of BYOD_AVAILABLE_EVAL_TASKS that map to CustomScorerEvaluation in the serverless API.
# llm_judge and rubric_llm_judge are BYOD but use BenchmarkEvaluation instead.
SERVERLESS_CUSTOM_SCORER_EVAL_TASKS: List[str] = [
    EvaluationTask.GEN_QA.value,
    EvaluationTask.RFT_EVAL.value,
]


# --- Escrow URI tagging constants (used by util/bedrock.py and util/sagemaker.py) ---

import hashlib

from amzn_nova_forge.core.enums import ModelStatus

_BEDROCK_STATUS_MAP = {
    "Creating": ModelStatus.CREATING,
    "Active": ModelStatus.ACTIVE,
    "Failed": ModelStatus.FAILED,
}

ESCROW_URI_TAG_KEY = "sagemaker.amazonaws.com/forge/escrow-uri"

INSPECT_LENS_DEFAULT_IMAGE_REPO = "763104351884.dkr.ecr.{region}.amazonaws.com/sagemaker-inspect-ai"


def get_inspect_lens_default_image_uri(region: str) -> str:
    """Return the default InspectLens image URI for the given region."""
    return INSPECT_LENS_DEFAULT_IMAGE_REPO.format(region=region)


def _escrow_tag_value(escrow_uri: str) -> str:
    """Normalize escrow URI for use as a tag value (max 256 chars)."""
    if len(escrow_uri) <= 256:
        return escrow_uri
    return escrow_uri[:224] + "-" + hashlib.sha256(escrow_uri.encode()).hexdigest()[:31]
