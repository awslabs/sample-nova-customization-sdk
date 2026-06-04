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
"""Validate operations and the ValidateMethod registry."""

from enum import Enum
from typing import Any, Optional

from ...util.iterator_utils import peek
from ...util.logging import logger
from .base import NovaForgeValidateOperation


class SchemaValidateOperation(NovaForgeValidateOperation):
    """Validate the dataset against schema requirements for a training method and model."""

    def execute(self, loader: Any, **kwargs) -> None:
        from ...model.model_enums import Model, Platform, TrainingMethod
        from ...recipe.recipe_config import EvaluationTask
        from ..dataset_validator.dataset_schema_resolver import resolve_schema_validator

        training_method: Optional[TrainingMethod] = kwargs.get("training_method")
        model: Optional[Model] = kwargs.get("model")
        eval_task: Optional[EvaluationTask] = kwargs.get("eval_task")
        platform: Optional[Platform] = kwargs.get("platform")

        if training_method is None or model is None:
            raise ValueError("training_method and model are required for schema validation.")

        dataset_iter = loader.dataset()
        peeked_value, dataset_iter = peek(dataset_iter)

        if peeked_value is None:
            logger.info("Validate: dataset is empty")
            return

        if training_method == TrainingMethod.EVALUATION and eval_task is None:
            logger.warning(
                "`eval_task` not provided for EVALUATION method. "
                "Using default evaluation validation. "
                "For RFT Multiturn evaluation, pass eval_task=EvaluationTask.RFT_MULTITURN_EVAL"
            )

        # For Serverless MTRL, use the serverless validator
        is_mtrl = training_method in (
            TrainingMethod.RFT_MULTITURN_LORA,
            TrainingMethod.RFT_MULTITURN_FULL,
        )
        if is_mtrl and platform == Platform.SMTJServerless:
            from amzn_nova_forge.dataset.dataset_validator.rft_multiturn_dataset_validator import (
                RFTMultiturnServerlessValidator,
            )

            validator: Optional[Any] = RFTMultiturnServerlessValidator(model)
        else:
            validator = resolve_schema_validator(training_method, model, eval_task)

        if validator is None:
            logger.info(
                "Validate: skipped — not available for model=%s / method=%s",
                model,
                training_method,
            )
            return

        row_kwargs: dict[str, Any] = dict(
            platform=platform,
            training_method=training_method,
        )
        validator.validate(dataset_iter, model, **row_kwargs)
        logger.info(
            "Validate: schema passed (model=%s, method=%s)", model.value, training_method.value
        )


class ValidateMethod(Enum):
    """Supported dataset validation methods."""

    SCHEMA = "schema"  # Deprecated, kept for backwards compatibility
    INVALID_RECORDS = "invalid_records"


def get_validate_operation(method: ValidateMethod) -> NovaForgeValidateOperation:
    """Factory that returns the operation instance for a given ValidateMethod."""
    registry = {
        ValidateMethod.SCHEMA: SchemaValidateOperation,  # Deprecated, kept for backwards compatibility
        ValidateMethod.INVALID_RECORDS: SchemaValidateOperation,
    }
    op_class = registry.get(method)
    if op_class is None:
        raise ValueError(
            f"Validate method '{method.value}' is not yet implemented. "
            f"Supported: {[m.value for m in ValidateMethod]}."
        )
    return op_class()
