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
"""Transform operations and the TransformMethod registry."""

from enum import Enum
from typing import Any, Callable, Dict, Iterator, List, Optional, cast
from urllib.parse import urlparse

import boto3
import jsonschema

from ...core.enums import EvaluationTask, Model, Platform, TrainingMethod
from ...util.iterator_utils import peek
from ...util.logging import logger
from ..data_state import DataLocation, DataState
from ..dataset_transformers import DatasetTransformer, TransformContext
from ..transform_format_schema import TRANSFORM_CONFIG
from .base import DataPrepError, NovaForgeTransformOperation, OperationResult


class SchemaTransformOperation(NovaForgeTransformOperation):
    """Transform the dataset between format schemas (e.g., generic -> Converse, OpenAI -> Converse)."""

    _TRANSFORMER_MAP = {
        "convert_to_converse_sft_nova_one": DatasetTransformer.convert_to_converse_sft_nova_one,
        "convert_to_converse_sft_nova_two": DatasetTransformer.convert_to_converse_sft_nova_two,
        "convert_openai_to_converse_sft_nova_one": DatasetTransformer.convert_openai_to_converse_sft_nova_one,
        "convert_openai_to_converse_sft_nova_two": DatasetTransformer.convert_openai_to_converse_sft_nova_two,
        "convert_to_openai_rft": DatasetTransformer.convert_to_openai_rft,
        "convert_to_evaluation": DatasetTransformer.convert_to_evaluation,
        "convert_to_cpt": DatasetTransformer.convert_to_cpt,
        "convert_to_rft_multiturn": DatasetTransformer.convert_to_rft_multiturn,
        "convert_to_rft_multiturn_serverless": DatasetTransformer.convert_to_rft_multiturn_serverless,
    }

    def execute(self, loader: Any, **kwargs) -> OperationResult:
        # Accept but ignore output_path — local transform operates in-memory.
        kwargs.pop("output_path", None)
        state = kwargs.pop("state", None)

        training_method: Optional[TrainingMethod] = kwargs.get("training_method")
        model: Optional[Model] = kwargs.get("model")
        eval_task: Optional[EvaluationTask] = kwargs.get("eval_task")
        region: Optional[str] = kwargs.get("region")
        platform = kwargs.get("platform")

        if training_method is None or model is None:
            raise ValueError("training_method and model are required for schema transforms.")

        if not self._has_data(loader):
            logger.info("Dataset is empty. Call load() method to load data first")
            return OperationResult(status="SKIPPED", output_state=state)

        training_method = self._resolve_eval_method(training_method, eval_task)
        transform_config = self._lookup_config(training_method, model)
        column_mappings = kwargs.get("column_mappings", {})
        multimodal_data_s3_path = kwargs.get("multimodal_data_s3_path")
        multimodal_data_bucket_owner = kwargs.get("multimodal_data_bucket_owner")

        # RFT Multiturn requires platform to determine output format
        is_mtrl = training_method in (
            TrainingMethod.RFT_MULTITURN_LORA,
            TrainingMethod.RFT_MULTITURN_FULL,
        )
        if is_mtrl and platform is None:
            raise ValueError(
                "platform is required for RFT Multiturn transforms. "
                "Pass platform=Platform.SMTJServerless or platform=Platform.SMHP. "
                "Example: loader.transform(method=TrainingMethod.RFT_MULTITURN_LORA, "
                "model=Model.NOVA_LITE_2, platform=Platform.SMTJServerless)"
            )

        if is_mtrl and platform == Platform.SMTJServerless:
            transform_config = dict(transform_config)
            transform_config["schema"] = None
            transform_config["transformers"] = [
                {
                    "source_schema": None,
                    "method": "convert_to_rft_multiturn_serverless",
                    "msg": "Transforming to flat RFT Multiturn format for Serverless.",
                },
            ]

        # Already in target format — nothing to do
        if transform_config["schema"] is not None and self._validate_against_schema(
            loader.dataset, transform_config["schema"]
        ):
            logger.info("Transform: %s", transform_config["success_msg"])
            return OperationResult(status="SUCCEEDED", output_state=state)

        self._apply_first_matching_transformer(
            loader,
            transform_config,
            column_mappings,
            multimodal_data_s3_path,
            multimodal_data_bucket_owner,
            region=region,
        )
        # Transform produces in-memory data — update state to LOCAL so that
        # _flush_pending() will materialize the dataset and prevent the lazy
        # generator from re-running the transform on subsequent terminal ops.
        local_state = DataState(
            path=state.path if state else "",
            format=state.format if state else "unknown",
            location=DataLocation.LOCAL,
            generator=loader.dataset,
        )
        return OperationResult(status="SUCCEEDED", output_state=local_state)

    @staticmethod
    def _has_data(loader: Any) -> bool:
        """Return True if the loader's dataset yields at least one record."""
        peeked, _ = peek(loader.dataset())
        return peeked is not None

    @staticmethod
    def _resolve_eval_method(training_method, eval_task):
        """Map EVALUATION + RFT_MULTITURN_EVAL to the concrete training method."""
        if training_method == TrainingMethod.EVALUATION and eval_task is None:
            logger.warning(
                "`eval_task` not provided for EVALUATION method. "
                "Using default evaluation transformation. "
                "For RFT Multiturn evaluation, pass "
                "eval_task=EvaluationTask.RFT_MULTITURN_EVAL"
            )

        if (
            training_method == TrainingMethod.EVALUATION
            and eval_task == EvaluationTask.RFT_MULTITURN_EVAL
        ):
            return TrainingMethod.RFT_MULTITURN_LORA

        return training_method

    @staticmethod
    def _lookup_config(training_method, model) -> Dict[str, Any]:
        """Find the TRANSFORM_CONFIG entry for this (method, model) pair."""
        for (methods, models), config in TRANSFORM_CONFIG.items():
            if (training_method in methods) and (
                models is None or models == model or (isinstance(models, tuple) and model in models)
            ):
                return cast(Dict[str, Any], config)

        raise ValueError(
            f"The combination of training method {training_method} and "
            f"model {model} is not yet supported.\n"
            f"Note: RFT is only supported on Nova 2.0."
        )

    def _apply_first_matching_transformer(
        self,
        loader: Any,
        transform_config: Dict[str, Any],
        column_mappings: dict,
        multimodal_data_s3_path: Optional[str],
        multimodal_data_bucket_owner: Optional[str],
        region: Optional[str] = None,
    ) -> None:
        """Try each transformer in order; apply the first whose source schema matches."""
        transformers: List[Dict[str, Any]] = transform_config.get("transformers", [])

        for transformer_info in transformers:
            source_schema = transformer_info.get("source_schema")
            should_apply = source_schema is None or self._validate_against_schema(
                loader.dataset, source_schema
            )

            if should_apply:
                method_name = transformer_info["method"]
                self._wire_transform_generator(
                    loader,
                    method_name,
                    transformer_info["msg"],
                    column_mappings,
                    source_schema,
                    multimodal_data_s3_path,
                    multimodal_data_bucket_owner,
                    region=region,
                )
                return

        raise DataPrepError(
            "Unable to transform dataset. No suitable transformer found for the given data format."
        )

    def _wire_transform_generator(
        self,
        loader: Any,
        method_name: str,
        description: str,
        column_mappings: dict,
        source_schema: Optional[dict],
        multimodal_data_s3_path: Optional[str],
        multimodal_data_bucket_owner: Optional[str],
        region: Optional[str] = None,
    ) -> None:
        """Replace loader.dataset with a generator that applies the transformer."""
        transformer_func = self._get_transformer_function(method_name)
        dataset_callable = loader.dataset

        error_suffix = (
            "\nMake sure to add the correct column mappings when calling transform()."
            if source_schema is None
            else ""
        )

        # Eagerly resolve multimodal S3 config so save() can validate bucket before iteration.
        transform_ctx = None
        if multimodal_data_s3_path is not None:
            if not multimodal_data_s3_path.startswith("s3://"):
                raise ValueError(
                    "multimodal_data_s3_path must be a valid S3 prefix starting with 's3://'"
                )
            parsed = urlparse(multimodal_data_s3_path)
            bucket = parsed.netloc
            prefix = parsed.path.lstrip("/")
            if prefix and not prefix.endswith("/"):
                prefix += "/"

            # Resolve bucket owner
            bucket_owner = multimodal_data_bucket_owner
            if bucket_owner is None:
                try:
                    sts_client = boto3.client("sts", region_name=region)
                    account_id = sts_client.get_caller_identity()["Account"]
                    logger.info("Auto-resolved bucket owner for %r: %s", bucket, account_id)
                    bucket_owner = account_id
                except Exception as e:
                    raise ValueError(f"Failed to resolve bucket owner for '{bucket}': {e}")

            transform_ctx = TransformContext(
                bucket=bucket,
                prefix=prefix,
                bucket_owner=bucket_owner,
            )
            loader._multimodal_image_bucket = bucket

        def transform_generator(
            captured_dataset=dataset_callable,
            captured_source_schema=source_schema,
            captured_error_suffix=error_suffix,
        ):
            logger.info("Transform: %s (runtime=Local)", description)
            logger.info("  Input: in-memory (dict)")
            logger.info("  Output: in-memory")
            # Create S3 client once for the entire transform pass.
            s3_client = (
                boto3.client("s3", region_name=region) if transform_ctx is not None else None
            )
            count = 0
            try:
                for rec in captured_dataset():
                    yield transformer_func(rec, column_mappings, transform_ctx, s3_client)
                    count += 1
            except Exception as e:
                error_type = (
                    "using generic format"
                    if captured_source_schema is None
                    else "from detected format"
                )
                raise DataPrepError(
                    f"Error transforming dataset {error_type}: {str(e)}{captured_error_suffix}"
                )
            logger.info("Transform complete: %s — %d samples processed", method_name, count)

        loader.dataset = transform_generator

    @staticmethod
    def _get_transformer_function(method_name: str):
        if method_name not in SchemaTransformOperation._TRANSFORMER_MAP:
            raise ValueError(f"Unknown transformer method: {method_name}")
        return SchemaTransformOperation._TRANSFORMER_MAP[method_name]

    @staticmethod
    def _validate_against_schema(dataset_callable, schema: dict) -> bool:
        try:
            for row in dataset_callable():
                jsonschema.validate(instance=row, schema=schema)
            return True
        except jsonschema.exceptions.ValidationError:
            return False


class TransformMethod(Enum):
    """Supported dataset transform methods."""

    SCHEMA = "schema"


def get_transform_operation(method: TransformMethod) -> NovaForgeTransformOperation:
    """Factory that returns the operation instance for a given TransformMethod."""
    registry = {
        TransformMethod.SCHEMA: SchemaTransformOperation,
    }
    op_class = registry.get(method)
    if op_class is None:
        raise ValueError(
            f"Transform method '{method.value}' is not yet implemented. "
            f"Supported: {[m.value for m in TransformMethod]}."
        )
    return op_class()
