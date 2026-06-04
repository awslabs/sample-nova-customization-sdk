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
"""ForgeInference — owns real-time and batch inference."""

import uuid
from datetime import datetime, timezone
from typing import Any, Dict, Optional, cast

import boto3

from amzn_nova_forge.core.constants import DEFAULT_REGION
from amzn_nova_forge.core.enums import Model, Platform, TrainingMethod
from amzn_nova_forge.core.job_cache import (
    build_cache_context,
    load_existing_result,
    persist_result,
)
from amzn_nova_forge.core.result import SMTJBatchInferenceResult, TrainingResult
from amzn_nova_forge.core.result.inference_result import InferenceResult
from amzn_nova_forge.core.runtime import RuntimeManager
from amzn_nova_forge.core.types import (
    ForgeConfig,
    JobConfig,
)
from amzn_nova_forge.manager.runtime_manager import SMHPRuntimeManager
from amzn_nova_forge.model.nova_model_customizer_util import (
    resolve_model_checkpoint_path,
    set_output_s3_path,
)
from amzn_nova_forge.monitor.log_monitor import CloudWatchLogMonitor
from amzn_nova_forge.recipe.recipe_builder import RecipeBuilder
from amzn_nova_forge.telemetry import UNKNOWN, Feature, _telemetry_emitter
from amzn_nova_forge.util.bedrock import invoke_model
from amzn_nova_forge.util.logging import logger
from amzn_nova_forge.util.platform_util import (
    detect_platform_from_path,
    validate_platform_compatibility,
)
from amzn_nova_forge.util.sagemaker import invoke_sagemaker_inference
from amzn_nova_forge.validation.endpoint_validator import (
    SAGEMAKER_ENDPOINT_ARN_REGEX,
    validate_endpoint_arn,
)


class ForgeInference:
    """Encapsulates real-time and batch inference for Nova models.

    ``model`` and ``infra`` are optional — only needed for ``invoke_batch()``.
    Real-time ``invoke()`` only needs ``region``.
    """

    def __init__(
        self,
        region: Optional[str] = None,
        model: Optional[Model] = None,
        infra: Optional[RuntimeManager] = None,
        config: Optional[ForgeConfig] = None,
        method: Optional[TrainingMethod] = None,
        hub_content_version: Optional[str] = None,
    ) -> None:
        self.region = region or boto3.session.Session().region_name or DEFAULT_REGION
        self.model = model
        self.infra = infra
        self.method = method
        self._config = config or ForgeConfig()
        self.hub_content_version = hub_content_version

        self._platform: Optional[Platform] = None
        if infra is not None:
            self._platform = infra.platform

        # Job caching context
        self._cache_context = build_cache_context(
            self._config,
            model=model,
            method=method,
            instance_type=infra.instance_type if infra else None,
            instance_count=infra.instance_count if infra else None,
        )

    @_telemetry_emitter(
        Feature.DEPLOY,
        "invoke",
        extra_info_fn=lambda self, *args, **kwargs: {
            "model": self.model.value if self.model else UNKNOWN,
        },
    )
    def invoke(
        self,
        endpoint_arn: str,
        request_body: Dict[str, Any],
        inference_component_name: Optional[str] = None,
    ) -> Any:
        """Invoke real-time inference against an endpoint.

        Args:
            endpoint_arn: ARN of the deployed endpoint.
            request_body: Inference request body.
            inference_component_name: Optional inference component to target on a
                SageMaker endpoint. Required for IC-enabled endpoints.

        Returns:
            Inference result.
        """
        validate_endpoint_arn(endpoint_arn=endpoint_arn)

        if SAGEMAKER_ENDPOINT_ARN_REGEX.match(endpoint_arn):
            runtime_client = boto3.client("sagemaker-runtime", region_name=self.region)
            endpoint_name = endpoint_arn.split("/")[-1]
            logger.info(f"Invoking SageMaker endpoint: {endpoint_name}")
            return invoke_sagemaker_inference(
                request_body,
                endpoint_name,
                runtime_client,
                inference_component_name=inference_component_name,
            )
        else:
            runtime_client = boto3.client("bedrock-runtime", region_name=self.region)
            return invoke_model(
                model_id=endpoint_arn,
                request_body=request_body,
                bedrock_runtime=runtime_client,
            )

    @_telemetry_emitter(
        Feature.BATCH_INFERENCE,
        "invoke_batch",
        extra_info_fn=lambda self, *args, **kwargs: {
            "model": self.model.value if self.model else UNKNOWN,
            "platform": self._platform if self._platform else UNKNOWN,
            "dryRun": kwargs.get("dry_run", False),
        },
    )
    def invoke_batch(
        self,
        job_name: str,
        input_path: str,
        output_s3_path: str,
        model_path: Optional[str] = None,
        recipe_path: Optional[str] = None,
        overrides: Optional[Dict[str, Any]] = None,
        dry_run: bool = False,
        job_result: Optional[TrainingResult] = None,
    ) -> Optional[InferenceResult]:
        """Launch a batch inference job.

        Args:
            job_name: Name for the batch inference job.
            input_path: S3 path to input data.
            output_s3_path: S3 path for inference outputs.
            model_path: Optional S3 path to the model checkpoint.
            recipe_path: Optional path to a YAML recipe file.
            overrides: Optional configuration overrides.
            dry_run: If True, only validate — do not start a job.
            job_result: Optional TrainingResult to extract checkpoint path
                from if model_path is not provided.

        Returns:
            InferenceResult on success, None if dry_run is True.

        Raises:
            ValueError: If model, infra, or method were not provided.
        """
        if self.model is None or self.infra is None or self._platform is None:
            raise ValueError(
                "model and infra are required for batch inference. "
                "Provide them in the ForgeInference constructor."
            )

        # Check job cache
        cached = load_existing_result(
            self._cache_context,
            job_name=job_name,
            job_type="batch_inference",
            model_path=model_path,
            recipe_path=recipe_path,
            overrides=overrides or {},
        )
        if cached:
            logger.info("Returning cached result for '%s'.", job_name)
            return cached  # type: ignore[return-value]

        platform: Platform = self._platform

        if self.method is None:
            raise ValueError(
                "method is required for batch inference (used for recipe resolution). "
                "Provide method=TrainingMethod.XXX in the ForgeInference constructor."
            )

        if self._platform == Platform.BEDROCK:
            raise NotImplementedError(
                "Batch inference is not supported on Bedrock platform. "
                "Use invoke() for single requests instead."
            )

        # Resolve model checkpoint
        resolved_model_path = resolve_model_checkpoint_path(
            model_path=model_path,
            job_result=job_result,
            customizer_job_id=None,
            customizer_output_s3_path=None,
            customizer_model_path=None,
        )

        if resolved_model_path is None:
            logger.warning(
                f"Could not resolve model checkpoint path for batch inference! "
                f"Falling back to base model {self.model}"
            )

        # Validate platform compatibility
        checkpoint_platform = None
        if resolved_model_path and resolved_model_path.startswith("s3://"):
            checkpoint_platform = detect_platform_from_path(resolved_model_path)

        if checkpoint_platform is None and job_result is not None:
            if job_result.model_artifacts.checkpoint_s3_path:
                checkpoint_platform = detect_platform_from_path(
                    job_result.model_artifacts.checkpoint_s3_path
                )

        validate_platform_compatibility(
            checkpoint_platform=checkpoint_platform,
            execution_platform=platform,
            checkpoint_source="batch inference model checkpoint",
        )

        resolved_output = set_output_s3_path(
            region=self.region,
            output_s3_path=output_s3_path,
            kms_key_id=self.infra.kms_key_id,
        )

        recipe_builder = RecipeBuilder(
            region=self.region,
            job_name=job_name,
            platform=platform,
            model=self.model,
            method=self.method,
            instance_type=self.infra.instance_type,
            instance_count=self.infra.instance_count,
            infra=self.infra,
            data_s3_path=input_path,
            output_s3_path=resolved_output,
            model_path=resolved_model_path,
            image_uri_override=self._config.image_uri,
            hub_content_version=self.hub_content_version,
        )

        (
            resolved_recipe_path,
            resolved_output_s3_path,
            resolved_data_s3_path,
            resolved_image_uri,
        ) = recipe_builder.build_and_validate(
            overrides=overrides,
            input_recipe_path=recipe_path,
            output_recipe_path=self._config.generated_recipe_dir,
            validation_config=self._config.validation_config,
        )

        if dry_run:
            return None

        unique_job_name = f"{job_name}-{uuid.uuid4()}"[:63].rstrip("-")
        start_time = datetime.now(timezone.utc)

        job_id = self.infra.execute(
            job_config=JobConfig(
                job_name=unique_job_name,
                data_s3_path=resolved_data_s3_path,
                output_s3_path=resolved_output_s3_path,
                image_uri=resolved_image_uri,
                recipe_path=resolved_recipe_path,
                input_s3_data_type="S3Prefix",
            )
        )

        inference_output_s3_path = (
            f"{resolved_output_s3_path.rstrip('/')}/{job_id}/output/output.tar.gz"
        )
        batch_result = SMTJBatchInferenceResult(
            job_id=job_id,
            started_time=start_time,
            inference_output_path=inference_output_s3_path,
            region=self.region,
        )
        logger.info(
            f"Started batch inference job '{job_id}'. \n"
            f"Artifacts will be published to {inference_output_s3_path}.\n"
            f"After opening the tar file, look for {recipe_builder.job_name}/eval_results/inference_output.jsonl."
        )
        persist_result(
            self._cache_context,
            batch_result,
            job_name=job_name,
            job_type="batch_inference",
            model_path=model_path,
            recipe_path=recipe_path,
            overrides=overrides or {},
        )

        return batch_result

    @_telemetry_emitter(
        Feature.BATCH_INFERENCE,
        "get_logs",
        extra_info_fn=lambda self, *args, **kwargs: {
            "model": self.model.value if self.model else UNKNOWN,
            "platform": self._platform if self._platform else UNKNOWN,
        },
    )
    def get_logs(
        self,
        job_result: Optional[InferenceResult] = None,
        job_id: Optional[str] = None,
        started_time: Optional[datetime] = None,
        limit: Optional[int] = None,
        start_from_head: bool = False,
        end_time: Optional[int] = None,
    ) -> None:
        """Stream CloudWatch logs for a batch inference job.

        Provide either a ``job_result`` or explicit ``job_id`` + ``started_time``.

        Raises:
            ValueError: If neither ``job_result`` nor both ``job_id`` and
                ``started_time`` are provided, or if the platform cannot be
                determined (no ``infra`` in constructor).
        """
        resolved_job_id = job_result.job_id if job_result else job_id
        resolved_started = job_result.started_time if job_result else started_time

        if not resolved_job_id or not resolved_started:
            raise ValueError(
                "No job reference provided. Pass either a job_result or explicit job_id and started_time."
            )

        if self._platform is None:
            raise ValueError(
                "Cannot determine platform. Provide infra in the ForgeInference constructor."
            )

        platform: Platform = self._platform
        kwargs: Dict[str, Any] = {}
        if self._platform == Platform.SMHP and self.infra is not None:
            kwargs["cluster_name"] = cast(SMHPRuntimeManager, self.infra).cluster_name
            kwargs["namespace"] = cast(SMHPRuntimeManager, self.infra).namespace

        monitor = CloudWatchLogMonitor(
            job_id=resolved_job_id,
            platform=platform,
            started_time=int(resolved_started.timestamp() * 1000),
            region=self.region,
            **kwargs,
        )
        monitor.show_logs(limit=limit, start_from_head=start_from_head, end_time=end_time)
