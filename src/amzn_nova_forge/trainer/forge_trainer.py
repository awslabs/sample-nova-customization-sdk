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
"""ForgeTrainer — owns the training workflow."""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Optional, cast

import boto3

from amzn_nova_forge.core.constants import (
    BATCH_TRACE_LOG_SUBDIR,
    DEFAULT_BATCH_TRACE_CACHE_DIR,
    DEFAULT_REGION,
    SUPPORTED_DATAMIXING_METHODS,
)
from amzn_nova_forge.core.enums import Model, Platform, TrainingMethod
from amzn_nova_forge.core.job_cache import (
    build_cache_context,
    load_existing_result,
    persist_result,
)
from amzn_nova_forge.core.result import (
    BedrockTrainingResult,
    SMHPTrainingResult,
    SMTJTrainingResult,
    TrainingResult,
)
from amzn_nova_forge.core.result.job_result import JobStatus
from amzn_nova_forge.core.runtime import RuntimeManager
from amzn_nova_forge.core.training_overrides import TrainingOverrides
from amzn_nova_forge.core.types import (
    ForgeConfig,
    JobConfig,
    ModelArtifacts,
    RecipeConfig,
    validate_region,
)
from amzn_nova_forge.manager.runtime_manager import SMHPRuntimeManager, SMTJServerlessRuntimeManager
from amzn_nova_forge.model.nova_model_customizer_util import set_output_s3_path
from amzn_nova_forge.monitor.log_monitor import CloudWatchLogMonitor
from amzn_nova_forge.recipe.recipe_builder import RecipeBuilder
from amzn_nova_forge.telemetry import Feature, _telemetry_emitter
from amzn_nova_forge.trainer.utils.batch_trace import run as batch_trace_run
from amzn_nova_forge.util.data_mixing import DataMixing
from amzn_nova_forge.util.data_utils import is_multimodal_data
from amzn_nova_forge.util.logging import logger
from amzn_nova_forge.util.metric_util import (
    _build_and_upload_training_metrics_csv,
    _parse_user_time,
)
from amzn_nova_forge.util.recipe import load_recipe_templates
from amzn_nova_forge.util.sagemaker import get_model_artifacts
from amzn_nova_forge.validation.endpoint_validator import is_sagemaker_arn
from amzn_nova_forge.validation.validator import validate_rft_lambda_name

if TYPE_CHECKING:
    from amzn_nova_forge.rft_multiturn import RFTMultiturnInfrastructure


class ForgeTrainer:
    """Encapsulates the training workflow for Nova model customization.

    Configuration is provided in the constructor; ``train()`` accepts only
    per-job parameters.  Follows the RecipeBuilder pattern — no shared
    mutable state between calls.
    """

    def __init__(
        self,
        model: Model,
        method: TrainingMethod,
        infra: RuntimeManager,
        training_data_s3_path: Optional[str] = None,
        model_s3_path: Optional[str] = None,
        model_arn: Optional[str] = None,
        data_mixing_enabled: bool = False,
        holdout_data_s3_path: Optional[str] = None,
        val_check_interval: Optional[int] = None,
        config: Optional[ForgeConfig] = None,
        region: Optional[str] = None,
        is_multimodal: Optional[bool] = None,
        hub_content_version: Optional[str] = None,
        enable_batch_sample_tracing: bool = False,
    ) -> None:
        self.model = model
        self.method = method
        self.infra = infra
        self.training_data_s3_path = training_data_s3_path
        self.model_arn = model_arn
        self.model_s3_path = model_s3_path
        self.holdout_data_s3_path = holdout_data_s3_path
        self.hub_content_version = hub_content_version
        if hub_content_version and hasattr(infra, "hub_content_version"):
            infra.hub_content_version = hub_content_version
        self._enable_batch_sample_tracing = enable_batch_sample_tracing
        self.val_check_interval: Optional[int] = None
        if val_check_interval is not None:
            if (
                isinstance(val_check_interval, bool)
                or not isinstance(val_check_interval, int)
                or val_check_interval < 1
            ):
                raise ValueError(
                    f"val_check_interval must be a positive integer, got: {val_check_interval}"
                )
            self.val_check_interval = val_check_interval
        if val_check_interval is not None and holdout_data_s3_path is None:
            logger.warning(
                "val_check_interval is set but no holdout_data_s3_path provided. val_check_interval will have no effect without validation data."
            )

        self._config = config or ForgeConfig()

        self.region = region or boto3.session.Session().region_name or DEFAULT_REGION
        validate_region(self.region)

        self._s3_client = None

        self._platform = infra.platform

        if self._platform == Platform.SMTJServerless:
            if self.model_s3_path is not None and self.model_arn is not None:
                raise ValueError(
                    "Cannot specify both model_s3_path and model_arn. "
                    "For SMTJServerless, use model_arn with a model package ARN."
                )
            if self.model_s3_path is not None:
                if not is_sagemaker_arn(self.model_s3_path):
                    raise ValueError(
                        f"For SMTJServerless, model_s3_path must be a SageMaker model package ARN, "
                        f"got: '{self.model_s3_path}'. Use model_arn instead."
                    )
                self.model_arn = self.model_s3_path
                self.model_s3_path = None
            if self.model_arn is not None and not is_sagemaker_arn(self.model_arn):
                raise ValueError(
                    f"model_arn must be a SageMaker model package ARN, got: '{self.model_arn}'."
                )
        if self._platform == Platform.BEDROCK and self.model_s3_path is not None:
            logger.warning(
                "model_path is not used for Bedrock platform. "
                "To specify a base model, pass base_model_identifier to "
                "BedrockRuntimeManager constructor instead."
            )
        if enable_batch_sample_tracing:
            if self._platform != Platform.SMHP:
                raise ValueError(
                    "enable_batch_sample_tracing is only supported on SageMaker HyperPod (SMHP)."
                )
            if method != TrainingMethod.CPT:
                raise ValueError("enable_batch_sample_tracing is only supported for CPT training.")

        self.output_s3_path = set_output_s3_path(
            region=self.region,
            output_s3_path=self._config.output_s3_path,
            kms_key_id=self.infra.kms_key_id,
        )

        # is_multimodal resolution (matches original NovaModelCustomizer lines 230-252)
        if not data_mixing_enabled:
            if is_multimodal is not None:
                logger.warning("is_multimodal is ignored because data_mixing_enabled=False.")
            self._is_multimodal = False
        elif is_multimodal is not None:
            self._is_multimodal = is_multimodal
        elif training_data_s3_path:
            self._is_multimodal = is_multimodal_data(training_data_s3_path)
            if self._is_multimodal:
                logger.info(
                    "Multimodal data detected. Using multimodal datamix recipes. "
                    "To skip auto-detection, pass is_multimodal=False."
                )
        else:
            self._is_multimodal = False

        # Data mixing setup (SMHP and SMTJServerless, CPT/SFT methods)
        self.data_mixing: Optional[DataMixing] = None
        if data_mixing_enabled:
            _datamix_platforms = (Platform.SMHP, Platform.SMTJServerless)
            if (
                self._platform not in _datamix_platforms
                or method not in SUPPORTED_DATAMIXING_METHODS
            ):
                raise ValueError(
                    f"Data mixing is only supported for {SUPPORTED_DATAMIXING_METHODS} "
                    "training methods on SageMaker HyperPod or SMTJServerless."
                )
            self.data_mixing = DataMixing()
            (
                _metadata,
                _template,
                overrides_template,
                _image_uri,
            ) = load_recipe_templates(
                model=model,
                method=method,
                platform=self._platform,
                region=self.region,
                data_mixing_enabled=True,
                instance_type=self.infra.instance_type,
                image_uri_override=self._config.image_uri,
                is_multimodal=self._is_multimodal,
                hub_content_version=self.hub_content_version,
            )
            if overrides_template:
                self.data_mixing._load_defaults_from_template(overrides_template)

        # Job caching context
        self._cache_context = build_cache_context(
            self._config,
            model=model,
            method=method,
            data_s3_path=training_data_s3_path,
            model_path=self.model_arn or self.model_s3_path,
            output_s3_path=self.output_s3_path,
            instance_type=infra.instance_type,
            instance_count=infra.instance_count,
        )

    @_telemetry_emitter(
        Feature.TRAINING,
        "train",
        extra_info_fn=lambda self, *args, **kwargs: {
            "method": self.method,
            "model": self.model.value,
            "platform": self._platform,
            "dryRun": kwargs.get("dry_run", False),
            "hasValidationData": self.holdout_data_s3_path is not None,
        },
    )
    def train(
        self,
        job_name: str,
        recipe_path: Optional[str] = None,
        overrides: Optional[TrainingOverrides] = None,
        rft_lambda_arn: Optional[str] = None,
        dry_run: bool = False,
        rft_multiturn_infra: Optional[RFTMultiturnInfrastructure] = None,
    ) -> Optional[TrainingResult]:
        """Launch a training job.

        Args:
            job_name: User-defined name for the training job.
            recipe_path: Optional path to a YAML recipe file.
            overrides: Optional dictionary of configuration overrides.
            rft_lambda_arn: Optional RFT Lambda ARN. Falls back to infra attribute.
            dry_run: If True, only validate — do not start a job.
            rft_multiturn_infra: Optional RFT multiturn infrastructure (passed
                through to RecipeBuilder for multi-turn RFT workflows).

        Returns:
            TrainingResult on success, None if dry_run is True.
        """
        # Check job cache
        cached = load_existing_result(
            self._cache_context,
            job_name=job_name,
            job_type="train",
            recipe_path=recipe_path,
            overrides=overrides or {},
        )
        if cached:
            logger.info("Returning cached result for '%s'.", job_name)
            return cached  # type: ignore[return-value]

        rft_lambda_arn = rft_lambda_arn or getattr(self.infra, "rft_lambda_arn", None)

        if rft_lambda_arn:
            validate_rft_lambda_name(rft_lambda_arn.split(":")[-1], self._platform)
            logger.info(f"Using reward lambda: {rft_lambda_arn}")

        # MTRL serverless: skip recipe generation, call directly
        is_mtrl_serverless = self._platform == Platform.SMTJServerless and self.method in (
            TrainingMethod.RFT_MULTITURN_LORA,
            TrainingMethod.RFT_MULTITURN_FULL,
        )
        if is_mtrl_serverless:
            if not self._config.mlflow_monitor or not self._config.mlflow_monitor.tracking_uri:
                raise ValueError(
                    "MLflow configuration is required for AgentRFT jobs. "
                    "Please provide an mlflow_monitor with a valid tracking_uri when "
                    "using RFT_MULTITURN methods on the SMTJServerless platform."
                )

            if dry_run:
                return None

            mtrl_infra = cast(SMTJServerlessRuntimeManager, self.infra)
            mlflow_uri = self._config.mlflow_monitor.tracking_uri

            unique_job_name = f"{job_name}-{uuid.uuid4().hex[:8]}"[:48].rstrip("-")
            start_time = datetime.now(timezone.utc)

            job_id = mtrl_infra.execute_mtrl(
                model=self.model,
                job_name=unique_job_name,
                data_s3_path=self.training_data_s3_path,
                output_s3_path=self.output_s3_path,
                mlflow_tracking_uri=mlflow_uri,
                overrides=dict(overrides) if overrides else None,
                model_path=self.model_arn,
            )

            training_result: TrainingResult = SMTJTrainingResult(
                job_id=job_id,
                started_time=start_time,
                method=self.method,
                model_type=self.model,
                model_artifacts=get_model_artifacts(
                    job_name=job_id,
                    infra=self.infra,
                    region=self.region,
                ),
                region=self.region,
            )

            logger.info(f"Started MTRL job '{training_result.job_id}'.")
            if training_result.model_artifacts.output_s3_path:
                logger.info(f"Output S3 path is: {training_result.model_artifacts.output_s3_path}.")

            persist_result(
                self._cache_context,
                training_result,
                job_name=job_name,
                job_type="train",
                recipe_path=recipe_path,
                overrides=overrides or {},
            )
            return training_result

        recipe_builder = RecipeBuilder(
            region=self.region,
            job_name=job_name,
            platform=self._platform,
            model=self.model,
            method=self.method,
            instance_type=self.infra.instance_type,
            instance_count=self.infra.instance_count,
            infra=self.infra,
            data_s3_path=self.training_data_s3_path,
            output_s3_path=self.output_s3_path,
            model_path=self.model_arn or self.model_s3_path,
            rft_lambda_arn=rft_lambda_arn,
            validation_data_s3_path=self.holdout_data_s3_path,
            val_check_interval=self.val_check_interval,
            data_mixing_instance=self.data_mixing,
            image_uri_override=self._config.image_uri,
            is_multimodal=self._is_multimodal,
            mlflow_monitor=self._config.mlflow_monitor,
            rft_multiturn_infra=rft_multiturn_infra,
            hub_content_version=self.hub_content_version,
            enable_batch_sample_tracing=self._enable_batch_sample_tracing,
        )

        (
            resolved_recipe_path,
            resolved_output_s3_path,
            resolved_data_s3_path,
            resolved_image_uri,
        ) = recipe_builder.build_and_validate(
            overrides=dict(overrides) if overrides else {},
            input_recipe_path=recipe_path,
            output_recipe_path=self._config.generated_recipe_dir,
            validation_config=self._config.validation_config,
        )

        if dry_run:
            return None

        unique_job_name = f"{job_name}-{uuid.uuid4()}"[:63].rstrip("-")
        start_time = datetime.now(timezone.utc)

        job_config_params: Dict[str, Any] = {
            "job_name": unique_job_name,
            "data_s3_path": resolved_data_s3_path,
            "output_s3_path": resolved_output_s3_path,
            "image_uri": resolved_image_uri,
            "recipe_path": resolved_recipe_path,
            "rft_lambda_arn": rft_lambda_arn,
            "validation_data_s3_path": self.holdout_data_s3_path,
            "input_s3_data_type": "Converse"
            if self.method not in (TrainingMethod.RFT_LORA, TrainingMethod.RFT_FULL)
            else "S3Prefix",
        }

        hp: Dict[str, str] = {}
        if self.val_check_interval is not None:
            hp["val_check_interval"] = str(self.val_check_interval)
        if hp:
            job_config_params["trainer_config_hyperparameters"] = hp

        if self._platform in (Platform.BEDROCK, Platform.SMTJServerless):
            job_config_params["method"] = self.method

        if self._platform == Platform.SMTJServerless and self.data_mixing:
            job_config_params["data_mixing_config"] = self.data_mixing.get_config()

        job_id = self.infra.execute(job_config=JobConfig(**job_config_params))

        if self._platform in (Platform.SMTJ, Platform.SMTJServerless):
            training_result = SMTJTrainingResult(
                job_id=job_id,
                started_time=start_time,
                method=self.method,
                model_type=self.model,
                model_artifacts=get_model_artifacts(
                    job_name=job_id,
                    infra=self.infra,
                    output_s3_path=resolved_output_s3_path,
                    region=self.region,
                ),
                region=self.region,
            )
        elif self._platform is Platform.BEDROCK:
            training_result = BedrockTrainingResult(
                job_id=job_id,
                started_time=start_time,
                method=self.method,
                model_type=self.model,
                model_artifacts=ModelArtifacts(
                    checkpoint_s3_path=None,
                    output_s3_path=resolved_output_s3_path,
                ),
                region=self.region,
            )
        else:
            cluster_name = cast(SMHPRuntimeManager, self.infra).cluster_name
            namespace = cast(SMHPRuntimeManager, self.infra).namespace
            training_result = SMHPTrainingResult(
                job_id=job_id,
                started_time=start_time,
                method=self.method,
                model_type=self.model,
                model_artifacts=get_model_artifacts(
                    job_name=unique_job_name,
                    infra=self.infra,
                    output_s3_path=resolved_output_s3_path,
                    region=self.region,
                ),
                cluster_name=cluster_name,
                namespace=namespace,
                region=self.region,
            )

        logger.info(f"Started job '{training_result.job_id}'.")
        if training_result.model_artifacts.checkpoint_s3_path:
            logger.info(
                f"Checkpoint S3 path is: {training_result.model_artifacts.checkpoint_s3_path}."
            )
        if training_result.model_artifacts.output_s3_path:
            logger.info(f"Output S3 path is: {training_result.model_artifacts.output_s3_path}.")

        persist_result(
            self._cache_context,
            training_result,
            job_name=job_name,
            job_type="train",
            recipe_path=recipe_path,
            overrides=overrides or {},
        )

        return training_result

    @_telemetry_emitter(
        Feature.TRAINING,
        "forgetrainer.get_config",
        extra_info_fn=lambda self, *args, **kwargs: {
            "method": self.method,
            "model": self.model,
            "platform": self._platform,
        },
    )
    def get_config(
        self,
        overrides: Optional[TrainingOverrides] = None,
    ) -> RecipeConfig:
        """Return overridable training parameters without starting a job.

        Not supported for RFT_MULTITURN methods (rft_multiturn_infra
        is not available outside of ``train()``).

        Args:
            overrides: Optional overrides to merge with defaults.

        Returns:
            Frozen RecipeConfig with all overridable parameters.
        """
        recipe_builder = RecipeBuilder(
            region=self.region,
            job_name="__config_preview__",
            platform=self._platform,
            model=self.model,
            method=self.method,
            instance_type=self.infra.instance_type,
            instance_count=self.infra.instance_count,
            infra=self.infra,
            data_s3_path=self.training_data_s3_path,
            output_s3_path=self.output_s3_path,
            model_path=self.model_arn or self.model_s3_path,
            validation_data_s3_path=self.holdout_data_s3_path,
            val_check_interval=self.val_check_interval,
            data_mixing_instance=self.data_mixing,
            image_uri_override=self._config.image_uri,
            is_multimodal=self._is_multimodal,
            mlflow_monitor=self._config.mlflow_monitor,
            hub_content_version=self.hub_content_version,
        )
        return recipe_builder.get_overridable_config(overrides=overrides)  # type: ignore[arg-type]

    @_telemetry_emitter(Feature.TRAINING, "get_logs")
    def get_logs(
        self,
        job_result: Optional[TrainingResult] = None,
        job_id: Optional[str] = None,
        started_time: Optional[datetime] = None,
        limit: Optional[int] = None,
        start_from_head: bool = False,
        end_time: Optional[int] = None,
        poll: int = 30,
        timeout: int = 7200,
    ) -> None:
        """Stream logs or wait for a training job.

        For MTRL jobs, delegates to ``job_result.wait()`` which displays a
        rich progress panel with status, metrics, and links.

        For all other jobs, streams CloudWatch logs.

        Provide either a ``job_result`` or explicit ``job_id`` + ``started_time``.

        Raises:
            ValueError: If neither ``job_result`` nor both ``job_id`` and
                ``started_time`` are provided.
        """
        # MTRL: delegate to wait() which shows progress
        if (
            job_result is not None
            and isinstance(job_result, SMTJTrainingResult)
            and job_result._is_mtrl
        ):
            job_result.wait(poll=poll, timeout=timeout)
            return

        resolved_job_id = job_result.job_id if job_result else job_id
        resolved_started = job_result.started_time if job_result else started_time

        if not resolved_job_id or not resolved_started:
            raise ValueError(
                "No job reference provided. Pass either a job_result or explicit job_id and started_time."
            )

        kwargs: Dict[str, Any] = {}
        if self._platform == Platform.SMHP:
            kwargs["cluster_name"] = cast(SMHPRuntimeManager, self.infra).cluster_name
            kwargs["namespace"] = cast(SMHPRuntimeManager, self.infra).namespace

        monitor = CloudWatchLogMonitor(
            job_id=resolved_job_id,
            platform=self._platform,
            started_time=int(resolved_started.timestamp() * 1000),
            region=self.region,
            **kwargs,
        )
        monitor.show_logs(limit=limit, start_from_head=start_from_head, end_time=end_time)

    @_telemetry_emitter(Feature.TRAINING, "trace_batch")
    def trace_batch(
        self,
        training_result: TrainingResult,
        step: int,
        output_path: str | None = None,
        cache_dir: str = DEFAULT_BATCH_TRACE_CACHE_DIR,
    ) -> Path | None:
        """Identify which input data lines were used in a specific training step.

        Requires that the training job was launched with
        ``enable_batch_sample_tracing=True``.

        Args:
            training_result: Result from a completed training job.
            step: Training step number to investigate.
            output_path: Output file for matched lines
                (default: ``step_<N>_samples.jsonl``).
            cache_dir: Directory for caching downloaded files and fingerprint
                indices (default: ``~/.nova-forge/batch_trace_cache/``).

        Returns:
            Path to the output file containing matched lines, or ``None`` if no
            matches were found.

        Raises:
            ValueError: If training data path or output path is not available.
            BatchTraceError: If batch tracing encounters an unrecoverable error.
        """
        if not self.training_data_s3_path:
            msg = "training_data_s3_path is required for batch tracing"
            raise ValueError(msg)

        output_s3_path = training_result.model_artifacts.output_s3_path
        if not output_s3_path:
            msg = "training_result.model_artifacts.output_s3_path is required for batch tracing"
            raise ValueError(msg)

        if not self._enable_batch_sample_tracing:
            logger.warning(
                "Batch sample tracing was not enabled for this trainer. "
                "If the training job was not launched with enable_batch_sample_tracing=True, "
                "no batch hash logs will be available."
            )

        log_dir = f"{output_s3_path.rstrip('/')}/{training_result.job_id}/{BATCH_TRACE_LOG_SUBDIR}/"

        if self._s3_client is None:
            self._s3_client = boto3.client("s3", region_name=self.region)

        return batch_trace_run(
            data_path=self.training_data_s3_path,
            log_dir=log_dir,
            step=step,
            output_path=output_path,
            s3_client=self._s3_client,
            cache_dir=cache_dir,
        )

    @_telemetry_emitter(Feature.TRAINING, "generate_training_metrics_csv")
    def generate_training_metrics_csv(
        self,
        job_result: Optional[SMHPTrainingResult] = None,
        job_id: Optional[str] = None,
        started_time=None,
        end_time=None,
        output_s3_path: Optional[str] = None,
    ) -> Optional[str]:
        """Generate step-wise training metrics CSV for an SMHP SFT job.

        Fetches CloudWatch logs for the specified job, extracts step-level
        training metrics (step number, epoch number, training loss), and
        uploads a ``step_wise_training_metrics.csv`` to the job's output S3 path.

        Args:
            job_result: An SMHPTrainingResult from a completed training job.
            job_id: The SMHP training job ID (used if job_result isn't provided).
            started_time: Job start time for log filtering. Accepts a datetime
                object or an ISO date str (e.g. "2025-05-26"). Defaults to 7 days.
            end_time: Optional end time to bound the log search. If not provided,
                searches up to the current time.
            output_s3_path: S3 URI for output. Defaults to ForgeTrainer's output path.

        Returns:
            S3 URI of the uploaded CSV, or None (no logs/metrics found).

        Raises:
            ValueError: If platform is not SMHP, method is not SFT_LORA/SFT_FULL,
                or required parameters are missing.
        """
        # Validate platform
        if self._platform != Platform.SMHP:
            raise ValueError(
                "generate_training_metrics_csv is only supported for SMHP platform. "
                f"Current platform: {self._platform.value}"
            )

        # Validate method
        if self.method not in (TrainingMethod.SFT_LORA, TrainingMethod.SFT_FULL):
            raise ValueError(
                "generate_training_metrics_csv is only supported for SFT training methods "
                f"(SFT_LORA, SFT_FULL). Current method: {self.method.value}"
            )

        # Extract parameters from job_result or use standalone params
        resolved_output_s3_path: Optional[str] = None
        resolved_job_id: Optional[str] = None
        if job_result is not None:
            resolved_job_id = job_result.job_id
            resolved_started_time = job_result.started_time
            resolved_cluster_name = job_result.cluster_name
            resolved_namespace = job_result.namespace
            resolved_output_s3_path = output_s3_path or job_result.model_artifacts.output_s3_path
        else:
            resolved_job_id = job_id
            resolved_started_time = started_time
            resolved_cluster_name = cast(SMHPRuntimeManager, self.infra).cluster_name
            resolved_namespace = cast(SMHPRuntimeManager, self.infra).namespace
            resolved_output_s3_path = output_s3_path or self.output_s3_path

        if not resolved_job_id:
            raise ValueError("No job_id provided. Pass either a job_result object or a job_id.")

        if not resolved_output_s3_path:
            raise ValueError(
                "output_s3_path is required but was not provided and could not be "
                "extracted from job_result."
            )

        # Check job status and emit warnings
        if job_result is not None:
            try:
                job_status, raw_status = job_result.get_job_status()
                if job_status == JobStatus.IN_PROGRESS:
                    logger.warning(
                        "Job '%s' is still in progress (status: %s). "
                        "Training metrics may be incomplete.",
                        resolved_job_id,
                        raw_status,
                    )
                elif job_status == JobStatus.FAILED:
                    logger.warning(
                        "Job '%s' has failed (status: %s). Training metrics may be partial.",
                        resolved_job_id,
                        raw_status,
                    )
            except Exception:
                logger.warning(
                    "Could not determine job status for '%s'. "
                    "Proceeding with best-effort log parsing.",
                    resolved_job_id,
                )

        # Fetch logs from CloudWatch
        resolved_started_time_dt = _parse_user_time(resolved_started_time)
        started_time_ms = int(resolved_started_time_dt.timestamp() * 1000)

        end_time_ms = None
        if end_time is not None:
            resolved_end_time_dt = _parse_user_time(end_time)
            end_time_ms = int(resolved_end_time_dt.timestamp() * 1000)

        # Create a monitor object and retrieve relevant job logs
        logger.info(
            "Fetching CloudWatch logs for job '%s' - this can take a few minutes.",
            resolved_job_id,
        )
        monitor = CloudWatchLogMonitor(
            job_id=resolved_job_id,
            platform=Platform.SMHP,
            started_time=started_time_ms,
            region=self.region,
            cluster_name=resolved_cluster_name,
            namespace=resolved_namespace,
        )
        log_events = monitor.get_logs(end_time=end_time_ms)
        logger.info("Retrieved %d log events.", len(log_events) if log_events else 0)

        logger.info("Parsing training metrics and generating CSV. ")
        return _build_and_upload_training_metrics_csv(
            job_id=resolved_job_id,
            log_events=log_events,
            output_s3_path=resolved_output_s3_path,
            training_method=self.method,
            region=self.region,
        )
