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
Main entrypoint for customizing and training Nova models.

This module provides the NovaModelCustomizer class which orchestrates the training process.
"""

import warnings
from datetime import datetime
from typing import Any, Dict, List, Optional, cast

import boto3

from amzn_nova_forge.core.constants import (
    REGION_TO_ESCROW_ACCOUNT_MAPPING,
    SUPPORTED_DATAMIXING_METHODS,
)
from amzn_nova_forge.core.enums import (
    DeploymentMode,
    DeployPlatform,
    EvaluationTask,
    Model,
    Platform,
    TrainingMethod,
)
from amzn_nova_forge.core.job_cache import JobCachingConfig
from amzn_nova_forge.core.result import (
    EvaluationResult,
    TrainingResult,
)
from amzn_nova_forge.core.result.inference_result import InferenceResult
from amzn_nova_forge.core.types import (
    DeploymentResult,
    EndpointInfo,
    ForgeConfig,
    RecipeConfig,
    ValidationConfig,
)
from amzn_nova_forge.manager.runtime_manager import (
    BedrockRuntimeManager,
    RuntimeManager,
    SMHPRuntimeManager,
    SMTJRuntimeManager,
    SMTJServerlessRuntimeManager,
)
from amzn_nova_forge.model.model_config import ModelDeployResult
from amzn_nova_forge.model.nova_model_customizer_util import (
    load_existing_result,
    persist_result,
    resolve_model_checkpoint_path,
    set_output_s3_path,
)
from amzn_nova_forge.monitor.log_monitor import CloudWatchLogMonitor
from amzn_nova_forge.monitor.mlflow_monitor import MLflowMonitor
from amzn_nova_forge.rft_multiturn import RFTMultiturnInfrastructure
from amzn_nova_forge.telemetry.constants import (
    UNKNOWN,
    Feature,
)
from amzn_nova_forge.telemetry.telemetry_logging import (
    _telemetry_emitter,
)
from amzn_nova_forge.util.checkpoint_util import extract_checkpoint_path_from_job_output
from amzn_nova_forge.util.data_mixing import DataMixing
from amzn_nova_forge.util.data_utils import is_multimodal_data
from amzn_nova_forge.util.logging import logger
from amzn_nova_forge.util.recipe import load_recipe_templates
from amzn_nova_forge.validation.endpoint_validator import (
    SageMakerEndpointEnvironment,
    is_sagemaker_arn,
)


def _resolve_deploy_platform(
    endpoint_arn: Optional[str], endpoint_info: Optional[EndpointInfo]
) -> Optional[DeployPlatform]:
    """Determine the deployment platform from an endpoint ARN or stored endpoint info.

    Returns DeployPlatform.SAGEMAKER when the ARN matches a SageMaker endpoint,
    DeployPlatform.BEDROCK_PT for other ARNs, the stored platform from endpoint_info,
    or None if neither is provided.
    """
    if endpoint_arn is not None:
        if is_sagemaker_arn(endpoint_arn):
            return DeployPlatform.SAGEMAKER
        return DeployPlatform.BEDROCK_PT
    elif endpoint_info is not None:
        return endpoint_info.platform
    return None


def _invoke_inference_extra_info(self, *args, **kwargs):
    info = {}
    if self.model is not None:
        info["model"] = self.model.value
    platform = _resolve_deploy_platform(kwargs.get("endpoint_arn"), self.endpoint_info)
    info["platform"] = platform if platform else UNKNOWN
    return info


class NovaModelCustomizer:
    def __init__(
        self,
        model: Model,
        method: TrainingMethod,
        infra: RuntimeManager,
        data_s3_path: Optional[str] = None,
        output_s3_path: Optional[str] = None,
        model_path: Optional[str] = None,
        validation_config: Optional[Dict[str, bool]] = None,
        generated_recipe_dir: Optional[str] = None,
        mlflow_monitor: Optional[MLflowMonitor] = None,
        deployment_mode: DeploymentMode = DeploymentMode.FAIL_IF_EXISTS,
        data_mixing_enabled: bool = False,
        image_uri: Optional[str] = None,
        enable_job_caching: bool = False,
        is_multimodal: Optional[bool] = None,
        hub_content_version: Optional[str] = None,
    ):
        """
        Initializes a NovaModelCustomizer instance.

        Args:
            model: The Nova model to be trained (e.g., NOVA_MICRO)
            method: The fine-tuning method (e.g., SFT_LORA, DPO)
            infra: Runtime infrastructure manager (e.g., SMTJRuntimeManager)
            data_s3_path: S3 path to the training dataset
            output_s3_path: Optional S3 path for output artifacts. If not provided, will be auto-generated
            model_path: Optional S3 path for model path
            validation_config: Optional dict to control validation. Keys: 'iam' (bool), 'infra' (bool), 'recipe' (bool).
                             Defaults to {'iam': True, 'infra': True, 'recipe': True}
            generated_recipe_dir: Optional path to save generated recipe YAMLs and persist job results.
                                If None, no result persistence occurs.
            mlflow_monitor: Optional MLflowMonitor instance for experiment tracking
            deployment_mode: Behavior when deploying to existing endpoint name. Options:
                           FAIL_IF_EXISTS (default), UPDATE_IF_EXISTS
            data_mixing_enabled: Enable data mixing. Default is False.
            image_uri: Optional custom ECR image URI to override the default training image.
                      Must be in format: <account>.dkr.ecr.<region>.amazonaws.com/<repository>:<tag>
            is_multimodal: Optional bool to explicitly set multimodal mode. If None (default),
                          auto-detects from data when applicable.
            enable_job_caching: Whether to enable job result caching. When enabled, completed
                              job results are cached to job_cache_dir (default: .cached-nova-jobs/)
                              and reused for identical job configurations.
            hub_content_version: Optional version of the hub content to retrieve from SageMaker Hub

        Raises:
            ValueError: If region is unsupported or model is invalid
        """
        self.job_id: Optional[str] = None  # This will be set after train/eval method invoked
        self.job_started_time: Optional[datetime] = (
            None  # This will be set after train/eval method invoked
        )
        self.cloud_watch_log_monitor: Optional[CloudWatchLogMonitor] = (
            None  # This will be set after get_logs method invoked
        )

        region = boto3.session.Session().region_name or "us-east-1"
        if region not in REGION_TO_ESCROW_ACCOUNT_MAPPING:
            raise ValueError(
                f"Region '{region}' is not supported for Nova training. "
                f"Supported regions are: {list(REGION_TO_ESCROW_ACCOUNT_MAPPING.keys())}"
            )

        self.region = region
        self._model = model
        self._image_uri = image_uri
        self._method = method
        self.infra = infra
        self._data_s3_path = data_s3_path
        self.model_path = model_path
        self.hub_content_version = hub_content_version
        self.validation_config = (
            ValidationConfig(**validation_config) if validation_config else None
        )
        self.deployment_mode = deployment_mode

        if isinstance(self.infra, SMTJRuntimeManager):
            self._platform = Platform.SMTJ
        elif isinstance(self.infra, SMTJServerlessRuntimeManager):
            self._platform = Platform.SMTJServerless
        elif isinstance(self.infra, BedrockRuntimeManager):
            self._platform = Platform.BEDROCK
        else:
            self._platform = Platform.SMHP

        # Warn if user passes model_path for Bedrock (should use base_model_identifier instead)
        if self._platform == Platform.BEDROCK and model_path is not None:
            logger.warning(
                "model_path is not used for Bedrock platform. "
                "To specify a base model, pass base_model_identifier to BedrockRuntimeManager constructor instead."
            )

        # For SMTJServerless, model_path must be a model package ARN for iterative training
        if (
            self._platform == Platform.SMTJServerless
            and model_path is not None
            and not is_sagemaker_arn(model_path)
        ):
            raise ValueError(
                f"For SMTJServerless, model_path must be a SageMaker model package ARN, "
                f"got: '{model_path}'. "
            )

        # Warn if user passes mlflow_monitor for Bedrock (not supported)
        if self._platform == Platform.BEDROCK and mlflow_monitor is not None:
            logger.warning("MLflow monitoring is not supported on the Bedrock platform.")

        self.instance_type = self.infra.instance_type

        self.output_s3_path = set_output_s3_path(
            region=self.region,
            output_s3_path=output_s3_path,
            kms_key_id=self.infra.kms_key_id,
        )

        self.generated_recipe_dir = generated_recipe_dir
        self.mlflow_monitor = mlflow_monitor

        # Initialize data mixing configuration
        self._data_mixing_enabled = data_mixing_enabled
        self.data_mixing = None

        # Auto-detect multimodal with optional override — only relevant when data mixing is enabled
        # Store user's explicit intent separately so data_s3_path setter can respect it.
        self._user_is_multimodal = is_multimodal  # None = auto-detect, True/False = explicit
        if not data_mixing_enabled:
            if is_multimodal is not None:
                logger.warning("is_multimodal is ignored because data_mixing_enabled=False.")
            _resolved_multimodal = False
        elif is_multimodal is not None:
            _resolved_multimodal = is_multimodal
        elif data_s3_path:
            _resolved_multimodal = is_multimodal_data(data_s3_path)
            if _resolved_multimodal:
                logger.info(
                    "Multimodal data detected. Using multimodal datamix recipes. "
                    "To skip auto-detection, pass is_multimodal=False."
                )
        else:
            _resolved_multimodal = False
        self._is_multimodal = _resolved_multimodal

        if data_mixing_enabled:
            self.data_mixing = DataMixing()
            self._init_data_mixing(self.model, self.method, self.platform)

        self.endpoint_info: Optional[EndpointInfo] = None

        # Deploy-decoupling state
        self.last_model_publish: Optional[ModelDeployResult] = None
        # Set of (platform, model_arn, escrow_path) tuples for dedup
        self._published_models: set = set()

        # Job caching configuration
        self.enable_job_caching = enable_job_caching
        self.job_cache_dir = ".cached-nova-jobs"
        self._job_caching_config = JobCachingConfig()

    @property
    def data_s3_path(self) -> Optional[str]:
        """Get the data S3 path."""
        return self._data_s3_path

    @data_s3_path.setter
    def data_s3_path(self, value: Optional[str]) -> None:
        """Set the data S3 path and re-run multimodal auto-detection if applicable.

        If the user originally passed is_multimodal=None (auto-detect), changing
        data_s3_path will re-scan the new path to keep is_multimodal consistent.
        If the user passed an explicit True/False, that value is preserved.
        Recipe templates are always reloaded when data_mixing_enabled=True so the
        correct text_with_datamix or mm_with_datamix recipe is selected.
        """
        self._data_s3_path = value
        if self.data_mixing_enabled:
            if self._user_is_multimodal is None:
                # Re-run auto-detection against the new path
                if value:
                    self._is_multimodal = is_multimodal_data(value)
                    if self._is_multimodal:
                        logger.info(
                            "Multimodal data detected. Using multimodal datamix recipes. "
                            "To skip auto-detection, pass is_multimodal=False."
                        )
                else:
                    self._is_multimodal = False
            # Always reload recipe templates so the correct datamix recipe is selected
            self._init_data_mixing(model=self.model, method=self.method, platform=self.platform)
            logger.info("data_s3_path changed. Datamixing configs set to default.")

    @property
    def is_multimodal(self) -> bool:
        """Get the is_multimodal flag."""
        return self._is_multimodal

    @is_multimodal.setter
    def is_multimodal(self, value: Optional[bool]) -> None:
        """Set the is_multimodal flag.

        Args:
            value: Explicit bool to force multimodal on/off. Pass None to re-run
                   auto-detection from the current data_s3_path (requires data_mixing_enabled=True).
        """
        self._user_is_multimodal = value  # track intent for data_s3_path setter
        if not self.data_mixing_enabled:
            logger.warning("is_multimodal is ignored because data_mixing_enabled=False.")
            self._is_multimodal = False
            return

        if value is not None:
            self._is_multimodal = value
        elif self._data_s3_path:
            self._is_multimodal = is_multimodal_data(self._data_s3_path)
            if self._is_multimodal:
                logger.info(
                    "Multimodal data detected. Using multimodal datamix recipes. "
                    "To skip auto-detection, pass is_multimodal=False."
                )
        else:
            self._is_multimodal = False

        # Reload recipe templates so the correct datamix recipe is selected
        self._init_data_mixing(model=self.model, method=self.method, platform=self.platform)
        logger.info("is_multimodal changed. Datamixing configs set to default.")

    @property
    def data_mixing_enabled(self) -> bool:
        """Get whether data mixing is enabled."""
        return self._data_mixing_enabled

    @data_mixing_enabled.setter
    def data_mixing_enabled(self, value: bool) -> None:
        """Enable or disable data mixing.

        Enabling: initializes DataMixing instance, re-runs multimodal detection,
        and loads recipe templates.
        Disabling: tears down DataMixing instance and resets is_multimodal to False.
        """
        if value == self._data_mixing_enabled:
            return  # no-op if unchanged

        self._data_mixing_enabled = value

        if value:
            # Enabling data mixing — initialize DataMixing and resolve is_multimodal
            self.data_mixing = DataMixing()
            if self._user_is_multimodal is not None:
                self._is_multimodal = self._user_is_multimodal
            elif self._data_s3_path:
                self._is_multimodal = is_multimodal_data(self._data_s3_path)
                if self._is_multimodal:
                    logger.info(
                        "Multimodal data detected. Using multimodal datamix recipes. "
                        "To skip auto-detection, pass is_multimodal=False."
                    )
            else:
                self._is_multimodal = False
            self._init_data_mixing(model=self.model, method=self.method, platform=self.platform)
            logger.info("data_mixing_enabled set to True. Datamixing configs set to default.")
        else:
            # Disabling data mixing — tear down
            self.data_mixing = None
            self._is_multimodal = False
            logger.info("data_mixing_enabled set to False. Data mixing disabled.")

    @property
    def model(self) -> Model:
        """Get the model attribute."""
        return self._model

    @model.setter
    def model(self, value: Model) -> None:
        """Set the model attribute and reinitialize data mixing if enabled."""
        if self.data_mixing_enabled:
            self._init_data_mixing(model=value, method=self.method, platform=self.platform)
            logger.info(f"Model changed to {value.name}. Datamixing configs set to default.")
        self._model = value

    @property
    def method(self) -> TrainingMethod:
        """Get the method attribute."""
        return self._method

    @method.setter
    def method(self, value: TrainingMethod) -> None:
        """Set the method attribute and reinitialize data mixing if enabled."""
        if self.data_mixing_enabled:
            self._init_data_mixing(model=self.model, method=value, platform=self.platform)
            logger.info(f"Method changed to {value.name}. Datamixing configs set to default.")
        self._method = value

    @property
    def platform(self) -> Platform:
        """Get the platform attribute."""
        return self._platform

    @platform.setter
    def platform(self, value: Platform) -> None:
        """Set the platform attribute and reinitialize data mixing if enabled."""
        if self.data_mixing_enabled:
            self._init_data_mixing(model=self.model, method=self.method, platform=value)
            logger.info(f"Platform changed to {value.name}. Datamixing configs set to default.")
        self._platform = value

    def _init_data_mixing(self, model: Model, method: TrainingMethod, platform: Platform) -> None:
        """
        Initialize data mixing configuration.
        """
        if not self.data_mixing_enabled:
            return

        # Data mixing is only supported on HyperPod or SMTJServerless for certain training methods
        if (
            platform not in (Platform.SMHP, Platform.SMTJServerless)
            or method not in SUPPORTED_DATAMIXING_METHODS
        ):
            raise ValueError(
                f"Data mixing is only supported for {SUPPORTED_DATAMIXING_METHODS} training methods on SageMaker HyperPod or SMTJServerless. "
                "Change platform to SMHP or SMTJServerless or change to a supported training method to use data mixing."
            )

        # Load recipe metadata and templates for non-evaluation methods
        # Eval requires "type" to be passed to load recipes, therefore we load them in evaluate()
        (
            self.recipe_metadata,
            self.recipe_template,
            self.overrides_template,
            self.image_uri,
        ) = load_recipe_templates(
            model=model,
            method=method,
            platform=platform,
            region=self.region,
            data_mixing_enabled=self.data_mixing_enabled,
            instance_type=self.instance_type,
            eval_task=getattr(self, "eval_task", None),
            image_uri_override=self._image_uri,
            is_multimodal=self.is_multimodal,
            hub_content_version=self.hub_content_version,
        )

        # Load default configuration into DataMixing instance if enabled
        if self.data_mixing and self.overrides_template:
            self.data_mixing._load_defaults_from_template(self.overrides_template)

    def _build_forge_config(self) -> ForgeConfig:
        """Build a ForgeConfig from the facade's current instance state.

        Caching is always disabled — the facade manages its own caching layer
        and delegates to service classes with caching off to prevent double-caching.
        """
        return ForgeConfig(
            kms_key_id=self.infra.kms_key_id,
            output_s3_path=self.output_s3_path,
            generated_recipe_dir=self.generated_recipe_dir,
            validation_config=self.validation_config,
            image_uri=self._image_uri,
            mlflow_monitor=self.mlflow_monitor,
            enable_job_caching=False,
            job_cache_dir=self.job_cache_dir,
            job_caching_config=self._job_caching_config,
        )

    def get_data_mixing_config(self) -> Dict[str, Any]:
        """
        Get the current data mixing configuration.

        Returns:
            Dictionary containing the data mixing configuration
        """
        if not self.data_mixing:
            return {}
        return self.data_mixing.get_config()

    def set_data_mixing_config(self, config: Dict[str, Any]) -> None:
        """
        Set the data mixing configuration.

        Args:
            config: Dictionary containing the data mixing configuration.
                   Keys should include nova_*_percent fields and customer_data_percent.
                   Any nova_*_percent fields not specified will be set to 0.

        Raises:
            ValueError: If data mixing is not enabled or invalid configuration
        """
        if not self.data_mixing:
            raise ValueError(
                "Data mixing is not enabled for this customizer. Set data_mixing = True in 'NovaModelCustomizer' object."
            )

        self.data_mixing.set_config(config, normalize=True)  # type: ignore[arg-type]

    @_telemetry_emitter(
        Feature.TRAINING,
        "train",
        extra_info_fn=lambda self, *args, **kwargs: {
            "method": self.method,
            "model": self.model.value,
            "platform": self.platform,
            "dryRun": kwargs.get("dry_run", False),
            "hasValidationData": kwargs.get("validation_data_s3_path") is not None,
        },
    )
    def train(
        self,
        job_name: str,
        recipe_path: Optional[str] = None,
        overrides: Optional[Dict[str, Any]] = None,
        rft_lambda_arn: Optional[str] = None,
        rft_multiturn_infra: Optional[RFTMultiturnInfrastructure] = None,
        validation_data_s3_path: Optional[str] = None,
        val_check_interval: Optional[int] = None,
        dry_run: bool = False,
    ) -> TrainingResult | None:
        """
        Generates the recipe YAML, configures runtime, and launches a training job.

        Args:
            job_name: User-defined name for the training job
            recipe_path: Optional path for a YAML recipe file (both S3 and local paths are accepted)
            overrides: Optional dictionary of configuration overrides. Example:
                {
                    'max_epochs': 10,
                    'lr': 5e-6,
                    'warmup_steps': 20,
                    'loraplus_lr_ratio': 16.0,
                    'global_batch_size': 128,
                    'max_length': 16384
                }
            rft_lambda_arn: Optional Lambda ARN for RFT reward function (only used for RFT training methods).
                If passed, takes priority over rft_lambda_arn set on the RuntimeManager.
            rft_multiturn_infra: Optional RFT multiturn infrastructure, required for RFT_MULTITURN methods
            validation_data_s3_path: Optional validation S3 path, applicable for CPT and SFT on SMTJ/SMTJServerless/SMHP, or any method on Bedrock (but is still optional)
            val_check_interval: Optional positive integer controlling how often (in training steps) validation is run.
                Defaults to 2500 if omitted. Only used when validation_data_s3_path is provided.
            dry_run: Actually starts a job if False, otherwise just performs validation. Default is False.

        Returns:
            TrainingResult: Metadata object containing job ID, method, start time, and model artifacts
            or None if dry_run is enabled

        Raises:
            Exception: If job execution fails
        """
        warnings.warn(
            "NovaModelCustomizer.train() is deprecated and will be removed in a future version. "
            "Use ForgeTrainer.train() instead.",
            DeprecationWarning,
            stacklevel=2,
        )

        from amzn_nova_forge.trainer.forge_trainer import ForgeTrainer

        # Facade caching (backward compat — uses customizer-level hash)
        rft_lambda_arn = rft_lambda_arn or getattr(self.infra, "rft_lambda_arn", None)
        existing_result = load_existing_result(
            self,
            job_name,
            "training",
            recipe_path=recipe_path,
            overrides=overrides or {},
            rft_lambda_arn=rft_lambda_arn,
            rft_multiturn_infra=rft_multiturn_infra,
            validation_data_s3_path=validation_data_s3_path,
        )
        if existing_result:
            return cast(TrainingResult, existing_result)

        # Translation layer: map facade state → ForgeTrainer constructor
        trainer = ForgeTrainer(
            model=self.model,
            method=self.method,
            infra=self.infra,
            training_data_s3_path=self.data_s3_path,
            model_s3_path=self.model_path,
            data_mixing_enabled=self.data_mixing_enabled,
            holdout_data_s3_path=validation_data_s3_path,
            val_check_interval=val_check_interval,
            config=self._build_forge_config(),
            region=self.region,
            is_multimodal=self.is_multimodal,
            hub_content_version=self.hub_content_version,
        )

        # Forward user-configured data mixing (ForgeTrainer creates a fresh instance with defaults)
        if self.data_mixing is not None:
            trainer.data_mixing = self.data_mixing

        # Delegate
        training_result = trainer.train(
            job_name=job_name,
            recipe_path=recipe_path,
            overrides=overrides,  # type: ignore[arg-type]
            rft_lambda_arn=rft_lambda_arn,
            dry_run=dry_run,
            rft_multiturn_infra=rft_multiturn_infra,
        )

        if training_result is None:  # dry_run
            return None

        # Update facade state from result
        self.job_id = training_result.job_id
        self.job_started_time = training_result.started_time

        # Facade caching persist
        persist_result(
            self,
            training_result,
            job_name,
            "training",
            recipe_path=recipe_path,
            overrides=overrides or {},
            rft_lambda_arn=rft_lambda_arn,
            rft_multiturn_infra=rft_multiturn_infra,
            validation_data_s3_path=validation_data_s3_path,
        )

        return training_result

    @_telemetry_emitter(
        Feature.TRAINING,
        "customizer.get_config",
        extra_info_fn=lambda self, *args, **kwargs: {
            "method": self.method,
            "model": self.model.value,
            "platform": self.platform,
        },
    )
    def get_config(
        self,
        overrides: Optional[Dict[str, Any]] = None,
    ) -> RecipeConfig:
        """Return the overridable training configuration.

        .. deprecated::
            Use ``ForgeTrainer.get_config()`` instead.
        """
        warnings.warn(
            "NovaModelCustomizer.get_config() is deprecated and will be removed in a future version. "
            "Use ForgeTrainer.get_config() instead.",
            DeprecationWarning,
            stacklevel=2,
        )

        from amzn_nova_forge.trainer.forge_trainer import ForgeTrainer

        trainer = ForgeTrainer(
            model=self.model,
            method=self.method,
            infra=self.infra,
            training_data_s3_path=self.data_s3_path,
            model_s3_path=self.model_path,
            data_mixing_enabled=self.data_mixing_enabled,
            config=self._build_forge_config(),
            region=self.region,
            is_multimodal=self.is_multimodal,
            hub_content_version=self.hub_content_version,
        )

        if self.data_mixing is not None:
            trainer.data_mixing = self.data_mixing

        return trainer.get_config(overrides=overrides)  # type: ignore[arg-type]

    @_telemetry_emitter(
        Feature.EVAL,
        "evaluate",
        extra_info_fn=lambda self, *args, **kwargs: {
            "method": self.method,
            "model": self.model.value,
            "platform": self.platform,
            "dryRun": kwargs.get("dry_run", False),
        },
    )
    def evaluate(
        self,
        job_name: str,
        eval_task: EvaluationTask,
        model_path: Optional[str] = None,
        subtask: Optional[str] = None,
        data_s3_path: Optional[str] = None,
        recipe_path: Optional[str] = None,
        overrides: Optional[Dict[str, Any]] = None,
        processor: Optional[Dict[str, Any]] = None,
        rl_env: Optional[Dict[str, Any]] = None,
        rft_multiturn_infra: Optional["RFTMultiturnInfrastructure"] = None,
        dry_run: bool = False,
        job_result: Optional[TrainingResult] = None,
    ) -> EvaluationResult | None:
        """
        Generates the recipe YAML, configures runtime, and launches an evaluation job.

        :param job_name: User-defined name for the evaluation job
        :param eval_task: The evaluation task to be performed, e.g. mmlu
        :param model_path: Optional S3 path for model path
        :param subtask: Optional subtask for evaluation
        :param data_s3_path: Optional S3 URI for the dataset
        :param recipe_path: Optional path for a YAML recipe file (both S3 and local paths are accepted)
        :param overrides: Optional dictionary of configuration overrides for eval job (Inference config). Example:
                {
                    'max_new_tokens': 2048,
                    'top_k': -1,
                    'top_p': 1.0,
                    'temperature': 0,
                    'top_logprobs': 10
                }
        :param processor: Optional, only needed for Bring Your Own Metrics Configuration. Example:
                {
                    'lambda_arn': 'arn:aws:lambda:<region>:<account_id>:function:<function-name>',
                    'preprocessing': { # Optional, default to True if not provided
                        'enabled': True
                    },
                    'postprocessing': { # Optional, default to True if not provided
                        'enabled': True
                    },
                    # Built-in aggregation function (valid options: min, max, average, sum), default to average
                    'aggregation': 'average'
                }
        :param rl_env: Optional, only needed for Bring your own Reinforcement learning environment (RFT Eval) config.
                Example:
                {
                    'reward_lambda_arn': 'arn:aws:lambda:<region>:<account_id>:function:<reward-function-name>'
                }
        :param rft_multiturn_infra: Optional RFT multiturn infrastructure, required for RFT_MULTITURN_EVAL
        :param dry_run: dry_run: Actually starts a job if False, otherwise just performs validation. Default is False.
        :param job_result: Optional TrainingResult object to extract checkpoint path from.
                          If provided and model_path is None, will automatically extract
                          the checkpoint path from the training job's output.
        :return: EvaluationResult: Metadata object containing job ID, start time, and evaluation output path
                 or None if dry_run is enabled
        """
        warnings.warn(
            "NovaModelCustomizer.evaluate() is deprecated and will be removed in a future version. "
            "Use ForgeEvaluator.evaluate() instead.",
            DeprecationWarning,
            stacklevel=2,
        )

        from amzn_nova_forge.evaluator.forge_evaluator import (
            EvalTaskConfig,
            ForgeEvaluator,
        )

        # Facade caching (backward compat)
        existing_result = load_existing_result(
            self,
            job_name,
            "evaluation",
            eval_task=eval_task,
            model_path=model_path,
            subtask=subtask,
            data_s3_path=data_s3_path,
            recipe_path=recipe_path,
            overrides=overrides or {},
            processor=processor,
            rl_env=rl_env,
            job_result=job_result,
        )
        if existing_result:
            return cast(EvaluationResult, existing_result)

        # Translation layer: resolve model_path from job_result / facade state
        resolved_model_path = resolve_model_checkpoint_path(
            model_path=model_path,
            job_result=job_result,
            customizer_job_id=self.job_id,
            customizer_output_s3_path=self.output_s3_path,
            customizer_model_path=self.model_path,
        )

        # Translation layer: flat params → EvalTaskConfig
        task_config = EvalTaskConfig(
            subtask=subtask,
            processor=processor,
            rl_env=rl_env,
            override_data_s3_path=data_s3_path,
        )

        evaluator = ForgeEvaluator(
            model=self.model,
            infra=self.infra,
            data_s3_path=self.data_s3_path,
            config=self._build_forge_config(),
            region=self.region,
            hub_content_version=self.hub_content_version,
        )

        evaluation_result = evaluator.evaluate(
            job_name=job_name,
            eval_task=eval_task,
            model_path=resolved_model_path,
            task_config=task_config,
            recipe_path=recipe_path,
            overrides=overrides,
            dry_run=dry_run,
            rft_multiturn_infra=rft_multiturn_infra,
        )

        if evaluation_result is None:  # dry_run
            return None

        # Update facade state
        self.job_id = evaluation_result.job_id
        self.job_started_time = evaluation_result.started_time

        # Facade caching persist
        persist_result(
            self,
            evaluation_result,
            job_name,
            "evaluation",
            eval_task=eval_task,
            model_path=model_path,
            subtask=subtask,
            data_s3_path=data_s3_path,
            recipe_path=recipe_path,
            overrides=overrides or {},
            processor=processor,
            rl_env=rl_env,
            job_result=job_result,
        )

        return evaluation_result

    def _build_deployer(self):
        """Create a ForgeDeployer configured from this customizer's state."""
        from amzn_nova_forge.deployer.forge_deployer import ForgeDeployer

        deployer = ForgeDeployer(
            region=self.region,
            model=self.model,
            deployment_mode=self.deployment_mode,
            config=ForgeConfig(
                kms_key_id=self.infra.kms_key_id if self.infra else None,
                validation_config=self.validation_config,
            ),
            method=self.method,
        )
        # Share session cache state
        deployer._published_models = self._published_models
        deployer.last_model_publish = self.last_model_publish
        return deployer

    @_telemetry_emitter(
        Feature.DEPLOY,
        "deploy",
        extra_info_fn=lambda self, *args, **kwargs: {
            "model": self.model.value,
            "platform": kwargs.get("deploy_platform", DeployPlatform.BEDROCK_OD),
        },
    )
    def deploy(
        self,
        model_artifact_path: Optional[str] = None,
        deploy_platform: DeployPlatform = DeployPlatform.BEDROCK_OD,
        unit_count: int = 1,
        endpoint_name: Optional[str] = None,
        job_result: Optional[TrainingResult] = None,
        execution_role_name: Optional[str] = None,
        sagemaker_instance_type: Optional[str] = "ml.p5.48xlarge",
        sagemaker_environment: Optional[SageMakerEndpointEnvironment] = None,
        skip_model_reuse: bool = False,
    ) -> DeploymentResult:
        """
        Deployment method supporting both Bedrock and SageMaker platforms.

        Args:
            model_artifact_path: S3 path to the trained model checkpoint. If not provided, will attempt to extract from job_result or the `job_id` field of the Customizer.
            deploy_platform: Platform to deploy to (Bedrock On-Demand, Provisioned Throughput, or SageMaker)
            unit_count: Used in Bedrock Provisioned Throughput number of PT to purchase or SageMaker number of initial instances
            endpoint_name: Name of the deployed model's endpoint (auto-generated if not provided)
            job_result: Training job result object to use for extracting checkpoint path and validating job completion. Also used to retrieve job_id if it's not provided.
            execution_role_name:  Optional IAM execution role name for Bedrock or SageMaker, defaults to BedrockDeployModelExecutionRole or SageMakerExecutionRoleName. If this role does not exist, it will be created.
            sagemaker_instance_type: Optional EC2 instance type for SageMaker deployment, defaults to ml.p5.48xlarge
            sagemaker_environment: Optional SageMaker endpoint environment config
            skip_model_reuse: If True, always create a new model (skip tag-based discovery of existing models)

        Returns:
            DeploymentResult with endpoint information
        """

        warnings.warn(
            "NovaModelCustomizer.deploy() is deprecated and will be removed in a future version. "
            "Use ForgeDeployer.deploy() instead.",
            DeprecationWarning,
            stacklevel=2,
        )

        # Translation layer: resolve checkpoint from job_result / facade state
        resolved_model_artifact_path = resolve_model_checkpoint_path(
            model_path=model_artifact_path,
            job_result=job_result,
            customizer_job_id=self.job_id,
            customizer_output_s3_path=self.output_s3_path,
            customizer_model_path=self.model_path,
            fail_on_error=True,
        )

        if resolved_model_artifact_path is None:
            raise ValueError(
                "Model artifact path could not be resolved. Provide a valid model_path or job_result"
            )

        deployer = self._build_deployer()

        # Delegate
        result = deployer.deploy(
            model_artifact_path=resolved_model_artifact_path,
            deploy_platform=deploy_platform,
            endpoint_name=endpoint_name,
            unit_count=unit_count,
            execution_role_name=execution_role_name,
            sagemaker_instance_type=sagemaker_instance_type,
            sagemaker_environment=sagemaker_environment,
            skip_model_reuse=skip_model_reuse,
        )

        # Backward compat: old Bedrock deploy stored output_s3_path here, not the checkpoint
        if deploy_platform in (DeployPlatform.BEDROCK_OD, DeployPlatform.BEDROCK_PT):
            result.endpoint.model_artifact_path = self.output_s3_path

        # Update facade state
        self.endpoint_info = result.endpoint
        self.job_started_time = result.created_at
        self.last_model_publish = deployer.last_model_publish
        self._published_models = deployer._published_models
        return result

    def find_published_model(
        self, platform: str, escrow_path: str, skip_model_reuse: bool = False
    ) -> Optional[str]:
        """Find a previously published model ARN for the given platform and escrow path.

        Delegates to ForgeDeployer.find_published_model().
        """
        warnings.warn(
            "NovaModelCustomizer.find_published_model() is deprecated and will be removed in a future version. "
            "Use ForgeDeployer.find_published_model() instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        deployer = self._build_deployer()
        return deployer.find_published_model(platform, escrow_path, skip_model_reuse)

    def deploy_to_sagemaker(
        self,
        instance_type: str,
        model_deploy_result: Optional[ModelDeployResult] = None,
        model_artifact_path: Optional[str] = None,
        unit_count: int = 1,
        endpoint_name: Optional[str] = None,
        sagemaker_environment: Optional[SageMakerEndpointEnvironment] = None,
        execution_role_name: Optional[str] = None,
        skip_model_reuse: bool = False,
    ) -> DeploymentResult:
        """Deploy a model to a SageMaker Inference endpoint.

        Delegates to ForgeDeployer._deploy_to_sagemaker() with model reuse support.
        """
        warnings.warn(
            "NovaModelCustomizer.deploy_to_sagemaker() is deprecated and will be removed in a future version. "
            "Use ForgeDeployer.deploy() with deploy_platform=DeployPlatform.SAGEMAKER instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        if model_deploy_result is not None and model_artifact_path is not None:
            raise ValueError(
                "Cannot provide both model_deploy_result and model_artifact_path. "
                "Use model_deploy_result to reuse an existing model, "
                "or model_artifact_path to create a new one."
            )

        # Resolve model_artifact_path from model_deploy_result if provided
        if model_artifact_path is None and model_deploy_result is not None:
            model_artifact_path = model_deploy_result.escrow_uri

        if not model_artifact_path:
            raise ValueError(
                "No model artifact path available. Provide model_artifact_path, "
                "or a model_deploy_result with a non-empty escrow_uri."
            )

        # Pre-populate session cache so the deployer reuses the existing model
        if model_deploy_result is not None:
            self._published_models.add(
                ("sagemaker", model_deploy_result.model_arn, model_artifact_path)
            )

        deployer = self._build_deployer()
        result = deployer._deploy_to_sagemaker(
            model_artifact_path=model_artifact_path,
            endpoint_name=endpoint_name,
            instance_type=instance_type,
            unit_count=unit_count,
            sagemaker_environment=sagemaker_environment,
            execution_role_name=execution_role_name,
            skip_model_reuse=skip_model_reuse,
        )

        # Sync state back
        self.last_model_publish = deployer.last_model_publish
        self._published_models = deployer._published_models
        self.endpoint_info = result.endpoint

        return result

    def create_custom_model(
        self,
        model_artifact_path: Optional[str] = None,
        job_result: Optional[TrainingResult] = None,
        endpoint_name: Optional[str] = None,
        execution_role_name: Optional[str] = None,
        tags: Optional[List[Dict[str, str]]] = None,
        skip_model_reuse: bool = False,
        custom_model_data_source: Optional[Dict[str, Any]] = None,
    ) -> ModelDeployResult:
        """Create a Bedrock custom model from S3 artifacts or a model package ARN.

        Delegates to ForgeDeployer.create_custom_model(). Handles job_result
        resolution at the facade level before delegating.
        """
        warnings.warn(
            "NovaModelCustomizer.create_custom_model() is deprecated and will be removed in a future version. "
            "Use ForgeDeployer.create_custom_model() instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        # Resolve model_artifact_path from job_result (facade-level concern).
        # Skip extraction when custom_model_data_source is already provided,
        # since it serves as a valid alternative to model_artifact_path.
        if (
            model_artifact_path is None
            and job_result is not None
            and custom_model_data_source is None
        ):
            model_artifact_path = extract_checkpoint_path_from_job_output(
                output_s3_path=job_result.model_artifacts.output_s3_path,
                job_result=job_result,
            )
            if model_artifact_path is None:
                raise ValueError(
                    f"Could not resolve checkpoint path from job result '{job_result.job_id}'. "
                    f"Provide model_artifact_path or custom_model_data_source explicitly."
                )

        if model_artifact_path is None and custom_model_data_source is None:
            raise ValueError(
                "Either model_artifact_path, job_result, or custom_model_data_source must be provided."
            )

        deployer = self._build_deployer()
        result = deployer.create_custom_model(
            model_artifact_path=model_artifact_path,
            endpoint_name=endpoint_name,
            execution_role_name=execution_role_name,
            tags=tags,
            skip_model_reuse=skip_model_reuse,
            custom_model_data_source=custom_model_data_source,
        )

        # Sync state back
        self.last_model_publish = deployer.last_model_publish
        self._published_models = deployer._published_models

        return result

    def deploy_to_bedrock(
        self,
        model_deploy_result: Optional[ModelDeployResult] = None,
        model_arn: Optional[str] = None,
        deploy_platform: DeployPlatform = DeployPlatform.BEDROCK_OD,
        pt_units: Optional[int] = None,
        endpoint_name: Optional[str] = None,
    ) -> DeploymentResult:
        """Deploy a published Bedrock custom model to an endpoint.

        Delegates to ForgeDeployer.deploy_to_bedrock().

        Resolution order for model ARN:
        1. model_deploy_result provided -> use its model_arn
        2. model_arn provided -> use directly
        3. self.last_model_publish -> auto-use
        4. None -> raise ValueError
        """
        warnings.warn(
            "NovaModelCustomizer.deploy_to_bedrock() is deprecated and will be removed in a future version. "
            "Use ForgeDeployer.deploy_to_bedrock() instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        deployer = self._build_deployer()
        result = deployer.deploy_to_bedrock(
            model_deploy_result=model_deploy_result,
            model_arn=model_arn,
            deploy_platform=deploy_platform,
            pt_units=pt_units,
            endpoint_name=endpoint_name,
        )

        # Sync state back
        self.last_model_publish = deployer.last_model_publish
        self.endpoint_info = result.endpoint
        self.job_started_time = result.created_at

        return result

    @_telemetry_emitter(
        Feature.DEPLOY,
        "predict",
        extra_info_fn=lambda self, *args, **kwargs: {
            "model": self.model.value,
            "platform": self.platform,
        },
    )
    def predict(self):
        warnings.warn(
            "NovaModelCustomizer.predict() is deprecated and will be removed in a future version. "
            "Use ForgeInference.invoke() instead.",
            DeprecationWarning,
            stacklevel=2,
        )

    @_telemetry_emitter(
        Feature.BATCH_INFERENCE,
        "batch_inference",
        extra_info_fn=lambda self, *args, **kwargs: {
            "method": self.method,
            "model": self.model.value,
            "platform": self.platform,
            "dryRun": kwargs.get("dry_run", False),
        },
    )
    def batch_inference(
        self,
        job_name: str,
        input_path: str,
        output_s3_path: str,
        model_path: Optional[str] = None,
        recipe_path: Optional[str] = None,
        overrides: Optional[Dict[str, Any]] = None,
        dry_run: bool = False,
        job_result: Optional[TrainingResult] = None,
    ) -> InferenceResult | None:
        """
        Launches a batch inference job on a trained model.

        :param job_name: Name for the batch inference job
        :param input_path: S3 path to input data for inference
        :param output_s3_path: S3 path for inference outputs
        :param model_path: Optional S3 path to the model
        :param recipe_path: Optional path for a YAML recipe file
        :param overrides: Optional configuration overrides for inference
        :param dry_run: Actually starts a job if False, otherwise just performs validation. Default is False.
        :param job_result: Optional TrainingResult object to extract checkpoint path from.
                          If provided and model_path is None, will automatically extract
                          the checkpoint path from the training job's output.
        :return: InferenceResult or None if dry_run is enabled
        """
        warnings.warn(
            "NovaModelCustomizer.batch_inference() is deprecated and will be removed in a future version. "
            "Use ForgeInference.invoke_batch() instead.",
            DeprecationWarning,
            stacklevel=2,
        )

        from amzn_nova_forge.inference.forge_inference import ForgeInference

        # Facade caching (backward compat)
        existing_result = load_existing_result(
            self,
            job_name,
            "inference",
            input_path=input_path,
            output_s3_path=output_s3_path,
            model_path=model_path,
            recipe_path=recipe_path,
            overrides=overrides or {},
            job_result=job_result,
        )
        if existing_result:
            return cast(InferenceResult, existing_result)

        # Translation layer: resolve model_path from job_result / facade state
        resolved_model_path = resolve_model_checkpoint_path(
            model_path=model_path,
            job_result=job_result,
            customizer_job_id=self.job_id,
            customizer_output_s3_path=self.output_s3_path,
            customizer_model_path=self.model_path,
        )

        inference = ForgeInference(
            region=self.region,
            model=self.model,
            infra=self.infra,
            config=self._build_forge_config(),
            method=self.method,
            hub_content_version=self.hub_content_version,
        )

        batch_inference_result = inference.invoke_batch(
            job_name=job_name,
            input_path=input_path,
            output_s3_path=output_s3_path,
            model_path=resolved_model_path,
            recipe_path=recipe_path,
            overrides=overrides,
            dry_run=dry_run,
        )

        if batch_inference_result is None:  # dry_run
            return None

        # Update facade state
        self.job_id = batch_inference_result.job_id
        self.job_started_time = batch_inference_result.started_time

        # Facade caching persist
        persist_result(
            self,
            batch_inference_result,
            job_name,
            "inference",
            input_path=input_path,
            output_s3_path=output_s3_path,
            model_path=model_path,
            recipe_path=recipe_path,
            overrides=overrides or {},
            job_result=job_result,
        )

        return batch_inference_result

    @_telemetry_emitter(
        Feature.MONITOR,
        "get_logs",
        extra_info_fn=lambda self, *args, **kwargs: {
            "platform": self.platform,
        },
    )
    def get_logs(
        self,
        limit: Optional[int] = None,
        start_from_head: bool = False,
        end_time: Optional[int] = None,
    ):
        warnings.warn(
            "NovaModelCustomizer.get_logs() is deprecated and will be removed in a future version. "
            "Use ForgeTrainer.get_logs(), ForgeEvaluator.get_logs(), or "
            "ForgeInference.get_logs() instead — those accept explicit "
            "job_result or job_id parameters and do not require a prior "
            "train()/evaluate() call.",
            DeprecationWarning,
            stacklevel=2,
        )
        if self.job_id and self.job_started_time:
            kwargs = {}
            if self.platform == Platform.SMHP:
                kwargs["cluster_name"] = cast(SMHPRuntimeManager, self.infra).cluster_name
                kwargs["namespace"] = cast(SMHPRuntimeManager, self.infra).namespace
            self.cloud_watch_log_monitor = self.cloud_watch_log_monitor or CloudWatchLogMonitor(
                job_id=self.job_id,
                platform=self.platform,
                started_time=int(self.job_started_time.timestamp() * 1000),
                region=self.region,
                **kwargs,
            )
            self.cloud_watch_log_monitor.show_logs(
                limit=limit, start_from_head=start_from_head, end_time=end_time
            )
        else:
            raise ValueError(
                "No job_id and job_started_time found. Call .train() or .evaluate() before calling .get_logs()."
            )

    @_telemetry_emitter(
        Feature.DEPLOY,
        "invoke_inference",
        extra_info_fn=_invoke_inference_extra_info,
    )
    def invoke_inference(self, request_body: Dict[str, Any], endpoint_arn: Optional[str] = None):
        """
        Invokes single inference against an endpoint
        :param request_body: Inference request body
        :param endpoint_arn: Optional endpoint ARN if user does not want to use previously deployed endpoint
        :return: Inference result. String for non-streaming response, generator for streaming response
        """
        warnings.warn(
            "NovaModelCustomizer.invoke_inference() is deprecated and will be removed in a future version. "
            "Use ForgeInference.invoke() instead.",
            DeprecationWarning,
            stacklevel=2,
        )

        from amzn_nova_forge.inference.forge_inference import ForgeInference

        # Translation layer: resolve endpoint from facade's implicit state
        if endpoint_arn is None and self.endpoint_info is None:
            raise ValueError(
                "endpoint_arn must be provided if no endpoint was previously deployed by Customizer"
            )
        elif endpoint_arn is not None:
            resolved_arn = endpoint_arn
        else:
            assert self.endpoint_info is not None  # guarded by the check above
            resolved_arn = self.endpoint_info.uri

        inference = ForgeInference(region=self.region)
        return inference.invoke(endpoint_arn=resolved_arn, request_body=request_body)

    @_telemetry_emitter(
        Feature.MONITOR,
        "monitor_metrics",
        extra_info_fn=lambda self, *args, **kwargs: {
            "model": self.model.value,
            "platform": self.platform,
        },
    )
    def monitor_metrics(self):
        warnings.warn(
            "NovaModelCustomizer.monitor_metrics() is deprecated and will be removed in a future version. "
            "This method was a no-op and has no replacement.",
            DeprecationWarning,
            stacklevel=2,
        )
