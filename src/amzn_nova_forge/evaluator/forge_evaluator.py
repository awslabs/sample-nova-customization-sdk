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
"""ForgeEvaluator — owns the evaluation workflow."""

from __future__ import annotations

import io
import os
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from tempfile import mkdtemp
from typing import TYPE_CHECKING, Any, Dict, Optional, cast

import boto3
import yaml

from amzn_nova_forge.core.constants import DEFAULT_REGION, get_inspect_lens_default_image_uri
from amzn_nova_forge.core.enums import EvaluationTask, Model, Platform, TrainingMethod
from amzn_nova_forge.core.job_cache import (
    build_cache_context,
    load_existing_result,
    persist_result,
)
from amzn_nova_forge.core.result import (
    EvaluationResult,
    SMHPEvaluationResult,
    SMTJEvaluationResult,
    SMTJTrainingResult,
    TrainingResult,
)
from amzn_nova_forge.core.runtime import RuntimeManager
from amzn_nova_forge.core.types import (
    ForgeConfig,
    JobConfig,
    validate_region,
)
from amzn_nova_forge.evaluator.inspect_lens_config import InspectLensConfig
from amzn_nova_forge.manager.runtime_manager import SMHPRuntimeManager, SMTJServerlessRuntimeManager
from amzn_nova_forge.model.nova_model_customizer_util import (
    requires_custom_eval_data,
    resolve_model_checkpoint_path,
    set_output_s3_path,
)
from amzn_nova_forge.monitor.log_monitor import CloudWatchLogMonitor
from amzn_nova_forge.recipe.recipe_builder import RecipeBuilder
from amzn_nova_forge.telemetry import Feature, _telemetry_emitter
from amzn_nova_forge.util.logging import logger
from amzn_nova_forge.util.platform_util import (
    detect_platform_from_path,
    validate_platform_compatibility,
)
from amzn_nova_forge.util.s3_utils import parse_s3_uri
from amzn_nova_forge.validation.inspect_lens_validator import validate_task_names_against_benchmarks
from amzn_nova_forge.validation.validator import validate_rft_lambda_name

if TYPE_CHECKING:
    from amzn_nova_forge.rft_multiturn import RFTMultiturnInfrastructure


@dataclass
class EvalTaskConfig:
    """Per-task configuration for an evaluation job.

    Attributes:
        subtask: Subtask identifier for benchmark evaluations.
        processor: Lambda processor config for BYOM evaluations.
        rl_env: RL environment config for HyperPod evaluations.
        override_data_s3_path: Override the default evaluation dataset path.
        evaluate_base_model: When True and used with RFT_MULTITURN_EVAL,
            evaluates both the base model and the fine-tuned model in a
            single pipeline for side-by-side comparison. Only applies to
            MTRL evaluation on SMTJServerless. Requires model_path to be set.
    """

    subtask: Optional[str] = None
    processor: Optional[Dict[str, Any]] = None
    rl_env: Optional[Dict[str, Any]] = None
    override_data_s3_path: Optional[str] = None
    evaluate_base_model: bool = False


class ForgeEvaluator:
    """Encapsulates the evaluation workflow for Nova model customization.

    Configuration is provided in the constructor; ``evaluate()`` accepts
    per-job parameters including the evaluation task.
    """

    def __init__(
        self,
        model: Model,
        infra: RuntimeManager,
        data_s3_path: Optional[str] = None,
        config: Optional[ForgeConfig] = None,
        region: Optional[str] = None,
        hub_content_version: Optional[str] = None,
    ) -> None:
        self.model = model
        self.infra = infra
        self.data_s3_path = data_s3_path
        self._config = config or ForgeConfig()
        self.hub_content_version = hub_content_version

        self.region = region or boto3.session.Session().region_name or DEFAULT_REGION
        validate_region(self.region)

        self._platform = infra.platform
        self._is_multimodal = False  # Default; evaluation multimodal support can be extended later

        self.output_s3_path = set_output_s3_path(
            region=self.region,
            output_s3_path=self._config.output_s3_path,
            kms_key_id=self.infra.kms_key_id,
        )

        # Job caching context
        self._cache_context = build_cache_context(
            self._config,
            model=model,
            method=TrainingMethod.EVALUATION,
            data_s3_path=data_s3_path,
            output_s3_path=self.output_s3_path,
            instance_type=infra.instance_type,
            instance_count=infra.instance_count,
        )

    @_telemetry_emitter(
        Feature.EVAL,
        "evaluate",
        extra_info_fn=lambda self, *args, **kwargs: {
            "model": self.model.value,
            "platform": self._platform,
            "dryRun": kwargs.get("dry_run", False),
        },
    )
    def evaluate(
        self,
        job_name: str,
        eval_task: EvaluationTask,
        model_path: Optional[str] = None,
        task_config: Optional[EvalTaskConfig] = None,
        recipe_path: Optional[str] = None,
        overrides: Optional[Dict[str, Any]] = None,
        dry_run: bool = False,
        job_result: Optional[TrainingResult] = None,
        rft_multiturn_infra: Optional[RFTMultiturnInfrastructure] = None,
        inspect_lens_config: Optional[InspectLensConfig] = None,
    ) -> Optional[EvaluationResult]:
        """Launch an evaluation job.

        Args:
            job_name: User-defined name for the evaluation job.
            eval_task: The evaluation task to perform (e.g. MMLU).
            model_path: Optional S3 path to the model checkpoint.
            task_config: Optional per-task configuration.
            recipe_path: Optional path to a YAML recipe file.
            overrides: Optional dictionary of configuration overrides.
            dry_run: If True, only validate — do not start a job.
            job_result: Optional TrainingResult to extract checkpoint path from.
            rft_multiturn_infra: Optional RFT multiturn infrastructure (passed
                through to RecipeBuilder for multi-turn RFT evaluation).
            inspect_lens_config: Required when eval_task is INSPECT_LENS.
                Configures benchmarks, inference provider, decoding, and output.

        Returns:
            EvaluationResult on success, None if dry_run is True.
        """
        if eval_task == EvaluationTask.INSPECT_LENS:
            if overrides:
                # Warn only if overrides contains keys that are not InspectLens decoding params
                _INSPECT_LENS_VALID_OVERRIDES = {
                    "temperature",
                    "top_p",
                    "top_k",
                    "max_tokens",
                    "max_connections",
                    "max_retries",
                    "timeout",
                }
                unknown = set(overrides.keys()) - _INSPECT_LENS_VALID_OVERRIDES
                if unknown:
                    logger.warning(
                        "'overrides' keys %s are not applied for EvaluationTask.INSPECT_LENS "
                        "and will be ignored. Use InspectLensConfig for benchmark, endpoint, "
                        "and output settings.",
                        sorted(unknown),
                    )

            # Check job cache before doing any work
            cached = load_existing_result(
                self._cache_context,
                job_name=job_name,
                job_type="inspect_lens",
                model_path=model_path,
                benchmarks_path=inspect_lens_config.benchmarks_path
                if inspect_lens_config
                else None,
                tasks=str(inspect_lens_config.tasks) if inspect_lens_config else None,
                inference_scenario=inspect_lens_config._infer_scenario()
                if inspect_lens_config
                else None,
                endpoint_name=inspect_lens_config.endpoint_name if inspect_lens_config else None,
                bedrock_model_id=inspect_lens_config.bedrock_model_id
                if inspect_lens_config
                else None,
                overrides=overrides or {},
            )
            if cached:
                logger.info("Returning cached InspectLens result for '%s'.", job_name)
                return cached  # type: ignore[return-value]

            return self._evaluate_inspect_lens(
                job_name=job_name,
                inspect_lens_config=inspect_lens_config,
                model_path=model_path,
                job_result=job_result,
                dry_run=dry_run,
                overrides=overrides,
            )
        if self._platform == Platform.BEDROCK:
            raise NotImplementedError(
                "Evaluation is not supported on the Bedrock platform. "
                "Use SageMaker platforms (SMTJ, SMHP) instead."
            )

        # MTRL evaluation on SMTJServerless delegates to the MultiTurnRLEvaluator
        if (
            eval_task == EvaluationTask.RFT_MULTITURN_EVAL
            and self._platform == Platform.SMTJServerless
        ):
            evaluate_base_model = False
            if task_config and hasattr(task_config, "evaluate_base_model"):
                evaluate_base_model = task_config.evaluate_base_model
            return self._execute_mtrl_eval(
                job_name=job_name,
                model_path=model_path,
                job_result=job_result,
                task_config=task_config,
                overrides=overrides,
                dry_run=dry_run,
                evaluate_base_model=evaluate_base_model,
            )

        # Check job cache
        cached = load_existing_result(
            self._cache_context,
            job_name=job_name,
            job_type="eval",
            model_path=model_path,
            recipe_path=recipe_path,
            overrides=overrides or {},
        )
        if cached:
            logger.info("Returning cached result for '%s'.", job_name)
            return cached  # type: ignore[return-value]

        tc = task_config or EvalTaskConfig()

        if tc.rl_env and tc.rl_env.get("reward_lambda_arn"):
            validate_rft_lambda_name(tc.rl_env["reward_lambda_arn"].split(":")[-1], self._platform)

        # Resolve model checkpoint
        resolved_model_path = resolve_model_checkpoint_path(
            model_path=model_path,
            job_result=job_result,
            customizer_job_id=None,
            customizer_output_s3_path=self.output_s3_path,
            customizer_model_path=None,
        )

        if resolved_model_path is None:
            logger.warning(
                f"Could not resolve model checkpoint path for evaluate job! "
                f"Falling back to base model {self.model}"
            )

        # Validate platform compatibility
        checkpoint_platform = None
        if resolved_model_path and resolved_model_path.startswith("s3://"):
            checkpoint_platform = detect_platform_from_path(resolved_model_path)

        if checkpoint_platform is None:
            if job_result is not None:
                if job_result.model_artifacts.checkpoint_s3_path:
                    checkpoint_platform = detect_platform_from_path(
                        job_result.model_artifacts.checkpoint_s3_path
                    )
            elif self.output_s3_path and self.output_s3_path.startswith("s3://"):
                checkpoint_platform = detect_platform_from_path(self.output_s3_path)

        validate_platform_compatibility(
            checkpoint_platform=checkpoint_platform,
            execution_platform=self._platform,
            checkpoint_source="evaluation model checkpoint",
        )

        # Resolve data path
        data_s3_path_for_job = tc.override_data_s3_path
        if data_s3_path_for_job is None and requires_custom_eval_data(eval_task):
            data_s3_path_for_job = self.data_s3_path

        if not requires_custom_eval_data(eval_task) and self.data_s3_path:
            logger.info(
                f"{eval_task} does not use custom data, ignoring ForgeEvaluator's data_s3_path."
            )

        # Resolve processor / rl_env with lambda auto-population
        resolved_processor = tc.processor
        resolved_rl_env = tc.rl_env
        infra_lambda_arn = getattr(self.infra, "rft_lambda_arn", None)

        if (
            eval_task == EvaluationTask.RFT_EVAL
            and resolved_processor
            and resolved_processor.get("lambda_arn")
        ):
            if resolved_rl_env is None:
                resolved_rl_env = {"reward_lambda_arn": resolved_processor["lambda_arn"]}
                logger.info(f"Using reward_lambda_arn: {resolved_processor['lambda_arn']}")
            resolved_processor = None

        if infra_lambda_arn and resolved_rl_env is None and eval_task == EvaluationTask.RFT_EVAL:
            resolved_rl_env = {"reward_lambda_arn": infra_lambda_arn}
            logger.info(f"Using reward_lambda_arn: {infra_lambda_arn}")

        recipe_builder = RecipeBuilder(
            region=self.region,
            job_name=job_name,
            platform=self._platform,
            model=self.model,
            method=TrainingMethod.EVALUATION,
            instance_type=self.infra.instance_type,
            instance_count=self.infra.instance_count,
            infra=self.infra,
            data_s3_path=data_s3_path_for_job,
            output_s3_path=self.output_s3_path,
            model_path=resolved_model_path,
            eval_task=eval_task,
            subtask=tc.subtask,
            processor_config=resolved_processor,
            rl_env_config=resolved_rl_env,
            image_uri_override=self._config.image_uri,
            is_multimodal=self._is_multimodal,
            mlflow_monitor=self._config.mlflow_monitor,
            rft_multiturn_infra=rft_multiturn_infra,
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
                method=TrainingMethod.EVALUATION,
                model_name_or_path=resolved_model_path,
            )
        )

        evaluation_result: EvaluationResult
        if self._platform in (Platform.SMTJ, Platform.SMTJServerless):
            eval_output_s3_path = (
                f"{resolved_output_s3_path.rstrip('/')}/{job_id}/output/output.tar.gz"
            )
            evaluation_result = SMTJEvaluationResult(
                job_id=job_id,
                eval_task=eval_task,
                started_time=start_time,
                eval_output_path=eval_output_s3_path,
                region=self.region,
            )
        else:
            cluster_name = cast(SMHPRuntimeManager, self.infra).cluster_name
            namespace = cast(SMHPRuntimeManager, self.infra).namespace
            eval_output_s3_path = f"{resolved_output_s3_path.rstrip('/')}/{job_id}/eval-result/"
            evaluation_result = SMHPEvaluationResult(
                job_id=job_id,
                eval_task=eval_task,
                started_time=start_time,
                eval_output_path=eval_output_s3_path,
                cluster_name=cluster_name,
                namespace=namespace,
                region=self.region,
            )

        logger.info(
            f"Started eval job '{job_id}'. Artifacts will be published to {eval_output_s3_path}"
        )
        persist_result(
            self._cache_context,
            evaluation_result,
            job_name=job_name,
            job_type="eval",
            model_path=model_path,
            recipe_path=recipe_path,
            overrides=overrides or {},
        )

        return evaluation_result

    def _evaluate_inspect_lens(
        self,
        job_name: str,
        inspect_lens_config: Optional[InspectLensConfig],
        model_path: Optional[str],
        job_result: Optional[TrainingResult],
        dry_run: bool,
        overrides: Optional[Dict[str, Any]] = None,
    ) -> Optional[EvaluationResult]:
        """Submit an InspectLens evaluation as a SageMaker Training Job.

        The training job runs the InspectLens container as an orchestrator only
        (no GPU needed).  Inference is delegated to a Bedrock endpoint or an
        existing SageMaker endpoint as configured in ``inspect_lens_config``.
        """
        if inspect_lens_config is None:
            raise ValueError(
                "inspect_lens_config is required when eval_task=EvaluationTask.INSPECT_LENS."
            )

        # Resolve the inference provider from model_path / job_result.
        # Priority: explicit model_path > job_result checkpoint > job_result model ARN.
        resolved_model_path = model_path
        if resolved_model_path is None and job_result is not None:
            resolved_model_path = (
                job_result.model_artifacts.checkpoint_s3_path
                or job_result.model_artifacts.output_model_arn
            )

        # Guard: an S3 checkpoint URI is not a valid Bedrock model ID.
        # If the resolved path is an S3 URI and no endpoint is configured, fall back to
        # the evaluator's model enum so the job doesn't fail at runtime.
        if (
            resolved_model_path is not None
            and resolved_model_path.startswith("s3://")
            and inspect_lens_config._infer_scenario() == "bedrock"
        ):
            logger.warning(
                "model_path '%s' is an S3 URI and cannot be used as a Bedrock model ID. "
                "Falling back to the evaluator model (%s) for Bedrock inference. "
                "To evaluate an S3 checkpoint, set endpoint_name or model_s3_uri + "
                "inference_image_uri on InspectLensConfig.",
                resolved_model_path,
                self.model.bedrock_model_id,
            )
            resolved_model_path = None

        inference_provider = self._build_inspect_lens_inference_provider(
            inspect_lens_config, resolved_model_path
        )

        # Resolve output S3 path: config override > evaluator default
        results_s3_path = inspect_lens_config.output_s3_path or (
            f"{self.output_s3_path.rstrip('/')}/inspectlens-results/"
        )

        # Generate run ID here — used for benchmarks, config, and job S3 paths
        run_id = str(uuid.uuid4())

        # Always use output_s3_path as the base for config and output.
        # benchmarks_path is just a reference in the YAML — no need to co-locate.
        base_s3_path = self.output_s3_path

        # Validate benchmarks_path is already an S3 URI.
        # Users must call upload_benchmarks() separately before starting the job.
        if (
            inspect_lens_config.benchmarks_path
            and not inspect_lens_config.benchmarks_path.startswith("s3://")
        ):
            raise ValueError(
                f"benchmarks_path must be an s3:// URI, got: '{inspect_lens_config.benchmarks_path}'. "
                f"Upload your local benchmarks first using evaluator.upload_benchmarks(local_dir, s3_path), "
                f"then pass the returned S3 URI as benchmarks_path in InspectLensConfig."
            )

        # Validate task names against @task functions in the benchmarks S3 path.
        # Done here (not in __post_init__) to avoid S3 calls during config construction.
        task_name_errors = validate_task_names_against_benchmarks(
            tasks=inspect_lens_config.tasks,
            benchmarks_path=inspect_lens_config.benchmarks_path,
            region=self.region,
        )
        if task_name_errors:
            bullet_list = "\n".join(f"  - {e}" for e in task_name_errors)
            raise ValueError(
                f"InspectLensConfig task validation failed with {len(task_name_errors)} error(s):\n{bullet_list}"
            )

        # Serialize config to YAML (needed for both dry_run local save and actual upload)
        # Populate MLflow tracking section from ForgeConfig.mlflow_monitor if provided.
        mlflow_monitor = self._config.mlflow_monitor
        mlflow_tracking_section: Optional[Dict[str, Any]] = None
        if mlflow_monitor and mlflow_monitor.tracking_uri:
            mlflow_tracking_section = {
                "mlflow_tracking_arn": mlflow_monitor.tracking_uri,
                "mlflow_experiment_name": mlflow_monitor.experiment_name or "nova-evals",
                "mlflow_tracing": True,
                "mlflow_log_artifacts": True,
            }
            logger.info(
                f"Populated InspectLens MLflow tracking from ForgeConfig.mlflow_monitor: "
                f"{mlflow_monitor.tracking_uri}"
            )

        config_dict = inspect_lens_config.to_yaml_dict(
            inference_provider=inference_provider,
            output_s3_path=results_s3_path,
            overrides=overrides,
        )
        if mlflow_tracking_section:
            config_dict["tracking"] = mlflow_tracking_section
        config_yaml = yaml.dump(config_dict, default_flow_style=False, sort_keys=False)

        # Save config YAML locally — always write to a temp dir (mirrors RecipeBuilder default),
        # or to generated_recipe_dir if explicitly set.
        if self._config.generated_recipe_dir:
            local_dir = os.path.expanduser(self._config.generated_recipe_dir)
        else:
            local_dir = mkdtemp()

        os.makedirs(local_dir, exist_ok=True)
        local_config_path = os.path.join(local_dir, f"{job_name}_inspect_config.yaml")
        with open(local_config_path, "w") as fh:
            fh.write(config_yaml)
        logger.info(f"InspectLens config saved locally to {local_config_path}")

        if dry_run:
            logger.info(
                f"[dry_run] InspectLens job '{job_name}' validated. "
                f"Inference provider: {list(inference_provider.keys())[0]}. "
                f"Results would be written to: {results_s3_path}"
            )
            return None

        # Truncate job_name first to guarantee the full UUID is always preserved
        config_s3_prefix = f"{base_s3_path.rstrip('/')}/{run_id}/config/"
        config_s3_uri = f"{config_s3_prefix}inspect_config.yaml"
        self._upload_config_to_s3(config_yaml, config_s3_uri)

        # Resolve container image: ForgeConfig > default (region-specific DLC image)
        image_uri = self._config.image_uri or get_inspect_lens_default_image_uri(self.region)

        start_time = datetime.now(timezone.utc)

        # Build environment vars for the SageMaker Training Job
        env_vars: Dict[str, Any] = dict(inspect_lens_config.environment or {})
        environment: Optional[Dict[str, Any]] = env_vars or None

        # InspectLens uses ml.m5.large (CPU orchestrator only — no GPU needed).
        # SMTJRuntimeManager.execute() uses ModelTrainer.from_recipe() which rejects
        # CPU instances, so we call create_training_job directly here.
        # TODO: extract sagemaker_client/execution_role resolution into a helper
        sm_client = getattr(self.infra, "sagemaker_client", None) or boto3.client(
            "sagemaker", region_name=self.region
        )
        execution_role = getattr(self.infra, "execution_role", None)
        if not execution_role:
            raise ValueError(
                "InspectLens requires an execution_role on the RuntimeManager. "
                "Pass execution_role to SMTJRuntimeManager."
            )

        job_id = self._submit_inspect_lens_training_job(
            job_name=job_name,
            run_id=run_id,
            base_s3_path=base_s3_path,
            config_s3_prefix=config_s3_prefix,
            image_uri=image_uri,
            environment=environment,
            sm_client=sm_client,
            execution_role=execution_role,
        )
        logger.info(
            f"InspectLens job '{job_id}' submitted. Results will be written to: {results_s3_path}"
        )

        evaluation_result = SMTJEvaluationResult(
            job_id=job_id,
            eval_task=EvaluationTask.INSPECT_LENS,
            started_time=start_time,
            eval_output_path=results_s3_path,
            sagemaker_client=sm_client,
        )
        persist_result(
            self._cache_context,
            evaluation_result,
            job_name=job_name,
            job_type="inspect_lens",
            model_path=model_path,
            benchmarks_path=inspect_lens_config.benchmarks_path,
            tasks=str(inspect_lens_config.tasks),
            inference_scenario=inspect_lens_config._infer_scenario(),
            endpoint_name=inspect_lens_config.endpoint_name,
            bedrock_model_id=inspect_lens_config.bedrock_model_id,
            overrides=overrides or {},
        )
        return evaluation_result

    def _submit_inspect_lens_training_job(
        self,
        job_name: str,
        run_id: str,
        base_s3_path: str,
        config_s3_prefix: str,
        image_uri: str,
        environment: Optional[Dict[str, Any]],
        sm_client: Any,
        execution_role: str,
    ) -> str:
        """Create the SageMaker Training Job for an InspectLens run and return the job name."""
        # Truncate job_name first to guarantee the full UUID is always preserved
        # TODO move to a separate helper function
        max_name_len = 63 - len(run_id) - 1  # 63 - 36 - 1 = 26
        unique_job_name = f"{job_name[:max_name_len]}-{run_id}"

        algorithm_spec: Dict[str, Any] = {
            "TrainingImage": image_uri,
            "TrainingInputMode": "File",
        }

        create_kwargs: Dict[str, Any] = {
            "TrainingJobName": unique_job_name,
            "AlgorithmSpecification": algorithm_spec,
            "RoleArn": execution_role,
            "InputDataConfig": [
                {
                    "ChannelName": "config",
                    "DataSource": {
                        "S3DataSource": {
                            "S3DataType": "S3Prefix",
                            "S3Uri": config_s3_prefix,
                            "S3DataDistributionType": "FullyReplicated",
                        }
                    },
                    "ContentType": "application/x-yaml",
                }
            ],
            "OutputDataConfig": {
                "S3OutputPath": f"{base_s3_path.rstrip('/')}/{run_id}/output/",
            },
            "ResourceConfig": {
                "InstanceType": self.infra.instance_type,
                "InstanceCount": 1,
                "VolumeSizeInGB": 30,
            },
            "StoppingCondition": {
                "MaxRuntimeInSeconds": getattr(self.infra, "max_job_runtime", 86400),
            },
        }
        if environment:
            create_kwargs["Environment"] = environment
        if self.infra.kms_key_id:
            create_kwargs["OutputDataConfig"]["KmsKeyId"] = self.infra.kms_key_id

        sm_client.create_training_job(**create_kwargs)
        return unique_job_name

    def _build_inspect_lens_inference_provider(
        self,
        inspect_lens_config: InspectLensConfig,
        model_path: Optional[str],
    ) -> Dict[str, Any]:
        """Build the ``inference_provider`` block for inspect_config.yaml.

        Priority:
          1. InspectLensConfig endpoint fields (endpoint_name / model_s3_uri)
          2. model_path / job_result passed to evaluate()
          3. Fallback: Bedrock with the evaluator's model enum value
        """
        scenario = inspect_lens_config._infer_scenario()

        if scenario == "existing_endpoint":
            ep_existing: Dict[str, Any] = {
                "endpoint_name": inspect_lens_config.endpoint_name,
                "region": self.region,
            }
            if inspect_lens_config.context_length:
                ep_existing["context_length"] = inspect_lens_config.context_length
            if inspect_lens_config.max_concurrency:
                ep_existing["max_concurrency"] = inspect_lens_config.max_concurrency
            ep_existing["enable_rai"] = inspect_lens_config.enable_rai
            return {"sagemaker_endpoint": ep_existing}

        if scenario == "create_endpoint":
            ep: Dict[str, Any] = {
                "endpoint_name": None,
                "region": self.region,
                "model_s3_uri": (inspect_lens_config.model_s3_uri or "").rstrip("/") + "/",
                "inference_image_uri": inspect_lens_config.inference_image_uri,
                "cleanup_endpoint": inspect_lens_config.cleanup_endpoint,
                "instance_count": inspect_lens_config.endpoint_instance_count,
                "enable_rai": inspect_lens_config.enable_rai,
                "endpoint_prefix": inspect_lens_config.endpoint_prefix,
            }
            if inspect_lens_config.endpoint_instance_type:
                ep["instance_type"] = inspect_lens_config.endpoint_instance_type
            if inspect_lens_config.endpoint_execution_role_arn:
                ep["execution_role_arn"] = inspect_lens_config.endpoint_execution_role_arn
            if inspect_lens_config.context_length:
                ep["context_length"] = inspect_lens_config.context_length
            if inspect_lens_config.max_concurrency:
                ep["max_concurrency"] = inspect_lens_config.max_concurrency
            if inspect_lens_config.endpoint_environment:
                ep["environment"] = inspect_lens_config.endpoint_environment
            return {"sagemaker_endpoint": ep}

        # Bedrock — resolve model_id from model_path, InspectLensConfig.bedrock_model_id,
        # or fall back to the evaluator's model enum (cross-region inference profile ID).
        if model_path is not None:
            bedrock_model_id = model_path
        elif inspect_lens_config.bedrock_model_id is not None:
            bedrock_model_id = inspect_lens_config.bedrock_model_id
        else:
            bedrock_model_id = self.model.bedrock_model_id
        return {
            "bedrock": {
                "model_id": bedrock_model_id,
                "region": self.region,
            }
        }

    @_telemetry_emitter(Feature.EVAL, "upload_benchmarks")
    def upload_benchmarks(self, local_dir: str, s3_path: str) -> str:
        """Upload a local benchmarks directory to S3.

        Call this before starting an InspectLens evaluation job. Pass the
        returned S3 URI as ``benchmarks_path`` in ``InspectLensConfig``.

        Args:
            local_dir: Path to the local directory containing benchmark ``.py``
                files with ``@task`` decorators.
            s3_path: S3 URI (``s3://bucket/prefix/``) where the benchmark files
                will be uploaded.

        Returns:
            The S3 URI where benchmarks were uploaded (same as ``s3_path``,
            normalized with a trailing slash).

        Raises:
            ValueError: If ``local_dir`` is not an existing directory or
                ``s3_path`` is not an ``s3://`` URI.
        """
        expanded = os.path.expanduser(local_dir)
        if not os.path.isdir(expanded):
            raise ValueError(f"local_dir '{local_dir}' is not an existing directory.")
        if not s3_path.startswith("s3://"):
            raise ValueError(f"s3_path must be an s3:// URI, got: '{s3_path}'")

        s3_path = s3_path.rstrip("/") + "/"
        s3_client = boto3.client("s3", region_name=self.region)
        bucket, prefix = parse_s3_uri(s3_path)
        uploaded = 0
        for fname in os.listdir(expanded):
            local_file = os.path.join(expanded, fname)
            if os.path.isfile(local_file) and (
                fname.endswith(".py") or fname in ("pyproject.toml", "requirements.txt")
            ):
                s3_key = f"{prefix}{fname}"
                s3_client.upload_file(local_file, bucket, s3_key)
                uploaded += 1
        logger.info(f"Uploaded {uploaded} benchmark file(s) from '{expanded}' to {s3_path}")
        return s3_path

    def _upload_config_to_s3(self, config_yaml: str, s3_uri: str) -> None:
        """Upload YAML string to an s3:// URI."""
        assert s3_uri.startswith("s3://"), f"Expected s3:// URI, got: {s3_uri}"
        bucket, key = parse_s3_uri(s3_uri)
        s3_client = boto3.client("s3", region_name=self.region)
        s3_client.put_object(
            Bucket=bucket,
            Key=key,
            Body=config_yaml.encode("utf-8"),
            ContentType="application/x-yaml",
        )
        logger.info(f"Uploaded InspectLens config to s3://{bucket}/{key}")

    def _execute_mtrl_eval(
        self,
        job_name: str,
        model_path: Optional[str] = None,
        job_result: Optional[TrainingResult] = None,
        task_config: Optional[EvalTaskConfig] = None,
        overrides: Optional[Dict[str, Any]] = None,
        dry_run: bool = False,
        evaluate_base_model: bool = False,
    ) -> Optional[EvaluationResult]:
        """Delegate MTRL evaluation to the MultiTurnRLEvaluator.

        This is called when eval_task is RFT_MULTITURN_EVAL on the
        SMTJServerless platform. It uses the sagemaker.train.evaluate
        MultiTurnRLEvaluator which creates a SageMaker Pipeline with
        the AgentRFTEvaluation job type.
        """
        infra = cast(SMTJServerlessRuntimeManager, self.infra)

        # Resolve model path from job_result if not explicitly provided
        resolved_model_path = model_path
        if resolved_model_path is None and job_result is not None:
            # For MTRL, use the model package ARN directly

            if isinstance(job_result, SMTJTrainingResult) and job_result._is_mtrl:
                resolved_model_path = job_result.model_artifacts.output_model_arn
            else:
                resolved_model_path = resolve_model_checkpoint_path(
                    model_path=None,
                    job_result=job_result,
                    customizer_job_id=None,
                    customizer_output_s3_path=self.output_s3_path,
                    customizer_model_path=None,
                )

        # Resolve MLflow tracking URI (required for AgentRFT evaluation jobs)
        if not self._config.mlflow_monitor or not self._config.mlflow_monitor.tracking_uri:
            raise ValueError(
                "MLflow configuration is required for AgentRFT evaluation jobs. "
                "Please provide an mlflow_monitor with a valid tracking_uri when "
                "using RFT_MULTITURN_EVAL on the SMTJServerless platform."
            )
        mlflow_uri = self._config.mlflow_monitor.tracking_uri

        if dry_run:
            logger.info(
                f"[dry_run] Would launch MTRL evaluation '{job_name}' "
                f"with model_path={resolved_model_path}"
            )
            return None

        # Pass training job name so the evaluator can attach and resolve model artifacts
        training_job_name = job_result.job_id if job_result is not None else None

        execution = infra.execute_mtrl_eval(
            model=self.model,
            data_s3_path=self.data_s3_path,
            output_s3_path=self.output_s3_path,
            mlflow_tracking_uri=mlflow_uri,
            model_path=resolved_model_path,
            overrides=overrides,
            training_job_name=training_job_name,
            evaluate_base_model=evaluate_base_model,
        )

        start_time = datetime.now(timezone.utc)
        eval_output_s3_path = f"{self.output_s3_path.rstrip('/')}/{job_name}/"

        evaluation_result = SMTJEvaluationResult(
            job_id=execution.arn,
            eval_task=EvaluationTask.RFT_MULTITURN_EVAL,
            started_time=start_time,
            eval_output_path=eval_output_s3_path,
            region=self.region,
        )
        evaluation_result._job_name = job_name  # type: ignore[attr-defined]
        # Attach the execution object for downstream access (wait, show_results, etc.)
        evaluation_result._mtrl_execution = execution  # type: ignore[attr-defined]

        logger.info(f"Started MTRL evaluation pipeline. Execution ARN: {execution.arn}")
        return evaluation_result

    @_telemetry_emitter(Feature.EVAL, "get_logs")
    def get_logs(
        self,
        job_result: Optional[EvaluationResult] = None,
        job_id: Optional[str] = None,
        started_time: Optional[datetime] = None,
        limit: Optional[int] = None,
        start_from_head: bool = False,
        end_time: Optional[int] = None,
    ) -> None:
        """Stream CloudWatch logs for an evaluation job.

        Raises:
            ValueError: If neither ``job_result`` nor both ``job_id`` and
                ``started_time`` are provided.
        """
        resolved_job_id = job_result.job_id if job_result else job_id
        resolved_started = job_result.started_time if job_result else started_time

        if not resolved_job_id or not resolved_started:
            raise ValueError(
                "No job reference provided. Pass either a job_result or explicit job_id and started_time."
            )

        is_mtrl_eval = (
            job_result
            and hasattr(job_result, "eval_task")
            and job_result.eval_task == EvaluationTask.RFT_MULTITURN_EVAL
        )

        if is_mtrl_eval:
            from amzn_nova_forge.monitor import MTRLLogMonitor

            monitor = MTRLLogMonitor.from_job_id(
                job_id=resolved_job_id,
                region=self.region,
                job_category="AgentRFTEvaluation",
            )
            monitor.show_logs(limit=limit)
            return

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
