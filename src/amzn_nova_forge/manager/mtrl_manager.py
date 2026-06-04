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
from typing import Any, Dict, List, Optional

import boto3

from amzn_nova_forge.core.enums import Model
from amzn_nova_forge.telemetry.constants import Feature
from amzn_nova_forge.telemetry.telemetry_logging import _telemetry_emitter
from amzn_nova_forge.util.logging import logger


class MTRLOperations:
    """Multi-turn RL training and evaluation operations for serverless runtime.

    Mixin class — expects to be composed with SMTJServerlessRuntimeManager
    which provides the attributes declared below.
    """

    model_package_group_arn: str
    execution_role: Optional[str]
    subnets: Optional[List[str]]
    security_group_ids: Optional[List[str]]
    kms_key_id: Optional[str]
    agent_core_arn: Optional[str]
    rft_lambda: Optional[str]
    region: str

    def _get_or_create_checkpoint_model_package_group_arn(self) -> str:
        raise NotImplementedError

    def _mtrl_common_kwargs(
        self,
        output_s3_path: Optional[str] = None,
        mlflow_tracking_uri: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Build common kwargs shared between MTRL training and evaluation."""
        kwargs: Dict[str, Any] = {}
        if output_s3_path:
            kwargs["s3_output_path"] = output_s3_path
        if self.model_package_group_arn:
            kwargs["output_model_package_group"] = self.model_package_group_arn
            kwargs["intermediate_checkpoint_model_package_group"] = (
                self._get_or_create_checkpoint_model_package_group_arn()
            )
        if mlflow_tracking_uri:
            kwargs["mlflow_app_arn"] = mlflow_tracking_uri
        if self.execution_role:
            kwargs["role"] = self.execution_role
        if self.subnets or self.security_group_ids:
            from sagemaker.core.shapes import VpcConfig

            kwargs["networking"] = VpcConfig(
                security_group_ids=self.security_group_ids or [],
                subnets=self.subnets or [],
            )
        if self.kms_key_id:
            kwargs["kms_key_arn"] = self.kms_key_id
        return kwargs

    @staticmethod
    def _apply_mtrl_overrides(obj, overrides: Optional[Dict[str, Any]], label: str = "") -> None:
        """Apply overrides to trainer/evaluator hyperparameters object."""
        if not overrides:
            return
        # obj.hyperparameters is a lazy property that fetches valid parameter specs
        # from the SageMaker Public Hub. We need it before applying overrides so that
        # setattr can validate each key/value against the recipe's allowed parameters.
        try:
            hp = obj.hyperparameters
        except (ValueError, RuntimeError) as e:
            skipped = ", ".join(f"{k}={v}" for k, v in overrides.items())
            logger.warning(
                f"Overrides not supported currently — "
                f"skipping {label}hyperparameters: [{skipped}]. "
                f"The job will proceed with default values."
            )
            return
        for key, value in overrides.items():
            try:
                setattr(hp, key, value)
            except AttributeError:
                logger.warning(
                    f"Skipping unsupported {label}hyperparameter '{key}' — not in recipe."
                )
            except (TypeError, ValueError) as e:
                raise ValueError(f"Invalid {label}hyperparameter '{key}={value}': {e}") from e

    @_telemetry_emitter(Feature.TRAINING, "execute_mtrl")
    def execute_mtrl(
        self,
        model: Model,
        job_name: str,
        data_s3_path: Optional[str] = None,
        output_s3_path: Optional[str] = None,
        mlflow_tracking_uri: Optional[str] = None,
        overrides: Optional[Dict[str, Any]] = None,
        model_path: Optional[str] = None,
    ) -> str:
        """Execute a multi-turn RFT job via sagemaker.train MultiTurnRLTrainer.

        Args:
            model: The Nova model enum.
            job_name: Base job name.
            data_s3_path: S3 path to training data.
            output_s3_path: S3 output path.
            mlflow_tracking_uri: MLflow tracking server ARN.
            overrides: Optional hyperparameter overrides.
            model_path: Model package ARN from a previous job for iterative training.

        Returns:
            The job name string from the SDK.
        """
        # TODO: Move to top-level import when MTRL is available in the PySdk
        from sagemaker.core.resources import ModelPackage
        from sagemaker.train.multi_turn_rl_trainer import MultiTurnRLTrainer as SmMTRL

        agent_arn = self.agent_core_arn or self.rft_lambda

        resolved_model: Any = model.hub_content_name
        if model_path and ":model-package/" in model_path:
            resolved_model = ModelPackage.get(model_path)

        kwargs = {
            "model": resolved_model,
            "agent_env": agent_arn,
            "accept_eula": True,
            **self._mtrl_common_kwargs(output_s3_path, mlflow_tracking_uri),
        }
        if data_s3_path:
            kwargs["training_dataset"] = data_s3_path

        sm_trainer = SmMTRL(**kwargs)
        self._apply_mtrl_overrides(sm_trainer, overrides)
        sm_trainer.base_job_name = job_name

        sm_job = sm_trainer.train(wait=False)
        logger.info(f"MTRL job created: {sm_job.job_name} ({sm_job.job_arn})")
        return sm_job.job_name

    @_telemetry_emitter(Feature.EVAL, "execute_mtrl_eval")
    def execute_mtrl_eval(
        self,
        model: Model,
        data_s3_path: Optional[str] = None,
        output_s3_path: Optional[str] = None,
        mlflow_tracking_uri: Optional[str] = None,
        model_path: Optional[str] = None,
        overrides: Optional[Dict[str, Any]] = None,
        training_job_name: Optional[str] = None,
        evaluate_base_model: bool = False,
    ):
        """Execute MTRL evaluation via sagemaker.train MultiTurnRLEvaluator.

        Constructs a MultiTurnRLTrainer, attaches the completed training job,
        and passes the trainer to MultiTurnRLEvaluator so it can auto-resolve
        the model package and agent config.

        Args:
            model: The Nova model enum.
            data_s3_path: S3 path to evaluation dataset.
            output_s3_path: S3 output path for results.
            mlflow_tracking_uri: MLflow tracking server ARN.
            model_path: Model package ARN of a fine-tuned model to evaluate.
            overrides: Optional hyperparameter overrides for evaluation.
            training_job_name: Name of the completed MTRL training job to attach.

        Returns:
            The MTRLEvaluationExecution object from the SDK.
        """
        # TODO: Move to top-level import when MTRL is available in the PySdk
        from sagemaker.train.evaluate import MultiTurnRLEvaluator
        from sagemaker.train.multi_turn_rl_trainer import MultiTurnRLTrainer as SmMTRL

        agent_arn = self.agent_core_arn or self.rft_lambda
        common_kwargs = self._mtrl_common_kwargs(output_s3_path, mlflow_tracking_uri)

        # The evaluator uses 'mlflow_resource_arn' not 'mlflow_app_arn', and accepts
        # 'kms_key_id' and 'region' which the trainer does not (BaseEvaluator fields).
        eval_common_kwargs = dict(common_kwargs)
        if "mlflow_app_arn" in eval_common_kwargs:
            eval_common_kwargs["mlflow_resource_arn"] = eval_common_kwargs.pop("mlflow_app_arn")
        if self.kms_key_id:
            eval_common_kwargs["kms_key_id"] = self.kms_key_id
        if self.region:
            eval_common_kwargs["region"] = self.region

        # Build a trainer instance and attach the completed job so the evaluator
        # can auto-resolve model artifacts and agent config.
        if training_job_name and model_path:
            trainer_kwargs = {
                "model": model.hub_content_name,
                "agent_env": agent_arn,
                "accept_eula": True,
                **common_kwargs,
            }
            if data_s3_path:
                trainer_kwargs["training_dataset"] = data_s3_path

            sm_trainer = SmMTRL(**trainer_kwargs)
            job = SmMTRL.attach(training_job_name)
            sm_trainer._latest_job = job
            sm_trainer.base_model_arn = sm_trainer._model_arn
            sm_trainer.base_model_name = model.hub_content_name
            sm_trainer.agent_config = agent_arn

            eval_kwargs = {
                "model": sm_trainer,
                "dataset": data_s3_path,
                **eval_common_kwargs,
            }
        else:
            # Fallback: pass model directly with agent_config.
            # If model_path is a model-package ARN (Restricted MPG), pass the
            # base hub-content name as `model` and inject the ARN post-init to
            # avoid pydantic validation failure on missing s3_uri.
            is_model_package_arn = model_path and ":model-package/" in model_path
            eval_kwargs = {
                "model": model.hub_content_name
                if is_model_package_arn
                else (model_path or model.hub_content_name),
                "dataset": data_s3_path,
                "agent_config": agent_arn,
                **eval_common_kwargs,
            }

        if evaluate_base_model:
            eval_kwargs["evaluate_base_model"] = True

        evaluator = MultiTurnRLEvaluator(**eval_kwargs)

        if not training_job_name and model_path and ":model-package/" in model_path:
            evaluator._source_model_package_arn_cache = model_path

        self._apply_mtrl_overrides(evaluator, overrides, label="eval ")

        execution = evaluator.evaluate()
        logger.info(f"MTRL evaluation pipeline started: {execution.arn}")
        return execution

    def _cleanup_mtrl(self, job_name: str) -> None:
        """Stop an MTRL job."""
        # TODO: Move to top-level import when MTRL is available in the PySdk
        from sagemaker.train.agent_rft_job import AgentRFTJob

        session = boto3.Session(region_name=self.region)
        rft_job = AgentRFTJob.get(job_name, session=session)
        rft_job.stop()
        logger.info(f"Stopped MTRL job '{job_name}'")
