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
import json
import os
import tempfile
import unittest
from unittest.mock import MagicMock, patch

import yaml

from amzn_nova_forge.core.enums import Model, Platform, TrainingMethod
from amzn_nova_forge.core.types import ValidationConfig
from amzn_nova_forge.manager.runtime_manager import (
    _METHOD_TO_SERVERLESS_CONFIG,
    DEFAULT_SMTJ_JOB_MAX_RUNTIME,
    JobConfig,
    SMTJServerlessRuntimeManager,
)


class TestSMTJServerlessRuntimeManager(unittest.TestCase):
    def setUp(self):
        self.mock_role = "arn:aws:iam::123456789012:role/SageMakerExecutionRole"
        self.model_package_group_name = "test-model-package-group"
        self.mock_group_arn = (
            "arn:aws:sagemaker:us-east-1:123456789012:model-package-group/test-model-package-group"
        )
        self.mock_hub_content_arn = (
            "arn:aws:sagemaker:us-east-1:123456789012:"
            "hub-content/SageMakerPublicHub/Model/nova-textgeneration-lite-v2/1.0.0"
        )

    @patch.object(SMTJServerlessRuntimeManager, "setup", return_value=None)
    def _create_manager(self, mock_setup, **kwargs):
        manager = SMTJServerlessRuntimeManager(
            model_package_group_name=self.model_package_group_name, **kwargs
        )
        manager.execution_role = self.mock_role
        manager.sagemaker_client = MagicMock()
        manager.sagemaker_session = MagicMock()
        manager.region = "us-east-1"
        manager.model_package_group_arn = self.mock_group_arn
        return manager

    # --- Initialization tests ---

    @patch.object(SMTJServerlessRuntimeManager, "setup", return_value=None)
    def test_initialization_defaults(self, mock_setup):
        manager = SMTJServerlessRuntimeManager(
            model_package_group_name=self.model_package_group_name
        )
        self.assertEqual(manager.model_package_group_name, self.model_package_group_name)
        self.assertIsNone(manager.instance_type)
        self.assertIsNone(manager.instance_count)
        self.assertFalse(manager.encrypt_inter_container_traffic)
        self.assertIsNone(manager.subnets)
        self.assertIsNone(manager.security_group_ids)
        self.assertEqual(manager.max_job_runtime, DEFAULT_SMTJ_JOB_MAX_RUNTIME)

    @patch.object(SMTJServerlessRuntimeManager, "setup", return_value=None)
    def test_initialization_custom_params(self, mock_setup):
        manager = SMTJServerlessRuntimeManager(
            model_package_group_name=self.model_package_group_name,
            execution_role=self.mock_role,
            kms_key_id="my-kms-key",
            encrypt_inter_container_traffic=True,
            subnets=["subnet-1"],
            security_group_ids=["sg-1"],
            max_job_runtime=3600,
        )
        self.assertEqual(manager.subnets, ["subnet-1"])
        self.assertEqual(manager.security_group_ids, ["sg-1"])
        self.assertTrue(manager.encrypt_inter_container_traffic)
        self.assertEqual(manager.max_job_runtime, 3600)
        self.assertEqual(manager.kms_key_id, "my-kms-key")

    # --- Setup tests ---

    @patch("amzn_nova_forge.manager.runtime_manager.Session")
    @patch("amzn_nova_forge.manager.runtime_manager.get_execution_role")
    @patch("amzn_nova_forge.manager.runtime_manager.boto3.client")
    @patch("amzn_nova_forge.manager.runtime_manager.boto3.session.Session")
    def test_setup_creates_model_package_group_if_not_exists(
        self,
        mock_boto_session_cls,
        mock_boto_client,
        mock_get_execution_role,
        mock_session_cls,
    ):
        mock_boto_session = MagicMock()
        mock_boto_session.region_name = "us-west-2"
        mock_boto_session_cls.return_value = mock_boto_session

        mock_client = MagicMock()
        mock_client.exceptions.ClientError = type("ClientError", (Exception,), {})
        mock_client.describe_model_package_group.side_effect = mock_client.exceptions.ClientError()
        mock_client.create_model_package_group.return_value = {
            "ModelPackageGroupArn": self.mock_group_arn
        }
        mock_boto_client.return_value = mock_client

        mock_get_execution_role.return_value = self.mock_role

        manager = SMTJServerlessRuntimeManager(
            model_package_group_name=self.model_package_group_name
        )

        self.assertEqual(manager.region, "us-west-2")
        self.assertEqual(manager.execution_role, self.mock_role)
        self.assertEqual(manager.model_package_group_arn, self.mock_group_arn)
        mock_client.create_model_package_group.assert_called_once_with(
            ModelPackageGroupName=self.model_package_group_name
        )

    @patch("amzn_nova_forge.manager.runtime_manager.Session")
    @patch("amzn_nova_forge.manager.runtime_manager.get_execution_role")
    @patch("amzn_nova_forge.manager.runtime_manager.boto3.client")
    @patch("amzn_nova_forge.manager.runtime_manager.boto3.session.Session")
    def test_setup_uses_existing_model_package_group(
        self,
        mock_boto_session_cls,
        mock_boto_client,
        mock_get_execution_role,
        mock_session_cls,
    ):
        mock_boto_session = MagicMock()
        mock_boto_session.region_name = None
        mock_boto_session_cls.return_value = mock_boto_session

        mock_client = MagicMock()
        mock_client.describe_model_package_group.return_value = {
            "ModelPackageGroupArn": self.mock_group_arn
        }
        mock_boto_client.return_value = mock_client

        mock_get_execution_role.return_value = self.mock_role

        manager = SMTJServerlessRuntimeManager(
            model_package_group_name=self.model_package_group_name
        )

        self.assertEqual(manager.region, "us-east-1")  # fallback when region_name is None
        self.assertEqual(manager.model_package_group_arn, self.mock_group_arn)
        mock_client.create_model_package_group.assert_not_called()

    @patch("amzn_nova_forge.manager.runtime_manager.Session")
    @patch("amzn_nova_forge.manager.runtime_manager.boto3.client")
    @patch("amzn_nova_forge.manager.runtime_manager.boto3.session.Session")
    def test_setup_uses_explicit_execution_role(
        self, mock_boto_session_cls, mock_boto_client, mock_session_cls
    ):
        mock_boto_session = MagicMock()
        mock_boto_session.region_name = "us-east-1"
        mock_boto_session_cls.return_value = mock_boto_session

        mock_client = MagicMock()
        mock_client.describe_model_package_group.return_value = {
            "ModelPackageGroupArn": self.mock_group_arn
        }
        mock_boto_client.return_value = mock_client

        manager = SMTJServerlessRuntimeManager(
            model_package_group_name=self.model_package_group_name,
            execution_role=self.mock_role,
        )

        self.assertEqual(manager.execution_role, self.mock_role)

    # --- _extract_hyperparameters tests ---

    def test_extract_hyperparameters_flat(self):
        manager = self._create_manager()
        recipe = {"lr": 0.001, "max_steps": 100, "global_batch_size": 8}
        result = manager._extract_hyperparameters(recipe)
        self.assertEqual(
            result,
            {
                "learning_rate": "0.001",
                "max_steps": "100",
                "global_batch_size": "8",
            },
        )

    def test_extract_hyperparameters_nested(self):
        manager = self._create_manager()
        recipe = {
            "run": {"name": "my-run", "max_steps": 50},
            "trainer": {"lr": 0.01},
        }
        result = manager._extract_hyperparameters(recipe)
        self.assertEqual(
            result,
            {
                "name": "my-run",
                "max_steps": "50",
                "learning_rate": "0.01",
            },
        )

    def test_extract_hyperparameters_three_levels(self):
        manager = self._create_manager()
        recipe = {"model": {"peft": {"alpha": 16}}}
        result = manager._extract_hyperparameters(recipe)
        self.assertEqual(result, {"lora_alpha": "16"})

    def test_extract_hyperparameters_four_levels(self):
        """Recursive extraction handles arbitrarily nested keys."""
        manager = self._create_manager()
        recipe = {"training_config": {"rollout": {"rewards": {"api_endpoint": {"alpha": 16}}}}}
        result = manager._extract_hyperparameters(recipe)
        self.assertEqual(result, {"lora_alpha": "16"})

    def test_extract_hyperparameters_skips_none(self):
        manager = self._create_manager()
        recipe = {"lr": None, "max_steps": 10}
        result = manager._extract_hyperparameters(recipe)
        self.assertEqual(result, {"max_steps": "10"})

    def test_extract_hyperparameters_empty_recipe(self):
        manager = self._create_manager()
        self.assertEqual(manager._extract_hyperparameters({}), {})

    # --- _build_serverless_job_config tests ---

    def test_build_serverless_job_config_sft_lora(self):
        manager = self._create_manager()
        config = manager._build_serverless_job_config(TrainingMethod.SFT_LORA, "arn:model")
        self.assertEqual(
            config,
            {
                "BaseModelArn": "arn:model",
                "AcceptEula": True,
                "JobType": "FineTuning",
                "CustomizationTechnique": "SFT",
                "Peft": "LORA",
            },
        )

    def test_build_serverless_job_config_sft_full(self):
        manager = self._create_manager()
        config = manager._build_serverless_job_config(TrainingMethod.SFT_FULL, "arn:model")
        self.assertNotIn("Peft", config)
        self.assertEqual(config["CustomizationTechnique"], "SFT")

    def test_build_serverless_job_config_dpo_lora(self):
        manager = self._create_manager()
        config = manager._build_serverless_job_config(TrainingMethod.DPO_LORA, "arn:model")
        self.assertEqual(config["CustomizationTechnique"], "DPO")
        self.assertEqual(config["Peft"], "LORA")

    def test_build_serverless_job_config_rft_lora(self):
        manager = self._create_manager()
        config = manager._build_serverless_job_config(TrainingMethod.RFT_LORA, "arn:model")
        self.assertEqual(
            config,
            {
                "BaseModelArn": "arn:model",
                "AcceptEula": True,
                "JobType": "FineTuning",
                "CustomizationTechnique": "RLVR",
                "Peft": "LORA",
            },
        )

    def test_build_serverless_job_config_rft_lora_with_evaluator_arn(self):
        manager = self._create_manager()
        hub_arn = "arn:aws:sagemaker:us-east-1:123456789012:hub-content/recipestest/JsonDoc/my-reward/0.0.1"
        config = manager._build_serverless_job_config(
            TrainingMethod.RFT_LORA, "arn:model", evaluator_arn=hub_arn
        )
        self.assertEqual(config["CustomizationTechnique"], "RLVR")
        self.assertEqual(config["Peft"], "LORA")
        self.assertEqual(config["EvaluatorArn"], hub_arn)

    def test_build_serverless_job_config_rft_lora_lambda_arn_not_evaluator(self):
        """Lambda ARN must NOT be set as EvaluatorArn for training jobs."""
        manager = self._create_manager()
        lambda_arn = "arn:aws:lambda:us-east-1:123456789012:function:my-reward"
        config = manager._build_serverless_job_config(
            TrainingMethod.RFT_LORA, "arn:model", evaluator_arn=lambda_arn
        )
        self.assertEqual(config["CustomizationTechnique"], "RLVR")
        self.assertNotIn("EvaluatorArn", config)

    def test_rft_lambda_setter_accepts_hub_content_arn(self):
        """Hub-content ARN set as rft_lambda is stored directly as rft_lambda_arn."""
        manager = self._create_manager()
        hub_arn = (
            "arn:aws:sagemaker:us-east-1:123456789012:hub-content/my-hub/JsonDoc/my-reward/0.0.1"
        )
        manager.rft_lambda = hub_arn
        self.assertEqual(manager.rft_lambda_arn, hub_arn)

    def test_rft_lambda_setter_lambda_arn_sets_rft_lambda_arn(self):
        manager = self._create_manager()
        lambda_arn = "arn:aws:lambda:us-east-1:123456789012:function:my-reward"
        manager.rft_lambda = lambda_arn
        self.assertEqual(manager.rft_lambda_arn, lambda_arn)

    def test_rft_lambda_setter_file_path_clears_rft_lambda_arn(self):
        manager = self._create_manager()
        manager.rft_lambda = "my_reward.py"
        self.assertIsNone(manager.rft_lambda_arn)

    # --- validate_lambda tests ---

    @patch("amzn_nova_forge.manager.runtime_manager.extract_lambda_arn_from_hub_content")
    @patch("amzn_nova_forge.manager.runtime_manager.RuntimeManager.validate_lambda")
    def test_validate_lambda_hub_content_arn_extracts_and_delegates(
        self, mock_super_validate, mock_extract
    ):
        """Hub-content ARN path: extracts Lambda ARN, delegates to super with extracted ARN, restores original."""
        manager = self._create_manager()
        hub_arn = (
            "arn:aws:sagemaker:us-east-1:123456789012:hub-content/my-hub/JsonDoc/my-reward/0.0.1"
        )
        lambda_arn = "arn:aws:lambda:us-east-1:123456789012:function:my-reward"
        manager._rft_lambda_arn = hub_arn
        mock_extract.return_value = lambda_arn

        manager.validate_lambda("s3://bucket/data.jsonl")

        mock_extract.assert_called_once_with(hub_arn, manager.sagemaker_client)
        mock_super_validate.assert_called_once_with("s3://bucket/data.jsonl", 10)
        # Original ARN is restored after validation
        self.assertEqual(manager._rft_lambda_arn, hub_arn)

    @patch("amzn_nova_forge.manager.runtime_manager.extract_lambda_arn_from_hub_content")
    @patch("amzn_nova_forge.manager.runtime_manager.RuntimeManager.validate_lambda")
    def test_validate_lambda_hub_content_arn_missing_lambda_logs_warning_and_returns(
        self, mock_super_validate, mock_extract
    ):
        """Hub-content ARN with missing Lambda reference: logs warning and returns early."""
        manager = self._create_manager()
        hub_arn = (
            "arn:aws:sagemaker:us-east-1:123456789012:hub-content/my-hub/JsonDoc/my-reward/0.0.1"
        )
        manager._rft_lambda_arn = hub_arn
        mock_extract.return_value = None

        manager.validate_lambda("s3://bucket/data.jsonl")

        mock_extract.assert_called_once_with(hub_arn, manager.sagemaker_client)
        mock_super_validate.assert_not_called()

    @patch("amzn_nova_forge.manager.runtime_manager.RuntimeManager.validate_lambda")
    def test_validate_lambda_non_hub_content_delegates_to_super(self, mock_super_validate):
        """Non-hub-content ARN path: delegates directly to super."""
        manager = self._create_manager()
        lambda_arn = "arn:aws:lambda:us-east-1:123456789012:function:my-reward"
        manager._rft_lambda_arn = lambda_arn

        manager.validate_lambda("s3://bucket/data.jsonl", validation_samples=5)

        mock_super_validate.assert_called_once_with("s3://bucket/data.jsonl", 5)

    def test_build_serverless_job_config_unsupported_method(self):
        manager = self._create_manager()
        with self.assertRaises(ValueError):
            manager._build_serverless_job_config(TrainingMethod.CPT, "arn:model")

    def test_build_serverless_job_config_rft_full_raises_value_error(self):
        """RFT_FULL raises a clear ValueError with a helpful message on SMTJServerless."""
        manager = self._create_manager()
        with self.assertRaises(ValueError) as ctx:
            manager._build_serverless_job_config(TrainingMethod.RFT_FULL, "arn:model")
        self.assertIn("not supported on SMTJServerless", str(ctx.exception))
        self.assertIn("RFT_LORA", str(ctx.exception))

    def test_build_serverless_job_config_eval_benchmark(self):
        manager = self._create_manager()
        config = manager._build_serverless_job_config(
            TrainingMethod.EVALUATION, "arn:model", eval_task="mmlu"
        )
        self.assertEqual(config["JobType"], "Evaluation")
        self.assertEqual(config["EvaluationType"], "BenchmarkEvaluation")
        self.assertNotIn("CustomizationTechnique", config)
        self.assertNotIn("EvaluatorArn", config)

    def test_build_serverless_job_config_eval_llm_judge_is_benchmark(self):
        manager = self._create_manager()
        config = manager._build_serverless_job_config(
            TrainingMethod.EVALUATION, "arn:model", eval_task="llm_judge"
        )
        self.assertEqual(config["EvaluationType"], "BenchmarkEvaluation")
        self.assertNotIn("EvaluatorArn", config)

    def test_build_serverless_job_config_eval_custom_scorer(self):
        manager = self._create_manager()
        for task in ("gen_qa", "rft_eval"):
            config = manager._build_serverless_job_config(
                TrainingMethod.EVALUATION, "arn:model", eval_task=task
            )
            self.assertEqual(config["EvaluationType"], "CustomScorerEvaluation")
            self.assertNotIn("EvaluatorArn", config)

    def test_build_serverless_job_config_eval_custom_scorer_with_evaluator_arn(self):
        """EvaluatorArn is NOT set for eval jobs — lambda goes in HyperParameters instead."""
        manager = self._create_manager()
        hub_arn = "arn:aws:sagemaker:us-east-1:123456789012:hub-content/recipestest/JsonDoc/my-reward/0.0.1"
        config = manager._build_serverless_job_config(
            TrainingMethod.EVALUATION,
            "arn:model",
            eval_task="rft_eval",
            evaluator_arn=hub_arn,
        )
        self.assertEqual(config["EvaluationType"], "CustomScorerEvaluation")
        self.assertNotIn("EvaluatorArn", config)

    def test_build_serverless_job_config_lambda_arn_not_set_as_evaluator_arn(self):
        """Lambda ARNs must NOT be sent as EvaluatorArn — only hub-content ARNs are accepted."""
        manager = self._create_manager()
        lambda_arn = "arn:aws:lambda:us-east-1:123456789012:function:my-reward"
        config = manager._build_serverless_job_config(
            TrainingMethod.EVALUATION,
            "arn:model",
            eval_task="rft_eval",
            evaluator_arn=lambda_arn,
        )
        self.assertEqual(config["EvaluationType"], "CustomScorerEvaluation")
        self.assertNotIn("EvaluatorArn", config)

    def test_build_serverless_job_config_evaluator_arn_not_set_for_benchmark(self):
        """EvaluatorArn must not be sent for benchmark evals even if provided."""
        manager = self._create_manager()
        hub_arn = "arn:aws:sagemaker:us-east-1:123456789012:hub-content/h/JsonDoc/fn/0.0.1"
        config = manager._build_serverless_job_config(
            TrainingMethod.EVALUATION,
            "arn:model",
            eval_task="mmlu",
            evaluator_arn=hub_arn,
        )
        self.assertEqual(config["EvaluationType"], "BenchmarkEvaluation")
        self.assertNotIn("EvaluatorArn", config)

    @patch(
        "amzn_nova_forge.manager.runtime_manager.get_hub_content",
        return_value={"HubContentArn": "arn:hub:content"},
    )
    def test_resolve_base_model_arn(self, mock_hub):
        manager = self._create_manager()
        arn = manager._resolve_base_model_arn(Model.NOVA_LITE_2)
        self.assertEqual(arn, "arn:hub:content")
        mock_hub.assert_called_once_with(
            hub_name="SageMakerPublicHub",
            hub_content_name="nova-textgeneration-lite-v2",
            hub_content_type="Model",
            region="us-east-1",
            hub_content_version=None,
        )

    @patch("amzn_nova_forge.manager.runtime_manager.get_hub_content")
    @patch("sagemaker.ai_registry.dataset.DataSet")
    def test_execute_success(self, mock_dataset_cls, mock_hub_content):
        manager = self._create_manager()
        mock_hub_content.return_value = {"HubContentArn": self.mock_hub_content_arn}

        mock_dataset = MagicMock()
        mock_dataset.arn = "arn:aws:sagemaker:us-east-1:123456789012:dataset/train-input"
        mock_dataset_cls.create.return_value = mock_dataset

        manager.sagemaker_client.create_training_job.return_value = {
            "TrainingJobArn": "arn:aws:sagemaker:us-east-1:123456789012:training-job/test-job"
        }

        recipe = {
            "run": {"model_type": "amazon.nova-2-lite-v1:0:256k"},
            "trainer": {"lr": 0.001},
        }
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(recipe, f)
            recipe_path = f.name

        try:
            job_config = JobConfig(
                job_name="test-serverless-job",
                image_uri="",
                recipe_path=recipe_path,
                output_s3_path="s3://output-bucket/output",
                data_s3_path="s3://input-bucket/data.jsonl",
                method=TrainingMethod.SFT_LORA,
            )
            job_id = manager.execute(job_config)
        finally:
            os.unlink(recipe_path)

        self.assertEqual(job_id, "test-job")
        call_kwargs = manager.sagemaker_client.create_training_job.call_args.kwargs
        self.assertEqual(call_kwargs["TrainingJobName"], "test-serverless-job")
        self.assertEqual(call_kwargs["RoleArn"], self.mock_role)
        self.assertIn("ServerlessJobConfig", call_kwargs)
        self.assertEqual(call_kwargs["ServerlessJobConfig"]["CustomizationTechnique"], "SFT")
        self.assertEqual(call_kwargs["ServerlessJobConfig"]["Peft"], "LORA")
        self.assertIn("InputDataConfig", call_kwargs)
        self.assertEqual(
            call_kwargs["ModelPackageConfig"]["ModelPackageGroupArn"],
            self.mock_group_arn,
        )

    @patch("amzn_nova_forge.manager.runtime_manager.get_hub_content")
    def test_execute_without_data_s3_path(self, mock_hub_content):
        manager = self._create_manager()
        mock_hub_content.return_value = {"HubContentArn": self.mock_hub_content_arn}
        manager.sagemaker_client.create_training_job.return_value = {
            "TrainingJobArn": "arn:aws:sagemaker:us-east-1:123456789012:training-job/test-job"
        }

        recipe = {"run": {"model_type": "amazon.nova-2-lite-v1:0:256k"}}
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(recipe, f)
            recipe_path = f.name

        try:
            job_config = JobConfig(
                job_name="test-job",
                image_uri="",
                recipe_path=recipe_path,
                output_s3_path="s3://output-bucket/output",
                method=TrainingMethod.SFT_LORA,
            )
            manager.execute(job_config)
        finally:
            os.unlink(recipe_path)

        call_kwargs = manager.sagemaker_client.create_training_job.call_args.kwargs
        self.assertNotIn("InputDataConfig", call_kwargs)

    @patch("amzn_nova_forge.manager.runtime_manager.get_hub_content")
    def test_execute_with_kms_key(self, mock_hub_content):
        manager = self._create_manager(kms_key_id="my-kms-key")
        mock_hub_content.return_value = {"HubContentArn": self.mock_hub_content_arn}
        manager.sagemaker_client.create_training_job.return_value = {
            "TrainingJobArn": "arn:aws:sagemaker:us-east-1:123456789012:training-job/test-job"
        }

        recipe = {"run": {"model_type": "amazon.nova-2-lite-v1:0:256k"}}
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(recipe, f)
            recipe_path = f.name

        try:
            job_config = JobConfig(
                job_name="test-job",
                image_uri="",
                recipe_path=recipe_path,
                output_s3_path="s3://output-bucket/output",
                method=TrainingMethod.SFT_LORA,
            )
            manager.execute(job_config)
        finally:
            os.unlink(recipe_path)

        call_kwargs = manager.sagemaker_client.create_training_job.call_args.kwargs
        self.assertEqual(call_kwargs["OutputDataConfig"]["KmsKeyId"], "my-kms-key")

    @patch("amzn_nova_forge.manager.runtime_manager.get_hub_content")
    def test_execute_with_vpc_config(self, mock_hub_content):
        manager = self._create_manager(subnets=["subnet-abc"], security_group_ids=["sg-123"])
        mock_hub_content.return_value = {"HubContentArn": self.mock_hub_content_arn}
        manager.sagemaker_client.create_training_job.return_value = {
            "TrainingJobArn": "arn:aws:sagemaker:us-east-1:123456789012:training-job/test-job"
        }

        recipe = {"run": {"model_type": "amazon.nova-2-lite-v1:0:256k"}}
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(recipe, f)
            recipe_path = f.name

        try:
            job_config = JobConfig(
                job_name="test-job",
                image_uri="",
                recipe_path=recipe_path,
                output_s3_path="s3://output-bucket/output",
                method=TrainingMethod.SFT_LORA,
            )
            manager.execute(job_config)
        finally:
            os.unlink(recipe_path)

        call_kwargs = manager.sagemaker_client.create_training_job.call_args.kwargs
        self.assertEqual(call_kwargs["VpcConfig"]["Subnets"], ["subnet-abc"])
        self.assertEqual(call_kwargs["VpcConfig"]["SecurityGroupIds"], ["sg-123"])

    @patch("amzn_nova_forge.manager.runtime_manager.get_hub_content")
    def test_execute_with_source_model_package_arn(self, mock_hub_content):
        """SourceModelPackageArn is included when model_name_or_path is a SageMaker ARN."""
        manager = self._create_manager()
        mock_hub_content.return_value = {"HubContentArn": self.mock_hub_content_arn}
        manager.sagemaker_client.create_training_job.return_value = {
            "TrainingJobArn": "arn:aws:sagemaker:us-east-1:123456789012:training-job/test-job"
        }
        model_package_arn = "arn:aws:sagemaker:us-east-1:123456789012:model-package/group/1"
        recipe = {
            "run": {
                "model_type": "amazon.nova-2-lite-v1:0:256k",
                "model_name_or_path": model_package_arn,
            }
        }
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(recipe, f)
            recipe_path = f.name
        try:
            job_config = JobConfig(
                job_name="test-job",
                image_uri="",
                recipe_path=recipe_path,
                output_s3_path="s3://output-bucket/output",
                method=TrainingMethod.SFT_LORA,
            )
            manager.execute(job_config)
        finally:
            os.unlink(recipe_path)

        call_kwargs = manager.sagemaker_client.create_training_job.call_args.kwargs
        self.assertEqual(
            call_kwargs["ModelPackageConfig"]["SourceModelPackageArn"],
            model_package_arn,
        )

    @patch("amzn_nova_forge.manager.runtime_manager.get_hub_content")
    @patch("sagemaker.ai_registry.dataset.DataSet")
    def test_execute_benchmark_eval_skips_input_data(self, mock_dataset_cls, mock_hub_content):
        """Built-in benchmark eval tasks skip InputDataConfig even when data_s3_path is set."""
        manager = self._create_manager()
        mock_hub_content.return_value = {"HubContentArn": self.mock_hub_content_arn}
        manager.sagemaker_client.create_training_job.return_value = {
            "TrainingJobArn": "arn:aws:sagemaker:us-east-1:123456789012:training-job/test-job"
        }
        recipe = {
            "run": {"model_type": "amazon.nova-2-lite-v1:0:256k"},
            "evaluation": {"task": "mmlu"},
        }
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(recipe, f)
            recipe_path = f.name
        try:
            job_config = JobConfig(
                job_name="test-eval-job",
                image_uri="",
                recipe_path=recipe_path,
                output_s3_path="s3://output-bucket/output",
                data_s3_path="s3://data-bucket/data.jsonl",
                method=TrainingMethod.EVALUATION,
            )
            manager.execute(job_config)
        finally:
            os.unlink(recipe_path)

        call_kwargs = manager.sagemaker_client.create_training_job.call_args.kwargs
        self.assertNotIn("InputDataConfig", call_kwargs)
        mock_dataset_cls.create.assert_not_called()

    @patch("amzn_nova_forge.manager.runtime_manager.get_hub_content")
    @patch("sagemaker.ai_registry.dataset.DataSet")
    def test_execute_eval_with_sagemaker_arn_includes_source_model_package(
        self, mock_dataset_cls, mock_hub_content
    ):
        """Eval job with a SageMaker ARN sets both ModelPackageGroupArn and SourceModelPackageArn."""
        manager = self._create_manager()
        mock_hub_content.return_value = {"HubContentArn": self.mock_hub_content_arn}
        manager.sagemaker_client.create_training_job.return_value = {
            "TrainingJobArn": "arn:aws:sagemaker:us-east-1:123456789012:training-job/test-job"
        }
        model_package_arn = "arn:aws:sagemaker:us-east-1:123456789012:model-package/group/1"
        recipe = {
            "run": {
                "model_type": "amazon.nova-2-lite-v1:0:256k",
                "model_name_or_path": model_package_arn,
            },
            "evaluation": {"task": "mmlu"},
        }
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(recipe, f)
            recipe_path = f.name
        try:
            job_config = JobConfig(
                job_name="test-eval-job",
                image_uri="",
                recipe_path=recipe_path,
                output_s3_path="s3://output-bucket/output",
                method=TrainingMethod.EVALUATION,
            )
            manager.execute(job_config)
        finally:
            os.unlink(recipe_path)

        call_kwargs = manager.sagemaker_client.create_training_job.call_args.kwargs
        self.assertIn("ModelPackageConfig", call_kwargs)
        self.assertEqual(
            call_kwargs["ModelPackageConfig"]["SourceModelPackageArn"],
            model_package_arn,
        )
        self.assertIn("ModelPackageGroupArn", call_kwargs["ModelPackageConfig"])
        self.assertEqual(
            call_kwargs["ModelPackageConfig"]["ModelPackageGroupArn"],
            manager.model_package_group_arn,
        )

    @patch("amzn_nova_forge.manager.runtime_manager.get_hub_content")
    @patch("sagemaker.ai_registry.dataset.DataSet")
    def test_execute_eval_without_sagemaker_arn_omits_model_package_config(
        self, mock_dataset_cls, mock_hub_content
    ):
        """Eval job without a SageMaker ARN omits ModelPackageConfig entirely."""
        manager = self._create_manager()
        mock_hub_content.return_value = {"HubContentArn": self.mock_hub_content_arn}
        manager.sagemaker_client.create_training_job.return_value = {
            "TrainingJobArn": "arn:aws:sagemaker:us-east-1:123456789012:training-job/test-job"
        }
        recipe = {
            "run": {"model_type": "amazon.nova-2-lite-v1:0:256k"},
            "evaluation": {"task": "mmlu"},
        }
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(recipe, f)
            recipe_path = f.name
        try:
            job_config = JobConfig(
                job_name="test-eval-job",
                image_uri="",
                recipe_path=recipe_path,
                output_s3_path="s3://output-bucket/output",
                method=TrainingMethod.EVALUATION,
            )
            manager.execute(job_config)
        finally:
            os.unlink(recipe_path)

        call_kwargs = manager.sagemaker_client.create_training_job.call_args.kwargs
        self.assertNotIn("ModelPackageConfig", call_kwargs)

    @patch("amzn_nova_forge.manager.runtime_manager.get_hub_content")
    @patch("sagemaker.ai_registry.dataset.DataSet")
    def test_execute_hub_content_arn_skips_registration(self, mock_dataset_cls, mock_hub_content):
        """Hub-content ARN passed directly as rft_lambda bypasses ImportHubContent."""
        hub_arn = (
            "arn:aws:sagemaker:us-east-1:123456789012:hub-content/my-hub/JsonDoc/my-reward/0.0.1"
        )
        manager = self._create_manager()
        manager.rft_lambda = hub_arn  # set hub-content ARN directly
        mock_hub_content.return_value = {"HubContentArn": self.mock_hub_content_arn}
        manager.sagemaker_client.create_training_job.return_value = {
            "TrainingJobArn": "arn:aws:sagemaker:us-east-1:123456789012:training-job/test-job"
        }
        mock_dataset = MagicMock()
        mock_dataset.arn = "arn:aws:sagemaker:us-east-1:123456789012:dataset/test"
        mock_dataset_cls.create.return_value = mock_dataset

        recipe = {"run": {"model_type": "amazon.nova-2-lite-v1:0:256k"}}
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(recipe, f)
            recipe_path = f.name
        try:
            job_config = JobConfig(
                job_name="test-job",
                image_uri="",
                recipe_path=recipe_path,
                output_s3_path="s3://output-bucket/output",
                data_s3_path="s3://data-bucket/data.jsonl",
                method=TrainingMethod.RFT_LORA,
            )
            manager.execute(job_config)
        finally:
            os.unlink(recipe_path)

        # ImportHubContent must NOT be called
        manager.sagemaker_client.import_hub_content.assert_not_called()
        call_kwargs = manager.sagemaker_client.create_training_job.call_args.kwargs
        self.assertEqual(call_kwargs["ServerlessJobConfig"]["EvaluatorArn"], hub_arn)

    @patch("amzn_nova_forge.manager.runtime_manager.get_hub_content")
    @patch("sagemaker.ai_registry.dataset.DataSet")
    def test_execute_rft_training_passes_evaluator_arn(self, mock_dataset_cls, mock_hub_content):
        """Lambda ARN is auto-registered as hub-content and passed as EvaluatorArn for RFT training."""
        manager = self._create_manager()
        mock_hub_content.return_value = {"HubContentArn": self.mock_hub_content_arn}
        manager.sagemaker_client.create_training_job.return_value = {
            "TrainingJobArn": "arn:aws:sagemaker:us-east-1:123456789012:training-job/test-job"
        }
        lambda_arn = "arn:aws:lambda:us-east-1:123456789012:function:my-reward"
        hub_arn = "arn:aws:sagemaker:us-east-1:123456789012:hub-content/test-model-package-group/JsonDoc/my-reward/1.0.0"
        mock_dataset = MagicMock()
        mock_dataset.arn = "arn:aws:sagemaker:us-east-1:123456789012:dataset/test"
        mock_dataset_cls.create.return_value = mock_dataset

        # Mock hub operations
        manager.sagemaker_client.exceptions.ResourceNotFound = type(
            "ResourceNotFound", (Exception,), {}
        )
        manager.sagemaker_client.exceptions.ResourceInUse = type("ResourceInUse", (Exception,), {})
        manager.sagemaker_client.describe_hub.side_effect = (
            manager.sagemaker_client.exceptions.ResourceNotFound()
        )
        manager.sagemaker_client.create_hub.return_value = {}
        manager.sagemaker_client.import_hub_content.return_value = {"HubContentArn": hub_arn}

        recipe = {
            "run": {
                "model_type": "amazon.nova-2-lite-v1:0:256k",
                "reward_lambda_arn": lambda_arn,
            },
        }
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(recipe, f)
            recipe_path = f.name
        try:
            job_config = JobConfig(
                job_name="test-rft-train-job",
                image_uri="",
                recipe_path=recipe_path,
                output_s3_path="s3://output-bucket/output",
                data_s3_path="s3://data-bucket/data.jsonl",
                method=TrainingMethod.RFT_LORA,
            )
            manager.execute(job_config)
        finally:
            os.unlink(recipe_path)

        # Verify Lambda was registered as hub-content
        manager.sagemaker_client.import_hub_content.assert_called_once()
        import_kwargs = manager.sagemaker_client.import_hub_content.call_args.kwargs
        self.assertIn(lambda_arn, import_kwargs["HubContentDocument"])
        self.assertEqual(import_kwargs["HubContentType"], "JsonDoc")

        # Verify hub-content ARN was passed as EvaluatorArn
        call_kwargs = manager.sagemaker_client.create_training_job.call_args.kwargs
        self.assertEqual(call_kwargs["ServerlessJobConfig"]["EvaluatorArn"], hub_arn)
        self.assertEqual(call_kwargs["ServerlessJobConfig"]["CustomizationTechnique"], "RLVR")

    @patch("amzn_nova_forge.manager.runtime_manager.get_hub_content")
    @patch("sagemaker.ai_registry.dataset.DataSet")
    def test_execute_rft_eval_sets_lambda_arn_and_type_in_hyperparams(
        self, mock_dataset_cls, mock_hub_content
    ):
        """RFT eval sets lambda_arn + lambda_type='rft' in HyperParameters, not EvaluatorArn."""
        manager = self._create_manager()
        mock_hub_content.return_value = {"HubContentArn": self.mock_hub_content_arn}
        manager.sagemaker_client.create_training_job.return_value = {
            "TrainingJobArn": "arn:aws:sagemaker:us-east-1:123456789012:training-job/test-job"
        }
        lambda_arn = "arn:aws:lambda:us-east-1:123456789012:function:my-reward"
        mock_dataset = MagicMock()
        mock_dataset.arn = "arn:aws:sagemaker:us-east-1:123456789012:dataset/test"
        mock_dataset_cls.create.return_value = mock_dataset

        recipe = {
            "run": {"model_type": "amazon.nova-2-lite-v1:0:256k"},
            "evaluation": {"task": "rft_eval"},
            "rl_env": {"reward_lambda_arn": lambda_arn},
        }
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(recipe, f)
            recipe_path = f.name
        try:
            job_config = JobConfig(
                job_name="test-rft-eval-job",
                image_uri="",
                recipe_path=recipe_path,
                output_s3_path="s3://output-bucket/output",
                data_s3_path="s3://data-bucket/data.jsonl",
                method=TrainingMethod.EVALUATION,
            )
            manager.execute(job_config)
        finally:
            os.unlink(recipe_path)

        call_kwargs = manager.sagemaker_client.create_training_job.call_args.kwargs
        # lambda_arn and lambda_type must be in HyperParameters
        self.assertEqual(call_kwargs["HyperParameters"]["lambda_arn"], lambda_arn)
        self.assertEqual(call_kwargs["HyperParameters"]["lambda_type"], "rft")
        # EvaluatorArn must NOT be in ServerlessJobConfig for eval
        self.assertNotIn("EvaluatorArn", call_kwargs["ServerlessJobConfig"])
        self.assertEqual(
            call_kwargs["ServerlessJobConfig"]["EvaluationType"],
            "CustomScorerEvaluation",
        )

    @patch("amzn_nova_forge.manager.runtime_manager.get_hub_content")
    @patch("sagemaker.ai_registry.dataset.DataSet")
    def test_execute_rft_eval_lambda_arn_not_set_as_evaluator_arn(
        self, mock_dataset_cls, mock_hub_content
    ):
        """Lambda ARNs in rl_env.reward_lambda_arn must NOT become EvaluatorArn."""
        manager = self._create_manager()
        mock_hub_content.return_value = {"HubContentArn": self.mock_hub_content_arn}
        manager.sagemaker_client.create_training_job.return_value = {
            "TrainingJobArn": "arn:aws:sagemaker:us-east-1:123456789012:training-job/test-job"
        }
        lambda_arn = "arn:aws:lambda:us-east-1:123456789012:function:my-reward"
        mock_dataset = MagicMock()
        mock_dataset.arn = "arn:aws:sagemaker:us-east-1:123456789012:dataset/test"
        mock_dataset_cls.create.return_value = mock_dataset

        recipe = {
            "run": {"model_type": "amazon.nova-2-lite-v1:0:256k"},
            "evaluation": {"task": "rft_eval"},
            "rl_env": {"reward_lambda_arn": lambda_arn},
        }
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(recipe, f)
            recipe_path = f.name
        try:
            job_config = JobConfig(
                job_name="test-rft-eval-job",
                image_uri="",
                recipe_path=recipe_path,
                output_s3_path="s3://output-bucket/output",
                data_s3_path="s3://data-bucket/data.jsonl",
                method=TrainingMethod.EVALUATION,
            )
            manager.execute(job_config)
        finally:
            os.unlink(recipe_path)

        call_kwargs = manager.sagemaker_client.create_training_job.call_args.kwargs
        self.assertNotIn("EvaluatorArn", call_kwargs["ServerlessJobConfig"])

    @patch("amzn_nova_forge.manager.runtime_manager.get_hub_content")
    @patch("sagemaker.ai_registry.dataset.DataSet")
    def test_execute_dataset_name_truncated_to_63_chars(self, mock_dataset_cls, mock_hub_content):
        """Dataset name is truncated so job_name[:51] + '-train-input' stays within 63 chars."""
        manager = self._create_manager()
        mock_hub_content.return_value = {"HubContentArn": self.mock_hub_content_arn}
        manager.sagemaker_client.create_training_job.return_value = {
            "TrainingJobArn": "arn:aws:sagemaker:us-east-1:123456789012:training-job/test-job"
        }
        mock_dataset = MagicMock()
        mock_dataset.arn = "arn:aws:sagemaker:us-east-1:123456789012:dataset/test"
        mock_dataset_cls.create.return_value = mock_dataset

        long_job_name = "a" * 63  # max job name length
        recipe = {"run": {"model_type": "amazon.nova-2-lite-v1:0:256k"}}
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(recipe, f)
            recipe_path = f.name
        try:
            job_config = JobConfig(
                job_name=long_job_name,
                image_uri="",
                recipe_path=recipe_path,
                output_s3_path="s3://output-bucket/output",
                data_s3_path="s3://data-bucket/data.jsonl",
                method=TrainingMethod.SFT_LORA,
            )
            manager.execute(job_config)
        finally:
            os.unlink(recipe_path)

        dataset_name_used = mock_dataset_cls.create.call_args[0][0]
        self.assertLessEqual(len(dataset_name_used), 63)
        self.assertTrue(dataset_name_used.endswith("-train-input"))

    @patch("amzn_nova_forge.manager.runtime_manager.get_hub_content")
    def test_execute_with_mlflow_config(self, mock_hub_content):
        manager = self._create_manager()
        mock_hub_content.return_value = {"HubContentArn": self.mock_hub_content_arn}
        manager.sagemaker_client.create_training_job.return_value = {
            "TrainingJobArn": "arn:aws:sagemaker:us-east-1:123456789012:training-job/test-job"
        }

        recipe = {
            "run": {
                "model_type": "amazon.nova-2-lite-v1:0:256k",
                "mlflow_tracking_uri": "arn:aws:sagemaker:us-east-1:123456789012:mlflow/my-server",
                "mlflow_experiment_name": "my-exp",
                "mlflow_run_name": "my-run",
            }
        }
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(recipe, f)
            recipe_path = f.name

        try:
            job_config = JobConfig(
                job_name="test-job",
                image_uri="",
                recipe_path=recipe_path,
                output_s3_path="s3://output-bucket/output",
                method=TrainingMethod.SFT_LORA,
            )
            manager.execute(job_config)
        finally:
            os.unlink(recipe_path)

        call_kwargs = manager.sagemaker_client.create_training_job.call_args.kwargs
        self.assertEqual(
            call_kwargs["MlflowConfig"]["MlflowResourceArn"],
            "arn:aws:sagemaker:us-east-1:123456789012:mlflow/my-server",
        )
        self.assertEqual(call_kwargs["MlflowConfig"]["MlflowExperimentName"], "my-exp")
        self.assertEqual(call_kwargs["MlflowConfig"]["MlflowRunName"], "my-run")

    def test_execute_fails_without_output_s3_path(self):
        manager = self._create_manager()
        job_config = JobConfig(
            job_name="test-job",
            image_uri="",
            recipe_path="/tmp/recipe.yaml",
            output_s3_path=None,
            method=TrainingMethod.SFT_LORA,
        )
        with self.assertRaises(AssertionError):
            manager.execute(job_config)

    def test_execute_fails_without_method(self):
        manager = self._create_manager()
        job_config = JobConfig(
            job_name="test-job",
            image_uri="",
            recipe_path="/tmp/recipe.yaml",
            output_s3_path="s3://bucket/output",
            method=None,
        )
        with self.assertRaises(AssertionError):
            manager.execute(job_config)

    @patch("amzn_nova_forge.manager.runtime_manager.get_hub_content")
    def test_execute_raises_on_api_error(self, mock_hub_content):
        manager = self._create_manager()
        mock_hub_content.return_value = {"HubContentArn": self.mock_hub_content_arn}
        manager.sagemaker_client.create_training_job.side_effect = Exception("API error")

        recipe = {"run": {"model_type": "amazon.nova-2-lite-v1:0:256k"}}
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(recipe, f)
            recipe_path = f.name

        try:
            job_config = JobConfig(
                job_name="test-job",
                image_uri="",
                recipe_path=recipe_path,
                output_s3_path="s3://bucket/output",
                method=TrainingMethod.SFT_LORA,
            )
            with self.assertRaises(Exception) as ctx:
                manager.execute(job_config)
            self.assertEqual(str(ctx.exception), "API error")
        finally:
            os.unlink(recipe_path)

    def test_cleanup_success(self):
        manager = self._create_manager()
        manager.cleanup("test-job")
        manager.sagemaker_client.stop_training_job.assert_called_once_with(
            TrainingJobName="test-job"
        )
        manager.sagemaker_client.close.assert_called_once()

    def test_cleanup_is_mtrl_false_calls_stop_training_job(self):
        manager = self._create_manager()
        manager.cleanup("test-job", is_mtrl=False)
        manager.sagemaker_client.stop_training_job.assert_called_once_with(
            TrainingJobName="test-job"
        )
        manager.sagemaker_client.close.assert_called_once()

    @patch.object(SMTJServerlessRuntimeManager, "_cleanup_mtrl")
    def test_cleanup_is_mtrl_true_calls_cleanup_mtrl(self, mock_cleanup_mtrl):
        manager = self._create_manager()
        manager.cleanup("mtrl-job-123", is_mtrl=True)
        mock_cleanup_mtrl.assert_called_once_with("mtrl-job-123")
        manager.sagemaker_client.stop_training_job.assert_not_called()

    @patch.object(SMTJServerlessRuntimeManager, "_cleanup_mtrl")
    def test_cleanup_is_mtrl_true_raises_on_error(self, mock_cleanup_mtrl):
        manager = self._create_manager()
        mock_cleanup_mtrl.side_effect = Exception("MTRL cleanup failed")
        with self.assertRaises(Exception) as ctx:
            manager.cleanup("mtrl-job-123", is_mtrl=True)
        self.assertEqual(str(ctx.exception), "MTRL cleanup failed")

    def test_cleanup_raises_on_error(self):
        manager = self._create_manager()
        manager.sagemaker_client.stop_training_job.side_effect = Exception("Cleanup failed")
        with self.assertRaises(Exception) as ctx:
            manager.cleanup("test-job")
        self.assertEqual(str(ctx.exception), "Cleanup failed")

    def test_required_calling_role_permissions(self):
        perms = SMTJServerlessRuntimeManager.required_calling_role_permissions(
            data_s3_path="s3://input/data.jsonl",
            output_s3_path="s3://output/results",
        )
        perm_actions = [p[0] if isinstance(p, tuple) else p for p in perms]
        self.assertIn("sagemaker:CreateTrainingJob", perm_actions)
        self.assertIn("sagemaker:DescribeTrainingJob", perm_actions)
        self.assertIn("iam:PassRole", perm_actions)
        self.assertIn("s3:GetObject", perm_actions)


class TestSMTJServerlessValidationDataset(unittest.TestCase):
    """Tests for SFT validation dataset support in SMTJServerlessRuntimeManager."""

    def setUp(self):
        self.mock_role = "arn:aws:iam::123456789012:role/SageMakerExecutionRole"
        self.model_package_group_name = "test-model-package-group"
        self.mock_group_arn = (
            "arn:aws:sagemaker:us-east-1:123456789012:model-package-group/test-model-package-group"
        )
        self.mock_hub_content_arn = (
            "arn:aws:sagemaker:us-east-1:123456789012:"
            "hub-content/SageMakerPublicHub/Model/nova-textgeneration-lite-v2/1.0.0"
        )

    @patch.object(SMTJServerlessRuntimeManager, "setup", return_value=None)
    def _create_manager(self, mock_setup, **kwargs):
        manager = SMTJServerlessRuntimeManager(
            model_package_group_name=self.model_package_group_name, **kwargs
        )
        manager.execution_role = self.mock_role
        manager.sagemaker_client = MagicMock()
        manager.sagemaker_session = MagicMock()
        manager.region = "us-east-1"
        manager.model_package_group_arn = self.mock_group_arn
        return manager

    @patch("amzn_nova_forge.manager.runtime_manager.get_hub_content")
    @patch("sagemaker.ai_registry.dataset.DataSet")
    def test_serverless_sft_with_validation_data(self, mock_dataset_cls, mock_hub_content):
        manager = self._create_manager()
        mock_hub_content.return_value = {"HubContentArn": self.mock_hub_content_arn}

        mock_dataset = MagicMock()
        mock_dataset.arn = "arn:aws:sagemaker:us-east-1:123456789012:dataset/train-input"
        mock_val_dataset = MagicMock()
        mock_val_dataset.arn = "arn:aws:sagemaker:us-east-1:123456789012:dataset/val-input"
        mock_dataset_cls.create.side_effect = [mock_dataset, mock_val_dataset]

        manager.sagemaker_client.create_training_job.return_value = {
            "TrainingJobArn": "arn:aws:sagemaker:us-east-1:123456789012:training-job/test-job"
        }

        recipe = {
            "run": {"model_type": "amazon.nova-2-lite-v1:0:256k"},
            "trainer": {"lr": 0.001},
        }
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(recipe, f)
            recipe_path = f.name

        try:
            job_config = JobConfig(
                job_name="test-serverless-val-job",
                image_uri="",
                recipe_path=recipe_path,
                output_s3_path="s3://output-bucket/output",
                data_s3_path="s3://input-bucket/train.jsonl",
                validation_data_s3_path="s3://input-bucket/val.jsonl",
                method=TrainingMethod.SFT_LORA,
            )
            manager.execute(job_config)
        finally:
            os.unlink(recipe_path)

        call_kwargs = manager.sagemaker_client.create_training_job.call_args.kwargs
        self.assertIn("InputDataConfig", call_kwargs)
        channel_names = [ch["ChannelName"] for ch in call_kwargs["InputDataConfig"]]
        self.assertIn("train", channel_names)
        self.assertIn("validation", channel_names)
        self.assertEqual(len(call_kwargs["InputDataConfig"]), 2)

    @patch("amzn_nova_forge.manager.runtime_manager.get_hub_content")
    @patch("sagemaker.ai_registry.dataset.DataSet")
    def test_serverless_sft_without_validation_data(self, mock_dataset_cls, mock_hub_content):
        manager = self._create_manager()
        mock_hub_content.return_value = {"HubContentArn": self.mock_hub_content_arn}

        mock_dataset = MagicMock()
        mock_dataset.arn = "arn:aws:sagemaker:us-east-1:123456789012:dataset/train-input"
        mock_dataset_cls.create.return_value = mock_dataset

        manager.sagemaker_client.create_training_job.return_value = {
            "TrainingJobArn": "arn:aws:sagemaker:us-east-1:123456789012:training-job/test-job"
        }

        recipe = {
            "run": {"model_type": "amazon.nova-2-lite-v1:0:256k"},
            "trainer": {"lr": 0.001},
        }
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(recipe, f)
            recipe_path = f.name

        try:
            job_config = JobConfig(
                job_name="test-serverless-no-val-job",
                image_uri="",
                recipe_path=recipe_path,
                output_s3_path="s3://output-bucket/output",
                data_s3_path="s3://input-bucket/train.jsonl",
                method=TrainingMethod.SFT_LORA,
            )
            manager.execute(job_config)
        finally:
            os.unlink(recipe_path)

        call_kwargs = manager.sagemaker_client.create_training_job.call_args.kwargs
        self.assertIn("InputDataConfig", call_kwargs)
        channel_names = [ch["ChannelName"] for ch in call_kwargs["InputDataConfig"]]
        self.assertIn("train", channel_names)
        self.assertNotIn("validation", channel_names)

    @patch("amzn_nova_forge.manager.runtime_manager.get_hub_content")
    @patch("sagemaker.ai_registry.dataset.DataSet")
    def test_serverless_validation_dataset_name_truncation(
        self, mock_dataset_cls, mock_hub_content
    ):
        manager = self._create_manager()
        mock_hub_content.return_value = {"HubContentArn": self.mock_hub_content_arn}

        mock_dataset = MagicMock()
        mock_dataset.arn = "arn:aws:sagemaker:us-east-1:123456789012:dataset/train-input"
        mock_val_dataset = MagicMock()
        mock_val_dataset.arn = "arn:aws:sagemaker:us-east-1:123456789012:dataset/val-input"
        mock_dataset_cls.create.side_effect = [mock_dataset, mock_val_dataset]

        manager.sagemaker_client.create_training_job.return_value = {
            "TrainingJobArn": "arn:aws:sagemaker:us-east-1:123456789012:training-job/test-job"
        }

        long_job_name = "a" * 63  # max job name length
        recipe = {"run": {"model_type": "amazon.nova-2-lite-v1:0:256k"}}
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(recipe, f)
            recipe_path = f.name

        try:
            job_config = JobConfig(
                job_name=long_job_name,
                image_uri="",
                recipe_path=recipe_path,
                output_s3_path="s3://output-bucket/output",
                data_s3_path="s3://data-bucket/data.jsonl",
                validation_data_s3_path="s3://data-bucket/val.jsonl",
                method=TrainingMethod.SFT_LORA,
            )
            manager.execute(job_config)
        finally:
            os.unlink(recipe_path)

        # Second DataSet.create call is for validation dataset
        val_dataset_name = mock_dataset_cls.create.call_args_list[1][0][0]
        self.assertLessEqual(len(val_dataset_name), 63)

    @patch("amzn_nova_forge.manager.runtime_manager.get_hub_content")
    @patch("sagemaker.ai_registry.dataset.DataSet")
    def test_serverless_val_check_interval_fallback_merge(self, mock_dataset_cls, mock_hub_content):
        manager = self._create_manager()
        mock_hub_content.return_value = {"HubContentArn": self.mock_hub_content_arn}

        mock_dataset = MagicMock()
        mock_dataset.arn = "arn:aws:sagemaker:us-east-1:123456789012:dataset/train-input"
        mock_dataset_cls.create.return_value = mock_dataset

        manager.sagemaker_client.create_training_job.return_value = {
            "TrainingJobArn": "arn:aws:sagemaker:us-east-1:123456789012:training-job/test-job"
        }

        # Recipe template WITHOUT val_check_interval
        recipe = {
            "run": {"model_type": "amazon.nova-2-lite-v1:0:256k"},
            "trainer": {"lr": 0.001},
        }
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(recipe, f)
            recipe_path = f.name

        try:
            job_config = JobConfig(
                job_name="test-val-interval-job",
                image_uri="",
                recipe_path=recipe_path,
                output_s3_path="s3://output-bucket/output",
                data_s3_path="s3://input-bucket/train.jsonl",
                method=TrainingMethod.SFT_LORA,
                trainer_config_hyperparameters={"val_check_interval": "500"},
            )
            manager.execute(job_config)
        finally:
            os.unlink(recipe_path)

        call_kwargs = manager.sagemaker_client.create_training_job.call_args.kwargs
        self.assertIn("val_check_interval", call_kwargs["HyperParameters"])
        self.assertEqual(call_kwargs["HyperParameters"]["val_check_interval"], "500")


class TestMethodToServerlessConfig(unittest.TestCase):
    def test_all_expected_methods_present(self):
        expected = {
            TrainingMethod.SFT_LORA,
            TrainingMethod.SFT_FULL,
            TrainingMethod.DPO_LORA,
            TrainingMethod.DPO_FULL,
            TrainingMethod.RFT_LORA,
            TrainingMethod.RFT_MULTITURN_LORA,
            # TrainingMethod.RFT_FULL,
        }
        self.assertEqual(set(_METHOD_TO_SERVERLESS_CONFIG.keys()), expected)

    def test_lora_methods_have_peft(self):
        for method in (
            TrainingMethod.SFT_LORA,
            TrainingMethod.DPO_LORA,
            TrainingMethod.RFT_LORA,
            TrainingMethod.RFT_MULTITURN_LORA,
        ):
            _, peft = _METHOD_TO_SERVERLESS_CONFIG[method]
            self.assertEqual(peft, "LORA")

    def test_full_methods_have_no_peft(self):
        for method in (
            TrainingMethod.SFT_FULL,
            TrainingMethod.DPO_FULL,
            # TrainingMethod.RFT_FULL,
        ):
            _, peft = _METHOD_TO_SERVERLESS_CONFIG[method]
            self.assertIsNone(peft)


class TestIsHubContentArn(unittest.TestCase):
    """Tests for _is_hub_content_arn module-level helper."""

    def test_valid_hub_content_arn(self):
        from amzn_nova_forge.manager.runtime_manager import _is_hub_content_arn

        arn = "arn:aws:sagemaker:us-east-1:123456789012:hub-content/my-hub/JsonDoc/my-reward/0.0.1"
        self.assertTrue(_is_hub_content_arn(arn))

    def test_lambda_arn_is_not_hub_content(self):
        from amzn_nova_forge.manager.runtime_manager import _is_hub_content_arn

        self.assertFalse(
            _is_hub_content_arn("arn:aws:lambda:us-east-1:123456789012:function:my-fn")
        )

    def test_model_package_arn_is_not_hub_content(self):
        from amzn_nova_forge.manager.runtime_manager import _is_hub_content_arn

        self.assertFalse(
            _is_hub_content_arn("arn:aws:sagemaker:us-east-1:123456789012:model-package/group/1")
        )

    def test_none_returns_false(self):
        from amzn_nova_forge.manager.runtime_manager import _is_hub_content_arn

        self.assertFalse(_is_hub_content_arn(None))

    def test_empty_string_returns_false(self):
        from amzn_nova_forge.manager.runtime_manager import _is_hub_content_arn

        self.assertFalse(_is_hub_content_arn(""))

    def test_missing_version_returns_false(self):
        from amzn_nova_forge.manager.runtime_manager import _is_hub_content_arn

        self.assertFalse(
            _is_hub_content_arn(
                "arn:aws:sagemaker:us-east-1:123456789012:hub-content/my-hub/JsonDoc/my-reward"
            )
        )


class TestIsHubContentArnValidator(unittest.TestCase):
    """Tests for is_hub_content_arn in validator.py."""

    def test_valid_hub_content_arn(self):
        from amzn_nova_forge.validation.validator import is_hub_content_arn

        arn = "arn:aws:sagemaker:us-east-1:123456789012:hub-content/my-hub/JsonDoc/my-reward/0.0.1"
        self.assertTrue(is_hub_content_arn(arn))

    def test_lambda_arn_returns_false(self):
        from amzn_nova_forge.validation.validator import is_hub_content_arn

        self.assertFalse(is_hub_content_arn("arn:aws:lambda:us-east-1:123456789012:function:fn"))

    def test_none_returns_false(self):
        from amzn_nova_forge.validation.validator import is_hub_content_arn

        self.assertFalse(is_hub_content_arn(None))

    def test_placeholder_returns_false(self):
        from amzn_nova_forge.validation.validator import is_hub_content_arn

        self.assertFalse(
            is_hub_content_arn("arn:aws:sagemaker:<region>:<account>:hub-content/h/T/n/0.0.1")
        )


class TestValidateRftWithHubContentArn(unittest.TestCase):
    """Tests for validate_rft accepting hub-content ARNs."""

    @patch("amzn_nova_forge.validation.validator.boto3.client")
    def test_hub_content_arn_passes_validation(self, mock_boto3_client):
        from unittest.mock import Mock

        from amzn_nova_forge.manager.runtime_manager import SMTJServerlessRuntimeManager
        from amzn_nova_forge.validation.validator import Validator

        mock_infra = Mock(spec=SMTJServerlessRuntimeManager)
        mock_infra.instance_type = None
        mock_infra.region = "us-east-1"
        hub_arn = (
            "arn:aws:sagemaker:us-east-1:123456789012:hub-content/my-hub/JsonDoc/my-reward/0.0.1"
        )

        # Hub-content ARN must not trigger a hub-content/rft_lambda_arn validation error.
        # Other unrelated errors (e.g. missing recipe fields) are acceptable.
        try:
            Validator.validate(
                platform=Platform.SMTJServerless,
                method=TrainingMethod.RFT_LORA,
                infra=mock_infra,
                recipe={},
                overrides_template={},
                validation_config=ValidationConfig(iam=False, infra=False),
                rft_lambda_arn=hub_arn,
            )
        except ValueError as e:
            self.assertNotIn(
                "hub-content",
                str(e).lower(),
                f"Hub-content ARN should be valid but got: {e}",
            )
            self.assertNotIn(
                "placeholder",
                str(e).lower(),
                f"Hub-content ARN should be valid but got: {e}",
            )
            self.assertNotIn(
                "rft_lambda_arn",
                str(e).lower(),
                f"Hub-content ARN should be valid but got: {e}",
            )


class TestRegisterLambdaAsHubContent(unittest.TestCase):
    """Tests for register_lambda_as_hub_content in sagemaker.py."""

    def _make_client(self):
        client = MagicMock()
        client.exceptions.ResourceNotFound = type("ResourceNotFound", (Exception,), {})
        client.exceptions.ResourceInUse = type("ResourceInUse", (Exception,), {})
        return client

    def test_registers_new_hub_content(self):
        import json

        from amzn_nova_forge.util.sagemaker import register_lambda_as_hub_content

        client = self._make_client()
        client.describe_hub.return_value = {}  # hub exists
        hub_arn = "arn:aws:sagemaker:us-east-1:123456789012:hub-content/my-hub/JsonDoc/my-fn/0.0.1"
        client.import_hub_content.return_value = {"HubContentArn": hub_arn}

        result = register_lambda_as_hub_content(
            lambda_arn="arn:aws:lambda:us-east-1:123456789012:function:my-fn",
            hub_name="my-hub",
            sagemaker_client=client,
        )

        self.assertEqual(result, hub_arn)
        client.import_hub_content.assert_called_once()
        call_kwargs = client.import_hub_content.call_args.kwargs
        self.assertEqual(call_kwargs["HubContentType"], "JsonDoc")
        self.assertEqual(call_kwargs["DocumentSchemaVersion"], "2.0.0")
        doc = json.loads(call_kwargs["HubContentDocument"])
        inner = json.loads(doc["JsonContent"])
        self.assertEqual(inner["EvaluatorType"], "RewardFunction")
        self.assertEqual(inner["Reference"], "arn:aws:lambda:us-east-1:123456789012:function:my-fn")

    def test_creates_hub_if_not_exists(self):
        from amzn_nova_forge.util.sagemaker import register_lambda_as_hub_content

        client = self._make_client()
        client.describe_hub.side_effect = client.exceptions.ResourceNotFound()
        client.create_hub.return_value = {}
        client.import_hub_content.return_value = {
            "HubContentArn": "arn:aws:sagemaker:us-east-1:123456789012:hub-content/h/JsonDoc/fn/0.0.1"
        }

        register_lambda_as_hub_content(
            lambda_arn="arn:aws:lambda:us-east-1:123456789012:function:fn",
            hub_name="new-hub",
            sagemaker_client=client,
        )

        client.create_hub.assert_called_once_with(
            HubName="new-hub",
            HubDescription="Private hub for Nova Forge serverless reward functions",
        )

    def test_reuses_existing_hub_content_same_arn(self):
        import json

        from amzn_nova_forge.util.sagemaker import register_lambda_as_hub_content

        lambda_arn = "arn:aws:lambda:us-east-1:123456789012:function:fn"
        existing_arn = "arn:aws:sagemaker:us-east-1:123456789012:hub-content/h/JsonDoc/fn/0.0.1"
        client = self._make_client()
        client.describe_hub.return_value = {}
        client.import_hub_content.side_effect = client.exceptions.ResourceInUse()
        client.describe_hub_content.return_value = {
            "HubContentArn": existing_arn,
            "HubContentDocument": json.dumps(
                {
                    "SubType": "AWS/Evaluator",
                    "JsonContent": json.dumps(
                        {"EvaluatorType": "RewardFunction", "Reference": lambda_arn}
                    ),
                }
            ),
        }

        result = register_lambda_as_hub_content(
            lambda_arn=lambda_arn,
            hub_name="h",
            sagemaker_client=client,
        )

        self.assertEqual(result, existing_arn)
        # Should not call import_hub_content a second time
        client.import_hub_content.assert_called_once()

    def test_bumps_version_when_lambda_arn_changed(self):
        import json

        from amzn_nova_forge.util.sagemaker import register_lambda_as_hub_content

        old_arn = "arn:aws:lambda:us-east-1:123456789012:function:old-fn"
        new_arn = "arn:aws:lambda:us-east-1:123456789012:function:new-fn"
        new_hub_arn = "arn:aws:sagemaker:us-east-1:123456789012:hub-content/h/JsonDoc/new-fn/0.0.2"
        client = self._make_client()
        client.describe_hub.return_value = {}
        client.import_hub_content.side_effect = [
            client.exceptions.ResourceInUse(),
            {"HubContentArn": new_hub_arn},
        ]
        client.describe_hub_content.return_value = {
            "HubContentArn": "arn:aws:sagemaker:us-east-1:123456789012:hub-content/h/JsonDoc/new-fn/0.0.1",
            "HubContentDocument": json.dumps(
                {
                    "SubType": "AWS/Evaluator",
                    "JsonContent": json.dumps(
                        {"EvaluatorType": "RewardFunction", "Reference": old_arn}
                    ),
                }
            ),
        }

        result = register_lambda_as_hub_content(
            lambda_arn=new_arn,
            hub_name="h",
            sagemaker_client=client,
        )

        self.assertEqual(result, new_hub_arn)
        # Second import_hub_content call should use bumped version
        second_call = client.import_hub_content.call_args_list[1].kwargs
        self.assertEqual(second_call["HubContentVersion"], "0.0.2")

    def test_uses_custom_evaluator_name(self):
        from amzn_nova_forge.util.sagemaker import register_lambda_as_hub_content

        client = self._make_client()
        client.describe_hub.return_value = {}
        client.import_hub_content.return_value = {
            "HubContentArn": "arn:aws:sagemaker:us-east-1:123456789012:hub-content/h/JsonDoc/my-custom-name/0.0.1"
        }

        register_lambda_as_hub_content(
            lambda_arn="arn:aws:lambda:us-east-1:123456789012:function:fn",
            hub_name="h",
            sagemaker_client=client,
            evaluator_name="my-custom-name",
        )

        call_kwargs = client.import_hub_content.call_args.kwargs
        self.assertEqual(call_kwargs["HubContentName"], "my-custom-name")


class TestExtractLambdaArnFromHubContent(unittest.TestCase):
    """Tests for extract_lambda_arn_from_hub_content in sagemaker.py."""

    def test_extracts_lambda_arn(self):
        import json

        from amzn_nova_forge.util.sagemaker import extract_lambda_arn_from_hub_content

        lambda_arn = "arn:aws:lambda:us-east-1:123456789012:function:my-fn"
        client = MagicMock()
        client.describe_hub_content.return_value = {
            "HubContentDocument": json.dumps(
                {
                    "SubType": "AWS/Evaluator",
                    "JsonContent": json.dumps(
                        {"EvaluatorType": "RewardFunction", "Reference": lambda_arn}
                    ),
                }
            )
        }

        result = extract_lambda_arn_from_hub_content(
            "arn:aws:sagemaker:us-east-1:123456789012:hub-content/my-hub/JsonDoc/my-reward/0.0.1",
            client,
        )

        self.assertEqual(result, lambda_arn)

    def test_returns_none_on_api_error(self):
        from amzn_nova_forge.util.sagemaker import extract_lambda_arn_from_hub_content

        client = MagicMock()
        client.describe_hub_content.side_effect = Exception("Not found")

        result = extract_lambda_arn_from_hub_content(
            "arn:aws:sagemaker:us-east-1:123456789012:hub-content/my-hub/JsonDoc/my-reward/0.0.1",
            client,
        )

        self.assertIsNone(result)

    def test_returns_none_when_reference_missing(self):
        import json

        from amzn_nova_forge.util.sagemaker import extract_lambda_arn_from_hub_content

        client = MagicMock()
        client.describe_hub_content.return_value = {
            "HubContentDocument": json.dumps(
                {
                    "SubType": "AWS/Evaluator",
                    "JsonContent": json.dumps({"EvaluatorType": "RewardFunction"}),
                }
            )
        }

        result = extract_lambda_arn_from_hub_content(
            "arn:aws:sagemaker:us-east-1:123456789012:hub-content/my-hub/JsonDoc/my-reward/0.0.1",
            client,
        )

        self.assertIsNone(result)


class TestExtractHyperparameters(unittest.TestCase):
    """Tests for SMTJServerlessRuntimeManager._extract_hyperparameters."""

    @patch.object(SMTJServerlessRuntimeManager, "setup", return_value=None)
    def setUp(self, mock_setup):
        self.manager = SMTJServerlessRuntimeManager(model_package_group_name="test-group")

    def test_basic_recipe_extraction(self):
        recipe = {"run": {"name": "my-job", "max_steps": 100, "global_batch_size": 64}}
        result = self.manager._extract_hyperparameters(recipe)
        self.assertEqual(result["name"], "my-job")
        self.assertEqual(result["max_steps"], "100")
        self.assertEqual(result["global_batch_size"], "64")

    def test_datamix_config_injected_as_hyperparameters(self):
        recipe = {"run": {"name": "my-job"}}
        data_mixing_config = {
            "customer_data_percent": 50,
            "nova_code_percent": 60,
            "nova_general_percent": 40,
            "dataset_catalog": "sft_1p5_text_chat",
        }
        result = self.manager._extract_hyperparameters(
            recipe, data_mixing_config=data_mixing_config
        )
        self.assertEqual(result["customer_data_percent"], "50")
        self.assertEqual(result["nova_code_percent"], "60")
        self.assertEqual(result["nova_general_percent"], "40")
        self.assertEqual(result["dataset_catalog"], "sft_1p5_text_chat")

    def test_dataset_catalog_included_in_hyperparameters(self):
        """dataset_catalog must be passed as a hyperparameter."""
        recipe = {}
        data_mixing_config = {"dataset_catalog": "sft_1p5_text_chat", "customer_data_percent": 80}
        result = self.manager._extract_hyperparameters(
            recipe, data_mixing_config=data_mixing_config
        )
        self.assertIn("dataset_catalog", result)
        self.assertEqual(result["dataset_catalog"], "sft_1p5_text_chat")

    def test_no_datamix_config_does_not_error(self):
        recipe = {"run": {"name": "my-job", "max_steps": 10}}
        result = self.manager._extract_hyperparameters(recipe, data_mixing_config=None)
        self.assertEqual(result["name"], "my-job")
        self.assertNotIn("customer_data_percent", result)

    def test_empty_datamix_config_does_not_error(self):
        recipe = {"run": {"name": "my-job"}}
        result = self.manager._extract_hyperparameters(recipe, data_mixing_config={})
        self.assertNotIn("customer_data_percent", result)


class TestSMTJServerlessMTRL(unittest.TestCase):
    """Tests for MTRL operations on SMTJServerlessRuntimeManager."""

    MOCK_AGENT_CORE_ARN = "arn:aws:bedrock-agentcore:us-east-1:123456789012:runtime/my-agent"
    MOCK_LAMBDA_ARN = "arn:aws:lambda:us-east-1:123456789012:function:my-reward-fn"
    MOCK_MODEL_PACKAGE_ARN = "arn:aws:sagemaker:us-east-1:123456789012:model-package/grp/1"

    @patch.object(SMTJServerlessRuntimeManager, "setup", return_value=None)
    def test_init_with_agent_core_arn(self, mock_setup):
        manager = SMTJServerlessRuntimeManager(
            model_package_group_name="test-group",
            agent_core_arn=self.MOCK_AGENT_CORE_ARN,
        )
        self.assertEqual(manager.agent_core_arn, self.MOCK_AGENT_CORE_ARN)

    @patch.object(SMTJServerlessRuntimeManager, "setup", return_value=None)
    def test_init_with_intermediate_model_package_group(self, mock_setup):
        manager = SMTJServerlessRuntimeManager(
            model_package_group_name="test-group",
            intermediate_model_package_group_name="test-checkpoints",
        )
        self.assertEqual(manager.intermediate_model_package_group_name, "test-checkpoints")

    @patch.object(SMTJServerlessRuntimeManager, "setup", return_value=None)
    def test_init_with_rft_lambda(self, mock_setup):
        manager = SMTJServerlessRuntimeManager(
            model_package_group_name="test-group",
            rft_lambda=self.MOCK_LAMBDA_ARN,
        )
        self.assertEqual(manager.rft_lambda, self.MOCK_LAMBDA_ARN)

    def _create_mtrl_ops(self, agent_core_arn=None, rft_lambda=None):
        from amzn_nova_forge.manager.mtrl_manager import MTRLOperations

        ops = MTRLOperations.__new__(MTRLOperations)
        ops.agent_core_arn = agent_core_arn
        ops.rft_lambda = rft_lambda
        ops.model_package_group_arn = (
            "arn:aws:sagemaker:us-east-1:123456789012:model-package-group/grp"
        )
        ops.execution_role = "arn:aws:iam::123456789012:role/role"
        ops.subnets = None
        ops.security_group_ids = None
        ops.kms_key_id = None
        ops.region = "us-east-1"
        return ops

    def _mock_sagemaker_modules(self):
        import types

        mock_sm_train = types.ModuleType("sagemaker.train")
        mock_sm_train_mtrl = types.ModuleType("sagemaker.train.multi_turn_rl_trainer")
        mock_sm_core = types.ModuleType("sagemaker.core")
        mock_sm_core_resources = types.ModuleType("sagemaker.core.resources")

        self._mock_trainer_cls = MagicMock()
        self._mock_trainer_instance = MagicMock()
        self._mock_trainer_cls.return_value = self._mock_trainer_instance
        self._mock_model_package_cls = MagicMock()

        mock_sm_train_mtrl.MultiTurnRLTrainer = self._mock_trainer_cls
        mock_sm_core_resources.ModelPackage = self._mock_model_package_cls

        return {
            "sagemaker.train": mock_sm_train,
            "sagemaker.train.multi_turn_rl_trainer": mock_sm_train_mtrl,
            "sagemaker.core": mock_sm_core,
            "sagemaker.core.resources": mock_sm_core_resources,
        }

    @patch("amzn_nova_forge.manager.mtrl_manager.logger")
    def test_execute_mtrl_with_agent_core(self, _mock_logger):
        ops = self._create_mtrl_ops(agent_core_arn=self.MOCK_AGENT_CORE_ARN)
        modules = self._mock_sagemaker_modules()

        mock_job = MagicMock()
        mock_job.job_name = "test-job-123"
        mock_job.job_arn = "arn:aws:sagemaker:us-east-1:123456789012:job/test-job-123"
        self._mock_trainer_instance.train.return_value = mock_job

        with patch.dict("sys.modules", modules):
            with patch.object(
                ops,
                "_get_or_create_checkpoint_model_package_group_arn",
                return_value="arn:checkpoint",
            ):
                result = ops.execute_mtrl(
                    model=Model.NOVA_LITE_2,
                    job_name="test-job",
                    data_s3_path="s3://bucket/data",
                    output_s3_path="s3://bucket/output",
                )

        self.assertEqual(result, "test-job-123")
        call_kwargs = self._mock_trainer_cls.call_args[1]
        self.assertEqual(call_kwargs["agent_env"], self.MOCK_AGENT_CORE_ARN)
        self.assertEqual(call_kwargs["model"], Model.NOVA_LITE_2.hub_content_name)

    @patch("amzn_nova_forge.manager.mtrl_manager.logger")
    def test_execute_mtrl_with_lambda(self, _mock_logger):
        ops = self._create_mtrl_ops(rft_lambda=self.MOCK_LAMBDA_ARN)
        modules = self._mock_sagemaker_modules()

        mock_job = MagicMock()
        mock_job.job_name = "lambda-job"
        mock_job.job_arn = "arn:aws:sagemaker:us-east-1:123456789012:job/lambda-job"
        self._mock_trainer_instance.train.return_value = mock_job

        with patch.dict("sys.modules", modules):
            with patch.object(
                ops,
                "_get_or_create_checkpoint_model_package_group_arn",
                return_value="arn:checkpoint",
            ):
                result = ops.execute_mtrl(
                    model=Model.NOVA_LITE_2,
                    job_name="lambda-job",
                    data_s3_path="s3://bucket/data",
                )

        call_kwargs = self._mock_trainer_cls.call_args[1]
        self.assertEqual(call_kwargs["agent_env"], self.MOCK_LAMBDA_ARN)

    @patch("amzn_nova_forge.manager.mtrl_manager.logger")
    def test_execute_mtrl_with_model_path_resolves_model_package(self, _mock_logger):
        ops = self._create_mtrl_ops(agent_core_arn=self.MOCK_AGENT_CORE_ARN)
        modules = self._mock_sagemaker_modules()

        mock_model_package = MagicMock()
        self._mock_model_package_cls.get.return_value = mock_model_package

        mock_job = MagicMock()
        mock_job.job_name = "iter-job"
        mock_job.job_arn = "arn:aws:sagemaker:us-east-1:123456789012:job/iter-job"
        self._mock_trainer_instance.train.return_value = mock_job

        with patch.dict("sys.modules", modules):
            with patch.object(
                ops,
                "_get_or_create_checkpoint_model_package_group_arn",
                return_value="arn:checkpoint",
            ):
                result = ops.execute_mtrl(
                    model=Model.NOVA_LITE_2,
                    job_name="iter-job",
                    data_s3_path="s3://bucket/data",
                    model_path=self.MOCK_MODEL_PACKAGE_ARN,
                )

        self.assertEqual(result, "iter-job")
        self._mock_model_package_cls.get.assert_called_once_with(self.MOCK_MODEL_PACKAGE_ARN)
        call_kwargs = self._mock_trainer_cls.call_args[1]
        self.assertEqual(call_kwargs["model"], mock_model_package)

    @patch("amzn_nova_forge.manager.mtrl_manager.logger")
    def test_execute_mtrl_without_model_path_uses_base_model(self, _mock_logger):
        ops = self._create_mtrl_ops(agent_core_arn=self.MOCK_AGENT_CORE_ARN)
        modules = self._mock_sagemaker_modules()

        mock_job = MagicMock()
        mock_job.job_name = "base-job"
        mock_job.job_arn = "arn:aws:sagemaker:us-east-1:123456789012:job/base-job"
        self._mock_trainer_instance.train.return_value = mock_job

        with patch.dict("sys.modules", modules):
            with patch.object(
                ops,
                "_get_or_create_checkpoint_model_package_group_arn",
                return_value="arn:checkpoint",
            ):
                ops.execute_mtrl(
                    model=Model.NOVA_LITE_2,
                    job_name="base-job",
                    data_s3_path="s3://bucket/data",
                )

        self._mock_model_package_cls.get.assert_not_called()
        call_kwargs = self._mock_trainer_cls.call_args[1]
        self.assertEqual(call_kwargs["model"], Model.NOVA_LITE_2.hub_content_name)

    @patch("amzn_nova_forge.manager.mtrl_manager.logger")
    def test_execute_mtrl_agent_core_takes_priority_over_lambda(self, _mock_logger):
        ops = self._create_mtrl_ops(
            agent_core_arn=self.MOCK_AGENT_CORE_ARN,
            rft_lambda=self.MOCK_LAMBDA_ARN,
        )
        modules = self._mock_sagemaker_modules()

        mock_job = MagicMock()
        mock_job.job_name = "priority-job"
        mock_job.job_arn = "arn:aws:sagemaker:us-east-1:123456789012:job/priority-job"
        self._mock_trainer_instance.train.return_value = mock_job

        with patch.dict("sys.modules", modules):
            with patch.object(
                ops,
                "_get_or_create_checkpoint_model_package_group_arn",
                return_value="arn:checkpoint",
            ):
                ops.execute_mtrl(
                    model=Model.NOVA_LITE_2,
                    job_name="priority-job",
                    data_s3_path="s3://bucket/data",
                )

        call_kwargs = self._mock_trainer_cls.call_args[1]
        self.assertEqual(call_kwargs["agent_env"], self.MOCK_AGENT_CORE_ARN)


class TestSMTJServerlessMTRLEval(unittest.TestCase):
    """Tests for MTRLOperations.execute_mtrl_eval."""

    MOCK_AGENT_CORE_ARN = "arn:aws:bedrock-agentcore:us-east-1:123456789012:runtime/my-agent"
    MOCK_MODEL_PACKAGE_ARN = "arn:aws:sagemaker:us-east-1:123456789012:model-package/grp/1"

    def _create_mtrl_ops(self):
        from amzn_nova_forge.manager.mtrl_manager import MTRLOperations

        ops = MTRLOperations.__new__(MTRLOperations)
        ops.agent_core_arn = self.MOCK_AGENT_CORE_ARN
        ops.rft_lambda = None
        ops.model_package_group_arn = (
            "arn:aws:sagemaker:us-east-1:123456789012:model-package-group/grp"
        )
        ops.execution_role = "arn:aws:iam::123456789012:role/role"
        ops.subnets = None
        ops.security_group_ids = None
        ops.kms_key_id = None
        ops.region = "us-east-1"
        return ops

    def _mock_sagemaker_modules(self):
        import types

        mock_sm_train = types.ModuleType("sagemaker.train")
        mock_sm_train_mtrl = types.ModuleType("sagemaker.train.multi_turn_rl_trainer")
        mock_sm_train_eval = types.ModuleType("sagemaker.train.evaluate")
        mock_sm_core = types.ModuleType("sagemaker.core")
        mock_sm_core_resources = types.ModuleType("sagemaker.core.resources")

        self._mock_trainer_cls = MagicMock()
        self._mock_trainer_instance = MagicMock()
        self._mock_trainer_cls.return_value = self._mock_trainer_instance

        self._mock_evaluator_cls = MagicMock()
        self._mock_evaluator_instance = MagicMock()
        self._mock_evaluator_cls.return_value = self._mock_evaluator_instance

        self._mock_execution = MagicMock()
        self._mock_execution.arn = "arn:aws:sagemaker:us-east-1:123456789012:pipeline/eval-exec"
        self._mock_evaluator_instance.evaluate.return_value = self._mock_execution

        mock_sm_train_mtrl.MultiTurnRLTrainer = self._mock_trainer_cls
        mock_sm_train_eval.MultiTurnRLEvaluator = self._mock_evaluator_cls
        mock_sm_core_resources.ModelPackage = MagicMock()

        return {
            "sagemaker.train": mock_sm_train,
            "sagemaker.train.multi_turn_rl_trainer": mock_sm_train_mtrl,
            "sagemaker.train.evaluate": mock_sm_train_eval,
            "sagemaker.core": mock_sm_core,
            "sagemaker.core.resources": mock_sm_core_resources,
        }

    @patch("amzn_nova_forge.manager.mtrl_manager.logger")
    def test_execute_mtrl_eval_with_training_job_name(self, _mock_logger):
        ops = self._create_mtrl_ops()
        modules = self._mock_sagemaker_modules()

        mock_attached_job = MagicMock()
        self._mock_trainer_cls.attach.return_value = mock_attached_job

        with patch.dict("sys.modules", modules):
            with patch.object(
                ops,
                "_get_or_create_checkpoint_model_package_group_arn",
                return_value="arn:checkpoint",
            ):
                result = ops.execute_mtrl_eval(
                    model=Model.NOVA_LITE_2,
                    data_s3_path="s3://bucket/eval-data",
                    output_s3_path="s3://bucket/output",
                    model_path=self.MOCK_MODEL_PACKAGE_ARN,
                    training_job_name="completed-training-job",
                )

        self.assertEqual(result, self._mock_execution)
        # Should have built a trainer and attached the job
        self._mock_trainer_cls.assert_called_once()
        self._mock_trainer_cls.attach.assert_called_once_with("completed-training-job")
        # Evaluator should receive the trainer as model
        eval_call_kwargs = self._mock_evaluator_cls.call_args[1]
        self.assertEqual(eval_call_kwargs["model"], self._mock_trainer_instance)

    @patch("amzn_nova_forge.manager.mtrl_manager.logger")
    def test_execute_mtrl_eval_fallback_without_training_job(self, _mock_logger):
        ops = self._create_mtrl_ops()
        modules = self._mock_sagemaker_modules()

        with patch.dict("sys.modules", modules):
            with patch.object(
                ops,
                "_get_or_create_checkpoint_model_package_group_arn",
                return_value="arn:checkpoint",
            ):
                result = ops.execute_mtrl_eval(
                    model=Model.NOVA_LITE_2,
                    data_s3_path="s3://bucket/eval-data",
                    model_path=self.MOCK_MODEL_PACKAGE_ARN,
                )

        self.assertEqual(result, self._mock_execution)
        # Should NOT attach a job — uses fallback path
        self._mock_trainer_cls.attach.assert_not_called()
        # Evaluator should receive model name (base hub content) since model_path is ARN
        eval_call_kwargs = self._mock_evaluator_cls.call_args[1]
        self.assertEqual(eval_call_kwargs["model"], Model.NOVA_LITE_2.hub_content_name)
        # Should inject model_package_arn post-init
        self.assertEqual(
            self._mock_evaluator_instance._source_model_package_arn_cache,
            self.MOCK_MODEL_PACKAGE_ARN,
        )

    @patch("amzn_nova_forge.manager.mtrl_manager.logger")
    def test_execute_mtrl_eval_applies_overrides(self, _mock_logger):
        ops = self._create_mtrl_ops()
        modules = self._mock_sagemaker_modules()

        with patch.dict("sys.modules", modules):
            with patch.object(
                ops,
                "_get_or_create_checkpoint_model_package_group_arn",
                return_value="arn:checkpoint",
            ):
                ops.execute_mtrl_eval(
                    model=Model.NOVA_LITE_2,
                    data_s3_path="s3://bucket/eval-data",
                    overrides={"global_batch_size": 8},
                )

        # Verify evaluate was called (overrides applied without error)
        self._mock_evaluator_instance.evaluate.assert_called_once()


if __name__ == "__main__":
    unittest.main()
