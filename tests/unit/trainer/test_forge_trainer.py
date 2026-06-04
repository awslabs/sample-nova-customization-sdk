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
import unittest
import unittest.mock
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock, PropertyMock, create_autospec, patch

from amzn_nova_forge.core.constants import DEFAULT_BATCH_TRACE_CACHE_DIR
from amzn_nova_forge.core.enums import Model, Platform, TrainingMethod
from amzn_nova_forge.core.result import (
    BedrockTrainingResult,
    SMHPTrainingResult,
    SMTJTrainingResult,
)
from amzn_nova_forge.core.types import ForgeConfig, ModelArtifacts
from amzn_nova_forge.manager.runtime_manager import (
    BedrockRuntimeManager,
    SMHPRuntimeManager,
    SMTJRuntimeManager,
)
from amzn_nova_forge.monitor.mlflow_monitor import MLflowMonitor
from amzn_nova_forge.trainer.forge_trainer import ForgeTrainer

FIXED_OUTPUT_PATH = "s3://sagemaker-nova-123456789012-us-east-1/output"


def _make_smtj_infra():
    infra = create_autospec(SMTJRuntimeManager)
    infra.instance_type = "ml.p5.48xlarge"
    infra.instance_count = 2
    infra.kms_key_id = None
    infra.platform = Platform.SMTJ
    infra.rft_lambda_arn = None
    return infra


def _make_bedrock_infra():
    infra = create_autospec(BedrockRuntimeManager)
    infra.instance_type = None
    infra.instance_count = None
    infra.kms_key_id = None
    infra.platform = Platform.BEDROCK
    infra.rft_lambda_arn = None
    return infra


def _make_smhp_infra():
    infra = create_autospec(SMHPRuntimeManager)
    infra.instance_type = "ml.p5.48xlarge"
    infra.instance_count = 2
    infra.kms_key_id = None
    infra.platform = Platform.SMHP
    infra.cluster_name = "my-cluster"
    infra.namespace = "kubeflow"
    infra.rft_lambda_arn = None
    return infra


class TestForgeTrainerInit(unittest.TestCase):
    """Tests for ForgeTrainer.__init__."""

    @patch(
        "amzn_nova_forge.trainer.forge_trainer.set_output_s3_path",
        return_value=FIXED_OUTPUT_PATH,
    )
    @patch("boto3.session.Session")
    def test_happy_path(self, mock_session, mock_set_output):
        type(mock_session.return_value).region_name = PropertyMock(return_value="us-east-1")
        infra = _make_smtj_infra()

        trainer = ForgeTrainer(
            model=Model.NOVA_MICRO,
            method=TrainingMethod.SFT_LORA,
            infra=infra,
            training_data_s3_path="s3://bucket/data",
        )

        self.assertEqual(trainer.model, Model.NOVA_MICRO)
        self.assertEqual(trainer.method, TrainingMethod.SFT_LORA)
        self.assertEqual(trainer.region, "us-east-1")
        self.assertEqual(trainer.output_s3_path, FIXED_OUTPUT_PATH)
        self.assertFalse(trainer._is_multimodal)
        self.assertIsNone(trainer.data_mixing)
        mock_set_output.assert_called_once()

    @patch("boto3.session.Session")
    def test_unsupported_region_raises(self, mock_session):
        type(mock_session.return_value).region_name = PropertyMock(return_value="ap-southeast-99")
        infra = _make_smtj_infra()

        with self.assertRaises(ValueError) as ctx:
            ForgeTrainer(
                model=Model.NOVA_MICRO,
                method=TrainingMethod.SFT_LORA,
                infra=infra,
                training_data_s3_path="s3://bucket/data",
            )
        self.assertIn("ap-southeast-99", str(ctx.exception))
        self.assertIn("not supported", str(ctx.exception))

    @patch(
        "amzn_nova_forge.trainer.forge_trainer.set_output_s3_path",
        return_value=FIXED_OUTPUT_PATH,
    )
    @patch("boto3.session.Session")
    def test_smtj_serverless_requires_sagemaker_arn(self, mock_session, _mock_output):
        type(mock_session.return_value).region_name = PropertyMock(return_value="us-east-1")
        infra = create_autospec(SMTJRuntimeManager)
        infra.instance_type = None
        infra.instance_count = None
        infra.kms_key_id = None
        infra.platform = Platform.SMTJServerless

        with self.assertRaises(ValueError) as ctx:
            ForgeTrainer(
                model=Model.NOVA_MICRO,
                method=TrainingMethod.SFT_LORA,
                infra=infra,
                training_data_s3_path="s3://bucket/data",
                model_s3_path="s3://bucket/checkpoint/",
            )
        self.assertIn("model package ARN", str(ctx.exception))

    @patch(
        "amzn_nova_forge.trainer.forge_trainer.set_output_s3_path",
        return_value=FIXED_OUTPUT_PATH,
    )
    @patch("boto3.session.Session")
    def test_bedrock_warns_on_model_s3_path(self, mock_session, _mock_output):
        type(mock_session.return_value).region_name = PropertyMock(return_value="us-east-1")
        infra = _make_bedrock_infra()

        with patch("amzn_nova_forge.trainer.forge_trainer.logger") as mock_logger:
            ForgeTrainer(
                model=Model.NOVA_MICRO,
                method=TrainingMethod.SFT_LORA,
                infra=infra,
                training_data_s3_path="s3://bucket/data",
                model_s3_path="s3://bucket/model",
            )
            mock_logger.warning.assert_called_once()
            self.assertIn(
                "model_path is not used for Bedrock",
                mock_logger.warning.call_args[0][0],
            )

    @patch(
        "amzn_nova_forge.trainer.forge_trainer.set_output_s3_path",
        return_value=FIXED_OUTPUT_PATH,
    )
    @patch("boto3.session.Session")
    def test_data_mixing_raises_for_unsupported_platform_or_method(
        self, mock_session, _mock_output
    ):
        type(mock_session.return_value).region_name = PropertyMock(return_value="us-east-1")
        # SMTJ (non-SMHP) should raise
        infra = _make_smtj_infra()
        with self.assertRaises(ValueError) as ctx:
            ForgeTrainer(
                model=Model.NOVA_MICRO,
                method=TrainingMethod.CPT,
                infra=infra,
                training_data_s3_path="s3://bucket/data",
                data_mixing_enabled=True,
            )
        self.assertIn("SageMaker HyperPod", str(ctx.exception))

        # SMHP with unsupported method (DPO_LORA) should raise
        infra_smhp = _make_smhp_infra()
        with self.assertRaises(ValueError) as ctx:
            ForgeTrainer(
                model=Model.NOVA_MICRO,
                method=TrainingMethod.DPO_LORA,
                infra=infra_smhp,
                training_data_s3_path="s3://bucket/data",
                data_mixing_enabled=True,
            )
        self.assertIn("Data mixing is only supported", str(ctx.exception))

    @patch("amzn_nova_forge.trainer.forge_trainer.load_recipe_templates")
    @patch(
        "amzn_nova_forge.trainer.forge_trainer.set_output_s3_path",
        return_value=FIXED_OUTPUT_PATH,
    )
    @patch("boto3.session.Session")
    def test_data_mixing_smhp_cpt_works(self, mock_session, _mock_output, mock_load_templates):
        type(mock_session.return_value).region_name = PropertyMock(return_value="us-east-1")
        mock_load_templates.return_value = ({}, {}, None, "image:latest")
        infra = _make_smhp_infra()

        trainer = ForgeTrainer(
            model=Model.NOVA_MICRO,
            method=TrainingMethod.CPT,
            infra=infra,
            training_data_s3_path="s3://bucket/data",
            data_mixing_enabled=True,
        )
        self.assertIsNotNone(trainer.data_mixing)
        mock_load_templates.assert_called_once()

    @patch("amzn_nova_forge.trainer.forge_trainer.is_multimodal_data", return_value=True)
    @patch("amzn_nova_forge.trainer.forge_trainer.load_recipe_templates")
    @patch(
        "amzn_nova_forge.trainer.forge_trainer.set_output_s3_path",
        return_value=FIXED_OUTPUT_PATH,
    )
    @patch("boto3.session.Session")
    def test_is_multimodal_auto_detection(
        self, mock_session, _mock_output, mock_load_templates, mock_is_mm
    ):
        type(mock_session.return_value).region_name = PropertyMock(return_value="us-east-1")
        mock_load_templates.return_value = ({}, {}, None, "image:latest")
        infra = _make_smhp_infra()

        trainer = ForgeTrainer(
            model=Model.NOVA_MICRO,
            method=TrainingMethod.CPT,
            infra=infra,
            training_data_s3_path="s3://bucket/data",
            data_mixing_enabled=True,
        )
        self.assertTrue(trainer._is_multimodal)
        mock_is_mm.assert_called_once_with("s3://bucket/data")

    @patch("boto3.session.Session")
    def test_batch_sample_tracing_rejects_non_smhp(self, mock_session):
        type(mock_session.return_value).region_name = PropertyMock(return_value="us-east-1")
        infra = _make_smtj_infra()

        with self.assertRaises(ValueError) as ctx:
            ForgeTrainer(
                model=Model.NOVA_MICRO,
                method=TrainingMethod.CPT,
                infra=infra,
                training_data_s3_path="s3://bucket/data",
                enable_batch_sample_tracing=True,
            )
        self.assertIn("SageMaker HyperPod", str(ctx.exception))

    @patch("boto3.session.Session")
    def test_batch_sample_tracing_rejects_non_cpt(self, mock_session):
        type(mock_session.return_value).region_name = PropertyMock(return_value="us-east-1")
        infra = _make_smhp_infra()

        with self.assertRaises(ValueError) as ctx:
            ForgeTrainer(
                model=Model.NOVA_MICRO,
                method=TrainingMethod.SFT_LORA,
                infra=infra,
                training_data_s3_path="s3://bucket/data",
                enable_batch_sample_tracing=True,
            )
        self.assertIn("CPT", str(ctx.exception))

    @patch(
        "amzn_nova_forge.trainer.forge_trainer.set_output_s3_path",
        return_value=FIXED_OUTPUT_PATH,
    )
    @patch("boto3.session.Session")
    def test_hub_content_version_propagates_to_infra(self, mock_session, _mock_output):
        type(mock_session.return_value).region_name = PropertyMock(return_value="us-east-1")
        infra = _make_smtj_infra()
        infra.hub_content_version = None

        ForgeTrainer(
            model=Model.NOVA_MICRO,
            method=TrainingMethod.SFT_LORA,
            infra=infra,
            training_data_s3_path="s3://bucket/data",
            hub_content_version="3.38.0",
        )

        self.assertEqual(infra.hub_content_version, "3.38.0")

    @patch(
        "amzn_nova_forge.trainer.forge_trainer.set_output_s3_path",
        return_value=FIXED_OUTPUT_PATH,
    )
    @patch("boto3.session.Session")
    def test_hub_content_version_none_does_not_set_infra(self, mock_session, _mock_output):
        type(mock_session.return_value).region_name = PropertyMock(return_value="us-east-1")
        infra = _make_smtj_infra()
        infra.hub_content_version = None

        ForgeTrainer(
            model=Model.NOVA_MICRO,
            method=TrainingMethod.SFT_LORA,
            infra=infra,
            training_data_s3_path="s3://bucket/data",
            hub_content_version=None,
        )

        self.assertIsNone(infra.hub_content_version)

    @patch(
        "amzn_nova_forge.trainer.forge_trainer.set_output_s3_path",
        return_value=FIXED_OUTPUT_PATH,
    )
    @patch("boto3.session.Session")
    def test_hub_content_version_skips_infra_without_attribute(self, mock_session, _mock_output):
        type(mock_session.return_value).region_name = PropertyMock(return_value="us-east-1")
        infra = _make_smtj_infra()
        # Remove the attribute so hasattr returns False
        if hasattr(infra, "hub_content_version"):
            del infra.hub_content_version

        ForgeTrainer(
            model=Model.NOVA_MICRO,
            method=TrainingMethod.SFT_LORA,
            infra=infra,
            training_data_s3_path="s3://bucket/data",
            hub_content_version="3.38.0",
        )

        self.assertFalse(hasattr(infra, "hub_content_version"))


class TestForgeTrainerTrain(unittest.TestCase):
    """Tests for ForgeTrainer.train()."""

    def setUp(self):
        self._boto3_client_patcher = patch("boto3.client")
        self._mock_boto3_client = self._boto3_client_patcher.start()

    def tearDown(self):
        self._boto3_client_patcher.stop()

    def _make_trainer(self, infra=None, **kwargs):
        infra = infra or _make_smtj_infra()
        defaults = dict(
            model=Model.NOVA_MICRO,
            method=TrainingMethod.SFT_LORA,
            infra=infra,
            training_data_s3_path="s3://bucket/data",
        )
        defaults.update(kwargs)
        with (
            patch(
                "amzn_nova_forge.trainer.forge_trainer.set_output_s3_path",
                return_value=FIXED_OUTPUT_PATH,
            ),
            patch("boto3.session.Session") as mock_session,
        ):
            type(mock_session.return_value).region_name = PropertyMock(return_value="us-east-1")
            return ForgeTrainer(**defaults)

    @patch("amzn_nova_forge.trainer.forge_trainer.get_model_artifacts")
    @patch("amzn_nova_forge.trainer.forge_trainer.RecipeBuilder")
    def test_train_returns_smtj_result(self, MockRecipeBuilder, mock_get_artifacts):
        mock_builder = MockRecipeBuilder.return_value
        mock_builder.build_and_validate.return_value = (
            "/tmp/recipe.yaml",
            "s3://bucket/output",
            "s3://bucket/data",
            "image:latest",
        )

        infra = _make_smtj_infra()
        infra.execute.return_value = "job-123"
        mock_get_artifacts.return_value = ModelArtifacts(
            checkpoint_s3_path="s3://bucket/checkpoint",
            output_s3_path="s3://bucket/output",
        )

        trainer = self._make_trainer(infra=infra)
        result = trainer.train(job_name="my-job")

        self.assertIsInstance(result, SMTJTrainingResult)
        self.assertEqual(result.job_id, "job-123")
        self.assertEqual(result.method, TrainingMethod.SFT_LORA)
        self.assertEqual(result.model_type, Model.NOVA_MICRO)
        infra.execute.assert_called_once()

    @patch("amzn_nova_forge.trainer.forge_trainer.RecipeBuilder")
    def test_dry_run_returns_none(self, MockRecipeBuilder):
        mock_builder = MockRecipeBuilder.return_value
        mock_builder.build_and_validate.return_value = (
            "/tmp/recipe.yaml",
            "s3://bucket/output",
            "s3://bucket/data",
            "image:latest",
        )

        infra = _make_smtj_infra()
        trainer = self._make_trainer(infra=infra)
        result = trainer.train(job_name="my-job", dry_run=True)

        self.assertIsNone(result)
        infra.execute.assert_not_called()

    @patch("amzn_nova_forge.trainer.forge_trainer.RecipeBuilder")
    def test_dry_run_with_validation_data(self, MockRecipeBuilder):
        mock_builder = MockRecipeBuilder.return_value
        mock_builder.build_and_validate.return_value = (
            "/tmp/recipe.yaml",
            "s3://bucket/output",
            "s3://bucket/data",
            "image:latest",
        )

        infra = _make_smtj_infra()
        trainer = self._make_trainer(
            infra=infra,
            holdout_data_s3_path="s3://bucket/validation.jsonl",
        )
        result = trainer.train(job_name="my-job", dry_run=True)

        self.assertIsNone(result)
        infra.execute.assert_not_called()
        MockRecipeBuilder.assert_called_once()
        call_kwargs = MockRecipeBuilder.call_args[1]
        self.assertEqual(call_kwargs["validation_data_s3_path"], "s3://bucket/validation.jsonl")
        mock_builder.build_and_validate.assert_called_once()

    @patch("amzn_nova_forge.trainer.forge_trainer.validate_rft_lambda_name")
    @patch("amzn_nova_forge.trainer.forge_trainer.get_model_artifacts")
    @patch("amzn_nova_forge.trainer.forge_trainer.RecipeBuilder")
    def test_rft_lambda_arn_validated_when_provided(
        self, MockRecipeBuilder, mock_get_artifacts, mock_validate_lambda
    ):
        mock_builder = MockRecipeBuilder.return_value
        mock_builder.build_and_validate.return_value = (
            "/tmp/recipe.yaml",
            "s3://bucket/output",
            "s3://bucket/data",
            "image:latest",
        )
        infra = _make_smtj_infra()
        infra.execute.return_value = "job-123"
        mock_get_artifacts.return_value = ModelArtifacts(
            checkpoint_s3_path=None, output_s3_path="s3://bucket/output"
        )

        trainer = self._make_trainer(infra=infra)
        trainer.train(
            job_name="my-job",
            rft_lambda_arn="arn:aws:lambda:us-east-1:123:function:my-func",
        )

        mock_validate_lambda.assert_called_once_with("my-func", Platform.SMTJ)

    @patch("amzn_nova_forge.trainer.forge_trainer.validate_rft_lambda_name")
    @patch("amzn_nova_forge.trainer.forge_trainer.get_model_artifacts")
    @patch("amzn_nova_forge.trainer.forge_trainer.RecipeBuilder")
    def test_rft_lambda_arn_falls_back_to_infra(
        self, MockRecipeBuilder, mock_get_artifacts, mock_validate_lambda
    ):
        mock_builder = MockRecipeBuilder.return_value
        mock_builder.build_and_validate.return_value = (
            "/tmp/recipe.yaml",
            "s3://bucket/output",
            "s3://bucket/data",
            "image:latest",
        )
        infra = _make_smtj_infra()
        infra.rft_lambda_arn = "arn:aws:lambda:us-east-1:123:function:infra-func"
        infra.execute.return_value = "job-123"
        mock_get_artifacts.return_value = ModelArtifacts(
            checkpoint_s3_path=None, output_s3_path="s3://bucket/output"
        )

        trainer = self._make_trainer(infra=infra)
        trainer.train(job_name="my-job")

        mock_validate_lambda.assert_called_once_with("infra-func", Platform.SMTJ)

    @patch("amzn_nova_forge.trainer.forge_trainer.RecipeBuilder")
    def test_bedrock_training_returns_bedrock_result(self, MockRecipeBuilder):
        mock_builder = MockRecipeBuilder.return_value
        mock_builder.build_and_validate.return_value = (
            "/tmp/recipe.yaml",
            "s3://bucket/output",
            "s3://bucket/data",
            "image:latest",
        )
        infra = _make_bedrock_infra()
        infra.execute.return_value = "bedrock-job-123"

        trainer = self._make_trainer(infra=infra)
        result = trainer.train(job_name="my-job")

        self.assertIsInstance(result, BedrockTrainingResult)
        self.assertEqual(result.job_id, "bedrock-job-123")
        self.assertIsNone(result.model_artifacts.checkpoint_s3_path)

    @patch("amzn_nova_forge.trainer.forge_trainer.get_model_artifacts")
    @patch("amzn_nova_forge.trainer.forge_trainer.RecipeBuilder")
    def test_smhp_training_returns_smhp_result(self, MockRecipeBuilder, mock_get_artifacts):
        mock_builder = MockRecipeBuilder.return_value
        mock_builder.build_and_validate.return_value = (
            "/tmp/recipe.yaml",
            "s3://bucket/output",
            "s3://bucket/data",
            "image:latest",
        )
        infra = _make_smhp_infra()
        infra.execute.return_value = "smhp-job-123"
        mock_get_artifacts.return_value = ModelArtifacts(
            checkpoint_s3_path="s3://bucket/checkpoint",
            output_s3_path="s3://bucket/output",
        )

        trainer = self._make_trainer(infra=infra)
        result = trainer.train(job_name="my-job")

        self.assertIsInstance(result, SMHPTrainingResult)
        self.assertEqual(result.job_id, "smhp-job-123")
        self.assertEqual(result.cluster_name, "my-cluster")
        self.assertEqual(result.namespace, "kubeflow")


class TestForgeTrainerCaching(unittest.TestCase):
    """Tests for caching integration in ForgeTrainer.train()."""

    def setUp(self):
        self._boto3_client_patcher = patch("boto3.client")
        self._boto3_client_patcher.start()

    def tearDown(self):
        self._boto3_client_patcher.stop()

    def _make_trainer(self, enable_caching=False):
        infra = _make_smtj_infra()
        config = ForgeConfig(enable_job_caching=enable_caching) if enable_caching else None
        with (
            patch(
                "amzn_nova_forge.trainer.forge_trainer.set_output_s3_path",
                return_value=FIXED_OUTPUT_PATH,
            ),
            patch("boto3.session.Session") as mock_session,
        ):
            type(mock_session.return_value).region_name = PropertyMock(return_value="us-east-1")
            return ForgeTrainer(
                model=Model.NOVA_MICRO,
                method=TrainingMethod.SFT_LORA,
                infra=infra,
                training_data_s3_path="s3://bucket/data",
                config=config,
            )

    @patch("amzn_nova_forge.trainer.forge_trainer.load_existing_result")
    def test_cached_result_short_circuits_train(self, mock_load):
        mock_cached = MagicMock(spec=SMTJTrainingResult)
        mock_load.return_value = mock_cached

        trainer = self._make_trainer(enable_caching=True)
        result = trainer.train(job_name="cached-job")

        self.assertIs(result, mock_cached)
        mock_load.assert_called_once()

    @patch("amzn_nova_forge.trainer.forge_trainer.persist_result")
    @patch("amzn_nova_forge.trainer.forge_trainer.get_model_artifacts")
    @patch("amzn_nova_forge.trainer.forge_trainer.RecipeBuilder")
    @patch("amzn_nova_forge.trainer.forge_trainer.load_existing_result", return_value=None)
    def test_persist_called_after_successful_train(
        self, mock_load, MockRecipeBuilder, mock_get_artifacts, mock_persist
    ):
        mock_builder = MockRecipeBuilder.return_value
        mock_builder.build_and_validate.return_value = (
            "/tmp/recipe.yaml",
            "s3://bucket/output",
            "s3://bucket/data",
            "image:latest",
        )
        infra = _make_smtj_infra()
        infra.execute.return_value = "job-123"
        mock_get_artifacts.return_value = ModelArtifacts(
            checkpoint_s3_path="s3://bucket/checkpoint",
            output_s3_path="s3://bucket/output",
        )

        with (
            patch(
                "amzn_nova_forge.trainer.forge_trainer.set_output_s3_path",
                return_value=FIXED_OUTPUT_PATH,
            ),
            patch("boto3.session.Session") as mock_session,
        ):
            type(mock_session.return_value).region_name = PropertyMock(return_value="us-east-1")
            trainer = ForgeTrainer(
                model=Model.NOVA_MICRO,
                method=TrainingMethod.SFT_LORA,
                infra=infra,
                training_data_s3_path="s3://bucket/data",
                config=ForgeConfig(enable_job_caching=True),
            )

        trainer.train(job_name="my-job")
        mock_persist.assert_called_once()
        call_kwargs = mock_persist.call_args
        self.assertEqual(call_kwargs[1]["job_name"], "my-job")
        self.assertEqual(call_kwargs[1]["job_type"], "train")

    @patch("amzn_nova_forge.trainer.forge_trainer.persist_result")
    @patch("amzn_nova_forge.trainer.forge_trainer.load_existing_result")
    @patch("amzn_nova_forge.trainer.forge_trainer.get_model_artifacts")
    @patch("amzn_nova_forge.trainer.forge_trainer.RecipeBuilder")
    def test_caching_noop_when_disabled(
        self, MockRecipeBuilder, mock_get_artifacts, mock_load, mock_persist
    ):
        mock_load.return_value = None
        mock_builder = MockRecipeBuilder.return_value
        mock_builder.build_and_validate.return_value = (
            "/tmp/recipe.yaml",
            "s3://bucket/output",
            "s3://bucket/data",
            "image:latest",
        )
        infra = _make_smtj_infra()
        infra.execute.return_value = "job-123"
        mock_get_artifacts.return_value = ModelArtifacts(
            checkpoint_s3_path="s3://bucket/ckpt", output_s3_path="s3://bucket/out"
        )

        trainer = self._make_trainer(enable_caching=False)
        trainer.train(job_name="my-job")

        # Cache functions are still called but context has caching disabled
        mock_load.assert_called_once()
        mock_persist.assert_called_once()
        self.assertFalse(trainer._cache_context.enable_job_caching)


class TestForgeTrainerGetLogs(unittest.TestCase):
    """Tests for ForgeTrainer.get_logs()."""

    def _make_trainer(self, infra=None):
        infra = infra or _make_smtj_infra()
        with (
            patch(
                "amzn_nova_forge.trainer.forge_trainer.set_output_s3_path",
                return_value=FIXED_OUTPUT_PATH,
            ),
            patch("boto3.session.Session") as mock_session,
        ):
            type(mock_session.return_value).region_name = PropertyMock(return_value="us-east-1")
            return ForgeTrainer(
                model=Model.NOVA_MICRO,
                method=TrainingMethod.SFT_LORA,
                infra=infra,
                training_data_s3_path="s3://bucket/data",
            )

    @patch("amzn_nova_forge.trainer.forge_trainer.CloudWatchLogMonitor")
    def test_get_logs_with_job_result(self, MockMonitor):
        trainer = self._make_trainer()
        mock_result = MagicMock()
        mock_result.job_id = "job-abc"
        mock_result.started_time = datetime(2025, 1, 1, tzinfo=timezone.utc)

        trainer.get_logs(job_result=mock_result)

        MockMonitor.assert_called_once_with(
            job_id="job-abc",
            platform=Platform.SMTJ,
            started_time=int(datetime(2025, 1, 1, tzinfo=timezone.utc).timestamp() * 1000),
            region="us-east-1",
        )
        MockMonitor.return_value.show_logs.assert_called_once()

    @patch("amzn_nova_forge.trainer.forge_trainer.CloudWatchLogMonitor")
    def test_get_logs_with_explicit_ids(self, MockMonitor):
        trainer = self._make_trainer()
        started = datetime(2025, 6, 15, tzinfo=timezone.utc)

        trainer.get_logs(job_id="explicit-job", started_time=started)

        MockMonitor.assert_called_once_with(
            job_id="explicit-job",
            platform=Platform.SMTJ,
            started_time=int(started.timestamp() * 1000),
            region="us-east-1",
        )

    @patch("amzn_nova_forge.trainer.forge_trainer.CloudWatchLogMonitor")
    def test_get_logs_missing_info_raises_value_error(self, MockMonitor):
        trainer = self._make_trainer()

        with self.assertRaises(ValueError) as ctx:
            trainer.get_logs()

        self.assertIn("job_result", str(ctx.exception))
        MockMonitor.assert_not_called()

    @patch("amzn_nova_forge.trainer.forge_trainer.CloudWatchLogMonitor")
    def test_get_logs_job_id_only_raises_value_error(self, MockMonitor):
        trainer = self._make_trainer()

        with self.assertRaises(ValueError) as ctx:
            trainer.get_logs(job_id="some-job")

        self.assertIn("job_result", str(ctx.exception))
        MockMonitor.assert_not_called()

    @patch("amzn_nova_forge.trainer.forge_trainer.CloudWatchLogMonitor")
    def test_get_logs_started_time_only_raises_value_error(self, MockMonitor):
        trainer = self._make_trainer()

        with self.assertRaises(ValueError) as ctx:
            trainer.get_logs(started_time=datetime(2025, 1, 1, tzinfo=timezone.utc))

        self.assertIn("job_result", str(ctx.exception))
        MockMonitor.assert_not_called()

    @patch("amzn_nova_forge.trainer.forge_trainer.CloudWatchLogMonitor")
    def test_get_logs_smhp_includes_cluster_kwargs(self, MockMonitor):
        infra = _make_smhp_infra()
        trainer = self._make_trainer(infra=infra)
        started = datetime(2025, 1, 1, tzinfo=timezone.utc)

        trainer.get_logs(job_id="smhp-job", started_time=started)

        MockMonitor.assert_called_once_with(
            job_id="smhp-job",
            platform=Platform.SMHP,
            started_time=int(started.timestamp() * 1000),
            cluster_name="my-cluster",
            namespace="kubeflow",
            region="us-east-1",
        )


class TestForgeTrainerTraceBatch(unittest.TestCase):
    """Tests for ForgeTrainer.trace_batch()."""

    def _make_trainer(self, **kwargs):
        infra = create_autospec(SMTJRuntimeManager)
        infra.instance_type = "ml.p5.48xlarge"
        infra.instance_count = 2
        infra.kms_key_id = None
        infra.platform = Platform.SMTJ
        infra.rft_lambda_arn = None

        defaults = dict(
            model=Model.NOVA_MICRO,
            method=TrainingMethod.SFT_LORA,
            infra=infra,
            training_data_s3_path="s3://bucket/data.jsonl",
        )
        defaults.update(kwargs)
        with (
            patch(
                "amzn_nova_forge.trainer.forge_trainer.set_output_s3_path",
                return_value="s3://bucket/output",
            ),
            patch("boto3.session.Session") as mock_session,
        ):
            type(mock_session.return_value).region_name = PropertyMock(return_value="us-east-1")
            return ForgeTrainer(**defaults)

    @patch("amzn_nova_forge.trainer.forge_trainer.batch_trace_run")
    def test_delegates_to_batch_trace_run(self, mock_run):
        mock_run.return_value = Path("step_42_samples.jsonl")
        trainer = self._make_trainer()

        result_obj = MagicMock()
        result_obj.job_id = "my-job-123"
        result_obj.model_artifacts = ModelArtifacts(
            checkpoint_s3_path="s3://bucket/checkpoint",
            output_s3_path="s3://bucket/output",
        )

        result = trainer.trace_batch(result_obj, step=42)

        mock_run.assert_called_once_with(
            data_path="s3://bucket/data.jsonl",
            log_dir="s3://bucket/output/my-job-123/batch_tracing/",
            step=42,
            output_path=None,
            s3_client=unittest.mock.ANY,
            cache_dir=DEFAULT_BATCH_TRACE_CACHE_DIR,
        )
        self.assertEqual(result, Path("step_42_samples.jsonl"))

    @patch("amzn_nova_forge.trainer.forge_trainer.batch_trace_run")
    def test_passes_output_path_and_cache_dir(self, mock_run):
        mock_run.return_value = Path("/tmp/out.jsonl")
        trainer = self._make_trainer()

        result_obj = MagicMock()
        result_obj.job_id = "my-job-456"
        result_obj.model_artifacts = ModelArtifacts(
            checkpoint_s3_path=None,
            output_s3_path="s3://bucket/output",
        )

        trainer.trace_batch(
            result_obj, step=10, output_path="/tmp/out.jsonl", cache_dir="/tmp/cache"
        )

        mock_run.assert_called_once_with(
            data_path="s3://bucket/data.jsonl",
            log_dir="s3://bucket/output/my-job-456/batch_tracing/",
            step=10,
            output_path="/tmp/out.jsonl",
            s3_client=unittest.mock.ANY,
            cache_dir="/tmp/cache",
        )

    def test_raises_when_no_training_data_path(self):
        trainer = self._make_trainer(training_data_s3_path=None)
        result_obj = MagicMock()
        result_obj.model_artifacts = ModelArtifacts(
            checkpoint_s3_path=None, output_s3_path="s3://bucket/output"
        )

        with self.assertRaises(ValueError) as ctx:
            trainer.trace_batch(result_obj, step=0)
        self.assertIn("training_data_s3_path", str(ctx.exception))

    def test_raises_when_no_output_s3_path(self):
        trainer = self._make_trainer()
        result_obj = MagicMock()
        result_obj.model_artifacts = ModelArtifacts(checkpoint_s3_path=None, output_s3_path="")

        with self.assertRaises(ValueError) as ctx:
            trainer.trace_batch(result_obj, step=0)
        self.assertIn("output_s3_path", str(ctx.exception))

    @patch("amzn_nova_forge.trainer.forge_trainer.batch_trace_run")
    def test_strips_trailing_slash_from_output_path(self, mock_run):
        mock_run.return_value = None
        trainer = self._make_trainer()

        result_obj = MagicMock()
        result_obj.job_id = "my-job-789"
        result_obj.model_artifacts = ModelArtifacts(
            checkpoint_s3_path=None,
            output_s3_path="s3://bucket/output/",
        )

        trainer.trace_batch(result_obj, step=5)

        call_kwargs = mock_run.call_args
        self.assertEqual(
            call_kwargs.kwargs["log_dir"], "s3://bucket/output/my-job-789/batch_tracing/"
        )

    @unittest.mock.patch("amzn_nova_forge.trainer.forge_trainer.batch_trace_run")
    def test_warns_when_tracing_not_enabled(self, mock_run):
        mock_run.return_value = None
        trainer = self._make_trainer()
        result = MagicMock()
        result.model_artifacts.output_s3_path = "s3://bucket/output"
        result.job_id = "job-123"
        with self.assertLogs("nova_forge_sdk", level="WARNING") as cm:
            trainer.trace_batch(training_result=result, step=1)
        self.assertTrue(any("not enabled" in msg for msg in cm.output))


class TestForgeTrainerDataMixingServerless(unittest.TestCase):
    """Tests for data mixing support on SMTJServerless."""

    def _make_serverless_infra(self):
        from amzn_nova_forge.manager.runtime_manager import SMTJServerlessRuntimeManager

        infra = create_autospec(SMTJServerlessRuntimeManager)
        infra.instance_type = None
        infra.instance_count = None
        infra.kms_key_id = None
        infra.platform = Platform.SMTJServerless
        return infra

    @patch("amzn_nova_forge.trainer.forge_trainer.load_recipe_templates")
    @patch(
        "amzn_nova_forge.trainer.forge_trainer.set_output_s3_path",
        return_value=FIXED_OUTPUT_PATH,
    )
    @patch("boto3.session.Session")
    def test_data_mixing_smtj_serverless_sft_lora_works(
        self, mock_session, _mock_output, mock_load_templates
    ):
        """SMTJServerless + SFT_LORA + data_mixing_enabled=True should succeed."""
        type(mock_session.return_value).region_name = PropertyMock(return_value="us-east-1")
        mock_load_templates.return_value = ({}, {}, None, "image:latest")
        infra = self._make_serverless_infra()

        trainer = ForgeTrainer(
            model=Model.NOVA_MICRO,
            method=TrainingMethod.SFT_LORA,
            infra=infra,
            training_data_s3_path="s3://bucket/data",
            data_mixing_enabled=True,
        )
        self.assertIsNotNone(trainer.data_mixing)

    @patch("amzn_nova_forge.trainer.forge_trainer.load_recipe_templates")
    @patch(
        "amzn_nova_forge.trainer.forge_trainer.set_output_s3_path",
        return_value=FIXED_OUTPUT_PATH,
    )
    @patch("boto3.session.Session")
    def test_data_mixing_smtj_serverless_sft_full_works(
        self, mock_session, _mock_output, mock_load_templates
    ):
        """SMTJServerless + SFT_FULL + data_mixing_enabled=True should succeed."""
        type(mock_session.return_value).region_name = PropertyMock(return_value="us-east-1")
        mock_load_templates.return_value = ({}, {}, None, "image:latest")
        infra = self._make_serverless_infra()

        trainer = ForgeTrainer(
            model=Model.NOVA_MICRO,
            method=TrainingMethod.SFT_FULL,
            infra=infra,
            training_data_s3_path="s3://bucket/data",
            data_mixing_enabled=True,
        )
        self.assertIsNotNone(trainer.data_mixing)

    @patch(
        "amzn_nova_forge.trainer.forge_trainer.set_output_s3_path",
        return_value=FIXED_OUTPUT_PATH,
    )
    @patch("boto3.session.Session")
    def test_data_mixing_smtj_serverless_unsupported_method_raises(
        self, mock_session, _mock_output
    ):
        """SMTJServerless + unsupported method (RFT_LORA) should raise."""
        type(mock_session.return_value).region_name = PropertyMock(return_value="us-east-1")
        infra = self._make_serverless_infra()

        with self.assertRaises(ValueError) as ctx:
            ForgeTrainer(
                model=Model.NOVA_MICRO,
                method=TrainingMethod.RFT_LORA,
                infra=infra,
                training_data_s3_path="s3://bucket/data",
                data_mixing_enabled=True,
            )
        self.assertIn("Data mixing is only supported", str(ctx.exception))
        self.assertIn("SMTJServerless", str(ctx.exception))

    @patch("amzn_nova_forge.trainer.forge_trainer.load_recipe_templates")
    @patch(
        "amzn_nova_forge.trainer.forge_trainer.set_output_s3_path",
        return_value=FIXED_OUTPUT_PATH,
    )
    @patch("boto3.session.Session")
    def test_data_mixing_config_in_job_config_params_for_serverless(
        self, mock_session, _mock_output, mock_load_templates
    ):
        """data_mixing_config must be passed in JobConfig to infra.execute() for SMTJServerless."""
        type(mock_session.return_value).region_name = PropertyMock(return_value="us-east-1")
        mock_load_templates.return_value = ({}, {}, None, "image:latest")
        infra = self._make_serverless_infra()
        infra.execute.return_value = "serverless-job-id"

        trainer = ForgeTrainer(
            model=Model.NOVA_MICRO,
            method=TrainingMethod.SFT_LORA,
            infra=infra,
            training_data_s3_path="s3://bucket/data",
            data_mixing_enabled=True,
            config=ForgeConfig(output_s3_path=FIXED_OUTPUT_PATH),
        )
        trainer.data_mixing.set_config({"customer_data_percent": 50, "nova_code_percent": 100})

        with (
            patch("amzn_nova_forge.trainer.forge_trainer.RecipeBuilder") as mock_rb_cls,
            patch("amzn_nova_forge.trainer.forge_trainer.load_existing_result", return_value=None),
            patch("amzn_nova_forge.trainer.forge_trainer.get_model_artifacts") as mock_gma,
            patch("boto3.client"),
        ):
            mock_rb = mock_rb_cls.return_value
            mock_rb.build_and_validate.return_value = (
                "/tmp/recipe.yaml",
                FIXED_OUTPUT_PATH,
                "s3://bucket/data",
                "image:latest",
            )
            mock_gma.return_value = MagicMock()
            trainer.train(job_name="test-datamix-job")

        infra.execute.assert_called_once()
        job_config = infra.execute.call_args.kwargs["job_config"]
        self.assertEqual(
            job_config.data_mixing_config,
            {"customer_data_percent": 50, "nova_code_percent": 100},
        )

    @patch("amzn_nova_forge.trainer.forge_trainer.load_recipe_templates")
    @patch(
        "amzn_nova_forge.trainer.forge_trainer.set_output_s3_path",
        return_value=FIXED_OUTPUT_PATH,
    )
    @patch("boto3.session.Session")
    def test_data_mixing_not_set_for_smhp(self, mock_session, _mock_output, mock_load_templates):
        """For SMHP, data_mixing_config must NOT be passed in JobConfig to infra.execute()."""
        type(mock_session.return_value).region_name = PropertyMock(return_value="us-east-1")
        mock_load_templates.return_value = ({}, {}, None, "image:latest")
        infra = _make_smhp_infra()
        infra.execute.return_value = "smhp-job-id"

        trainer = ForgeTrainer(
            model=Model.NOVA_MICRO,
            method=TrainingMethod.SFT_LORA,
            infra=infra,
            training_data_s3_path="s3://bucket/data",
            data_mixing_enabled=True,
            config=ForgeConfig(output_s3_path=FIXED_OUTPUT_PATH),
        )
        trainer.data_mixing.set_config({"customer_data_percent": 50, "nova_code_percent": 100})

        with (
            patch("amzn_nova_forge.trainer.forge_trainer.RecipeBuilder") as mock_rb_cls,
            patch("amzn_nova_forge.trainer.forge_trainer.load_existing_result", return_value=None),
            patch("amzn_nova_forge.trainer.forge_trainer.get_model_artifacts") as mock_gma,
        ):
            mock_rb = mock_rb_cls.return_value
            mock_rb.build_and_validate.return_value = (
                "/tmp/recipe.yaml",
                FIXED_OUTPUT_PATH,
                "s3://bucket/data",
                "image:latest",
            )
            mock_gma.return_value = MagicMock()
            trainer.train(job_name="test-datamix-smhp-job")

        infra.execute.assert_called_once()
        job_config = infra.execute.call_args.kwargs["job_config"]
        self.assertIsNone(job_config.data_mixing_config)


class TestForgeTrainerValCheckInterval(unittest.TestCase):
    """Tests for val_check_interval in ForgeTrainer."""

    def _make_trainer(self, **kwargs):
        infra = _make_smtj_infra()
        defaults = dict(
            model=Model.NOVA_MICRO,
            method=TrainingMethod.SFT_LORA,
            infra=infra,
            training_data_s3_path="s3://bucket/data",
        )
        defaults.update(kwargs)
        with (
            patch(
                "amzn_nova_forge.trainer.forge_trainer.set_output_s3_path",
                return_value=FIXED_OUTPUT_PATH,
            ),
            patch("boto3.session.Session") as mock_session,
        ):
            type(mock_session.return_value).region_name = PropertyMock(return_value="us-east-1")
            return ForgeTrainer(**defaults)

    @patch(
        "amzn_nova_forge.trainer.forge_trainer.set_output_s3_path",
        return_value=FIXED_OUTPUT_PATH,
    )
    @patch("boto3.session.Session")
    def test_val_check_interval_valid(self, mock_session, _mock_output):
        type(mock_session.return_value).region_name = PropertyMock(return_value="us-east-1")
        trainer = ForgeTrainer(
            model=Model.NOVA_MICRO,
            method=TrainingMethod.SFT_LORA,
            infra=_make_smtj_infra(),
            training_data_s3_path="s3://bucket/data",
            holdout_data_s3_path="s3://bucket/val",
            val_check_interval=500,
        )
        self.assertEqual(trainer.val_check_interval, 500)

    def test_val_check_interval_zero_raises(self):
        with self.assertRaises(ValueError):
            self._make_trainer(val_check_interval=0)

    def test_val_check_interval_negative_raises(self):
        with self.assertRaises(ValueError):
            self._make_trainer(val_check_interval=-1)

    def test_val_check_interval_float_raises(self):
        with self.assertRaises((ValueError, TypeError)):
            self._make_trainer(val_check_interval=5.0)

    def test_val_check_interval_without_validation_data_warns(self):
        with patch("amzn_nova_forge.trainer.forge_trainer.logger") as mock_logger:
            self._make_trainer(val_check_interval=500, holdout_data_s3_path=None)
            mock_logger.warning.assert_called_once()
            self.assertIn(
                "val_check_interval",
                mock_logger.warning.call_args[0][0],
            )

    @patch("amzn_nova_forge.trainer.forge_trainer.get_model_artifacts")
    @patch("amzn_nova_forge.trainer.forge_trainer.RecipeBuilder")
    def test_val_check_interval_threads_to_job_config(self, MockRecipeBuilder, mock_get_artifacts):
        mock_builder = MockRecipeBuilder.return_value
        mock_builder.build_and_validate.return_value = (
            "/tmp/recipe.yaml",
            "s3://bucket/output",
            "s3://bucket/data",
            "image:latest",
        )
        infra = _make_smtj_infra()
        infra.execute.return_value = "job-123"
        mock_get_artifacts.return_value = ModelArtifacts(
            checkpoint_s3_path="s3://bucket/checkpoint",
            output_s3_path="s3://bucket/output",
        )

        with patch("boto3.client"):
            trainer = self._make_trainer(
                infra=infra,
                val_check_interval=500,
                holdout_data_s3_path="s3://bucket/val",
            )
            trainer.train(job_name="my-job")

        call_args = infra.execute.call_args
        job_config = call_args.kwargs["job_config"]
        self.assertEqual(job_config.trainer_config_hyperparameters, {"val_check_interval": "500"})

    @patch("amzn_nova_forge.trainer.forge_trainer.get_model_artifacts")
    @patch("amzn_nova_forge.trainer.forge_trainer.RecipeBuilder")
    def test_val_check_interval_none_no_hyperparameters(
        self, MockRecipeBuilder, mock_get_artifacts
    ):
        mock_builder = MockRecipeBuilder.return_value
        mock_builder.build_and_validate.return_value = (
            "/tmp/recipe.yaml",
            "s3://bucket/output",
            "s3://bucket/data",
            "image:latest",
        )
        infra = _make_smtj_infra()
        infra.execute.return_value = "job-123"
        mock_get_artifacts.return_value = ModelArtifacts(
            checkpoint_s3_path="s3://bucket/checkpoint",
            output_s3_path="s3://bucket/output",
        )

        with patch("boto3.client"):
            trainer = self._make_trainer(infra=infra)
            trainer.train(job_name="my-job")

        call_args = infra.execute.call_args
        job_config = call_args.kwargs["job_config"]
        self.assertIsNone(job_config.trainer_config_hyperparameters)


class TestForgeTrainerGetConfig(unittest.TestCase):
    """Tests for ForgeTrainer.get_config()."""

    @patch(
        "amzn_nova_forge.trainer.forge_trainer.set_output_s3_path",
        return_value=FIXED_OUTPUT_PATH,
    )
    @patch("boto3.session.Session")
    def _make_trainer(self, mock_session, mock_set_output, **kwargs):
        type(mock_session.return_value).region_name = PropertyMock(return_value="us-east-1")
        defaults = dict(
            model=Model.NOVA_MICRO,
            method=TrainingMethod.SFT_LORA,
            infra=_make_smtj_infra(),
            training_data_s3_path="s3://bucket/data",
        )
        defaults.update(kwargs)
        return ForgeTrainer(**defaults)

    @patch("amzn_nova_forge.trainer.forge_trainer.RecipeBuilder")
    def test_returns_recipe_config(self, mock_rb_cls):
        from amzn_nova_forge.core.types import ConfigParameter, RecipeConfig

        expected = RecipeConfig(
            model=Model.NOVA_MICRO,
            method=TrainingMethod.SFT_LORA,
            platform=Platform.SMTJ,
            parameters=(ConfigParameter(name="lr", type="float", default=5e-6),),
        )
        mock_rb_cls.return_value.get_overridable_config.return_value = expected

        trainer = self._make_trainer()
        result = trainer.get_config()

        self.assertEqual(result, expected)
        mock_rb_cls.return_value.get_overridable_config.assert_called_once_with(overrides=None)

    @patch("amzn_nova_forge.trainer.forge_trainer.RecipeBuilder")
    def test_passes_overrides_through(self, mock_rb_cls):
        from amzn_nova_forge.core.types import RecipeConfig

        mock_rb_cls.return_value.get_overridable_config.return_value = RecipeConfig(
            model=Model.NOVA_MICRO,
            method=TrainingMethod.SFT_LORA,
            platform=Platform.SMTJ,
            parameters=(),
        )

        trainer = self._make_trainer()
        overrides = {"lr": 1e-5, "max_epochs": 3}
        trainer.get_config(overrides=overrides)

        mock_rb_cls.return_value.get_overridable_config.assert_called_once_with(overrides=overrides)

    @patch("amzn_nova_forge.trainer.forge_trainer.RecipeBuilder")
    def test_does_not_start_job(self, mock_rb_cls):
        from amzn_nova_forge.core.types import RecipeConfig

        mock_rb_cls.return_value.get_overridable_config.return_value = RecipeConfig(
            model=Model.NOVA_MICRO,
            method=TrainingMethod.SFT_LORA,
            platform=Platform.SMTJ,
            parameters=(),
        )

        trainer = self._make_trainer()
        trainer.get_config()

        trainer.infra.execute.assert_not_called()

    @patch("amzn_nova_forge.trainer.forge_trainer.RecipeBuilder")
    def test_creates_recipe_builder_with_trainer_state(self, mock_rb_cls):
        from amzn_nova_forge.core.types import RecipeConfig

        mock_rb_cls.return_value.get_overridable_config.return_value = RecipeConfig(
            model=Model.NOVA_MICRO,
            method=TrainingMethod.SFT_LORA,
            platform=Platform.SMTJ,
            parameters=(),
        )

        trainer = self._make_trainer()
        trainer.get_config()

        call_kwargs = mock_rb_cls.call_args.kwargs
        self.assertEqual(call_kwargs["model"], Model.NOVA_MICRO)
        self.assertEqual(call_kwargs["method"], TrainingMethod.SFT_LORA)
        self.assertEqual(call_kwargs["platform"], Platform.SMTJ)
        self.assertEqual(call_kwargs["job_name"], "__config_preview__")

    def test_get_config_rft_multiturn_raises(self):
        trainer = self._make_trainer(
            model=Model.NOVA_LITE_2, method=TrainingMethod.RFT_MULTITURN_LORA
        )
        with self.assertRaises(ValueError) as ctx:
            trainer.get_config()
        self.assertIn("rft_multiturn_infra", str(ctx.exception))


class TestForgeTrainerModelArn(unittest.TestCase):
    """Tests for model_arn parameter on ForgeTrainer for SMTJServerless."""

    MOCK_MODEL_PACKAGE_ARN = "arn:aws:sagemaker:us-east-1:123456789012:model-package/my-group/1"

    def _make_serverless_infra(self):
        from amzn_nova_forge.manager.runtime_manager import SMTJServerlessRuntimeManager

        infra = create_autospec(SMTJServerlessRuntimeManager)
        infra.instance_type = None
        infra.instance_count = None
        infra.kms_key_id = None
        infra.platform = Platform.SMTJServerless
        infra.rft_lambda_arn = None
        infra.hub_content_version = None
        return infra

    @patch(
        "amzn_nova_forge.trainer.forge_trainer.set_output_s3_path",
        return_value=FIXED_OUTPUT_PATH,
    )
    @patch("boto3.session.Session")
    def test_model_arn_accepted_for_serverless(self, mock_session, _mock_output):
        type(mock_session.return_value).region_name = PropertyMock(return_value="us-east-1")
        infra = self._make_serverless_infra()

        trainer = ForgeTrainer(
            model=Model.NOVA_LITE_2,
            method=TrainingMethod.RFT_MULTITURN_LORA,
            infra=infra,
            training_data_s3_path="s3://bucket/data",
            model_arn=self.MOCK_MODEL_PACKAGE_ARN,
        )

        self.assertEqual(trainer.model_arn, self.MOCK_MODEL_PACKAGE_ARN)
        self.assertIsNone(trainer.model_s3_path)

    @patch(
        "amzn_nova_forge.trainer.forge_trainer.set_output_s3_path",
        return_value=FIXED_OUTPUT_PATH,
    )
    @patch("boto3.session.Session")
    def test_model_s3_path_migrated_to_model_arn_for_serverless(self, mock_session, _mock_output):
        type(mock_session.return_value).region_name = PropertyMock(return_value="us-east-1")
        infra = self._make_serverless_infra()

        trainer = ForgeTrainer(
            model=Model.NOVA_LITE_2,
            method=TrainingMethod.RFT_MULTITURN_LORA,
            infra=infra,
            training_data_s3_path="s3://bucket/data",
            model_s3_path=self.MOCK_MODEL_PACKAGE_ARN,
        )

        self.assertEqual(trainer.model_arn, self.MOCK_MODEL_PACKAGE_ARN)
        self.assertIsNone(trainer.model_s3_path)

    @patch(
        "amzn_nova_forge.trainer.forge_trainer.set_output_s3_path",
        return_value=FIXED_OUTPUT_PATH,
    )
    @patch("boto3.session.Session")
    def test_both_model_arn_and_model_s3_path_raises(self, mock_session, _mock_output):
        type(mock_session.return_value).region_name = PropertyMock(return_value="us-east-1")
        infra = self._make_serverless_infra()

        with self.assertRaises(ValueError) as ctx:
            ForgeTrainer(
                model=Model.NOVA_LITE_2,
                method=TrainingMethod.RFT_MULTITURN_LORA,
                infra=infra,
                training_data_s3_path="s3://bucket/data",
                model_s3_path=self.MOCK_MODEL_PACKAGE_ARN,
                model_arn=self.MOCK_MODEL_PACKAGE_ARN,
            )
        self.assertIn("Cannot specify both", str(ctx.exception))

    @patch(
        "amzn_nova_forge.trainer.forge_trainer.set_output_s3_path",
        return_value=FIXED_OUTPUT_PATH,
    )
    @patch("boto3.session.Session")
    def test_model_arn_invalid_raises(self, mock_session, _mock_output):
        type(mock_session.return_value).region_name = PropertyMock(return_value="us-east-1")
        infra = self._make_serverless_infra()

        with self.assertRaises(ValueError) as ctx:
            ForgeTrainer(
                model=Model.NOVA_LITE_2,
                method=TrainingMethod.RFT_MULTITURN_LORA,
                infra=infra,
                training_data_s3_path="s3://bucket/data",
                model_arn="s3://bucket/not-an-arn/",
            )
        self.assertIn("model_arn must be a SageMaker model package ARN", str(ctx.exception))

    @patch(
        "amzn_nova_forge.trainer.forge_trainer.set_output_s3_path",
        return_value=FIXED_OUTPUT_PATH,
    )
    @patch("boto3.session.Session")
    def test_no_model_arn_trains_from_base(self, mock_session, _mock_output):
        type(mock_session.return_value).region_name = PropertyMock(return_value="us-east-1")
        infra = self._make_serverless_infra()

        trainer = ForgeTrainer(
            model=Model.NOVA_LITE_2,
            method=TrainingMethod.RFT_MULTITURN_LORA,
            infra=infra,
            training_data_s3_path="s3://bucket/data",
        )

        self.assertIsNone(trainer.model_arn)
        self.assertIsNone(trainer.model_s3_path)


class TestForgeTrainerMTRLTrain(unittest.TestCase):
    """Tests for the MTRL serverless training path in ForgeTrainer.train()."""

    MOCK_MLFLOW_ARN = "arn:aws:sagemaker:us-east-1:123456789012:mlflow-tracking-server/my-server"

    def _make_serverless_infra(self):
        from amzn_nova_forge.manager.runtime_manager import SMTJServerlessRuntimeManager

        infra = create_autospec(SMTJServerlessRuntimeManager)
        infra.instance_type = None
        infra.instance_count = None
        infra.kms_key_id = None
        infra.platform = Platform.SMTJServerless
        infra.rft_lambda_arn = "arn:aws:bedrock-agentcore:us-east-1:123456789012:runtime/agent"
        infra.hub_content_version = None
        infra.execute_mtrl.return_value = "mtrl-job-id-123"
        return infra

    def _make_mlflow_config(self):
        mock_monitor = MagicMock(spec=MLflowMonitor)
        mock_monitor.tracking_uri = self.MOCK_MLFLOW_ARN
        return ForgeConfig(mlflow_monitor=mock_monitor)

    @patch(
        "amzn_nova_forge.trainer.forge_trainer.set_output_s3_path",
        return_value=FIXED_OUTPUT_PATH,
    )
    @patch("boto3.session.Session")
    def test_mtrl_train_raises_without_mlflow_config(self, mock_session, _mock_output):
        """AgentRFT jobs must have an MLflow config; raise ValueError if missing."""
        type(mock_session.return_value).region_name = PropertyMock(return_value="us-east-1")
        infra = self._make_serverless_infra()

        trainer = ForgeTrainer(
            model=Model.NOVA_LITE_2,
            method=TrainingMethod.RFT_MULTITURN_LORA,
            infra=infra,
            training_data_s3_path="s3://bucket/data",
        )

        with self.assertRaises(ValueError) as ctx:
            trainer.train(job_name="my-mtrl-job")
        self.assertIn("MLflow configuration is required", str(ctx.exception))

    @patch(
        "amzn_nova_forge.trainer.forge_trainer.set_output_s3_path",
        return_value=FIXED_OUTPUT_PATH,
    )
    @patch("boto3.session.Session")
    def test_mtrl_train_raises_without_mlflow_tracking_uri(self, mock_session, _mock_output):
        """AgentRFT jobs must have a tracking_uri; raise ValueError if None."""
        type(mock_session.return_value).region_name = PropertyMock(return_value="us-east-1")
        infra = self._make_serverless_infra()

        mock_monitor = MagicMock(spec=MLflowMonitor)
        mock_monitor.tracking_uri = None

        trainer = ForgeTrainer(
            model=Model.NOVA_LITE_2,
            method=TrainingMethod.RFT_MULTITURN_LORA,
            infra=infra,
            training_data_s3_path="s3://bucket/data",
            config=ForgeConfig(mlflow_monitor=mock_monitor),
        )

        with self.assertRaises(ValueError) as ctx:
            trainer.train(job_name="my-mtrl-job")
        self.assertIn("MLflow configuration is required", str(ctx.exception))

    @patch(
        "amzn_nova_forge.trainer.forge_trainer.set_output_s3_path",
        return_value=FIXED_OUTPUT_PATH,
    )
    @patch("boto3.session.Session")
    def test_mtrl_train_raises_for_rft_multiturn_full_without_mlflow(
        self, mock_session, _mock_output
    ):
        """RFT_MULTITURN_FULL also requires MLflow config."""
        type(mock_session.return_value).region_name = PropertyMock(return_value="us-east-1")
        infra = self._make_serverless_infra()

        trainer = ForgeTrainer(
            model=Model.NOVA_LITE_2,
            method=TrainingMethod.RFT_MULTITURN_FULL,
            infra=infra,
            training_data_s3_path="s3://bucket/data",
        )

        with self.assertRaises(ValueError) as ctx:
            trainer.train(job_name="my-mtrl-full-job")
        self.assertIn("MLflow configuration is required", str(ctx.exception))

    @patch("amzn_nova_forge.trainer.forge_trainer.get_model_artifacts")
    @patch("amzn_nova_forge.trainer.forge_trainer.persist_result")
    @patch(
        "amzn_nova_forge.trainer.forge_trainer.set_output_s3_path",
        return_value=FIXED_OUTPUT_PATH,
    )
    @patch("boto3.session.Session")
    def test_mtrl_serverless_train_returns_smtj_training_result(
        self, mock_session, _mock_output, mock_persist, mock_get_artifacts
    ):
        type(mock_session.return_value).region_name = PropertyMock(return_value="us-east-1")
        infra = self._make_serverless_infra()

        mock_get_artifacts.return_value = ModelArtifacts(
            output_s3_path="s3://bucket/output/",
            output_model_arn=None,
        )

        trainer = ForgeTrainer(
            model=Model.NOVA_LITE_2,
            method=TrainingMethod.RFT_MULTITURN_LORA,
            infra=infra,
            training_data_s3_path="s3://bucket/data",
            config=self._make_mlflow_config(),
        )

        with patch("amzn_nova_forge.trainer.forge_trainer.SMTJTrainingResult") as MockResult:
            mock_result_instance = MagicMock()
            mock_result_instance.job_id = "mtrl-job-id-123"
            mock_result_instance.model_artifacts = ModelArtifacts(
                output_s3_path="s3://bucket/output/"
            )
            MockResult.return_value = mock_result_instance

            result = trainer.train(job_name="my-mtrl-job")

        infra.execute_mtrl.assert_called_once()
        call_kwargs = infra.execute_mtrl.call_args[1]
        self.assertEqual(call_kwargs["model"], Model.NOVA_LITE_2)
        self.assertEqual(call_kwargs["data_s3_path"], "s3://bucket/data")
        self.assertEqual(call_kwargs["output_s3_path"], FIXED_OUTPUT_PATH)
        self.assertIsNone(call_kwargs["model_path"])
        self.assertEqual(result, mock_result_instance)

    @patch("amzn_nova_forge.trainer.forge_trainer.get_model_artifacts")
    @patch("amzn_nova_forge.trainer.forge_trainer.persist_result")
    @patch(
        "amzn_nova_forge.trainer.forge_trainer.set_output_s3_path",
        return_value=FIXED_OUTPUT_PATH,
    )
    @patch("boto3.session.Session")
    def test_mtrl_serverless_train_passes_model_arn(
        self, mock_session, _mock_output, mock_persist, mock_get_artifacts
    ):
        type(mock_session.return_value).region_name = PropertyMock(return_value="us-east-1")
        infra = self._make_serverless_infra()
        model_arn = "arn:aws:sagemaker:us-east-1:123456789012:model-package/grp/1"

        mock_get_artifacts.return_value = ModelArtifacts(
            output_s3_path="s3://bucket/output/",
            output_model_arn=None,
        )

        trainer = ForgeTrainer(
            model=Model.NOVA_LITE_2,
            method=TrainingMethod.RFT_MULTITURN_LORA,
            infra=infra,
            training_data_s3_path="s3://bucket/data",
            model_arn=model_arn,
            config=self._make_mlflow_config(),
        )

        with patch("amzn_nova_forge.trainer.forge_trainer.SMTJTrainingResult") as MockResult:
            mock_result_instance = MagicMock()
            mock_result_instance.job_id = "mtrl-job-id-456"
            mock_result_instance.model_artifacts = ModelArtifacts(
                output_s3_path="s3://bucket/output/"
            )
            MockResult.return_value = mock_result_instance

            trainer.train(job_name="my-iterative-job")

        call_kwargs = infra.execute_mtrl.call_args[1]
        self.assertEqual(call_kwargs["model_path"], model_arn)

    @patch("amzn_nova_forge.trainer.forge_trainer.get_model_artifacts")
    @patch("amzn_nova_forge.trainer.forge_trainer.persist_result")
    @patch(
        "amzn_nova_forge.trainer.forge_trainer.set_output_s3_path",
        return_value=FIXED_OUTPUT_PATH,
    )
    @patch("boto3.session.Session")
    def test_mtrl_serverless_train_passes_overrides(
        self, mock_session, _mock_output, mock_persist, mock_get_artifacts
    ):
        type(mock_session.return_value).region_name = PropertyMock(return_value="us-east-1")
        infra = self._make_serverless_infra()

        mock_get_artifacts.return_value = ModelArtifacts(
            output_s3_path="s3://bucket/output/",
        )

        trainer = ForgeTrainer(
            model=Model.NOVA_LITE_2,
            method=TrainingMethod.RFT_MULTITURN_LORA,
            infra=infra,
            training_data_s3_path="s3://bucket/data",
            config=self._make_mlflow_config(),
        )

        with patch("amzn_nova_forge.trainer.forge_trainer.SMTJTrainingResult") as MockResult:
            mock_result_instance = MagicMock()
            mock_result_instance.job_id = "mtrl-job-id-789"
            mock_result_instance.model_artifacts = ModelArtifacts(
                output_s3_path="s3://bucket/output/"
            )
            MockResult.return_value = mock_result_instance

            trainer.train(
                job_name="my-overrides-job",
                overrides={"global_batch_size": 16, "max_steps": 50},
            )

        call_kwargs = infra.execute_mtrl.call_args[1]
        self.assertEqual(call_kwargs["overrides"], {"global_batch_size": 16, "max_steps": 50})


class TestForgeTrainerGetLogsMTRL(unittest.TestCase):
    """Tests for ForgeTrainer.get_logs() with MTRL SMTJTrainingResult."""

    def _make_trainer(self):
        from amzn_nova_forge.manager.runtime_manager import SMTJServerlessRuntimeManager

        infra = create_autospec(SMTJServerlessRuntimeManager)
        infra.instance_type = None
        infra.instance_count = None
        infra.kms_key_id = None
        infra.platform = Platform.SMTJServerless
        infra.rft_lambda_arn = None
        infra.hub_content_version = None

        with (
            patch(
                "amzn_nova_forge.trainer.forge_trainer.set_output_s3_path",
                return_value=FIXED_OUTPUT_PATH,
            ),
            patch("boto3.session.Session") as mock_session,
        ):
            type(mock_session.return_value).region_name = PropertyMock(return_value="us-east-1")
            return ForgeTrainer(
                model=Model.NOVA_LITE_2,
                method=TrainingMethod.RFT_MULTITURN_LORA,
                infra=infra,
                training_data_s3_path="s3://bucket/data",
            )

    @patch("amzn_nova_forge.trainer.forge_trainer.CloudWatchLogMonitor")
    def test_get_logs_mtrl_delegates_to_wait(self, MockMonitor):
        """When job_result is an MTRL SMTJTrainingResult, get_logs() should call wait() instead of streaming CloudWatch logs."""
        trainer = self._make_trainer()

        mock_result = MagicMock(spec=SMTJTrainingResult)
        mock_result._is_mtrl = True
        mock_result.job_id = "mtrl-job-abc"
        mock_result.started_time = datetime(2025, 5, 1, tzinfo=timezone.utc)

        trainer.get_logs(job_result=mock_result)

        # Should call wait() on the job_result
        mock_result.wait.assert_called_once_with(poll=30, timeout=7200)
        # Should NOT create a CloudWatch monitor
        MockMonitor.assert_not_called()

    @patch("amzn_nova_forge.trainer.forge_trainer.CloudWatchLogMonitor")
    def test_get_logs_mtrl_passes_custom_poll_and_timeout(self, MockMonitor):
        """MTRL get_logs() should pass through custom poll and timeout values."""
        trainer = self._make_trainer()

        mock_result = MagicMock(spec=SMTJTrainingResult)
        mock_result._is_mtrl = True
        mock_result.job_id = "mtrl-job-def"
        mock_result.started_time = datetime(2025, 5, 1, tzinfo=timezone.utc)

        trainer.get_logs(job_result=mock_result, poll=60, timeout=7200)

        mock_result.wait.assert_called_once_with(poll=60, timeout=7200)
        MockMonitor.assert_not_called()

    @patch("amzn_nova_forge.trainer.forge_trainer.CloudWatchLogMonitor")
    def test_get_logs_non_mtrl_smtj_result_still_streams_cloudwatch(self, MockMonitor):
        """Non-MTRL SMTJTrainingResult should still stream CloudWatch logs normally."""
        trainer = self._make_trainer()

        mock_result = MagicMock(spec=SMTJTrainingResult)
        mock_result._is_mtrl = False
        mock_result.job_id = "smtj-job-xyz"
        mock_result.started_time = datetime(2025, 5, 1, tzinfo=timezone.utc)

        trainer.get_logs(job_result=mock_result)

        # Should NOT call wait()
        mock_result.wait.assert_not_called()
        # Should create a CloudWatch monitor
        MockMonitor.assert_called_once()
        MockMonitor.return_value.show_logs.assert_called_once()


if __name__ == "__main__":
    unittest.main()
