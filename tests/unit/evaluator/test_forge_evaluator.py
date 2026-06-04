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
import os
import tempfile
import unittest
from datetime import datetime, timezone
from unittest.mock import MagicMock, PropertyMock, create_autospec, patch

from amzn_nova_forge.core.enums import EvaluationTask, Model, Platform, TrainingMethod
from amzn_nova_forge.core.result import (
    SMHPEvaluationResult,
    SMTJEvaluationResult,
    SMTJTrainingResult,
)
from amzn_nova_forge.core.result.training_result import TrainingResult
from amzn_nova_forge.core.types import ForgeConfig, ModelArtifacts
from amzn_nova_forge.evaluator.forge_evaluator import EvalTaskConfig, ForgeEvaluator
from amzn_nova_forge.evaluator.inspect_lens_config import InspectLensConfig
from amzn_nova_forge.manager.runtime_manager import (
    SMHPRuntimeManager,
    SMTJRuntimeManager,
    SMTJServerlessRuntimeManager,
)
from amzn_nova_forge.monitor import MLflowMonitor


class TestForgeEvaluatorInit(unittest.TestCase):
    """Tests for ForgeEvaluator constructor."""

    def setUp(self):
        self.model = Model.NOVA_MICRO
        self.mock_infra = create_autospec(SMTJRuntimeManager)
        self.mock_infra.kms_key_id = None
        self.mock_infra.instance_type = "ml.p5.48xlarge"
        self.mock_infra.instance_count = 2
        self.mock_infra.platform = Platform.SMTJ

    @patch(
        "amzn_nova_forge.evaluator.forge_evaluator.set_output_s3_path",
        return_value="s3://bucket/output",
    )
    @patch("boto3.session.Session")
    @patch("boto3.client")
    def test_init_happy_path(self, mock_client, mock_session, mock_set_output):
        type(mock_session.return_value).region_name = PropertyMock(return_value="us-east-1")
        evaluator = ForgeEvaluator(
            model=self.model,
            infra=self.mock_infra,
            data_s3_path="s3://bucket/data",
        )
        self.assertEqual(evaluator.model, Model.NOVA_MICRO)
        self.assertEqual(evaluator.region, "us-east-1")
        self.assertEqual(evaluator.data_s3_path, "s3://bucket/data")
        self.assertEqual(evaluator.output_s3_path, "s3://bucket/output")

    @patch("boto3.session.Session")
    def test_init_unsupported_region_raises(self, mock_session):
        type(mock_session.return_value).region_name = PropertyMock(
            return_value="unsupported-region"
        )
        with self.assertRaises(ValueError) as ctx:
            ForgeEvaluator(
                model=self.model,
                infra=self.mock_infra,
            )
        self.assertIn("unsupported-region", str(ctx.exception))
        self.assertIn("not supported", str(ctx.exception))

    @patch(
        "amzn_nova_forge.evaluator.forge_evaluator.set_output_s3_path",
        return_value="s3://bucket/output",
    )
    @patch("boto3.session.Session")
    @patch("boto3.client")
    def test_platform_resolved_from_infra(self, mock_client, mock_session, mock_set_output):
        type(mock_session.return_value).region_name = PropertyMock(return_value="us-east-1")
        smhp_infra = create_autospec(SMHPRuntimeManager)
        smhp_infra.kms_key_id = None
        smhp_infra.platform = Platform.SMHP
        smhp_infra.instance_type = "ml.p5.48xlarge"
        smhp_infra.instance_count = 2

        evaluator = ForgeEvaluator(model=self.model, infra=smhp_infra)
        self.assertEqual(evaluator._platform, Platform.SMHP)


class TestForgeEvaluatorEvaluate(unittest.TestCase):
    """Tests for ForgeEvaluator.evaluate()."""

    def setUp(self):
        self.model = Model.NOVA_MICRO
        self.mock_infra = create_autospec(SMTJRuntimeManager)
        self.mock_infra.kms_key_id = None
        self.mock_infra.instance_type = "ml.p5.48xlarge"
        self.mock_infra.instance_count = 2
        self.mock_infra.platform = Platform.SMTJ
        self.mock_infra.rft_lambda_arn = None
        self.mock_infra.execute.return_value = "job-123"

        self._patcher_set_output = patch(
            "amzn_nova_forge.evaluator.forge_evaluator.set_output_s3_path",
            return_value="s3://bucket/output",
        )
        self._patcher_session = patch("boto3.session.Session")
        self._patcher_client = patch("boto3.client")

        self._patcher_set_output.start()
        mock_session = self._patcher_session.start()
        self._patcher_client.start()

        type(mock_session.return_value).region_name = PropertyMock(return_value="us-east-1")
        self.evaluator = ForgeEvaluator(
            model=self.model,
            infra=self.mock_infra,
            data_s3_path="s3://bucket/data",
        )

    def tearDown(self):
        self._patcher_client.stop()
        self._patcher_session.stop()
        self._patcher_set_output.stop()

    def test_bedrock_platform_raises_not_implemented(self):
        self.evaluator._platform = Platform.BEDROCK
        with self.assertRaises(NotImplementedError) as ctx:
            self.evaluator.evaluate(
                job_name="test-eval",
                eval_task=EvaluationTask.MMLU,
            )
        self.assertIn("Bedrock", str(ctx.exception))

    @patch("amzn_nova_forge.evaluator.forge_evaluator.validate_platform_compatibility")
    @patch("amzn_nova_forge.evaluator.forge_evaluator.detect_platform_from_path")
    @patch(
        "amzn_nova_forge.evaluator.forge_evaluator.resolve_model_checkpoint_path",
        return_value="s3://bucket/checkpoint",
    )
    @patch("amzn_nova_forge.evaluator.forge_evaluator.RecipeBuilder")
    def test_smtj_evaluate_returns_smtj_result(
        self,
        mock_recipe_cls,
        mock_resolve,
        mock_detect,
        mock_validate_compat,
    ):
        mock_detect.return_value = Platform.SMTJ
        mock_builder = MagicMock()
        mock_builder.build_and_validate.return_value = (
            "/tmp/recipe.yaml",
            "s3://bucket/output",
            "s3://bucket/data",
            "image:latest",
        )
        mock_recipe_cls.return_value = mock_builder

        result = self.evaluator.evaluate(
            job_name="test-eval",
            eval_task=EvaluationTask.MMLU,
        )

        self.assertIsInstance(result, SMTJEvaluationResult)
        self.assertEqual(result.job_id, "job-123")
        self.assertEqual(result.eval_task, EvaluationTask.MMLU)
        self.mock_infra.execute.assert_called_once()

    @patch("amzn_nova_forge.evaluator.forge_evaluator.validate_platform_compatibility")
    @patch("amzn_nova_forge.evaluator.forge_evaluator.detect_platform_from_path")
    @patch(
        "amzn_nova_forge.evaluator.forge_evaluator.resolve_model_checkpoint_path",
        return_value="s3://bucket/checkpoint",
    )
    @patch("amzn_nova_forge.evaluator.forge_evaluator.RecipeBuilder")
    def test_smhp_evaluate_returns_smhp_result(
        self,
        mock_recipe_cls,
        mock_resolve,
        mock_detect,
        mock_validate_compat,
    ):
        mock_detect.return_value = Platform.SMHP
        mock_builder = MagicMock()
        mock_builder.build_and_validate.return_value = (
            "/tmp/recipe.yaml",
            "s3://bucket/output",
            "s3://bucket/data",
            "image:latest",
        )
        mock_recipe_cls.return_value = mock_builder

        smhp_infra = create_autospec(SMHPRuntimeManager)
        smhp_infra.kms_key_id = None
        smhp_infra.instance_type = "ml.p5.48xlarge"
        smhp_infra.instance_count = 2
        smhp_infra.platform = Platform.SMHP
        smhp_infra.rft_lambda_arn = None
        smhp_infra.execute.return_value = "smhp-job-456"
        smhp_infra.cluster_name = "my-cluster"
        smhp_infra.namespace = "kubeflow"

        self.evaluator.infra = smhp_infra
        self.evaluator._platform = Platform.SMHP

        result = self.evaluator.evaluate(
            job_name="test-eval",
            eval_task=EvaluationTask.MMLU,
        )

        self.assertIsInstance(result, SMHPEvaluationResult)
        self.assertEqual(result.job_id, "smhp-job-456")
        self.assertEqual(result.cluster_name, "my-cluster")
        self.assertEqual(result.namespace, "kubeflow")

    @patch("amzn_nova_forge.evaluator.forge_evaluator.validate_platform_compatibility")
    @patch("amzn_nova_forge.evaluator.forge_evaluator.detect_platform_from_path")
    @patch(
        "amzn_nova_forge.evaluator.forge_evaluator.resolve_model_checkpoint_path",
        return_value="s3://bucket/checkpoint",
    )
    @patch("amzn_nova_forge.evaluator.forge_evaluator.RecipeBuilder")
    def test_dry_run_returns_none(
        self,
        mock_recipe_cls,
        mock_resolve,
        mock_detect,
        mock_validate_compat,
    ):
        mock_detect.return_value = Platform.SMTJ
        mock_builder = MagicMock()
        mock_builder.build_and_validate.return_value = (
            "/tmp/recipe.yaml",
            "s3://bucket/output",
            "s3://bucket/data",
            "image:latest",
        )
        mock_recipe_cls.return_value = mock_builder

        result = self.evaluator.evaluate(
            job_name="test-eval",
            eval_task=EvaluationTask.MMLU,
            dry_run=True,
        )

        self.assertIsNone(result)
        self.mock_infra.execute.assert_not_called()

    @patch("amzn_nova_forge.evaluator.forge_evaluator.validate_platform_compatibility")
    @patch("amzn_nova_forge.evaluator.forge_evaluator.detect_platform_from_path")
    @patch("amzn_nova_forge.evaluator.forge_evaluator.resolve_model_checkpoint_path")
    @patch("amzn_nova_forge.evaluator.forge_evaluator.RecipeBuilder")
    def test_model_path_passed_to_resolve(
        self,
        mock_recipe_cls,
        mock_resolve,
        mock_detect,
        mock_validate_compat,
    ):
        mock_detect.return_value = None
        mock_resolve.return_value = "s3://bucket/my-checkpoint"
        mock_builder = MagicMock()
        mock_builder.build_and_validate.return_value = (
            "/tmp/recipe.yaml",
            "s3://bucket/output",
            "s3://bucket/data",
            "image:latest",
        )
        mock_recipe_cls.return_value = mock_builder

        self.evaluator.evaluate(
            job_name="test-eval",
            eval_task=EvaluationTask.MMLU,
            model_path="s3://bucket/my-checkpoint",
        )

        mock_resolve.assert_called_once_with(
            model_path="s3://bucket/my-checkpoint",
            job_result=None,
            customizer_job_id=None,
            customizer_output_s3_path="s3://bucket/output",
            customizer_model_path=None,
        )

    @patch("amzn_nova_forge.evaluator.forge_evaluator.validate_platform_compatibility")
    @patch("amzn_nova_forge.evaluator.forge_evaluator.detect_platform_from_path")
    @patch("amzn_nova_forge.evaluator.forge_evaluator.resolve_model_checkpoint_path")
    @patch("amzn_nova_forge.evaluator.forge_evaluator.RecipeBuilder")
    def test_job_result_used_for_checkpoint(
        self,
        mock_recipe_cls,
        mock_resolve,
        mock_detect,
        mock_validate_compat,
    ):
        mock_detect.return_value = None
        mock_resolve.return_value = "s3://bucket/from-result"
        mock_builder = MagicMock()
        mock_builder.build_and_validate.return_value = (
            "/tmp/recipe.yaml",
            "s3://bucket/output",
            "s3://bucket/data",
            "image:latest",
        )
        mock_recipe_cls.return_value = mock_builder

        mock_job_result = MagicMock(spec=TrainingResult)
        mock_job_result.model_artifacts = ModelArtifacts(
            checkpoint_s3_path="s3://bucket/checkpoint",
            output_s3_path="s3://bucket/output",
        )

        self.evaluator.evaluate(
            job_name="test-eval",
            eval_task=EvaluationTask.MMLU,
            job_result=mock_job_result,
        )

        mock_resolve.assert_called_once_with(
            model_path=None,
            job_result=mock_job_result,
            customizer_job_id=None,
            customizer_output_s3_path="s3://bucket/output",
            customizer_model_path=None,
        )

    @patch("amzn_nova_forge.evaluator.forge_evaluator.validate_platform_compatibility")
    @patch("amzn_nova_forge.evaluator.forge_evaluator.detect_platform_from_path")
    @patch(
        "amzn_nova_forge.evaluator.forge_evaluator.resolve_model_checkpoint_path",
        return_value=None,
    )
    @patch("amzn_nova_forge.evaluator.forge_evaluator.RecipeBuilder")
    def test_override_data_s3_path_used(
        self,
        mock_recipe_cls,
        mock_resolve,
        mock_detect,
        mock_validate_compat,
    ):
        mock_detect.return_value = None
        mock_builder = MagicMock()
        mock_builder.build_and_validate.return_value = (
            "/tmp/recipe.yaml",
            "s3://bucket/output",
            "s3://custom/data",
            "image:latest",
        )
        mock_recipe_cls.return_value = mock_builder

        tc = EvalTaskConfig(override_data_s3_path="s3://custom/data")
        self.evaluator.evaluate(
            job_name="test-eval",
            eval_task=EvaluationTask.GEN_QA,
            task_config=tc,
        )

        # RecipeBuilder should receive the override data path
        _, kwargs = mock_recipe_cls.call_args
        self.assertEqual(kwargs["data_s3_path"], "s3://custom/data")

    @patch("amzn_nova_forge.evaluator.forge_evaluator.validate_platform_compatibility")
    @patch("amzn_nova_forge.evaluator.forge_evaluator.detect_platform_from_path")
    @patch(
        "amzn_nova_forge.evaluator.forge_evaluator.resolve_model_checkpoint_path",
        return_value=None,
    )
    @patch("amzn_nova_forge.evaluator.forge_evaluator.RecipeBuilder")
    def test_byod_eval_uses_constructor_data_s3_path(
        self,
        mock_recipe_cls,
        mock_resolve,
        mock_detect,
        mock_validate_compat,
    ):
        mock_detect.return_value = None
        mock_builder = MagicMock()
        mock_builder.build_and_validate.return_value = (
            "/tmp/recipe.yaml",
            "s3://bucket/output",
            "s3://bucket/data",
            "image:latest",
        )
        mock_recipe_cls.return_value = mock_builder

        # GEN_QA is a BYOD task, so constructor's data_s3_path should be used
        self.evaluator.evaluate(
            job_name="test-eval",
            eval_task=EvaluationTask.GEN_QA,
        )

        _, kwargs = mock_recipe_cls.call_args
        self.assertEqual(kwargs["data_s3_path"], "s3://bucket/data")

    @patch("amzn_nova_forge.evaluator.forge_evaluator.validate_platform_compatibility")
    @patch("amzn_nova_forge.evaluator.forge_evaluator.detect_platform_from_path")
    @patch(
        "amzn_nova_forge.evaluator.forge_evaluator.resolve_model_checkpoint_path",
        return_value=None,
    )
    @patch("amzn_nova_forge.evaluator.forge_evaluator.RecipeBuilder")
    def test_non_byod_eval_ignores_data_s3_path(
        self,
        mock_recipe_cls,
        mock_resolve,
        mock_detect,
        mock_validate_compat,
    ):
        mock_detect.return_value = None
        mock_builder = MagicMock()
        mock_builder.build_and_validate.return_value = (
            "/tmp/recipe.yaml",
            "s3://bucket/output",
            None,
            "image:latest",
        )
        mock_recipe_cls.return_value = mock_builder

        # MMLU is NOT a BYOD task, data_s3_path should be None
        self.evaluator.evaluate(
            job_name="test-eval",
            eval_task=EvaluationTask.MMLU,
        )

        _, kwargs = mock_recipe_cls.call_args
        self.assertIsNone(kwargs["data_s3_path"])

    @patch("amzn_nova_forge.evaluator.forge_evaluator.validate_platform_compatibility")
    @patch("amzn_nova_forge.evaluator.forge_evaluator.detect_platform_from_path")
    @patch(
        "amzn_nova_forge.evaluator.forge_evaluator.resolve_model_checkpoint_path",
        return_value=None,
    )
    @patch("amzn_nova_forge.evaluator.forge_evaluator.RecipeBuilder")
    def test_rft_eval_processor_lambda_arn_to_rl_env(
        self,
        mock_recipe_cls,
        mock_resolve,
        mock_detect,
        mock_validate_compat,
    ):
        mock_detect.return_value = None
        mock_builder = MagicMock()
        mock_builder.build_and_validate.return_value = (
            "/tmp/recipe.yaml",
            "s3://bucket/output",
            None,
            "image:latest",
        )
        mock_recipe_cls.return_value = mock_builder

        tc = EvalTaskConfig(
            processor={"lambda_arn": "arn:aws:lambda:us-east-1:123:function:MyFunc"}
        )
        self.evaluator.evaluate(
            job_name="test-eval",
            eval_task=EvaluationTask.RFT_EVAL,
            task_config=tc,
        )

        _, kwargs = mock_recipe_cls.call_args
        self.assertEqual(
            kwargs["rl_env_config"],
            {"reward_lambda_arn": "arn:aws:lambda:us-east-1:123:function:MyFunc"},
        )
        # processor should be cleared
        self.assertIsNone(kwargs["processor_config"])

    @patch("amzn_nova_forge.evaluator.forge_evaluator.validate_platform_compatibility")
    @patch("amzn_nova_forge.evaluator.forge_evaluator.detect_platform_from_path")
    @patch(
        "amzn_nova_forge.evaluator.forge_evaluator.resolve_model_checkpoint_path",
        return_value=None,
    )
    @patch("amzn_nova_forge.evaluator.forge_evaluator.RecipeBuilder")
    def test_infra_rft_lambda_arn_auto_populates_rl_env(
        self,
        mock_recipe_cls,
        mock_resolve,
        mock_detect,
        mock_validate_compat,
    ):
        mock_detect.return_value = None
        mock_builder = MagicMock()
        mock_builder.build_and_validate.return_value = (
            "/tmp/recipe.yaml",
            "s3://bucket/output",
            None,
            "image:latest",
        )
        mock_recipe_cls.return_value = mock_builder

        self.mock_infra.rft_lambda_arn = "arn:aws:lambda:us-east-1:123:function:InfraLambda"

        self.evaluator.evaluate(
            job_name="test-eval",
            eval_task=EvaluationTask.RFT_EVAL,
        )

        _, kwargs = mock_recipe_cls.call_args
        self.assertEqual(
            kwargs["rl_env_config"],
            {"reward_lambda_arn": "arn:aws:lambda:us-east-1:123:function:InfraLambda"},
        )

    @patch("amzn_nova_forge.evaluator.forge_evaluator.validate_rft_lambda_name")
    @patch("amzn_nova_forge.evaluator.forge_evaluator.validate_platform_compatibility")
    @patch("amzn_nova_forge.evaluator.forge_evaluator.detect_platform_from_path")
    @patch(
        "amzn_nova_forge.evaluator.forge_evaluator.resolve_model_checkpoint_path",
        return_value=None,
    )
    @patch("amzn_nova_forge.evaluator.forge_evaluator.RecipeBuilder")
    def test_rl_env_reward_lambda_arn_validated(
        self,
        mock_recipe_cls,
        mock_resolve,
        mock_detect,
        mock_validate_compat,
        mock_validate_lambda,
    ):
        mock_detect.return_value = None
        mock_builder = MagicMock()
        mock_builder.build_and_validate.return_value = (
            "/tmp/recipe.yaml",
            "s3://bucket/output",
            None,
            "image:latest",
        )
        mock_recipe_cls.return_value = mock_builder

        tc = EvalTaskConfig(
            rl_env={"reward_lambda_arn": "arn:aws:lambda:us-east-1:123:function:MyFunc"}
        )
        self.evaluator.evaluate(
            job_name="test-eval",
            eval_task=EvaluationTask.RFT_EVAL,
            task_config=tc,
        )

        mock_validate_lambda.assert_called_once_with("MyFunc", Platform.SMTJ)


class TestForgeEvaluatorCaching(unittest.TestCase):
    """Tests for caching integration in ForgeEvaluator.evaluate()."""

    def setUp(self):
        self.model = Model.NOVA_MICRO
        self.mock_infra = create_autospec(SMTJRuntimeManager)
        self.mock_infra.kms_key_id = None
        self.mock_infra.instance_type = "ml.p5.48xlarge"
        self.mock_infra.instance_count = 2
        self.mock_infra.platform = Platform.SMTJ
        self.mock_infra.rft_lambda_arn = None
        self.mock_infra.execute.return_value = "eval-job-123"

        self._patcher_set_output = patch(
            "amzn_nova_forge.evaluator.forge_evaluator.set_output_s3_path",
            return_value="s3://bucket/output",
        )
        self._patcher_session = patch("boto3.session.Session")
        self._patcher_client = patch("boto3.client")

        self._patcher_set_output.start()
        mock_session = self._patcher_session.start()
        self._patcher_client.start()

        type(mock_session.return_value).region_name = PropertyMock(return_value="us-east-1")

    def tearDown(self):
        self._patcher_client.stop()
        self._patcher_session.stop()
        self._patcher_set_output.stop()

    @patch("amzn_nova_forge.evaluator.forge_evaluator.load_existing_result")
    def test_cached_result_short_circuits_evaluate(self, mock_load):
        mock_cached = MagicMock(spec=SMTJEvaluationResult)
        mock_load.return_value = mock_cached

        evaluator = ForgeEvaluator(
            model=self.model,
            infra=self.mock_infra,
            config=ForgeConfig(enable_job_caching=True),
        )
        result = evaluator.evaluate(job_name="cached-eval", eval_task=EvaluationTask.MMLU)

        self.assertIs(result, mock_cached)
        self.mock_infra.execute.assert_not_called()
        mock_load.assert_called_once()

    @patch("amzn_nova_forge.evaluator.forge_evaluator.persist_result")
    @patch("amzn_nova_forge.evaluator.forge_evaluator.validate_platform_compatibility")
    @patch("amzn_nova_forge.evaluator.forge_evaluator.detect_platform_from_path")
    @patch(
        "amzn_nova_forge.evaluator.forge_evaluator.resolve_model_checkpoint_path",
        return_value="s3://bucket/checkpoint",
    )
    @patch("amzn_nova_forge.evaluator.forge_evaluator.RecipeBuilder")
    @patch(
        "amzn_nova_forge.evaluator.forge_evaluator.load_existing_result",
        return_value=None,
    )
    def test_persist_called_after_successful_evaluate(
        self,
        mock_load,
        mock_recipe_cls,
        mock_resolve,
        mock_detect,
        mock_validate_compat,
        mock_persist,
    ):
        mock_detect.return_value = Platform.SMTJ
        mock_builder = MagicMock()
        mock_builder.build_and_validate.return_value = (
            "/tmp/recipe.yaml",
            "s3://bucket/output",
            "s3://bucket/data",
            "image:latest",
        )
        mock_recipe_cls.return_value = mock_builder

        evaluator = ForgeEvaluator(
            model=self.model,
            infra=self.mock_infra,
            config=ForgeConfig(enable_job_caching=True),
        )
        evaluator.evaluate(job_name="test-eval", eval_task=EvaluationTask.MMLU)

        mock_persist.assert_called_once()
        call_kwargs = mock_persist.call_args
        self.assertEqual(call_kwargs[1]["job_name"], "test-eval")
        self.assertEqual(call_kwargs[1]["job_type"], "eval")

    @patch("amzn_nova_forge.evaluator.forge_evaluator.load_existing_result")
    def test_cache_load_receives_model_path(self, mock_load):
        mock_load.return_value = MagicMock(spec=SMTJEvaluationResult)

        evaluator = ForgeEvaluator(
            model=self.model,
            infra=self.mock_infra,
            config=ForgeConfig(enable_job_caching=True),
        )
        evaluator.evaluate(
            job_name="test-eval",
            eval_task=EvaluationTask.MMLU,
            model_path="s3://bucket/my-checkpoint",
        )

        call_kwargs = mock_load.call_args[1]
        self.assertEqual(call_kwargs["model_path"], "s3://bucket/my-checkpoint")


class TestForgeEvaluatorGetLogs(unittest.TestCase):
    """Tests for ForgeEvaluator.get_logs()."""

    def setUp(self):
        self.mock_infra = create_autospec(SMTJRuntimeManager)
        self.mock_infra.kms_key_id = None
        self.mock_infra.instance_type = "ml.p5.48xlarge"
        self.mock_infra.instance_count = 2
        self.mock_infra.platform = Platform.SMTJ

        self._patcher_set_output = patch(
            "amzn_nova_forge.evaluator.forge_evaluator.set_output_s3_path",
            return_value="s3://bucket/output",
        )
        self._patcher_session = patch("boto3.session.Session")
        self._patcher_client = patch("boto3.client")

        self._patcher_set_output.start()
        mock_session = self._patcher_session.start()
        self._patcher_client.start()

        type(mock_session.return_value).region_name = PropertyMock(return_value="us-east-1")
        self.evaluator = ForgeEvaluator(
            model=Model.NOVA_MICRO,
            infra=self.mock_infra,
        )

    def tearDown(self):
        self._patcher_client.stop()
        self._patcher_session.stop()
        self._patcher_set_output.stop()

    @patch("amzn_nova_forge.evaluator.forge_evaluator.CloudWatchLogMonitor")
    def test_get_logs_with_job_result(self, mock_monitor_cls):
        mock_monitor = MagicMock()
        mock_monitor_cls.return_value = mock_monitor

        started = datetime(2025, 1, 1, tzinfo=timezone.utc)
        job_result = MagicMock()
        job_result.job_id = "eval-job-123"
        job_result.started_time = started

        self.evaluator.get_logs(job_result=job_result)

        mock_monitor_cls.assert_called_once_with(
            job_id="eval-job-123",
            platform=Platform.SMTJ,
            started_time=int(started.timestamp() * 1000),
            region="us-east-1",
        )
        mock_monitor.show_logs.assert_called_once_with(
            limit=None, start_from_head=False, end_time=None
        )

    @patch("amzn_nova_forge.evaluator.forge_evaluator.CloudWatchLogMonitor")
    def test_get_logs_missing_params_raises_value_error(self, mock_monitor_cls):
        with self.assertRaises(ValueError) as ctx:
            self.evaluator.get_logs()

        self.assertIn("job_result", str(ctx.exception))
        mock_monitor_cls.assert_not_called()

    @patch("amzn_nova_forge.evaluator.forge_evaluator.CloudWatchLogMonitor")
    def test_get_logs_job_id_only_raises_value_error(self, mock_monitor_cls):
        with self.assertRaises(ValueError) as ctx:
            self.evaluator.get_logs(job_id="some-job")

        self.assertIn("job_result", str(ctx.exception))
        mock_monitor_cls.assert_not_called()

    @patch("amzn_nova_forge.evaluator.forge_evaluator.CloudWatchLogMonitor")
    def test_get_logs_started_time_only_raises_value_error(self, mock_monitor_cls):
        with self.assertRaises(ValueError) as ctx:
            self.evaluator.get_logs(started_time=datetime(2025, 1, 1, tzinfo=timezone.utc))

        self.assertIn("job_result", str(ctx.exception))
        mock_monitor_cls.assert_not_called()

    @patch("amzn_nova_forge.evaluator.forge_evaluator.CloudWatchLogMonitor")
    def test_get_logs_smhp_includes_cluster_namespace(self, mock_monitor_cls):
        mock_monitor = MagicMock()
        mock_monitor_cls.return_value = mock_monitor

        smhp_infra = create_autospec(SMHPRuntimeManager)
        smhp_infra.kms_key_id = None
        smhp_infra.instance_type = "ml.p5.48xlarge"
        smhp_infra.instance_count = 2
        smhp_infra.platform = Platform.SMHP
        smhp_infra.cluster_name = "my-cluster"
        smhp_infra.namespace = "kubeflow"

        self.evaluator.infra = smhp_infra
        self.evaluator._platform = Platform.SMHP

        started = datetime(2025, 1, 1, tzinfo=timezone.utc)
        job_result = MagicMock()
        job_result.job_id = "smhp-eval-789"
        job_result.started_time = started

        self.evaluator.get_logs(job_result=job_result)

        mock_monitor_cls.assert_called_once_with(
            job_id="smhp-eval-789",
            platform=Platform.SMHP,
            started_time=int(started.timestamp() * 1000),
            cluster_name="my-cluster",
            namespace="kubeflow",
            region="us-east-1",
        )


class TestForgeEvaluatorInspectLens(unittest.TestCase):
    """Tests for ForgeEvaluator InspectLens path."""

    def _make_evaluator(self, image_uri=None):
        mock_infra = create_autospec(SMTJRuntimeManager)
        mock_infra.kms_key_id = None
        mock_infra.instance_type = "ml.m5.large"
        mock_infra.instance_count = 1
        mock_infra.platform = Platform.SMTJ
        mock_infra.execution_role = "arn:aws:iam::123:role/MyRole"
        mock_infra.max_job_runtime = 7200

        config = ForgeConfig(
            output_s3_path="s3://bucket/output/",
            image_uri=image_uri
            or "123456789012.dkr.ecr.us-east-1.amazonaws.com/inspect-lens:beta-latest",
        )

        with (
            patch(
                "amzn_nova_forge.evaluator.forge_evaluator.set_output_s3_path",
                return_value="s3://bucket/output/",
            ),
            patch("boto3.session.Session") as mock_session,
            patch("boto3.client"),
        ):
            type(mock_session.return_value).region_name = PropertyMock(return_value="us-east-1")
            evaluator = ForgeEvaluator(
                model=Model.NOVA_MICRO,
                infra=mock_infra,
                config=config,
            )
        return evaluator

    def test_dry_run_returns_none(self):
        evaluator = self._make_evaluator()
        config = InspectLensConfig(
            benchmarks_path="s3://bucket/benchmarks/boolq/",
            tasks=[{"name": "boolq_pt", "limit": 50}],
            output_s3_path="s3://bucket/results/",
        )
        with (
            patch("boto3.client"),
            patch("amzn_nova_forge.evaluator.forge_evaluator.yaml") as mock_yaml,
        ):
            mock_yaml.dump.return_value = "inference_provider: {}\n"
            result = evaluator.evaluate(
                job_name="test-job",
                eval_task=EvaluationTask.INSPECT_LENS,
                inspect_lens_config=config,
                dry_run=True,
            )
        self.assertIsNone(result)

    def test_missing_inspect_lens_config_raises(self):
        evaluator = self._make_evaluator()
        with self.assertRaises(ValueError) as ctx:
            evaluator.evaluate(
                job_name="test-job",
                eval_task=EvaluationTask.INSPECT_LENS,
                inspect_lens_config=None,
            )
        self.assertIn("inspect_lens_config is required", str(ctx.exception))

    def test_overrides_warning_for_non_decoding_keys(self):
        evaluator = self._make_evaluator()
        config = InspectLensConfig(
            benchmarks_path="s3://bucket/benchmarks/boolq/",
            output_s3_path="s3://bucket/results/",
        )
        with (
            patch("boto3.client"),
            patch("amzn_nova_forge.evaluator.forge_evaluator.yaml") as mock_yaml,
            patch("amzn_nova_forge.evaluator.forge_evaluator.logger") as mock_logger,
        ):
            mock_yaml.dump.return_value = "inference_provider: {}\n"
            evaluator.evaluate(
                job_name="test-job",
                eval_task=EvaluationTask.INSPECT_LENS,
                inspect_lens_config=config,
                overrides={"benchmarks_path": "s3://other/"},
                dry_run=True,
            )
        # Should warn about unknown key
        warning_calls = [str(c) for c in mock_logger.warning.call_args_list]
        self.assertTrue(any("benchmarks_path" in w for w in warning_calls))

    def test_valid_decoding_overrides_no_warning(self):
        evaluator = self._make_evaluator()
        config = InspectLensConfig(
            benchmarks_path="s3://bucket/benchmarks/boolq/",
            output_s3_path="s3://bucket/results/",
        )
        with (
            patch("boto3.client"),
            patch("amzn_nova_forge.evaluator.forge_evaluator.yaml") as mock_yaml,
            patch("amzn_nova_forge.evaluator.forge_evaluator.logger") as mock_logger,
        ):
            mock_yaml.dump.return_value = "inference_provider: {}\n"
            evaluator.evaluate(
                job_name="test-job",
                eval_task=EvaluationTask.INSPECT_LENS,
                inspect_lens_config=config,
                overrides={"temperature": 0.0, "max_tokens": 512, "max_connections": 4},
                dry_run=True,
            )
        # No warning for valid decoding keys
        warning_calls = [str(c) for c in mock_logger.warning.call_args_list]
        self.assertFalse(any("overrides" in w for w in warning_calls))

    def test_inference_provider_bedrock_default(self):
        evaluator = self._make_evaluator()
        config = InspectLensConfig(
            benchmarks_path="s3://bucket/benchmarks/",
            output_s3_path="s3://bucket/results/",
        )
        provider = evaluator._build_inspect_lens_inference_provider(config, model_path=None)
        self.assertIn("bedrock", provider)
        self.assertIn("us.amazon.nova-micro-v1:0", provider["bedrock"]["model_id"])

    def test_inference_provider_existing_endpoint(self):
        evaluator = self._make_evaluator()
        config = InspectLensConfig(
            benchmarks_path="s3://bucket/benchmarks/",
            endpoint_name="my-endpoint",
        )
        provider = evaluator._build_inspect_lens_inference_provider(config, model_path=None)
        self.assertIn("sagemaker_endpoint", provider)
        self.assertEqual(provider["sagemaker_endpoint"]["endpoint_name"], "my-endpoint")

    def test_inference_provider_model_path_overrides_bedrock(self):
        evaluator = self._make_evaluator()
        config = InspectLensConfig(
            benchmarks_path="s3://bucket/benchmarks/",
            output_s3_path="s3://bucket/results/",
        )
        provider = evaluator._build_inspect_lens_inference_provider(
            config, model_path="us.amazon.nova-lite-v1:0"
        )
        self.assertIn("bedrock", provider)
        self.assertEqual(provider["bedrock"]["model_id"], "us.amazon.nova-lite-v1:0")

    def test_cache_hit_returns_cached_result(self):
        """When job caching is enabled and a matching result exists, return it without submitting."""
        evaluator = self._make_evaluator()
        config = InspectLensConfig(
            benchmarks_path="s3://bucket/benchmarks/boolq/",
            tasks=[{"name": "boolq_pt", "limit": 50}],
            output_s3_path="s3://bucket/results/",
        )
        mock_cached = MagicMock()

        with patch(
            "amzn_nova_forge.evaluator.forge_evaluator.load_existing_result",
            return_value=mock_cached,
        ) as mock_load:
            result = evaluator.evaluate(
                job_name="test-job",
                eval_task=EvaluationTask.INSPECT_LENS,
                inspect_lens_config=config,
            )

        self.assertIs(result, mock_cached)
        mock_load.assert_called_once_with(
            evaluator._cache_context,
            job_name="test-job",
            job_type="inspect_lens",
            model_path=None,
            benchmarks_path="s3://bucket/benchmarks/boolq/",
            tasks=str([{"name": "boolq_pt", "limit": 50}]),
            inference_scenario="bedrock",
            endpoint_name=None,
            bedrock_model_id=None,
            overrides={},
        )

    def test_mlflow_tracking_injected_into_config_dict(self):
        """When ForgeConfig.mlflow_monitor is set, tracking section must appear in the YAML dict."""
        mock_infra = create_autospec(SMTJRuntimeManager)
        mock_infra.kms_key_id = None
        mock_infra.instance_type = "ml.m5.large"
        mock_infra.instance_count = 1
        mock_infra.platform = Platform.SMTJ
        mock_infra.execution_role = "arn:aws:iam::123:role/MyRole"
        mock_infra.max_job_runtime = 7200

        # Patch validate_mlflow_overrides to avoid real AWS calls during MLflowMonitor init
        with patch(
            "amzn_nova_forge.monitor.mlflow_monitor.validate_mlflow_overrides",
            return_value=[],
        ):
            mlflow_monitor = MLflowMonitor(
                tracking_uri="arn:aws:sagemaker:us-east-1:123456789012:mlflow-app/app-xxx",
                experiment_name="nova-evals",
            )

        forge_config = ForgeConfig(
            output_s3_path="s3://bucket/output/",
            image_uri="123456789012.dkr.ecr.us-east-1.amazonaws.com/inspect-lens:beta-latest",
            mlflow_monitor=mlflow_monitor,
        )

        with (
            patch(
                "amzn_nova_forge.evaluator.forge_evaluator.set_output_s3_path",
                return_value="s3://bucket/output/",
            ),
            patch("boto3.session.Session") as mock_session,
            patch("boto3.client"),
        ):
            type(mock_session.return_value).region_name = PropertyMock(return_value="us-east-1")
            evaluator = ForgeEvaluator(
                model=Model.NOVA_MICRO,
                infra=mock_infra,
                config=forge_config,
            )

        inspect_config = InspectLensConfig(
            benchmarks_path="s3://bucket/benchmarks/",
            output_s3_path="s3://bucket/results/",
        )

        captured_dict = {}

        def capture_yaml_dump(d, **kwargs):
            captured_dict.update(d)
            return "mocked_yaml\n"

        with (
            patch("boto3.client"),
            patch("amzn_nova_forge.evaluator.forge_evaluator.yaml") as mock_yaml,
        ):
            mock_yaml.dump.side_effect = capture_yaml_dump
            evaluator.evaluate(
                job_name="test-mlflow-job",
                eval_task=EvaluationTask.INSPECT_LENS,
                inspect_lens_config=inspect_config,
                dry_run=True,
            )

        self.assertIn("tracking", captured_dict)
        tracking = captured_dict["tracking"]
        self.assertEqual(
            tracking["mlflow_tracking_arn"],
            "arn:aws:sagemaker:us-east-1:123456789012:mlflow-app/app-xxx",
        )
        self.assertEqual(tracking["mlflow_experiment_name"], "nova-evals")
        self.assertTrue(tracking["mlflow_tracing"])
        self.assertTrue(tracking["mlflow_log_artifacts"])


class TestInspectLensS3Paths(unittest.TestCase):
    """Tests for run_id-based S3 path layout in _evaluate_inspect_lens."""

    FIXED_RUN_ID = "aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee"

    def _make_evaluator(self, output_s3_path="s3://bucket/output/"):
        mock_infra = create_autospec(SMTJRuntimeManager)
        mock_infra.kms_key_id = None
        mock_infra.instance_type = "ml.m5.large"
        mock_infra.instance_count = 1
        mock_infra.platform = Platform.SMTJ
        mock_infra.execution_role = "arn:aws:iam::123:role/MyRole"
        mock_infra.max_job_runtime = 7200

        config = ForgeConfig(
            output_s3_path=output_s3_path,
            image_uri="123456789012.dkr.ecr.us-east-1.amazonaws.com/inspect-lens:beta-latest",
        )

        with (
            patch(
                "amzn_nova_forge.evaluator.forge_evaluator.set_output_s3_path",
                return_value=output_s3_path,
            ),
            patch("boto3.session.Session") as mock_session,
            patch("boto3.client"),
        ):
            type(mock_session.return_value).region_name = PropertyMock(return_value="us-east-1")
            evaluator = ForgeEvaluator(
                model=Model.NOVA_MICRO,
                infra=mock_infra,
                config=config,
            )
        return evaluator

    def _run_evaluate(self, evaluator, inspect_lens_config, mock_sm_client):
        """Helper: run evaluate() with all external calls mocked."""
        mock_sm_client.create_training_job.return_value = {}
        with (
            patch("amzn_nova_forge.evaluator.forge_evaluator.uuid") as mock_uuid,
            patch("amzn_nova_forge.evaluator.forge_evaluator.yaml") as mock_yaml,
            patch("amzn_nova_forge.evaluator.forge_evaluator.boto3") as mock_boto3,
        ):
            mock_uuid.uuid4.return_value = self.FIXED_RUN_ID
            mock_yaml.dump.return_value = "inference_provider: {}\n"
            mock_boto3.client.return_value = mock_sm_client
            result = evaluator.evaluate(
                job_name="test-job",
                eval_task=EvaluationTask.INSPECT_LENS,
                inspect_lens_config=inspect_lens_config,
            )
        return result, mock_sm_client

    def test_local_benchmarks_path_raises_valueerror(self):
        """Local benchmarks_path should raise ValueError — user must upload first."""
        evaluator = self._make_evaluator(output_s3_path="s3://bucket/output/")

        with tempfile.TemporaryDirectory() as tmpdir:
            with open(os.path.join(tmpdir, "boolq_pt.py"), "w") as f:
                f.write("@task\ndef boolq_pt():\n    pass\n")

            # Bypass InspectLensConfig validation to test the evaluator-level check
            cfg = object.__new__(InspectLensConfig)
            for field, default in InspectLensConfig.__dataclass_fields__.items():
                try:
                    setattr(cfg, field, default.default)
                except Exception:
                    setattr(cfg, field, default.default_factory())
            cfg.benchmarks_path = tmpdir
            cfg.tasks = [{"name": "boolq_pt"}]

            with (
                patch("amzn_nova_forge.evaluator.forge_evaluator.boto3") as mock_boto3,
            ):
                mock_boto3.client.return_value = MagicMock()
                with self.assertRaises(ValueError) as ctx:
                    evaluator.evaluate(
                        job_name="test-job",
                        eval_task=EvaluationTask.INSPECT_LENS,
                        inspect_lens_config=cfg,
                    )
                self.assertIn("upload_benchmarks", str(ctx.exception))

    def test_upload_benchmarks(self):
        """upload_benchmarks() uploads .py files to S3 and returns the S3 URI."""
        evaluator = self._make_evaluator(output_s3_path="s3://bucket/output/")

        with tempfile.TemporaryDirectory() as tmpdir:
            with open(os.path.join(tmpdir, "boolq_pt.py"), "w") as f:
                f.write("@task\ndef boolq_pt():\n    pass\n")
            with open(os.path.join(tmpdir, "README.md"), "w") as f:
                f.write("# not uploaded\n")

            with patch("amzn_nova_forge.evaluator.forge_evaluator.boto3") as mock_boto3:
                mock_s3 = MagicMock()
                mock_boto3.client.return_value = mock_s3

                result = evaluator.upload_benchmarks(tmpdir, "s3://bucket/my-benchmarks/")

            self.assertEqual(result, "s3://bucket/my-benchmarks/")
            mock_s3.upload_file.assert_called_once()
            call_args = mock_s3.upload_file.call_args
            self.assertIn("boolq_pt.py", call_args[0][0])
            self.assertEqual(call_args[0][1], "bucket")
            self.assertEqual(call_args[0][2], "my-benchmarks/boolq_pt.py")

    def test_upload_benchmarks_invalid_local_dir_raises(self):
        """upload_benchmarks() raises ValueError for non-existent directory."""
        evaluator = self._make_evaluator()
        with self.assertRaises(ValueError) as ctx:
            evaluator.upload_benchmarks("/nonexistent/path", "s3://bucket/benchmarks/")
        self.assertIn("not an existing directory", str(ctx.exception))

    def test_upload_benchmarks_invalid_s3_path_raises(self):
        """upload_benchmarks() raises ValueError for non-S3 path."""
        evaluator = self._make_evaluator()
        with tempfile.TemporaryDirectory() as tmpdir:
            with self.assertRaises(ValueError) as ctx:
                evaluator.upload_benchmarks(tmpdir, "/local/path/")
            self.assertIn("s3://", str(ctx.exception))

    def test_s3_benchmarks_config_colocated_with_benchmarks(self):
        """S3 benchmarks_path in a different bucket → config and output still under output_s3_path."""
        evaluator = self._make_evaluator(output_s3_path="s3://bucket/output/")

        cfg = InspectLensConfig(
            benchmarks_path="s3://separate-benchmarks-bucket/my-project/benchmarks/my_benchmarks/",
            tasks=[{"name": "boolq_pt"}],
        )

        with (
            patch("amzn_nova_forge.evaluator.forge_evaluator.uuid") as mock_uuid,
            patch("amzn_nova_forge.evaluator.forge_evaluator.yaml") as mock_yaml,
            patch("amzn_nova_forge.evaluator.forge_evaluator.boto3") as mock_boto3,
        ):
            mock_uuid.uuid4.return_value = self.FIXED_RUN_ID
            mock_yaml.dump.return_value = "inference_provider: {}\n"
            mock_s3 = MagicMock()
            mock_boto3.client.return_value = mock_s3
            mock_s3.create_training_job = MagicMock(return_value={})

            evaluator.evaluate(
                job_name="test-job",
                eval_task=EvaluationTask.INSPECT_LENS,
                inspect_lens_config=cfg,
            )

        # Config always goes under output_s3_path/<run_id>/config/ — not the separate benchmarks bucket
        put_calls = mock_s3.put_object.call_args_list
        config_call = next(c for c in put_calls if "inspect_config.yaml" in str(c))
        kwargs = config_call[1]
        self.assertEqual(kwargs["Bucket"], "bucket")
        self.assertEqual(kwargs["Key"], f"output/{self.FIXED_RUN_ID}/config/inspect_config.yaml")

        # SageMaker output also under output_s3_path/<run_id>/output/
        create_call = mock_s3.create_training_job.call_args
        output_path = create_call[1]["OutputDataConfig"]["S3OutputPath"]
        self.assertEqual(output_path, f"s3://bucket/output/{self.FIXED_RUN_ID}/output/")

    def test_run_id_in_job_name(self):
        """unique_job_name should contain the run_id UUID."""
        evaluator = self._make_evaluator()

        cfg = InspectLensConfig(
            benchmarks_path="s3://bucket/benchmarks/",
            tasks=[{"name": "boolq_pt"}],
        )

        with (
            patch("amzn_nova_forge.evaluator.forge_evaluator.uuid") as mock_uuid,
            patch("amzn_nova_forge.evaluator.forge_evaluator.yaml") as mock_yaml,
            patch("amzn_nova_forge.evaluator.forge_evaluator.boto3") as mock_boto3,
        ):
            mock_uuid.uuid4.return_value = self.FIXED_RUN_ID
            mock_yaml.dump.return_value = "inference_provider: {}\n"
            mock_s3 = MagicMock()
            mock_boto3.client.return_value = mock_s3
            mock_s3.create_training_job = MagicMock(return_value={})

            result = evaluator.evaluate(
                job_name="my-eval",
                eval_task=EvaluationTask.INSPECT_LENS,
                inspect_lens_config=cfg,
            )

        self.assertIn(self.FIXED_RUN_ID, result.job_id)
        self.assertTrue(result.job_id.startswith("my-eval-"))

    def test_s3_benchmarks_bucket_only_path(self):
        """S3 benchmarks_path in a separate bucket → output still goes under output_s3_path."""
        evaluator = self._make_evaluator(output_s3_path="s3://bucket/output/")

        cfg = InspectLensConfig(
            benchmarks_path="s3://separate-benchmarks-bucket/benchmarks/",
            tasks=[{"name": "boolq_pt"}],
        )

        with (
            patch("amzn_nova_forge.evaluator.forge_evaluator.uuid") as mock_uuid,
            patch("amzn_nova_forge.evaluator.forge_evaluator.yaml") as mock_yaml,
            patch("amzn_nova_forge.evaluator.forge_evaluator.boto3") as mock_boto3,
        ):
            mock_uuid.uuid4.return_value = self.FIXED_RUN_ID
            mock_yaml.dump.return_value = "inference_provider: {}\n"
            mock_s3 = MagicMock()
            mock_boto3.client.return_value = mock_s3
            mock_s3.create_training_job = MagicMock(return_value={})

            evaluator.evaluate(
                job_name="test-job",
                eval_task=EvaluationTask.INSPECT_LENS,
                inspect_lens_config=cfg,
            )

        # parent of s3://separate-benchmarks-bucket/benchmarks/ → s3://separate-benchmarks-bucket/
        create_call = mock_s3.create_training_job.call_args
        output_path = create_call[1]["OutputDataConfig"]["S3OutputPath"]
        # Output always goes under output_s3_path, not the benchmarks bucket
        self.assertEqual(output_path, f"s3://bucket/output/{self.FIXED_RUN_ID}/output/")


class TestForgeEvaluatorMTRL(unittest.TestCase):
    """Tests for ForgeEvaluator._execute_mtrl_eval via evaluate()."""

    def setUp(self):
        self.model = Model.NOVA_LITE_2
        self.mock_infra = create_autospec(SMTJServerlessRuntimeManager)
        self.mock_infra.kms_key_id = None
        self.mock_infra.instance_type = None
        self.mock_infra.instance_count = None
        self.mock_infra.platform = Platform.SMTJServerless
        self.mock_infra.rft_lambda_arn = None
        self.mock_infra.hub_content_version = None

        # Mock execute_mtrl_eval return value
        mock_execution = MagicMock()
        mock_execution.arn = (
            "arn:aws:sagemaker:us-east-1:123456789012:pipeline-execution/mtrl-eval-exec"
        )
        self.mock_infra.execute_mtrl_eval.return_value = mock_execution

        self._patcher_set_output = patch(
            "amzn_nova_forge.evaluator.forge_evaluator.set_output_s3_path",
            return_value="s3://bucket/output",
        )
        self._patcher_session = patch("boto3.session.Session")
        self._patcher_client = patch("boto3.client")

        self._patcher_set_output.start()
        mock_session = self._patcher_session.start()
        self._patcher_client.start()

        type(mock_session.return_value).region_name = PropertyMock(return_value="us-east-1")

        mock_mlflow = MagicMock(spec=MLflowMonitor)
        mock_mlflow.tracking_uri = (
            "arn:aws:sagemaker:us-east-1:123456789012:mlflow-tracking-server/my-server"
        )

        self.evaluator = ForgeEvaluator(
            model=self.model,
            infra=self.mock_infra,
            data_s3_path="s3://bucket/data",
            config=ForgeConfig(mlflow_monitor=mock_mlflow),
        )

    def tearDown(self):
        self._patcher_client.stop()
        self._patcher_session.stop()
        self._patcher_set_output.stop()

    def test_mtrl_eval_resolves_model_path_from_job_result(self):
        """_execute_mtrl_eval resolves model_path from MTRL job_result's output_model_arn."""
        mock_job_result = MagicMock(spec=SMTJTrainingResult)
        mock_job_result._is_mtrl = True
        mock_job_result.job_id = "mtrl-train-123"
        mock_job_result.model_artifacts = ModelArtifacts(
            output_s3_path="s3://bucket/output/",
            output_model_arn="arn:aws:sagemaker:us-east-1:123456789012:model-package/my-group/1",
        )

        result = self.evaluator.evaluate(
            job_name="test-mtrl-eval",
            eval_task=EvaluationTask.RFT_MULTITURN_EVAL,
            job_result=mock_job_result,
        )

        self.mock_infra.execute_mtrl_eval.assert_called_once()
        call_kwargs = self.mock_infra.execute_mtrl_eval.call_args[1]
        self.assertEqual(
            call_kwargs["model_path"],
            "arn:aws:sagemaker:us-east-1:123456789012:model-package/my-group/1",
        )
        self.assertEqual(call_kwargs["training_job_name"], "mtrl-train-123")
        self.assertIsInstance(result, SMTJEvaluationResult)

    def test_mtrl_eval_uses_explicit_model_path(self):
        """_execute_mtrl_eval uses explicit model_path when provided directly."""
        explicit_arn = "arn:aws:sagemaker:us-east-1:123456789012:model-package/explicit-group/2"

        result = self.evaluator.evaluate(
            job_name="test-mtrl-eval-explicit",
            eval_task=EvaluationTask.RFT_MULTITURN_EVAL,
            model_path=explicit_arn,
        )

        self.mock_infra.execute_mtrl_eval.assert_called_once()
        call_kwargs = self.mock_infra.execute_mtrl_eval.call_args[1]
        self.assertEqual(call_kwargs["model_path"], explicit_arn)
        self.assertIsNone(call_kwargs["training_job_name"])
        self.assertIsInstance(result, SMTJEvaluationResult)

    def test_mtrl_eval_dry_run_returns_none(self):
        """_execute_mtrl_eval with dry_run=True returns None without calling infra."""
        result = self.evaluator.evaluate(
            job_name="test-mtrl-eval-dryrun",
            eval_task=EvaluationTask.RFT_MULTITURN_EVAL,
            dry_run=True,
        )

        self.assertIsNone(result)
        self.mock_infra.execute_mtrl_eval.assert_not_called()

    def test_mtrl_eval_evaluate_base_model_passed_to_infra(self):
        """evaluate_base_model from EvalTaskConfig is passed to execute_mtrl_eval."""
        model_path = "arn:aws:sagemaker:us-east-1:123456789012:model-package/rmp/1"

        result = self.evaluator.evaluate(
            job_name="test-mtrl-eval-both",
            eval_task=EvaluationTask.RFT_MULTITURN_EVAL,
            model_path=model_path,
            task_config=EvalTaskConfig(evaluate_base_model=True),
        )

        self.mock_infra.execute_mtrl_eval.assert_called_once()
        call_kwargs = self.mock_infra.execute_mtrl_eval.call_args[1]
        self.assertEqual(call_kwargs["model_path"], model_path)
        self.assertTrue(call_kwargs["evaluate_base_model"])
        self.assertIsInstance(result, SMTJEvaluationResult)

    def test_mtrl_eval_evaluate_base_model_defaults_to_false(self):
        """evaluate_base_model defaults to False when not in task_config."""
        model_path = "arn:aws:sagemaker:us-east-1:123456789012:model-package/rmp/1"

        result = self.evaluator.evaluate(
            job_name="test-mtrl-eval-ft-only",
            eval_task=EvaluationTask.RFT_MULTITURN_EVAL,
            model_path=model_path,
        )

        self.mock_infra.execute_mtrl_eval.assert_called_once()
        call_kwargs = self.mock_infra.execute_mtrl_eval.call_args[1]
        self.assertFalse(call_kwargs["evaluate_base_model"])

    def test_evaluate_base_model_only_used_for_mtrl(self):
        """evaluate_base_model in EvalTaskConfig is only extracted for RFT_MULTITURN_EVAL."""
        # When using MTRL without evaluate_base_model, it should default to False
        self.evaluator.evaluate(
            job_name="test-mtrl-no-base",
            eval_task=EvaluationTask.RFT_MULTITURN_EVAL,
        )

        call_kwargs = self.mock_infra.execute_mtrl_eval.call_args[1]
        self.assertFalse(call_kwargs["evaluate_base_model"])

        # When using MTRL WITH evaluate_base_model=True, it should be True
        self.mock_infra.execute_mtrl_eval.reset_mock()
        self.evaluator.evaluate(
            job_name="test-mtrl-with-base",
            eval_task=EvaluationTask.RFT_MULTITURN_EVAL,
            model_path="arn:aws:sagemaker:us-east-1:123:model-package/rmp/1",
            task_config=EvalTaskConfig(evaluate_base_model=True),
        )

        call_kwargs = self.mock_infra.execute_mtrl_eval.call_args[1]
        self.assertTrue(call_kwargs["evaluate_base_model"])

    @patch(
        "amzn_nova_forge.evaluator.forge_evaluator.set_output_s3_path",
        return_value="s3://bucket/output",
    )
    @patch("boto3.client")
    @patch("boto3.session.Session")
    def test_mtrl_eval_raises_without_mlflow_config(self, mock_session, _mock_client, _mock_output):
        """AgentRFT eval jobs must have an MLflow config; raise ValueError if missing."""
        type(mock_session.return_value).region_name = PropertyMock(return_value="us-east-1")
        mock_infra = create_autospec(SMTJServerlessRuntimeManager)
        mock_infra.kms_key_id = None
        mock_infra.instance_type = None
        mock_infra.instance_count = None
        mock_infra.platform = Platform.SMTJServerless
        mock_infra.rft_lambda_arn = None
        mock_infra.hub_content_version = None

        evaluator = ForgeEvaluator(
            model=Model.NOVA_LITE_2,
            infra=mock_infra,
            data_s3_path="s3://bucket/data",
            config=ForgeConfig(),  # No mlflow_monitor
        )

        with self.assertRaises(ValueError) as ctx:
            evaluator.evaluate(
                job_name="test-mtrl-eval-no-mlflow",
                eval_task=EvaluationTask.RFT_MULTITURN_EVAL,
                model_path="arn:aws:sagemaker:us-east-1:123:model-package/rmp/1",
            )
        self.assertIn("MLflow configuration is required", str(ctx.exception))

    @patch(
        "amzn_nova_forge.evaluator.forge_evaluator.set_output_s3_path",
        return_value="s3://bucket/output",
    )
    @patch("boto3.client")
    @patch("boto3.session.Session")
    def test_mtrl_eval_raises_without_mlflow_tracking_uri(
        self, mock_session, _mock_client, _mock_output
    ):
        """AgentRFT eval jobs must have a tracking_uri; raise ValueError if None."""
        type(mock_session.return_value).region_name = PropertyMock(return_value="us-east-1")
        mock_infra = create_autospec(SMTJServerlessRuntimeManager)
        mock_infra.kms_key_id = None
        mock_infra.instance_type = None
        mock_infra.instance_count = None
        mock_infra.platform = Platform.SMTJServerless
        mock_infra.rft_lambda_arn = None
        mock_infra.hub_content_version = None

        mock_monitor = MagicMock(spec=MLflowMonitor)
        mock_monitor.tracking_uri = None

        evaluator = ForgeEvaluator(
            model=Model.NOVA_LITE_2,
            infra=mock_infra,
            data_s3_path="s3://bucket/data",
            config=ForgeConfig(mlflow_monitor=mock_monitor),
        )

        with self.assertRaises(ValueError) as ctx:
            evaluator.evaluate(
                job_name="test-mtrl-eval-no-tracking-uri",
                eval_task=EvaluationTask.RFT_MULTITURN_EVAL,
                model_path="arn:aws:sagemaker:us-east-1:123:model-package/rmp/1",
            )
        self.assertIn("MLflow configuration is required", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
