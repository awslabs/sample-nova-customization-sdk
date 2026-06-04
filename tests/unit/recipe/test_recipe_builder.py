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
from unittest.mock import MagicMock, Mock, patch

import pytest
import yaml

from amzn_nova_forge.core.enums import (
    EvaluationTask,
    Model,
    Platform,
    TrainingMethod,
    Version,
)
from amzn_nova_forge.core.types import ValidationConfig
from amzn_nova_forge.manager.runtime_manager import RuntimeManager
from amzn_nova_forge.monitor import MLflowMonitor
from amzn_nova_forge.recipe.recipe_builder import RecipeBuilder
from amzn_nova_forge.util.data_mixing import DataMixing


class TestRecipeBuilder(unittest.TestCase):
    def setUp(self):
        self.region = "us-east-1"
        self.job_name = "job"
        self.platform = Platform.SMTJ
        self.method = TrainingMethod.SFT_LORA
        self.instance_type = "ml.g5.12xlarge"
        self.instance_count = 1
        self.data_s3 = "s3://bucket/data"
        self.output_s3 = "s3://bucket/output"

        self.mock_model = Mock(spec=Model)
        self.mock_model.name = "nova-micro"
        self.mock_model.value = "nova_micro"
        self.mock_model.version = Version.ONE
        self.mock_model.model_type = "test-model"
        self.mock_model.model_path = "models/test"

        self.mock_infra = Mock(spec=RuntimeManager)
        self.mock_infra.instance_type = self.instance_type
        self.mock_infra.instance_count = self.instance_count

    def test_initialization(self):
        builder = RecipeBuilder(
            region=self.region,
            job_name=self.job_name,
            platform=self.platform,
            model=self.mock_model,
            method=self.method,
            instance_type=self.instance_type,
            instance_count=self.instance_count,
            infra=self.mock_infra,
            output_s3_path=self.output_s3,
            data_s3_path=self.data_s3,
        )

        self.assertEqual(builder.region, self.region)
        self.assertEqual(builder.job_name, self.job_name)
        self.assertEqual(builder.platform, self.platform)
        self.assertEqual(builder.model, self.mock_model)
        self.assertEqual(builder.method, self.method)
        self.assertEqual(builder.model_type, "test-model")
        self.assertEqual(builder.model_name_or_path, "models/test")
        self.assertEqual(builder.data_s3_path, self.data_s3)
        self.assertEqual(builder.output_s3_path, self.output_s3)
        self.assertEqual(builder.mlflow_tracking_uri, None)

    def test_initialization_with_model_path_override(self):
        custom_model_path = "s3://bucket/custom-checkpoint"
        builder = RecipeBuilder(
            region=self.region,
            job_name=self.job_name,
            platform=self.platform,
            model=self.mock_model,
            method=self.method,
            instance_type=self.instance_type,
            instance_count=self.instance_count,
            infra=self.mock_infra,
            output_s3_path=self.output_s3,
            data_s3_path=self.data_s3,
            model_path=custom_model_path,
        )

        self.assertEqual(builder.model_name_or_path, custom_model_path)

    def test_initialization_with_mlflow_monitor(self):
        mock_mlflow = Mock(spec=MLflowMonitor)
        mock_mlflow.tracking_uri = "test-uri"
        mock_mlflow.experiment_name = "test-experiment"
        mock_mlflow.run_name = "test-run"

        builder = RecipeBuilder(
            region=self.region,
            job_name=self.job_name,
            platform=self.platform,
            model=self.mock_model,
            method=self.method,
            instance_type=self.instance_type,
            instance_count=self.instance_count,
            infra=self.mock_infra,
            output_s3_path=self.output_s3,
            data_s3_path=self.data_s3,
            mlflow_monitor=mock_mlflow,
        )

        self.assertEqual(builder.mlflow_tracking_uri, "test-uri")
        self.assertEqual(builder.mlflow_experiment_name, "test-experiment")
        self.assertEqual(builder.mlflow_run_name, "test-run")

    def test_initialization_with_mlflow_monitor_default_names(self):
        mock_mlflow = Mock(spec=MLflowMonitor)
        mock_mlflow.tracking_uri = "test-uri"
        mock_mlflow.experiment_name = None
        mock_mlflow.run_name = None

        builder = RecipeBuilder(
            region=self.region,
            job_name=self.job_name,
            platform=self.platform,
            model=self.mock_model,
            method=self.method,
            instance_type=self.instance_type,
            instance_count=self.instance_count,
            infra=self.mock_infra,
            output_s3_path=self.output_s3,
            data_s3_path=self.data_s3,
            mlflow_monitor=mock_mlflow,
        )

        self.assertEqual(builder.mlflow_experiment_name, self.job_name)
        self.assertTrue(builder.mlflow_run_name.startswith(self.job_name))

    def test_initialization_rft_method(self):
        rft_lambda = "arn:aws:lambda:us-east-1:123456789012:function:reward"
        builder = RecipeBuilder(
            region=self.region,
            job_name=self.job_name,
            platform=self.platform,
            model=self.mock_model,
            method=TrainingMethod.RFT_LORA,
            instance_type=self.instance_type,
            instance_count=self.instance_count,
            infra=self.mock_infra,
            output_s3_path=self.output_s3,
            data_s3_path=self.data_s3,
            rft_lambda_arn=rft_lambda,
        )

        self.assertEqual(builder.rft_lambda_arn, rft_lambda)

    @patch("amzn_nova_forge.recipe.recipe_builder.logger")
    def test_initialization_rft_lambda_arn_ignored_for_non_rft_method(self, mock_logger):
        rft_lambda = "arn:aws:lambda:us-east-1:123456789012:function:reward"
        builder = RecipeBuilder(
            region=self.region,
            job_name=self.job_name,
            platform=self.platform,
            model=self.mock_model,
            method=TrainingMethod.SFT_LORA,
            instance_type=self.instance_type,
            instance_count=self.instance_count,
            infra=self.mock_infra,
            output_s3_path=self.output_s3,
            data_s3_path=self.data_s3,
            rft_lambda_arn=rft_lambda,
        )

        mock_logger.info.assert_called_with(
            "'rft_lambda_arn' is only required for RFT. Will ignore."
        )

        self.assertFalse(hasattr(builder, "rft_lambda_arn"))

    def test_initialization_evaluation_method(self):
        builder = RecipeBuilder(
            region=self.region,
            job_name=self.job_name,
            platform=self.platform,
            model=self.mock_model,
            method=TrainingMethod.EVALUATION,
            instance_type=self.instance_type,
            instance_count=self.instance_count,
            infra=self.mock_infra,
            output_s3_path=self.output_s3,
            data_s3_path=self.data_s3,
            eval_task=EvaluationTask.MMLU,
        )

        self.assertEqual(builder.eval_task, EvaluationTask.MMLU)
        self.assertIsNotNone(builder.strategy)
        self.assertIsNotNone(builder.metric)

    def test_initialization_evaluation_method_missing_eval_task(self):
        with self.assertRaises(ValueError) as context:
            RecipeBuilder(
                region=self.region,
                job_name=self.job_name,
                platform=self.platform,
                model=self.mock_model,
                method=TrainingMethod.EVALUATION,
                instance_type=self.instance_type,
                instance_count=self.instance_count,
                infra=self.mock_infra,
                output_s3_path=self.output_s3,
                data_s3_path=self.data_s3,
                eval_task=None,
            )
        self.assertEqual(
            str(context.exception),
            "'eval_task' is a required parameter when calling evaluate().",
        )

    def test_initialization_cpt_method(self):
        validation_data_s3 = "s3://bucket/validation-data"
        builder = RecipeBuilder(
            region=self.region,
            job_name=self.job_name,
            platform=self.platform,
            model=self.mock_model,
            method=TrainingMethod.CPT,
            instance_type=self.instance_type,
            instance_count=self.instance_count,
            infra=self.mock_infra,
            output_s3_path=self.output_s3,
            data_s3_path=self.data_s3,
            validation_data_s3_path=validation_data_s3,
        )

        self.assertEqual(builder.validation_data_s3_path, validation_data_s3)

    @patch("amzn_nova_forge.recipe.recipe_builder.logger")
    def test_initialization_validation_data_s3_path_ignored_for_non_cpt_method(self, mock_logger):
        validation_data_s3 = "s3://bucket/validation-data"
        builder = RecipeBuilder(
            region=self.region,
            job_name=self.job_name,
            platform=self.platform,
            model=self.mock_model,
            method=TrainingMethod.DPO_LORA,
            instance_type=self.instance_type,
            instance_count=self.instance_count,
            infra=self.mock_infra,
            output_s3_path=self.output_s3,
            data_s3_path=self.data_s3,
            validation_data_s3_path=validation_data_s3,
        )

        mock_logger.info.assert_called_with(
            "'validation_data_s3_path' is only applicable for CPT, SFT on SMTJ/SMHP/SMTJServerless, or any method on Bedrock. Will ignore."
        )

        self.assertIsNone(builder.validation_data_s3_path)

    def test_initialization_dpo_lora_method(self):
        builder = RecipeBuilder(
            region=self.region,
            job_name=self.job_name,
            platform=self.platform,
            model=self.mock_model,
            method=TrainingMethod.DPO_LORA,
            instance_type=self.instance_type,
            instance_count=self.instance_count,
            infra=self.mock_infra,
            output_s3_path=self.output_s3,
            data_s3_path=self.data_s3,
        )

        self.assertEqual(builder.method, TrainingMethod.DPO_LORA)
        self.assertEqual(builder.model, self.mock_model)

    def test_initialization_dpo_full_method(self):
        builder = RecipeBuilder(
            region=self.region,
            job_name=self.job_name,
            platform=self.platform,
            model=self.mock_model,
            method=TrainingMethod.DPO_FULL,
            instance_type=self.instance_type,
            instance_count=self.instance_count,
            infra=self.mock_infra,
            output_s3_path=self.output_s3,
            data_s3_path=self.data_s3,
        )

        self.assertEqual(builder.method, TrainingMethod.DPO_FULL)
        self.assertEqual(builder.model, self.mock_model)

    def test_load_input_recipe_valid_yaml(self):
        builder = RecipeBuilder(
            region=self.region,
            job_name=self.job_name,
            platform=self.platform,
            model=self.mock_model,
            method=self.method,
            instance_type=self.instance_type,
            instance_count=self.instance_count,
            infra=self.mock_infra,
            output_s3_path=self.output_s3,
            data_s3_path=self.data_s3,
        )

        test_recipe = {"run": {"name": "test", "model_type": "test-model"}}
        yaml_content = yaml.dump(test_recipe)

        with patch(
            "amzn_nova_forge.recipe.recipe_builder.load_file_as_string",
            return_value=yaml_content,
        ):
            builder._load_input_recipe("test_path.yaml")

        self.assertEqual(builder.input_recipe_dict, test_recipe)

    def test_load_input_recipe_not_yaml(self):
        builder = RecipeBuilder(
            region=self.region,
            job_name=self.job_name,
            platform=self.platform,
            model=self.mock_model,
            method=self.method,
            instance_type=self.instance_type,
            instance_count=self.instance_count,
            infra=self.mock_infra,
            output_s3_path=self.output_s3,
            data_s3_path=self.data_s3,
        )

        with patch(
            "amzn_nova_forge.recipe.recipe_builder.load_file_as_string",
            return_value="invalid: yaml: content:",
        ):
            with self.assertRaises(ValueError) as context:
                builder._load_input_recipe("test_path.yaml")

            self.assertIn("Failed to parse", str(context.exception))

    def test_load_input_recipe_invalid_yaml(self):
        builder = RecipeBuilder(
            region=self.region,
            job_name=self.job_name,
            platform=self.platform,
            model=self.mock_model,
            method=self.method,
            instance_type=self.instance_type,
            instance_count=self.instance_count,
            infra=self.mock_infra,
            output_s3_path=self.output_s3,
            data_s3_path=self.data_s3,
        )

        yaml_content = "- item1\n- item2\n- item3"

        with patch(
            "amzn_nova_forge.recipe.recipe_builder.load_file_as_string",
            return_value=yaml_content,
        ):
            with self.assertRaises(ValueError) as context:
                builder._load_input_recipe("test_path.yaml")

            self.assertIn("Failed to parse provided recipe", str(context.exception))

    def test_generate_recipe_path_with_provided_path(self):
        builder = RecipeBuilder(
            region=self.region,
            job_name=self.job_name,
            platform=self.platform,
            model=self.mock_model,
            method=self.method,
            instance_type=self.instance_type,
            instance_count=self.instance_count,
            infra=self.mock_infra,
            output_s3_path=self.output_s3,
            data_s3_path=self.data_s3,
        )

        custom_path = "/custom/path/recipe.yaml"
        recipe_path = builder._generate_recipe_path(custom_path)

        self.assertEqual(recipe_path.path, custom_path)

    def test_generate_recipe_path_smtj(self):
        builder = RecipeBuilder(
            region=self.region,
            job_name=self.job_name,
            platform=Platform.SMTJ,
            model=self.mock_model,
            method=self.method,
            instance_type=self.instance_type,
            instance_count=self.instance_count,
            infra=self.mock_infra,
            output_s3_path=self.output_s3,
            data_s3_path=self.data_s3,
        )

        recipe_path = builder._generate_recipe_path()
        path = recipe_path.path

        self.assertTrue(path.endswith(".yaml"))
        self.assertIn(self.job_name, path)
        self.assertTrue(recipe_path.temp)

    @patch("builtins.__import__", side_effect=ModuleNotFoundError("hyperpod_cli missing"))
    def test_generate_recipe_path_missing_hyperpod_cli(self, _):
        builder = RecipeBuilder(
            region=self.region,
            job_name=self.job_name,
            platform=Platform.SMHP,
            model=self.mock_model,
            method=self.method,
            instance_type=self.instance_type,
            instance_count=self.instance_count,
            infra=self.mock_infra,
            output_s3_path=self.output_s3,
            data_s3_path=self.data_s3,
        )

        with self.assertRaises(RuntimeError) as context:
            builder._generate_recipe_path()

        self.assertIn("HyperPod CLI is a required dependency", str(context.exception))

    def test_generate_recipe_path_hyperpod(self):
        mock_hyperpod_cli = MagicMock()
        mock_hyperpod_cli.__file__ = "/tmp/hyperpod_cli/__init__.py"

        with patch.dict("sys.modules", {"hyperpod_cli": mock_hyperpod_cli}):
            builder = RecipeBuilder(
                region=self.region,
                job_name=self.job_name,
                platform=Platform.SMHP,
                model=self.mock_model,
                method=self.method,
                instance_type=self.instance_type,
                instance_count=self.instance_count,
                infra=self.mock_infra,
                output_s3_path=self.output_s3,
                data_s3_path=self.data_s3,
            )

            recipe_path = builder._generate_recipe_path()
            path = recipe_path.path
            self.assertTrue(
                path.startswith(
                    "/tmp/hyperpod_cli/sagemaker_hyperpod_recipes/recipes_collection/recipes/fine-tuning/nova/nova_1_0/nova_micro/SFT/"
                )
            )
            self.assertTrue(path.endswith(".yaml"))

    def test_generate_recipe_path_hyperpod_nova_v2(self):
        mock_hyperpod_cli = MagicMock()
        mock_hyperpod_cli.__file__ = "/tmp/hyperpod_cli/__init__.py"

        with patch.dict("sys.modules", {"hyperpod_cli": mock_hyperpod_cli}):
            builder = RecipeBuilder(
                region=self.region,
                job_name=self.job_name,
                platform=Platform.SMHP,
                model=Model.NOVA_LITE_2,
                method=self.method,
                instance_type=self.instance_type,
                instance_count=self.instance_count,
                infra=self.mock_infra,
                output_s3_path=self.output_s3,
                data_s3_path=self.data_s3,
            )

            recipe_path = builder._generate_recipe_path()
            path = recipe_path.path
            self.assertTrue(
                path.startswith(
                    "/tmp/hyperpod_cli/sagemaker_hyperpod_recipes/recipes_collection/recipes/fine-tuning/nova/nova_2_0/nova_lite/SFT/"
                )
            )
            self.assertTrue(path.endswith(".yaml"))

    def test_generate_recipe_path_hyperpod_evaluation(self):
        mock_hyperpod_cli = MagicMock()
        mock_hyperpod_cli.__file__ = "/tmp/hyperpod_cli/__init__.py"

        with patch.dict("sys.modules", {"hyperpod_cli": mock_hyperpod_cli}):
            builder = RecipeBuilder(
                region=self.region,
                job_name=self.job_name,
                platform=Platform.SMHP,
                model=self.mock_model,
                method=TrainingMethod.EVALUATION,
                instance_type=self.instance_type,
                instance_count=self.instance_count,
                infra=self.mock_infra,
                output_s3_path=self.output_s3,
                data_s3_path=self.data_s3,
                eval_task=EvaluationTask.MMLU,
            )

            recipe_path = builder._generate_recipe_path()
            path = recipe_path.path

            self.assertTrue(
                path.startswith(
                    "/tmp/hyperpod_cli/sagemaker_hyperpod_recipes/recipes_collection/recipes/evaluation/nova/nova_1_0/nova_micro/"
                )
            )
            self.assertTrue(path.endswith(".yaml"))

    def test_generate_recipe_path_hyperpod_cpt(self):
        mock_hyperpod_cli = MagicMock()
        mock_hyperpod_cli.__file__ = "/tmp/hyperpod_cli/__init__.py"

        with patch.dict("sys.modules", {"hyperpod_cli": mock_hyperpod_cli}):
            builder = RecipeBuilder(
                region=self.region,
                job_name=self.job_name,
                platform=Platform.SMHP,
                model=self.mock_model,
                method=TrainingMethod.CPT,
                instance_type=self.instance_type,
                instance_count=self.instance_count,
                infra=self.mock_infra,
                output_s3_path=self.output_s3,
                data_s3_path=self.data_s3,
            )

            recipe_path = builder._generate_recipe_path()
            path = recipe_path.path

            self.assertTrue(
                path.startswith(
                    "/tmp/hyperpod_cli/sagemaker_hyperpod_recipes/recipes_collection/recipes/training/nova/nova_1_0/nova_micro/"
                )
            )
            self.assertTrue(path.endswith(".yaml"))

    def test_generate_recipe_path_hyperpod_forge(self):
        mock_hyperpod_cli = MagicMock()
        mock_hyperpod_cli.__file__ = "/tmp/hyperpod_cli/__init__.py"

        with patch.dict("sys.modules", {"hyperpod_cli": mock_hyperpod_cli}):
            builder = RecipeBuilder(
                region=self.region,
                job_name=self.job_name,
                platform=Platform.SMHP,
                model=self.mock_model,
                method=TrainingMethod.CPT,
                instance_type=self.instance_type,
                instance_count=self.instance_count,
                infra=self.mock_infra,
                output_s3_path=self.output_s3,
                data_s3_path=self.data_s3,
                data_mixing_instance=DataMixing(),
            )

            recipe_path = builder._generate_recipe_path()
            path = recipe_path.path

            self.assertTrue(
                path.startswith(
                    "/tmp/hyperpod_cli/sagemaker_hyperpod_recipes/recipes_collection/recipes/training/nova/forge/nova_1_0/nova_micro/"
                )
            )
            self.assertTrue(path.endswith(".yaml"))

    def test_generate_recipe_path_hyperpod_dpo_lora(self):
        mock_hyperpod_cli = MagicMock()
        mock_hyperpod_cli.__file__ = "/tmp/hyperpod_cli/__init__.py"

        with patch.dict("sys.modules", {"hyperpod_cli": mock_hyperpod_cli}):
            builder = RecipeBuilder(
                region=self.region,
                job_name=self.job_name,
                platform=Platform.SMHP,
                model=self.mock_model,
                method=TrainingMethod.DPO_LORA,
                instance_type=self.instance_type,
                instance_count=self.instance_count,
                infra=self.mock_infra,
                output_s3_path=self.output_s3,
                data_s3_path=self.data_s3,
            )

            recipe_path = builder._generate_recipe_path()
            path = recipe_path.path

            self.assertTrue(
                path.startswith(
                    "/tmp/hyperpod_cli/sagemaker_hyperpod_recipes/recipes_collection/recipes/fine-tuning/nova/nova_1_0/nova_micro/"
                )
            )
            self.assertTrue(path.endswith(".yaml"))

    def test_generate_recipe_path_hyperpod_dpo_full(self):
        mock_hyperpod_cli = MagicMock()
        mock_hyperpod_cli.__file__ = "/tmp/hyperpod_cli/__init__.py"

        with patch.dict("sys.modules", {"hyperpod_cli": mock_hyperpod_cli}):
            builder = RecipeBuilder(
                region=self.region,
                job_name=self.job_name,
                platform=Platform.SMHP,
                model=self.mock_model,
                method=TrainingMethod.DPO_FULL,
                instance_type=self.instance_type,
                instance_count=self.instance_count,
                infra=self.mock_infra,
                output_s3_path=self.output_s3,
                data_s3_path=self.data_s3,
            )

            recipe_path = builder._generate_recipe_path()
            path = recipe_path.path

            self.assertTrue(
                path.startswith(
                    "/tmp/hyperpod_cli/sagemaker_hyperpod_recipes/recipes_collection/recipes/fine-tuning/nova/nova_1_0/nova_micro/"
                )
            )
            self.assertTrue(path.endswith(".yaml"))

    @patch("amzn_nova_forge.util.recipe.get_hub_recipe_metadata")
    @patch("amzn_nova_forge.util.recipe.download_templates_from_s3")
    @patch("amzn_nova_forge.recipe.recipe_builder.Validator")
    def test_build_and_validate_success(self, mock_validator, mock_download, mock_metadata):
        # Setup mocks
        mock_metadata.return_value = {"recipe_uri": "s3://bucket/recipe"}

        recipe_template = {
            "run": {
                "name": "{{name}}",
                "model_type": self.mock_model.model_type,
                "model_name_or_path": self.mock_model.model_path,
                "data_s3_path": "{{data_s3_path}}",
                "output_s3_path": "{{output_s3_path}}",
                "replicas": "{{replicas}}",
            }
        }

        overrides_template = {
            "name": {"default": "", "type": "string"},
            "data_s3_path": {"default": "", "type": "string"},
            "output_s3_path": {"default": "", "type": "string"},
            "replicas": {"default": 1, "type": "int"},
        }

        mock_download.return_value = (recipe_template, overrides_template, "image_uri")

        builder = RecipeBuilder(
            region=self.region,
            job_name=self.job_name,
            platform=self.platform,
            model=self.mock_model,
            method=self.method,
            instance_type=self.instance_type,
            instance_count=self.instance_count,
            infra=self.mock_infra,
            output_s3_path=self.output_s3,
            data_s3_path=self.data_s3,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, "recipe.yaml")

            (recipe_path, *_) = builder.build_and_validate(
                output_recipe_path=output_path,
            )

            # Verify file was created
            self.assertTrue(os.path.exists(recipe_path))

            # Verify content
            with open(recipe_path, "r") as f:
                config = yaml.safe_load(f)

            self.assertEqual(config["run"]["name"], self.job_name)
            self.assertEqual(config["run"]["data_s3_path"], self.data_s3)
            self.assertEqual(config["run"]["output_s3_path"], self.output_s3)

            # Verify validator was called
            mock_validator.validate.assert_called_once()

    @patch("amzn_nova_forge.util.recipe.get_hub_recipe_metadata")
    @patch("amzn_nova_forge.util.recipe.download_templates_from_s3")
    @patch("amzn_nova_forge.recipe.recipe_builder.Validator")
    def test_build_and_validate_with_overrides(self, mock_validator, mock_download, mock_metadata):
        mock_metadata.return_value = {"recipe_uri": "s3://bucket/recipe"}

        recipe_template = {
            "run": {"name": "{{name}}"},
            "training_config": {
                "learning_rate": "{{learning_rate}}",
            },
        }

        overrides_template = {
            "name": {"default": "", "type": "string"},
            "learning_rate": {"default": 0.001, "type": "float"},
        }

        mock_download.return_value = (recipe_template, overrides_template, "image_uri")

        builder = RecipeBuilder(
            region=self.region,
            job_name=self.job_name,
            platform=self.platform,
            model=self.mock_model,
            method=self.method,
            instance_type=self.instance_type,
            instance_count=self.instance_count,
            infra=self.mock_infra,
            output_s3_path=self.output_s3,
            data_s3_path=self.data_s3,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, "recipe.yaml")

            overrides = {"learning_rate": 0.01}
            recipe_path, *_ = builder.build_and_validate(
                overrides=overrides,
                output_recipe_path=output_path,
            )

            with open(recipe_path, "r") as f:
                config = yaml.safe_load(f)

            self.assertEqual(config["training_config"]["learning_rate"], 0.01)

    @patch("amzn_nova_forge.util.recipe.get_hub_recipe_metadata")
    @patch("amzn_nova_forge.util.recipe.download_templates_from_s3")
    @patch("amzn_nova_forge.recipe.recipe_builder.Validator")
    def test_build_and_validate_with_scientific_notation_override(
        self, mock_validator, mock_download, mock_metadata
    ):
        mock_metadata.return_value = {"recipe_uri": "s3://bucket/recipe"}

        recipe_template = {
            "eps": 1e-5,
        }

        overrides_template = {}

        mock_download.return_value = (recipe_template, overrides_template, "image_uri")

        builder = RecipeBuilder(
            region=self.region,
            job_name=self.job_name,
            platform=self.platform,
            model=self.mock_model,
            method=self.method,
            instance_type=self.instance_type,
            instance_count=self.instance_count,
            infra=self.mock_infra,
            output_s3_path=self.output_s3,
            data_s3_path=self.data_s3,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, "recipe.yaml")

            overrides = {"eps": 5e-6}
            recipe_path, *_ = builder.build_and_validate(
                overrides=overrides,
                output_recipe_path=output_path,
            )

            with open(recipe_path, "r") as f:
                config = yaml.safe_load(f)

            self.assertEqual(config["eps"], 5e-6)
            self.assertTrue(type(config["eps"]), float)

    @patch("amzn_nova_forge.util.recipe.get_hub_recipe_metadata")
    @patch("amzn_nova_forge.util.recipe.download_templates_from_s3")
    @patch("amzn_nova_forge.recipe.recipe_builder.Validator")
    def test_build_and_validate_with_override_replicas_not_in_overrides_template(
        self, mock_validator, mock_download, mock_metadata
    ):
        mock_metadata.return_value = {"recipe_uri": "s3://bucket/recipe"}

        recipe_template = {
            "run": {"name": "{{name}}", "replicas": 6},
        }

        overrides_template = {
            "name": {"default": "", "type": "string"},
        }

        mock_download.return_value = (recipe_template, overrides_template, "image_uri")

        builder = RecipeBuilder(
            region=self.region,
            job_name=self.job_name,
            platform=self.platform,
            model=self.mock_model,
            method=self.method,
            instance_type=self.instance_type,
            instance_count=self.instance_count,
            infra=self.mock_infra,
            output_s3_path=self.output_s3,
            data_s3_path=self.data_s3,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, "recipe.yaml")

            overrides = {"replicas": 10}
            recipe_path, *_ = builder.build_and_validate(
                overrides=overrides,
                output_recipe_path=output_path,
            )

            with open(recipe_path, "r") as f:
                config = yaml.safe_load(f)

            self.assertEqual(config["run"]["replicas"], 10)

    @patch("amzn_nova_forge.util.recipe.get_hub_recipe_metadata")
    @patch("amzn_nova_forge.util.recipe.download_templates_from_s3")
    @patch("amzn_nova_forge.recipe.recipe_builder.Validator")
    def test_build_and_validate_with_override_name_different_from_recipe_name(
        self, mock_validator, mock_download, mock_metadata
    ):
        mock_metadata.return_value = {"recipe_uri": "s3://bucket/recipe"}

        recipe_template = {
            "run": {"name": "{{name}}"},
            "training_config": {
                "alpha": "{{lora_alpha}}",
            },
        }

        overrides_template = {
            "name": {"default": "", "type": "string"},
            "lora_alpha": {"default": 128, "type": "int"},
        }

        mock_download.return_value = (recipe_template, overrides_template, "image_uri")

        builder = RecipeBuilder(
            region=self.region,
            job_name=self.job_name,
            platform=self.platform,
            model=self.mock_model,
            method=self.method,
            instance_type=self.instance_type,
            instance_count=self.instance_count,
            infra=self.mock_infra,
            output_s3_path=self.output_s3,
            data_s3_path=self.data_s3,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, "recipe.yaml")

            overrides = {"alpha": 5}
            recipe_path, *_ = builder.build_and_validate(
                overrides=overrides,
                output_recipe_path=output_path,
            )

            with open(recipe_path, "r") as f:
                config = yaml.safe_load(f)

            self.assertEqual(config["training_config"]["alpha"], 5)

    @patch("amzn_nova_forge.util.recipe.get_hub_recipe_metadata")
    @patch("amzn_nova_forge.util.recipe.download_templates_from_s3")
    @patch("amzn_nova_forge.recipe.recipe_builder.Validator")
    def test_build_and_validate_with_mlflow(self, mock_validator, mock_download, mock_metadata):
        mock_metadata.return_value = {"recipe_uri": "s3://bucket/recipe"}

        recipe_template = {
            "run": {
                "name": "{{name}}",
                "replicas": 1,
                "mlflow_tracking_uri": "{{mlflow_tracking_uri}}",
                "mlflow_experiment_name": "{{mlflow_experiment_name}}",
                "mlflow_run_name": "{{mlflow_run_name}}",
            }
        }

        overrides_template = {
            "name": {"default": "", "type": "string"},
        }

        mock_download.return_value = (recipe_template, overrides_template, "image_uri")

        mock_mlflow = Mock(spec=MLflowMonitor)
        mock_mlflow.tracking_uri = "test-uri"
        mock_mlflow.experiment_name = "test-exp"
        mock_mlflow.run_name = "test-run"

        builder = RecipeBuilder(
            region=self.region,
            job_name=self.job_name,
            platform=self.platform,
            model=self.mock_model,
            method=self.method,
            instance_type=self.instance_type,
            instance_count=self.instance_count,
            infra=self.mock_infra,
            output_s3_path=self.output_s3,
            data_s3_path=self.data_s3,
            mlflow_monitor=mock_mlflow,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, "recipe.yaml")
            recipe_path, *_ = builder.build_and_validate(
                output_recipe_path=output_path,
            )

            with open(recipe_path, "r") as f:
                config = yaml.safe_load(f)

            self.assertEqual(config["run"]["mlflow_tracking_uri"], "test-uri")
            self.assertEqual(config["run"]["mlflow_experiment_name"], "test-exp")
            self.assertEqual(config["run"]["mlflow_run_name"], "test-run")

    @patch("amzn_nova_forge.util.recipe.get_hub_recipe_metadata")
    @patch("amzn_nova_forge.util.recipe.download_templates_from_s3")
    @patch("amzn_nova_forge.recipe.recipe_builder.Validator")
    def test_build_and_validate_rft(self, mock_validator, mock_download, mock_metadata):
        mock_metadata.return_value = {"recipe_uri": "s3://bucket/recipe"}

        recipe_template = {
            "run": {
                "name": "{{name}}",
                "replicas": 1,
                "lambda_arn": "{{reward_lambda_arn}}",
            }
        }

        overrides_template = {
            "name": {"default": "", "type": "string"},
            "reward_lambda_arn": {"default": "", "type": "string"},
        }

        mock_download.return_value = (recipe_template, overrides_template, "image_uri")

        rft_lambda = "arn:aws:lambda:us-east-1:123456789012:function:reward"

        builder = RecipeBuilder(
            region=self.region,
            job_name=self.job_name,
            platform=self.platform,
            model=self.mock_model,
            method=TrainingMethod.RFT_LORA,
            instance_type=self.instance_type,
            instance_count=self.instance_count,
            infra=self.mock_infra,
            output_s3_path=self.output_s3,
            data_s3_path=self.data_s3,
            rft_lambda_arn=rft_lambda,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, "recipe.yaml")
            recipe_path, *_ = builder.build_and_validate(
                output_recipe_path=output_path,
            )

            with open(recipe_path, "r") as f:
                config = yaml.safe_load(f)

            self.assertEqual(config["run"]["lambda_arn"], rft_lambda)

    @patch("amzn_nova_forge.util.recipe.get_hub_recipe_metadata")
    @patch("amzn_nova_forge.util.recipe.download_templates_from_s3")
    @patch("amzn_nova_forge.recipe.recipe_builder.Validator")
    def test_build_and_validate_evaluation_basic(
        self, mock_validator, mock_download, mock_metadata
    ):
        mock_metadata.return_value = {"recipe_uri": "s3://bucket/recipe"}

        recipe_template = {
            "run": {"name": "{{name}}", "replicas": 1},
            "evaluation": {
                "task": "{{task}}",
                "strategy": "{{strategy}}",
                "metric": "{{metric}}",
            },
        }

        overrides_template = {
            "name": {"default": "", "type": "string"},
        }

        mock_download.return_value = (recipe_template, overrides_template, "image_uri")

        builder = RecipeBuilder(
            region=self.region,
            job_name=self.job_name,
            platform=self.platform,
            model=self.mock_model,
            method=TrainingMethod.EVALUATION,
            instance_type=self.instance_type,
            instance_count=self.instance_count,
            infra=self.mock_infra,
            output_s3_path=self.output_s3,
            data_s3_path=self.data_s3,
            eval_task=EvaluationTask.MMLU,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, "recipe.yaml")
            recipe_path, *_ = builder.build_and_validate(
                output_recipe_path=output_path,
            )

            with open(recipe_path, "r") as f:
                yaml_content = f.read()

            self.assertIn("task:", yaml_content)
            self.assertIn("strategy:", yaml_content)
            self.assertIn("metric:", yaml_content)

    @patch("amzn_nova_forge.util.recipe.get_hub_recipe_metadata")
    @patch("amzn_nova_forge.util.recipe.download_templates_from_s3")
    @patch("amzn_nova_forge.recipe.recipe_builder.Validator")
    def test_build_and_validate_evaluation_with_subtask(
        self, mock_validator, mock_download, mock_metadata
    ):
        mock_metadata.return_value = {"recipe_uri": "s3://bucket/recipe"}

        recipe_template = {
            "run": {"name": "{{name}}", "replicas": 1},
            "evaluation": {
                "task": "{{task}}",
                "subtask": "{{subtask}}",
                "strategy": "{{strategy}}",
                "metric": "{{metric}}",
            },
        }

        overrides_template = {
            "name": {"default": "", "type": "string"},
        }

        mock_download.return_value = (recipe_template, overrides_template, "image_uri")

        builder = RecipeBuilder(
            region=self.region,
            job_name=self.job_name,
            platform=self.platform,
            model=self.mock_model,
            method=TrainingMethod.EVALUATION,
            instance_type=self.instance_type,
            instance_count=self.instance_count,
            infra=self.mock_infra,
            output_s3_path=self.output_s3,
            data_s3_path=self.data_s3,
            eval_task=EvaluationTask.MMLU,
            subtask="college_biology",
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, "recipe.yaml")
            recipe_path, *_ = builder.build_and_validate(
                output_recipe_path=output_path,
            )

            with open(recipe_path, "r") as f:
                yaml_content = f.read()

            self.assertIn("subtask: college_biology", yaml_content)

    @patch("amzn_nova_forge.util.recipe.get_hub_recipe_metadata")
    @patch("amzn_nova_forge.util.recipe.download_templates_from_s3")
    @patch("amzn_nova_forge.recipe.recipe_builder.Validator")
    def test_build_and_validate_evaluation_with_processor_config(
        self, mock_validator, mock_download, mock_metadata
    ):
        mock_metadata.return_value = {"recipe_uri": "s3://bucket/recipe"}

        recipe_template = {
            "run": {"name": "{{name}}", "replicas": 1},
            "evaluation": {"task": "{{task}}"},
            "processor": {
                "lambda_arn": "{{lambda_arn}}",
                "lambda_type": "{{lambda_type}}",
                "preprocessing": {"enabled": "{{preprocessing}}"},
                "postprocessing": {"enabled": "{{postprocessing}}"},
                "aggregation": "{{aggregation}}",
            },
        }

        overrides_template = {
            "name": {"default": "", "type": "string"},
        }

        mock_download.return_value = (recipe_template, overrides_template, "image_uri")

        processor_config = {
            "lambda_arn": "arn:aws:lambda:us-east-1:123456789012:function:processor",
            "lambda_type": "custom",
            "preprocessing": {"enabled": True},
            "postprocessing": {"enabled": False},
            "aggregation": "average",
        }

        builder = RecipeBuilder(
            region=self.region,
            job_name=self.job_name,
            platform=self.platform,
            model=self.mock_model,
            method=TrainingMethod.EVALUATION,
            instance_type=self.instance_type,
            instance_count=self.instance_count,
            infra=self.mock_infra,
            output_s3_path=self.output_s3,
            data_s3_path=self.data_s3,
            eval_task=EvaluationTask.MMLU,
            processor_config=processor_config,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, "recipe.yaml")
            recipe_path, *_ = builder.build_and_validate(
                output_recipe_path=output_path,
            )

            with open(recipe_path, "r") as f:
                yaml_content = f.read()

            self.assertIn(
                "lambda_arn: arn:aws:lambda:us-east-1:123456789012:function:processor",
                yaml_content,
            )
            self.assertIn("lambda_type: custom", yaml_content)
            self.assertIn("aggregation: average", yaml_content)
            self.assertIn("preprocessing:\n    enabled: true", yaml_content)
            self.assertIn("postprocessing:\n    enabled: false", yaml_content)

    @patch("amzn_nova_forge.util.recipe.get_hub_recipe_metadata")
    @patch("amzn_nova_forge.util.recipe.download_templates_from_s3")
    @patch("amzn_nova_forge.recipe.recipe_builder.Validator")
    def test_build_and_validate_evaluation_with_empty_processor_config(
        self, mock_validator, mock_download, mock_metadata
    ):
        mock_metadata.return_value = {"recipe_uri": "s3://bucket/recipe"}

        recipe_template = {
            "run": {"name": "{{name}}"},
            "replicas": 1,
            "evaluation": {"task": "{{task}}"},
            "processor": {
                "lambda_arn": "{{lambda_arn}}",
                "lambda_type": "{{lambda_type}}",
                "preprocessing": {"enabled": "{{preprocessing}}"},
                "postprocessing": {"enabled": "{{postprocessing}}"},
                "aggregation": "{{aggregation}}",
            },
        }

        overrides_template = {
            "name": {"default": "", "type": "string"},
            "lambda_arn": {"default": "arn", "type": "string"},
            "lambda_type": {"default": "type", "type": "string"},
            "preprocessing": {"default": True, "type": "boolean"},
            "postprocessing": {"default": False, "type": "boolean"},
            "aggregation": {"default": "average", "type": "string"},
        }

        mock_download.return_value = (recipe_template, overrides_template, "image_uri")

        builder = RecipeBuilder(
            region=self.region,
            job_name=self.job_name,
            platform=self.platform,
            model=self.mock_model,
            method=TrainingMethod.EVALUATION,
            instance_type=self.instance_type,
            instance_count=self.instance_count,
            infra=self.mock_infra,
            output_s3_path=self.output_s3,
            data_s3_path=self.data_s3,
            eval_task=EvaluationTask.MMLU,
            processor_config={},
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, "recipe.yaml")
            recipe_path, *_ = builder.build_and_validate(
                output_recipe_path=output_path,
            )

            with open(recipe_path, "r") as f:
                yaml_content = f.read()

            self.assertIn(
                "lambda_arn: arn",
                yaml_content,
            )
            self.assertIn("lambda_type: type", yaml_content)
            self.assertIn("aggregation: average", yaml_content)
            self.assertIn("preprocessing:\n    enabled: true", yaml_content)
            self.assertIn("postprocessing:\n    enabled: false", yaml_content)

    @patch("amzn_nova_forge.util.recipe.get_hub_recipe_metadata")
    @patch("amzn_nova_forge.util.recipe.download_templates_from_local")
    @patch("amzn_nova_forge.recipe.recipe_builder.Validator")
    def test_build_and_validate_evaluation_with_rl_env_config(
        self, mock_validator, mock_download, mock_metadata
    ):
        mock_metadata.return_value = {
            "RecipeTemplatePath": "/path/to/recipe",
            "OverrideParamsPath": "/path/to/override.json",
            "ImageUri": "image_uri",
        }

        recipe_template = {
            "run": {"name": "{{name}}", "replicas": 1},
            "rl_env": {
                "reward_lambda_arn": "{{reward_lambda_arn}}",
            },
        }

        overrides_template = {
            "name": {"default": "", "type": "string"},
        }

        mock_download.return_value = (recipe_template, overrides_template, "image_uri")

        rl_env_config = {
            "reward_lambda_arn": "arn:aws:lambda:us-east-1:123456789012:function:rl-reward"
        }

        builder = RecipeBuilder(
            region=self.region,
            job_name=self.job_name,
            platform=self.platform,
            model=self.mock_model,
            method=TrainingMethod.EVALUATION,
            instance_type=self.instance_type,
            instance_count=self.instance_count,
            infra=self.mock_infra,
            output_s3_path=self.output_s3,
            data_s3_path=self.data_s3,
            eval_task=EvaluationTask.RFT_EVAL,
            rl_env_config=rl_env_config,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, "recipe.yaml")
            recipe_path, *_ = builder.build_and_validate(
                output_recipe_path=output_path,
            )

            with open(recipe_path, "r") as f:
                config = yaml.safe_load(f)

            self.assertEqual(
                config["rl_env"]["reward_lambda_arn"],
                "arn:aws:lambda:us-east-1:123456789012:function:rl-reward",
            )

    @patch("amzn_nova_forge.util.recipe.get_hub_recipe_metadata")
    @patch("amzn_nova_forge.util.recipe.download_templates_from_s3")
    @patch("amzn_nova_forge.recipe.recipe_builder.Validator")
    def test_build_and_validate_cpt(self, mock_validator, mock_download, mock_metadata):
        mock_metadata.return_value = {"recipe_uri": "s3://bucket/recipe"}

        recipe_template = {
            "run": {
                "name": "{{name}}",
                "replicas": 1,
                "validation_data_s3_path": "{{validation_s3_path}}",
            }
        }

        overrides_template = {
            "name": {"default": "", "type": "string"},
            "validation_s3_path": {"default": "", "type": "string"},
        }

        mock_download.return_value = (recipe_template, overrides_template, "image_uri")

        validation_data_s3 = "s3://bucket/validation-data"

        builder = RecipeBuilder(
            region=self.region,
            job_name=self.job_name,
            platform=self.platform,
            model=self.mock_model,
            method=TrainingMethod.CPT,
            instance_type=self.instance_type,
            instance_count=self.instance_count,
            infra=self.mock_infra,
            output_s3_path=self.output_s3,
            data_s3_path=self.data_s3,
            validation_data_s3_path=validation_data_s3,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, "recipe.yaml")
            recipe_path, *_ = builder.build_and_validate(output_recipe_path=output_path)

            with open(recipe_path, "r") as f:
                config = yaml.safe_load(f)

            self.assertEqual(config["run"]["validation_data_s3_path"], validation_data_s3)

    @patch("amzn_nova_forge.util.recipe.get_hub_recipe_metadata")
    @patch("amzn_nova_forge.util.recipe.download_templates_from_s3")
    @patch("amzn_nova_forge.recipe.recipe_builder.Validator")
    def test_build_and_validate_cpt_without_validation_data(
        self, mock_validator, mock_download, mock_metadata
    ):
        mock_metadata.return_value = {"recipe_uri": "s3://bucket/recipe"}

        recipe_template = {
            "run": {
                "name": "{{name}}",
                "replicas": 1,
                "validation_data_s3_path": "{{validation_s3_path}}",
            }
        }

        overrides_template = {
            "name": {"default": "", "type": "string"},
            "validation_s3_path": {"default": "", "type": "string"},
        }

        mock_download.return_value = (recipe_template, overrides_template, "image_uri")

        builder = RecipeBuilder(
            region=self.region,
            job_name=self.job_name,
            platform=self.platform,
            model=self.mock_model,
            method=TrainingMethod.CPT,
            instance_type=self.instance_type,
            instance_count=self.instance_count,
            infra=self.mock_infra,
            output_s3_path=self.output_s3,
            data_s3_path=self.data_s3,
            validation_data_s3_path=None,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, "recipe.yaml")
            recipe_path, *_ = builder.build_and_validate(output_recipe_path=output_path)

            with open(recipe_path, "r") as f:
                config = yaml.safe_load(f)

            self.assertEqual("", config["run"]["validation_data_s3_path"])

    @patch("amzn_nova_forge.util.recipe.get_hub_recipe_metadata")
    @patch("amzn_nova_forge.util.recipe.download_templates_from_s3")
    @patch("amzn_nova_forge.recipe.recipe_builder.Validator")
    def test_build_and_validate_dpo_lora(self, mock_validator, mock_download, mock_metadata):
        mock_metadata.return_value = {"recipe_uri": "s3://bucket/recipe"}

        recipe_template = {
            "run": {
                "name": "{{name}}",
                "replicas": 1,
            },
            "training_config": {"model": {"dpo_cfg": {"beta": "{{beta}}"}}},
        }

        overrides_template = {
            "name": {"default": "", "type": "string"},
            "beta": {"default": 0.1, "type": "float"},
        }

        mock_download.return_value = (recipe_template, overrides_template, "image_uri")

        builder = RecipeBuilder(
            region=self.region,
            job_name=self.job_name,
            platform=self.platform,
            model=self.mock_model,
            method=TrainingMethod.DPO_LORA,
            instance_type=self.instance_type,
            instance_count=self.instance_count,
            infra=self.mock_infra,
            output_s3_path=self.output_s3,
            data_s3_path=self.data_s3,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, "recipe.yaml")
            recipe_path, *_ = builder.build_and_validate(output_recipe_path=output_path)

            with open(recipe_path, "r") as f:
                config = yaml.safe_load(f)

            self.assertEqual(config["training_config"]["model"]["dpo_cfg"]["beta"], 0.1)

    @patch("amzn_nova_forge.util.recipe.get_hub_recipe_metadata")
    @patch("amzn_nova_forge.util.recipe.download_templates_from_s3")
    @patch("amzn_nova_forge.recipe.recipe_builder.Validator")
    def test_build_and_validate_dpo_full(self, mock_validator, mock_download, mock_metadata):
        mock_metadata.return_value = {"recipe_uri": "s3://bucket/recipe"}

        recipe_template = {
            "run": {
                "name": "{{name}}",
                "replicas": 1,
            },
            "training_config": {"model": {"dpo_cfg": {"beta": "{{beta}}"}}},
        }

        overrides_template = {
            "name": {"default": "", "type": "string"},
            "beta": {"default": 0.2, "type": "float"},
        }

        mock_download.return_value = (recipe_template, overrides_template, "image_uri")

        builder = RecipeBuilder(
            region=self.region,
            job_name=self.job_name,
            platform=self.platform,
            model=self.mock_model,
            method=TrainingMethod.DPO_FULL,
            instance_type=self.instance_type,
            instance_count=self.instance_count,
            infra=self.mock_infra,
            output_s3_path=self.output_s3,
            data_s3_path=self.data_s3,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, "recipe.yaml")
            recipe_path, *_ = builder.build_and_validate(output_recipe_path=output_path)

            with open(recipe_path, "r") as f:
                config = yaml.safe_load(f)

            self.assertEqual(config["training_config"]["model"]["dpo_cfg"]["beta"], 0.2)

    @patch("amzn_nova_forge.util.recipe.get_hub_recipe_metadata")
    @patch("amzn_nova_forge.util.recipe.download_templates_from_s3")
    @patch("amzn_nova_forge.recipe.recipe_builder.Validator")
    def test_build_and_validate_ignores_distributed_fused_adam(
        self, mock_validator, mock_download, mock_metadata
    ):
        mock_metadata.return_value = {"recipe_uri": "s3://bucket/recipe"}

        recipe_template = {
            "run": {"name": "{{name}}", "replicas": 1},
            "training_config": {
                "model": {
                    "optim": {
                        "name": "distributed_fused_adam",
                        "lr": "{{learning_rate}}",
                    }
                }
            },
        }

        overrides_template = {
            "name": {"default": "", "type": "string"},
            "learning_rate": {"default": 0.001, "type": "float"},
        }

        mock_download.return_value = (recipe_template, overrides_template, "image_uri")

        job_name = "my-training-job"
        builder = RecipeBuilder(
            region=self.region,
            job_name=job_name,
            platform=self.platform,
            model=self.mock_model,
            method=self.method,
            instance_type=self.instance_type,
            instance_count=self.instance_count,
            infra=self.mock_infra,
            output_s3_path=self.output_s3,
            data_s3_path=self.data_s3,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, "recipe.yaml")

            # Test 1: Without explicit override - should use job_name
            recipe_path, *_ = builder.build_and_validate(
                output_recipe_path=output_path,
            )

            with open(recipe_path, "r") as f:
                config = yaml.safe_load(f)

            self.assertEqual(config["run"]["name"], job_name)
            self.assertEqual(
                config["training_config"]["model"]["optim"]["name"],
                "distributed_fused_adam",
            )

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, "recipe.yaml")

            # Test 2: With explicit override for "name" - should use override value
            overrides = {"name": "overridden_name"}
            recipe_path, *_ = builder.build_and_validate(
                overrides=overrides,
                output_recipe_path=output_path,
            )

            with open(recipe_path, "r") as f:
                config = yaml.safe_load(f)

            self.assertEqual(config["run"]["name"], "overridden_name")
            self.assertEqual(
                config["training_config"]["model"]["optim"]["name"],
                "distributed_fused_adam",
            )

    @patch("amzn_nova_forge.util.recipe.get_hub_recipe_metadata")
    @patch("amzn_nova_forge.util.recipe.download_templates_from_s3")
    @patch("amzn_nova_forge.recipe.recipe_builder.Validator")
    @patch("amzn_nova_forge.recipe.recipe_builder.load_file_as_string")
    def test_build_and_validate_with_input_recipe_success(
        self, mock_load_file, mock_validator, mock_download, mock_metadata
    ):
        mock_metadata.return_value = {"recipe_uri": "s3://bucket/recipe"}

        recipe_template = {
            "run": {
                "name": "{{name}}",
                "replicas": 1,
                "model_type": self.mock_model.model_type,
                "data_s3_path": "{{data_s3_path}}",
                "output_s3_path": "{{output_s3_path}}",
            },
            "training_config": {
                "epochs": "{{epochs}}",
                "name": "distributed_fused_adam",
            },
        }

        overrides_template = {
            "name": {"default": "", "type": "string"},
            "data_s3_path": {"default": "", "type": "string"},
            "output_s3_path": {"default": "", "type": "string"},
            "epochs": {"default": 10, "type": "int"},
        }

        mock_download.return_value = (recipe_template, overrides_template, "image_uri")

        input_recipe = {
            "run": {
                "name": "custom-job-name",
                "model_type": self.mock_model.model_type,
                "data_s3_path": "s3://bucket/custom-data",
                "output_s3_path": "s3://bucket/custom-output",
            },
            "training_config": {
                "learning_rate": 0.005,
                "batch_size": 64,
                "epochs": 20,
                "name": "distributed_fused_adam",
            },
        }
        mock_load_file.return_value = yaml.dump(input_recipe)

        builder = RecipeBuilder(
            region=self.region,
            job_name=self.job_name,
            platform=self.platform,
            model=self.mock_model,
            method=self.method,
            instance_type=self.instance_type,
            instance_count=self.instance_count,
            infra=self.mock_infra,
            output_s3_path=self.output_s3,
            data_s3_path=self.data_s3,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, "recipe.yaml")
            input_recipe_path = os.path.join(tmpdir, "input_recipe.yaml")

            recipe_path, *_ = builder.build_and_validate(
                input_recipe_path=input_recipe_path,
                output_recipe_path=output_path,
            )

            self.assertTrue(os.path.exists(recipe_path))

            with open(recipe_path, "r") as f:
                config = yaml.safe_load(f)

            self.assertEqual(config["run"]["name"], "custom-job-name")
            self.assertEqual(config["run"]["data_s3_path"], "s3://bucket/custom-data")
            self.assertEqual(config["run"]["output_s3_path"], "s3://bucket/custom-output")
            self.assertEqual(config["training_config"]["epochs"], 20)
            self.assertEqual(config["training_config"]["name"], "distributed_fused_adam")

            mock_validator.validate.assert_called_once()

            mock_load_file.assert_called_once_with(input_recipe_path, ".yaml")

    @patch("amzn_nova_forge.recipe.recipe_builder.load_file_as_string")
    def test_load_input_recipe_converts_scientific_notation(self, mock_load_file):
        builder = RecipeBuilder(
            region=self.region,
            job_name=self.job_name,
            platform=self.platform,
            model=self.mock_model,
            method=self.method,
            instance_type=self.instance_type,
            instance_count=self.instance_count,
            infra=self.mock_infra,
            output_s3_path=self.output_s3,
            data_s3_path=self.data_s3,
        )

        input_recipe = {
            "run": {
                "name": "e-run",
                "replicas": 1,
            },
            "training_config": {
                "model": {
                    "optim": {
                        "lr": "1e-5",
                        "name": "adam",
                        "eps": "1e-06",
                        "sched": {
                            "min_lr": "1e-6",
                            "warmup_steps": 10,
                        },
                    }
                }
            },
        }
        mock_load_file.return_value = yaml.dump(input_recipe)

        builder._load_input_recipe("test_path.yaml")

        self.assertEqual(builder.input_recipe_dict["training_config"]["model"]["optim"]["lr"], 1e-5)
        self.assertEqual(
            builder.input_recipe_dict["training_config"]["model"]["optim"]["eps"], 1e-06
        )
        self.assertEqual(
            builder.input_recipe_dict["training_config"]["model"]["optim"]["sched"]["min_lr"],
            1e-6,
        )

        self.assertIsInstance(
            builder.input_recipe_dict["training_config"]["model"]["optim"]["lr"], float
        )
        self.assertIsInstance(
            builder.input_recipe_dict["training_config"]["model"]["optim"]["eps"], float
        )
        self.assertIsInstance(
            builder.input_recipe_dict["training_config"]["model"]["optim"]["sched"]["min_lr"],
            float,
        )

        self.assertEqual(builder.input_recipe_dict["run"]["name"], "e-run")
        self.assertEqual(builder.input_recipe_dict["run"]["replicas"], 1)
        self.assertEqual(
            builder.input_recipe_dict["training_config"]["model"]["optim"]["name"],
            "adam",
        )
        self.assertEqual(
            builder.input_recipe_dict["training_config"]["model"]["optim"]["sched"]["warmup_steps"],
            10,
        )

    @patch("amzn_nova_forge.recipe.recipe_builder.load_file_as_string")
    def test_load_input_recipe_handles_lists_with_scientific_notation(self, mock_load_file):
        builder = RecipeBuilder(
            region=self.region,
            job_name=self.job_name,
            platform=self.platform,
            model=self.mock_model,
            method=self.method,
            instance_type=self.instance_type,
            instance_count=self.instance_count,
            infra=self.mock_infra,
            output_s3_path=self.output_s3,
            data_s3_path=self.data_s3,
        )

        input_recipe = {
            "training_config": {
                "learning_rates": ["1e-5", "1e-4", "1e-3"],
                "other_values": [0.9, 0.999],
                "names": ["config1", "config2"],
            }
        }
        mock_load_file.return_value = yaml.dump(input_recipe)

        builder._load_input_recipe("test_path.yaml")

        self.assertEqual(
            builder.input_recipe_dict["training_config"]["learning_rates"],
            [1e-5, 1e-4, 1e-3],
        )
        self.assertIsInstance(
            builder.input_recipe_dict["training_config"]["learning_rates"][0], float
        )
        self.assertIsInstance(
            builder.input_recipe_dict["training_config"]["learning_rates"][1], float
        )
        self.assertIsInstance(
            builder.input_recipe_dict["training_config"]["learning_rates"][2], float
        )

        self.assertEqual(builder.input_recipe_dict["training_config"]["other_values"], [0.9, 0.999])
        self.assertEqual(
            builder.input_recipe_dict["training_config"]["names"],
            ["config1", "config2"],
        )

    @patch("amzn_nova_forge.recipe.recipe_builder.load_file_as_string")
    def test_load_input_recipe_preserves_all_data_types(self, mock_load_file):
        builder = RecipeBuilder(
            region=self.region,
            job_name=self.job_name,
            platform=self.platform,
            model=self.mock_model,
            method=self.method,
            instance_type=self.instance_type,
            instance_count=self.instance_count,
            infra=self.mock_infra,
            output_s3_path=self.output_s3,
            data_s3_path=self.data_s3,
        )

        input_recipe = {
            "string_value": "test",
            "int_value": 42,
            "float_value": 3.14,
            "bool_true": True,
            "bool_false": False,
            "none_value": None,
            "scientific_string": "1e-5",
            "list_value": [1, 2, 3],
            "nested_dict": {"key": "value"},
        }
        mock_load_file.return_value = yaml.dump(input_recipe)

        builder._load_input_recipe("test_path.yaml")

        self.assertEqual(builder.input_recipe_dict["string_value"], "test")
        self.assertEqual(builder.input_recipe_dict["int_value"], 42)
        self.assertEqual(builder.input_recipe_dict["float_value"], 3.14)
        self.assertEqual(builder.input_recipe_dict["bool_true"], True)
        self.assertEqual(builder.input_recipe_dict["bool_false"], False)
        self.assertIsNone(builder.input_recipe_dict["none_value"])
        self.assertEqual(builder.input_recipe_dict["list_value"], [1, 2, 3])
        self.assertEqual(builder.input_recipe_dict["nested_dict"], {"key": "value"})

        self.assertIsInstance(builder.input_recipe_dict["string_value"], str)
        self.assertIsInstance(builder.input_recipe_dict["int_value"], int)
        self.assertIsInstance(builder.input_recipe_dict["float_value"], float)
        self.assertIsInstance(builder.input_recipe_dict["bool_true"], bool)
        self.assertIsInstance(builder.input_recipe_dict["bool_false"], bool)
        self.assertIsInstance(builder.input_recipe_dict["list_value"], list)
        self.assertIsInstance(builder.input_recipe_dict["nested_dict"], dict)

        self.assertEqual(builder.input_recipe_dict["scientific_string"], 1e-5)
        self.assertIsInstance(builder.input_recipe_dict["scientific_string"], float)

    @patch("amzn_nova_forge.recipe.recipe_builder.load_file_as_string")
    def test_load_input_recipe_handles_uppercase_scientific_notation(self, mock_load_file):
        builder = RecipeBuilder(
            region=self.region,
            job_name=self.job_name,
            platform=self.platform,
            model=self.mock_model,
            method=self.method,
            instance_type=self.instance_type,
            instance_count=self.instance_count,
            infra=self.mock_infra,
            output_s3_path=self.output_s3,
            data_s3_path=self.data_s3,
        )

        input_recipe = {
            "lowercase": "1e-5",
            "uppercase": "1E-5",
            "mixed_case": "2.5E-3",
        }
        mock_load_file.return_value = yaml.dump(input_recipe)

        builder._load_input_recipe("test_path.yaml")

        self.assertEqual(builder.input_recipe_dict["lowercase"], 1e-5)
        self.assertEqual(builder.input_recipe_dict["uppercase"], 1e-5)
        self.assertEqual(builder.input_recipe_dict["mixed_case"], 2.5e-3)
        self.assertIsInstance(builder.input_recipe_dict["lowercase"], float)
        self.assertIsInstance(builder.input_recipe_dict["uppercase"], float)
        self.assertIsInstance(builder.input_recipe_dict["mixed_case"], float)

    @patch("amzn_nova_forge.recipe.recipe_builder.load_file_as_string")
    def test_load_input_recipe_handles_invalid_scientific_notation_gracefully(self, mock_load_file):
        builder = RecipeBuilder(
            region=self.region,
            job_name=self.job_name,
            platform=self.platform,
            model=self.mock_model,
            method=self.method,
            instance_type=self.instance_type,
            instance_count=self.instance_count,
            infra=self.mock_infra,
            output_s3_path=self.output_s3,
            data_s3_path=self.data_s3,
        )

        input_recipe = {
            "valid": "1e-5",
            "invalid1": "test-e-value",
            "invalid2": "e-5",
            "invalid3": "experiment",
        }
        mock_load_file.return_value = yaml.dump(input_recipe)

        builder._load_input_recipe("test_path.yaml")

        self.assertEqual(builder.input_recipe_dict["valid"], 1e-5)
        self.assertIsInstance(builder.input_recipe_dict["valid"], float)

        self.assertEqual(builder.input_recipe_dict["invalid1"], "test-e-value")
        self.assertEqual(builder.input_recipe_dict["invalid2"], "e-5")
        self.assertEqual(builder.input_recipe_dict["invalid3"], "experiment")
        self.assertIsInstance(builder.input_recipe_dict["invalid1"], str)
        self.assertIsInstance(builder.input_recipe_dict["invalid2"], str)
        self.assertIsInstance(builder.input_recipe_dict["invalid3"], str)

    @patch("amzn_nova_forge.util.recipe.get_hub_recipe_metadata")
    @patch("amzn_nova_forge.util.recipe.download_templates_from_s3")
    @patch("amzn_nova_forge.recipe.recipe_builder.Validator")
    @patch("amzn_nova_forge.recipe.recipe_builder.load_file_as_string")
    def test_build_and_validate_with_input_recipe_replicas_not_in_override_template(
        self, mock_load_file, mock_validator, mock_download, mock_metadata
    ):
        mock_metadata.return_value = {"recipe_uri": "s3://bucket/recipe"}

        recipe_template = {
            "run": {
                "name": "{{name}}",
                "replicas": 10,
            },
        }

        overrides_template = {
            "name": {"default": "", "type": "string"},
        }

        mock_download.return_value = (recipe_template, overrides_template, "image_uri")

        input_recipe = {
            "run": {
                "name": "custom-job-name",
                "replicas": 20,
            },
        }
        mock_load_file.return_value = yaml.dump(input_recipe)

        builder = RecipeBuilder(
            region=self.region,
            job_name=self.job_name,
            platform=self.platform,
            model=self.mock_model,
            method=self.method,
            instance_type=self.instance_type,
            instance_count=self.instance_count,
            infra=self.mock_infra,
            output_s3_path=self.output_s3,
            data_s3_path=self.data_s3,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, "recipe.yaml")
            input_recipe_path = os.path.join(tmpdir, "input_recipe.yaml")

            recipe_path, *_ = builder.build_and_validate(
                input_recipe_path=input_recipe_path,
                output_recipe_path=output_path,
            )

            self.assertTrue(os.path.exists(recipe_path))

            with open(recipe_path, "r") as f:
                config = yaml.safe_load(f)

            self.assertEqual(config["run"]["replicas"], 20)

    @patch("amzn_nova_forge.util.recipe.get_hub_recipe_metadata")
    @patch("amzn_nova_forge.util.recipe.download_templates_from_s3")
    @patch("amzn_nova_forge.recipe.recipe_builder.Validator")
    @patch("amzn_nova_forge.recipe.recipe_builder.load_file_as_string")
    def test_build_and_validate_with_input_recipe_override_name_different_from_recipe_name(
        self, mock_load_file, mock_validator, mock_download, mock_metadata
    ):
        mock_metadata.return_value = {"recipe_uri": "s3://bucket/recipe"}

        recipe_template = {
            "run": {
                "name": "{{name}}",
                "replicas": 1,
                "model_type": self.mock_model.model_type,
                "data_s3_path": "{{data_s3_path}}",
                "output_s3_path": "{{output_s3_path}}",
            },
            "training_config": {
                "alpha": "{{lora_alpha}}",
            },
        }

        overrides_template = {
            "name": {"default": "", "type": "string"},
            "data_s3_path": {"default": "", "type": "string"},
            "output_s3_path": {"default": "", "type": "string"},
            "lora_alpha": {"default": 10, "type": "int"},
        }

        mock_download.return_value = (recipe_template, overrides_template, "image_uri")

        input_recipe = {
            "run": {
                "name": "custom-job-name",
                "model_type": self.mock_model.model_type,
                "data_s3_path": "s3://bucket/custom-data",
                "output_s3_path": "s3://bucket/custom-output",
            },
            "training_config": {
                "alpha": 20,
            },
        }
        mock_load_file.return_value = yaml.dump(input_recipe)

        builder = RecipeBuilder(
            region=self.region,
            job_name=self.job_name,
            platform=self.platform,
            model=self.mock_model,
            method=self.method,
            instance_type=self.instance_type,
            instance_count=self.instance_count,
            infra=self.mock_infra,
            output_s3_path=self.output_s3,
            data_s3_path=self.data_s3,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, "recipe.yaml")
            input_recipe_path = os.path.join(tmpdir, "input_recipe.yaml")

            recipe_path, *_ = builder.build_and_validate(
                input_recipe_path=input_recipe_path,
                output_recipe_path=output_path,
            )

            self.assertTrue(os.path.exists(recipe_path))

            with open(recipe_path, "r") as f:
                config = yaml.safe_load(f)

            self.assertEqual(config["run"]["name"], "custom-job-name")
            self.assertEqual(config["run"]["data_s3_path"], "s3://bucket/custom-data")
            self.assertEqual(config["run"]["output_s3_path"], "s3://bucket/custom-output")
            self.assertEqual(config["training_config"]["alpha"], 20)

            mock_validator.validate.assert_called_once()

            mock_load_file.assert_called_once_with(input_recipe_path, ".yaml")

    @patch("amzn_nova_forge.recipe.recipe_builder.logger")
    @patch("amzn_nova_forge.util.recipe.get_hub_recipe_metadata")
    @patch("amzn_nova_forge.util.recipe.download_templates_from_s3")
    @patch("amzn_nova_forge.recipe.recipe_builder.Validator")
    def test_model_type_override_is_ignored(
        self, mock_validator, mock_download, mock_metadata, mock_logger
    ):
        mock_metadata.return_value = {"recipe_uri": "s3://bucket/recipe"}

        recipe_template = {
            "run": {
                "name": "{{name}}",
                "model_type": "{{model_type}}",
            }
        }

        overrides_template = {
            "name": {"default": "", "type": "string"},
            "model_type": {"default": "test-model", "type": "string"},
        }

        mock_download.return_value = (recipe_template, overrides_template, "image_uri")

        builder = RecipeBuilder(
            region=self.region,
            job_name=self.job_name,
            platform=self.platform,
            model=self.mock_model,
            method=self.method,
            instance_type=self.instance_type,
            instance_count=self.instance_count,
            infra=self.mock_infra,
            output_s3_path=self.output_s3,
            data_s3_path=self.data_s3,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, "recipe.yaml")

            overrides = {"model_type": "different-model"}
            recipe_path, *_ = builder.build_and_validate(
                overrides=overrides,
                output_recipe_path=output_path,
            )

            mock_logger.warning.assert_any_call(
                f"Override for 'model_type' will be ignored. If you wish to use a different model than {self.mock_model.name}, please update your NovaModelCustomizer object."
            )

            with open(recipe_path, "r") as f:
                config = yaml.safe_load(f)

            self.assertEqual(config["run"]["model_type"], "test-model")

    @patch("amzn_nova_forge.recipe.recipe_builder.logger")
    @patch("amzn_nova_forge.util.recipe.get_hub_recipe_metadata")
    @patch("amzn_nova_forge.util.recipe.download_templates_from_s3")
    @patch("amzn_nova_forge.recipe.recipe_builder.Validator")
    @patch("amzn_nova_forge.recipe.recipe_builder.load_file_as_string")
    def test_model_type_from_input_recipe_ignored_when_different(
        self, mock_load_file, mock_validator, mock_download, mock_metadata, mock_logger
    ):
        mock_metadata.return_value = {"recipe_uri": "s3://bucket/recipe"}

        recipe_template = {
            "run": {
                "name": "{{name}}",
                "model_type": "{{model_type}}",
            }
        }

        overrides_template = {
            "name": {"default": "", "type": "string"},
            "model_type": {"default": "test-model", "type": "string"},
        }

        mock_download.return_value = (recipe_template, overrides_template, "image_uri")

        input_recipe = {
            "run": {
                "name": "test",
                "model_type": "different-model",
            }
        }
        mock_load_file.return_value = yaml.dump(input_recipe)

        builder = RecipeBuilder(
            region=self.region,
            job_name=self.job_name,
            platform=self.platform,
            model=self.mock_model,
            method=self.method,
            instance_type=self.instance_type,
            instance_count=self.instance_count,
            infra=self.mock_infra,
            output_s3_path=self.output_s3,
            data_s3_path=self.data_s3,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, "recipe.yaml")
            input_recipe_path = os.path.join(tmpdir, "input_recipe.yaml")

            recipe_path, *_ = builder.build_and_validate(
                input_recipe_path=input_recipe_path,
                output_recipe_path=output_path,
            )

            mock_logger.warning.assert_any_call(
                f"model_type 'different-model' will be ignored from your input recipe. If you wish to use a different model than {self.mock_model.name}, please update your NovaModelCustomizer object."
            )

            with open(recipe_path, "r") as f:
                config = yaml.safe_load(f)

            self.assertEqual(config["run"]["model_type"], "test-model")

    @patch("amzn_nova_forge.recipe.recipe_builder.logger")
    @patch("amzn_nova_forge.util.recipe.get_hub_recipe_metadata")
    @patch("amzn_nova_forge.util.recipe.download_templates_from_s3")
    @patch("amzn_nova_forge.recipe.recipe_builder.Validator")
    def test_model_name_or_path_non_s3_override_ignored(
        self, mock_validator, mock_download, mock_metadata, mock_logger
    ):
        mock_metadata.return_value = {"recipe_uri": "s3://bucket/recipe"}

        recipe_template = {
            "run": {
                "name": "{{name}}",
                "model_name_or_path": "{{model_name_or_path}}",
            }
        }

        overrides_template = {
            "name": {"default": "", "type": "string"},
            "model_name_or_path": {"default": "models/test", "type": "string"},
        }

        mock_download.return_value = (recipe_template, overrides_template, "image_uri")

        builder = RecipeBuilder(
            region=self.region,
            job_name=self.job_name,
            platform=self.platform,
            model=self.mock_model,
            method=self.method,
            instance_type=self.instance_type,
            instance_count=self.instance_count,
            infra=self.mock_infra,
            output_s3_path=self.output_s3,
            data_s3_path=self.data_s3,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, "recipe.yaml")

            overrides = {"model_name_or_path": "local/path/to/model"}
            recipe_path, *_ = builder.build_and_validate(
                overrides=overrides,
                output_recipe_path=output_path,
            )

            mock_logger.warning.assert_any_call(
                f"Override for 'model_name_or_path' will be ignored. If you wish to use a different model than {self.mock_model.name}, please update your NovaModelCustomizer object."
            )

            with open(recipe_path, "r") as f:
                config = yaml.safe_load(f)

            self.assertEqual(config["run"]["model_name_or_path"], "models/test")

    @patch("amzn_nova_forge.recipe.recipe_builder.validate_checkpoint_uri")
    @patch("amzn_nova_forge.util.recipe.get_hub_recipe_metadata")
    @patch("amzn_nova_forge.util.recipe.download_templates_from_s3")
    @patch("amzn_nova_forge.recipe.recipe_builder.Validator")
    def test_model_name_or_path_s3_override_validated(
        self, mock_validator, mock_download, mock_metadata, mock_validate_checkpoint
    ):
        mock_metadata.return_value = {"recipe_uri": "s3://bucket/recipe"}

        recipe_template = {
            "run": {
                "name": "{{name}}",
                "model_name_or_path": "model/test",
            }
        }

        overrides_template = {
            "name": {"default": "", "type": "string"},
        }

        mock_download.return_value = (recipe_template, overrides_template, "image_uri")

        builder = RecipeBuilder(
            region=self.region,
            job_name=self.job_name,
            platform=self.platform,
            model=self.mock_model,
            method=self.method,
            instance_type=self.instance_type,
            instance_count=self.instance_count,
            infra=self.mock_infra,
            output_s3_path=self.output_s3,
            data_s3_path=self.data_s3,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, "recipe.yaml")

            s3_checkpoint = "s3://bucket/custom-checkpoint"
            overrides = {"model_name_or_path": s3_checkpoint}
            recipe_path, *_ = builder.build_and_validate(
                overrides=overrides,
                output_recipe_path=output_path,
            )

            mock_validate_checkpoint.assert_called_with(
                checkpoint_uri=s3_checkpoint, region=self.region
            )

            with open(recipe_path, "r") as f:
                config = yaml.safe_load(f)

            self.assertEqual(config["run"]["model_name_or_path"], s3_checkpoint)

    @patch("amzn_nova_forge.util.recipe.get_hub_recipe_metadata")
    @patch("amzn_nova_forge.util.recipe.download_templates_from_s3")
    @patch("amzn_nova_forge.recipe.recipe_builder.Validator")
    def test_model_name_or_path_model_package_arn_override_accepted(
        self, mock_validator, mock_download, mock_metadata
    ):
        mock_metadata.return_value = {"recipe_uri": "s3://bucket/recipe"}

        recipe_template = {
            "run": {
                "name": "{{name}}",
                "model_name_or_path": "{{model_name_or_path}}",
            }
        }

        overrides_template = {
            "name": {"default": "", "type": "string"},
            "model_name_or_path": {"default": "models/test", "type": "string"},
        }

        mock_download.return_value = (recipe_template, overrides_template, "image_uri")

        builder = RecipeBuilder(
            region=self.region,
            job_name=self.job_name,
            platform=self.platform,
            model=self.mock_model,
            method=self.method,
            instance_type=self.instance_type,
            instance_count=self.instance_count,
            infra=self.mock_infra,
            output_s3_path=self.output_s3,
            data_s3_path=self.data_s3,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, "recipe.yaml")

            mp_arn = "arn:aws:sagemaker:us-east-1:123456789012:model-package/my-mpg/1"
            overrides = {"model_name_or_path": mp_arn}
            recipe_path, *_ = builder.build_and_validate(
                overrides=overrides,
                output_recipe_path=output_path,
            )

            with open(recipe_path, "r") as f:
                config = yaml.safe_load(f)

            self.assertEqual(config["run"]["model_name_or_path"], mp_arn)

    @patch("amzn_nova_forge.util.recipe.get_hub_recipe_metadata")
    @patch("amzn_nova_forge.util.recipe.download_templates_from_s3")
    @patch("amzn_nova_forge.recipe.recipe_builder.Validator")
    @patch("amzn_nova_forge.recipe.recipe_builder.load_file_as_string")
    def test_model_name_or_path_model_package_arn_from_input_recipe_accepted(
        self, mock_load_file, mock_validator, mock_download, mock_metadata
    ):
        mock_metadata.return_value = {"recipe_uri": "s3://bucket/recipe"}

        recipe_template = {
            "run": {
                "name": "{{name}}",
                "model_name_or_path": "{{model_name_or_path}}",
            }
        }

        overrides_template = {
            "name": {"default": "", "type": "string"},
            "model_name_or_path": {"default": "models/test", "type": "string"},
        }

        mock_download.return_value = (recipe_template, overrides_template, "image_uri")

        mp_arn = "arn:aws:sagemaker:us-east-1:123456789012:model-package/my-mpg/1"
        input_recipe = {
            "run": {
                "name": "test",
                "model_name_or_path": mp_arn,
            }
        }
        mock_load_file.return_value = yaml.dump(input_recipe)

        builder = RecipeBuilder(
            region=self.region,
            job_name=self.job_name,
            platform=self.platform,
            model=self.mock_model,
            method=self.method,
            instance_type=self.instance_type,
            instance_count=self.instance_count,
            infra=self.mock_infra,
            output_s3_path=self.output_s3,
            data_s3_path=self.data_s3,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, "recipe.yaml")
            input_recipe_path = os.path.join(tmpdir, "input_recipe.yaml")

            recipe_path, *_ = builder.build_and_validate(
                input_recipe_path=input_recipe_path,
                output_recipe_path=output_path,
            )

            with open(recipe_path, "r") as f:
                config = yaml.safe_load(f)

            self.assertEqual(config["run"]["model_name_or_path"], mp_arn)

    @patch("amzn_nova_forge.recipe.recipe_builder.logger")
    @patch("amzn_nova_forge.util.checkpoint_util.validate_checkpoint_uri")
    @patch("amzn_nova_forge.util.recipe.get_hub_recipe_metadata")
    @patch("amzn_nova_forge.util.recipe.download_templates_from_s3")
    @patch("amzn_nova_forge.recipe.recipe_builder.Validator")
    @patch("amzn_nova_forge.recipe.recipe_builder.load_file_as_string")
    def test_model_name_or_path_from_input_recipe_non_s3_ignored(
        self,
        mock_load_file,
        mock_validator,
        mock_download,
        mock_metadata,
        mock_validate_checkpoint,
        mock_logger,
    ):
        mock_metadata.return_value = {"recipe_uri": "s3://bucket/recipe"}

        recipe_template = {
            "run": {
                "name": "{{name}}",
                "model_name_or_path": "{{model_name_or_path}}",
            }
        }

        overrides_template = {
            "name": {"default": "", "type": "string"},
            "model_name_or_path": {"default": "models/test", "type": "string"},
        }

        mock_download.return_value = (recipe_template, overrides_template, "image_uri")

        input_recipe = {
            "run": {
                "name": "test",
                "model_name_or_path": "different/model/path",
            }
        }
        mock_load_file.return_value = yaml.dump(input_recipe)

        builder = RecipeBuilder(
            region=self.region,
            job_name=self.job_name,
            platform=self.platform,
            model=self.mock_model,
            method=self.method,
            instance_type=self.instance_type,
            instance_count=self.instance_count,
            infra=self.mock_infra,
            output_s3_path=self.output_s3,
            data_s3_path=self.data_s3,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, "recipe.yaml")
            input_recipe_path = os.path.join(tmpdir, "input_recipe.yaml")

            recipe_path, *_ = builder.build_and_validate(
                input_recipe_path=input_recipe_path,
                output_recipe_path=output_path,
            )

            mock_logger.warning.assert_any_call(
                f"model_name_or_path 'different/model/path' will be ignored from your input recipe. If you wish to use a different model than {self.mock_model.name}, please update your NovaModelCustomizer object."
            )

            with open(recipe_path, "r") as f:
                config = yaml.safe_load(f)

            self.assertEqual(config["run"]["model_name_or_path"], "models/test")

    @patch("amzn_nova_forge.recipe.recipe_builder.validate_checkpoint_uri")
    @patch("amzn_nova_forge.util.recipe.get_hub_recipe_metadata")
    @patch("amzn_nova_forge.util.recipe.download_templates_from_s3")
    @patch("amzn_nova_forge.recipe.recipe_builder.Validator")
    @patch("amzn_nova_forge.recipe.recipe_builder.load_file_as_string")
    def test_model_name_or_path_from_input_recipe_s3_validated(
        self,
        mock_load_file,
        mock_validator,
        mock_download,
        mock_metadata,
        mock_validate_checkpoint,
    ):
        mock_metadata.return_value = {"recipe_uri": "s3://bucket/recipe"}

        recipe_template = {
            "run": {
                "name": "{{name}}",
                "model_name_or_path": "models/test",
            }
        }

        overrides_template = {
            "name": {"default": "", "type": "string"},
        }

        mock_download.return_value = (recipe_template, overrides_template, "image_uri")

        s3_checkpoint = "s3://bucket/input-checkpoint"
        input_recipe = {
            "run": {
                "name": "test",
                "model_name_or_path": s3_checkpoint,
            }
        }
        mock_load_file.return_value = yaml.dump(input_recipe)

        builder = RecipeBuilder(
            region=self.region,
            job_name=self.job_name,
            platform=self.platform,
            model=self.mock_model,
            method=self.method,
            instance_type=self.instance_type,
            instance_count=self.instance_count,
            infra=self.mock_infra,
            output_s3_path=self.output_s3,
            data_s3_path=self.data_s3,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, "recipe.yaml")
            input_recipe_path = os.path.join(tmpdir, "input_recipe.yaml")

            recipe_path, *_ = builder.build_and_validate(
                input_recipe_path=input_recipe_path,
                output_recipe_path=output_path,
            )

            mock_validate_checkpoint.assert_called_with(
                checkpoint_uri=s3_checkpoint, region=self.region
            )

            with open(recipe_path, "r") as f:
                config = yaml.safe_load(f)

            self.assertEqual(config["run"]["model_name_or_path"], s3_checkpoint)

    @patch("amzn_nova_forge.recipe.recipe_builder.logger")
    @patch("amzn_nova_forge.util.recipe.get_hub_recipe_metadata")
    @patch("amzn_nova_forge.util.recipe.download_templates_from_s3")
    @patch("amzn_nova_forge.recipe.recipe_builder.Validator")
    def test_task_override_is_ignored(
        self, mock_validator, mock_download, mock_metadata, mock_logger
    ):
        mock_metadata.return_value = {"recipe_uri": "s3://bucket/recipe"}

        recipe_template = {
            "run": {
                "name": "{{name}}",
            },
            "evaluation": {
                "task": "{{task}}",
            },
        }

        overrides_template = {
            "name": {"default": "", "type": "string"},
            "task": {"default": EvaluationTask.MMLU.value, "type": "string"},
        }

        mock_download.return_value = (recipe_template, overrides_template, "image_uri")

        builder = RecipeBuilder(
            region=self.region,
            job_name=self.job_name,
            platform=self.platform,
            model=self.mock_model,
            method=TrainingMethod.EVALUATION,
            instance_type=self.instance_type,
            instance_count=self.instance_count,
            infra=self.mock_infra,
            output_s3_path=self.output_s3,
            data_s3_path=self.data_s3,
            eval_task=EvaluationTask.MMLU,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, "recipe.yaml")

            overrides = {"task": EvaluationTask.MATH.value}
            recipe_path, *_ = builder.build_and_validate(
                overrides=overrides,
                output_recipe_path=output_path,
            )

            self.assertTrue(os.path.exists(recipe_path))

            mock_logger.warning.assert_any_call(
                f"Override for 'task' will be ignored. If you wish to use an evaluation task other than {EvaluationTask.MMLU.name}, please pass a different value for 'eval_task' when calling evaluate()."
            )

            with open(recipe_path, "r") as f:
                config = yaml.safe_load(f)

            self.assertEqual(config["evaluation"]["task"], EvaluationTask.MMLU.value)

    @patch("amzn_nova_forge.recipe.recipe_builder.logger")
    @patch("amzn_nova_forge.util.recipe.get_hub_recipe_metadata")
    @patch("amzn_nova_forge.util.recipe.download_templates_from_s3")
    @patch("amzn_nova_forge.recipe.recipe_builder.Validator")
    @patch("amzn_nova_forge.recipe.recipe_builder.load_file_as_string")
    def test_task_from_input_recipe_ignored_when_different(
        self, mock_load_file, mock_validator, mock_download, mock_metadata, mock_logger
    ):
        mock_metadata.return_value = {"recipe_uri": "s3://bucket/recipe"}

        recipe_template = {
            "run": {
                "name": "{{name}}",
            },
            "evaluation": {
                "task": "{{task}}",
            },
        }

        overrides_template = {
            "name": {"default": "", "type": "string"},
            "task": {"default": EvaluationTask.MMLU, "type": "string"},
        }

        mock_download.return_value = (recipe_template, overrides_template, "image_uri")

        input_recipe = {
            "run": {
                "name": "test",
            },
            "evaluation": {
                "task": EvaluationTask.MATH.value,
            },
        }
        mock_load_file.return_value = yaml.dump(input_recipe)

        builder = RecipeBuilder(
            region=self.region,
            job_name=self.job_name,
            platform=self.platform,
            model=self.mock_model,
            method=TrainingMethod.EVALUATION,
            instance_type=self.instance_type,
            instance_count=self.instance_count,
            infra=self.mock_infra,
            output_s3_path=self.output_s3,
            data_s3_path=self.data_s3,
            eval_task=EvaluationTask.MMLU,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, "recipe.yaml")
            input_recipe_path = os.path.join(tmpdir, "input_recipe.yaml")

            recipe_path, *_ = builder.build_and_validate(
                input_recipe_path=input_recipe_path,
                output_recipe_path=output_path,
            )

            self.assertTrue(os.path.exists(recipe_path))

            mock_logger.warning.assert_any_call(
                f"task '{EvaluationTask.MATH.value}' will be ignored from your input recipe. If you wish to use a different evaluation task than {EvaluationTask.MMLU.name}, please pass a different value for 'eval_task' when calling evaluate()."
            )

    @patch("amzn_nova_forge.recipe.recipe_builder.logger")
    @patch("amzn_nova_forge.util.recipe.get_hub_recipe_metadata")
    @patch("amzn_nova_forge.util.recipe.download_templates_from_s3")
    @patch("amzn_nova_forge.recipe.recipe_builder.Validator")
    @patch("amzn_nova_forge.recipe.recipe_builder.load_file_as_string")
    def test_task_from_input_recipe_accepted_when_same(
        self, mock_load_file, mock_validator, mock_download, mock_metadata, mock_logger
    ):
        mock_metadata.return_value = {"recipe_uri": "s3://bucket/recipe"}

        recipe_template = {
            "run": {
                "name": "{{name}}",
            },
            "evaluation": {
                "task": "{{task}}",
            },
        }

        overrides_template = {
            "name": {"default": "", "type": "string"},
            "task": {"default": EvaluationTask.MMLU.value, "type": "string"},
        }

        mock_download.return_value = (recipe_template, overrides_template, "image_uri")

        input_recipe = {
            "run": {
                "name": "test",
            },
            "evaluation": {
                "task": EvaluationTask.MMLU.value,
            },
        }
        mock_load_file.return_value = yaml.dump(input_recipe)

        builder = RecipeBuilder(
            region=self.region,
            job_name=self.job_name,
            platform=self.platform,
            model=self.mock_model,
            method=TrainingMethod.EVALUATION,
            instance_type=self.instance_type,
            instance_count=self.instance_count,
            infra=self.mock_infra,
            output_s3_path=self.output_s3,
            data_s3_path=self.data_s3,
            eval_task=EvaluationTask.MMLU,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, "recipe.yaml")
            input_recipe_path = os.path.join(tmpdir, "input_recipe.yaml")

            recipe_path, *_ = builder.build_and_validate(
                input_recipe_path=input_recipe_path,
                output_recipe_path=output_path,
            )

            mock_logger.warning.assert_not_called()

    @patch("amzn_nova_forge.recipe.recipe_builder.logger")
    @patch("amzn_nova_forge.util.recipe.get_hub_recipe_metadata")
    @patch("amzn_nova_forge.util.recipe.download_templates_from_s3")
    @patch("amzn_nova_forge.recipe.recipe_builder.Validator")
    def test_non_overrideable_params_are_ignored(
        self, mock_validator, mock_download, mock_metadata, mock_logger
    ):
        mock_metadata.return_value = {"recipe_uri": "s3://bucket/recipe"}

        recipe_template = {
            "run": {"name": "{{name}}"},
            "training_config": {
                "model": {
                    "peft": {
                        "peft_scheme": "lora",
                    }
                }
            },
        }

        overrides_template = {
            "name": {"default": "", "type": "string"},
            "peft_scheme": {"default": "lora", "type": "string"},
        }

        mock_download.return_value = (recipe_template, overrides_template, "image_uri")

        builder = RecipeBuilder(
            region=self.region,
            job_name=self.job_name,
            platform=self.platform,
            model=self.mock_model,
            method=self.method,
            instance_type=self.instance_type,
            instance_count=self.instance_count,
            infra=self.mock_infra,
            output_s3_path=self.output_s3,
            data_s3_path=self.data_s3,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, "recipe.yaml")

            overrides = {"peft_scheme": "custom_peft"}
            recipe_path, *_ = builder.build_and_validate(
                overrides=overrides,
                output_recipe_path=output_path,
            )

            mock_logger.warning.assert_any_call(
                "'peft_scheme' is not an overrideable parameter. Will be ignored."
            )

            with open(recipe_path, "r") as f:
                config = yaml.safe_load(f)

            self.assertEqual(config["training_config"]["model"]["peft"]["peft_scheme"], "lora")

    @patch("amzn_nova_forge.util.recipe.get_hub_recipe_metadata")
    @patch("amzn_nova_forge.util.recipe.download_templates_from_local")
    @patch("amzn_nova_forge.recipe.recipe_builder.Validator")
    def test_build_and_validate_evaluation_uses_local_templates_for_special_tasks(
        self, mock_validator, mock_download_local, mock_metadata
    ):
        mock_metadata.return_value = {
            "RecipeTemplatePath": "/path/to/recipe.yaml",
            "OverrideParamsPath": "/path/to/override.json",
            "ImageUri": "image_uri",
        }

        recipe_template = {
            "run": {
                "name": "{{name}}",
                "replicas": 1,
                "model_type": "{{model_type}}",
                "model_name_or_path": "{{model_name_or_path}}",
            },
            "evaluation": {
                "task": "{{task}}",
                "strategy": "{{strategy}}",
                "metric": "{{metric}}",
            },
        }

        overrides_template = {
            "name": {"default": "", "type": "string"},
            "model_type": {"default": "", "type": "string"},
            "model_name_or_path": {"default": "", "type": "string"},
        }

        mock_download_local.return_value = (
            recipe_template,
            overrides_template,
            "image_uri",
        )

        special_tasks = [
            EvaluationTask.LLM_JUDGE,
            EvaluationTask.RUBRIC_LLM_JUDGE,
            EvaluationTask.RFT_EVAL,
        ]

        for eval_task in special_tasks:
            with self.subTest(eval_task=eval_task.value):
                mock_download_local.reset_mock()

                builder = RecipeBuilder(
                    region=self.region,
                    job_name=self.job_name,
                    platform=self.platform,
                    model=self.mock_model,
                    method=TrainingMethod.EVALUATION,
                    instance_type=self.instance_type,
                    instance_count=self.instance_count,
                    infra=self.mock_infra,
                    output_s3_path=self.output_s3,
                    data_s3_path=self.data_s3,
                    eval_task=eval_task,
                )

                with tempfile.TemporaryDirectory() as tmpdir:
                    output_path = os.path.join(tmpdir, "recipe.yaml")

                    recipe_path, *_ = builder.build_and_validate(output_recipe_path=output_path)

                    mock_download_local.assert_called_once_with(mock_metadata.return_value)

                    self.assertTrue(os.path.exists(recipe_path))

                    with open(recipe_path, "r") as f:
                        config = yaml.safe_load(f)

                    self.assertEqual(config["run"]["name"], self.job_name)

    @patch("amzn_nova_forge.util.recipe.get_hub_recipe_metadata")
    @patch("amzn_nova_forge.util.recipe.download_templates_from_local")
    @patch("amzn_nova_forge.recipe.recipe_builder.Validator")
    def test_data_s3_path_never_null(self, mock_validator, mock_download_local, mock_metadata):
        """
        Ensure data_s3_path defaults to empty string, not null.

        The eval containers currently don't support `null` or the absence of
        data_s3_path - they require an empty string '' when no custom data is used.
        """
        mock_metadata.return_value = {
            "RecipeTemplatePath": "/path/to/recipe.yaml",
            "OverrideParamsPath": "/path/to/override.json",
            "ImageUri": "image_uri",
        }

        recipe_template = {
            "run": {
                "name": "{{name}}",
                "data_s3_path": "{{data_s3_path}}",
            },
        }

        overrides_template = {
            "name": {"default": "", "type": "string"},
            "data_s3_path": {"default": "", "type": "string"},
        }

        mock_download_local.return_value = (
            recipe_template,
            overrides_template,
            "image_uri",
        )

        builder = RecipeBuilder(
            region=self.region,
            job_name=self.job_name,
            platform=self.platform,
            model=self.mock_model,
            method=TrainingMethod.EVALUATION,
            instance_type=self.instance_type,
            instance_count=self.instance_count,
            infra=self.mock_infra,
            output_s3_path=self.output_s3,
            data_s3_path=None,
            eval_task=EvaluationTask.LLM_JUDGE,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, "recipe.yaml")
            builder.build_and_validate(output_recipe_path=output_path)

            with open(output_path, "r") as f:
                config = yaml.safe_load(f)

            self.assertEqual(config["run"]["data_s3_path"], "")

    @patch("amzn_nova_forge.util.recipe.get_hub_recipe_metadata")
    @patch("amzn_nova_forge.util.recipe.download_templates_from_s3")
    @patch("amzn_nova_forge.recipe.recipe_builder.Validator")
    def test_build_and_validate_updates_infra_instance_count(
        self, mock_validator, mock_download, mock_metadata
    ):
        mock_metadata.return_value = {"recipe_uri": "s3://bucket/recipe"}

        recipe_template = {
            "run": {
                "name": "{{name}}",
                "replicas": "{{replicas}}",
            }
        }

        overrides_template = {
            "name": {"default": "", "type": "string"},
            "replicas": {"default": 1, "type": "int"},
        }

        mock_download.return_value = (recipe_template, overrides_template, "image_uri")

        builder = RecipeBuilder(
            region=self.region,
            job_name=self.job_name,
            platform=self.platform,
            model=self.mock_model,
            method=self.method,
            instance_type=self.instance_type,
            instance_count=1,
            infra=self.mock_infra,
            output_s3_path=self.output_s3,
            data_s3_path=self.data_s3,
        )

        self.assertEqual(self.mock_infra.instance_count, 1)

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, "recipe.yaml")

            recipe_path, *_ = builder.build_and_validate(
                output_recipe_path=output_path,
                overrides={"replicas": 10},
            )

            self.assertEqual(self.mock_infra.instance_count, 10)

            with open(recipe_path, "r") as f:
                config = yaml.safe_load(f)

            self.assertEqual(config["run"]["replicas"], 10)

    def test_initialization_with_data_mixing(self):
        """Test RecipeBuilder initialization with data_mixing flag."""
        builder = RecipeBuilder(
            region=self.region,
            job_name=self.job_name,
            platform=self.platform,
            model=self.mock_model,
            method=self.method,
            instance_type=self.instance_type,
            instance_count=self.instance_count,
            infra=self.mock_infra,
            output_s3_path=self.output_s3,
            data_s3_path=self.data_s3,
            data_mixing_instance=None,
        )

        self.assertIsNone(builder.data_mixing_instance)

    def test_initialization_without_data_mixing(self):
        """Test RecipeBuilder initialization without data_mixing flag."""
        builder = RecipeBuilder(
            region=self.region,
            job_name=self.job_name,
            platform=self.platform,
            model=self.mock_model,
            method=self.method,
            instance_type=self.instance_type,
            instance_count=self.instance_count,
            infra=self.mock_infra,
            output_s3_path=self.output_s3,
            data_s3_path=self.data_s3,
        )

        self.assertIsNone(builder.data_mixing_instance)

    @patch("amzn_nova_forge.util.recipe.get_hub_recipe_metadata")
    @patch("amzn_nova_forge.util.recipe.download_templates_from_s3")
    @patch("amzn_nova_forge.recipe.recipe_builder.Validator")
    def test_build_and_validate_with_data_mixing_instance(
        self, mock_validator, mock_download, mock_metadata
    ):
        """Test build_and_validate with DataMixing instance parameter."""
        mock_metadata.return_value = {"recipe_uri": "s3://bucket/recipe"}

        recipe_template = {
            "run": {"name": "{{name}}"},
            "data_mixing": {
                "percent": "{{customer_data_percent}}",
                "code": "{{nova_code_percent}}",
                "general": "{{nova_general_percent}}",
            },
        }

        overrides_template = {
            "name": {"default": "", "type": "string"},
            "customer_data_percent": {"default": 50, "type": "int"},
            "nova_code_percent": {"default": 25, "type": "int"},
            "nova_general_percent": {"default": 75, "type": "int"},
        }

        mock_download.return_value = (recipe_template, overrides_template, "image_uri")

        # Create a DataMixing instance with configuration
        data_mixing_instance = DataMixing()
        data_mixing_instance.set_config(
            {
                "customer_data_percent": 70,
                "nova_code_percent": 40,
                "nova_general_percent": 60,
            }
        )

        builder = RecipeBuilder(
            region=self.region,
            job_name=self.job_name,
            platform=self.platform,
            model=self.mock_model,
            method=self.method,
            instance_type=self.instance_type,
            instance_count=self.instance_count,
            infra=self.mock_infra,
            output_s3_path=self.output_s3,
            data_s3_path=self.data_s3,
            data_mixing_instance=data_mixing_instance,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, "recipe.yaml")

            recipe_path, *_ = builder.build_and_validate(output_recipe_path=output_path)

            with open(recipe_path, "r") as f:
                config = yaml.safe_load(f)
            # DataMixing instance values should be used
            self.assertEqual(config["data_mixing"]["percent"], 70)
            self.assertEqual(config["data_mixing"]["code"], 40)
            self.assertEqual(config["data_mixing"]["general"], 60)

    @patch("amzn_nova_forge.util.recipe.get_hub_recipe_metadata")
    @patch("amzn_nova_forge.util.recipe.download_templates_from_s3")
    @patch("amzn_nova_forge.recipe.recipe_builder.Validator")
    @patch("amzn_nova_forge.recipe.recipe_builder.load_file_as_string")
    def test_data_mixing_with_empty_sources(
        self, mock_load_file, mock_validator, mock_download, mock_metadata
    ):
        """Test data mixing with empty sources section."""
        mock_metadata.return_value = {"recipe_uri": "s3://bucket/recipe"}

        recipe_template = {
            "run": {"name": "{{name}}"},
            "data_mixing": {
                "customer_data_percent": "{{customer_data_percent}}",
            },
        }

        overrides_template = {
            "name": {"default": "", "type": "string"},
            "customer_data_percent": {"default": 100, "type": "int"},
        }

        mock_download.return_value = (recipe_template, overrides_template, "image_uri")

        input_recipe = {
            "run": {"name": "test-job"},
            "data_mixing": {
                "sources": {},  # Empty sources
                "dataset_catalog": "catalog_name",
            },
        }
        mock_load_file.return_value = yaml.dump(input_recipe)

        builder = RecipeBuilder(
            region=self.region,
            job_name=self.job_name,
            platform=self.platform,
            model=self.mock_model,
            method=self.method,
            instance_type=self.instance_type,
            instance_count=self.instance_count,
            infra=self.mock_infra,
            output_s3_path=self.output_s3,
            data_s3_path=self.data_s3,
            data_mixing_instance=None,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, "recipe.yaml")
            input_recipe_path = os.path.join(tmpdir, "input_recipe.yaml")

            recipe_path, *_ = builder.build_and_validate(
                input_recipe_path=input_recipe_path,
                output_recipe_path=output_path,
            )

            with open(recipe_path, "r") as f:
                config = yaml.safe_load(f)

            # Should use default value
            self.assertEqual(config["data_mixing"]["customer_data_percent"], 100)

    @patch("amzn_nova_forge.util.recipe.get_hub_recipe_metadata")
    @patch("amzn_nova_forge.util.recipe.download_templates_from_s3")
    @patch("amzn_nova_forge.recipe.recipe_builder.Validator")
    @patch("amzn_nova_forge.recipe.recipe_builder.load_file_as_string")
    def test_data_mixing_without_data_mixing_flag(
        self, mock_load_file, mock_validator, mock_download, mock_metadata
    ):
        """Test that data_mixing section is ignored when data_mixing flag is False."""
        mock_metadata.return_value = {"recipe_uri": "s3://bucket/recipe"}

        recipe_template = {
            "run": {"name": "{{name}}"},
        }

        overrides_template = {
            "name": {"default": "", "type": "string"},
        }

        mock_download.return_value = (recipe_template, overrides_template, "image_uri")

        input_recipe = {
            "run": {"name": "test-job"},
            "data_mixing": {
                "sources": {
                    "customer_data": {"percent": 60},
                    "nova_data": {
                        "code": 30,
                        "general": 70,
                    },
                },
            },
        }
        mock_load_file.return_value = yaml.dump(input_recipe)

        builder = RecipeBuilder(
            region=self.region,
            job_name=self.job_name,
            platform=self.platform,
            model=self.mock_model,
            method=self.method,
            instance_type=self.instance_type,
            instance_count=self.instance_count,
            infra=self.mock_infra,
            output_s3_path=self.output_s3,
            data_s3_path=self.data_s3,
            data_mixing_instance=None,  # data_mixing is disabled
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, "recipe.yaml")
            input_recipe_path = os.path.join(tmpdir, "input_recipe.yaml")

            recipe_path, *_ = builder.build_and_validate(
                input_recipe_path=input_recipe_path,
                output_recipe_path=output_path,
            )

            with open(recipe_path, "r") as f:
                config = yaml.safe_load(f)

            # data_mixing should not be in the output
            self.assertNotIn("data_mixing", config)

    @patch("amzn_nova_forge.recipe.recipe_builder.logger")
    @patch("amzn_nova_forge.util.recipe.get_hub_recipe_metadata")
    @patch("amzn_nova_forge.util.recipe.download_templates_from_s3")
    @patch("amzn_nova_forge.recipe.recipe_builder.Validator")
    def test_data_mixing_fields_ignored_in_overrides_with_data_mixing_instance(
        self, mock_validator, mock_download, mock_metadata, mock_logger
    ):
        """Test that data mixing fields in overrides are ignored when using data_mixing_instance."""
        mock_metadata.return_value = {"recipe_uri": "s3://bucket/recipe"}

        recipe_template = {
            "run": {"name": "{{name}}"},
            "data_mixing": {
                "customer_data_percent": "{{customer_data_percent}}",
                "nova_code_percent": "{{nova_code_percent}}",
                "nova_general_percent": "{{nova_general_percent}}",
            },
        }

        overrides_template = {
            "name": {"default": "", "type": "string"},
            "customer_data_percent": {"default": 50, "type": "int"},
            "nova_code_percent": {"default": 25, "type": "int"},
            "nova_general_percent": {"default": 75, "type": "int"},
        }

        mock_download.return_value = (recipe_template, overrides_template, "image_uri")

        # Create a DataMixing instance with configuration
        data_mixing_instance = DataMixing()
        data_mixing_instance.set_config(
            {
                "customer_data_percent": 70,
                "nova_code_percent": 40,
                "nova_general_percent": 60,
            }
        )

        builder = RecipeBuilder(
            region=self.region,
            job_name=self.job_name,
            platform=self.platform,
            model=self.mock_model,
            method=self.method,
            instance_type=self.instance_type,
            instance_count=self.instance_count,
            infra=self.mock_infra,
            output_s3_path=self.output_s3,
            data_s3_path=self.data_s3,
            data_mixing_instance=data_mixing_instance,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, "recipe.yaml")

            # Try to override data mixing fields
            overrides = {
                "customer_data_percent": 90,
                "nova_code_percent": 10,
                "nova_general_percent": 90,
            }

            recipe_path, *_ = builder.build_and_validate(
                output_recipe_path=output_path,
                overrides=overrides,
            )

            # Check that warnings were logged for each data mixing field
            mock_logger.warning.assert_any_call(
                "The following data mixing keys in overrides recipe will be ignored: customer_data_percent. "
                "Data mixing configuration can only be set using set_datamixing_config()."
            )
            mock_logger.warning.assert_any_call(
                "The following data mixing keys in overrides recipe will be ignored: nova_code_percent. "
                "Data mixing configuration can only be set using set_datamixing_config()."
            )
            mock_logger.warning.assert_any_call(
                "The following data mixing keys in overrides recipe will be ignored: nova_general_percent. "
                "Data mixing configuration can only be set using set_datamixing_config()."
            )

            with open(recipe_path, "r") as f:
                config = yaml.safe_load(f)

            # DataMixing instance values should be used, not the overrides
            self.assertEqual(config["data_mixing"]["customer_data_percent"], 70)
            self.assertEqual(config["data_mixing"]["nova_code_percent"], 40)
            self.assertEqual(config["data_mixing"]["nova_general_percent"], 60)

    @patch("amzn_nova_forge.recipe.recipe_builder.logger")
    @patch("amzn_nova_forge.util.recipe.get_hub_recipe_metadata")
    @patch("amzn_nova_forge.util.recipe.download_templates_from_s3")
    @patch("amzn_nova_forge.recipe.recipe_builder.Validator")
    @patch("amzn_nova_forge.recipe.recipe_builder.load_file_as_string")
    def test_data_mixing_fields_ignored_in_input_recipe_with_data_mixing_instance(
        self, mock_load_file, mock_validator, mock_download, mock_metadata, mock_logger
    ):
        """Test that data mixing fields in input recipe are ignored when using data_mixing_instance."""
        mock_metadata.return_value = {"recipe_uri": "s3://bucket/recipe"}

        recipe_template = {
            "run": {"name": "{{name}}"},
            "data_mixing": {
                "customer_data_percent": "{{customer_data_percent}}",
                "nova_code_percent": "{{nova_code_percent}}",
                "nova_general_percent": "{{nova_general_percent}}",
            },
        }

        overrides_template = {
            "name": {"default": "", "type": "string"},
            "customer_data_percent": {"default": 50, "type": "int"},
            "nova_code_percent": {"default": 25, "type": "int"},
            "nova_general_percent": {"default": 75, "type": "int"},
        }

        mock_download.return_value = (recipe_template, overrides_template, "image_uri")

        # Input recipe with data mixing fields
        input_recipe = {
            "run": {"name": "test-job"},
            "customer_data_percent": 85,
            "nova_code_percent": 15,
            "nova_general_percent": 85,
        }
        mock_load_file.return_value = yaml.dump(input_recipe)

        # Create a DataMixing instance with configuration
        data_mixing_instance = DataMixing()
        data_mixing_instance.set_config(
            {
                "customer_data_percent": 60,
                "nova_code_percent": 30,
                "nova_general_percent": 70,
            }
        )

        builder = RecipeBuilder(
            region=self.region,
            job_name=self.job_name,
            platform=self.platform,
            model=self.mock_model,
            method=self.method,
            instance_type=self.instance_type,
            instance_count=self.instance_count,
            infra=self.mock_infra,
            output_s3_path=self.output_s3,
            data_s3_path=self.data_s3,
            data_mixing_instance=data_mixing_instance,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, "recipe.yaml")
            input_recipe_path = os.path.join(tmpdir, "input_recipe.yaml")

            recipe_path, *_ = builder.build_and_validate(
                input_recipe_path=input_recipe_path,
                output_recipe_path=output_path,
            )

            # Check that warnings were logged for each data mixing field
            mock_logger.warning.assert_any_call(
                "The following data mixing keys in input recipe will be ignored: customer_data_percent. "
                "Data mixing configuration can only be set using set_datamixing_config()."
            )
            mock_logger.warning.assert_any_call(
                "The following data mixing keys in input recipe will be ignored: nova_code_percent. "
                "Data mixing configuration can only be set using set_datamixing_config()."
            )
            mock_logger.warning.assert_any_call(
                "The following data mixing keys in input recipe will be ignored: nova_general_percent. "
                "Data mixing configuration can only be set using set_datamixing_config()."
            )

            with open(recipe_path, "r") as f:
                config = yaml.safe_load(f)

            # DataMixing instance values should be used, not the input recipe values
            self.assertEqual(config["data_mixing"]["customer_data_percent"], 60)
            self.assertEqual(config["data_mixing"]["nova_code_percent"], 30)
            self.assertEqual(config["data_mixing"]["nova_general_percent"], 70)

    @patch("amzn_nova_forge.util.recipe.get_hub_recipe_metadata")
    @patch("amzn_nova_forge.util.recipe.download_templates_from_s3")
    @patch("amzn_nova_forge.recipe.recipe_builder.Validator")
    def test_data_mixing_fields_used_from_overrides_when_no_data_mixing_instance(
        self, mock_validator, mock_download, mock_metadata
    ):
        """Test that data mixing fields in overrides are used when no data_mixing_instance is provided."""
        mock_metadata.return_value = {"recipe_uri": "s3://bucket/recipe"}

        recipe_template = {
            "run": {"name": "{{name}}"},
            "data_mixing": {
                "customer_data_percent": "{{customer_data_percent}}",
                "nova_code_percent": "{{nova_code_percent}}",
            },
        }

        overrides_template = {
            "name": {"default": "", "type": "string"},
            "customer_data_percent": {"default": 50, "type": "int"},
            "nova_code_percent": {"default": 100, "type": "int"},
        }

        mock_download.return_value = (recipe_template, overrides_template, "image_uri")

        builder = RecipeBuilder(
            region=self.region,
            job_name=self.job_name,
            platform=self.platform,
            model=self.mock_model,
            method=self.method,
            instance_type=self.instance_type,
            instance_count=self.instance_count,
            infra=self.mock_infra,
            output_s3_path=self.output_s3,
            data_s3_path=self.data_s3,
            data_mixing_instance=None,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, "recipe.yaml")

            # Override data mixing fields without providing a data_mixing_instance
            overrides = {
                "customer_data_percent": 80,
                "nova_code_percent": 100,
            }

            recipe_path, *_ = builder.build_and_validate(
                output_recipe_path=output_path,
                overrides=overrides,
            )

            with open(recipe_path, "r") as f:
                config = yaml.safe_load(f)

            # Override values should be used since no data_mixing_instance was provided
            self.assertEqual(config["data_mixing"]["customer_data_percent"], 80)
            self.assertEqual(config["data_mixing"]["nova_code_percent"], 100)

    @patch("amzn_nova_forge.util.recipe.get_hub_recipe_metadata")
    @patch("amzn_nova_forge.util.recipe.download_templates_from_s3")
    @patch("amzn_nova_forge.recipe.recipe_builder.Validator")
    def test_save_steps_override_with_integer(self, mock_validator, mock_download, mock_metadata):
        """Test that save_steps can be overridden with an integer value"""
        mock_metadata.return_value = {"recipe_uri": "s3://bucket/recipe"}

        recipe_template = {
            "run": {"name": "{{name}}"},
            "training_config": {
                "trainer": {
                    "max_steps": "{{max_steps}}",
                    "save_steps": "{{save_steps}}",
                }
            },
        }

        overrides_template = {
            "name": {"default": "test-job", "type": "string"},
            "max_steps": {"default": 1000, "type": "integer"},
            "save_steps": {
                "default": "${oc.select:training_config.trainer.max_steps}",
                "type": "string",
            },
        }

        mock_download.return_value = (recipe_template, overrides_template, "image_uri")

        builder = RecipeBuilder(
            region=self.region,
            job_name=self.job_name,
            platform=self.platform,
            model=self.mock_model,
            method=self.method,
            instance_type=self.instance_type,
            instance_count=self.instance_count,
            infra=self.mock_infra,
            output_s3_path=self.output_s3,
            data_s3_path=self.data_s3,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, "recipe.yaml")

            # Override save_steps with an integer
            overrides = {"save_steps": 100}
            recipe_path, *_ = builder.build_and_validate(
                overrides=overrides,
                output_recipe_path=output_path,
            )

            with open(recipe_path, "r") as f:
                config = yaml.safe_load(f)

            # save_steps should be an integer, not a string
            self.assertEqual(config["training_config"]["trainer"]["save_steps"], 100)
            self.assertIsInstance(config["training_config"]["trainer"]["save_steps"], int)

    @patch("amzn_nova_forge.util.recipe.get_hub_recipe_metadata")
    @patch("amzn_nova_forge.util.recipe.download_templates_from_s3")
    def test_save_steps_override_with_numeric_string(self, mock_download, mock_metadata):
        """Test that save_steps numeric string raises validation error"""
        mock_metadata.return_value = {"recipe_uri": "s3://bucket/recipe"}

        recipe_template = {
            "run": {"name": "{{name}}"},
            "training_config": {
                "trainer": {
                    "max_steps": "{{max_steps}}",
                    "save_steps": "{{save_steps}}",
                }
            },
        }

        overrides_template = {
            "name": {"default": "test-job", "type": "string"},
            "max_steps": {"default": 1000, "type": "integer"},
            "save_steps": {
                "default": "${oc.select:training_config.trainer.max_steps}",
                "type": "string",
            },
        }

        mock_download.return_value = (recipe_template, overrides_template, "image_uri")

        # Create a temporary recipe file for RFT multiturn infra
        with tempfile.TemporaryDirectory() as tmpdir:
            temp_recipe_path = os.path.join(tmpdir, "temp_recipe.yaml")
            with open(temp_recipe_path, "w") as f:
                yaml.dump(recipe_template, f)

            # Mock RFT multiturn infra with required methods
            mock_rft_infra = Mock()
            mock_rft_infra.get_recipe_path.return_value = temp_recipe_path
            mock_rft_infra.get_recipe_overrides.return_value = {}

            # Mock infra with cluster_name for SMHP
            mock_infra = Mock(spec=RuntimeManager)
            mock_infra.instance_type = self.instance_type
            mock_infra.instance_count = self.instance_count
            mock_infra.cluster_name = "test-cluster"
            mock_infra.region = self.region

            builder = RecipeBuilder(
                region=self.region,
                job_name=self.job_name,
                platform=Platform.SMHP,  # RFT multiturn requires HyperPod
                model=Model.NOVA_LITE_2,  # Use actual Model enum for RFT multiturn
                method=TrainingMethod.RFT_MULTITURN_LORA,  # Use RFT multiturn method
                instance_type=self.instance_type,
                instance_count=self.instance_count,
                infra=mock_infra,
                output_s3_path=self.output_s3,
                data_s3_path=self.data_s3,
                rft_multiturn_infra=mock_rft_infra,  # Required for RFT multiturn
            )

            output_path = os.path.join(tmpdir, "recipe.yaml")

            # Override save_steps with a numeric string - should raise error
            overrides = {"save_steps": "50"}

            # Should raise error - numeric strings are not valid, must be integers
            with self.assertRaises(ValueError) as context:
                builder.build_and_validate(
                    overrides=overrides,
                    output_recipe_path=output_path,
                    validation_config=ValidationConfig(
                        infra=False,
                        iam=False,
                    ),  # Skip AWS validation
                )

            self.assertIn("must be an integer, not a string", str(context.exception))
            self.assertIn("Use 50 instead of", str(context.exception))

    @patch("amzn_nova_forge.util.recipe.get_hub_recipe_metadata")
    @patch("amzn_nova_forge.util.recipe.download_templates_from_s3")
    @patch("amzn_nova_forge.recipe.recipe_builder.Validator")
    def test_save_steps_omegaconf_interpolation_preserved(
        self, mock_validator, mock_download, mock_metadata
    ):
        """Test that OmegaConf interpolation strings are preserved for save_steps"""
        mock_metadata.return_value = {"recipe_uri": "s3://bucket/recipe"}

        recipe_template = {
            "run": {"name": "{{name}}"},
            "training_config": {
                "trainer": {
                    "max_steps": "{{max_steps}}",
                    "save_steps": "{{save_steps}}",
                }
            },
        }

        overrides_template = {
            "name": {"default": "test-job", "type": "string"},
            "max_steps": {"default": 1000, "type": "integer"},
            "save_steps": {
                "default": "${oc.select:training_config.trainer.max_steps}",
                "type": "string",
            },
        }

        mock_download.return_value = (recipe_template, overrides_template, "image_uri")

        builder = RecipeBuilder(
            region=self.region,
            job_name=self.job_name,
            platform=self.platform,
            model=self.mock_model,
            method=self.method,
            instance_type=self.instance_type,
            instance_count=self.instance_count,
            infra=self.mock_infra,
            output_s3_path=self.output_s3,
            data_s3_path=self.data_s3,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, "recipe.yaml")

            # Don't override save_steps, use default OmegaConf interpolation
            overrides = {}
            recipe_path, *_ = builder.build_and_validate(
                overrides=overrides,
                output_recipe_path=output_path,
            )

            with open(recipe_path, "r") as f:
                config = yaml.safe_load(f)

            # save_steps should preserve the OmegaConf interpolation string
            self.assertEqual(
                config["training_config"]["trainer"]["save_steps"],
                "${oc.select:training_config.trainer.max_steps}",
            )
            self.assertIsInstance(config["training_config"]["trainer"]["save_steps"], str)

    @patch("amzn_nova_forge.util.recipe.get_hub_recipe_metadata")
    @patch("amzn_nova_forge.recipe.recipe_builder.load_file_as_string")
    @patch("amzn_nova_forge.util.recipe.download_templates_from_s3")
    def test_save_steps_input_recipe_with_numeric_string(
        self, mock_download, mock_load_file, mock_metadata
    ):
        """Test that save_steps numeric string from input recipe raises validation error"""
        mock_metadata.return_value = {"recipe_uri": "s3://bucket/recipe"}

        recipe_template = {
            "run": {"name": "{{name}}"},
            "training_config": {
                "trainer": {
                    "max_steps": "{{max_steps}}",
                    "save_steps": "{{save_steps}}",
                }
            },
        }

        overrides_template = {
            "name": {"default": "test-job", "type": "string"},
            "max_steps": {"default": 1000, "type": "integer"},
            "save_steps": {
                "default": "${oc.select:training_config.trainer.max_steps}",
                "type": "string",
            },
        }

        mock_download.return_value = (recipe_template, overrides_template, "image_uri")

        # Input recipe with save_steps as numeric string
        input_recipe_content = """
        run:
          name: my-job
        training_config:
          trainer:
            max_steps: 200
            save_steps: "75"
        """
        mock_load_file.return_value = input_recipe_content

        # Create a temporary recipe file for RFT multiturn infra
        with tempfile.TemporaryDirectory() as tmpdir:
            temp_recipe_path = os.path.join(tmpdir, "temp_recipe.yaml")
            with open(temp_recipe_path, "w") as f:
                yaml.dump(recipe_template, f)

            # Mock RFT multiturn infra with required methods
            mock_rft_infra = Mock()
            mock_rft_infra.get_recipe_path.return_value = temp_recipe_path
            mock_rft_infra.get_recipe_overrides.return_value = {}

            # Mock infra with cluster_name for SMHP
            mock_infra = Mock(spec=RuntimeManager)
            mock_infra.instance_type = self.instance_type
            mock_infra.instance_count = self.instance_count
            mock_infra.cluster_name = "test-cluster"
            mock_infra.region = self.region

            builder = RecipeBuilder(
                region=self.region,
                job_name=self.job_name,
                platform=Platform.SMHP,  # RFT multiturn requires HyperPod
                model=Model.NOVA_LITE_2,  # Use actual Model enum for RFT multiturn
                method=TrainingMethod.RFT_MULTITURN_FULL,  # Use RFT multiturn method
                instance_type=self.instance_type,
                instance_count=self.instance_count,
                infra=mock_infra,
                output_s3_path=self.output_s3,
                data_s3_path=self.data_s3,
                rft_multiturn_infra=mock_rft_infra,  # Required for RFT multiturn
            )

            output_path = os.path.join(tmpdir, "recipe.yaml")
            input_recipe_path = os.path.join(tmpdir, "input_recipe.yaml")

            # Should raise error - numeric strings are not valid, must be integers
            with self.assertRaises(ValueError) as context:
                builder.build_and_validate(
                    input_recipe_path=input_recipe_path,
                    output_recipe_path=output_path,
                    validation_config=ValidationConfig(
                        infra=False,
                        iam=False,
                    ),  # Skip AWS validation
                )

            self.assertIn("must be an integer, not a string", str(context.exception))

    @patch("amzn_nova_forge.util.recipe.get_hub_recipe_metadata")
    @patch("amzn_nova_forge.util.recipe.download_templates_from_s3")
    def test_save_steps_override_does_not_affect_other_params(self, mock_download, mock_metadata):
        """Test that save_steps validation doesn't affect other string parameters"""
        mock_metadata.return_value = {"recipe_uri": "s3://bucket/recipe"}

        recipe_template = {
            "run": {"name": "{{name}}"},
            "training_config": {
                "trainer": {
                    "max_steps": "{{max_steps}}",
                    "save_steps": "{{save_steps}}",
                },
                "custom_param": "{{custom_param}}",
            },
        }

        overrides_template = {
            "name": {"default": "test-job", "type": "string"},
            "max_steps": {"default": 1000, "type": "integer"},
            "save_steps": {
                "default": "${oc.select:training_config.trainer.max_steps}",
                "type": "string",
            },
            "custom_param": {"default": "default_value", "type": "string"},
        }

        mock_download.return_value = (recipe_template, overrides_template, "image_uri")

        # Create a temporary recipe file for RFT multiturn infra
        with tempfile.TemporaryDirectory() as tmpdir:
            temp_recipe_path = os.path.join(tmpdir, "temp_recipe.yaml")
            with open(temp_recipe_path, "w") as f:
                yaml.dump(recipe_template, f)

            # Mock RFT multiturn infra with required methods
            mock_rft_infra = Mock()
            mock_rft_infra.get_recipe_path.return_value = temp_recipe_path
            mock_rft_infra.get_recipe_overrides.return_value = {}

            # Mock infra with cluster_name for SMHP
            mock_infra = Mock(spec=RuntimeManager)
            mock_infra.instance_type = self.instance_type
            mock_infra.instance_count = self.instance_count
            mock_infra.cluster_name = "test-cluster"
            mock_infra.region = self.region

            builder = RecipeBuilder(
                region=self.region,
                job_name=self.job_name,
                platform=Platform.SMHP,  # RFT multiturn requires HyperPod
                model=Model.NOVA_LITE_2,  # Use actual Model enum for RFT multiturn
                method=TrainingMethod.RFT_MULTITURN_LORA,  # Use RFT multiturn method
                instance_type=self.instance_type,
                instance_count=self.instance_count,
                infra=mock_infra,
                output_s3_path=self.output_s3,
                data_s3_path=self.data_s3,
                rft_multiturn_infra=mock_rft_infra,  # Required for RFT multiturn
            )

            output_path = os.path.join(tmpdir, "recipe.yaml")

            # Override save_steps with integer and custom_param with numeric string
            # save_steps must be integer, but custom_param can be a numeric string
            overrides = {"save_steps": 30, "custom_param": "100"}

            # Should NOT raise error - save_steps is integer, custom_param can be string
            recipe_path, *_ = builder.build_and_validate(
                overrides=overrides,
                output_recipe_path=output_path,
                validation_config=ValidationConfig(infra=False, iam=False),  # Skip AWS validation
            )

            # Verify the recipe was created successfully
            self.assertTrue(os.path.exists(recipe_path))

    @patch("amzn_nova_forge.recipe.recipe_builder.logger")
    @patch("amzn_nova_forge.util.recipe.get_hub_recipe_metadata")
    @patch("amzn_nova_forge.util.recipe.download_templates_from_s3")
    @patch("amzn_nova_forge.recipe.recipe_builder.Validator")
    def test_data_s3_path_override_ignored_when_multimodal(
        self, mock_validator, mock_download, mock_metadata, mock_logger
    ):
        """data_s3_path in overrides is ignored when is_multimodal=True (used for datamix recipe selection)."""
        mock_metadata.return_value = {"recipe_uri": "s3://bucket/recipe"}

        recipe_template = {
            "run": {"name": "{{name}}"},
            "data": {"data_s3_path": "{{data_s3_path}}"},
        }
        overrides_template = {
            "name": {"default": "", "type": "string"},
            "data_s3_path": {"default": "s3://bucket/original", "type": "string"},
        }
        mock_download.return_value = (recipe_template, overrides_template, "image_uri")

        builder = RecipeBuilder(
            region=self.region,
            job_name=self.job_name,
            platform=self.platform,
            model=self.mock_model,
            method=self.method,
            instance_type=self.instance_type,
            instance_count=self.instance_count,
            infra=self.mock_infra,
            output_s3_path=self.output_s3,
            data_s3_path="s3://bucket/original",
            is_multimodal=True,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, "recipe.yaml")
            builder.build_and_validate(
                overrides={"data_s3_path": "s3://bucket/override"},
                output_recipe_path=output_path,
            )

            mock_logger.warning.assert_any_call(
                "Override for 'data_s3_path' will be ignored. If you wish to pass 'data_s3_path', please update your NovaModelCustomizer object."
            )

    @patch("amzn_nova_forge.recipe.recipe_builder.logger")
    @patch("amzn_nova_forge.util.recipe.get_hub_recipe_metadata")
    @patch("amzn_nova_forge.util.recipe.download_templates_from_s3")
    @patch("amzn_nova_forge.recipe.recipe_builder.Validator")
    @patch("amzn_nova_forge.recipe.recipe_builder.load_file_as_string")
    def test_data_s3_path_in_input_recipe_ignored_when_multimodal(
        self, mock_load_file, mock_validator, mock_download, mock_metadata, mock_logger
    ):
        """data_s3_path in input recipe is ignored when is_multimodal=True."""
        mock_metadata.return_value = {"recipe_uri": "s3://bucket/recipe"}

        recipe_template = {
            "run": {"name": "{{name}}"},
            "data": {"data_s3_path": "{{data_s3_path}}"},
        }
        overrides_template = {
            "name": {"default": "", "type": "string"},
            "data_s3_path": {"default": "s3://bucket/original", "type": "string"},
        }
        mock_download.return_value = (recipe_template, overrides_template, "image_uri")

        input_recipe = {
            "run": {"name": "test"},
            "data": {"data_s3_path": "s3://bucket/from-recipe"},
        }
        mock_load_file.return_value = yaml.dump(input_recipe)

        builder = RecipeBuilder(
            region=self.region,
            job_name=self.job_name,
            platform=self.platform,
            model=self.mock_model,
            method=self.method,
            instance_type=self.instance_type,
            instance_count=self.instance_count,
            infra=self.mock_infra,
            output_s3_path=self.output_s3,
            data_s3_path="s3://bucket/original",
            is_multimodal=True,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, "recipe.yaml")
            input_recipe_path = os.path.join(tmpdir, "input_recipe.yaml")

            builder.build_and_validate(
                input_recipe_path=input_recipe_path,
                output_recipe_path=output_path,
            )

            mock_logger.warning.assert_any_call(
                "Override for 'data_s3_path' will be ignored. If you wish to pass 'data_s3_path', please update your NovaModelCustomizer object."
            )

    @patch("amzn_nova_forge.recipe.recipe_builder.logger")
    @patch("amzn_nova_forge.util.recipe.get_hub_recipe_metadata")
    @patch("amzn_nova_forge.util.recipe.download_templates_from_s3")
    @patch("amzn_nova_forge.recipe.recipe_builder.Validator")
    def test_data_s3_path_override_allowed_when_not_multimodal(
        self, mock_validator, mock_download, mock_metadata, mock_logger
    ):
        """data_s3_path in overrides is applied normally when is_multimodal=False."""
        mock_metadata.return_value = {"recipe_uri": "s3://bucket/recipe"}

        recipe_template = {
            "run": {"name": "{{name}}"},
            "data": {"data_s3_path": "{{data_s3_path}}"},
        }
        overrides_template = {
            "name": {"default": "", "type": "string"},
            "data_s3_path": {"default": "s3://bucket/original", "type": "string"},
        }
        mock_download.return_value = (recipe_template, overrides_template, "image_uri")

        builder = RecipeBuilder(
            region=self.region,
            job_name=self.job_name,
            platform=self.platform,
            model=self.mock_model,
            method=self.method,
            instance_type=self.instance_type,
            instance_count=self.instance_count,
            infra=self.mock_infra,
            output_s3_path=self.output_s3,
            data_s3_path="s3://bucket/original",
            is_multimodal=False,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, "recipe.yaml")
            builder.build_and_validate(
                overrides={"data_s3_path": "s3://bucket/override"},
                output_recipe_path=output_path,
            )

            # Warning should NOT be emitted
            warning_calls = [str(c) for c in mock_logger.warning.call_args_list]
            self.assertFalse(
                any("data_s3_path" in c for c in warning_calls),
                "data_s3_path warning should not be emitted when is_multimodal=False",
            )

            with open(output_path, "r") as f:
                config = yaml.safe_load(f)
            self.assertEqual(config["data"]["data_s3_path"], "s3://bucket/override")

    @patch("amzn_nova_forge.util.recipe.get_hub_recipe_metadata")
    @patch("amzn_nova_forge.util.recipe.download_templates_from_s3")
    @patch("amzn_nova_forge.recipe.recipe_builder.Validator")
    def test_build_and_validate_batch_sample_tracing_non_bedrock(
        self, mock_validator, mock_download, mock_metadata
    ):
        """enable_batch_sample_tracing=True on non-Bedrock platform injects flag into training_config."""
        mock_metadata.return_value = {"recipe_uri": "s3://bucket/recipe"}

        recipe_template = {
            "run": {"name": "{{name}}"},
        }

        overrides_template = {
            "name": {"default": "", "type": "string"},
        }

        mock_download.return_value = (recipe_template, overrides_template, "image_uri")

        builder = RecipeBuilder(
            region=self.region,
            job_name=self.job_name,
            platform=Platform.SMTJ,
            model=self.mock_model,
            method=self.method,
            instance_type=self.instance_type,
            instance_count=self.instance_count,
            infra=self.mock_infra,
            output_s3_path=self.output_s3,
            data_s3_path=self.data_s3,
            enable_batch_sample_tracing=True,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, "recipe.yaml")
            builder.build_and_validate(output_recipe_path=output_path)

            with open(output_path, "r") as f:
                config = yaml.safe_load(f)

            self.assertIn("training_config", config)
            self.assertTrue(config["training_config"]["enable_batch_sample_tracing"])

    def test_sft_lora_smtj_stores_validation_path(self):
        validation_data_s3 = "s3://bucket/validation-data"
        builder = RecipeBuilder(
            region=self.region,
            job_name=self.job_name,
            platform=Platform.SMTJ,
            model=self.mock_model,
            method=TrainingMethod.SFT_LORA,
            instance_type=self.instance_type,
            instance_count=self.instance_count,
            infra=self.mock_infra,
            output_s3_path=self.output_s3,
            data_s3_path=self.data_s3,
            validation_data_s3_path=validation_data_s3,
        )

        self.assertEqual(builder.validation_data_s3_path, validation_data_s3)

    def test_sft_full_smtj_stores_validation_path(self):
        validation_data_s3 = "s3://bucket/validation-data"
        builder = RecipeBuilder(
            region=self.region,
            job_name=self.job_name,
            platform=Platform.SMTJ,
            model=self.mock_model,
            method=TrainingMethod.SFT_FULL,
            instance_type=self.instance_type,
            instance_count=self.instance_count,
            infra=self.mock_infra,
            output_s3_path=self.output_s3,
            data_s3_path=self.data_s3,
            validation_data_s3_path=validation_data_s3,
        )

        self.assertEqual(builder.validation_data_s3_path, validation_data_s3)

    def test_sft_lora_smtj_serverless_stores_validation_path(self):
        validation_data_s3 = "s3://bucket/validation-data"
        builder = RecipeBuilder(
            region=self.region,
            job_name=self.job_name,
            platform=Platform.SMTJServerless,
            model=self.mock_model,
            method=TrainingMethod.SFT_LORA,
            instance_type=self.instance_type,
            instance_count=self.instance_count,
            infra=self.mock_infra,
            output_s3_path=self.output_s3,
            data_s3_path=self.data_s3,
            validation_data_s3_path=validation_data_s3,
        )

        self.assertEqual(builder.validation_data_s3_path, validation_data_s3)

    def test_sft_lora_smhp_stores_validation_path(self):
        validation_data_s3 = "s3://bucket/validation-data"
        builder = RecipeBuilder(
            region=self.region,
            job_name=self.job_name,
            platform=Platform.SMHP,
            model=self.mock_model,
            method=TrainingMethod.SFT_LORA,
            instance_type=self.instance_type,
            instance_count=self.instance_count,
            infra=self.mock_infra,
            output_s3_path=self.output_s3,
            data_s3_path=self.data_s3,
            validation_data_s3_path=validation_data_s3,
        )

        self.assertEqual(builder.validation_data_s3_path, validation_data_s3)

    def test_sft_no_validation_data_attribute_is_none(self):
        builder = RecipeBuilder(
            region=self.region,
            job_name=self.job_name,
            platform=Platform.SMTJ,
            model=self.mock_model,
            method=TrainingMethod.SFT_LORA,
            instance_type=self.instance_type,
            instance_count=self.instance_count,
            infra=self.mock_infra,
            output_s3_path=self.output_s3,
            data_s3_path=self.data_s3,
            validation_data_s3_path=None,
        )

        self.assertIsNone(builder.validation_data_s3_path)

    def test_validation_data_s3_path_initialized_to_none(self):
        builder = RecipeBuilder(
            region=self.region,
            job_name=self.job_name,
            platform=Platform.SMTJ,
            model=self.mock_model,
            method=TrainingMethod.SFT_LORA,
            instance_type=self.instance_type,
            instance_count=self.instance_count,
            infra=self.mock_infra,
            output_s3_path=self.output_s3,
            data_s3_path=self.data_s3,
        )

        self.assertIsNone(builder.validation_data_s3_path)

    @patch("amzn_nova_forge.util.recipe.get_hub_recipe_metadata")
    @patch("amzn_nova_forge.util.recipe.download_templates_from_s3")
    @patch("amzn_nova_forge.recipe.recipe_builder.Validator")
    def test_build_and_validate_batch_sample_tracing_default_false(
        self, mock_validator, mock_download, mock_metadata
    ):
        """enable_batch_sample_tracing defaults to False — flag absent from recipe."""
        mock_metadata.return_value = {"recipe_uri": "s3://bucket/recipe"}

        recipe_template = {
            "run": {"name": "{{name}}"},
        }

        overrides_template = {
            "name": {"default": "", "type": "string"},
        }

        mock_download.return_value = (recipe_template, overrides_template, "image_uri")

        builder = RecipeBuilder(
            region=self.region,
            job_name=self.job_name,
            platform=self.platform,
            model=self.mock_model,
            method=self.method,
            instance_type=self.instance_type,
            instance_count=self.instance_count,
            infra=self.mock_infra,
            output_s3_path=self.output_s3,
            data_s3_path=self.data_s3,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, "recipe.yaml")
            builder.build_and_validate(output_recipe_path=output_path)

            with open(output_path, "r") as f:
                config = yaml.safe_load(f)

            # training_config should either not exist or not contain the flag
            if "training_config" in config:
                self.assertNotIn("enable_batch_sample_tracing", config["training_config"])

    @patch("amzn_nova_forge.util.recipe.get_hub_recipe_metadata")
    @patch("amzn_nova_forge.util.recipe.download_templates_from_s3")
    @patch("amzn_nova_forge.recipe.recipe_builder.Validator")
    def test_sft_lora_smhp_injects_val_check_interval(
        self, mock_validator, mock_download, mock_metadata
    ):
        mock_metadata.return_value = {"recipe_uri": "s3://bucket/recipe"}

        recipe_template = {
            "run": {
                "name": "{{name}}",
                "replicas": 1,
                "validation_data_s3_path": "{{validation_data_s3_path}}",
                "val_check_interval": "{{val_check_interval}}",
            }
        }

        overrides_template = {
            "name": {"default": "", "type": "string"},
            "validation_data_s3_path": {"default": "", "type": "string"},
            "val_check_interval": {"default": 1000, "type": "integer"},
        }

        mock_download.return_value = (recipe_template, overrides_template, "image_uri")

        builder = RecipeBuilder(
            region=self.region,
            job_name=self.job_name,
            platform=Platform.SMHP,
            model=self.mock_model,
            method=TrainingMethod.SFT_LORA,
            instance_type=self.instance_type,
            instance_count=self.instance_count,
            infra=self.mock_infra,
            output_s3_path=self.output_s3,
            data_s3_path=self.data_s3,
            validation_data_s3_path="s3://bucket/validation-data",
            val_check_interval=500,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, "recipe.yaml")
            recipe_path, *_ = builder.build_and_validate(output_recipe_path=output_path)

            with open(recipe_path, "r") as f:
                config = yaml.safe_load(f)

            self.assertEqual(config["run"]["val_check_interval"], 500)
            self.assertEqual(
                config["run"]["validation_data_s3_path"], "s3://bucket/validation-data"
            )

    @patch("amzn_nova_forge.util.recipe.get_hub_recipe_metadata")
    @patch("amzn_nova_forge.util.recipe.download_templates_from_s3")
    @patch("amzn_nova_forge.recipe.recipe_builder.Validator")
    def test_val_check_interval_not_injected_when_none(
        self, mock_validator, mock_download, mock_metadata
    ):
        mock_metadata.return_value = {"recipe_uri": "s3://bucket/recipe"}

        recipe_template = {
            "run": {
                "name": "{{name}}",
                "replicas": 1,
                "val_check_interval": "{{val_check_interval}}",
            }
        }

        overrides_template = {
            "name": {"default": "", "type": "string"},
            "val_check_interval": {"default": 1000, "type": "integer"},
        }

        mock_download.return_value = (recipe_template, overrides_template, "image_uri")

        builder = RecipeBuilder(
            region=self.region,
            job_name=self.job_name,
            platform=Platform.SMHP,
            model=self.mock_model,
            method=TrainingMethod.SFT_LORA,
            instance_type=self.instance_type,
            instance_count=self.instance_count,
            infra=self.mock_infra,
            output_s3_path=self.output_s3,
            data_s3_path=self.data_s3,
            val_check_interval=None,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, "recipe.yaml")
            recipe_path, *_ = builder.build_and_validate(output_recipe_path=output_path)

            with open(recipe_path, "r") as f:
                config = yaml.safe_load(f)

            # Should use the template default (1000), not an injected value
            self.assertEqual(config["run"]["val_check_interval"], 1000)


class TestRecipeBuilderMlflowPlaceholder(unittest.TestCase):
    """Tests that mlflow placeholders are resolved even when no MLflowMonitor is provided."""

    def setUp(self):
        self.region = "us-east-1"
        self.job_name = "test-job"
        self.platform = Platform.SMTJ
        self.method = TrainingMethod.SFT_LORA
        self.instance_type = "ml.g5.12xlarge"
        self.instance_count = 1
        self.data_s3 = "s3://bucket/data"
        self.output_s3 = "s3://bucket/output"

        self.mock_model = Mock(spec=Model)
        self.mock_model.name = "nova-micro"
        self.mock_model.value = "nova_micro"
        self.mock_model.version = Version.ONE
        self.mock_model.model_type = "test-model"
        self.mock_model.model_path = "models/test"

        self.mock_infra = Mock(spec=RuntimeManager)
        self.mock_infra.instance_type = self.instance_type
        self.mock_infra.instance_count = self.instance_count

    @patch("amzn_nova_forge.util.recipe.get_hub_recipe_metadata")
    @patch("amzn_nova_forge.util.recipe.download_templates_from_s3")
    @patch("amzn_nova_forge.recipe.recipe_builder.Validator")
    def test_mlflow_placeholders_resolved_to_empty_when_no_monitor(
        self, mock_validator, mock_download, mock_metadata
    ):
        """When no mlflow_monitor is provided, {{mlflow_*}} placeholders must resolve to ''."""
        mock_metadata.return_value = {"recipe_uri": "s3://bucket/recipe"}

        recipe_template = {
            "run": {
                "name": "{{name}}",
                "mlflow_tracking_uri": "{{mlflow_tracking_uri}}",
                "mlflow_experiment_name": "{{mlflow_experiment_name}}",
                "mlflow_run_name": "{{mlflow_run_name}}",
            }
        }
        overrides_template = {"name": {"default": "", "type": "string"}}
        mock_download.return_value = (recipe_template, overrides_template, "image_uri")

        builder = RecipeBuilder(
            region=self.region,
            job_name=self.job_name,
            platform=self.platform,
            model=self.mock_model,
            method=self.method,
            instance_type=self.instance_type,
            instance_count=self.instance_count,
            infra=self.mock_infra,
            output_s3_path=self.output_s3,
            data_s3_path=self.data_s3,
            mlflow_monitor=None,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, "recipe.yaml")
            recipe_path, *_ = builder.build_and_validate(output_recipe_path=output_path)

            with open(recipe_path, "r") as f:
                config = yaml.safe_load(f)

        # Placeholders must be resolved to empty strings, not left as "{{...}}"
        self.assertEqual(config["run"]["mlflow_tracking_uri"], "")
        self.assertEqual(config["run"]["mlflow_experiment_name"], "")
        self.assertEqual(config["run"]["mlflow_run_name"], "")

    @patch("amzn_nova_forge.util.recipe.get_hub_recipe_metadata")
    @patch("amzn_nova_forge.util.recipe.download_templates_from_s3")
    @patch("amzn_nova_forge.recipe.recipe_builder.Validator")
    def test_mlflow_placeholders_not_raw_when_no_monitor(
        self, mock_validator, mock_download, mock_metadata
    ):
        """Ensure raw {{...}} strings are never present in the built recipe."""
        mock_metadata.return_value = {"recipe_uri": "s3://bucket/recipe"}

        recipe_template = {
            "run": {
                "name": "{{name}}",
                "mlflow_tracking_uri": "{{mlflow_tracking_uri}}",
            }
        }
        overrides_template = {"name": {"default": "", "type": "string"}}
        mock_download.return_value = (recipe_template, overrides_template, "image_uri")

        builder = RecipeBuilder(
            region=self.region,
            job_name=self.job_name,
            platform=self.platform,
            model=self.mock_model,
            method=self.method,
            instance_type=self.instance_type,
            instance_count=self.instance_count,
            infra=self.mock_infra,
            output_s3_path=self.output_s3,
            data_s3_path=self.data_s3,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, "recipe.yaml")
            recipe_path, *_ = builder.build_and_validate(output_recipe_path=output_path)

            with open(recipe_path, "r") as f:
                raw = f.read()

        self.assertNotIn("{{mlflow_tracking_uri}}", raw)


class TestGetOverridableConfig(unittest.TestCase):
    def setUp(self):
        self.region = "us-east-1"
        self.platform = Platform.SMTJ
        self.method = TrainingMethod.SFT_LORA

        self.mock_model = Mock(spec=Model)
        self.mock_model.name = "nova-micro"
        self.mock_model.value = "nova_micro"
        self.mock_model.version = Version.ONE
        self.mock_model.model_type = "test-model"
        self.mock_model.model_path = "models/test"

        self.mock_infra = Mock(spec=RuntimeManager)
        self.mock_infra.instance_type = "ml.g5.12xlarge"
        self.mock_infra.instance_count = 1

        self.overrides_template = {
            "lr": {
                "type": "float",
                "default": 5e-6,
                "min": 1e-6,
                "max": 1e-4,
                "description": "Learning rate",
            },
            "max_epochs": {
                "type": "integer",
                "default": 2,
                "min": 1,
                "max": 5,
                "description": "Max epochs",
            },
            "reasoning_effort": {
                "type": "string",
                "default": "medium",
                "enum": ["low", "medium", "high"],
            },
            "model_type": {"type": "string", "default": "test-model"},
            "model_name_or_path": {"type": "string", "default": "models/test"},
            "peft_scheme": {"type": "string", "default": "lora"},
            "name": {"type": "string", "default": "job"},
            "data_s3_path": {"type": "string", "default": "s3://bucket/data"},
            "output_s3_path": {"type": "string", "default": "s3://bucket/output"},
            "task": {"type": "string", "default": "sft"},
        }
        self.recipe_template = {"run": {"lr": "{{lr}}", "max_epochs": "{{max_epochs}}"}}

        self.load_patcher = patch("amzn_nova_forge.recipe.recipe_builder.load_recipe_templates")
        self.mock_load = self.load_patcher.start()
        self.mock_load.return_value = (
            None,
            self.recipe_template,
            self.overrides_template,
            "test-image-uri",
        )
        self.addCleanup(self.load_patcher.stop)

    def _make_builder(self, **kwargs):
        defaults = dict(
            region=self.region,
            job_name="__config_preview__",
            platform=self.platform,
            model=self.mock_model,
            method=self.method,
            instance_type="ml.g5.12xlarge",
            instance_count=1,
            infra=self.mock_infra,
            output_s3_path="s3://bucket/output",
            data_s3_path="s3://bucket/data",
        )
        defaults.update(kwargs)
        return RecipeBuilder(**defaults)

    def test_returns_recipe_config(self):
        from amzn_nova_forge.core.types import RecipeConfig

        builder = self._make_builder()
        config = builder.get_overridable_config()
        self.assertIsInstance(config, RecipeConfig)
        self.assertEqual(config.model, self.mock_model)
        self.assertEqual(config.method, self.method)
        self.assertEqual(config.platform, self.platform)

    def test_excludes_non_overridable_keys(self):
        builder = self._make_builder()
        config = builder.get_overridable_config()
        param_names = {p.name for p in config.parameters}
        for excluded in (
            "model_type",
            "model_name_or_path",
            "peft_scheme",
            "name",
            "data_s3_path",
            "output_s3_path",
            "task",
        ):
            self.assertNotIn(excluded, param_names)

    def test_includes_overridable_keys(self):
        builder = self._make_builder()
        config = builder.get_overridable_config()
        param_names = {p.name for p in config.parameters}
        self.assertIn("lr", param_names)
        self.assertIn("max_epochs", param_names)
        self.assertIn("reasoning_effort", param_names)

    def test_preserves_constraints(self):
        builder = self._make_builder()
        config = builder.get_overridable_config()
        lr_param = next(p for p in config.parameters if p.name == "lr")
        self.assertEqual(lr_param.min, 1e-6)
        self.assertEqual(lr_param.max, 1e-4)
        self.assertEqual(lr_param.description, "Learning rate")

    def test_preserves_enum(self):
        builder = self._make_builder()
        config = builder.get_overridable_config()
        effort_param = next(p for p in config.parameters if p.name == "reasoning_effort")
        self.assertEqual(effort_param.enum, ("low", "medium", "high"))

    def test_with_overrides_merges_into_defaults(self):
        builder = self._make_builder()
        config = builder.get_overridable_config(overrides={"lr": 1e-5})
        lr_param = next(p for p in config.parameters if p.name == "lr")
        self.assertEqual(lr_param.default, 1e-5)

    def test_skips_non_dict_metadata(self):
        self.overrides_template["stray_string"] = "not a dict"
        self.overrides_template["stray_int"] = 42
        builder = self._make_builder()
        config = builder.get_overridable_config()
        param_names = {p.name for p in config.parameters}
        self.assertNotIn("stray_string", param_names)
        self.assertNotIn("stray_int", param_names)

    def test_config_is_frozen(self):
        builder = self._make_builder()
        config = builder.get_overridable_config()
        with self.assertRaises(AttributeError):
            config.model = self.mock_model


class TestReasoningEffortNoneOverride(unittest.TestCase):
    """Tests for reasoning_effort=None override handling."""

    def setUp(self):
        self.region = "us-east-1"
        self.job_name = "test-reasoning-none"
        self.platform = Platform.SMTJ
        self.instance_type = "ml.p5.48xlarge"
        self.instance_count = 4
        self.data_s3 = "s3://bucket/rft-data.jsonl"
        self.output_s3 = "s3://bucket/output"

        self.mock_model = Mock(spec=Model)
        self.mock_model.name = "NOVA_LITE_2"
        self.mock_model.value = "nova_lite_2"
        self.mock_model.version = Version.TWO
        self.mock_model.model_type = "nova_lite_2"
        self.mock_model.model_path = "models/nova-lite-2"

        self.mock_infra = Mock(spec=RuntimeManager)
        self.mock_infra.instance_type = self.instance_type
        self.mock_infra.instance_count = self.instance_count
        self.mock_infra.platform = Platform.SMTJ

    @patch("amzn_nova_forge.util.recipe._get_smhp_replicas_enum", return_value=None)
    @patch("amzn_nova_forge.util.recipe.get_hub_recipe_metadata")
    @patch("amzn_nova_forge.util.recipe.download_templates_from_s3")
    def test_reasoning_effort_none_passes_validation(
        self, mock_download, mock_metadata, _mock_replicas
    ):
        """reasoning_effort=None override must not raise ValueError during build_and_validate."""
        mock_metadata.return_value = {"recipe_uri": "s3://bucket/recipe"}

        recipe_template = {
            "run": {
                "name": "{{name}}",
                "replicas": 1,
                "lambda_arn": "{{reward_lambda_arn}}",
            },
            "training_config": {
                "max_steps": "{{max_steps}}",
                "lr": "{{lr}}",
                "reasoning_effort": "medium",
            },
        }

        overrides_template = {
            "name": {"default": "", "type": "string", "required": True},
            "reward_lambda_arn": {"default": "", "type": "string", "required": True},
            "max_steps": {"default": 100, "type": "integer", "min": 1, "max": 10000},
            "lr": {"default": 5e-6, "type": "float"},
            "reasoning_effort": {
                "type": "string",
                "enum": ["low", "medium", "high"],
                "default": "medium",
            },
        }

        mock_download.return_value = (recipe_template, overrides_template, "image_uri")

        rft_lambda = "arn:aws:lambda:us-east-1:123456789012:function:reward"
        builder = RecipeBuilder(
            region=self.region,
            job_name=self.job_name,
            platform=self.platform,
            model=self.mock_model,
            method=TrainingMethod.RFT_LORA,
            instance_type=self.instance_type,
            instance_count=self.instance_count,
            infra=self.mock_infra,
            output_s3_path=self.output_s3,
            data_s3_path=self.data_s3,
            rft_lambda_arn=rft_lambda,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, "recipe.yaml")
            recipe_path, *_ = builder.build_and_validate(
                output_recipe_path=output_path,
                overrides={"reasoning_effort": None},
                validation_config=ValidationConfig(iam=False, infra=False, recipe=True),
            )

            with open(recipe_path, "r") as f:
                recipe = yaml.safe_load(f)

            assert recipe["training_config"]["reasoning_effort"] is None

    @patch("amzn_nova_forge.util.recipe._get_smhp_replicas_enum", return_value=None)
    @patch("amzn_nova_forge.util.recipe.get_hub_recipe_metadata")
    @patch("amzn_nova_forge.util.recipe.download_templates_from_s3")
    def test_reasoning_effort_none_placeholder_path_still_validates(
        self, mock_download, mock_metadata, _mock_replicas
    ):
        """reasoning_effort=None on a {{placeholder}}-same-key template still raises.

        The fix only targets the literal-value branch (line 575-585). When the
        template uses {{reasoning_effort}} as a placeholder, the overrides_template
        retains its original required: True, so None is correctly rejected.
        """
        mock_metadata.return_value = {"recipe_uri": "s3://bucket/recipe"}

        recipe_template = {
            "run": {
                "name": "{{name}}",
                "replicas": 1,
                "lambda_arn": "{{reward_lambda_arn}}",
            },
            "training_config": {
                "max_steps": "{{max_steps}}",
                "lr": "{{lr}}",
                "reasoning_effort": "{{reasoning_effort}}",
            },
        }

        overrides_template = {
            "name": {"default": "", "type": "string", "required": True},
            "reward_lambda_arn": {"default": "", "type": "string", "required": True},
            "max_steps": {"default": 100, "type": "integer", "min": 1, "max": 10000},
            "lr": {"default": 5e-6, "type": "float"},
            "reasoning_effort": {
                "type": "string",
                "required": True,
                "enum": ["low", "medium", "high"],
                "default": "medium",
            },
        }

        mock_download.return_value = (recipe_template, overrides_template, "image_uri")

        rft_lambda = "arn:aws:lambda:us-east-1:123456789012:function:reward"
        builder = RecipeBuilder(
            region=self.region,
            job_name=self.job_name,
            platform=self.platform,
            model=self.mock_model,
            method=TrainingMethod.RFT_LORA,
            instance_type=self.instance_type,
            instance_count=self.instance_count,
            infra=self.mock_infra,
            output_s3_path=self.output_s3,
            data_s3_path=self.data_s3,
            rft_lambda_arn=rft_lambda,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, "recipe.yaml")
            with self.assertRaises(ValueError) as ctx:
                builder.build_and_validate(
                    output_recipe_path=output_path,
                    overrides={"reasoning_effort": None},
                    validation_config=ValidationConfig(iam=False, infra=False, recipe=True),
                )
            self.assertIn("expects string", str(ctx.exception))
            self.assertIn("NoneType", str(ctx.exception))

    @patch("amzn_nova_forge.util.recipe._get_smhp_replicas_enum", return_value=None)
    @patch("amzn_nova_forge.util.recipe.get_hub_recipe_metadata")
    @patch("amzn_nova_forge.util.recipe.download_templates_from_s3")
    def test_non_nullable_field_none_still_raises(
        self, mock_download, mock_metadata, _mock_replicas
    ):
        """max_steps=None must still raise — only fields typed as X | None are nullable."""
        mock_metadata.return_value = {"recipe_uri": "s3://bucket/recipe"}

        recipe_template = {
            "run": {
                "name": "{{name}}",
                "replicas": 1,
                "lambda_arn": "{{reward_lambda_arn}}",
            },
            "training_config": {
                "max_steps": 100,
                "lr": "{{lr}}",
            },
        }

        overrides_template = {
            "name": {"default": "", "type": "string", "required": True},
            "reward_lambda_arn": {"default": "", "type": "string", "required": True},
            "max_steps": {"default": 100, "type": "integer", "min": 1, "max": 10000},
            "lr": {"default": 5e-6, "type": "float"},
        }

        mock_download.return_value = (recipe_template, overrides_template, "image_uri")

        rft_lambda = "arn:aws:lambda:us-east-1:123456789012:function:reward"
        builder = RecipeBuilder(
            region=self.region,
            job_name=self.job_name,
            platform=self.platform,
            model=self.mock_model,
            method=TrainingMethod.RFT_LORA,
            instance_type=self.instance_type,
            instance_count=self.instance_count,
            infra=self.mock_infra,
            output_s3_path=self.output_s3,
            data_s3_path=self.data_s3,
            rft_lambda_arn=rft_lambda,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, "recipe.yaml")
            with self.assertRaises(ValueError) as ctx:
                builder.build_and_validate(
                    output_recipe_path=output_path,
                    overrides={"max_steps": None},
                    validation_config=ValidationConfig(iam=False, infra=False, recipe=True),
                )
            self.assertIn("expects int", str(ctx.exception))
            self.assertIn("NoneType", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
