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
import re
import unittest
from datetime import datetime, timezone
from unittest.mock import MagicMock, PropertyMock, create_autospec, patch

from amzn_nova_forge.core.enums import Model, Platform, TrainingMethod
from amzn_nova_forge.core.result import SMTJBatchInferenceResult
from amzn_nova_forge.core.result.training_result import TrainingResult
from amzn_nova_forge.core.types import ForgeConfig, ModelArtifacts
from amzn_nova_forge.inference.forge_inference import ForgeInference
from amzn_nova_forge.manager.runtime_manager import (
    SMHPRuntimeManager,
    SMTJRuntimeManager,
)


class TestForgeInferenceConstructor(unittest.TestCase):
    """Tests for ForgeInference.__init__."""

    @patch("boto3.session.Session")
    def test_minimal_constructor_region_only(self, mock_session):
        type(mock_session.return_value).region_name = PropertyMock(return_value="us-west-2")
        inf = ForgeInference(region="us-west-2")

        self.assertEqual(inf.region, "us-west-2")
        self.assertIsNone(inf.model)
        self.assertIsNone(inf.infra)
        self.assertIsNone(inf._platform)

    @patch("boto3.session.Session")
    def test_full_constructor(self, mock_session):
        type(mock_session.return_value).region_name = PropertyMock(return_value="us-east-1")
        mock_infra = create_autospec(SMTJRuntimeManager)
        mock_infra.platform = Platform.SMTJ
        config = ForgeConfig(image_uri="test-image")

        inf = ForgeInference(
            region="us-east-1",
            model=Model.NOVA_MICRO,
            infra=mock_infra,
            config=config,
        )

        self.assertEqual(inf.region, "us-east-1")
        self.assertEqual(inf.model, Model.NOVA_MICRO)
        self.assertIs(inf.infra, mock_infra)
        self.assertEqual(inf._config.image_uri, "test-image")

    @patch("boto3.session.Session")
    def test_platform_resolved_from_infra(self, mock_session):
        type(mock_session.return_value).region_name = PropertyMock(return_value="us-east-1")
        mock_infra = create_autospec(SMTJRuntimeManager)
        mock_infra.platform = Platform.SMTJ

        inf = ForgeInference(region="us-east-1", infra=mock_infra)
        self.assertEqual(inf._platform, Platform.SMTJ)

    @patch("boto3.session.Session")
    def test_platform_is_none_when_no_infra(self, mock_session):
        type(mock_session.return_value).region_name = PropertyMock(return_value="us-east-1")
        inf = ForgeInference(region="us-east-1")
        self.assertIsNone(inf._platform)


class TestForgeInferenceInvoke(unittest.TestCase):
    """Tests for ForgeInference.invoke (real-time)."""

    @patch("boto3.session.Session")
    def setUp(self, mock_session):
        type(mock_session.return_value).region_name = PropertyMock(return_value="us-east-1")
        self.inf = ForgeInference(region="us-east-1")

    @patch("amzn_nova_forge.inference.forge_inference.invoke_sagemaker_inference")
    @patch("boto3.client")
    @patch("amzn_nova_forge.inference.forge_inference.SAGEMAKER_ENDPOINT_ARN_REGEX")
    @patch("amzn_nova_forge.inference.forge_inference.validate_endpoint_arn")
    def test_invoke_sagemaker_endpoint(
        self, mock_validate, mock_regex, mock_boto_client, mock_invoke_sm
    ):
        arn = "arn:aws:sagemaker:us-east-1:123456789012:endpoint/my-endpoint"
        mock_regex.match.return_value = re.match(".*", arn)
        mock_runtime_client = MagicMock()
        mock_boto_client.return_value = mock_runtime_client
        mock_invoke_sm.return_value = {"result": "ok"}

        result = self.inf.invoke(endpoint_arn=arn, request_body={"prompt": "hello"})

        mock_validate.assert_called_once_with(endpoint_arn=arn)
        mock_regex.match.assert_called_once_with(arn)
        mock_boto_client.assert_called_once_with("sagemaker-runtime", region_name="us-east-1")
        mock_invoke_sm.assert_called_once_with(
            {"prompt": "hello"}, "my-endpoint", mock_runtime_client, inference_component_name=None
        )
        self.assertEqual(result, {"result": "ok"})

    @patch("amzn_nova_forge.inference.forge_inference.invoke_model")
    @patch("boto3.client")
    @patch("amzn_nova_forge.inference.forge_inference.SAGEMAKER_ENDPOINT_ARN_REGEX")
    @patch("amzn_nova_forge.inference.forge_inference.validate_endpoint_arn")
    def test_invoke_bedrock_endpoint(
        self, mock_validate, mock_regex, mock_boto_client, mock_invoke_model
    ):
        arn = "arn:aws:bedrock:us-east-1:123456789012:provisioned-model/my-model"
        mock_regex.match.return_value = None  # Not a SageMaker ARN
        mock_runtime_client = MagicMock()
        mock_boto_client.return_value = mock_runtime_client
        mock_invoke_model.return_value = {"output": "generated"}

        result = self.inf.invoke(endpoint_arn=arn, request_body={"prompt": "hello"})

        mock_validate.assert_called_once_with(endpoint_arn=arn)
        mock_boto_client.assert_called_once_with("bedrock-runtime", region_name="us-east-1")
        mock_invoke_model.assert_called_once_with(
            model_id=arn,
            request_body={"prompt": "hello"},
            bedrock_runtime=mock_runtime_client,
        )
        self.assertEqual(result, {"output": "generated"})


class TestForgeInferenceInvokeBatch(unittest.TestCase):
    """Tests for ForgeInference.invoke_batch."""

    def _make_inference(self, platform=Platform.SMTJ, method=TrainingMethod.SFT_LORA):
        with patch("boto3.session.Session") as mock_session:
            type(mock_session.return_value).region_name = PropertyMock(return_value="us-east-1")
            mock_infra = create_autospec(SMTJRuntimeManager)
            mock_infra.platform = platform
            mock_infra.kms_key_id = None
            mock_infra.instance_type = "ml.p4d.24xlarge"
            mock_infra.instance_count = 1
            mock_infra.execute.return_value = "job-123"
            inf = ForgeInference(
                region="us-east-1",
                model=Model.NOVA_MICRO,
                infra=mock_infra,
                method=method,
            )
            return inf, mock_infra

    def test_missing_model_raises_value_error(self):
        with patch("boto3.session.Session") as mock_session:
            type(mock_session.return_value).region_name = PropertyMock(return_value="us-east-1")
            inf = ForgeInference(region="us-east-1", method=TrainingMethod.SFT_LORA)

        with self.assertRaises(ValueError) as ctx:
            inf.invoke_batch(
                job_name="test",
                input_path="s3://bucket/input",
                output_s3_path="s3://bucket/output",
            )
        self.assertIn("model and infra are required", str(ctx.exception))

    def test_missing_method_raises_value_error(self):
        inf, _ = self._make_inference(method=None)
        with self.assertRaises(ValueError) as ctx:
            inf.invoke_batch(
                job_name="test",
                input_path="s3://bucket/input",
                output_s3_path="s3://bucket/output",
            )
        self.assertIn("method is required", str(ctx.exception))

    def test_bedrock_platform_raises_not_implemented(self):
        inf, _ = self._make_inference(platform=Platform.BEDROCK)
        with self.assertRaises(NotImplementedError) as ctx:
            inf.invoke_batch(
                job_name="test",
                input_path="s3://bucket/input",
                output_s3_path="s3://bucket/output",
            )
        self.assertIn("not supported on Bedrock", str(ctx.exception))

    @patch("amzn_nova_forge.inference.forge_inference.validate_platform_compatibility")
    @patch("amzn_nova_forge.inference.forge_inference.detect_platform_from_path")
    @patch("amzn_nova_forge.inference.forge_inference.RecipeBuilder")
    @patch("amzn_nova_forge.inference.forge_inference.set_output_s3_path")
    @patch("amzn_nova_forge.inference.forge_inference.resolve_model_checkpoint_path")
    @patch("boto3.client")
    def test_successful_batch_inference(
        self,
        mock_boto_client,
        mock_resolve_ckpt,
        mock_set_output,
        mock_recipe_builder_cls,
        mock_detect_platform,
        mock_validate_compat,
    ):
        inf, mock_infra = self._make_inference()

        mock_resolve_ckpt.return_value = "s3://bucket/checkpoint"
        mock_set_output.return_value = "s3://bucket/output"
        mock_detect_platform.return_value = Platform.SMTJ

        mock_recipe_builder = MagicMock()
        mock_recipe_builder.build_and_validate.return_value = (
            "/tmp/recipe.yaml",
            "s3://bucket/output",
            "s3://bucket/input",
            "test-image-uri",
        )
        mock_recipe_builder.job_name = "test-job"
        mock_recipe_builder_cls.return_value = mock_recipe_builder

        result = inf.invoke_batch(
            job_name="test-job",
            input_path="s3://bucket/input",
            output_s3_path="s3://bucket/output",
        )

        self.assertIsInstance(result, SMTJBatchInferenceResult)
        self.assertEqual(result.job_id, "job-123")
        mock_infra.execute.assert_called_once()

    @patch("amzn_nova_forge.inference.forge_inference.validate_platform_compatibility")
    @patch("amzn_nova_forge.inference.forge_inference.detect_platform_from_path")
    @patch("amzn_nova_forge.inference.forge_inference.RecipeBuilder")
    @patch("amzn_nova_forge.inference.forge_inference.set_output_s3_path")
    @patch("amzn_nova_forge.inference.forge_inference.resolve_model_checkpoint_path")
    @patch("boto3.client")
    def test_dry_run_returns_none(
        self,
        mock_boto_client,
        mock_resolve_ckpt,
        mock_set_output,
        mock_recipe_builder_cls,
        mock_detect_platform,
        mock_validate_compat,
    ):
        inf, mock_infra = self._make_inference()

        mock_resolve_ckpt.return_value = "s3://bucket/checkpoint"
        mock_set_output.return_value = "s3://bucket/output"
        mock_detect_platform.return_value = Platform.SMTJ

        mock_recipe_builder = MagicMock()
        mock_recipe_builder.build_and_validate.return_value = (
            "/tmp/recipe.yaml",
            "s3://bucket/output",
            "s3://bucket/input",
            "test-image-uri",
        )
        mock_recipe_builder_cls.return_value = mock_recipe_builder

        result = inf.invoke_batch(
            job_name="test-job",
            input_path="s3://bucket/input",
            output_s3_path="s3://bucket/output",
            dry_run=True,
        )

        self.assertIsNone(result)
        mock_infra.execute.assert_not_called()

    @patch("amzn_nova_forge.inference.forge_inference.validate_platform_compatibility")
    @patch("amzn_nova_forge.inference.forge_inference.detect_platform_from_path")
    @patch("amzn_nova_forge.inference.forge_inference.RecipeBuilder")
    @patch("amzn_nova_forge.inference.forge_inference.set_output_s3_path")
    @patch("amzn_nova_forge.inference.forge_inference.resolve_model_checkpoint_path")
    @patch("boto3.client")
    def test_model_path_passed_through(
        self,
        mock_boto_client,
        mock_resolve_ckpt,
        mock_set_output,
        mock_recipe_builder_cls,
        mock_detect_platform,
        mock_validate_compat,
    ):
        inf, _ = self._make_inference()

        mock_resolve_ckpt.return_value = "s3://bucket/explicit-checkpoint"
        mock_set_output.return_value = "s3://bucket/output"
        mock_detect_platform.return_value = Platform.SMTJ

        mock_recipe_builder = MagicMock()
        mock_recipe_builder.build_and_validate.return_value = (
            "/tmp/recipe.yaml",
            "s3://bucket/output",
            "s3://bucket/input",
            "test-image-uri",
        )
        mock_recipe_builder.job_name = "test-job"
        mock_recipe_builder_cls.return_value = mock_recipe_builder

        inf.invoke_batch(
            job_name="test-job",
            input_path="s3://bucket/input",
            output_s3_path="s3://bucket/output",
            model_path="s3://bucket/explicit-checkpoint",
        )

        mock_resolve_ckpt.assert_called_once_with(
            model_path="s3://bucket/explicit-checkpoint",
            job_result=None,
            customizer_job_id=None,
            customizer_output_s3_path=None,
            customizer_model_path=None,
        )

    @patch("amzn_nova_forge.inference.forge_inference.validate_platform_compatibility")
    @patch("amzn_nova_forge.inference.forge_inference.detect_platform_from_path")
    @patch("amzn_nova_forge.inference.forge_inference.RecipeBuilder")
    @patch("amzn_nova_forge.inference.forge_inference.set_output_s3_path")
    @patch("amzn_nova_forge.inference.forge_inference.resolve_model_checkpoint_path")
    @patch("boto3.client")
    def test_job_result_used_to_resolve_checkpoint_path(
        self,
        mock_boto_client,
        mock_resolve_ckpt,
        mock_set_output,
        mock_recipe_builder_cls,
        mock_detect_platform,
        mock_validate_compat,
    ):
        inf, _ = self._make_inference()

        mock_job_result = MagicMock(spec=TrainingResult)
        mock_job_result.model_artifacts = ModelArtifacts(
            checkpoint_s3_path="s3://bucket/training-checkpoint",
            output_s3_path="s3://bucket/output",
        )

        mock_resolve_ckpt.return_value = "s3://bucket/training-checkpoint"
        mock_set_output.return_value = "s3://bucket/output"
        mock_detect_platform.return_value = Platform.SMTJ

        mock_recipe_builder = MagicMock()
        mock_recipe_builder.build_and_validate.return_value = (
            "/tmp/recipe.yaml",
            "s3://bucket/output",
            "s3://bucket/input",
            "test-image-uri",
        )
        mock_recipe_builder.job_name = "test-job"
        mock_recipe_builder_cls.return_value = mock_recipe_builder

        inf.invoke_batch(
            job_name="test-job",
            input_path="s3://bucket/input",
            output_s3_path="s3://bucket/output",
            job_result=mock_job_result,
        )

        mock_resolve_ckpt.assert_called_once_with(
            model_path=None,
            job_result=mock_job_result,
            customizer_job_id=None,
            customizer_output_s3_path=None,
            customizer_model_path=None,
        )

    @patch("amzn_nova_forge.inference.forge_inference.validate_platform_compatibility")
    @patch("amzn_nova_forge.inference.forge_inference.detect_platform_from_path")
    @patch("amzn_nova_forge.inference.forge_inference.RecipeBuilder")
    @patch("amzn_nova_forge.inference.forge_inference.set_output_s3_path")
    @patch("amzn_nova_forge.inference.forge_inference.resolve_model_checkpoint_path")
    @patch("boto3.client")
    def test_output_path_uses_job_id(
        self,
        mock_boto_client,
        mock_resolve_ckpt,
        mock_set_output,
        mock_recipe_builder_cls,
        mock_detect_platform,
        mock_validate_compat,
    ):
        inf, mock_infra = self._make_inference()

        mock_resolve_ckpt.return_value = "s3://bucket/checkpoint"
        mock_set_output.return_value = "s3://bucket/output"
        mock_detect_platform.return_value = Platform.SMTJ

        mock_recipe_builder = MagicMock()
        mock_recipe_builder.build_and_validate.return_value = (
            "/tmp/recipe.yaml",
            "s3://bucket/output",
            "s3://bucket/input",
            "test-image-uri",
        )
        mock_recipe_builder.job_name = "my-batch-job"
        mock_recipe_builder_cls.return_value = mock_recipe_builder

        result = inf.invoke_batch(
            job_name="my-batch-job",
            input_path="s3://bucket/input",
            output_s3_path="s3://bucket/output",
        )

        # Output path should use job_id (returned by infra.execute), not user-provided job_name
        self.assertIn("job-123/output/output.tar.gz", result.inference_output_path)


class TestForgeInferenceCaching(unittest.TestCase):
    """Tests for caching integration in ForgeInference.invoke_batch()."""

    def _make_inference(self, enable_caching=False):
        with patch("boto3.session.Session") as mock_session:
            type(mock_session.return_value).region_name = PropertyMock(return_value="us-east-1")
            mock_infra = create_autospec(SMTJRuntimeManager)
            mock_infra.platform = Platform.SMTJ
            mock_infra.kms_key_id = None
            mock_infra.instance_type = "ml.p4d.24xlarge"
            mock_infra.instance_count = 1
            mock_infra.execute.return_value = "batch-job-123"
            config = ForgeConfig(enable_job_caching=enable_caching) if enable_caching else None
            inf = ForgeInference(
                region="us-east-1",
                model=Model.NOVA_MICRO,
                infra=mock_infra,
                method=TrainingMethod.SFT_LORA,
                config=config,
            )
            return inf, mock_infra

    @patch("amzn_nova_forge.inference.forge_inference.load_existing_result")
    def test_cached_result_short_circuits_batch_inference(self, mock_load):
        mock_cached = MagicMock(spec=SMTJBatchInferenceResult)
        mock_load.return_value = mock_cached

        inf, mock_infra = self._make_inference(enable_caching=True)
        result = inf.invoke_batch(
            job_name="cached-batch",
            input_path="s3://bucket/input",
            output_s3_path="s3://bucket/output",
        )

        self.assertIs(result, mock_cached)
        mock_infra.execute.assert_not_called()
        mock_load.assert_called_once()

    @patch("amzn_nova_forge.inference.forge_inference.persist_result")
    @patch("amzn_nova_forge.inference.forge_inference.validate_platform_compatibility")
    @patch("amzn_nova_forge.inference.forge_inference.detect_platform_from_path")
    @patch("amzn_nova_forge.inference.forge_inference.RecipeBuilder")
    @patch("amzn_nova_forge.inference.forge_inference.set_output_s3_path")
    @patch("amzn_nova_forge.inference.forge_inference.resolve_model_checkpoint_path")
    @patch("boto3.client")
    @patch(
        "amzn_nova_forge.inference.forge_inference.load_existing_result",
        return_value=None,
    )
    def test_persist_called_after_successful_batch_inference(
        self,
        mock_load,
        mock_boto_client,
        mock_resolve,
        mock_set_output,
        mock_recipe_cls,
        mock_detect,
        mock_validate_compat,
        mock_persist,
    ):
        mock_resolve.return_value = "s3://bucket/checkpoint"
        mock_set_output.return_value = "s3://bucket/output"
        mock_detect.return_value = Platform.SMTJ
        mock_builder = MagicMock()
        mock_builder.build_and_validate.return_value = (
            "/tmp/recipe.yaml",
            "s3://bucket/output",
            "s3://bucket/input",
            "test-image-uri",
        )
        mock_builder.job_name = "test-batch"
        mock_recipe_cls.return_value = mock_builder

        inf, _ = self._make_inference(enable_caching=True)
        inf.invoke_batch(
            job_name="test-batch",
            input_path="s3://bucket/input",
            output_s3_path="s3://bucket/output",
        )

        mock_persist.assert_called_once()
        call_kwargs = mock_persist.call_args
        self.assertEqual(call_kwargs[1]["job_name"], "test-batch")
        self.assertEqual(call_kwargs[1]["job_type"], "batch_inference")

    @patch("amzn_nova_forge.inference.forge_inference.load_existing_result")
    def test_cache_load_receives_model_path(self, mock_load):
        mock_load.return_value = MagicMock(spec=SMTJBatchInferenceResult)

        inf, _ = self._make_inference(enable_caching=True)
        inf.invoke_batch(
            job_name="test-batch",
            input_path="s3://bucket/input",
            output_s3_path="s3://bucket/output",
            model_path="s3://bucket/my-checkpoint",
        )

        call_kwargs = mock_load.call_args[1]
        self.assertEqual(call_kwargs["model_path"], "s3://bucket/my-checkpoint")


class TestForgeInferenceGetLogs(unittest.TestCase):
    """Tests for ForgeInference.get_logs."""

    @patch("boto3.session.Session")
    def _make_inference_with_platform(self, platform, mock_session):
        type(mock_session.return_value).region_name = PropertyMock(return_value="us-east-1")
        mock_infra = create_autospec(SMTJRuntimeManager)
        mock_infra.platform = platform
        return ForgeInference(
            region="us-east-1",
            model=Model.NOVA_MICRO,
            infra=mock_infra,
        )

    @patch("amzn_nova_forge.inference.forge_inference.CloudWatchLogMonitor")
    def test_get_logs_with_job_result(self, mock_monitor_cls):
        inf = self._make_inference_with_platform(Platform.SMTJ)
        mock_job_result = MagicMock()
        mock_job_result.job_id = "job-abc"
        mock_job_result.started_time = datetime(2024, 1, 1, tzinfo=timezone.utc)

        mock_monitor = MagicMock()
        mock_monitor_cls.return_value = mock_monitor

        inf.get_logs(job_result=mock_job_result)

        mock_monitor_cls.assert_called_once_with(
            job_id="job-abc",
            platform=Platform.SMTJ,
            started_time=int(datetime(2024, 1, 1, tzinfo=timezone.utc).timestamp() * 1000),
            region="us-east-1",
        )
        mock_monitor.show_logs.assert_called_once_with(
            limit=None, start_from_head=False, end_time=None
        )

    def test_get_logs_missing_params_raises_value_error(self):
        inf = self._make_inference_with_platform(Platform.SMTJ)

        with self.assertRaises(ValueError) as ctx:
            inf.get_logs()  # No job_result, no job_id, no started_time

        self.assertIn("job_result", str(ctx.exception))

    def test_get_logs_job_id_only_raises_value_error(self):
        inf = self._make_inference_with_platform(Platform.SMTJ)

        with self.assertRaises(ValueError) as ctx:
            inf.get_logs(job_id="some-job")

        self.assertIn("job_result", str(ctx.exception))

    def test_get_logs_started_time_only_raises_value_error(self):
        inf = self._make_inference_with_platform(Platform.SMTJ)

        with self.assertRaises(ValueError) as ctx:
            inf.get_logs(started_time=datetime(2024, 1, 1, tzinfo=timezone.utc))

        self.assertIn("job_result", str(ctx.exception))

    def test_get_logs_no_platform_raises_value_error(self):
        with patch("boto3.session.Session") as mock_session:
            type(mock_session.return_value).region_name = PropertyMock(return_value="us-east-1")
            inf = ForgeInference(region="us-east-1")  # No infra -> no platform

        with self.assertRaises(ValueError) as ctx:
            inf.get_logs(
                job_id="job-xyz",
                started_time=datetime(2024, 1, 1, tzinfo=timezone.utc),
            )

        self.assertIn("platform", str(ctx.exception))

    @patch("amzn_nova_forge.inference.forge_inference.CloudWatchLogMonitor")
    def test_get_logs_smhp_includes_cluster_and_namespace(self, mock_monitor_cls):
        with patch("boto3.session.Session") as mock_session:
            type(mock_session.return_value).region_name = PropertyMock(return_value="us-east-1")
            mock_infra = create_autospec(SMHPRuntimeManager)
            mock_infra.platform = Platform.SMHP
            mock_infra.cluster_name = "my-cluster"
            mock_infra.namespace = "my-namespace"
            inf = ForgeInference(
                region="us-east-1",
                model=Model.NOVA_MICRO,
                infra=mock_infra,
            )

        mock_monitor = MagicMock()
        mock_monitor_cls.return_value = mock_monitor

        start = datetime(2024, 6, 15, tzinfo=timezone.utc)
        inf.get_logs(job_id="job-smhp", started_time=start)

        mock_monitor_cls.assert_called_once_with(
            job_id="job-smhp",
            platform=Platform.SMHP,
            started_time=int(start.timestamp() * 1000),
            cluster_name="my-cluster",
            namespace="my-namespace",
            region="us-east-1",
        )
        mock_monitor.show_logs.assert_called_once()


if __name__ == "__main__":
    unittest.main()
