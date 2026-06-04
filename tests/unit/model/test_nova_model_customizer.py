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
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock, PropertyMock, create_autospec, patch

import boto3
from botocore.exceptions import ClientError

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
    SMTJBatchInferenceResult,
    SMTJEvaluationResult,
)
from amzn_nova_forge.core.result.job_result import JobStatus
from amzn_nova_forge.core.result.training_result import (
    SMTJTrainingResult,
    TrainingResult,
)
from amzn_nova_forge.core.types import EndpointInfo, ModelArtifacts
from amzn_nova_forge.manager.runtime_manager import (
    SMHPRuntimeManager,
    SMTJRuntimeManager,
    SMTJServerlessRuntimeManager,
)
from amzn_nova_forge.model.model_config import ModelDeployResult
from amzn_nova_forge.model.nova_model_customizer import (
    NovaModelCustomizer,
    _invoke_inference_extra_info,
    _resolve_deploy_platform,
)
from amzn_nova_forge.model.nova_model_customizer_util import (
    collect_all_parameters,
    generate_job_hash,
    get_recipe_directory,
    get_result_file_path,
    load_existing_result,
    matches_job_cache_criteria,
    persist_result,
    should_persist_results,
)
from amzn_nova_forge.util.bedrock import BEDROCK_EXECUTION_ROLE_NAME
from amzn_nova_forge.util.recipe import RecipePath
from amzn_nova_forge.validation.validator import (
    verify_rft_lambda,
)


class MockRecipePath(RecipePath):
    """Mock RecipePath that acts as a context manager for tests"""

    def __init__(self, path: str):
        super().__init__(path, temp=False)

    def close(self):
        # Override to prevent actual cleanup in tests
        pass


class TestNovaModelCustomizer(unittest.TestCase):
    def setUp(self):
        self.model = Model.NOVA_MICRO
        self.method = TrainingMethod.SFT_LORA
        self.data_s3_path = "s3://test-bucket/data"
        self.output_s3_path = "s3://test-bucket/output"

        self.mock_runtime_manager = create_autospec(SMTJRuntimeManager)
        self.mock_runtime_manager.platform = Platform.SMTJ
        self.mock_runtime_manager.instance_count = 2

        # Keep mocks alive for the full test lifecycle (not just __init__) so
        # that facade delegation to service classes can construct boto3 clients.
        p_client = patch("boto3.client")
        p_session = patch("boto3.session.Session")
        p_exec_role = patch("sagemaker.core.helper.session_helper.get_execution_role")
        p_hub_meta = patch("amzn_nova_forge.util.recipe.get_hub_recipe_metadata")
        p_download = patch("amzn_nova_forge.util.recipe.download_templates_from_s3")

        mock_client = p_client.start()
        mock_session = p_session.start()
        mock_get_execution_role = p_exec_role.start()
        mock_get_hub_metadata = p_hub_meta.start()
        mock_download_s3 = p_download.start()

        self.addCleanup(p_client.stop)
        self.addCleanup(p_session.stop)
        self.addCleanup(p_exec_role.stop)
        self.addCleanup(p_hub_meta.stop)
        self.addCleanup(p_download.stop)

        type(mock_session.return_value).region_name = PropertyMock(return_value="us-east-1")
        mock_get_execution_role.return_value = (
            "arn:aws:iam::123456789012:role/SageMakerExecutionRole"
        )

        # Mock recipe metadata and templates
        mock_get_hub_metadata.return_value = {
            "SmtjRecipeTemplateS3Uri": "s3://test-bucket/recipe.yaml",
            "SmtjOverrideParamsS3Uri": "s3://test-bucket/overrides.json",
        }
        mock_download_s3.return_value = ({}, {})

        mock_sts = MagicMock()
        mock_sts.get_caller_identity.return_value = {"Account": "123456789012"}
        mock_s3 = MagicMock()
        mock_s3.head_bucket.return_value = {}

        # Add IAM and SageMaker mocking for validation
        mock_iam = MagicMock()
        mock_iam.get_role.return_value = {
            "Role": {
                "AssumeRolePolicyDocument": {
                    "Statement": [
                        {
                            "Effect": "Allow",
                            "Principal": {"Service": "sagemaker.amazonaws.com"},
                            "Action": "sts:AssumeRole",
                        }
                    ]
                }
            }
        }
        mock_sagemaker = MagicMock()

        def client_side_effect(service, **kwargs):
            if service == "sts":
                return mock_sts
            elif service == "s3":
                return mock_s3
            elif service == "iam":
                return mock_iam
            elif service == "sagemaker":
                return mock_sagemaker
            return MagicMock()

        mock_client.side_effect = client_side_effect
        self._mock_boto_client = mock_client

        self.customizer = NovaModelCustomizer(
            model=self.model,
            method=self.method,
            infra=self.mock_runtime_manager,
            data_s3_path=self.data_s3_path,
            output_s3_path=self.output_s3_path,
        )

    def test_init_raises_value_error_on_unsupported_region(self):
        with patch("boto3.session.Session") as mock_session:
            type(mock_session.return_value).region_name = PropertyMock(
                return_value="unsupported-region"
            )
            with self.assertRaises(ValueError) as context:
                NovaModelCustomizer(
                    model=self.model,
                    method=self.method,
                    infra=self.mock_runtime_manager,
                    data_s3_path=self.data_s3_path,
                    output_s3_path=self.output_s3_path,
                )
            self.assertIn("unsupported-region", str(context.exception))
            self.assertIn("not supported", str(context.exception))

    @patch.object(SMTJServerlessRuntimeManager, "setup", return_value=None)
    def test_serverless_model_path_must_be_arn(self, mock_setup):
        """model_path for SMTJServerless must be a SageMaker model package ARN."""
        runtime = SMTJServerlessRuntimeManager(model_package_group_name="test-group")
        runtime.execution_role = "arn:aws:iam::123:role/test"
        runtime.sagemaker_client = MagicMock()
        runtime.region = "us-east-1"
        runtime.model_package_group_arn = "arn:aws:sagemaker:us-east-1:123:model-package-group/test"

        with (
            patch("boto3.client"),
            patch(
                "amzn_nova_forge.model.nova_model_customizer.set_output_s3_path",
                return_value="s3://bucket/output",
            ),
        ):
            # S3 path should raise ValueError
            with self.assertRaises(ValueError) as ctx:
                NovaModelCustomizer(
                    model=Model.NOVA_MICRO,
                    method=TrainingMethod.SFT_LORA,
                    infra=runtime,
                    model_path="s3://bucket/checkpoint/",
                )
            self.assertIn("model package ARN", str(ctx.exception))

            # Valid ARN should not raise
            NovaModelCustomizer(
                model=Model.NOVA_MICRO,
                method=TrainingMethod.SFT_LORA,
                infra=runtime,
                model_path="arn:aws:sagemaker:us-east-1:123456789012:model-package/group/1",
            )

            # None should not raise (no iterative training)
            NovaModelCustomizer(
                model=Model.NOVA_MICRO,
                method=TrainingMethod.SFT_LORA,
                infra=runtime,
                model_path=None,
            )

    def test_set_model_config_variants(self):
        for model in Model:
            mock_infra = create_autospec(SMTJRuntimeManager)

            with (
                patch("boto3.client") as mock_client,
                patch(
                    "amzn_nova_forge.util.recipe.get_hub_recipe_metadata"
                ) as mock_get_hub_metadata,
                patch("amzn_nova_forge.util.recipe.download_templates_from_s3") as mock_download_s3,
            ):
                mock_get_hub_metadata.return_value = {
                    "SmtjRecipeTemplateS3Uri": "s3://test-bucket/recipe.yaml",
                    "SmtjOverrideParamsS3Uri": "s3://test-bucket/overrides.json",
                }
                mock_download_s3.return_value = ({}, {})

                mock_sts = MagicMock()
                mock_sts.get_caller_identity.return_value = {"Account": "123456789012"}
                mock_s3 = MagicMock()
                mock_s3.head_bucket.return_value = {}

                def client_side_effect(service, **kwargs):
                    if service == "sts":
                        return mock_sts
                    elif service == "s3":
                        return mock_s3
                    return MagicMock()

                mock_client.side_effect = client_side_effect

                customizer = NovaModelCustomizer(
                    model=model,
                    method=self.method,
                    infra=mock_infra,
                    data_s3_path=self.data_s3_path,
                    output_s3_path=self.output_s3_path,
                )

                self.assertEqual(customizer.model.model_type, model.model_type)
                self.assertEqual(customizer.model.model_path, model.model_path)

    def test_invalid_model_raises_error(self):
        mock_infra = create_autospec(SMTJRuntimeManager)

        with (
            patch("boto3.client") as mock_client,
            patch("amzn_nova_forge.util.recipe.get_hub_recipe_metadata") as mock_get_hub_metadata,
            patch("amzn_nova_forge.util.recipe.download_templates_from_s3") as mock_download_s3,
        ):
            mock_get_hub_metadata.return_value = {
                "SmtjRecipeTemplateS3Uri": "s3://test-bucket/recipe.yaml",
                "SmtjOverrideParamsS3Uri": "s3://test-bucket/overrides.json",
            }
            mock_download_s3.return_value = ({}, {})

            mock_sts = MagicMock()
            mock_sts.get_caller_identity.return_value = {"Account": "123456789012"}
            mock_s3 = MagicMock()
            mock_s3.head_bucket.return_value = {}

            def client_side_effect(service, **kwargs):
                if service == "sts":
                    return mock_sts
                elif service == "s3":
                    return mock_s3
                return MagicMock()

            mock_client.side_effect = client_side_effect

            customizer = NovaModelCustomizer(
                model=Model.NOVA_MICRO,
                method=self.method,
                infra=mock_infra,
                data_s3_path=self.data_s3_path,
                output_s3_path=self.output_s3_path,
            )

            self.assertIsNotNone(customizer)

    def test_customizer_attributes(self):
        self.assertEqual(self.customizer.model, Model.NOVA_MICRO)
        self.assertEqual(self.customizer.method, TrainingMethod.SFT_LORA)
        self.assertEqual(self.customizer.data_s3_path, self.data_s3_path)
        self.assertEqual(self.customizer.output_s3_path, self.output_s3_path)
        self.assertEqual(self.customizer.region, "us-east-1")

    def test_model_setter_getter(self):
        """Test that model setter and getter work correctly."""
        # Test initial value
        self.assertEqual(self.customizer.model, Model.NOVA_MICRO)

        # Test setter - change to NOVA_LITE
        self.customizer.model = Model.NOVA_LITE
        self.assertEqual(self.customizer.model, Model.NOVA_LITE)

        # Test setter - change to NOVA_PRO
        self.customizer.model = Model.NOVA_PRO
        self.assertEqual(self.customizer.model, Model.NOVA_PRO)

    def test_multiple_training_methods(self):
        for method in TrainingMethod:
            mock_infra = create_autospec(SMTJRuntimeManager)

            with (
                patch("boto3.client") as mock_client,
                patch(
                    "amzn_nova_forge.util.recipe.get_hub_recipe_metadata"
                ) as mock_get_hub_metadata,
                patch("amzn_nova_forge.util.recipe.download_templates_from_s3") as mock_download_s3,
            ):
                mock_get_hub_metadata.return_value = {
                    "SmtjRecipeTemplateS3Uri": "s3://test-bucket/recipe.yaml",
                    "SmtjOverrideParamsS3Uri": "s3://test-bucket/overrides.json",
                }
                mock_download_s3.return_value = ({}, {})

                mock_sts = MagicMock()
                mock_sts.get_caller_identity.return_value = {"Account": "123456789012"}
                mock_s3 = MagicMock()
                mock_s3.head_bucket.return_value = {}

                def client_side_effect(service, **kwargs):
                    if service == "sts":
                        return mock_sts
                    elif service == "s3":
                        return mock_s3
                    return MagicMock()

                mock_client.side_effect = client_side_effect

                customizer = NovaModelCustomizer(
                    model=Model.NOVA_MICRO,
                    method=method,
                    infra=mock_infra,
                    data_s3_path=self.data_s3_path,
                    output_s3_path=self.output_s3_path,
                )

                self.assertEqual(customizer.method, method)

    @patch("amzn_nova_forge.util.recipe.download_templates_from_s3")
    @patch("amzn_nova_forge.util.recipe.get_hub_recipe_metadata")
    @patch("boto3.session.Session")
    @patch("boto3.client")
    def test_auto_generate_output_path_bucket_exists(
        self, mock_client, mock_session, mock_get_hub_metadata, mock_download_s3
    ):
        type(mock_session.return_value).region_name = PropertyMock(return_value="us-east-1")
        mock_get_hub_metadata.return_value = {
            "SmtjRecipeTemplateS3Uri": "s3://test-bucket/recipe.yaml",
            "SmtjOverrideParamsS3Uri": "s3://test-bucket/overrides.json",
        }
        mock_download_s3.return_value = ({}, {})

        mock_sts = MagicMock()
        mock_sts.get_caller_identity.return_value = {"Account": "123456789012"}
        mock_s3 = MagicMock()
        mock_s3.head_bucket.return_value = {}

        def client_side_effect(service, **kwargs):
            if service == "sts":
                return mock_sts
            elif service == "s3":
                return mock_s3
            return MagicMock()

        mock_client.side_effect = client_side_effect

        mock_infra = create_autospec(SMTJRuntimeManager)
        customizer = NovaModelCustomizer(
            model=Model.NOVA_MICRO,
            method=TrainingMethod.SFT_LORA,
            infra=mock_infra,
            data_s3_path=self.data_s3_path,
        )

        expected_output = "s3://sagemaker-nova-123456789012-us-east-1/output"
        self.assertEqual(customizer.output_s3_path, expected_output)
        mock_s3.head_bucket.assert_called_once_with(Bucket="sagemaker-nova-123456789012-us-east-1")
        mock_s3.create_bucket.assert_not_called()

    @patch("amzn_nova_forge.util.recipe.download_templates_from_s3")
    @patch("amzn_nova_forge.util.recipe.get_hub_recipe_metadata")
    @patch("boto3.client")
    def test_auto_generate_output_path_bucket_creation_fails(
        self, mock_client, mock_get_hub_metadata, mock_download_s3
    ):
        mock_get_hub_metadata.return_value = {
            "SmtjRecipeTemplateS3Uri": "s3://test-bucket/recipe.yaml",
            "SmtjOverrideParamsS3Uri": "s3://test-bucket/overrides.json",
        }
        mock_download_s3.return_value = ({}, {})

        mock_sts = MagicMock()
        mock_sts.get_caller_identity.return_value = {"Account": "123456789012"}
        mock_s3 = MagicMock()
        mock_s3.head_bucket.side_effect = Exception("Bucket does not exist")
        mock_s3.create_bucket.side_effect = Exception("Access Denied")

        def client_side_effect(service, **kwargs):
            if service == "sts":
                return mock_sts
            elif service == "s3":
                return mock_s3
            return MagicMock()

        mock_client.side_effect = client_side_effect

        mock_infra = create_autospec(SMTJRuntimeManager)

        with self.assertRaises(Exception) as context:
            NovaModelCustomizer(
                model=Model.NOVA_MICRO,
                method=TrainingMethod.SFT_LORA,
                infra=mock_infra,
                data_s3_path=self.data_s3_path,
            )

        self.assertIn("Failed to create output bucket", str(context.exception))
        self.assertIn("Access Denied", str(context.exception))

    @patch("amzn_nova_forge.util.recipe.download_templates_from_s3")
    @patch("amzn_nova_forge.util.recipe.get_hub_recipe_metadata")
    @patch("boto3.client")
    def test_explicit_output_path_bucket_exists(
        self, mock_client, mock_get_hub_metadata, mock_download_s3
    ):
        mock_get_hub_metadata.return_value = {
            "SmtjRecipeTemplateS3Uri": "s3://test-bucket/recipe.yaml",
            "SmtjOverrideParamsS3Uri": "s3://test-bucket/overrides.json",
        }
        mock_download_s3.return_value = ({}, {})

        mock_sts = MagicMock()
        mock_sts.get_caller_identity.return_value = {"Account": "123456789012"}
        mock_s3 = MagicMock()

        def client_side_effect(service, **kwargs):
            if service == "sts":
                return mock_sts
            elif service == "s3":
                return mock_s3
            return MagicMock()

        mock_client.side_effect = client_side_effect

        mock_infra = create_autospec(SMTJRuntimeManager)
        custom_output = "s3://my-custom-bucket/my-output/"

        customizer = NovaModelCustomizer(
            model=Model.NOVA_MICRO,
            method=TrainingMethod.SFT_LORA,
            infra=mock_infra,
            data_s3_path=self.data_s3_path,
            output_s3_path=custom_output,
        )

        self.assertEqual(customizer.output_s3_path, "s3://my-custom-bucket/my-output")
        mock_s3.head_bucket.assert_called_once()
        mock_s3.create_bucket.assert_not_called()

    @patch("amzn_nova_forge.util.recipe.download_templates_from_s3")
    @patch("amzn_nova_forge.util.recipe.get_hub_recipe_metadata")
    @patch("boto3.client")
    def test_explicit_output_path_bucket_does_not_exist(
        self, mock_client, mock_get_hub_metadata, mock_download_s3
    ):
        mock_get_hub_metadata.return_value = {
            "SmtjRecipeTemplateS3Uri": "s3://test-bucket/recipe.yaml",
            "SmtjOverrideParamsS3Uri": "s3://test-bucket/overrides.json",
        }
        mock_download_s3.return_value = ({}, {})

        s3_exceptions = boto3.client("s3").exceptions

        mock_sts = MagicMock()
        mock_sts.get_caller_identity.return_value = {"Account": "123456789012"}

        mock_s3 = MagicMock()
        mock_s3.exceptions = s3_exceptions
        mock_s3.head_bucket.side_effect = ClientError(
            error_response={
                "Error": {
                    "Code": "NoSuchBucket",
                    "Message": "The specified bucket does not exist",
                }
            },
            operation_name="HeadBucket",
        )

        def client_side_effect(service, **kwargs):
            if service == "sts":
                return mock_sts
            elif service == "s3":
                return mock_s3
            return MagicMock()

        mock_client.side_effect = client_side_effect

        mock_infra = create_autospec(SMTJRuntimeManager)
        custom_output = "s3://new-bucket/my-output/"

        customizer = NovaModelCustomizer(
            model=Model.NOVA_MICRO,
            method=TrainingMethod.SFT_LORA,
            infra=mock_infra,
            data_s3_path=self.data_s3_path,
            output_s3_path=custom_output,
        )

        self.assertEqual(customizer.output_s3_path, "s3://new-bucket/my-output")

        mock_s3.head_bucket.assert_called_once_with(
            Bucket="new-bucket", ExpectedBucketOwner="123456789012"
        )
        mock_s3.create_bucket.assert_called_once_with(Bucket="new-bucket")

    def test_data_mixing_raises_error_on_unsupported_platform(self):
        mock_infra = create_autospec(SMTJRuntimeManager)

        with (
            patch("boto3.client") as mock_client,
            patch("amzn_nova_forge.util.recipe.get_hub_recipe_metadata") as mock_get_hub_metadata,
            patch("amzn_nova_forge.util.recipe.download_templates_from_s3") as mock_download_s3,
        ):
            mock_get_hub_metadata.return_value = {
                "SmtjRecipeTemplateS3Uri": "s3://test-bucket/recipe.yaml",
                "SmtjOverrideParamsS3Uri": "s3://test-bucket/overrides.json",
            }
            mock_download_s3.return_value = ({}, {})

            mock_sts = MagicMock()
            mock_sts.get_caller_identity.return_value = {"Account": "123456789012"}
            mock_s3 = MagicMock()
            mock_s3.head_bucket.return_value = {}

            def client_side_effect(service, **kwargs):
                if service == "sts":
                    return mock_sts
                elif service == "s3":
                    return mock_s3
                return MagicMock()

            mock_client.side_effect = client_side_effect

            with self.assertRaises(ValueError) as context:
                NovaModelCustomizer(
                    model=Model.NOVA_MICRO,
                    method=TrainingMethod.CPT,
                    infra=mock_infra,  # SMTJ, not SMHP
                    data_s3_path=self.data_s3_path,
                    output_s3_path=self.output_s3_path,
                    data_mixing_enabled=True,
                )

            self.assertIn("Data mixing is only supported for", str(context.exception))

    def test_data_mixing_raises_error_on_unsupported_training_method(self):
        mock_infra = create_autospec(SMHPRuntimeManager)
        mock_infra.platform = Platform.SMHP

        with (
            patch("boto3.client") as mock_client,
            patch("amzn_nova_forge.util.recipe.get_hub_recipe_metadata") as mock_get_hub_metadata,
            patch("amzn_nova_forge.util.recipe.download_templates_from_s3") as mock_download_s3,
        ):
            mock_get_hub_metadata.return_value = {
                "SmtjRecipeTemplateS3Uri": "s3://test-bucket/recipe.yaml",
                "SmtjOverrideParamsS3Uri": "s3://test-bucket/overrides.json",
            }
            mock_download_s3.return_value = ({}, {})

            mock_sts = MagicMock()
            mock_sts.get_caller_identity.return_value = {"Account": "123456789012"}
            mock_s3 = MagicMock()
            mock_s3.head_bucket.return_value = {}

            def client_side_effect(service, **kwargs):
                if service == "sts":
                    return mock_sts
                elif service == "s3":
                    return mock_s3
                return MagicMock()

            mock_client.side_effect = client_side_effect

            with self.assertRaises(ValueError) as context:
                NovaModelCustomizer(
                    model=Model.NOVA_MICRO,
                    method=TrainingMethod.RFT_FULL,  # Not supported for datamixing
                    infra=mock_infra,
                    data_s3_path=self.data_s3_path,
                    output_s3_path=self.output_s3_path,
                    data_mixing_enabled=True,
                )

            self.assertIn("Data mixing is only supported for", str(context.exception))

    def _make_customizer(self, mock_client, **kwargs):
        """Helper to create a NovaModelCustomizer with standard mocks."""
        mock_sts = MagicMock()
        mock_sts.get_caller_identity.return_value = {"Account": "123456789012"}
        mock_s3 = MagicMock()
        mock_s3.head_bucket.return_value = {}

        def client_side_effect(service, **kw):
            if service == "sts":
                return mock_sts
            elif service == "s3":
                return mock_s3
            return MagicMock()

        mock_client.side_effect = client_side_effect
        mock_infra = create_autospec(SMTJRuntimeManager)

        return NovaModelCustomizer(
            model=Model.NOVA_MICRO,
            method=TrainingMethod.SFT_LORA,
            infra=mock_infra,
            data_s3_path=self.data_s3_path,
            output_s3_path=self.output_s3_path,
            **kwargs,
        )

    @patch("boto3.client")
    def test_is_multimodal_false_when_data_mixing_disabled(self, mock_client):
        """is_multimodal is always False when data_mixing_enabled=False, regardless of is_multimodal arg."""
        customizer = self._make_customizer(
            mock_client, data_mixing_enabled=False, is_multimodal=True
        )
        self.assertFalse(customizer.is_multimodal)

    @patch("boto3.client")
    def test_is_multimodal_explicit_true_honoured(self, mock_client):
        """Explicit is_multimodal=True is honoured when data_mixing_enabled=True."""
        with patch(
            "amzn_nova_forge.model.nova_model_customizer.NovaModelCustomizer._init_data_mixing"
        ):
            mock_infra = create_autospec(SMHPRuntimeManager)
            mock_infra.platform = Platform.SMHP
            mock_sts = MagicMock()
            mock_sts.get_caller_identity.return_value = {"Account": "123456789012"}
            mock_s3 = MagicMock()
            mock_s3.head_bucket.return_value = {}

            def client_side_effect(service, **kw):
                if service == "sts":
                    return mock_sts
                elif service == "s3":
                    return mock_s3
                return MagicMock()

            mock_client.side_effect = client_side_effect

            customizer = NovaModelCustomizer(
                model=Model.NOVA_LITE,
                method=TrainingMethod.SFT_LORA,
                infra=mock_infra,
                data_s3_path=self.data_s3_path,
                output_s3_path=self.output_s3_path,
                data_mixing_enabled=True,
                is_multimodal=True,
            )
            self.assertTrue(customizer.is_multimodal)

    @patch("amzn_nova_forge.model.nova_model_customizer.is_multimodal_data")
    @patch("boto3.client")
    def test_is_multimodal_auto_detects_from_data_s3_path(
        self, mock_client, mock_is_multimodal_data
    ):
        """When is_multimodal=None and data_s3_path is set, auto-detection runs."""
        mock_is_multimodal_data.return_value = True

        with patch(
            "amzn_nova_forge.model.nova_model_customizer.NovaModelCustomizer._init_data_mixing"
        ):
            mock_infra = create_autospec(SMHPRuntimeManager)
            mock_infra.platform = Platform.SMHP
            mock_sts = MagicMock()
            mock_sts.get_caller_identity.return_value = {"Account": "123456789012"}
            mock_s3 = MagicMock()
            mock_s3.head_bucket.return_value = {}

            def client_side_effect(service, **kw):
                if service == "sts":
                    return mock_sts
                elif service == "s3":
                    return mock_s3
                return MagicMock()

            mock_client.side_effect = client_side_effect

            customizer = NovaModelCustomizer(
                model=Model.NOVA_LITE,
                method=TrainingMethod.SFT_LORA,
                infra=mock_infra,
                data_s3_path=self.data_s3_path,
                output_s3_path=self.output_s3_path,
                data_mixing_enabled=True,
                is_multimodal=None,
            )
            self.assertTrue(customizer.is_multimodal)
            mock_is_multimodal_data.assert_called_once_with(self.data_s3_path)

    @patch("boto3.client")
    def test_is_multimodal_defaults_false_when_no_data_s3_path(self, mock_client):
        """When is_multimodal=None and no data_s3_path, is_multimodal defaults to False."""
        with patch(
            "amzn_nova_forge.model.nova_model_customizer.NovaModelCustomizer._init_data_mixing"
        ):
            mock_infra = create_autospec(SMHPRuntimeManager)
            mock_infra.platform = Platform.SMHP
            mock_sts = MagicMock()
            mock_sts.get_caller_identity.return_value = {"Account": "123456789012"}
            mock_s3 = MagicMock()
            mock_s3.head_bucket.return_value = {}

            def client_side_effect(service, **kw):
                if service == "sts":
                    return mock_sts
                elif service == "s3":
                    return mock_s3
                return MagicMock()

            mock_client.side_effect = client_side_effect

            customizer = NovaModelCustomizer(
                model=Model.NOVA_LITE,
                method=TrainingMethod.SFT_LORA,
                infra=mock_infra,
                data_s3_path=None,
                output_s3_path=self.output_s3_path,
                data_mixing_enabled=True,
                is_multimodal=None,
            )
            self.assertFalse(customizer.is_multimodal)

    def _make_smhp_customizer(self, mock_client, **kwargs):
        """Helper to create a data-mixing-enabled SMHP customizer with _init_data_mixing patched."""
        mock_sts = MagicMock()
        mock_sts.get_caller_identity.return_value = {"Account": "123456789012"}
        mock_s3 = MagicMock()
        mock_s3.head_bucket.return_value = {}

        def client_side_effect(service, **kw):
            if service == "sts":
                return mock_sts
            elif service == "s3":
                return mock_s3
            return MagicMock()

        mock_client.side_effect = client_side_effect
        mock_infra = create_autospec(SMHPRuntimeManager)
        mock_infra.platform = Platform.SMHP

        with patch(
            "amzn_nova_forge.model.nova_model_customizer.NovaModelCustomizer._init_data_mixing"
        ):
            customizer = NovaModelCustomizer(
                model=Model.NOVA_LITE,
                method=TrainingMethod.SFT_LORA,
                infra=mock_infra,
                data_s3_path=self.data_s3_path,
                output_s3_path=self.output_s3_path,
                data_mixing_enabled=True,
                **kwargs,
            )
        return customizer

    # --- data_s3_path setter ---

    @patch("amzn_nova_forge.model.nova_model_customizer.is_multimodal_data")
    @patch("boto3.client")
    def test_data_s3_path_setter_reruns_autodetect_when_user_intent_none(
        self, mock_client, mock_is_multimodal_data
    ):
        """Changing data_s3_path re-runs auto-detection when _user_is_multimodal is None."""
        mock_is_multimodal_data.return_value = False
        customizer = self._make_smhp_customizer(mock_client, is_multimodal=None)

        mock_is_multimodal_data.return_value = True
        with patch.object(customizer, "_init_data_mixing") as mock_init:
            customizer.data_s3_path = "s3://bucket/new_mm_data.jsonl"
            self.assertTrue(customizer.is_multimodal)
            mock_init.assert_called_once()

    @patch("amzn_nova_forge.model.nova_model_customizer.is_multimodal_data")
    @patch("boto3.client")
    def test_data_s3_path_setter_preserves_explicit_is_multimodal(
        self, mock_client, mock_is_multimodal_data
    ):
        """Changing data_s3_path does NOT re-run auto-detection when is_multimodal was explicit."""
        mock_is_multimodal_data.return_value = False
        customizer = self._make_smhp_customizer(mock_client, is_multimodal=True)

        with patch.object(customizer, "_init_data_mixing") as mock_init:
            customizer.data_s3_path = "s3://bucket/text_data.jsonl"
            # explicit True preserved — auto-detection not called
            mock_is_multimodal_data.assert_not_called()
            self.assertTrue(customizer.is_multimodal)
            # recipe still reloaded
            mock_init.assert_called_once()

    @patch("boto3.client")
    def test_data_s3_path_setter_reloads_recipe_always(self, mock_client):
        """data_s3_path setter always reloads recipe templates when data_mixing_enabled."""
        customizer = self._make_smhp_customizer(mock_client, is_multimodal=True)
        with patch.object(customizer, "_init_data_mixing") as mock_init:
            customizer.data_s3_path = "s3://bucket/other.jsonl"
            mock_init.assert_called_once()

    # --- is_multimodal setter ---

    @patch("boto3.client")
    def test_is_multimodal_setter_explicit_true_reloads_recipe(self, mock_client):
        """Setting is_multimodal=True updates flag and reloads recipe."""
        customizer = self._make_smhp_customizer(mock_client, is_multimodal=False)
        with patch.object(customizer, "_init_data_mixing") as mock_init:
            customizer.is_multimodal = True
            self.assertTrue(customizer.is_multimodal)
            mock_init.assert_called_once()

    @patch("amzn_nova_forge.model.nova_model_customizer.is_multimodal_data")
    @patch("boto3.client")
    def test_is_multimodal_setter_none_reruns_autodetect(
        self, mock_client, mock_is_multimodal_data
    ):
        """Setting is_multimodal=None re-runs auto-detection from current data_s3_path."""
        mock_is_multimodal_data.return_value = True
        customizer = self._make_smhp_customizer(mock_client, is_multimodal=False)
        with patch.object(customizer, "_init_data_mixing"):
            customizer.is_multimodal = None
            mock_is_multimodal_data.assert_called_with(self.data_s3_path)
            self.assertTrue(customizer.is_multimodal)

    @patch("boto3.client")
    def test_is_multimodal_setter_updates_user_intent(self, mock_client):
        """is_multimodal setter updates _user_is_multimodal so data_s3_path setter respects it."""
        customizer = self._make_smhp_customizer(mock_client, is_multimodal=None)
        with patch.object(customizer, "_init_data_mixing"):
            customizer.is_multimodal = True
            self.assertEqual(customizer._user_is_multimodal, True)
            customizer.is_multimodal = None
            self.assertIsNone(customizer._user_is_multimodal)

    @patch("boto3.client")
    def test_is_multimodal_setter_noop_when_data_mixing_disabled(self, mock_client):
        """is_multimodal setter warns and stays False when data_mixing_enabled=False."""
        customizer = self._make_customizer(mock_client, data_mixing_enabled=False)
        customizer.is_multimodal = True
        self.assertFalse(customizer.is_multimodal)

    # --- data_mixing_enabled setter ---

    @patch("amzn_nova_forge.model.nova_model_customizer.is_multimodal_data")
    @patch("boto3.client")
    def test_data_mixing_enabled_setter_enables_and_detects_multimodal(
        self, mock_client, mock_is_multimodal_data
    ):
        """Enabling data_mixing_enabled initializes DataMixing and runs auto-detection."""
        mock_is_multimodal_data.return_value = True
        customizer = self._make_customizer(mock_client, data_mixing_enabled=False)
        self.assertIsNone(customizer.data_mixing)

        with patch.object(customizer, "_init_data_mixing") as mock_init:
            customizer.data_mixing_enabled = True
            self.assertTrue(customizer.data_mixing_enabled)
            self.assertIsNotNone(customizer.data_mixing)
            self.assertTrue(customizer.is_multimodal)
            mock_init.assert_called_once()

    @patch("boto3.client")
    def test_data_mixing_enabled_setter_disables_tears_down(self, mock_client):
        """Disabling data_mixing_enabled resets DataMixing and is_multimodal."""
        customizer = self._make_smhp_customizer(mock_client, is_multimodal=True)
        self.assertTrue(customizer.data_mixing_enabled)

        customizer.data_mixing_enabled = False
        self.assertFalse(customizer.data_mixing_enabled)
        self.assertIsNone(customizer.data_mixing)
        self.assertFalse(customizer.is_multimodal)

    @patch("amzn_nova_forge.model.nova_model_customizer.is_multimodal_data")
    @patch("boto3.client")
    def test_data_mixing_enabled_setter_disable_reenable_preserves_user_intent(
        self, mock_client, mock_is_multimodal_data
    ):
        """Disabling then re-enabling data_mixing restores explicit is_multimodal intent."""
        mock_is_multimodal_data.return_value = False
        customizer = self._make_smhp_customizer(mock_client, is_multimodal=True)

        customizer.data_mixing_enabled = False
        self.assertFalse(customizer.is_multimodal)  # reset on disable

        with patch.object(customizer, "_init_data_mixing"):
            customizer.data_mixing_enabled = True
        # explicit True intent restored
        self.assertTrue(customizer.is_multimodal)
        mock_is_multimodal_data.assert_not_called()

    @patch("boto3.client")
    def test_data_mixing_enabled_setter_noop_if_unchanged(self, mock_client):
        """Setting data_mixing_enabled to its current value is a no-op."""
        customizer = self._make_smhp_customizer(mock_client)
        with patch.object(customizer, "_init_data_mixing") as mock_init:
            customizer.data_mixing_enabled = True  # already True
            mock_init.assert_not_called()

    # --- cross-setter consistency ---

    @patch("amzn_nova_forge.model.nova_model_customizer.is_multimodal_data")
    @patch("boto3.client")
    def test_set_is_multimodal_then_change_data_s3_path_preserves_explicit(
        self, mock_client, mock_is_multimodal_data
    ):
        """Explicit is_multimodal=True survives a data_s3_path change (no re-detection)."""
        mock_is_multimodal_data.return_value = False
        customizer = self._make_smhp_customizer(mock_client, is_multimodal=None)
        # init may have called is_multimodal_data during construction; reset before testing setter behavior
        mock_is_multimodal_data.reset_mock()

        with patch.object(customizer, "_init_data_mixing"):
            customizer.is_multimodal = True  # explicit override
            mock_is_multimodal_data.reset_mock()  # is_multimodal setter with value=True doesn't call detection
            customizer.data_s3_path = "s3://bucket/text_only.jsonl"
            mock_is_multimodal_data.assert_not_called()
            self.assertTrue(customizer.is_multimodal)

    @patch("amzn_nova_forge.model.nova_model_customizer.is_multimodal_data")
    @patch("boto3.client")
    def test_reset_user_intent_to_none_then_change_path_reruns_detection(
        self, mock_client, mock_is_multimodal_data
    ):
        """After resetting is_multimodal=None, changing data_s3_path re-runs detection."""
        mock_is_multimodal_data.return_value = False
        customizer = self._make_smhp_customizer(mock_client, is_multimodal=True)

        with patch.object(customizer, "_init_data_mixing"):
            customizer.is_multimodal = None  # reset to auto-detect
            mock_is_multimodal_data.return_value = True
            customizer.data_s3_path = "s3://bucket/mm_data.jsonl"
            mock_is_multimodal_data.assert_called_with("s3://bucket/mm_data.jsonl")
            self.assertTrue(customizer.is_multimodal)

    # --- model/method/platform setter consistency ---

    @patch("boto3.client")
    def test_model_setter_calls_init_data_mixing_with_new_model(self, mock_client):
        """model setter calls _init_data_mixing with the new model value."""
        customizer = self._make_smhp_customizer(mock_client, is_multimodal=True)
        with patch.object(customizer, "_init_data_mixing") as mock_init:
            customizer.model = Model.NOVA_PRO
            mock_init.assert_called_once_with(
                model=Model.NOVA_PRO,
                method=customizer.method,
                platform=customizer.platform,
            )
            self.assertEqual(customizer.model, Model.NOVA_PRO)

    @patch("boto3.client")
    def test_method_setter_calls_init_data_mixing_with_new_method(self, mock_client):
        """method setter calls _init_data_mixing with the new method value."""
        customizer = self._make_smhp_customizer(mock_client, is_multimodal=True)
        with patch.object(customizer, "_init_data_mixing") as mock_init:
            customizer.method = TrainingMethod.SFT_FULL
            mock_init.assert_called_once_with(
                model=customizer.model,
                method=TrainingMethod.SFT_FULL,
                platform=customizer.platform,
            )
            self.assertEqual(customizer.method, TrainingMethod.SFT_FULL)

    @patch("boto3.client")
    def test_platform_setter_calls_init_data_mixing_with_new_platform(self, mock_client):
        """platform setter calls _init_data_mixing with the new platform value."""
        customizer = self._make_smhp_customizer(mock_client, is_multimodal=True)
        with patch.object(customizer, "_init_data_mixing") as mock_init:
            customizer.platform = Platform.SMHP
            mock_init.assert_called_once_with(
                model=customizer.model,
                method=customizer.method,
                platform=Platform.SMHP,
            )

    @patch("boto3.client")
    def test_model_setter_noop_init_when_data_mixing_disabled(self, mock_client):
        """model setter does NOT call _init_data_mixing when data_mixing_enabled=False."""
        customizer = self._make_customizer(mock_client, data_mixing_enabled=False)
        with patch.object(customizer, "_init_data_mixing") as mock_init:
            customizer.model = Model.NOVA_PRO
            mock_init.assert_not_called()
            self.assertEqual(customizer.model, Model.NOVA_PRO)

    @patch("boto3.client")
    def test_is_multimodal_preserved_across_model_change(self, mock_client):
        """is_multimodal flag is read at _init_data_mixing call time — explicit value survives model change."""
        customizer = self._make_smhp_customizer(mock_client, is_multimodal=True)
        # _init_data_mixing reads self.is_multimodal; verify it's still True after model change
        with patch.object(customizer, "_init_data_mixing", wraps=lambda **kw: None) as mock_init:
            customizer.model = Model.NOVA_PRO
            # is_multimodal was not touched by model setter
            self.assertTrue(customizer.is_multimodal)

    def test_get_logs_without_job_state_raises_value_error(self):
        with self.assertRaises(ValueError) as ctx:
            self.customizer.get_logs()

        self.assertIn(".train()", str(ctx.exception))


class TestTrain(TestNovaModelCustomizer):
    @patch("amzn_nova_forge.recipe.recipe_builder.RecipeBuilder.build_and_validate")
    @patch("uuid.uuid4")
    @patch("boto3.client")
    def test_train_job_name_truncation(self, mock_boto_client, mock_uuid, mock_build_and_validate):
        mock_uuid.return_value = MagicMock()
        mock_uuid.return_value.__str__ = lambda x: "a" * 100  # Very long UUID
        mock_build_and_validate.return_value = (
            "mock_recipe.yaml",
            self.output_s3_path,
            self.data_s3_path,
            "image",
        )
        self.mock_runtime_manager.execute.return_value = "job-789"

        mock_sagemaker = MagicMock()
        mock_sagemaker.describe_training_job.return_value = {
            "CheckpointConfig": {"S3Uri": "s3://test-bucket/checkpoints/model"},
            "OutputDataConfig": {"S3OutputPath": "s3://output-bucket/output"},
        }
        mock_boto_client.return_value = mock_sagemaker

        long_job_name = "b" * 100  # Very long name

        self.customizer.train(job_name=long_job_name)

        call_args = self.mock_runtime_manager.execute.call_args
        job_config = call_args.kwargs["job_config"]
        # UUID is appended and the result is truncated to 63 chars
        self.assertEqual(len(job_config.job_name), 63)
        self.assertTrue(job_config.job_name.startswith("b"))
        # Must not end with a hyphen (SageMaker constraint)
        self.assertTrue(job_config.job_name[-1] != "-")

    @patch("amzn_nova_forge.recipe.recipe_builder.RecipeBuilder.build_and_validate")
    @patch("uuid.uuid4")
    @patch("boto3.client")
    def test_train_sft_basic_success(self, mock_boto_client, mock_uuid, mock_build_and_validate):
        mock_uuid.return_value = MagicMock()
        mock_uuid.return_value.__str__ = lambda x: "test-uuid-1234"
        mock_build_and_validate.return_value = (
            "mock_recipe.yaml",
            self.output_s3_path,
            self.data_s3_path,
            "image",
        )

        mock_sagemaker = MagicMock()
        mock_sagemaker.describe_training_job.return_value = {
            "CheckpointConfig": {"S3Uri": "s3://test-bucket/checkpoints/model"},
            "OutputDataConfig": {"S3OutputPath": "s3://output-bucket/output"},
        }
        mock_boto_client.return_value = mock_sagemaker

        expected_job_id = "job-123"
        self.mock_runtime_manager.execute.return_value = expected_job_id

        result = self.customizer.train(job_name="test-job")

        self.assertIsInstance(result, TrainingResult)
        self.assertEqual(result.method, TrainingMethod.SFT_LORA)
        self.assertEqual(result.job_id, expected_job_id)
        self.assertEqual(
            result.model_artifacts.checkpoint_s3_path,
            "s3://test-bucket/checkpoints/model",
        )
        self.assertEqual(result.model_artifacts.output_s3_path, "s3://output-bucket/output")
        self.assertIsInstance(result.started_time, datetime)

        self.mock_runtime_manager.execute.assert_called_once()
        call_args = self.mock_runtime_manager.execute.call_args
        job_config = call_args.kwargs["job_config"]
        self.assertIn("test-job", job_config.job_name)
        self.assertEqual(job_config.data_s3_path, self.data_s3_path)
        self.assertEqual(job_config.output_s3_path, self.output_s3_path)
        self.assertEqual(job_config.recipe_path, "mock_recipe.yaml")

    @patch("amzn_nova_forge.recipe.recipe_builder.RecipeBuilder.build_and_validate")
    @patch("uuid.uuid4")
    @patch("boto3.client")
    def test_train_rft_basic_success(self, mock_boto_client, mock_uuid, mock_build_and_validate):
        mock_uuid.return_value = MagicMock()
        mock_uuid.return_value.__str__ = lambda x: "test-uuid-1234"
        mock_build_and_validate.return_value = (
            "mock_recipe.yaml",
            self.output_s3_path,
            self.data_s3_path,
            "image",
        )

        mock_sagemaker = MagicMock()
        mock_sagemaker.describe_training_job.return_value = {
            "CheckpointConfig": {"S3Uri": "s3://test-bucket/checkpoints/model"},
            "OutputDataConfig": {"S3OutputPath": "s3://output-bucket/output"},
        }
        mock_boto_client.return_value = mock_sagemaker

        expected_job_id = "job-123"
        self.mock_runtime_manager.execute.return_value = expected_job_id

        self.customizer.method = TrainingMethod.RFT_LORA
        result = self.customizer.train(
            job_name="test-job",
            rft_lambda_arn="arn:aws:lambda:us-east-1:123456789012:function:my-function-name",
        )

        self.assertIsInstance(result, TrainingResult)
        self.assertEqual(result.method, TrainingMethod.RFT_LORA)
        self.assertEqual(result.job_id, expected_job_id)
        self.assertEqual(
            result.model_artifacts.checkpoint_s3_path,
            "s3://test-bucket/checkpoints/model",
        )
        self.assertEqual(result.model_artifacts.output_s3_path, "s3://output-bucket/output")
        self.assertIsInstance(result.started_time, datetime)

        self.mock_runtime_manager.execute.assert_called_once()
        call_args = self.mock_runtime_manager.execute.call_args
        job_config = call_args.kwargs["job_config"]
        self.assertIn("test-job", job_config.job_name)
        self.assertEqual(job_config.data_s3_path, self.data_s3_path)
        self.assertEqual(job_config.output_s3_path, self.output_s3_path)
        self.assertEqual(job_config.recipe_path, "mock_recipe.yaml")

    @patch("amzn_nova_forge.trainer.forge_trainer.ForgeTrainer.__init__", return_value=None)
    @patch("amzn_nova_forge.trainer.forge_trainer.ForgeTrainer.train")
    def test_train_passes_val_check_interval_to_forge_trainer(self, mock_ft_train, mock_ft_init):
        mock_ft_train.return_value = MagicMock()
        self.customizer.train(job_name="test-job", val_check_interval=500)
        init_kwargs = mock_ft_init.call_args.kwargs
        self.assertEqual(init_kwargs["val_check_interval"], 500)

    @patch("amzn_nova_forge.recipe.recipe_builder.RecipeBuilder.build_and_validate")
    def test_train_runtime_manager_failure(self, mock_build_and_validate):
        mock_build_and_validate.return_value = (
            "mock_recipe.yaml",
            self.output_s3_path,
            self.data_s3_path,
            "image",
        )
        self.mock_runtime_manager.execute.side_effect = Exception("Runtime failure")

        with self.assertRaises(Exception) as context:
            self.customizer.train(job_name="test-job")

        self.assertEqual(str(context.exception), "Runtime failure")

    @patch("amzn_nova_forge.recipe.recipe_builder.RecipeBuilder.build_and_validate")
    def test_train_recipe_build_failure(self, mock_build_and_validate):
        mock_build_and_validate.side_effect = Exception("Recipe build failed")

        with self.assertRaises(Exception) as context:
            self.customizer.train(job_name="test-job")

        self.assertIn("Recipe build failed", str(context.exception))

    @patch("amzn_nova_forge.recipe.recipe_builder.RecipeBuilder.build_and_validate")
    def test_train_dry_run(self, mock_build_and_validate):
        mock_build_and_validate.return_value = (
            "mock_recipe.yaml",
            self.output_s3_path,
            self.data_s3_path,
            "image",
        )

        result = self.customizer.train(job_name="test-job", dry_run=True)

        self.assertIsNone(result)
        self.mock_runtime_manager.execute.assert_not_called()

    @patch("amzn_nova_forge.recipe.recipe_builder.RecipeBuilder.build_and_validate")
    @patch("uuid.uuid4")
    @patch("boto3.client")
    def test_train_smhp_basic_success(self, mock_boto_client, mock_uuid, mock_build_and_validate):
        mock_uuid.return_value = MagicMock()
        mock_uuid.return_value.__str__ = lambda x: "test-uuid-smhp"
        mock_build_and_validate.return_value = (
            "mock_recipe.yaml",
            self.output_s3_path,
            self.data_s3_path,
            "image",
        )

        mock_sagemaker = MagicMock()
        mock_boto_client.return_value = mock_sagemaker

        mock_smhp_infra = create_autospec(SMHPRuntimeManager)
        mock_smhp_infra.platform = Platform.SMHP
        mock_smhp_infra.cluster_name = "test-cluster"
        mock_smhp_infra.namespace = "test-namespace"
        mock_smhp_infra.rft_lambda_arn = None

        expected_job_id = "smhp-job-123"
        mock_smhp_infra.execute.return_value = expected_job_id

        with (
            patch("boto3.client") as mock_client,
            patch("amzn_nova_forge.util.recipe.get_hub_recipe_metadata") as mock_get_hub_metadata,
            patch("amzn_nova_forge.util.recipe.download_templates_from_s3") as mock_download_s3,
        ):
            mock_get_hub_metadata.return_value = {
                "SmtjRecipeTemplateS3Uri": "s3://test-bucket/recipe.yaml",
                "SmtjOverrideParamsS3Uri": "s3://test-bucket/overrides.json",
            }
            mock_download_s3.return_value = ({}, {})

            mock_sts = MagicMock()
            mock_sts.get_caller_identity.return_value = {"Account": "123456789012"}
            mock_s3 = MagicMock()
            mock_s3.head_bucket.return_value = {}

            def client_side_effect(service, **kwargs):
                if service == "sts":
                    return mock_sts
                elif service == "s3":
                    return mock_s3
                return MagicMock()

            mock_client.side_effect = client_side_effect

            smhp_customizer = NovaModelCustomizer(
                model=self.model,
                method=self.method,
                infra=mock_smhp_infra,
                data_s3_path=self.data_s3_path,
                output_s3_path=self.output_s3_path,
            )

        result = smhp_customizer.train(job_name="test-smhp-job")

        self.assertIsInstance(result, TrainingResult)
        self.assertEqual(result.method, TrainingMethod.SFT_LORA)
        self.assertEqual(result.job_id, expected_job_id)
        self.assertIsInstance(result.started_time, datetime)

        from amzn_nova_forge.core.result import SMHPTrainingResult

        self.assertIsInstance(result, SMHPTrainingResult)
        self.assertEqual(result.cluster_name, "test-cluster")
        self.assertEqual(result.namespace, "test-namespace")

        mock_smhp_infra.execute.assert_called_once()
        call_args = mock_smhp_infra.execute.call_args
        job_config = call_args.kwargs["job_config"]
        self.assertIn("test-smhp-job", job_config.job_name)
        self.assertEqual(job_config.data_s3_path, self.data_s3_path)
        self.assertEqual(job_config.output_s3_path, self.output_s3_path)


class TestEvaluate(TestNovaModelCustomizer):
    @patch("boto3.client")
    @patch("amzn_nova_forge.recipe.recipe_builder.RecipeBuilder.build_and_validate")
    @patch("uuid.uuid4")
    def test_evaluate_basic_success(self, mock_uuid, mock_build_and_validate, mock_boto_client):
        mock_uuid.return_value = MagicMock()
        mock_uuid.return_value.__str__ = lambda x: "test-eval-uuid"
        mock_build_and_validate.return_value = (
            "mock_eval_recipe.yaml",
            self.output_s3_path,
            self.data_s3_path,
            "image",
        )
        mock_boto_client.return_value = MagicMock()

        expected_job_id = "eval-job-123"
        self.mock_runtime_manager.execute.return_value = expected_job_id

        result = self.customizer.evaluate(
            job_name="test-eval-job",
            eval_task=EvaluationTask.MMLU,
            model_path="s3://test/model",
        )

        self.assertIsInstance(result, SMTJEvaluationResult)
        self.assertEqual(result.job_id, expected_job_id)
        self.assertIsInstance(result.started_time, datetime)
        self.assertTrue(result.eval_output_path.endswith("/output/output.tar.gz"))

        self.mock_runtime_manager.execute.assert_called_once()
        call_args = self.mock_runtime_manager.execute.call_args
        job_config = call_args.kwargs["job_config"]
        self.assertIn("test-eval-job", job_config.job_name)
        self.assertEqual(job_config.data_s3_path, self.data_s3_path)
        self.assertEqual(job_config.output_s3_path, self.output_s3_path)
        self.assertEqual(job_config.recipe_path, "mock_eval_recipe.yaml")
        self.assertEqual(job_config.input_s3_data_type, "S3Prefix")

    @patch("boto3.client")
    @patch("amzn_nova_forge.recipe.recipe_builder.RecipeBuilder.build_and_validate")
    def test_evaluate_runtime_manager_failure(self, mock_build_and_validate, mock_boto_client):
        mock_build_and_validate.return_value = (
            "mock_eval_recipe.yaml",
            self.output_s3_path,
            self.data_s3_path,
            "image",
        )
        mock_boto_client.return_value = MagicMock()
        self.mock_runtime_manager.execute.side_effect = Exception("Eval runtime failure")

        with self.assertRaises(Exception) as context:
            self.customizer.evaluate(
                job_name="test-eval-job",
                eval_task=EvaluationTask.MMLU,
                model_path="s3://test/model",
            )

        self.assertEqual(str(context.exception), "Eval runtime failure")

    @patch("boto3.client")
    @patch("amzn_nova_forge.recipe.recipe_builder.RecipeBuilder.build_and_validate")
    def test_evaluate_dry_run_returns_none(self, mock_build_and_validate, mock_boto_client):
        mock_build_and_validate.return_value = (
            "mock_eval_recipe.yaml",
            self.output_s3_path,
            self.data_s3_path,
            "image",
        )
        mock_boto_client.return_value = MagicMock()

        result = self.customizer.evaluate(
            job_name="test-eval-job",
            eval_task=EvaluationTask.MMLU,
            dry_run=True,
            model_path="s3://test/model",
        )

        self.assertIsNone(result)
        self.mock_runtime_manager.execute.assert_not_called()

    @patch("boto3.client")
    @patch("amzn_nova_forge.recipe.recipe_builder.RecipeBuilder.build_and_validate")
    @patch("uuid.uuid4")
    def test_evaluate_with_processor_config(
        self, mock_uuid, mock_build_and_validate, mock_boto_client
    ):
        mock_uuid.return_value = MagicMock()
        mock_uuid.return_value.__str__ = lambda x: "test-eval-uuid"
        mock_build_and_validate.return_value = (
            "mock_eval_recipe.yaml",
            self.output_s3_path,
            self.data_s3_path,
            "image",
        )
        mock_boto_client.return_value = MagicMock()

        expected_job_id = "eval-job-123"
        self.mock_runtime_manager.execute.return_value = expected_job_id

        processor_config = {
            "lambda_arn": "arn:aws:lambda:us-east-1:123456789012:function:test-func",
            "preprocessing": {"enabled": True},
            "postprocessing": {"enabled": True},
            "aggregation": "average",
        }

        result = self.customizer.evaluate(
            job_name="test-eval-job",
            eval_task=EvaluationTask.MMLU,
            processor=processor_config,
            model_path="s3://test/model",
        )

        self.assertIsInstance(result, SMTJEvaluationResult)
        self.assertEqual(result.job_id, expected_job_id)

    @patch("boto3.client")
    @patch(
        "amzn_nova_forge.recipe.recipe_builder.RecipeBuilder.__init__",
        return_value=None,
    )
    @patch("amzn_nova_forge.recipe.recipe_builder.RecipeBuilder.build_and_validate")
    @patch("uuid.uuid4")
    def test_evaluate_does_not_use_customizer_data_for_builtin_tasks(
        self, mock_uuid, mock_build_and_validate, mock_recipe_init, mock_boto_client
    ):
        """Test that built-in eval tasks (MMLU) don't use customizer's data_s3_path"""
        mock_uuid.return_value = MagicMock()
        mock_uuid.return_value.__str__ = lambda x: "test-eval-uuid"
        mock_build_and_validate.return_value = (
            "mock_eval_recipe.yaml",
            self.output_s3_path,
            None,
            "image",
        )
        mock_boto_client.return_value = MagicMock()

        # Built-in eval tasks should not use it

        expected_job_id = "eval-job-builtin"
        self.mock_runtime_manager.execute.return_value = expected_job_id

        self.customizer.evaluate(
            job_name="test-eval-job",
            eval_task=EvaluationTask.MMLU,  # Built-in task
            model_path="s3://test/model",
        )

        # Verify RecipeBuilder was initialized with None for data_s3_path
        init_call_kwargs = mock_recipe_init.call_args[1]
        self.assertIsNone(init_call_kwargs["data_s3_path"])

    @patch(
        "amzn_nova_forge.recipe.recipe_builder.RecipeBuilder.__init__",
        return_value=None,
    )
    @patch("boto3.client")
    @patch("amzn_nova_forge.recipe.recipe_builder.RecipeBuilder.build_and_validate")
    @patch("uuid.uuid4")
    def test_evaluate_uses_customizer_data_for_byod_tasks(
        self, mock_uuid, mock_build_and_validate, mock_boto_client, mock_recipe_init
    ):
        """Test that BYOD eval tasks (GEN_QA) use customizer's data_s3_path as fallback"""
        mock_uuid.return_value = MagicMock()
        mock_uuid.return_value.__str__ = lambda x: "test-eval-uuid"
        mock_build_and_validate.return_value = (
            "mock_eval_recipe.yaml",
            self.output_s3_path,
            "s3://training-data/train.jsonl",
            "image",
        )
        mock_boto_client.return_value = MagicMock()

        # Customizer has data_s3_path from initialization
        # BYOD eval tasks should use it
        training_data_path = self.data_s3_path  # Use the one from setUp

        expected_job_id = "eval-job-byod"
        self.mock_runtime_manager.execute.return_value = expected_job_id

        self.customizer.evaluate(
            job_name="test-eval-job",
            eval_task=EvaluationTask.GEN_QA,  # BYOD task
            model_path="s3://test/model",
        )

        # Verify RecipeBuilder was initialized with training data path
        init_call_kwargs = mock_recipe_init.call_args[1]
        self.assertEqual(init_call_kwargs["data_s3_path"], training_data_path)

    @patch("boto3.client")
    @patch("amzn_nova_forge.recipe.recipe_builder.RecipeBuilder.build_and_validate")
    @patch("uuid.uuid4")
    def test_evaluate_smhp_basic_success(
        self, mock_uuid, mock_build_and_validate, mock_boto_client
    ):
        mock_uuid.return_value = MagicMock()
        mock_uuid.return_value.__str__ = lambda x: "test-eval-smhp-uuid"
        mock_build_and_validate.return_value = (
            "mock_eval_recipe.yaml",
            self.output_s3_path,
            self.data_s3_path,
            "image",
        )
        mock_boto_client.return_value = MagicMock()

        mock_smhp_infra = create_autospec(SMHPRuntimeManager)
        mock_smhp_infra.platform = Platform.SMHP
        mock_smhp_infra.cluster_name = "test-eval-cluster"
        mock_smhp_infra.namespace = "test-eval-namespace"

        expected_job_id = "smhp-eval-job-456"
        mock_smhp_infra.execute.return_value = expected_job_id

        with patch("boto3.client") as mock_client:
            mock_sts = MagicMock()
            mock_sts.get_caller_identity.return_value = {"Account": "123456789012"}
            mock_s3 = MagicMock()
            mock_s3.head_bucket.return_value = {}

            def client_side_effect(service, **kwargs):
                if service == "sts":
                    return mock_sts
                elif service == "s3":
                    return mock_s3
                return MagicMock()

            mock_client.side_effect = client_side_effect

            with (
                patch(
                    "amzn_nova_forge.util.recipe.get_hub_recipe_metadata"
                ) as mock_get_hub_metadata,
                patch("amzn_nova_forge.util.recipe.download_templates_from_s3") as mock_download_s3,
            ):
                mock_get_hub_metadata.return_value = {
                    "SmtjRecipeTemplateS3Uri": "s3://test-bucket/recipe.yaml",
                    "SmtjOverrideParamsS3Uri": "s3://test-bucket/overrides.json",
                }
                mock_download_s3.return_value = ({}, {})

                smhp_customizer = NovaModelCustomizer(
                    model=self.model,
                    method=self.method,
                    infra=mock_smhp_infra,
                    data_s3_path=self.data_s3_path,
                    output_s3_path=self.output_s3_path,
                )

        result = smhp_customizer.evaluate(
            job_name="test-smhp-eval-job",
            eval_task=EvaluationTask.MMLU,
            model_path="s3://test/model",
        )

        self.assertIsInstance(result, EvaluationResult)
        self.assertEqual(result.job_id, expected_job_id)
        self.assertEqual(result.eval_task, EvaluationTask.MMLU)
        self.assertIsInstance(result.started_time, datetime)

        from amzn_nova_forge.core.result import SMHPEvaluationResult

        self.assertIsInstance(result, SMHPEvaluationResult)
        self.assertEqual(result.cluster_name, "test-eval-cluster")
        self.assertEqual(result.namespace, "test-eval-namespace")
        self.assertTrue(result.eval_output_path.endswith("/eval-result/"))

        mock_smhp_infra.execute.assert_called_once()
        call_args = mock_smhp_infra.execute.call_args
        job_config = call_args.kwargs["job_config"]
        self.assertIn("test-smhp-eval-job", job_config.job_name)
        self.assertEqual(job_config.input_s3_data_type, "S3Prefix")

    @patch("boto3.client")
    @patch(
        "amzn_nova_forge.recipe.recipe_builder.RecipeBuilder.__init__",
        return_value=None,
    )
    @patch("amzn_nova_forge.recipe.recipe_builder.RecipeBuilder.build_and_validate")
    @patch("uuid.uuid4")
    def test_evaluate_passes_evaluation_method_to_recipe_builder(
        self, mock_uuid, mock_build_and_validate, mock_recipe_init, mock_boto_client
    ):
        """Test that evaluate() passes TrainingMethod.EVALUATION to RecipeBuilder, not the customizer's training method."""
        mock_uuid.return_value = MagicMock()
        mock_uuid.return_value.__str__ = lambda x: "test-eval-uuid"
        mock_build_and_validate.return_value = (
            "mock_eval_recipe.yaml",
            self.output_s3_path,
            self.data_s3_path,
            "image",
        )
        mock_boto_client.return_value = MagicMock()

        expected_job_id = "eval-job-method-check"
        self.mock_runtime_manager.execute.return_value = expected_job_id

        self.customizer.evaluate(
            job_name="test-eval-method",
            eval_task=EvaluationTask.MMLU,
            model_path="s3://test/model",
        )

        init_call_kwargs = mock_recipe_init.call_args[1]
        self.assertEqual(init_call_kwargs["method"], TrainingMethod.EVALUATION)


class TestFindPublishedModel(TestNovaModelCustomizer):
    """Tests for find_published_model()."""

    @patch("boto3.client")
    def test_skip_model_reuse_returns_none(self, mock_boto_client):
        result = self.customizer.find_published_model(
            "bedrock", "s3://path/", skip_model_reuse=True
        )
        self.assertIsNone(result)
        mock_boto_client.assert_not_called()

    def test_session_cache_hit(self):
        self.customizer._published_models.add(("bedrock", "arn:bedrock:model/cached", "s3://path/"))
        result = self.customizer.find_published_model("bedrock", "s3://path/")
        self.assertEqual(result, "arn:bedrock:model/cached")

    def test_session_cache_miss_different_path(self):
        self.customizer._published_models.add(
            ("bedrock", "arn:bedrock:model/cached", "s3://other/")
        )
        with patch("boto3.client") as mock_boto_client:
            mock_tagging = MagicMock()
            mock_tagging.get_resources.return_value = {"ResourceTagMappingList": []}
            mock_boto_client.return_value = mock_tagging
            result = self.customizer.find_published_model("bedrock", "s3://path/")
        self.assertIsNone(result)

    @patch("boto3.client")
    def test_bedrock_api_lookup_found(self, mock_boto_client):
        mock_tagging = MagicMock()
        mock_tagging.get_resources.return_value = {
            "ResourceTagMappingList": [
                {"ResourceARN": "arn:aws:bedrock:us-east-1:123456789012:custom-model/found"}
            ]
        }
        mock_bedrock = MagicMock()
        mock_bedrock.meta.region_name = "us-east-1"
        mock_boto_client.side_effect = lambda svc, **kw: (
            mock_tagging if svc == "resourcegroupstaggingapi" else mock_bedrock
        )

        result = self.customizer.find_published_model("bedrock", "s3://path/")
        self.assertEqual(result, "arn:aws:bedrock:us-east-1:123456789012:custom-model/found")
        # Should be cached now
        self.assertIn(
            (
                "bedrock",
                "arn:aws:bedrock:us-east-1:123456789012:custom-model/found",
                "s3://path/",
            ),
            self.customizer._published_models,
        )

    @patch("boto3.client")
    def test_sagemaker_api_lookup_found(self, mock_boto_client):
        mock_tagging = MagicMock()
        mock_tagging.get_resources.return_value = {
            "ResourceTagMappingList": [
                {"ResourceARN": "arn:aws:sagemaker:us-east-1:123456789012:model/found"}
            ]
        }
        mock_sm = MagicMock()
        mock_sm.meta.region_name = "us-east-1"
        mock_boto_client.side_effect = lambda svc, **kw: (
            mock_tagging if svc == "resourcegroupstaggingapi" else mock_sm
        )

        result = self.customizer.find_published_model("sagemaker", "s3://path/")
        self.assertEqual(result, "arn:aws:sagemaker:us-east-1:123456789012:model/found")

    @patch("boto3.client")
    def test_api_lookup_not_found(self, mock_boto_client):
        mock_tagging = MagicMock()
        mock_tagging.get_resources.return_value = {"ResourceTagMappingList": []}
        mock_boto_client.return_value = mock_tagging

        result = self.customizer.find_published_model("bedrock", "s3://path/")
        self.assertIsNone(result)

    @patch("boto3.client")
    def test_api_exception_returns_none(self, mock_boto_client):
        mock_boto_client.side_effect = Exception("connection error")

        result = self.customizer.find_published_model("bedrock", "s3://path/")
        self.assertIsNone(result)


class TestDeploy(TestNovaModelCustomizer):
    def _make_boto_client_dispatcher(self, **service_mocks):
        """Create a boto3.client mock that returns different mocks per service.

        Always includes a resourcegroupstaggingapi mock returning empty results
        (no existing models) unless overridden.
        """
        default_tagging = MagicMock()
        default_tagging.get_resources.return_value = {"ResourceTagMappingList": []}

        mocks = {"resourcegroupstaggingapi": default_tagging, **service_mocks}

        def dispatcher(service_name, **kwargs):
            return mocks.get(service_name, MagicMock())

        return dispatcher

    @patch("boto3.client")
    @patch("amzn_nova_forge.deployer.forge_deployer.create_bedrock_execution_role")
    @patch("amzn_nova_forge.deployer.forge_deployer.monitor_model_create")
    def test_use_deployment_name_if_user_provided(
        self, mock_monitor, mock_bedrock_role_creation, mock_boto_client
    ):
        mock_bedrock = MagicMock()
        mock_bedrock.create_custom_model.return_value = {"modelArn": "test:model:arn"}
        mock_boto_client.side_effect = self._make_boto_client_dispatcher(bedrock=mock_bedrock)

        mock_bedrock.get_custom_model.return_value = {"modelStatus": "ACTIVE"}
        mock_bedrock.create_custom_model_deployment.return_value = {
            "customModelDeploymentArn": "test:on-demand-deployment:arn"
        }
        mock_bedrock_role_creation.return_value = {"Role": {"Arn": "bedrock:role:arn"}}
        mock_monitor.return_value = None

        result = self.customizer.deploy(
            model_artifact_path="s3://test-bucket/model",
            deploy_platform=DeployPlatform.BEDROCK_OD,
            endpoint_name="test-endpoint-name",
        )

        self.assertEqual("test-endpoint-name", result.endpoint.endpoint_name)

    @patch("boto3.client")
    @patch("amzn_nova_forge.deployer.forge_deployer.create_bedrock_execution_role")
    def test_bedrock_role_already_created(self, mock_bedrock_role_creation, mock_boto_client):
        mock_bedrock = MagicMock()
        mock_bedrock.create_custom_model.return_value = {"modelArn": "test:model:arn"}
        mock_boto_client.side_effect = self._make_boto_client_dispatcher(bedrock=mock_bedrock)

        self.endpoint_name = "test-endpoint-name"

        mock_bedrock_role_creation.side_effect = Exception(
            "Failed to find or create the BedrockDeployModelExecutionRole:"
        )

        with self.assertRaises(Exception) as context:
            self.customizer.deploy(
                model_artifact_path="s3://test-bucket/model",
                deploy_platform=DeployPlatform.BEDROCK_PT,
            )
        self.assertIn(
            "Failed to find or create the BedrockDeployModelExecutionRole:",
            str(context.exception),
        )

    @patch("boto3.client")
    @patch("amzn_nova_forge.deployer.forge_deployer.create_bedrock_execution_role")
    @patch("amzn_nova_forge.deployer.forge_deployer.monitor_model_create")
    def test_deploy_bedrock_od_success(
        self, mock_monitor, mock_bedrock_role_creation, mock_boto_client
    ):
        mock_bedrock = MagicMock()
        mock_bedrock.create_custom_model.return_value = {"modelArn": "test:model:arn"}
        mock_boto_client.side_effect = self._make_boto_client_dispatcher(bedrock=mock_bedrock)

        mock_bedrock.get_custom_model.return_value = {"modelStatus": "ACTIVE"}
        mock_bedrock.create_custom_model_deployment.return_value = {
            "customModelDeploymentArn": "test:on-demand-deployment:arn"
        }
        mock_bedrock_role_creation.return_value = {"Role": {"Arn": "bedrock:role:arn"}}
        mock_monitor.return_value = None

        result = self.customizer.deploy(
            model_artifact_path="s3://test-bucket/model",
            deploy_platform=DeployPlatform.BEDROCK_OD,
        )

        mock_bedrock_role_creation.assert_called_once()
        mock_monitor.assert_called_once()
        self.assertEqual(result.endpoint.platform, DeployPlatform.BEDROCK_OD)
        self.assertEqual(result.endpoint.uri, "test:on-demand-deployment:arn")

    @patch("boto3.client")
    @patch("amzn_nova_forge.deployer.forge_deployer.create_bedrock_execution_role")
    @patch("amzn_nova_forge.deployer.forge_deployer.monitor_model_create")
    def test_deploy_bedrock_pt_success(
        self, mock_monitor, mock_bedrock_role_creation, mock_boto_client
    ):
        mock_bedrock = MagicMock()
        mock_bedrock.create_custom_model.return_value = {"modelArn": "test:model:arn"}
        mock_boto_client.side_effect = self._make_boto_client_dispatcher(bedrock=mock_bedrock)

        mock_bedrock.get_custom_model.return_value = {"modelStatus": "ACTIVE"}
        mock_bedrock.create_provisioned_model_throughput.return_value = {
            "provisionedModelArn": "test:provisioned-throughput-deployment:arn"
        }
        mock_bedrock_role_creation.return_value = {"Role": {"Arn": "bedrock:role:arn"}}
        mock_monitor.return_value = None

        result = self.customizer.deploy(
            model_artifact_path="s3://test-bucket/model",
            deploy_platform=DeployPlatform.BEDROCK_PT,
        )

        mock_bedrock_role_creation.assert_called_once()
        mock_monitor.assert_called_once()
        self.assertEqual(result.endpoint.platform, DeployPlatform.BEDROCK_PT)
        self.assertEqual(result.endpoint.uri, "test:provisioned-throughput-deployment:arn")

    @patch("boto3.client")
    @patch("amzn_nova_forge.deployer.forge_deployer.create_sagemaker_execution_role")
    @patch("amzn_nova_forge.deployer.forge_deployer.create_sagemaker_endpoint")
    @patch("amzn_nova_forge.deployer.forge_deployer.create_sagemaker_model")
    @patch(
        "amzn_nova_forge.deployer.forge_deployer.find_sagemaker_model_by_tag",
        return_value=None,
    )
    def test_deploy_sagemaker_success(
        self,
        mock_find_by_tag,
        mock_create_model,
        mock_create_endpoint,
        mock_sagemaker_role_creation,
        mock_boto_client,
    ):
        mock_sagemaker_role_creation.return_value = {"Role": {"Arn": "sagemaker:role:arn"}}
        mock_create_model.return_value = "arn:aws:sagemaker:us-east-1:123456789012:model/test-model"
        mock_create_endpoint.return_value = "sagemaker:endpoint:arn"

        result = self.customizer.deploy(
            model_artifact_path="s3://test-bucket/model",
            deploy_platform=DeployPlatform.SAGEMAKER,
            sagemaker_instance_type="ml.p5.48xlarge",
        )

        mock_sagemaker_role_creation.assert_called_once()
        mock_create_model.assert_called_once()
        mock_create_endpoint.assert_called_once()
        self.assertEqual(result.endpoint.platform, DeployPlatform.SAGEMAKER)
        self.assertEqual(result.endpoint.uri, "sagemaker:endpoint:arn")
        self.assertIsNotNone(result.model_publish)
        self.assertEqual(
            result.model_publish.model_arn,
            "arn:aws:sagemaker:us-east-1:123456789012:model/test-model",
        )

    @patch("boto3.client")
    @patch("amzn_nova_forge.deployer.forge_deployer.create_sagemaker_execution_role")
    @patch("amzn_nova_forge.deployer.forge_deployer.create_sagemaker_endpoint")
    @patch("amzn_nova_forge.deployer.forge_deployer.create_sagemaker_model")
    @patch("amzn_nova_forge.model.nova_model_customizer.resolve_model_checkpoint_path")
    @patch(
        "amzn_nova_forge.deployer.forge_deployer.find_sagemaker_model_by_tag",
        return_value=None,
    )
    def test_deploy_sagemaker_with_job_success(
        self,
        mock_find_by_tag,
        mock_checkpoint_resolution,
        mock_create_model,
        mock_create_endpoint,
        mock_sagemaker_role_creation,
        mock_boto_client,
    ):
        mock_sagemaker = MagicMock()
        mock_boto_client.return_value = mock_sagemaker
        mock_sagemaker_role_creation.return_value = {"Role": {"Arn": "sagemaker:role:arn"}}
        mock_create_model.return_value = (
            "arn:aws:sagemaker:us-east-1:123:model/nova-micro-sft-lora-sagemaker-model"
        )
        mock_create_endpoint.return_value = "sagemaker:endpoint:arn"
        mock_checkpoint_resolution.return_value = "s3://xn---checkpointbucket/ckpt"

        mock_job_result = MagicMock()
        mock_job_result.model_type = Model.NOVA_MICRO

        result = self.customizer.deploy(
            job_result=mock_job_result,
            deploy_platform=DeployPlatform.SAGEMAKER,
            sagemaker_instance_type="ml.p5.48xlarge",
        )

        mock_create_model.assert_called_once_with(
            region="us-east-1",
            model_name="nova-micro-sft-lora-sagemaker-model",
            model_s3_location="s3://xn---checkpointbucket/ckpt/",
            sagemaker_execution_role_arn="sagemaker:role:arn",
            sagemaker_client=mock_sagemaker,
            environment={
                "CONTEXT_LENGTH": "4000",
                "MAX_CONCURRENCY": "1",
            },
            deployment_mode=DeploymentMode.FAIL_IF_EXISTS,
            tags=[
                {
                    "Key": "sagemaker.amazonaws.com/forge/escrow-uri",
                    "Value": "s3://xn---checkpointbucket/ckpt/",
                }
            ],
            model_package_name=None,
        )
        mock_create_endpoint.assert_called_once_with(
            model_name="nova-micro-sft-lora-sagemaker-model",
            endpoint_config_name="nova-micro-sft-lora-sagemaker-config",
            endpoint_name="nova-micro-sft-lora-sagemaker",
            instance_type="ml.p5.48xlarge",
            sagemaker_client=mock_sagemaker,
            initial_instance_count=1,
            deployment_mode=DeploymentMode.FAIL_IF_EXISTS,
        )

        self.assertEqual(result.endpoint.platform, DeployPlatform.SAGEMAKER)
        self.assertEqual(result.endpoint.uri, "sagemaker:endpoint:arn")
        self.assertIsNotNone(result.model_publish)

    @patch("boto3.client")
    @patch("amzn_nova_forge.deployer.forge_deployer.create_bedrock_execution_role")
    @patch("amzn_nova_forge.deployer.forge_deployer.monitor_model_create")
    def test_create_custom_model_failure(
        self, mock_monitor, mock_bedrock_role_creation, mock_boto_client
    ):
        mock_bedrock = MagicMock()
        mock_bedrock.create_custom_model.return_value = {"modelArn": "test:model:arn"}
        mock_boto_client.side_effect = self._make_boto_client_dispatcher(bedrock=mock_bedrock)

        mock_bedrock.get_custom_model.return_value = {"modelStatus": "ACTIVE"}
        mock_bedrock.create_custom_model.side_effect = Exception("Failed to create custom model")
        mock_bedrock.create_custom_model_deployment.return_value = {
            "customModelDeploymentArn": "test:on-demand-deployment:arn"
        }
        mock_bedrock_role_creation.return_value = {"Role": {"Arn": "bedrock:role:arn"}}
        mock_monitor.return_value = None

        with self.assertRaises(Exception) as context:
            self.customizer.deploy(
                model_artifact_path="s3://test-bucket/model",
                deploy_platform=DeployPlatform.BEDROCK_OD,
            )

        self.assertIn("Failed to create custom model", str(context.exception))

    @patch("boto3.client")
    @patch("amzn_nova_forge.deployer.forge_deployer.create_sagemaker_execution_role")
    @patch("amzn_nova_forge.deployer.forge_deployer.create_sagemaker_endpoint")
    @patch("amzn_nova_forge.deployer.forge_deployer.create_sagemaker_model")
    @patch(
        "amzn_nova_forge.deployer.forge_deployer.find_sagemaker_model_by_tag",
        return_value=None,
    )
    def test_deploy_sagemaker_failure(
        self,
        mock_find_by_tag,
        mock_create_model,
        mock_create_endpoint,
        mock_sagemaker_role_creation,
        mock_boto_client,
    ):
        mock_sagemaker_role_creation.return_value = {"Role": {"Arn": "sagemaker:role:arn"}}
        mock_create_model.return_value = "arn:aws:sagemaker:us-east-1:123456789012:model/test-model"
        mock_create_endpoint.side_effect = Exception("Failed to create deployment")

        with self.assertRaises(Exception) as context:
            self.customizer.deploy(
                model_artifact_path="s3://test-bucket/model",
                deploy_platform=DeployPlatform.SAGEMAKER,
                sagemaker_instance_type="ml.p5.48xlarge",
            )
        self.assertIn("Failed to create deployment", str(context.exception))
        mock_create_model.assert_called_once()

    @patch("boto3.client")
    @patch("amzn_nova_forge.deployer.forge_deployer.create_sagemaker_execution_role")
    @patch("amzn_nova_forge.deployer.forge_deployer.create_sagemaker_endpoint")
    @patch("amzn_nova_forge.deployer.forge_deployer.create_sagemaker_model")
    def test_deploy_to_sagemaker_with_model_deploy_result(
        self,
        mock_create_model,
        mock_create_endpoint,
        mock_sagemaker_role_creation,
        mock_boto_client,
    ):
        """deploy_to_sagemaker with model_deploy_result skips model creation."""
        mock_sagemaker_role_creation.return_value = {"Role": {"Arn": "sagemaker:role:arn"}}
        mock_create_model.return_value = "arn:aws:sagemaker:us-east-1:123456789012:model/test-model"
        mock_create_endpoint.return_value = "sagemaker:endpoint:arn"
        mock_boto_client.side_effect = self._make_boto_client_dispatcher()

        model_result = ModelDeployResult(
            model_arn="arn:aws:sagemaker:us-east-1:123:model/existing-model",
            model_name="existing-model",
            escrow_uri="s3://escrow/path/",
            created_at=datetime.now(timezone.utc),
        )

        result = self.customizer.deploy_to_sagemaker(
            model_deploy_result=model_result,
            instance_type="ml.g5.12xlarge",
            endpoint_name="test-ep",
        )

        mock_create_model.assert_not_called()
        mock_create_endpoint.assert_called_once()
        self.assertEqual(result.endpoint.uri, "sagemaker:endpoint:arn")
        self.assertIsNotNone(result.model_publish)

    @patch("boto3.client")
    @patch("amzn_nova_forge.deployer.forge_deployer.create_sagemaker_execution_role")
    @patch("amzn_nova_forge.deployer.forge_deployer.create_sagemaker_endpoint")
    @patch("amzn_nova_forge.deployer.forge_deployer.create_sagemaker_model")
    def test_deploy_to_sagemaker_endpoint_failure_shows_retry_hint(
        self,
        mock_create_model,
        mock_create_endpoint,
        mock_sagemaker_role_creation,
        mock_boto_client,
    ):
        """Endpoint failure error message includes model ARN and retry hint."""
        mock_sagemaker_role_creation.return_value = {"Role": {"Arn": "sagemaker:role:arn"}}
        mock_create_model.return_value = "arn:aws:sagemaker:us-east-1:123456789012:model/test-model"
        mock_create_endpoint.side_effect = Exception("Endpoint creation failed")
        mock_boto_client.side_effect = self._make_boto_client_dispatcher()

        with self.assertRaises(RuntimeError) as context:
            self.customizer.deploy_to_sagemaker(
                model_artifact_path="s3://test-bucket/model/",
                instance_type="ml.g5.12xlarge",
            )
        err = str(context.exception)
        self.assertIn("last_model_publish", err)
        self.assertIn("SageMaker endpoint creation failed", err)

    def test_deploy_to_bedrock_with_model_deploy_result(self):
        """deploy_to_bedrock with model_deploy_result uses its ARN."""

        mock_bedrock = MagicMock()
        mock_bedrock.create_custom_model_deployment.return_value = {
            "customModelDeploymentArn": "test:deployment:arn"
        }

        with patch("boto3.client") as mock_boto_client:
            mock_boto_client.side_effect = self._make_boto_client_dispatcher(bedrock=mock_bedrock)

            model_result = ModelDeployResult(
                model_arn="arn:aws:bedrock:us-east-1:123:custom-model/my-model",
                model_name="my-model",
                escrow_uri="s3://escrow/path/",
                created_at=datetime.now(timezone.utc),
            )

            result = self.customizer.deploy_to_bedrock(
                model_deploy_result=model_result,
                endpoint_name="test-ep",
            )

            self.assertEqual(result.model_publish, model_result)
            self.assertEqual(result.endpoint.uri, "test:deployment:arn")

    def test_deploy_to_bedrock_no_arn_raises(self):
        """deploy_to_bedrock with no ARN source raises ValueError."""
        with self.assertRaises(ValueError) as ctx:
            self.customizer.deploy_to_bedrock()
        self.assertIn("No model ARN", str(ctx.exception))

    def test_deploy_to_sagemaker_no_input_raises(self):
        """deploy_to_sagemaker with neither model_deploy_result nor model_artifact_path raises."""
        with self.assertRaises(ValueError):
            self.customizer.deploy_to_sagemaker(instance_type="ml.g5.12xlarge")

    @patch("boto3.client")
    @patch("amzn_nova_forge.deployer.forge_deployer.create_sagemaker_execution_role")
    @patch("amzn_nova_forge.deployer.forge_deployer.create_sagemaker_endpoint")
    @patch("amzn_nova_forge.deployer.forge_deployer.create_sagemaker_model")
    def test_deploy_to_sagemaker_skip_model_reuse(
        self,
        mock_create_model,
        mock_create_endpoint,
        mock_sagemaker_role_creation,
        mock_boto_client,
    ):
        """deploy_to_sagemaker with skip_model_reuse=True skips tag-based model discovery."""
        mock_sagemaker_role_creation.return_value = {"Role": {"Arn": "sagemaker:role:arn"}}
        mock_create_model.return_value = "arn:aws:sagemaker:us-east-1:123456789012:model/test-model"
        mock_create_endpoint.return_value = "sagemaker:endpoint:arn"
        mock_boto_client.side_effect = self._make_boto_client_dispatcher()

        with patch(
            "amzn_nova_forge.deployer.forge_deployer.find_sagemaker_model_by_tag"
        ) as mock_find_tag:
            result = self.customizer.deploy_to_sagemaker(
                model_artifact_path="s3://test-bucket/model/",
                instance_type="ml.g5.12xlarge",
                skip_model_reuse=True,
            )

        # find_sagemaker_model_by_tag should NOT be called when skip_model_reuse=True
        mock_find_tag.assert_not_called()
        mock_create_model.assert_called_once()
        self.assertEqual(result.endpoint.uri, "sagemaker:endpoint:arn")

    @patch("boto3.client")
    def test_deploy_to_bedrock_waits_on_creating_model(self, mock_boto_client):
        """deploy_to_bedrock waits when model status is Creating."""
        mock_bedrock = MagicMock()
        mock_bedrock.get_custom_model.side_effect = [
            {"modelStatus": "Creating"},  # first call from status check
            {"modelStatus": "Active"},  # second call from wait_for_model_ready
        ]
        mock_bedrock.create_custom_model_deployment.return_value = {
            "customModelDeploymentArn": "test:deployment:arn"
        }
        mock_boto_client.side_effect = self._make_boto_client_dispatcher(bedrock=mock_bedrock)

        model_result = ModelDeployResult(
            model_arn="arn:aws:bedrock:us-east-1:123456789012:custom-model/my-model",
            model_name="my-model",
            escrow_uri="s3://escrow/path/",
            created_at=datetime.now(timezone.utc),
        )

        result = self.customizer.deploy_to_bedrock(
            model_deploy_result=model_result,
            endpoint_name="test-ep",
        )
        self.assertEqual(result.endpoint.uri, "test:deployment:arn")

    @patch("boto3.client")
    def test_deploy_to_bedrock_rejects_failed_model(self, mock_boto_client):
        """deploy_to_bedrock raises ValueError when model status is Failed."""
        mock_bedrock = MagicMock()
        mock_bedrock.get_custom_model.return_value = {"modelStatus": "Failed"}
        mock_boto_client.side_effect = self._make_boto_client_dispatcher(bedrock=mock_bedrock)

        model_result = ModelDeployResult(
            model_arn="arn:aws:bedrock:us-east-1:123456789012:custom-model/my-model",
            model_name="my-model",
            escrow_uri="s3://escrow/path/",
            created_at=datetime.now(timezone.utc),
        )

        with self.assertRaises(ValueError) as ctx:
            self.customizer.deploy_to_bedrock(
                model_deploy_result=model_result,
                endpoint_name="test-ep",
            )
        self.assertIn("Failed", str(ctx.exception))

    @patch("boto3.client")
    @patch("amzn_nova_forge.deployer.forge_deployer.create_bedrock_execution_role")
    @patch("amzn_nova_forge.deployer.forge_deployer.monitor_model_create")
    def test_auto_generate_deployment_name_if_not_provided(
        self, mock_monitor, mock_bedrock_role_creation, mock_boto_client
    ):
        mock_bedrock = MagicMock()
        mock_bedrock.create_custom_model.return_value = {"modelArn": "test:model:arn"}
        mock_boto_client.side_effect = self._make_boto_client_dispatcher(bedrock=mock_bedrock)

        mock_bedrock.get_custom_model.return_value = {"modelStatus": "ACTIVE"}
        mock_bedrock.create_custom_model_deployment.return_value = {
            "customModelDeploymentArn": "test:on-demand-deployment:arn"
        }
        mock_bedrock_role_creation.return_value = {"Role": {"Arn": "bedrock:role:arn"}}
        mock_monitor.return_value = None

        result = self.customizer.deploy(
            model_artifact_path="s3://test-bucket/model",
            deploy_platform=DeployPlatform.BEDROCK_OD,
        )

        self.assertEqual(
            "nova-micro-sft-lora-us-east-1",
            result.endpoint.endpoint_name,
        )

    @patch("boto3.client")
    @patch("amzn_nova_forge.deployer.forge_deployer.create_bedrock_execution_role")
    @patch("amzn_nova_forge.deployer.forge_deployer.monitor_model_create")
    @patch("amzn_nova_forge.deployer.forge_deployer.check_existing_deployment")
    def test_deploy_fails_when_endpoint_exists_and_fail_if_exists_mode(
        self,
        mock_check_existing,
        mock_monitor,
        mock_bedrock_role_creation,
        mock_boto_client,
    ):
        """Test that deploy fails when endpoint exists and deployment_mode=FAIL_IF_EXISTS (default)"""
        # Mock existing deployment found
        mock_check_existing.return_value = (
            "arn:aws:bedrock:us-east-1:123456789012:custom-model-deployment/existing"
        )

        with self.assertRaises(Exception) as context:
            self.customizer.deploy(
                model_artifact_path="s3://test-bucket/model.tar.gz",
                endpoint_name="test-endpoint",
            )

        error_msg = str(context.exception)
        self.assertIn("already exists", error_msg)
        self.assertIn("Change deployment_mode", error_msg)

        # Verify we checked for existing deployment
        mock_check_existing.assert_called_once_with(
            "test-endpoint", DeployPlatform.BEDROCK_OD, region="us-east-1"
        )

        # Verify we didn't proceed with deployment
        mock_bedrock_role_creation.assert_not_called()
        mock_monitor.assert_not_called()

    @patch("boto3.client")
    @patch("amzn_nova_forge.deployer.forge_deployer.create_bedrock_execution_role")
    @patch("amzn_nova_forge.deployer.forge_deployer.monitor_model_create")
    @patch("amzn_nova_forge.deployer.forge_deployer.check_existing_deployment")
    @patch("amzn_nova_forge.deployer.forge_deployer.update_provisioned_throughput_model")
    @patch("amzn_nova_forge.validation.validator.Validator._validate_calling_role_permissions")
    def test_deploy_pt_update_success_uses_existing_arn(
        self,
        mock_validate_perms,
        mock_update_pt,
        mock_check_existing,
        mock_monitor,
        mock_bedrock_role_creation,
        mock_boto_client,
    ):
        """Test PT update success: uses existing ARN, no new deployment created"""
        existing_arn = "test:existing-pt:arn"
        mock_check_existing.return_value = existing_arn
        mock_validate_perms.return_value = None
        mock_update_pt.return_value = None  # Success

        mock_bedrock = MagicMock()
        mock_bedrock.create_custom_model.return_value = {"modelArn": "test:new-model:arn"}
        mock_boto_client.side_effect = self._make_boto_client_dispatcher(bedrock=mock_bedrock)
        mock_bedrock_role_creation.return_value = {"Role": {"Arn": "test:role:arn"}}

        with (
            patch("amzn_nova_forge.util.recipe.get_hub_recipe_metadata") as mock_get_hub_metadata_2,
            patch("amzn_nova_forge.util.recipe.download_templates_from_s3") as mock_download_s3_2,
        ):
            mock_get_hub_metadata_2.return_value = {
                "SmtjRecipeTemplateS3Uri": "s3://test-bucket/recipe.yaml",
                "SmtjOverrideParamsS3Uri": "s3://test-bucket/overrides.json",
            }
            mock_download_s3_2.return_value = ({}, {})

            pt_customizer = NovaModelCustomizer(
                model=Model.NOVA_LITE,
                method=TrainingMethod.SFT_LORA,
                infra=self.mock_runtime_manager,
                data_s3_path="s3://test-bucket/data.jsonl",
                deployment_mode=DeploymentMode.UPDATE_IF_EXISTS,
            )

        result = pt_customizer.deploy(
            model_artifact_path="s3://test-bucket/model.tar.gz",
            deploy_platform=DeployPlatform.BEDROCK_PT,
            unit_count=10,
            endpoint_name="test-pt-endpoint",
        )

        # Verify update was called
        mock_update_pt.assert_called_once_with(
            existing_arn, "test:new-model:arn", "test-pt-endpoint", region="us-east-1"
        )

        # Verify NO new deployment created
        mock_bedrock.create_provisioned_model_throughput.assert_not_called()

        # Verify result uses existing ARN
        self.assertEqual(result.endpoint.uri, existing_arn)

    @patch("boto3.client")
    @patch("amzn_nova_forge.deployer.forge_deployer.create_bedrock_execution_role")
    @patch("amzn_nova_forge.deployer.forge_deployer.monitor_model_create")
    @patch("amzn_nova_forge.deployer.forge_deployer.check_existing_deployment")
    def test_deploy_succeeds_when_no_existing_endpoint(
        self,
        mock_check_existing,
        mock_monitor,
        mock_bedrock_role_creation,
        mock_boto_client,
    ):
        """Test that deploy succeeds normally when no existing endpoint found"""
        # Mock no existing deployment found
        mock_check_existing.return_value = None

        # Mock bedrock client and responses
        mock_bedrock = MagicMock()
        mock_bedrock.create_custom_model.return_value = {"modelArn": "test:model:arn"}
        mock_bedrock.create_custom_model_deployment.return_value = {
            "customModelDeploymentArn": "test:new-deployment:arn"
        }
        mock_boto_client.side_effect = self._make_boto_client_dispatcher(bedrock=mock_bedrock)

        # Mock role creation
        mock_bedrock_role_creation.return_value = {"Role": {"Arn": "bedrock:role:arn"}}
        mock_monitor.return_value = None

        result = self.customizer.deploy(
            model_artifact_path="s3://test-bucket/model.tar.gz",
            endpoint_name="new-endpoint",
        )

        # Verify we checked for existing deployment (pre-flight + deploy)
        self.assertEqual(mock_check_existing.call_count, 2)
        mock_check_existing.assert_called_with(
            "new-endpoint", DeployPlatform.BEDROCK_OD, region="us-east-1"
        )

        # Verify deployment proceeded successfully
        mock_bedrock_role_creation.assert_called_once()
        mock_monitor.assert_called_once()
        self.assertEqual(result.endpoint.endpoint_name, "new-endpoint")
        self.assertEqual(result.endpoint.uri, "test:new-deployment:arn")

    @patch("boto3.client")
    @patch("amzn_nova_forge.deployer.forge_deployer.create_bedrock_execution_role")
    @patch("amzn_nova_forge.deployer.forge_deployer.monitor_model_create")
    @patch(
        "amzn_nova_forge.model.nova_model_customizer_util.extract_checkpoint_path_from_job_output"
    )
    def test_deploy_with_job_result(
        self, mock_extract, mock_monitor, mock_bedrock_role_creation, mock_boto_client
    ):
        """Test deploy with job_result parameter"""
        mock_bedrock = MagicMock()
        mock_bedrock.create_custom_model.return_value = {"modelArn": "test:model:arn"}
        mock_bedrock.create_custom_model_deployment.return_value = {
            "customModelDeploymentArn": "test:deployment:arn"
        }
        mock_boto_client.side_effect = self._make_boto_client_dispatcher(bedrock=mock_bedrock)
        mock_bedrock_role_creation.return_value = {"Role": {"Arn": "bedrock:role:arn"}}
        mock_monitor.return_value = None
        mock_extract.return_value = "s3://extracted/checkpoint/path"

        # Mock job result
        mock_job_result = MagicMock()
        mock_job_result.model_artifacts.output_s3_path = "s3://output/path"
        mock_job_result.job_id = "test-job-123"
        mock_job_result.get_job_status.return_value = (JobStatus.COMPLETED, "Completed")

        result = self.customizer.deploy(job_result=mock_job_result)

        mock_extract.assert_called_once_with(
            output_s3_path="s3://output/path",
            job_result=mock_job_result,
        )
        self.assertEqual(result.endpoint.platform, DeployPlatform.BEDROCK_OD)

    @patch("boto3.client")
    @patch("amzn_nova_forge.deployer.forge_deployer.create_bedrock_execution_role")
    @patch("amzn_nova_forge.deployer.forge_deployer.monitor_model_create")
    @patch(
        "amzn_nova_forge.model.nova_model_customizer_util.extract_checkpoint_path_from_job_output"
    )
    def test_deploy_with_last_job_id(
        self, mock_extract, mock_monitor, mock_bedrock_role_creation, mock_boto_client
    ):
        """Test deploy using last job ID when no parameters provided"""
        mock_bedrock = MagicMock()
        mock_bedrock.create_custom_model.return_value = {"modelArn": "test:model:arn"}
        mock_bedrock.create_custom_model_deployment.return_value = {
            "customModelDeploymentArn": "test:deployment:arn"
        }
        mock_boto_client.side_effect = self._make_boto_client_dispatcher(bedrock=mock_bedrock)
        mock_bedrock_role_creation.return_value = {"Role": {"Arn": "bedrock:role:arn"}}
        mock_monitor.return_value = None
        mock_extract.return_value = "s3://extracted/checkpoint/path"

        # Set last job ID
        self.customizer.job_id = "last-job-456"

        result = self.customizer.deploy()

        mock_extract.assert_called_once_with(
            output_s3_path=self.customizer.output_s3_path, job_id="last-job-456"
        )
        self.assertEqual(result.endpoint.platform, DeployPlatform.BEDROCK_OD)

    @patch("boto3.client")
    def test_deploy_no_checkpoint_path_available(self, mock_boto_client):
        """Test deploy fails when no checkpoint path can be determined"""
        with self.assertRaises(Exception) as context:
            self.customizer.deploy()

        self.assertIn("No model path provided", str(context.exception))

    @patch("boto3.client")
    @patch(
        "amzn_nova_forge.model.nova_model_customizer_util.extract_checkpoint_path_from_job_output"
    )
    def test_deploy_extraction_fails(self, mock_extract, mock_boto_client):
        """Test deploy fails when checkpoint extraction fails"""
        mock_extract.side_effect = Exception("Extraction failed")

        mock_job_result = MagicMock()
        mock_job_result.model_artifacts.output_s3_path = "s3://output/path"
        mock_job_result.job_id = "test-job-123"

        with self.assertRaises(Exception) as context:
            self.customizer.deploy(job_result=mock_job_result)

        self.assertIn("Extraction failed", str(context.exception))

    @patch("boto3.client")
    @patch("amzn_nova_forge.deployer.forge_deployer.monitor_model_create")
    def test_user_role_does_not_attach_policies_by_default(self, mock_monitor, mock_boto_client):
        """When user passes execution_role_name, the role is looked up as-is — no create/attach."""
        mock_iam = MagicMock()
        mock_bedrock = MagicMock()
        mock_bedrock.create_custom_model.return_value = {"modelArn": "test:model:arn"}
        mock_bedrock.create_custom_model_deployment.return_value = {
            "customModelDeploymentArn": "test:deployment:arn"
        }
        mock_iam.get_role.return_value = {"Role": {"Arn": "user:role:arn"}}
        mock_boto_client.side_effect = self._make_boto_client_dispatcher(
            iam=mock_iam, bedrock=mock_bedrock
        )
        mock_monitor.return_value = None

        self.customizer.deploy(
            model_artifact_path="s3://test-bucket/model",
            deploy_platform=DeployPlatform.BEDROCK_OD,
            execution_role_name="my-existing-role",
        )

        mock_iam.get_role.assert_called_once_with(RoleName="my-existing-role")
        mock_iam.create_policy.assert_not_called()
        mock_iam.attach_role_policy.assert_not_called()

    @patch("boto3.client")
    @patch("amzn_nova_forge.deployer.forge_deployer.create_bedrock_execution_role")
    @patch("amzn_nova_forge.deployer.forge_deployer.monitor_model_create")
    def test_default_role_always_manages_policies(
        self, mock_monitor, mock_bedrock_role_creation, mock_boto_client
    ):
        """When no execution_role_name is given, the SDK-managed default role is created/managed."""
        mock_bedrock = MagicMock()
        mock_bedrock.create_custom_model.return_value = {"modelArn": "test:model:arn"}
        mock_bedrock.create_custom_model_deployment.return_value = {
            "customModelDeploymentArn": "test:deployment:arn"
        }
        mock_iam = MagicMock()
        mock_boto_client.side_effect = self._make_boto_client_dispatcher(
            bedrock=mock_bedrock, iam=mock_iam
        )
        mock_bedrock_role_creation.return_value = {"Role": {"Arn": "default:role:arn"}}
        mock_monitor.return_value = None

        self.customizer.deploy(
            model_artifact_path="s3://test-bucket/model",
            deploy_platform=DeployPlatform.BEDROCK_OD,
        )

        mock_bedrock_role_creation.assert_called_once_with(
            iam_client=mock_iam,
            role_name=BEDROCK_EXECUTION_ROLE_NAME,
        )

    @patch("boto3.client")
    def test_user_role_not_found_raises_clear_error(self, mock_boto_client):
        """When user-specified role doesn't exist, raise a clear error."""
        mock_iam = MagicMock()
        mock_iam.get_role.side_effect = Exception("IAM role 'nonexistent-role' does not exist.")
        mock_boto_client.side_effect = self._make_boto_client_dispatcher(iam=mock_iam)

        with self.assertRaises(Exception) as context:
            self.customizer.deploy(
                model_artifact_path="s3://test-bucket/model",
                deploy_platform=DeployPlatform.BEDROCK_OD,
                execution_role_name="nonexistent-role",
            )

        self.assertIn("does not exist", str(context.exception))

    @patch("boto3.client")
    @patch("amzn_nova_forge.deployer.forge_deployer.create_sagemaker_endpoint")
    @patch("amzn_nova_forge.deployer.forge_deployer.create_sagemaker_model")
    def test_sagemaker_user_role_does_not_attach_policies_by_default(
        self, mock_create_model, mock_create_endpoint, mock_boto_client
    ):
        """For SageMaker, user-specified role is looked up as-is — no create/attach."""
        mock_iam = MagicMock()
        mock_iam.get_role.return_value = {"Role": {"Arn": "sagemaker:user:role:arn"}}
        mock_create_model.return_value = "arn:aws:sagemaker:us-east-1:123:model/test"
        mock_create_endpoint.return_value = "sagemaker:endpoint:arn"
        mock_boto_client.side_effect = self._make_boto_client_dispatcher(iam=mock_iam)

        self.customizer.deploy(
            model_artifact_path="s3://test-bucket/model",
            deploy_platform=DeployPlatform.SAGEMAKER,
            execution_role_name="my-sagemaker-role",
            sagemaker_instance_type="ml.p5.48xlarge",
        )

        mock_iam.get_role.assert_called_once_with(RoleName="my-sagemaker-role")
        mock_iam.create_policy.assert_not_called()
        mock_iam.attach_role_policy.assert_not_called()

    @patch("boto3.client")
    @patch("amzn_nova_forge.recipe.recipe_builder.RecipeBuilder.build_and_validate")
    @patch("uuid.uuid4")
    def test_batch_inference_basic_success(
        self, mock_uuid, mock_build_and_validate, mock_boto_client
    ):
        mock_uuid.return_value = MagicMock()
        mock_uuid.return_value.__str__ = lambda x: "test-inference-uuid"
        mock_build_and_validate.return_value = (
            "mock_inference_recipe.yaml",
            self.output_s3_path,
            self.data_s3_path,
            "image",
        )
        mock_boto_client.return_value = MagicMock()

        expected_job_id = "inference-job-123"
        self.mock_runtime_manager.execute.return_value = expected_job_id

        result = self.customizer.batch_inference(
            job_name="test-inference-job",
            input_path="s3://test-bucket/model-s3-path",
            output_s3_path="s3://test-bucket/output",
        )

        self.assertIsInstance(result, SMTJBatchInferenceResult)
        self.assertEqual(result.job_id, expected_job_id)
        self.assertIsInstance(result.started_time, datetime)
        self.assertTrue(result.inference_output_path.endswith("/output/output.tar.gz"))

        self.mock_runtime_manager.execute.assert_called_once()
        call_args = self.mock_runtime_manager.execute.call_args
        job_config = call_args.kwargs["job_config"]
        self.assertIn("test-inference-job", job_config.job_name)
        self.assertEqual(job_config.data_s3_path, self.data_s3_path)
        self.assertEqual(job_config.output_s3_path, self.output_s3_path)
        self.assertEqual(job_config.recipe_path, "mock_inference_recipe.yaml")
        self.assertEqual(job_config.input_s3_data_type, "S3Prefix")

    @patch("boto3.client")
    @patch("amzn_nova_forge.recipe.recipe_builder.RecipeBuilder.build_and_validate")
    def test_batch_inference_runtime_manager_failure(
        self, mock_build_and_validate, mock_boto_client
    ):
        mock_build_and_validate.return_value = (
            "mock_inference_recipe.yaml",
            self.output_s3_path,
            self.data_s3_path,
            "image",
        )
        mock_boto_client.return_value = MagicMock()
        self.mock_runtime_manager.execute.side_effect = Exception("Inference runtime failure")

        with self.assertRaises(Exception) as context:
            self.customizer.batch_inference(
                job_name="test-inference-job",
                input_path="s3://test-bucket/input",
                output_s3_path="s3://test-bucket/inference-data",
            )

        self.assertEqual(str(context.exception), "Inference runtime failure")

    @patch("boto3.client")
    @patch("amzn_nova_forge.recipe.recipe_builder.RecipeBuilder.build_and_validate")
    def test_batch_inference_dry_run_returns_none(self, mock_build_and_validate, mock_boto_client):
        mock_build_and_validate.return_value = (
            "mock_inference_recipe.yaml",
            self.output_s3_path,
            self.data_s3_path,
            "image",
        )
        mock_boto_client.return_value = MagicMock()

        result = self.customizer.batch_inference(
            job_name="test-inference-job",
            input_path="s3://test-bucket/input",
            output_s3_path="s3://test-bucket/inference-data",
            dry_run=True,
        )

        self.assertIsNone(result)
        self.mock_runtime_manager.execute.assert_not_called()


class TestPlatformValidation(TestNovaModelCustomizer):
    """Tests for platform compatibility validation in evaluate() and batch_inference()"""

    def setUp(self):
        super().setUp()
        # Use NOVA_LITE_2 for platform tests (supports both evaluate and batch_inference)
        self.customizer.model = Model.NOVA_LITE_2

    def test_customizer_attributes(self):
        """Override parent test to check NOVA_LITE_2"""
        self.assertEqual(self.customizer.model, Model.NOVA_LITE_2)
        self.assertEqual(self.customizer.method, TrainingMethod.SFT_LORA)
        self.assertEqual(self.customizer.data_s3_path, self.data_s3_path)
        self.assertEqual(self.customizer.output_s3_path, self.output_s3_path)
        self.assertEqual(self.customizer.region, "us-east-1")

    def test_model_setter_getter(self):
        """Override parent test - temporarily set to NOVA_MICRO to test parent logic"""
        # Temporarily set to NOVA_MICRO to test the parent's setter/getter logic
        self.customizer.model = Model.NOVA_MICRO

        # Run parent test logic
        super().test_model_setter_getter()

        # Restore to NOVA_LITE_2 for other tests in this class
        self.customizer.model = Model.NOVA_LITE_2

    @patch("boto3.client")
    @patch("amzn_nova_forge.model.nova_model_customizer.resolve_model_checkpoint_path")
    def test_evaluate_platform_mismatch_smhp_checkpoint_smtj_execution(
        self, mock_resolve, mock_boto_client
    ):
        """Test that SMHP checkpoint on SMTJ execution raises ValueError"""
        mock_boto_client.return_value = MagicMock()
        mock_resolve.return_value = "s3://customer-escrow-123-hp-abc/checkpoint"

        with self.assertRaises(ValueError) as context:
            self.customizer.evaluate(job_name="test-eval", eval_task=EvaluationTask.MMLU)

        error_msg = str(context.exception)
        self.assertIn("Platform mismatch", error_msg)
        self.assertIn("SMHP", error_msg)
        self.assertIn("SMTJ", error_msg)

    @patch("boto3.client")
    @patch("amzn_nova_forge.model.nova_model_customizer.resolve_model_checkpoint_path")
    def test_evaluate_platform_mismatch_smtj_checkpoint_smhp_execution(
        self, mock_resolve, mock_boto_client
    ):
        """Test that SMTJ checkpoint on SMHP execution raises ValueError"""
        mock_boto_client.return_value = MagicMock()
        mock_resolve.return_value = "s3://customer-escrow-123-smtj-abc/checkpoint"

        # Create SMHP customizer
        mock_smhp_infra = create_autospec(SMHPRuntimeManager)
        mock_smhp_infra.platform = Platform.SMHP
        mock_smhp_infra.cluster_name = "test-cluster"
        mock_smhp_infra.namespace = "test-ns"
        self.customizer.infra = mock_smhp_infra
        self.customizer.platform = Platform.SMHP

        with self.assertRaises(ValueError) as context:
            self.customizer.evaluate(job_name="test-eval", eval_task=EvaluationTask.MMLU)

        error_msg = str(context.exception)
        self.assertIn("Platform mismatch", error_msg)
        self.assertIn("SMTJ", error_msg)
        self.assertIn("SMHP", error_msg)

    @patch("boto3.client")
    @patch("amzn_nova_forge.recipe.recipe_builder.RecipeBuilder.build_and_validate")
    @patch("amzn_nova_forge.model.nova_model_customizer.resolve_model_checkpoint_path")
    @patch("uuid.uuid4")
    def test_evaluate_platform_match_smtj(
        self, mock_uuid, mock_resolve, mock_build, mock_boto_client
    ):
        """Test that SMTJ checkpoint on SMTJ execution succeeds"""
        mock_uuid.return_value = MagicMock()
        mock_uuid.return_value.__str__ = lambda x: "test-uuid"
        mock_boto_client.return_value = MagicMock()
        mock_resolve.return_value = "s3://customer-escrow-123-smtj-abc/checkpoint"
        mock_build.return_value = (
            "recipe.yaml",
            self.output_s3_path,
            self.data_s3_path,
            "image",
        )
        self.mock_runtime_manager.execute.return_value = "job-123"

        result = self.customizer.evaluate(job_name="test-eval", eval_task=EvaluationTask.MMLU)

        self.assertIsNotNone(result)
        self.mock_runtime_manager.execute.assert_called_once()

    @patch("boto3.client")
    @patch("amzn_nova_forge.recipe.recipe_builder.RecipeBuilder.build_and_validate")
    @patch("amzn_nova_forge.model.nova_model_customizer.resolve_model_checkpoint_path")
    @patch("uuid.uuid4")
    def test_evaluate_platform_match_smhp(
        self, mock_uuid, mock_resolve, mock_build, mock_boto_client
    ):
        """Test that SMHP checkpoint on SMHP execution succeeds"""
        mock_uuid.return_value = MagicMock()
        mock_uuid.return_value.__str__ = lambda x: "test-uuid"
        mock_boto_client.return_value = MagicMock()
        mock_resolve.return_value = "s3://customer-escrow-123-hp-abc/checkpoint"
        mock_build.return_value = (
            "recipe.yaml",
            self.output_s3_path,
            self.data_s3_path,
            "image",
        )

        # Configure as SMHP
        mock_smhp_infra = create_autospec(SMHPRuntimeManager)
        mock_smhp_infra.platform = Platform.SMHP
        mock_smhp_infra.cluster_name = "test-cluster"
        mock_smhp_infra.namespace = "test-ns"
        mock_smhp_infra.execute.return_value = "job-456"
        self.customizer.infra = mock_smhp_infra
        self.customizer.platform = Platform.SMHP

        result = self.customizer.evaluate(job_name="test-eval", eval_task=EvaluationTask.MMLU)

        self.assertIsNotNone(result)
        mock_smhp_infra.execute.assert_called_once()

    @patch("amzn_nova_forge.util.logging.logger")
    @patch("boto3.client")
    @patch("amzn_nova_forge.recipe.recipe_builder.RecipeBuilder.build_and_validate")
    @patch("amzn_nova_forge.model.nova_model_customizer.resolve_model_checkpoint_path")
    @patch("uuid.uuid4")
    def test_evaluate_unknown_checkpoint_platform_logs_warning(
        self, mock_uuid, mock_resolve, mock_build, mock_boto_client, mock_logger
    ):
        """Test that unknown checkpoint platform logs warning but continues"""
        mock_uuid.return_value = MagicMock()
        mock_uuid.return_value.__str__ = lambda x: "test-uuid"
        mock_boto_client.return_value = MagicMock()
        mock_resolve.return_value = "s3://my-custom-bucket/checkpoint"
        mock_build.return_value = (
            "recipe.yaml",
            self.output_s3_path,
            self.data_s3_path,
            "image",
        )
        self.mock_runtime_manager.execute.return_value = "job-789"

        result = self.customizer.evaluate(job_name="test-eval", eval_task=EvaluationTask.MMLU)

        mock_logger.warning.assert_called_once()
        warning_msg = mock_logger.warning.call_args[0][0]
        self.assertIn("Cannot determine platform", warning_msg)
        self.assertIsNotNone(result)
        self.mock_runtime_manager.execute.assert_called_once()

    @patch("boto3.client")
    @patch("amzn_nova_forge.model.nova_model_customizer.resolve_model_checkpoint_path")
    def test_batch_inference_platform_mismatch_smhp_to_smtj(self, mock_resolve, mock_boto_client):
        """Test that SMHP checkpoint on SMTJ batch_inference raises ValueError"""
        mock_boto_client.return_value = MagicMock()
        mock_resolve.return_value = "s3://customer-escrow-123-hp-abc/checkpoint"

        with self.assertRaises(ValueError) as context:
            self.customizer.batch_inference(
                job_name="test-inference",
                input_path="s3://test/input",
                output_s3_path="s3://test/output",
            )

        error_msg = str(context.exception)
        self.assertIn("Platform mismatch", error_msg)
        self.assertIn("SMHP", error_msg)
        self.assertIn("SMTJ", error_msg)

    @patch("boto3.client")
    @patch("amzn_nova_forge.model.nova_model_customizer.resolve_model_checkpoint_path")
    def test_batch_inference_platform_mismatch_smtj_to_smhp(self, mock_resolve, mock_boto_client):
        """Test that SMTJ checkpoint on SMHP batch_inference raises ValueError"""
        mock_boto_client.return_value = MagicMock()
        mock_resolve.return_value = "s3://customer-escrow-123-smtj-abc/checkpoint"

        # Configure as SMHP
        mock_smhp_infra = create_autospec(SMHPRuntimeManager)
        mock_smhp_infra.platform = Platform.SMHP
        mock_smhp_infra.cluster_name = "test-cluster"
        mock_smhp_infra.namespace = "test-ns"
        self.customizer.infra = mock_smhp_infra
        self.customizer.platform = Platform.SMHP

        with self.assertRaises(ValueError) as context:
            self.customizer.batch_inference(
                job_name="test-inference",
                input_path="s3://test/input",
                output_s3_path="s3://test/output",
            )

        error_msg = str(context.exception)
        self.assertIn("Platform mismatch", error_msg)
        self.assertIn("SMTJ", error_msg)
        self.assertIn("SMHP", error_msg)

    @patch("boto3.client")
    @patch("amzn_nova_forge.recipe.recipe_builder.RecipeBuilder.build_and_validate")
    @patch("amzn_nova_forge.model.nova_model_customizer.resolve_model_checkpoint_path")
    @patch("uuid.uuid4")
    def test_batch_inference_platform_match_smtj(
        self, mock_uuid, mock_resolve, mock_build, mock_boto_client
    ):
        """Test that SMTJ checkpoint on SMTJ batch_inference succeeds"""
        mock_uuid.return_value = MagicMock()
        mock_uuid.return_value.__str__ = lambda x: "test-uuid"
        mock_boto_client.return_value = MagicMock()
        mock_resolve.return_value = "s3://customer-escrow-123-smtj-abc/checkpoint"
        mock_build.return_value = (
            "recipe.yaml",
            self.output_s3_path,
            self.data_s3_path,
            "image",
        )
        self.mock_runtime_manager.execute.return_value = "job-123"

        result = self.customizer.batch_inference(
            job_name="test-inference",
            input_path="s3://test/input",
            output_s3_path="s3://test/output",
        )

        self.assertIsNotNone(result)
        self.mock_runtime_manager.execute.assert_called_once()

    @patch("boto3.client")
    @patch("amzn_nova_forge.recipe.recipe_builder.RecipeBuilder.build_and_validate")
    @patch("amzn_nova_forge.model.nova_model_customizer.resolve_model_checkpoint_path")
    @patch("uuid.uuid4")
    def test_batch_inference_platform_match_smhp(
        self, mock_uuid, mock_resolve, mock_build, mock_boto_client
    ):
        """Test that SMHP checkpoint on SMHP batch_inference succeeds"""
        mock_uuid.return_value = MagicMock()
        mock_uuid.return_value.__str__ = lambda x: "test-uuid"
        mock_boto_client.return_value = MagicMock()
        mock_resolve.return_value = "s3://customer-escrow-123-hp-abc/checkpoint"
        mock_build.return_value = (
            "recipe.yaml",
            self.output_s3_path,
            self.data_s3_path,
            "image",
        )

        # Configure as SMHP
        mock_smhp_infra = create_autospec(SMHPRuntimeManager)
        mock_smhp_infra.platform = Platform.SMHP
        mock_smhp_infra.cluster_name = "test-cluster"
        mock_smhp_infra.namespace = "test-ns"
        mock_smhp_infra.execute.return_value = "job-456"
        self.customizer.infra = mock_smhp_infra
        self.customizer.platform = Platform.SMHP

        result = self.customizer.batch_inference(
            job_name="test-inference",
            input_path="s3://test/input",
            output_s3_path="s3://test/output",
        )

        self.assertIsNotNone(result)
        mock_smhp_infra.execute.assert_called_once()

    @patch("amzn_nova_forge.util.logging.logger")
    @patch("boto3.client")
    @patch("amzn_nova_forge.recipe.recipe_builder.RecipeBuilder.build_and_validate")
    @patch("amzn_nova_forge.model.nova_model_customizer.resolve_model_checkpoint_path")
    @patch("uuid.uuid4")
    def test_batch_inference_unknown_checkpoint_platform_logs_warning(
        self, mock_uuid, mock_resolve, mock_build, mock_boto_client, mock_logger
    ):
        """Test that unknown checkpoint platform logs warning but continues"""
        mock_uuid.return_value = MagicMock()
        mock_uuid.return_value.__str__ = lambda x: "test-uuid"
        mock_boto_client.return_value = MagicMock()
        mock_resolve.return_value = "s3://my-custom-bucket/checkpoint"
        mock_build.return_value = (
            "recipe.yaml",
            self.output_s3_path,
            self.data_s3_path,
            "image",
        )
        self.mock_runtime_manager.execute.return_value = "job-789"

        result = self.customizer.batch_inference(
            job_name="test-inference",
            input_path="s3://test/input",
            output_s3_path="s3://test/output",
        )

        mock_logger.warning.assert_called_once()
        warning_msg = mock_logger.warning.call_args[0][0]
        self.assertIn("Cannot determine platform", warning_msg)
        self.assertIsNotNone(result)
        self.mock_runtime_manager.execute.assert_called_once()

    def test_invoke_inference_no_endpoint_raises_error(self):
        """Test that an error is raised when no endpoint arn or endpoint info is provided"""
        with self.assertRaises(ValueError):
            self.customizer.invoke_inference({"messages": []})

    @patch("boto3.client")
    @patch("amzn_nova_forge.inference.forge_inference.invoke_sagemaker_inference")
    def test_invoke_inference_sagemaker_endpoint(self, mock_invoke_sagemaker, mock_boto3_client):
        # Setup
        endpoint_arn = "arn:aws:sagemaker:us-east-1:123456789012:endpoint/test-endpoint"
        request_body = {
            "messages": [{"role": "user", "content": "Hello"}],
            "max_tokens": 100,
            "stream": False,
        }
        mock_runtime_client = MagicMock()
        mock_boto3_client.return_value = mock_runtime_client
        mock_invoke_sagemaker.return_value = "Inference Result"

        # Execute
        result = self.customizer.invoke_inference(request_body, endpoint_arn)

        # Assert
        mock_boto3_client.assert_called_once_with("sagemaker-runtime", region_name="us-east-1")
        mock_invoke_sagemaker.assert_called_once_with(
            request_body, "test-endpoint", mock_runtime_client, inference_component_name=None
        )
        assert result == "Inference Result"

    @patch("boto3.client")
    @patch("amzn_nova_forge.inference.forge_inference.invoke_model")
    def test_invoke_inference_bedrock_endpoint(self, mock_invoke_model, mock_boto3_client):
        endpoint_arn = "arn:aws:bedrock:us-east-1:123456789012:custom-model-deployment/test-model"
        request_body = {
            "system": [{"text": "You are an assistant"}],
            "messages": [{"role": "user", "content": [{"text": "Hello"}]}],
        }
        mock_runtime_client = MagicMock()
        mock_boto3_client.return_value = mock_runtime_client
        mock_invoke_model.return_value = "Bedrock Inference Result"

        # Execute
        result = self.customizer.invoke_inference(request_body, endpoint_arn)

        # Assert
        mock_boto3_client.assert_called_once_with("bedrock-runtime", region_name="us-east-1")
        mock_invoke_model.assert_called_once_with(
            model_id=endpoint_arn,
            request_body=request_body,
            bedrock_runtime=mock_runtime_client,
        )
        assert result == "Bedrock Inference Result"


class TestLambdaVerification(TestNovaModelCustomizer):
    """Test suite for RFT lambda verification functionality"""

    def setUp(self):
        super().setUp()
        self.method = TrainingMethod.RFT_LORA
        self.data_s3_path = "s3://test-bucket/data.jsonl"
        self.customizer = NovaModelCustomizer(
            model=self.model,
            method=self.method,
            infra=self.mock_runtime_manager,
            data_s3_path=self.data_s3_path,
            output_s3_path=self.output_s3_path,
        )

    def test_customizer_attributes(self):
        """Override parent test to check RFT_LORA method."""
        self.assertEqual(self.customizer.model, Model.NOVA_MICRO)
        self.assertEqual(self.customizer.method, TrainingMethod.RFT_LORA)
        self.assertEqual(self.customizer.data_s3_path, self.data_s3_path)
        self.assertEqual(self.customizer.output_s3_path, self.output_s3_path)
        self.assertEqual(self.customizer.region, "us-east-1")

    @patch("amzn_nova_forge.trainer.forge_trainer.SMTJTrainingResult")
    @patch("amzn_nova_forge.trainer.forge_trainer.get_model_artifacts")
    @patch("amzn_nova_forge.trainer.forge_trainer.RecipeBuilder")
    def test_train_reads_rft_lambda_arn_from_infra(self, mock_rb, mock_gma, mock_training_result):
        """train() uses infra.rft_lambda_arn when no arg is passed."""
        arn = "arn:aws:lambda:us-east-1:123456789012:function:my-SageMaker-fn"
        self.mock_runtime_manager.rft_lambda_arn = arn
        self.mock_runtime_manager.rft_lambda = arn

        mock_rb_instance = MagicMock()
        mock_rb.return_value = mock_rb_instance
        mock_rb_instance.build_and_validate.return_value = (
            "recipe.yaml",
            self.output_s3_path,
            self.data_s3_path,
            None,
        )
        self.mock_runtime_manager.execute.return_value = "job-123"

        with patch(
            "amzn_nova_forge.trainer.forge_trainer.validate_rft_lambda_name"
        ) as mock_validate_name:
            self.customizer.method = TrainingMethod.RFT_LORA
            self.customizer.train(job_name="test-rft-job")
            mock_validate_name.assert_called_once_with(arn.split(":")[-1], self.customizer.platform)

    @patch("amzn_nova_forge.trainer.forge_trainer.SMTJTrainingResult")
    @patch("amzn_nova_forge.trainer.forge_trainer.get_model_artifacts")
    @patch("amzn_nova_forge.trainer.forge_trainer.RecipeBuilder")
    def test_train_prefers_direct_arg_over_infra_rft_lambda_arn(
        self, mock_rb, mock_gma, mock_training_result
    ):
        """train() prefers the directly-passed rft_lambda_arn over infra.rft_lambda_arn."""
        infra_arn = "arn:aws:lambda:us-east-1:123456789012:function:infra-fn"
        direct_arn = "arn:aws:lambda:us-east-1:123456789012:function:direct-fn"
        self.mock_runtime_manager.rft_lambda_arn = infra_arn
        self.mock_runtime_manager.rft_lambda = infra_arn

        mock_rb_instance = MagicMock()
        mock_rb.return_value = mock_rb_instance
        mock_rb_instance.build_and_validate.return_value = (
            "recipe.yaml",
            self.output_s3_path,
            self.data_s3_path,
            None,
        )
        self.mock_runtime_manager.execute.return_value = "job-123"

        with patch(
            "amzn_nova_forge.trainer.forge_trainer.validate_rft_lambda_name"
        ) as mock_validate_name:
            self.customizer.method = TrainingMethod.RFT_LORA
            self.customizer.train(job_name="test-rft-job", rft_lambda_arn=direct_arn)
            # Should validate the direct ARN, not the infra one
            mock_validate_name.assert_called_once_with(
                direct_arn.split(":")[-1], self.customizer.platform
            )

    @patch("amzn_nova_forge.trainer.forge_trainer.SMTJTrainingResult")
    @patch("amzn_nova_forge.trainer.forge_trainer.get_model_artifacts")
    @patch("amzn_nova_forge.trainer.forge_trainer.RecipeBuilder")
    def test_train_uses_rft_lambda_arn_when_infra_has_none(
        self, mock_rb, mock_gma, mock_training_result
    ):
        """train() uses the directly-passed rft_lambda_arn when infra has none."""
        arn = "arn:aws:lambda:us-east-1:123456789012:function:my-SageMaker-fn"
        self.mock_runtime_manager.rft_lambda_arn = None
        self.mock_runtime_manager.rft_lambda = None

        mock_rb_instance = MagicMock()
        mock_rb.return_value = mock_rb_instance
        mock_rb_instance.build_and_validate.return_value = (
            "recipe.yaml",
            self.output_s3_path,
            self.data_s3_path,
            None,
        )
        self.mock_runtime_manager.execute.return_value = "job-123"

        with patch(
            "amzn_nova_forge.trainer.forge_trainer.validate_rft_lambda_name"
        ) as mock_validate_name:
            self.customizer.method = TrainingMethod.RFT_LORA
            self.customizer.train(job_name="test-rft-job", rft_lambda_arn=arn)
            mock_validate_name.assert_called_once_with(arn.split(":")[-1], self.customizer.platform)

    @patch("amzn_nova_forge.evaluator.forge_evaluator.SMTJEvaluationResult")
    @patch("amzn_nova_forge.evaluator.forge_evaluator.RecipeBuilder")
    def test_evaluate_auto_populates_rl_env_from_infra_lambda_arn(self, mock_rb, mock_eval_result):
        """evaluate() sets rl_env.reward_lambda_arn from infra.rft_lambda_arn when not provided."""
        arn = "arn:aws:lambda:us-east-1:123456789012:function:my-SageMaker-fn"
        self.mock_runtime_manager.rft_lambda_arn = arn

        mock_rb_instance = MagicMock()
        mock_rb.return_value = mock_rb_instance
        mock_rb_instance.build_and_validate.return_value = (
            "recipe.yaml",
            self.output_s3_path,
            self.data_s3_path,
            None,
        )
        self.mock_runtime_manager.execute.return_value = "eval-job-123"

        self.customizer.method = TrainingMethod.EVALUATION
        self.customizer.evaluate(
            job_name="test-eval-job",
            eval_task=EvaluationTask.RFT_EVAL,
            data_s3_path=self.data_s3_path,
        )

        call_kwargs = mock_rb.call_args.kwargs
        self.assertEqual(call_kwargs["rl_env_config"], {"reward_lambda_arn": arn})
        self.assertIsNone(call_kwargs["processor_config"])

    @patch("amzn_nova_forge.evaluator.forge_evaluator.SMTJEvaluationResult")
    @patch("amzn_nova_forge.evaluator.forge_evaluator.RecipeBuilder")
    def test_evaluate_migrates_processor_lambda_arn_to_rl_env(self, mock_rb, mock_eval_result):
        """evaluate() migrates processor.lambda_arn to rl_env.reward_lambda_arn for RFT_EVAL."""
        arn = "arn:aws:lambda:us-east-1:123456789012:function:my-SageMaker-fn"
        self.mock_runtime_manager.rft_lambda_arn = None

        mock_rb_instance = MagicMock()
        mock_rb.return_value = mock_rb_instance
        mock_rb_instance.build_and_validate.return_value = (
            "recipe.yaml",
            self.output_s3_path,
            self.data_s3_path,
            None,
        )
        self.mock_runtime_manager.execute.return_value = "eval-job-123"

        self.customizer.method = TrainingMethod.EVALUATION
        self.customizer.evaluate(
            job_name="test-eval-job",
            eval_task=EvaluationTask.RFT_EVAL,
            data_s3_path=self.data_s3_path,
            processor={"lambda_arn": arn},
        )

        call_kwargs = mock_rb.call_args.kwargs
        self.assertEqual(call_kwargs["rl_env_config"], {"reward_lambda_arn": arn})
        self.assertIsNone(call_kwargs["processor_config"])

    @patch("amzn_nova_forge.evaluator.forge_evaluator.SMTJEvaluationResult")
    @patch("amzn_nova_forge.evaluator.forge_evaluator.RecipeBuilder")
    def test_evaluate_does_not_overwrite_existing_rl_env(self, mock_rb, mock_eval_result):
        """evaluate() does not overwrite rl_env when it is already provided."""
        infra_arn = "arn:aws:lambda:us-east-1:123456789012:function:infra-fn"
        explicit_arn = "arn:aws:lambda:us-east-1:123456789012:function:explicit-fn"
        self.mock_runtime_manager.rft_lambda_arn = infra_arn

        mock_rb_instance = MagicMock()
        mock_rb.return_value = mock_rb_instance
        mock_rb_instance.build_and_validate.return_value = (
            "recipe.yaml",
            self.output_s3_path,
            self.data_s3_path,
            None,
        )
        self.mock_runtime_manager.execute.return_value = "eval-job-123"

        self.customizer.method = TrainingMethod.EVALUATION
        self.customizer.evaluate(
            job_name="test-eval-job",
            eval_task=EvaluationTask.RFT_EVAL,
            data_s3_path=self.data_s3_path,
            rl_env={"reward_lambda_arn": explicit_arn},
        )

        call_kwargs = mock_rb.call_args.kwargs
        # Explicit rl_env must not be overwritten by infra_arn
        self.assertEqual(call_kwargs["rl_env_config"], {"reward_lambda_arn": explicit_arn})

    def test_verify_rft_lambda_no_data_s3_path(self):
        """Test verify_rft_lambda raises error when data_s3_path is not set"""
        with self.assertRaises(ValueError) as context:
            verify_rft_lambda(
                lambda_arn="arn:aws:lambda:us-east-1:123456789012:function:test",
                sample_count=10,
                data_s3_path=None,
                region="us-east-1",
                platform=Platform.SMTJ,
            )

        self.assertIn("Cannot verify RFT lambda", str(context.exception))
        self.assertIn("data_s3_path is not set", str(context.exception))

    def test_verify_rft_lambda_invalid_s3_path(self):
        """Test verify_rft_lambda raises error for invalid S3 path"""
        with self.assertRaises(ValueError) as context:
            verify_rft_lambda(
                lambda_arn="arn:aws:lambda:us-east-1:123456789012:function:test",
                sample_count=10,
                data_s3_path="invalid-path",
                region="us-east-1",
                platform=Platform.SMTJ,
            )

        self.assertIn("Invalid S3 path", str(context.exception))
        self.assertIn("s3://bucket-name/path/to/data.jsonl", str(context.exception))

    @patch("boto3.client")
    def test_verify_rft_lambda_s3_read_failure(self, mock_boto_client):
        """Test verify_rft_lambda handles S3 read failures"""
        mock_s3 = MagicMock()
        mock_s3.get_object.side_effect = Exception("Access Denied")
        mock_boto_client.return_value = mock_s3

        with self.assertRaises(ValueError) as context:
            verify_rft_lambda(
                lambda_arn="arn:aws:lambda:us-east-1:123456789012:function:test",
                sample_count=10,
                data_s3_path=self.data_s3_path,
                region="us-east-1",
                platform=Platform.SMTJ,
            )

        self.assertIn("Failed to read samples", str(context.exception))
        self.assertIn("Access Denied", str(context.exception))
        self.assertIn("verify the S3 path is correct", str(context.exception))

    @patch("boto3.client")
    def test_verify_rft_lambda_invalid_json(self, mock_boto_client):
        """Test verify_rft_lambda handles invalid JSON in data file"""
        mock_s3 = MagicMock()
        mock_response = MagicMock()
        mock_response["Body"].iter_lines.return_value = [
            b'{"valid": "json"}',
            b"invalid json here",
        ]
        mock_s3.get_object.return_value = mock_response
        mock_boto_client.return_value = mock_s3

        with self.assertRaises(ValueError) as context:
            verify_rft_lambda(
                lambda_arn="arn:aws:lambda:us-east-1:123456789012:function:test",
                sample_count=10,
                data_s3_path=self.data_s3_path,
                region="us-east-1",
                platform=Platform.SMTJ,
            )

        self.assertIn("Failed to parse JSON", str(context.exception))
        self.assertIn("line 2", str(context.exception))

    @patch("boto3.client")
    def test_verify_rft_lambda_empty_data_file(self, mock_boto_client):
        """Test verify_rft_lambda handles empty data file"""
        mock_s3 = MagicMock()
        mock_response = MagicMock()
        mock_response["Body"].iter_lines.return_value = []
        mock_s3.get_object.return_value = mock_response
        mock_boto_client.return_value = mock_s3

        with self.assertRaises(ValueError) as context:
            verify_rft_lambda(
                lambda_arn="arn:aws:lambda:us-east-1:123456789012:function:test",
                sample_count=10,
                data_s3_path=self.data_s3_path,
                region="us-east-1",
                platform=Platform.SMTJ,
            )

        self.assertIn("No samples found", str(context.exception))
        self.assertIn("ensure the data file contains valid JSONL data", str(context.exception))

    @patch("boto3.client")
    def test_verify_rft_lambda_lambda_invocation_failure(self, mock_boto_client):
        """Test verify_rft_lambda handles lambda invocation failures"""
        mock_s3 = MagicMock()
        mock_response = MagicMock()
        mock_response["Body"].iter_lines.return_value = [
            b'{"prompt": "test", "completion": "test"}',
        ]
        mock_s3.get_object.return_value = mock_response

        mock_lambda = MagicMock()
        mock_lambda.invoke.return_value = {"StatusCode": 500}

        def client_side_effect(service, **kwargs):
            if service == "s3":
                return mock_s3
            elif service == "lambda":
                return mock_lambda
            return MagicMock()

        mock_boto_client.side_effect = client_side_effect

        with self.assertRaises(ValueError) as context:
            verify_rft_lambda(
                lambda_arn="arn:aws:lambda:us-east-1:123456789012:function:test",
                sample_count=1,
                data_s3_path=self.data_s3_path,
                region="us-east-1",
                platform=Platform.SMTJ,
            )

        self.assertIn("RFT lambda verification failed", str(context.exception))

    @patch("boto3.client")
    def test_verify_rft_lambda_missing_reward_field(self, mock_boto_client):
        """Test verify_rft_lambda detects missing reward field in response"""
        mock_s3 = MagicMock()
        mock_response = MagicMock()
        mock_response["Body"].iter_lines.return_value = [
            b'{"prompt": "test", "completion": "test"}',
        ]
        mock_s3.get_object.return_value = mock_response

        mock_lambda = MagicMock()
        mock_payload = MagicMock()
        mock_payload.read.return_value = b'{"result": "no reward"}'
        mock_lambda.invoke.return_value = {"StatusCode": 200, "Payload": mock_payload}

        def client_side_effect(service, **kwargs):
            if service == "s3":
                return mock_s3
            elif service == "lambda":
                return mock_lambda
            return MagicMock()

        mock_boto_client.side_effect = client_side_effect

        with self.assertRaises(ValueError) as context:
            verify_rft_lambda(
                lambda_arn="arn:aws:lambda:us-east-1:123456789012:function:test",
                sample_count=1,
                data_s3_path=self.data_s3_path,
                region="us-east-1",
                platform=Platform.SMTJ,
            )

        self.assertIn("RFT lambda verification failed", str(context.exception))

    @patch("boto3.client")
    def test_verify_rft_lambda_invalid_reward_type(self, mock_boto_client):
        """Test verify_rft_lambda detects non-numeric reward values"""
        mock_s3 = MagicMock()
        mock_response = MagicMock()
        mock_response["Body"].iter_lines.return_value = [
            b'{"prompt": "test", "completion": "test"}',
        ]
        mock_s3.get_object.return_value = mock_response

        mock_lambda = MagicMock()
        mock_payload = MagicMock()
        mock_payload.read.return_value = b'{"reward": "not a number"}'
        mock_lambda.invoke.return_value = {"StatusCode": 200, "Payload": mock_payload}

        def client_side_effect(service, **kwargs):
            if service == "s3":
                return mock_s3
            elif service == "lambda":
                return mock_lambda
            return MagicMock()

        mock_boto_client.side_effect = client_side_effect

        with self.assertRaises(ValueError) as context:
            verify_rft_lambda(
                lambda_arn="arn:aws:lambda:us-east-1:123456789012:function:test",
                sample_count=1,
                data_s3_path=self.data_s3_path,
                region="us-east-1",
                platform=Platform.SMTJ,
            )

        self.assertIn("RFT lambda verification failed", str(context.exception))

    @patch("boto3.client")
    def test_verify_rft_lambda_success_with_int_reward(self, mock_boto_client):
        """Test verify_rft_lambda succeeds with integer reward"""
        mock_s3 = MagicMock()
        mock_response = MagicMock()
        mock_response["Body"].iter_lines.return_value = [
            b'{"id": "sample_1", "messages": [{"role": "user", "content": "test"}]}',
        ]
        mock_s3.get_object.return_value = mock_response

        mock_lambda = MagicMock()
        mock_payload = MagicMock()
        mock_payload.read.return_value = b'[{"id": "sample_1", "aggregate_reward_score": 5}]'
        mock_lambda.invoke.return_value = {"StatusCode": 200, "Payload": mock_payload}

        def client_side_effect(service, **kwargs):
            if service == "s3":
                return mock_s3
            elif service == "lambda":
                return mock_lambda
            return MagicMock()

        mock_boto_client.side_effect = client_side_effect

        # Should not raise any exception
        verify_rft_lambda(
            lambda_arn="arn:aws:lambda:us-east-1:123456789012:function:test",
            sample_count=1,
            data_s3_path=self.data_s3_path,
            region="us-east-1",
            platform=Platform.SMTJ,
        )

    @patch("boto3.client")
    def test_verify_rft_lambda_success_with_float_reward(self, mock_boto_client):
        """Test verify_rft_lambda succeeds with float reward"""
        mock_s3 = MagicMock()
        mock_response = MagicMock()
        mock_response["Body"].iter_lines.return_value = [
            b'{"id": "sample_1", "messages": [{"role": "user", "content": "test"}]}',
        ]
        mock_s3.get_object.return_value = mock_response

        mock_lambda = MagicMock()
        mock_payload = MagicMock()
        mock_payload.read.return_value = b'[{"id": "sample_1", "aggregate_reward_score": 3.14}]'
        mock_lambda.invoke.return_value = {"StatusCode": 200, "Payload": mock_payload}

        def client_side_effect(service, **kwargs):
            if service == "s3":
                return mock_s3
            elif service == "lambda":
                return mock_lambda
            return MagicMock()

        mock_boto_client.side_effect = client_side_effect

        # Should not raise any exception
        verify_rft_lambda(
            lambda_arn="arn:aws:lambda:us-east-1:123456789012:function:test",
            sample_count=1,
            data_s3_path=self.data_s3_path,
            region="us-east-1",
            platform=Platform.SMTJ,
        )

    @patch("boto3.client")
    def test_verify_rft_lambda_multiple_samples(self, mock_boto_client):
        """Test verify_rft_lambda verifies multiple samples"""
        mock_s3 = MagicMock()
        mock_response = MagicMock()
        mock_response["Body"].iter_lines.return_value = [
            b'{"id": "sample_1", "messages": [{"role": "user", "content": "test1"}]}',
            b'{"id": "sample_2", "messages": [{"role": "user", "content": "test2"}]}',
            b'{"id": "sample_3", "messages": [{"role": "user", "content": "test3"}]}',
        ]
        mock_s3.get_object.return_value = mock_response

        mock_lambda = MagicMock()
        mock_payload = MagicMock()
        mock_payload.read.return_value = b'[{"id": "sample_1", "aggregate_reward_score": 1.0}, {"id": "sample_2", "aggregate_reward_score": 1.0}, {"id": "sample_3", "aggregate_reward_score": 1.0}]'
        mock_lambda.invoke.return_value = {"StatusCode": 200, "Payload": mock_payload}

        def client_side_effect(service, **kwargs):
            if service == "s3":
                return mock_s3
            elif service == "lambda":
                return mock_lambda
            return MagicMock()

        mock_boto_client.side_effect = client_side_effect

        # Should not raise any exception
        verify_rft_lambda(
            lambda_arn="arn:aws:lambda:us-east-1:123456789012:function:test",
            sample_count=3,
            data_s3_path=self.data_s3_path,
            region="us-east-1",
            platform=Platform.SMTJ,
        )

    @patch("boto3.client")
    def test_verify_rft_lambda_limits_samples_read(self, mock_boto_client):
        """Test verify_rft_lambda only reads requested number of samples"""
        mock_s3 = MagicMock()
        mock_response = MagicMock()
        # Provide more samples than requested
        mock_response["Body"].iter_lines.return_value = [
            b'{"id": "sample_1", "messages": [{"role": "user", "content": "test1"}]}',
            b'{"id": "sample_2", "messages": [{"role": "user", "content": "test2"}]}',
            b'{"id": "sample_3", "messages": [{"role": "user", "content": "test3"}]}',
            b'{"id": "sample_4", "messages": [{"role": "user", "content": "test4"}]}',
            b'{"id": "sample_5", "messages": [{"role": "user", "content": "test5"}]}',
        ]
        mock_s3.get_object.return_value = mock_response

        mock_lambda = MagicMock()
        mock_payload = MagicMock()
        mock_payload.read.return_value = b'[{"id": "sample_1", "aggregate_reward_score": 1.0}, {"id": "sample_2", "aggregate_reward_score": 1.0}]'
        mock_lambda.invoke.return_value = {"StatusCode": 200, "Payload": mock_payload}

        def client_side_effect(service, **kwargs):
            if service == "s3":
                return mock_s3
            elif service == "lambda":
                return mock_lambda
            return MagicMock()

        mock_boto_client.side_effect = client_side_effect

        # Request only 2 samples
        verify_rft_lambda(
            lambda_arn="arn:aws:lambda:us-east-1:123456789012:function:test",
            sample_count=2,
            data_s3_path=self.data_s3_path,
            region="us-east-1",
            platform=Platform.SMTJ,
        )

    @patch("boto3.client")
    def test_verify_rft_lambda_error_includes_troubleshooting(self, mock_boto_client):
        """Test verify_rft_lambda error messages include troubleshooting guidance"""
        mock_s3 = MagicMock()
        mock_response = MagicMock()
        mock_response["Body"].iter_lines.return_value = [
            b'{"id": "sample_1", "messages": [{"role": "user", "content": "test"}]}',
        ]
        mock_s3.get_object.return_value = mock_response

        mock_lambda = MagicMock()
        mock_lambda.invoke.side_effect = Exception("Lambda timeout")

        def client_side_effect(service, **kwargs):
            if service == "s3":
                return mock_s3
            elif service == "lambda":
                return mock_lambda
            return MagicMock()

        mock_boto_client.side_effect = client_side_effect

        with self.assertRaises(ValueError) as context:
            verify_rft_lambda(
                lambda_arn="arn:aws:lambda:us-east-1:123456789012:function:test",
                sample_count=1,
                data_s3_path=self.data_s3_path,
                region="us-east-1",
                platform=Platform.SMTJ,
            )

        error_msg = str(context.exception)
        self.assertIn("RFT lambda verification failed", error_msg)


class TestJobCachingAndPersistence(unittest.TestCase):
    def setUp(self):
        self.model = Model.NOVA_MICRO
        self.method = TrainingMethod.SFT_LORA
        self.data_s3_path = "s3://test-bucket/data"
        self.output_s3_path = "s3://test-bucket/output"
        self.temp_dir = tempfile.mkdtemp()

        self.mock_runtime_manager = create_autospec(SMTJRuntimeManager)
        self.mock_runtime_manager.platform = Platform.SMTJ
        self.mock_runtime_manager.instance_count = 2

        with (
            patch("boto3.client") as mock_client,
            patch(
                "sagemaker.core.helper.session_helper.get_execution_role"
            ) as mock_get_execution_role,
        ):
            mock_get_execution_role.return_value = (
                "arn:aws:iam::123456789012:role/SageMakerExecutionRole"
            )

            mock_sts = MagicMock()
            mock_sts.get_caller_identity.return_value = {"Account": "123456789012"}
            mock_s3 = MagicMock()
            mock_s3.head_bucket.return_value = {}

            mock_iam = MagicMock()
            mock_iam.get_role.return_value = {
                "Role": {
                    "AssumeRolePolicyDocument": {
                        "Statement": [
                            {
                                "Effect": "Allow",
                                "Principal": {"Service": "sagemaker.amazonaws.com"},
                                "Action": "sts:AssumeRole",
                            }
                        ]
                    }
                }
            }
            mock_sagemaker = MagicMock()

            def client_side_effect(service, **kwargs):
                if service == "sts":
                    return mock_sts
                elif service == "s3":
                    return mock_s3
                elif service == "iam":
                    return mock_iam
                elif service == "sagemaker":
                    return mock_sagemaker
                return MagicMock()

            mock_client.side_effect = client_side_effect

            self.customizer = NovaModelCustomizer(
                model=self.model,
                method=self.method,
                infra=self.mock_runtime_manager,
                data_s3_path=self.data_s3_path,
                output_s3_path=self.output_s3_path,
                generated_recipe_dir=self.temp_dir,
                enable_job_caching=True,
            )
            self.customizer.job_cache_dir = self.temp_dir

    def tearDown(self):
        import shutil

        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def _create_aws_mocked_customizer(
        self,
        temp_dir,
        runtime_manager_params=None,
        customizer_params=None,
        training_job_status="Completed",
    ):
        """
        Helper method to create a NovaModelCustomizer with AWS mocking for tests that need custom instances.

        Args:
            temp_dir: Temporary directory for generated recipes
            runtime_manager_params: Dict of parameters for SMTJRuntimeManager constructor
            customizer_params: Dict of parameters for NovaModelCustomizer constructor
            training_job_status: Status to return from mock SageMaker describe_training_job

        Returns:
            Tuple of (customizer, mock_sagemaker, client_side_effect)
        """
        import inspect

        # Default parameters
        default_runtime_params = {
            "instance_type": "ml.g5.12xlarge",
            "instance_count": 1,
        }
        default_customizer_params = {
            "model": Model.NOVA_LITE_2,
            "method": TrainingMethod.SFT_LORA,
            "data_s3_path": "s3://test-bucket/data.jsonl",
            "generated_recipe_dir": temp_dir,
        }

        # Merge with provided parameters
        runtime_params = {**default_runtime_params, **(runtime_manager_params or {})}
        customizer_params = {**default_customizer_params, **(customizer_params or {})}

        # Validate parameters using introspection
        runtime_sig = inspect.signature(SMTJRuntimeManager.__init__)
        customizer_sig = inspect.signature(NovaModelCustomizer.__init__)

        # Filter out 'self' parameter and validate
        valid_runtime_params = {
            k: v for k, v in runtime_params.items() if k in runtime_sig.parameters and k != "self"
        }
        valid_customizer_params = {
            k: v
            for k, v in customizer_params.items()
            if k in customizer_sig.parameters and k != "self"
        }

        # Set up AWS mocks
        mock_sts = MagicMock()
        mock_sts.get_caller_identity.return_value = {"Account": "123456789012"}
        mock_s3 = MagicMock()
        mock_s3.head_bucket.return_value = {}
        mock_s3.create_bucket.return_value = {}
        mock_iam = MagicMock()
        mock_iam.get_role.return_value = {
            "Role": {
                "AssumeRolePolicyDocument": {
                    "Statement": [
                        {
                            "Effect": "Allow",
                            "Principal": {"Service": "sagemaker.amazonaws.com"},
                            "Action": "sts:AssumeRole",
                        }
                    ]
                }
            }
        }
        mock_sagemaker = MagicMock()
        mock_sagemaker.describe_training_job.return_value = {
            "TrainingJobStatus": training_job_status,
            "CheckpointConfig": {"S3Uri": "s3://test/checkpoint"},
            "OutputDataConfig": {"S3OutputPath": "s3://test/output"},
        }

        def client_side_effect(service, **kwargs):
            if service == "sts":
                return mock_sts
            elif service == "s3":
                return mock_s3
            elif service == "iam":
                return mock_iam
            elif service == "sagemaker":
                return mock_sagemaker
            return MagicMock()

        # Create runtime manager and customizer with validated parameters
        runtime = create_autospec(SMTJRuntimeManager)
        runtime.instance_count = valid_runtime_params.get("instance_count", 1)
        runtime.instance_type = valid_runtime_params.get("instance_type", "ml.g5.12xlarge")
        valid_customizer_params["infra"] = runtime
        customizer = NovaModelCustomizer(**valid_customizer_params)
        if temp_dir:
            customizer.job_cache_dir = temp_dir

        return customizer, mock_sagemaker, client_side_effect

    def test_init_with_job_caching_enabled(self):
        """Test initialization with job caching enabled"""
        self.assertTrue(self.customizer.enable_job_caching)
        self.assertIsNotNone(self.customizer._job_caching_config)
        self.assertEqual(self.customizer.generated_recipe_dir, self.temp_dir)

    def test_should_persist_results_true_when_job_caching_enabled(self):
        """Test _should_persist_results returns True when enable_job_caching is True"""
        self.assertTrue(should_persist_results(self.customizer))

    def test_should_persist_results_false_when_job_cache_dir_empty(self):
        """Test _should_persist_results returns False when job_cache_dir is empty"""
        self.customizer.enable_job_caching = True
        self.customizer.job_cache_dir = ""
        self.assertFalse(should_persist_results(self.customizer))

    def test_should_persist_results_false_when_job_caching_disabled(self):
        """Test _should_persist_results returns False when enable_job_caching is False"""
        with (
            patch("boto3.client") as mock_client,
            patch(
                "sagemaker.core.helper.session_helper.get_execution_role"
            ) as mock_get_execution_role,
        ):
            mock_get_execution_role.return_value = (
                "arn:aws:iam::123456789012:role/SageMakerExecutionRole"
            )

            customizer, mock_sagemaker, client_side_effect = self._create_aws_mocked_customizer(
                temp_dir=self.temp_dir,
                customizer_params={
                    "enable_job_caching": False,
                },
            )
            mock_client.side_effect = client_side_effect

        self.assertFalse(should_persist_results(customizer))

    def test_get_recipe_directory_input_types(self):
        """Test _get_recipe_directory with different input types"""
        # Test with directory path
        directory_path = get_recipe_directory(self.customizer.generated_recipe_dir)
        self.assertEqual(directory_path, self.temp_dir)

        # Test with file path
        with (
            patch("boto3.client") as mock_client,
            patch(
                "sagemaker.core.helper.session_helper.get_execution_role"
            ) as mock_get_execution_role,
        ):
            mock_get_execution_role.return_value = (
                "arn:aws:iam::123456789012:role/SageMakerExecutionRole"
            )

            file_path = "/path/to/recipe.yaml"
            customizer, mock_sagemaker, client_side_effect = self._create_aws_mocked_customizer(
                temp_dir=None, customizer_params={"generated_recipe_dir": file_path}
            )
            mock_client.side_effect = client_side_effect

        directory_path = get_recipe_directory(customizer.generated_recipe_dir)
        self.assertEqual(directory_path, "/path/to")

        # Test with None
        with (
            patch("boto3.client") as mock_client,
            patch(
                "sagemaker.core.helper.session_helper.get_execution_role"
            ) as mock_get_execution_role,
        ):
            mock_get_execution_role.return_value = (
                "arn:aws:iam::123456789012:role/SageMakerExecutionRole"
            )

            customizer, mock_sagemaker, client_side_effect = self._create_aws_mocked_customizer(
                temp_dir=None, customizer_params={"generated_recipe_dir": None}
            )
            mock_client.side_effect = client_side_effect

        directory_path = get_recipe_directory(customizer.generated_recipe_dir)
        self.assertIsNone(directory_path)

    def test_generate_job_hash(self):
        """Test _generate_job_hash produces consistent hashes for same parameters and different hashes for different parameters"""
        # Test consistency - same parameters produce same hash
        params = {"epochs": 3, "lr": 0.001, "batch_size": 32}
        hash1 = generate_job_hash(self.customizer, "test-job", "train", **params)
        hash2 = generate_job_hash(self.customizer, "test-job", "train", **params)
        self.assertEqual(hash1, hash2)

        # Test differentiation - different parameters produce different hashes
        hash3 = generate_job_hash(
            self.customizer,
            "test-job",
            "training",
            overrides={"max_epochs": 3, "lr": 0.001},
        )
        hash4 = generate_job_hash(
            self.customizer,
            "test-job",
            "training",
            overrides={"max_epochs": 5, "lr": 0.001},
        )
        self.assertNotEqual(hash3, hash4)

        # Test that train-specific parameters like rft_multiturn_infra and
        # validation_data_s3_path produce different hashes when they differ
        hash5 = generate_job_hash(
            self.customizer,
            "test-job",
            "training",
            validation_data_s3_path="s3://bucket/val1.jsonl",
        )
        hash6 = generate_job_hash(
            self.customizer,
            "test-job",
            "training",
            validation_data_s3_path="s3://bucket/val2.jsonl",
        )
        self.assertNotEqual(hash5, hash6)

    def test_generate_job_hash_includes_overrides(self):
        """Test _generate_job_hash includes override params in hash by default"""
        # With default config (include_recipe=True), override params are included in hash
        hash1 = generate_job_hash(
            self.customizer,
            "test-job",
            "train",
            overrides={"max_epochs": 3, "lr": 0.001},
        )
        hash2 = generate_job_hash(
            self.customizer,
            "test-job",
            "train",
            overrides={"max_epochs": 5, "lr": 0.01},
        )
        # Hashes differ because overrides are included
        self.assertNotEqual(hash1, hash2)

        # Same overrides should produce same hash
        hash3 = generate_job_hash(
            self.customizer,
            "test-job",
            "train",
            overrides={"max_epochs": 3, "lr": 0.001},
        )
        self.assertEqual(hash1, hash3)

    def test_get_result_file_path_with_job_caching(self):
        """Test _get_result_file_path with job caching enabled"""
        file_path = get_result_file_path(
            self.customizer,
            "test-job",
            "training",
            overrides={"max_epochs": 3, "lr": 0.001},
        )

        # Should generate a path with short hash of the full segmented hash
        filename = file_path.name
        self.assertTrue(filename.startswith("test-job_training_"))
        self.assertTrue(filename.endswith(".json"))
        # Extract timestamp from last part of filename (before .json)
        filename_parts = filename.replace(".json", "").split("_")
        timestamp_part = filename_parts[-1]  # Last part is the timestamp
        self.assertEqual(len(timestamp_part), 17)  # YYYYMMDDHHMMSSMMM should be 17 chars
        # Verify it's all digits (valid timestamp format)
        self.assertTrue(timestamp_part.isdigit())

    def test_get_result_file_path_raises_error_when_persistence_disabled(self):
        """Test _get_result_file_path raises error when persistence is disabled"""
        with (
            patch("boto3.client") as mock_client,
            patch(
                "sagemaker.core.helper.session_helper.get_execution_role"
            ) as mock_get_execution_role,
        ):
            mock_get_execution_role.return_value = (
                "arn:aws:iam::123456789012:role/SageMakerExecutionRole"
            )

            customizer, mock_sagemaker, client_side_effect = self._create_aws_mocked_customizer(
                temp_dir=self.temp_dir,
                customizer_params={"enable_job_caching": False},
            )
            mock_client.side_effect = client_side_effect

        with self.assertRaises(ValueError) as context:
            get_result_file_path(customizer, "test-job", "training")

        self.assertIn(
            "Cannot get result file path when persistence is disabled",
            str(context.exception),
        )

    @patch("boto3.client")
    def test_load_existing_result_success(self, mock_boto_client):
        """Test _load_existing_result successfully loads existing result when hash matches"""

        # Configure the mock to return completed status
        def mock_describe_training_job(*args, **kwargs):
            print(f"describe_training_job called with: {args}, {kwargs}")
            response = {
                "TrainingJobStatus": "Completed",
                "CheckpointConfig": {"S3Uri": "s3://test/checkpoint"},
                "OutputDataConfig": {"S3OutputPath": "s3://test/output"},
            }
            print(f"Returning response: {response}")
            return response

        mock_boto_client.return_value.describe_training_job.side_effect = mock_describe_training_job

        mock_artifacts = ModelArtifacts(
            checkpoint_s3_path="s3://test/checkpoint", output_s3_path="s3://test/output"
        )

        # Create a real job result that can be properly serialized
        job_result = SMTJTrainingResult(
            job_id="test-job-123",
            started_time=datetime.now(),
            method=TrainingMethod.SFT_LORA,
            model_type=Model.NOVA_LITE_2,
            model_artifacts=mock_artifacts,
            sagemaker_client=mock_boto_client.return_value,
        )

        # Persist the result which should add the job cache hash
        persist_result(
            self.customizer,
            job_result,
            "test-job",
            "training",
            overrides={"max_epochs": 3},
        )

        # Try to load the result with same parameters
        print(
            f"Mock configured: {mock_boto_client.return_value.describe_training_job.return_value}"
        )
        loaded_result = load_existing_result(
            self.customizer, "test-job", "training", overrides={"max_epochs": 3}
        )
        print(f"Loaded result: {loaded_result}")

        # Should find and return the result
        self.assertIsNotNone(loaded_result)
        self.assertEqual(loaded_result.job_id, "test-job-123")

        # Try to load with different parameters - should not find it
        loaded_result_different = load_existing_result(
            self.customizer, "test-job", "training", overrides={"max_epochs": 5}
        )
        self.assertIsNone(loaded_result_different)

    def test_load_existing_result_file_not_found(self):
        """Test _load_existing_result returns None when file doesn't exist"""
        non_existent_file = os.path.join(self.temp_dir, "non_existent.json")
        result = load_existing_result(self.customizer, non_existent_file, "train")
        self.assertIsNone(result)

    def test_persist_result_success(self):
        """Test _persist_result successfully saves result"""
        # Create a real job result object
        mock_artifacts = ModelArtifacts(
            checkpoint_s3_path="s3://test/checkpoint", output_s3_path="s3://test/output"
        )

        result = SMTJTrainingResult(
            job_id="test-job-123",
            started_time=datetime.now(),
            method=TrainingMethod.SFT_LORA,
            model_type=Model.NOVA_LITE_2,
            model_artifacts=mock_artifacts,
            sagemaker_client=MagicMock(),
        )

        persist_result(self.customizer, result, "test-job", "training", overrides={"max_epochs": 3})

        # Verify file was created
        files = list(Path(self.temp_dir).glob("*.json"))
        self.assertEqual(len(files), 1)

        # Verify file contains job cache hash
        with open(files[0], "r") as f:
            loaded_data = json.load(f)
        self.assertIn("_job_cache_hash", loaded_data)

    @patch("amzn_nova_forge.recipe.recipe_builder.RecipeBuilder.build_and_validate")
    @patch("uuid.uuid4")
    @patch("boto3.client")
    def test_train_with_job_caching_existing_result(self, mock_boto_client, mock_uuid, mock_build):
        """Test train method returns existing result when job cache finds match"""
        mock_uuid.return_value = MagicMock()
        mock_uuid.return_value.__str__ = lambda x: "test-uuid-1234"
        mock_build.return_value = (
            "mock_recipe.yaml",
            "s3://test/output",
            "s3://test/data",
            "mock-image-uri",
        )
        mock_boto_client.return_value = MagicMock()

        # Create and persist an existing completed result
        mock_artifacts = ModelArtifacts(
            checkpoint_s3_path="s3://test/checkpoint", output_s3_path="s3://test/output"
        )

        existing_result = SMTJTrainingResult(
            job_id="existing-job-123",
            started_time=datetime.now(),
            method=TrainingMethod.SFT_LORA,
            model_type=Model.NOVA_LITE_2,
            model_artifacts=mock_artifacts,
        )

        # Persist the result with same parameters that will be used in train call
        persist_result(
            self.customizer,
            existing_result,
            "test-job",
            "training",
            recipe_path=None,
            overrides={},
            rft_lambda_arn=None,
        )

        # Mock the result to appear completed by configuring the existing mock
        mock_boto_client.return_value.describe_training_job.return_value = {
            "TrainingJobStatus": "Completed",
            "CheckpointConfig": {"S3Uri": "s3://test/checkpoint"},
            "OutputDataConfig": {"S3OutputPath": "s3://test/output"},
        }

        result = self.customizer.train(job_name="test-job")

        # Should return existing result without calling runtime manager
        self.mock_runtime_manager.execute.assert_not_called()
        self.assertEqual(result.job_id, "existing-job-123")
        self.assertEqual(result.job_id, "existing-job-123")

    @patch("amzn_nova_forge.recipe.recipe_builder.RecipeBuilder.build_and_validate")
    @patch("uuid.uuid4")
    @patch("boto3.client")
    def test_train_with_job_caching_no_existing_result(
        self, mock_boto_client, mock_uuid, mock_build
    ):
        """Test train method executes normally when no existing result found"""
        mock_uuid.return_value = MagicMock()
        mock_uuid.return_value.__str__ = lambda x: "test-uuid-1234"
        mock_build.return_value = (
            "mock_recipe.yaml",
            "s3://test/output",
            "s3://test/data",
            "mock-image-uri",
        )

        mock_sagemaker = MagicMock()
        mock_sagemaker.describe_training_job.return_value = {
            "CheckpointConfig": {"S3Uri": "s3://test-bucket/checkpoints/model"},
            "OutputDataConfig": {"S3OutputPath": "s3://output-bucket/output"},
        }
        mock_boto_client.return_value = mock_sagemaker

        expected_job_id = "job-123"
        self.mock_runtime_manager.execute.return_value = expected_job_id

        result = self.customizer.train(job_name="test-job")

        # Should execute normally and persist result
        self.mock_runtime_manager.execute.assert_called_once()
        self.assertEqual(result.job_id, expected_job_id)

        # Verify result was persisted - check that a file was created
        files = list(Path(self.temp_dir).glob("*.json"))
        self.assertEqual(len(files), 1)

        # Verify the file contains the expected job_id
        with open(files[0], "r") as f:
            import json

            data = json.load(f)
        self.assertEqual(data["job_id"], expected_job_id)

    @patch("boto3.client")
    @patch("amzn_nova_forge.recipe.recipe_builder.RecipeBuilder.build_and_validate")
    @patch("uuid.uuid4")
    def test_evaluate_with_job_caching_existing_result(
        self, mock_uuid, mock_build, mock_boto_client
    ):
        """Test evaluate method returns existing result when job cache finds match"""
        mock_uuid.return_value = MagicMock()
        mock_uuid.return_value.__str__ = lambda x: "test-eval-uuid"
        mock_build.return_value = (
            "mock_eval_recipe.yaml",
            "s3://test/output",
            "s3://test/data",
        )
        mock_boto_client.return_value = MagicMock()

        # Create and persist an existing completed result
        existing_result = SMTJEvaluationResult(
            job_id="existing-eval-123",
            started_time=datetime.now(),
            eval_task=EvaluationTask.MMLU,
            eval_output_path="s3://test/eval/output",
        )

        # Persist the result with same parameters that will be used in evaluate call
        persist_result(
            self.customizer,
            existing_result,
            "test-eval-job",
            "evaluation",
            eval_task=EvaluationTask.MMLU,
            model_path=None,
            subtask=None,
            data_s3_path="s3://test-bucket/eval-data",
            recipe_path=None,
            overrides={},
            processor=None,
            rl_env=None,
        )

        # Mock the result to appear completed by mocking the SageMaker API call
        mock_sagemaker = MagicMock()
        mock_sagemaker.describe_training_job.return_value = {
            "TrainingJobStatus": "Completed",
            "CheckpointConfig": {"S3Uri": "s3://test/checkpoint"},
            "OutputDataConfig": {"S3OutputPath": "s3://test/output"},
        }
        with patch("boto3.client", return_value=mock_sagemaker):
            result = self.customizer.evaluate(
                job_name="test-eval-job",
                eval_task=EvaluationTask.MMLU,
                data_s3_path="s3://test-bucket/eval-data",
            )

            # Should return existing result without calling runtime manager
            self.mock_runtime_manager.execute.assert_not_called()
            self.assertEqual(result.job_id, "existing-eval-123")

    @patch("boto3.client")
    @patch("amzn_nova_forge.recipe.recipe_builder.RecipeBuilder.build_and_validate")
    @patch("uuid.uuid4")
    def test_batch_inference_with_job_caching_existing_result(
        self, mock_uuid, mock_build, mock_boto_client
    ):
        """Test batch_inference method returns existing result when job cache finds match"""
        mock_uuid.return_value = MagicMock()
        mock_uuid.return_value.__str__ = lambda x: "test-inference-uuid"
        mock_build.return_value = (
            "mock_inference_recipe.yaml",
            "s3://test/output",
            "s3://test/data",
        )
        mock_boto_client.return_value = MagicMock()

        # Create and persist an existing completed result
        existing_result = SMTJBatchInferenceResult(
            job_id="existing-inference-123",
            started_time=datetime.now(),
            inference_output_path="s3://test/inference/output",
        )

        # Persist the result with same parameters that will be used in batch_inference call
        persist_result(
            self.customizer,
            existing_result,
            "test-inference-job",
            "inference",
            input_path="s3://test-bucket/input",
            output_s3_path="s3://test-bucket/output",
            model_path=None,
            endpoint=None,
            recipe_path=None,
            overrides={},
        )

        # Mock the result to appear completed by mocking the SageMaker API call
        mock_sagemaker = MagicMock()
        mock_sagemaker.describe_training_job.return_value = {
            "TrainingJobStatus": "Completed",
            "CheckpointConfig": {"S3Uri": "s3://test/checkpoint"},
            "OutputDataConfig": {"S3OutputPath": "s3://test/output"},
        }
        with patch("boto3.client", return_value=mock_sagemaker):
            result = self.customizer.batch_inference(
                job_name="test-inference-job",
                input_path="s3://test-bucket/input",
                output_s3_path="s3://test-bucket/output",
            )

            # Should return existing result without calling runtime manager
            self.mock_runtime_manager.execute.assert_not_called()
            self.assertEqual(result.job_id, "existing-inference-123")

    def test_job_caching_disabled_by_default(self):
        """Test that job caching is disabled by default"""
        with tempfile.TemporaryDirectory() as temp_dir:
            with (
                patch("boto3.client") as mock_boto,
                patch(
                    "sagemaker.core.helper.session_helper.get_execution_role"
                ) as mock_get_execution_role,
            ):
                mock_get_execution_role.return_value = (
                    "arn:aws:iam::123456789012:role/SageMakerExecutionRole"
                )

                # Test default - job caching disabled by default
                customizer, _, client_side_effect = self._create_aws_mocked_customizer(
                    temp_dir=None,
                    customizer_params={
                        "generated_recipe_dir": None,
                    },
                )
                mock_boto.side_effect = client_side_effect

                # Job caching is disabled by default
                self.assertFalse(customizer.enable_job_caching)
                self.assertFalse(should_persist_results(customizer))

                # Test with enable_job_caching=True - uses temp_dir for job_cache_dir
                customizer_enabled, _, client_side_effect2 = self._create_aws_mocked_customizer(
                    temp_dir=temp_dir,
                    customizer_params={
                        "generated_recipe_dir": None,
                        "enable_job_caching": True,
                    },
                )
                mock_boto.side_effect = client_side_effect2

                # With enable_job_caching=True, persistence is enabled
                self.assertTrue(customizer_enabled.enable_job_caching)
                self.assertTrue(should_persist_results(customizer_enabled))

    def test_job_caching_with_failed_jobs(self):
        """Test that failed jobs are not returned by job caching"""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Use helper method to create mocked customizer with failed job status
            with (
                patch("boto3.client") as mock_boto,
                patch(
                    "sagemaker.core.helper.session_helper.get_execution_role"
                ) as mock_get_execution_role,
            ):
                mock_get_execution_role.return_value = (
                    "arn:aws:iam::123456789012:role/SageMakerExecutionRole"
                )

                customizer, mock_sagemaker, client_side_effect = self._create_aws_mocked_customizer(
                    temp_dir,
                    training_job_status="Failed",
                    customizer_params={"enable_job_caching": True},
                )
                mock_boto.side_effect = client_side_effect

                # Create and persist a failed result
                failed_result = SMTJTrainingResult(
                    job_id="failed-job-456",
                    started_time=datetime.now(),
                    method=TrainingMethod.SFT_LORA,
                    model_type=Model.NOVA_LITE_2,
                    model_artifacts=ModelArtifacts(
                        checkpoint_s3_path="s3://test/checkpoint",
                        output_s3_path="s3://test/output",
                    ),
                    sagemaker_client=mock_sagemaker,
                )

                job_params = {
                    "data_s3_path": "s3://test-bucket/data.jsonl",
                    "instance_type": "ml.g5.12xlarge",
                    "instance_count": 1,
                    "method": TrainingMethod.SFT_LORA,
                    "model": Model.NOVA_LITE_2,
                    "overrides": {"max_epochs": 10},
                }

                # Persist the failed result
                persist_result(customizer, failed_result, "failed-job", "training", **job_params)

                # Try to load it - should return None because job failed
                loaded_failed = load_existing_result(
                    customizer, "failed-job", "training", **job_params
                )
                self.assertIsNone(loaded_failed, "Should not return failed job results")

    def test_job_cache_hash_collision_handling(self):
        """Test that hash collisions are handled correctly"""
        with tempfile.TemporaryDirectory() as temp_dir:
            with (
                patch("boto3.client") as mock_boto,
                patch(
                    "sagemaker.core.helper.session_helper.get_execution_role"
                ) as mock_get_execution_role,
            ):
                mock_get_execution_role.return_value = (
                    "arn:aws:iam::123456789012:role/SageMakerExecutionRole"
                )

                customizer, mock_sagemaker, client_side_effect = self._create_aws_mocked_customizer(
                    temp_dir,
                    customizer_params={
                        "model": Model.NOVA_LITE_2,
                        "method": TrainingMethod.SFT_LORA,
                        "data_s3_path": "s3://test-bucket/data.jsonl",
                        "generated_recipe_dir": temp_dir,
                        "enable_job_caching": True,
                    },
                )
                mock_boto.side_effect = client_side_effect

                job_params1 = {
                    "data_s3_path": "s3://test-bucket/data.jsonl",
                    "instance_type": "ml.g5.12xlarge",
                    "instance_count": 1,
                    "method": TrainingMethod.SFT_LORA,
                    "model": Model.NOVA_LITE_2,
                    "overrides": {"max_epochs": 3},
                }

                job_params2 = {
                    "data_s3_path": "s3://test-bucket/data.jsonl",
                    "instance_type": "ml.g5.12xlarge",
                    "instance_count": 1,
                    "method": TrainingMethod.SFT_LORA,
                    "model": Model.NOVA_LITE_2,
                    "overrides": {"max_epochs": 5},  # Different parameter
                }

                # Create and persist first result with job_params1
                result1 = SMTJTrainingResult(
                    job_id="job-1",
                    started_time=datetime.now(),
                    method=TrainingMethod.SFT_LORA,
                    model_type=Model.NOVA_LITE_2,
                    model_artifacts=ModelArtifacts(
                        checkpoint_s3_path="s3://test/checkpoint",
                        output_s3_path="s3://test/output",
                    ),
                    sagemaker_client=mock_sagemaker,
                )
                persist_result(customizer, result1, "same-job", "training", **job_params1)

                # Small delay to ensure different timestamps
                import time

                time.sleep(0.001)

                # Create and persist second result with job_params2 (same job name, different params)
                result2 = SMTJTrainingResult(
                    job_id="job-2",
                    started_time=datetime.now(),
                    method=TrainingMethod.SFT_LORA,
                    model_type=Model.NOVA_LITE_2,
                    model_artifacts=ModelArtifacts(
                        checkpoint_s3_path="s3://test/checkpoint",
                        output_s3_path="s3://test/output",
                    ),
                    sagemaker_client=mock_sagemaker,
                )
                persist_result(customizer, result2, "same-job", "training", **job_params2)

                # Should have 2 files with same job name but different timestamps
                files = list(Path(temp_dir).glob("*.json"))
                self.assertEqual(
                    len(files),
                    2,
                    f"Expected 2 files with same job name but different params, found {len(files)}",
                )

                # Loading with job_params1 should return result1
                loaded_result1 = load_existing_result(
                    customizer, "same-job", "training", **job_params1
                )
                self.assertIsNotNone(loaded_result1, "Should find matching result for job_params1")
                self.assertEqual(loaded_result1.job_id, "job-1")

                # Loading with job_params2 should return result2
                loaded_result2 = load_existing_result(
                    customizer, "same-job", "training", **job_params2
                )
                self.assertIsNotNone(loaded_result2, "Should find matching result for job_params2")
                self.assertEqual(loaded_result2.job_id, "job-2")

    def test_complete_job_caching_workflow(self):
        """Test complete job caching workflow with real JobResult objects"""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Use helper method to create mocked customizer
            with (
                patch("boto3.client") as mock_boto,
                patch(
                    "sagemaker.core.helper.session_helper.get_execution_role"
                ) as mock_get_execution_role,
            ):
                mock_get_execution_role.return_value = (
                    "arn:aws:iam::123456789012:role/SageMakerExecutionRole"
                )

                customizer, mock_sagemaker, client_side_effect = self._create_aws_mocked_customizer(
                    temp_dir, customizer_params={"enable_job_caching": True}
                )
                mock_boto.side_effect = client_side_effect

                # Step 1: Create and persist a training result
                original_result = SMTJTrainingResult(
                    job_id="original-job-123",
                    started_time=datetime.now(),
                    method=TrainingMethod.SFT_LORA,
                    model_type=Model.NOVA_LITE_2,
                    model_artifacts=ModelArtifacts(
                        checkpoint_s3_path="s3://test/checkpoint",
                        output_s3_path="s3://test/output",
                    ),
                    sagemaker_client=mock_sagemaker,
                )

                # Persist the result with specific parameters
                job_params = {
                    "data_s3_path": "s3://test-bucket/data.jsonl",
                    "instance_type": "ml.g5.12xlarge",
                    "instance_count": 1,
                    "method": TrainingMethod.SFT_LORA,
                    "model": Model.NOVA_LITE_2,
                    "overrides": {"max_epochs": 3},
                }

                persist_result(customizer, original_result, "test-job", "training", **job_params)

                # Verify file was created
                files = list(Path(temp_dir).glob("*.json"))
                self.assertEqual(
                    len(files),
                    1,
                    f"Expected 1 file after persistence, found {len(files)}",
                )

                # Step 2: Try to load existing result with same parameters
                loaded_result = load_existing_result(
                    customizer, "test-job", "training", **job_params
                )

                self.assertIsNotNone(
                    loaded_result,
                    "Should have found existing result with matching parameters",
                )
                self.assertEqual(
                    loaded_result.job_id,
                    "original-job-123",
                    f"Expected job_id 'original-job-123', got '{loaded_result.job_id}'",
                )
                self.assertTrue(
                    hasattr(loaded_result, "_job_cache_hash"),
                    "Loaded result should have job cache hash",
                )

                # Step 3: Try to load with different parameters (should not find)
                different_params = job_params.copy()
                different_params["overrides"] = {"max_epochs": 5}  # Different parameter

                loaded_different = load_existing_result(
                    customizer, "test-job", "training", **different_params
                )
                self.assertIsNone(
                    loaded_different, "Should not find result with different parameters"
                )

    def test_job_caching_checks_all_override_parameters(self):
        """Test that job caching checks all override parameters, not just defaults"""
        # Create hashes with custom override parameter not in DEFAULT_OVERRIDE_PARAMS
        hash1 = generate_job_hash(
            self.customizer,
            "test-job",
            "training",
            overrides={"custom_param": "value1", "lr": 0.001},
        )
        hash2 = generate_job_hash(
            self.customizer,
            "test-job",
            "training",
            overrides={"custom_param": "value2", "lr": 0.001},
        )

        # Should not match because custom_param is different
        self.assertFalse(
            matches_job_cache_criteria(self.customizer._job_caching_config, hash1, hash2)
        )

        # Create hashes with same custom parameter
        hash3 = generate_job_hash(
            self.customizer,
            "test-job",
            "training",
            overrides={"custom_param": "value1", "lr": 0.001},
        )
        hash4 = generate_job_hash(
            self.customizer,
            "test-job",
            "training",
            overrides={"custom_param": "value1", "lr": 0.001},
        )

        # Should match because all parameters are the same
        self.assertTrue(
            matches_job_cache_criteria(self.customizer._job_caching_config, hash3, hash4)
        )

    def test_matches_criteria_include_infra(self):
        """Test that include_infra=True causes infra differences to reject cache hits."""
        hash1 = "instance_type:aaaa,model:bbbb,method:cccc"
        hash2 = "instance_type:xxxx,model:bbbb,method:cccc"
        config = self.customizer._job_caching_config.model_copy(update={"include_infra": True})
        self.assertFalse(matches_job_cache_criteria(config, hash1, hash2))
        # Same infra should match
        self.assertTrue(matches_job_cache_criteria(config, hash1, hash1))

    def test_matches_criteria_exclude_params(self):
        """Test that exclude_params removes specific fields from comparison."""
        hash1 = "model:aaaa,method:bbbb,data_s3_path:cccc,job_type:dddd,model_path:eeee"
        hash2 = "model:xxxx,method:bbbb,data_s3_path:cccc,job_type:dddd,model_path:eeee"
        config = self.customizer._job_caching_config.model_copy(
            update={"exclude_params": ["model"]}
        )
        # model differs but is excluded, so should match
        self.assertTrue(matches_job_cache_criteria(config, hash1, hash2))

    def test_matches_criteria_wildcard_exclude(self):
        """Test that exclude_params=['*'] skips all built-in groups, only checking include_params."""
        hash1 = "model:aaaa,method:bbbb,custom:1111"
        hash2 = "model:xxxx,method:yyyy,custom:1111"
        config = JobCachingConfig(exclude_params=["*"], include_params=["custom"])
        # Everything differs except custom, but wildcard exclude skips built-in checks
        self.assertTrue(matches_job_cache_criteria(config, hash1, hash2))
        # Different custom should fail even with wildcard exclude
        hash3 = "model:xxxx,method:yyyy,custom:9999"
        self.assertFalse(matches_job_cache_criteria(config, hash1, hash3))

    def test_matches_criteria_include_core_false(self):
        """Test that include_core=False skips core field comparison."""
        hash1 = "model:aaaa,method:bbbb,data_s3_path:cccc,job_type:dddd,model_path:eeee"
        hash2 = "model:xxxx,method:bbbb,data_s3_path:cccc,job_type:dddd,model_path:eeee"
        config = self.customizer._job_caching_config.model_copy(update={"include_core": False})
        # model differs but core is not checked
        self.assertTrue(matches_job_cache_criteria(config, hash1, hash2))

    def test_collect_all_parameters_happy_path(self):
        """Test that collect_all_parameters merges infra, customizer, and job params with correct priority."""
        params = collect_all_parameters(
            self.customizer, "test-job", "training", custom_key="custom_value"
        )
        # Customizer fields present
        self.assertEqual(params["model"], self.customizer.model.value)
        self.assertEqual(params["method"], self.customizer.method.value)
        self.assertEqual(params["data_s3_path"], self.customizer.data_s3_path)
        # Job params present and highest priority
        self.assertEqual(params["custom_key"], "custom_value")
        # Infra fields prefixed with infra_
        self.assertIn("infra_instance_count", params)

    def test_collect_all_parameters_skips_private_and_callable(self):
        """Test that infra private attrs and callables are excluded."""
        self.customizer.infra._private = "secret"
        self.customizer.infra.some_method = lambda: None
        params = collect_all_parameters(self.customizer, "test-job", "training")
        self.assertNotIn("infra__private", params)
        self.assertNotIn("infra_some_method", params)
        # Cleanup
        del self.customizer.infra._private
        del self.customizer.infra.some_method


class TestResolveDeployPlatform(unittest.TestCase):
    """Tests for the _resolve_deploy_platform standalone helper."""

    def test_sagemaker_arn_returns_sagemaker(self):
        arn = "arn:aws:sagemaker:us-east-1:123456789012:endpoint/my-endpoint"
        assert _resolve_deploy_platform(arn, None) == DeployPlatform.SAGEMAKER

    def test_bedrock_arn_returns_bedrock_pt(self):
        arn = "arn:aws:bedrock:us-east-1:123456789012:custom-model-deployment/my-model"
        assert _resolve_deploy_platform(arn, None) == DeployPlatform.BEDROCK_PT

    def test_endpoint_info_returns_its_platform(self):
        info = EndpointInfo(
            platform=DeployPlatform.SAGEMAKER,
            endpoint_name="ep",
            uri="arn:aws:sagemaker:us-east-1:123456789012:endpoint/ep",
            model_artifact_path="s3://bucket/path",
        )
        assert _resolve_deploy_platform(None, info) == DeployPlatform.SAGEMAKER

    def test_neither_provided_returns_none(self):
        assert _resolve_deploy_platform(None, None) is None

    def test_arn_takes_precedence_over_endpoint_info(self):
        info = EndpointInfo(
            platform=DeployPlatform.BEDROCK_PT,
            endpoint_name="ep",
            uri="arn:aws:bedrock:us-east-1:123456789012:custom-model-deployment/m",
            model_artifact_path="s3://bucket/path",
        )
        sm_arn = "arn:aws:sagemaker:us-east-1:123456789012:endpoint/my-endpoint"
        assert _resolve_deploy_platform(sm_arn, info) == DeployPlatform.SAGEMAKER


class TestInvokeInferenceExtraInfo(unittest.TestCase):
    """Tests for the _invoke_inference_extra_info telemetry helper."""

    def test_sagemaker_arn_in_kwargs(self):
        obj = MagicMock()
        obj.endpoint_info = None
        obj.model = Model.NOVA_MICRO
        result = _invoke_inference_extra_info(
            obj, endpoint_arn="arn:aws:sagemaker:us-east-1:123456789012:endpoint/ep"
        )
        assert result == {
            "model": Model.NOVA_MICRO.value,
            "platform": DeployPlatform.SAGEMAKER,
        }

    def test_bedrock_arn_in_kwargs(self):
        obj = MagicMock()
        obj.endpoint_info = None
        obj.model = Model.NOVA_MICRO
        result = _invoke_inference_extra_info(
            obj,
            endpoint_arn="arn:aws:bedrock:us-east-1:123456789012:custom-model-deployment/m",
        )
        assert result == {
            "model": Model.NOVA_MICRO.value,
            "platform": DeployPlatform.BEDROCK_PT,
        }

    def test_no_arn_with_endpoint_info(self):
        obj = MagicMock()
        obj.endpoint_info = EndpointInfo(
            platform=DeployPlatform.SAGEMAKER,
            endpoint_name="ep",
            uri="arn:aws:sagemaker:us-east-1:123456789012:endpoint/ep",
            model_artifact_path="s3://bucket/path",
        )
        obj.model = Model.NOVA_MICRO
        result = _invoke_inference_extra_info(obj)
        assert result == {
            "model": Model.NOVA_MICRO.value,
            "platform": DeployPlatform.SAGEMAKER,
        }

    def test_no_arn_no_endpoint_info_returns_unknown(self):
        obj = MagicMock()
        obj.endpoint_info = None
        obj.model = Model.NOVA_MICRO
        result = _invoke_inference_extra_info(obj)
        assert result == {"model": Model.NOVA_MICRO.value, "platform": "UNKNOWN"}


class TestDataMixingServerlessPlatform(unittest.TestCase):
    """Tests for data mixing support on SMTJServerless in NovaModelCustomizer."""

    def setUp(self):
        self.data_s3_path = "s3://bucket/data.jsonl"
        self.output_s3_path = "s3://bucket/output"

    def _patch_clients(self, mock_client):
        mock_sts = MagicMock()
        mock_sts.get_caller_identity.return_value = {"Account": "123456789012"}
        mock_s3 = MagicMock()
        mock_s3.head_bucket.return_value = {}

        def client_side_effect(service, **kw):
            if service == "sts":
                return mock_sts
            elif service == "s3":
                return mock_s3
            return MagicMock()

        mock_client.side_effect = client_side_effect

    @patch("amzn_nova_forge.model.nova_model_customizer.load_recipe_templates")
    @patch("boto3.client")
    def test_smtj_serverless_sft_lora_data_mixing_succeeds(self, mock_client, mock_load_recipes):
        """SMTJServerless + SFT_LORA + data_mixing_enabled=True must not raise."""
        self._patch_clients(mock_client)
        mock_load_recipes.return_value = (MagicMock(), MagicMock(), MagicMock(), MagicMock())
        infra = create_autospec(SMTJServerlessRuntimeManager)
        infra.platform = Platform.SMTJServerless

        customizer = NovaModelCustomizer(
            model=Model.NOVA_MICRO,
            method=TrainingMethod.SFT_LORA,
            infra=infra,
            data_s3_path=self.data_s3_path,
            output_s3_path=self.output_s3_path,
            data_mixing_enabled=True,
        )
        self.assertTrue(customizer.data_mixing_enabled)
        mock_load_recipes.assert_called_once()

    @patch("amzn_nova_forge.model.nova_model_customizer.load_recipe_templates")
    @patch("boto3.client")
    def test_smtj_serverless_sft_full_data_mixing_succeeds(self, mock_client, mock_load_recipes):
        """SMTJServerless + SFT_FULL + data_mixing_enabled=True must not raise."""
        self._patch_clients(mock_client)
        mock_load_recipes.return_value = (MagicMock(), MagicMock(), MagicMock(), MagicMock())
        infra = create_autospec(SMTJServerlessRuntimeManager)
        infra.platform = Platform.SMTJServerless

        customizer = NovaModelCustomizer(
            model=Model.NOVA_MICRO,
            method=TrainingMethod.SFT_FULL,
            infra=infra,
            data_s3_path=self.data_s3_path,
            output_s3_path=self.output_s3_path,
            data_mixing_enabled=True,
        )
        self.assertTrue(customizer.data_mixing_enabled)

    @patch("amzn_nova_forge.model.nova_model_customizer.load_recipe_templates")
    def test_init_data_mixing_accepts_smtj_serverless(self, mock_load_recipes):
        """_init_data_mixing must not raise for Platform.SMTJServerless + SFT_LORA."""
        mock_load_recipes.return_value = (MagicMock(), MagicMock(), MagicMock(), MagicMock())
        infra = create_autospec(SMTJServerlessRuntimeManager)
        infra.platform = Platform.SMTJServerless

        with patch("boto3.client"):
            customizer = NovaModelCustomizer(
                model=Model.NOVA_MICRO,
                method=TrainingMethod.SFT_LORA,
                infra=infra,
                data_s3_path=self.data_s3_path,
                output_s3_path=self.output_s3_path,
                data_mixing_enabled=False,  # don't trigger _init_data_mixing in __init__
            )

        # Call _init_data_mixing directly — must not raise
        mock_load_recipes.reset_mock()
        customizer.data_mixing_enabled = True
        customizer._init_data_mixing(
            model=Model.NOVA_MICRO,
            method=TrainingMethod.SFT_LORA,
            platform=Platform.SMTJServerless,
        )
        # Verify load_recipe_templates was called with SMTJServerless + data_mixing_enabled=True
        mock_load_recipes.assert_called()
        call_kwargs = mock_load_recipes.call_args.kwargs
        self.assertEqual(call_kwargs["platform"], Platform.SMTJServerless)
        self.assertTrue(call_kwargs["data_mixing_enabled"])

    @patch("boto3.client")
    def test_smtj_serverless_unsupported_method_raises(self, mock_client):
        """SMTJServerless + RFT_LORA + data_mixing_enabled=True must raise."""
        self._patch_clients(mock_client)
        infra = create_autospec(SMTJServerlessRuntimeManager)
        infra.platform = Platform.SMTJServerless

        with self.assertRaises(ValueError) as ctx:
            NovaModelCustomizer(
                model=Model.NOVA_MICRO,
                method=TrainingMethod.RFT_LORA,
                infra=infra,
                data_s3_path=self.data_s3_path,
                output_s3_path=self.output_s3_path,
                data_mixing_enabled=True,
            )
        self.assertIn("Data mixing is only supported for", str(ctx.exception))
        self.assertIn("SMTJServerless", str(ctx.exception))

    @patch("boto3.client")
    def test_smtj_non_serverless_still_raises(self, mock_client):
        """Regular SMTJ (non-serverless) must still raise for data mixing."""
        self._patch_clients(mock_client)
        infra = create_autospec(SMTJRuntimeManager)
        infra.platform = Platform.SMTJ

        with self.assertRaises(ValueError) as ctx:
            NovaModelCustomizer(
                model=Model.NOVA_MICRO,
                method=TrainingMethod.SFT_LORA,
                infra=infra,
                data_s3_path=self.data_s3_path,
                output_s3_path=self.output_s3_path,
                data_mixing_enabled=True,
            )
        self.assertIn("Data mixing is only supported for", str(ctx.exception))


class TestNovaModelCustomizerGetConfig(unittest.TestCase):
    def setUp(self):
        self.model = Model.NOVA_MICRO
        self.method = TrainingMethod.SFT_LORA
        self.data_s3_path = "s3://test-bucket/data"
        self.output_s3_path = "s3://test-bucket/output"

        self.mock_runtime_manager = create_autospec(SMTJRuntimeManager)
        self.mock_runtime_manager.platform = Platform.SMTJ
        self.mock_runtime_manager.instance_count = 2
        self.mock_runtime_manager.kms_key_id = None

        p_session = patch("boto3.session.Session")
        mock_session = p_session.start()
        self.addCleanup(p_session.stop)
        type(mock_session.return_value).region_name = PropertyMock(return_value="us-east-1")

        p_set_output = patch(
            "amzn_nova_forge.model.nova_model_customizer.set_output_s3_path",
            return_value="s3://sagemaker-nova-123456789012-us-east-1/output",
        )
        p_set_output.start()
        self.addCleanup(p_set_output.stop)

    def _make_customizer(self):
        return NovaModelCustomizer(
            model=self.model,
            method=self.method,
            infra=self.mock_runtime_manager,
            data_s3_path=self.data_s3_path,
            output_s3_path=self.output_s3_path,
        )

    @patch("amzn_nova_forge.trainer.forge_trainer.ForgeTrainer.get_config")
    @patch("amzn_nova_forge.trainer.forge_trainer.ForgeTrainer.__init__", return_value=None)
    def test_get_config_emits_deprecation_warning(self, mock_init, mock_get_config):
        from amzn_nova_forge.core.types import RecipeConfig

        mock_get_config.return_value = RecipeConfig(
            model=self.model,
            method=self.method,
            platform=Platform.SMTJ,
            parameters=(),
        )

        customizer = self._make_customizer()
        with self.assertWarns(DeprecationWarning):
            customizer.get_config()

    @patch("amzn_nova_forge.trainer.forge_trainer.ForgeTrainer.get_config")
    @patch("amzn_nova_forge.trainer.forge_trainer.ForgeTrainer.__init__", return_value=None)
    def test_get_config_delegates_to_forge_trainer(self, mock_init, mock_get_config):
        from amzn_nova_forge.core.types import ConfigParameter, RecipeConfig

        expected = RecipeConfig(
            model=self.model,
            method=self.method,
            platform=Platform.SMTJ,
            parameters=(ConfigParameter(name="lr", type="float", default=5e-6),),
        )
        mock_get_config.return_value = expected

        customizer = self._make_customizer()
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            result = customizer.get_config()

        self.assertEqual(result, expected)
        mock_get_config.assert_called_once_with(overrides=None)

    @patch("amzn_nova_forge.trainer.forge_trainer.ForgeTrainer.get_config")
    @patch("amzn_nova_forge.trainer.forge_trainer.ForgeTrainer.__init__", return_value=None)
    def test_get_config_passes_overrides(self, mock_init, mock_get_config):
        from amzn_nova_forge.core.types import RecipeConfig

        mock_get_config.return_value = RecipeConfig(
            model=self.model,
            method=self.method,
            platform=Platform.SMTJ,
            parameters=(),
        )

        customizer = self._make_customizer()
        overrides = {"lr": 1e-5}
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            customizer.get_config(overrides=overrides)

        mock_get_config.assert_called_once_with(overrides=overrides)


class TestCreateCustomModelDataSource(TestNovaModelCustomizer):
    """Tests for facade-level create_custom_model() with custom_model_data_source."""

    def test_raises_when_no_source_provided(self):
        """Validation at facade level requires at least one source."""
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            with self.assertRaises(ValueError) as ctx:
                self.customizer.create_custom_model()

        self.assertIn("custom_model_data_source", str(ctx.exception))

    def test_delegates_custom_model_data_source_to_deployer(self):
        """custom_model_data_source is passed through to ForgeDeployer."""
        import warnings

        data_source = {"modelPackageArnDataSource": {"modelPackageArn": "arn:pkg"}}

        with patch.object(self.customizer, "_build_deployer") as mock_build:
            mock_deployer = MagicMock()
            mock_deployer.create_custom_model.return_value = MagicMock(
                model_arn="arn:test", model_name="test"
            )
            mock_build.return_value = mock_deployer

            with warnings.catch_warnings():
                warnings.simplefilter("ignore", DeprecationWarning)
                self.customizer.create_custom_model(custom_model_data_source=data_source)

            mock_deployer.create_custom_model.assert_called_once_with(
                model_artifact_path=None,
                endpoint_name=None,
                execution_role_name=None,
                tags=None,
                skip_model_reuse=False,
                custom_model_data_source=data_source,
            )


if __name__ == "__main__":
    unittest.main()
