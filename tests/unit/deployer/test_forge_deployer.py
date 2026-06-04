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
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

from amzn_nova_forge.core.constants import ESCROW_URI_TAG_KEY
from amzn_nova_forge.core.enums import (
    DeploymentMode,
    DeployPlatform,
    Model,
    TrainingMethod,
)
from amzn_nova_forge.core.result.job_result import JobStatus
from amzn_nova_forge.core.types import DeploymentResult, EndpointInfo, ForgeConfig
from amzn_nova_forge.deployer.forge_deployer import ForgeDeployer

PATCH_PREFIX = "amzn_nova_forge.deployer.forge_deployer"


@patch(f"{PATCH_PREFIX}.validate_region")
class TestForgeDeployerInit(unittest.TestCase):
    """Constructor tests."""

    def test_happy_path_defaults(self, mock_validate_region):
        deployer = ForgeDeployer(region="us-east-1", model=Model.NOVA_MICRO)
        self.assertEqual(deployer.region, "us-east-1")
        self.assertEqual(deployer.model, Model.NOVA_MICRO)
        self.assertEqual(deployer.deployment_mode, DeploymentMode.FAIL_IF_EXISTS)
        self.assertIsNone(deployer.method)
        self.assertIsInstance(deployer._config, ForgeConfig)
        mock_validate_region.assert_called_once_with("us-east-1")

    def test_unsupported_region_raises(self, mock_validate_region):
        mock_validate_region.side_effect = ValueError("not supported")
        with self.assertRaises(ValueError):
            ForgeDeployer(region="invalid-region", model=Model.NOVA_MICRO)

    def test_deployment_mode_default(self, mock_validate_region):
        deployer = ForgeDeployer(region="us-east-1", model=Model.NOVA_LITE)
        self.assertEqual(deployer.deployment_mode, DeploymentMode.FAIL_IF_EXISTS)

    def test_method_stored(self, mock_validate_region):
        deployer = ForgeDeployer(
            region="us-east-1",
            model=Model.NOVA_MICRO,
            method=TrainingMethod.SFT_LORA,
        )
        self.assertEqual(deployer.method, TrainingMethod.SFT_LORA)


@patch(f"{PATCH_PREFIX}.find_bedrock_model_by_tag", return_value=None)
@patch(f"{PATCH_PREFIX}.validate_region")
class TestDeployBedrock(unittest.TestCase):
    """Tests for deploy() targeting Bedrock platforms."""

    def _make_deployer(self, **kwargs):
        defaults = dict(region="us-east-1", model=Model.NOVA_MICRO)
        defaults.update(kwargs)
        return ForgeDeployer(**defaults)

    # ---- Bedrock OD happy path ----

    @patch(f"{PATCH_PREFIX}.check_existing_deployment", return_value=None)
    @patch(f"{PATCH_PREFIX}.monitor_model_create")
    @patch(f"{PATCH_PREFIX}.create_bedrock_execution_role")
    @patch("boto3.client")
    def test_successful_bedrock_od_deployment(
        self,
        mock_boto_client,
        mock_create_role,
        mock_monitor,
        mock_check_existing,
        mock_validate_region,
        mock_find_by_tag,
    ):
        mock_bedrock = MagicMock()
        mock_iam = MagicMock()

        def client_side_effect(service, **kwargs):
            if service == "bedrock":
                return mock_bedrock
            if service == "iam":
                return mock_iam
            return MagicMock()

        mock_boto_client.side_effect = client_side_effect

        mock_create_role.return_value = {
            "Role": {"Arn": "arn:aws:iam::123456789012:role/BedrockRole"}
        }
        mock_bedrock.create_custom_model.return_value = {
            "modelArn": "arn:aws:bedrock:us-east-1:123456789012:custom-model/my-model"
        }
        mock_bedrock.create_custom_model_deployment.return_value = {
            "customModelDeploymentArn": "arn:aws:bedrock:us-east-1:123456789012:deployment/my-deploy"
        }

        deployer = self._make_deployer()
        result = deployer.deploy(
            model_artifact_path="s3://bucket/model",
            deploy_platform=DeployPlatform.BEDROCK_OD,
        )

        self.assertIsInstance(result, DeploymentResult)
        self.assertEqual(result.endpoint.platform, DeployPlatform.BEDROCK_OD)
        self.assertEqual(
            result.endpoint.uri,
            "arn:aws:bedrock:us-east-1:123456789012:deployment/my-deploy",
        )
        mock_bedrock.create_custom_model.assert_called_once()
        mock_monitor.assert_called_once()

    # ---- Bedrock PT uses create_provisioned_model_throughput ----

    @patch(f"{PATCH_PREFIX}.check_existing_deployment", return_value=None)
    @patch(f"{PATCH_PREFIX}.monitor_model_create")
    @patch(f"{PATCH_PREFIX}.create_bedrock_execution_role")
    @patch("boto3.client")
    def test_bedrock_pt_uses_provisioned_throughput(
        self,
        mock_boto_client,
        mock_create_role,
        mock_monitor,
        mock_check_existing,
        mock_validate_region,
        mock_find_by_tag,
    ):
        mock_bedrock = MagicMock()
        mock_iam = MagicMock()

        def client_side_effect(service, **kwargs):
            if service == "bedrock":
                return mock_bedrock
            if service == "iam":
                return mock_iam
            return MagicMock()

        mock_boto_client.side_effect = client_side_effect

        mock_create_role.return_value = {
            "Role": {"Arn": "arn:aws:iam::123456789012:role/BedrockRole"}
        }
        mock_bedrock.create_custom_model.return_value = {
            "modelArn": "arn:aws:bedrock:us-east-1:123456789012:custom-model/my-model"
        }
        mock_bedrock.create_provisioned_model_throughput.return_value = {
            "provisionedModelArn": "arn:aws:bedrock:us-east-1:123456789012:pt/my-pt"
        }

        deployer = self._make_deployer()
        result = deployer.deploy(
            model_artifact_path="s3://bucket/model",
            deploy_platform=DeployPlatform.BEDROCK_PT,
            unit_count=2,
        )

        self.assertIsInstance(result, DeploymentResult)
        self.assertEqual(result.endpoint.platform, DeployPlatform.BEDROCK_PT)
        mock_bedrock.create_provisioned_model_throughput.assert_called_once()
        call_kwargs = mock_bedrock.create_provisioned_model_throughput.call_args[1]
        self.assertEqual(call_kwargs["modelUnits"], 2)

    # ---- FAIL_IF_EXISTS raises when deployment exists ----

    @patch(
        f"{PATCH_PREFIX}.check_existing_deployment",
        return_value="arn:aws:bedrock:us-east-1:123456789012:deployment/existing",
    )
    @patch("boto3.client")
    def test_fail_if_exists_raises(
        self,
        mock_boto_client,
        mock_check_existing,
        mock_validate_region,
        mock_find_by_tag,
    ):
        mock_boto_client.return_value = MagicMock()

        deployer = self._make_deployer(deployment_mode=DeploymentMode.FAIL_IF_EXISTS)
        with self.assertRaises(Exception) as ctx:
            deployer.deploy(
                model_artifact_path="s3://bucket/model",
                deploy_platform=DeployPlatform.BEDROCK_OD,
            )
        self.assertIn("already exists", str(ctx.exception))

    # ---- UPDATE_IF_EXISTS attempts PT update ----

    @patch(f"{PATCH_PREFIX}.update_provisioned_throughput_model")
    @patch(f"{PATCH_PREFIX}.Validator._validate_calling_role_permissions")
    @patch(f"{PATCH_PREFIX}.get_required_bedrock_update_permissions", return_value=[])
    @patch(
        f"{PATCH_PREFIX}.check_existing_deployment",
        return_value="arn:aws:bedrock:us-east-1:123456789012:pt/existing-pt",
    )
    @patch(f"{PATCH_PREFIX}.monitor_model_create")
    @patch(f"{PATCH_PREFIX}.create_bedrock_execution_role")
    @patch("boto3.client")
    def test_update_if_exists_attempts_pt_update(
        self,
        mock_boto_client,
        mock_create_role,
        mock_monitor,
        mock_check_existing,
        mock_get_perms,
        mock_validate_perms,
        mock_update_pt,
        mock_validate_region,
        mock_find_by_tag,
    ):
        mock_bedrock = MagicMock()
        mock_iam = MagicMock()

        def client_side_effect(service, **kwargs):
            if service == "bedrock":
                return mock_bedrock
            if service == "iam":
                return mock_iam
            return MagicMock()

        mock_boto_client.side_effect = client_side_effect

        mock_create_role.return_value = {
            "Role": {"Arn": "arn:aws:iam::123456789012:role/BedrockRole"}
        }
        mock_bedrock.create_custom_model.return_value = {
            "modelArn": "arn:aws:bedrock:us-east-1:123456789012:custom-model/my-model"
        }

        deployer = self._make_deployer(deployment_mode=DeploymentMode.UPDATE_IF_EXISTS)
        result = deployer.deploy(
            model_artifact_path="s3://bucket/model",
            deploy_platform=DeployPlatform.BEDROCK_PT,
        )

        mock_update_pt.assert_called_once_with(
            "arn:aws:bedrock:us-east-1:123456789012:pt/existing-pt",
            "arn:aws:bedrock:us-east-1:123456789012:custom-model/my-model",
            "nova-micro-us-east-1",
            region="us-east-1",
        )
        self.assertEqual(
            result.endpoint.uri,
            "arn:aws:bedrock:us-east-1:123456789012:pt/existing-pt",
        )

    # ---- UPDATE_IF_EXISTS fails for non-PT platform ----

    @patch(
        f"{PATCH_PREFIX}.check_existing_deployment",
        return_value="arn:aws:bedrock:us-east-1:123456789012:deployment/existing",
    )
    @patch("boto3.client")
    def test_update_if_exists_fails_for_non_pt(
        self,
        mock_boto_client,
        mock_check_existing,
        mock_validate_region,
        mock_find_by_tag,
    ):
        mock_boto_client.return_value = MagicMock()

        deployer = self._make_deployer(deployment_mode=DeploymentMode.UPDATE_IF_EXISTS)
        with self.assertRaises(Exception) as ctx:
            deployer.deploy(
                model_artifact_path="s3://bucket/model",
                deploy_platform=DeployPlatform.BEDROCK_OD,
            )
        self.assertIn("UPDATE_IF_EXISTS", str(ctx.exception))
        self.assertIn("Provisioned Throughput", str(ctx.exception))

    # ---- Existing Bedrock model ARN skips model creation ----

    @patch(f"{PATCH_PREFIX}.check_existing_deployment", return_value=None)
    @patch(f"{PATCH_PREFIX}.create_bedrock_execution_role")
    @patch("boto3.client")
    def test_existing_bedrock_model_arn_skips_creation(
        self,
        mock_boto_client,
        mock_create_role,
        mock_check_existing,
        mock_validate_region,
        mock_find_by_tag,
    ):
        mock_bedrock = MagicMock()
        mock_iam = MagicMock()

        def client_side_effect(service, **kwargs):
            if service == "bedrock":
                return mock_bedrock
            if service == "iam":
                return mock_iam
            return MagicMock()

        mock_boto_client.side_effect = client_side_effect

        mock_create_role.return_value = {
            "Role": {"Arn": "arn:aws:iam::123456789012:role/BedrockRole"}
        }
        model_arn = "arn:aws:bedrock:us-east-1:123456789012:custom-model/existing"
        mock_bedrock.create_custom_model_deployment.return_value = {
            "customModelDeploymentArn": "arn:aws:bedrock:us-east-1:123456789012:deployment/deploy"
        }

        deployer = self._make_deployer()
        result = deployer.deploy(
            model_artifact_path=model_arn,
            deploy_platform=DeployPlatform.BEDROCK_OD,
        )

        mock_bedrock.create_custom_model.assert_not_called()
        self.assertIsInstance(result, DeploymentResult)

    # ---- Endpoint name auto-generation ----

    @patch(f"{PATCH_PREFIX}.check_existing_deployment", return_value=None)
    @patch(f"{PATCH_PREFIX}.monitor_model_create")
    @patch(f"{PATCH_PREFIX}.create_bedrock_execution_role")
    @patch("boto3.client")
    def test_endpoint_name_includes_method(
        self,
        mock_boto_client,
        mock_create_role,
        mock_monitor,
        mock_check_existing,
        mock_validate_region,
        mock_find_by_tag,
    ):
        mock_bedrock = MagicMock()
        mock_iam = MagicMock()

        def client_side_effect(service, **kwargs):
            if service == "bedrock":
                return mock_bedrock
            if service == "iam":
                return mock_iam
            return MagicMock()

        mock_boto_client.side_effect = client_side_effect

        mock_create_role.return_value = {
            "Role": {"Arn": "arn:aws:iam::123456789012:role/BedrockRole"}
        }
        mock_bedrock.create_custom_model.return_value = {
            "modelArn": "arn:aws:bedrock:us-east-1:123456789012:custom-model/m"
        }
        mock_bedrock.create_custom_model_deployment.return_value = {
            "customModelDeploymentArn": "arn:aws:bedrock:us-east-1:123456789012:deployment/d"
        }

        deployer = self._make_deployer(method=TrainingMethod.SFT_LORA)
        result = deployer.deploy(
            model_artifact_path="s3://bucket/model",
            deploy_platform=DeployPlatform.BEDROCK_OD,
        )

        self.assertIn("sft-lora", result.endpoint.endpoint_name)
        self.assertIn("us-east-1", result.endpoint.endpoint_name)

    @patch(f"{PATCH_PREFIX}.check_existing_deployment", return_value=None)
    @patch(f"{PATCH_PREFIX}.monitor_model_create")
    @patch(f"{PATCH_PREFIX}.create_bedrock_execution_role")
    @patch("boto3.client")
    def test_endpoint_name_without_method(
        self,
        mock_boto_client,
        mock_create_role,
        mock_monitor,
        mock_check_existing,
        mock_validate_region,
        mock_find_by_tag,
    ):
        mock_bedrock = MagicMock()
        mock_iam = MagicMock()

        def client_side_effect(service, **kwargs):
            if service == "bedrock":
                return mock_bedrock
            if service == "iam":
                return mock_iam
            return MagicMock()

        mock_boto_client.side_effect = client_side_effect

        mock_create_role.return_value = {
            "Role": {"Arn": "arn:aws:iam::123456789012:role/BedrockRole"}
        }
        mock_bedrock.create_custom_model.return_value = {
            "modelArn": "arn:aws:bedrock:us-east-1:123456789012:custom-model/m"
        }
        mock_bedrock.create_custom_model_deployment.return_value = {
            "customModelDeploymentArn": "arn:aws:bedrock:us-east-1:123456789012:deployment/d"
        }

        deployer = self._make_deployer()  # method=None by default
        result = deployer.deploy(
            model_artifact_path="s3://bucket/model",
            deploy_platform=DeployPlatform.BEDROCK_OD,
        )

        # Should NOT contain any method substring
        self.assertNotIn("sft", result.endpoint.endpoint_name)
        self.assertIn("us-east-1", result.endpoint.endpoint_name)

    # ---- KMS key handling ----

    @patch(f"{PATCH_PREFIX}.check_existing_deployment", return_value=None)
    @patch(f"{PATCH_PREFIX}.monitor_model_create")
    @patch(f"{PATCH_PREFIX}.create_bedrock_execution_role")
    @patch("boto3.client")
    def test_kms_key_full_arn_used_directly(
        self,
        mock_boto_client,
        mock_create_role,
        mock_monitor,
        mock_check_existing,
        mock_validate_region,
        mock_find_by_tag,
    ):
        mock_bedrock = MagicMock()
        mock_iam = MagicMock()

        def client_side_effect(service, **kwargs):
            if service == "bedrock":
                return mock_bedrock
            if service == "iam":
                return mock_iam
            return MagicMock()

        mock_boto_client.side_effect = client_side_effect

        kms_arn = "arn:aws:kms:us-east-1:123456789012:key/my-key-id"
        config = ForgeConfig(kms_key_id=kms_arn)
        mock_create_role.return_value = {
            "Role": {"Arn": "arn:aws:iam::123456789012:role/BedrockRole"}
        }
        mock_bedrock.create_custom_model.return_value = {
            "modelArn": "arn:aws:bedrock:us-east-1:123456789012:custom-model/m"
        }
        mock_bedrock.create_custom_model_deployment.return_value = {
            "customModelDeploymentArn": "arn:aws:bedrock:us-east-1:123456789012:deployment/d"
        }

        deployer = self._make_deployer(config=config)
        deployer.deploy(
            model_artifact_path="s3://bucket/model",
            deploy_platform=DeployPlatform.BEDROCK_OD,
        )

        create_kwargs = mock_bedrock.create_custom_model.call_args[1]
        self.assertEqual(create_kwargs["modelKmsKeyArn"], kms_arn)

    @patch(f"{PATCH_PREFIX}.check_existing_deployment", return_value=None)
    @patch(f"{PATCH_PREFIX}.monitor_model_create")
    @patch(f"{PATCH_PREFIX}.create_bedrock_execution_role")
    @patch("boto3.client")
    def test_kms_key_id_gets_constructed_to_arn(
        self,
        mock_boto_client,
        mock_create_role,
        mock_monitor,
        mock_check_existing,
        mock_validate_region,
        mock_find_by_tag,
    ):
        mock_bedrock = MagicMock()
        mock_iam = MagicMock()
        mock_sts = MagicMock()
        mock_sts.get_caller_identity.return_value = {"Account": "123456789012"}

        def client_side_effect(service, **kwargs):
            if service == "bedrock":
                return mock_bedrock
            if service == "iam":
                return mock_iam
            if service == "sts":
                return mock_sts
            return MagicMock()

        mock_boto_client.side_effect = client_side_effect

        config = ForgeConfig(kms_key_id="my-key-id")
        mock_create_role.return_value = {
            "Role": {"Arn": "arn:aws:iam::123456789012:role/BedrockRole"}
        }
        mock_bedrock.create_custom_model.return_value = {
            "modelArn": "arn:aws:bedrock:us-east-1:123456789012:custom-model/m"
        }
        mock_bedrock.create_custom_model_deployment.return_value = {
            "customModelDeploymentArn": "arn:aws:bedrock:us-east-1:123456789012:deployment/d"
        }

        deployer = self._make_deployer(config=config)
        deployer.deploy(
            model_artifact_path="s3://bucket/model",
            deploy_platform=DeployPlatform.BEDROCK_OD,
        )

        create_kwargs = mock_bedrock.create_custom_model.call_args[1]
        expected_arn = "arn:aws:kms:us-east-1:123456789012:key/my-key-id"
        self.assertEqual(create_kwargs["modelKmsKeyArn"], expected_arn)

    # ---- execution_role_name uses get_role ----

    @patch(f"{PATCH_PREFIX}.check_existing_deployment", return_value=None)
    @patch(f"{PATCH_PREFIX}.monitor_model_create")
    @patch("boto3.client")
    def test_execution_role_name_uses_get_role(
        self,
        mock_boto_client,
        mock_monitor,
        mock_check_existing,
        mock_validate_region,
        mock_find_by_tag,
    ):
        mock_bedrock = MagicMock()
        mock_iam = MagicMock()
        mock_iam.get_role.return_value = {
            "Role": {"Arn": "arn:aws:iam::123456789012:role/CustomRole"}
        }

        def client_side_effect(service, **kwargs):
            if service == "bedrock":
                return mock_bedrock
            if service == "iam":
                return mock_iam
            return MagicMock()

        mock_boto_client.side_effect = client_side_effect

        mock_bedrock.create_custom_model.return_value = {
            "modelArn": "arn:aws:bedrock:us-east-1:123456789012:custom-model/m"
        }
        mock_bedrock.create_custom_model_deployment.return_value = {
            "customModelDeploymentArn": "arn:aws:bedrock:us-east-1:123456789012:deployment/d"
        }

        deployer = self._make_deployer()
        deployer.deploy(
            model_artifact_path="s3://bucket/model",
            deploy_platform=DeployPlatform.BEDROCK_OD,
            execution_role_name="CustomRole",
        )

        mock_iam.get_role.assert_called_once_with(RoleName="CustomRole")


@patch(f"{PATCH_PREFIX}.find_sagemaker_model_by_tag", return_value=None)
@patch(f"{PATCH_PREFIX}.validate_region")
class TestDeploySageMaker(unittest.TestCase):
    """Tests for deploy() targeting SageMaker."""

    def _make_deployer(self, **kwargs):
        defaults = dict(region="us-east-1", model=Model.NOVA_MICRO)
        defaults.update(kwargs)
        return ForgeDeployer(**defaults)

    @patch(f"{PATCH_PREFIX}.create_sagemaker_endpoint")
    @patch(f"{PATCH_PREFIX}.create_sagemaker_model")
    @patch(f"{PATCH_PREFIX}._validate_sagemaker_instance_type_for_model_deployment")
    @patch(f"{PATCH_PREFIX}.create_sagemaker_execution_role")
    @patch("boto3.client")
    def test_successful_sagemaker_deployment(
        self,
        mock_boto_client,
        mock_create_role,
        mock_validate_instance,
        mock_create_model,
        mock_create_endpoint,
        mock_validate_region,
        mock_find_by_tag,
    ):
        mock_iam = MagicMock()

        def client_side_effect(service, **kwargs):
            if service == "iam":
                return mock_iam
            return MagicMock()

        mock_boto_client.side_effect = client_side_effect

        mock_create_role.return_value = {
            "Role": {"Arn": "arn:aws:iam::123456789012:role/SageMakerRole"}
        }
        mock_create_model.return_value = (
            "arn:aws:sagemaker:us-east-1:123456789012:model/my-ep-model"
        )
        mock_create_endpoint.return_value = (
            "arn:aws:sagemaker:us-east-1:123456789012:endpoint/my-ep"
        )

        deployer = self._make_deployer()
        result = deployer.deploy(
            model_artifact_path="s3://bucket/model",
            deploy_platform=DeployPlatform.SAGEMAKER,
        )

        self.assertIsInstance(result, DeploymentResult)
        self.assertEqual(result.endpoint.platform, DeployPlatform.SAGEMAKER)
        mock_create_endpoint.assert_called_once()

    @patch(f"{PATCH_PREFIX}.validate_unit_count")
    def test_sagemaker_instance_type_none_raises(
        self, mock_validate_unit, mock_validate_region, mock_find_by_tag
    ):
        deployer = self._make_deployer()
        with self.assertRaises(ValueError) as ctx:
            deployer.deploy(
                model_artifact_path="s3://bucket/model",
                deploy_platform=DeployPlatform.SAGEMAKER,
                sagemaker_instance_type=None,
            )
        self.assertIn("sagemaker_instance_type cannot be None", str(ctx.exception))

    @patch(f"{PATCH_PREFIX}.validate_unit_count")
    @patch(f"{PATCH_PREFIX}._validate_sagemaker_instance_type_for_model_deployment")
    def test_bedrock_model_arn_raises_for_sagemaker(
        self,
        mock_validate_instance,
        mock_validate_unit,
        mock_validate_region,
        mock_find_by_tag,
    ):
        deployer = self._make_deployer()
        with self.assertRaises(ValueError) as ctx:
            deployer.deploy(
                model_artifact_path="arn:aws:bedrock:us-east-1:123:custom-model/foo",
                deploy_platform=DeployPlatform.SAGEMAKER,
            )
        self.assertIn("Cannot deploy Bedrock-customized models", str(ctx.exception))

    @patch(f"{PATCH_PREFIX}.create_sagemaker_endpoint")
    @patch(f"{PATCH_PREFIX}.create_sagemaker_model")
    @patch(f"{PATCH_PREFIX}._validate_sagemaker_instance_type_for_model_deployment")
    @patch(f"{PATCH_PREFIX}.create_sagemaker_execution_role")
    @patch("boto3.client")
    def test_environment_variables_validated(
        self,
        mock_boto_client,
        mock_create_role,
        mock_validate_instance,
        mock_create_model,
        mock_create_endpoint,
        mock_validate_region,
        mock_find_by_tag,
    ):
        mock_iam = MagicMock()

        def client_side_effect(service, **kwargs):
            if service == "iam":
                return mock_iam
            return MagicMock()

        mock_boto_client.side_effect = client_side_effect

        mock_create_role.return_value = {
            "Role": {"Arn": "arn:aws:iam::123456789012:role/SageMakerRole"}
        }
        mock_create_model.return_value = "arn:aws:sagemaker:us-east-1:123456789012:model/ep-model"
        mock_create_endpoint.return_value = "arn:aws:sagemaker:us-east-1:123456789012:endpoint/ep"

        from amzn_nova_forge.validation.endpoint_validator import SageMakerEndpointEnvironment

        env = SageMakerEndpointEnvironment(context_length=8000, max_concurrency=4)
        deployer = self._make_deployer()
        deployer.deploy(
            model_artifact_path="s3://bucket/model",
            deploy_platform=DeployPlatform.SAGEMAKER,
            sagemaker_environment=env,
        )

        # Verify the model was created with the env dict
        mock_create_model.assert_called_once()
        call_kwargs = mock_create_model.call_args
        self.assertEqual(call_kwargs.kwargs.get("environment", {}).get("CONTEXT_LENGTH"), "8000")

    @patch(f"{PATCH_PREFIX}.create_sagemaker_endpoint")
    @patch(f"{PATCH_PREFIX}.create_sagemaker_model")
    @patch(f"{PATCH_PREFIX}._validate_sagemaker_instance_type_for_model_deployment")
    @patch(f"{PATCH_PREFIX}.create_sagemaker_execution_role")
    @patch("boto3.client")
    def test_sagemaker_endpoint_name_includes_method(
        self,
        mock_boto_client,
        mock_create_role,
        mock_validate_instance,
        mock_create_model,
        mock_create_endpoint,
        mock_validate_region,
        mock_find_by_tag,
    ):
        mock_iam = MagicMock()

        def client_side_effect(service, **kwargs):
            if service == "iam":
                return mock_iam
            return MagicMock()

        mock_boto_client.side_effect = client_side_effect

        mock_create_role.return_value = {
            "Role": {"Arn": "arn:aws:iam::123456789012:role/SageMakerRole"}
        }
        mock_create_model.return_value = "arn:aws:sagemaker:us-east-1:123456789012:model/ep-model"
        mock_create_endpoint.return_value = "arn:aws:sagemaker:us-east-1:123456789012:endpoint/ep"

        deployer = self._make_deployer(method=TrainingMethod.SFT_LORA)
        result = deployer.deploy(
            model_artifact_path="s3://bucket/model",
            deploy_platform=DeployPlatform.SAGEMAKER,
        )

        self.assertIn("sft-lora", result.endpoint.endpoint_name)
        self.assertIn("sagemaker", result.endpoint.endpoint_name)


@patch(f"{PATCH_PREFIX}.validate_region")
class TestGetStatus(unittest.TestCase):
    """Tests for get_status and get_status_by_arn."""

    def _make_deployer(self, **kwargs):
        defaults = dict(region="us-east-1", model=Model.NOVA_MICRO)
        defaults.update(kwargs)
        return ForgeDeployer(**defaults)

    def test_get_status_delegates_to_result_status(self, mock_validate_region):
        deployer = self._make_deployer()
        endpoint = EndpointInfo(
            platform=DeployPlatform.BEDROCK_OD,
            endpoint_name="ep",
            uri="arn:aws:bedrock:us-east-1:123:deployment/d",
            model_artifact_path="s3://bucket/model",
        )
        result = DeploymentResult(endpoint=endpoint, created_at=datetime.now(timezone.utc))

        with patch.object(
            DeploymentResult,
            "status",
            new_callable=lambda: property(lambda self: JobStatus.COMPLETED),
        ):
            status = deployer.get_status(result)
            self.assertEqual(status, JobStatus.COMPLETED)

    @patch(f"{PATCH_PREFIX}.check_deployment_status", return_value="InProgress")
    def test_get_status_by_arn_returns_job_status(self, mock_check_status, mock_validate_region):
        deployer = self._make_deployer()
        status = deployer.get_status_by_arn(
            "arn:aws:bedrock:us-east-1:123456789012:deployment/d", DeployPlatform.BEDROCK_OD
        )
        self.assertEqual(status, JobStatus.IN_PROGRESS)
        mock_check_status.assert_called_once_with(
            "arn:aws:bedrock:us-east-1:123456789012:deployment/d",
            DeployPlatform.BEDROCK_OD,
            region="us-east-1",
        )

    @patch(f"{PATCH_PREFIX}.check_deployment_status", return_value=None)
    def test_get_status_by_arn_returns_none_for_unknown(
        self, mock_check_status, mock_validate_region
    ):
        deployer = self._make_deployer()
        status = deployer.get_status_by_arn(
            "arn:aws:bedrock:us-east-1:123:deployment/d", DeployPlatform.BEDROCK_OD
        )
        self.assertIsNone(status)


@patch(f"{PATCH_PREFIX}.validate_region")
class TestGetLogs(unittest.TestCase):
    """Tests for get_logs."""

    def _make_deployer(self, **kwargs):
        defaults = dict(region="us-east-1", model=Model.NOVA_MICRO)
        defaults.update(kwargs)
        return ForgeDeployer(**defaults)

    @patch(f"{PATCH_PREFIX}.check_deployment_status", return_value="Completed")
    def test_get_logs_with_job_result(self, mock_check_status, mock_validate_region):
        deployer = self._make_deployer()
        endpoint = EndpointInfo(
            platform=DeployPlatform.BEDROCK_OD,
            endpoint_name="ep",
            uri="arn:aws:bedrock:us-east-1:123456789012:deployment/d",
            model_artifact_path="s3://bucket/model",
        )
        result = DeploymentResult(endpoint=endpoint, created_at=datetime.now(timezone.utc))

        deployer.get_logs(job_result=result)
        mock_check_status.assert_called_once_with(
            "arn:aws:bedrock:us-east-1:123456789012:deployment/d",
            DeployPlatform.BEDROCK_OD,
            region="us-east-1",
        )

    @patch(f"{PATCH_PREFIX}.check_sagemaker_deployment_status", return_value="InProgress")
    def test_get_logs_with_endpoint_arn_only(self, mock_check_status, mock_validate_region):
        deployer = self._make_deployer()
        deployer.get_logs(endpoint_arn="arn:aws:sagemaker:us-east-1:123456789012:endpoint/ep")
        mock_check_status.assert_called_once_with(
            "arn:aws:sagemaker:us-east-1:123456789012:endpoint/ep",
            region="us-east-1",
        )

    def test_get_logs_no_arn_raises_value_error(self, mock_validate_region):
        deployer = self._make_deployer()
        with self.assertRaises(ValueError) as ctx:
            deployer.get_logs()

        self.assertIn("endpoint_arn", str(ctx.exception))


@patch(f"{PATCH_PREFIX}.validate_region")
class TestCreateInferenceComponent(unittest.TestCase):
    """Tests for ForgeDeployer.create_inference_component()."""

    def _make_deployer(self, **kwargs):
        defaults = dict(region="us-east-1", model=Model.NOVA_MICRO)
        defaults.update(kwargs)
        return ForgeDeployer(**defaults)

    @patch(f"{PATCH_PREFIX}.create_inference_component")
    @patch("boto3.client")
    def test_successful_create_inference_component(
        self, mock_boto_client, mock_create_ic, mock_validate_region
    ):
        mock_sagemaker = MagicMock()
        mock_boto_client.return_value = mock_sagemaker

        expected_result = DeploymentResult(
            endpoint=EndpointInfo(
                platform=DeployPlatform.SAGEMAKER,
                endpoint_name="my-endpoint",
                uri="arn:aws:sagemaker:us-east-1:123456789012:inference-component/my-ic",
                model_artifact_path="my-model",
            ),
            created_at=datetime.now(timezone.utc),
        )
        mock_create_ic.return_value = expected_result

        deployer = self._make_deployer()
        result = deployer.create_inference_component(
            inference_component_name="my-ic",
            model_name="my-model",
            num_cpus=4,
            num_accelerators=1,
            min_memory_in_mb=8192,
            endpoint_name="my-endpoint",
            variant_name="primary",
            copy_count=1,
        )

        self.assertEqual(result, expected_result)
        mock_boto_client.assert_called_once_with("sagemaker", region_name="us-east-1")
        mock_create_ic.assert_called_once_with(
            inference_component_name="my-ic",
            endpoint_name="my-endpoint",
            variant_name="primary",
            model_name="my-model",
            num_cpus=4,
            num_accelerators=1,
            min_memory_in_mb=8192,
            copy_count=1,
            sagemaker_client=mock_sagemaker,
            region="us-east-1",
        )

    @patch(f"{PATCH_PREFIX}.create_inference_component")
    @patch("boto3.client")
    def test_create_inference_component_uses_default_variant_and_copy_count(
        self, mock_boto_client, mock_create_ic, mock_validate_region
    ):
        mock_sagemaker = MagicMock()
        mock_boto_client.return_value = mock_sagemaker
        mock_create_ic.return_value = MagicMock(spec=DeploymentResult)

        deployer = self._make_deployer()
        deployer.create_inference_component(
            inference_component_name="my-ic",
            model_name="my-model",
            num_cpus=2,
            num_accelerators=4,
            min_memory_in_mb=16384,
            endpoint_name="my-endpoint",
        )

        call_kwargs = mock_create_ic.call_args[1]
        self.assertEqual(call_kwargs["variant_name"], "primary")
        self.assertEqual(call_kwargs["copy_count"], 1)


@patch(f"{PATCH_PREFIX}.validate_region")
class TestMonitorInferenceComponent(unittest.TestCase):
    def _make_deployer(self, **kwargs):
        defaults = dict(region="us-east-1", model=Model.NOVA_MICRO)
        defaults.update(kwargs)
        return ForgeDeployer(**defaults)

    @patch(f"{PATCH_PREFIX}.monitor_inference_component")
    @patch("boto3.client")
    def test_successful_monitor_returns_status(
        self, mock_boto_client, mock_monitor_ic, mock_validate_region
    ):
        mock_sagemaker = MagicMock()
        mock_boto_client.return_value = mock_sagemaker
        mock_monitor_ic.return_value = "InService"

        deployer = self._make_deployer()
        result = deployer.monitor_inference_component(
            inference_component_name="my-ic",
        )

        self.assertEqual(result, "InService")
        mock_boto_client.assert_called_once_with("sagemaker", region_name="us-east-1")
        mock_monitor_ic.assert_called_once_with(
            inference_component_name="my-ic",
            sagemaker_client=mock_sagemaker,
        )

    @patch(f"{PATCH_PREFIX}.monitor_inference_component")
    @patch("boto3.client")
    def test_monitor_propagates_exception_on_failure(
        self, mock_boto_client, mock_monitor_ic, mock_validate_region
    ):
        mock_sagemaker = MagicMock()
        mock_boto_client.return_value = mock_sagemaker
        mock_monitor_ic.side_effect = Exception("Inference component 'my-ic' reached Failed status")

        deployer = self._make_deployer()
        with self.assertRaises(Exception) as ctx:
            deployer.monitor_inference_component(inference_component_name="my-ic")
        self.assertIn("Failed status", str(ctx.exception))


@patch(f"{PATCH_PREFIX}.find_sagemaker_model_by_tag", return_value=None)
@patch(f"{PATCH_PREFIX}.validate_region")
class TestDeploySageMakerWithInferenceComponentConfigs(unittest.TestCase):
    """Tests for deploy() with inference_component_configs on SageMaker."""

    def _make_deployer(self, **kwargs):
        defaults = dict(region="us-east-1", model=Model.NOVA_MICRO)
        defaults.update(kwargs)
        return ForgeDeployer(**defaults)

    def _make_ic_config(self, **overrides):
        from amzn_nova_forge.util.sagemaker import InferenceComponentConfig

        defaults = dict(
            inference_component_name="my-ic",
            num_cpus=15,
            num_accelerators=4,
            min_memory_in_mb=25000,
        )
        defaults.update(overrides)
        return InferenceComponentConfig(**defaults)

    @patch(f"{PATCH_PREFIX}.create_sagemaker_endpoint")
    @patch(f"{PATCH_PREFIX}.create_sagemaker_model")
    @patch(f"{PATCH_PREFIX}._validate_sagemaker_instance_type_for_model_deployment")
    @patch(f"{PATCH_PREFIX}.validate_inference_component_resources")
    @patch(f"{PATCH_PREFIX}.create_sagemaker_execution_role")
    @patch("boto3.client")
    def test_validate_inference_component_resources_called_for_each_config(
        self,
        mock_boto_client,
        mock_create_role,
        mock_validate_ic_resources,
        mock_validate_instance,
        mock_create_model,
        mock_create_endpoint,
        mock_validate_region,
        mock_find_by_tag,
    ):
        mock_iam = MagicMock()

        def client_side_effect(service, **kwargs):
            if service == "iam":
                return mock_iam
            return MagicMock()

        mock_boto_client.side_effect = client_side_effect
        mock_create_role.return_value = {
            "Role": {"Arn": "arn:aws:iam::123456789012:role/SageMakerRole"}
        }
        mock_create_model.return_value = (
            "arn:aws:sagemaker:us-east-1:123456789012:model/my-ep-model"
        )
        mock_create_endpoint.return_value = (
            "arn:aws:sagemaker:us-east-1:123456789012:endpoint/my-ep"
        )

        ic_config_1 = self._make_ic_config(inference_component_name="ic-1")
        ic_config_2 = self._make_ic_config(inference_component_name="ic-2")

        deployer = self._make_deployer()
        deployer.deploy(
            model_artifact_path="s3://bucket/model",
            deploy_platform=DeployPlatform.SAGEMAKER,
            inference_component_configs=[ic_config_1, ic_config_2],
        )

        self.assertEqual(mock_validate_ic_resources.call_count, 2)
        mock_validate_ic_resources.assert_any_call(ic_config_1, Model.NOVA_MICRO)
        mock_validate_ic_resources.assert_any_call(ic_config_2, Model.NOVA_MICRO)

    @patch(f"{PATCH_PREFIX}.create_sagemaker_endpoint")
    @patch(f"{PATCH_PREFIX}.create_sagemaker_model")
    @patch(f"{PATCH_PREFIX}._validate_sagemaker_instance_type_for_model_deployment")
    @patch(f"{PATCH_PREFIX}.validate_inference_component_resources")
    @patch(f"{PATCH_PREFIX}.create_sagemaker_execution_role")
    @patch(f"{PATCH_PREFIX}.logger")
    @patch("boto3.client")
    def test_multi_ic_warning_logged_when_more_than_one_config(
        self,
        mock_boto_client,
        mock_logger,
        mock_create_role,
        mock_validate_ic_resources,
        mock_validate_instance,
        mock_create_model,
        mock_create_endpoint,
        mock_validate_region,
        mock_find_by_tag,
    ):
        mock_iam = MagicMock()

        def client_side_effect(service, **kwargs):
            if service == "iam":
                return mock_iam
            return MagicMock()

        mock_boto_client.side_effect = client_side_effect
        mock_create_role.return_value = {
            "Role": {"Arn": "arn:aws:iam::123456789012:role/SageMakerRole"}
        }
        mock_create_model.return_value = (
            "arn:aws:sagemaker:us-east-1:123456789012:model/my-ep-model"
        )
        mock_create_endpoint.return_value = (
            "arn:aws:sagemaker:us-east-1:123456789012:endpoint/my-ep"
        )

        ic_configs = [
            self._make_ic_config(inference_component_name="ic-1"),
            self._make_ic_config(inference_component_name="ic-2"),
            self._make_ic_config(inference_component_name="ic-3"),
        ]

        deployer = self._make_deployer()
        deployer.deploy(
            model_artifact_path="s3://bucket/model",
            deploy_platform=DeployPlatform.SAGEMAKER,
            inference_component_configs=ic_configs,
        )

        mock_logger.warning.assert_called()
        warning_msg = mock_logger.warning.call_args[0][0]
        self.assertIn("3", warning_msg)
        self.assertIn("inference components", warning_msg)

    @patch(f"{PATCH_PREFIX}.create_sagemaker_endpoint")
    @patch(f"{PATCH_PREFIX}.create_sagemaker_model")
    @patch(f"{PATCH_PREFIX}._validate_sagemaker_instance_type_for_model_deployment")
    @patch(f"{PATCH_PREFIX}.validate_inference_component_resources")
    @patch(f"{PATCH_PREFIX}.create_sagemaker_execution_role")
    @patch(f"{PATCH_PREFIX}.logger")
    @patch("boto3.client")
    def test_no_multi_ic_warning_for_single_config(
        self,
        mock_boto_client,
        mock_logger,
        mock_create_role,
        mock_validate_ic_resources,
        mock_validate_instance,
        mock_create_model,
        mock_create_endpoint,
        mock_validate_region,
        mock_find_by_tag,
    ):
        mock_iam = MagicMock()

        def client_side_effect(service, **kwargs):
            if service == "iam":
                return mock_iam
            return MagicMock()

        mock_boto_client.side_effect = client_side_effect
        mock_create_role.return_value = {
            "Role": {"Arn": "arn:aws:iam::123456789012:role/SageMakerRole"}
        }
        mock_create_model.return_value = (
            "arn:aws:sagemaker:us-east-1:123456789012:model/my-ep-model"
        )
        mock_create_endpoint.return_value = (
            "arn:aws:sagemaker:us-east-1:123456789012:endpoint/my-ep"
        )

        ic_configs = [self._make_ic_config(inference_component_name="ic-1")]

        deployer = self._make_deployer()
        deployer.deploy(
            model_artifact_path="s3://bucket/model",
            deploy_platform=DeployPlatform.SAGEMAKER,
            inference_component_configs=ic_configs,
        )

        # No warning should be logged about multiple inference components
        for call in mock_logger.warning.call_args_list:
            self.assertNotIn("inference components", call[0][0])

    @patch(f"{PATCH_PREFIX}.create_sagemaker_endpoint")
    @patch(f"{PATCH_PREFIX}.create_sagemaker_model")
    @patch(f"{PATCH_PREFIX}._validate_sagemaker_instance_type_for_model_deployment")
    @patch(f"{PATCH_PREFIX}.validate_inference_component_resources")
    @patch(f"{PATCH_PREFIX}.create_sagemaker_execution_role")
    @patch("boto3.client")
    def test_create_sagemaker_endpoint_receives_ic_configs_and_role_arn(
        self,
        mock_boto_client,
        mock_create_role,
        mock_validate_ic_resources,
        mock_validate_instance,
        mock_create_model,
        mock_create_endpoint,
        mock_validate_region,
        mock_find_by_tag,
    ):
        mock_iam = MagicMock()

        def client_side_effect(service, **kwargs):
            if service == "iam":
                return mock_iam
            return MagicMock()

        mock_boto_client.side_effect = client_side_effect
        mock_create_role.return_value = {
            "Role": {"Arn": "arn:aws:iam::123456789012:role/SageMakerRole"}
        }
        mock_create_model.return_value = (
            "arn:aws:sagemaker:us-east-1:123456789012:model/my-ep-model"
        )
        mock_create_endpoint.return_value = (
            "arn:aws:sagemaker:us-east-1:123456789012:endpoint/my-ep"
        )

        ic_configs = [self._make_ic_config(inference_component_name="ic-1")]

        deployer = self._make_deployer()
        deployer.deploy(
            model_artifact_path="s3://bucket/model",
            deploy_platform=DeployPlatform.SAGEMAKER,
            inference_component_configs=ic_configs,
        )

        mock_create_endpoint.assert_called_once()
        call_kwargs = mock_create_endpoint.call_args[1]
        self.assertEqual(call_kwargs["inference_component_configs"], ic_configs)
        self.assertEqual(
            call_kwargs["execution_role_arn"],
            "arn:aws:iam::123456789012:role/SageMakerRole",
        )

    @patch(f"{PATCH_PREFIX}.create_sagemaker_endpoint")
    @patch(f"{PATCH_PREFIX}.create_sagemaker_model")
    @patch(f"{PATCH_PREFIX}._validate_sagemaker_instance_type_for_model_deployment")
    @patch(f"{PATCH_PREFIX}.create_sagemaker_execution_role")
    @patch("boto3.client")
    def test_create_sagemaker_endpoint_no_ic_kwargs_when_configs_none(
        self,
        mock_boto_client,
        mock_create_role,
        mock_validate_instance,
        mock_create_model,
        mock_create_endpoint,
        mock_validate_region,
        mock_find_by_tag,
    ):
        mock_iam = MagicMock()

        def client_side_effect(service, **kwargs):
            if service == "iam":
                return mock_iam
            return MagicMock()

        mock_boto_client.side_effect = client_side_effect
        mock_create_role.return_value = {
            "Role": {"Arn": "arn:aws:iam::123456789012:role/SageMakerRole"}
        }
        mock_create_model.return_value = (
            "arn:aws:sagemaker:us-east-1:123456789012:model/my-ep-model"
        )
        mock_create_endpoint.return_value = (
            "arn:aws:sagemaker:us-east-1:123456789012:endpoint/my-ep"
        )

        deployer = self._make_deployer()
        deployer.deploy(
            model_artifact_path="s3://bucket/model",
            deploy_platform=DeployPlatform.SAGEMAKER,
            inference_component_configs=[],
        )

        mock_create_endpoint.assert_called_once()
        call_kwargs = mock_create_endpoint.call_args[1]
        self.assertNotIn("inference_component_configs", call_kwargs)
        self.assertNotIn("execution_role_arn", call_kwargs)


class TestDeploymentResultRegion(unittest.TestCase):
    """Tests for region propagation through DeploymentResult.status."""

    def test_deployment_result_status_passes_region(self):
        endpoint = EndpointInfo(
            platform=DeployPlatform.BEDROCK_OD,
            endpoint_name="ep",
            uri="arn:test",
            model_artifact_path="s3://x",
            region="eu-west-1",
        )
        result = DeploymentResult(endpoint=endpoint, created_at=datetime.now(timezone.utc))
        with patch.object(DeploymentResult, "_status_checker") as mock_checker:
            mock_checker.return_value = "Active"
            _ = result.status
            mock_checker.assert_called_once_with("arn:test", DeployPlatform.BEDROCK_OD, "eu-west-1")


@patch(f"{PATCH_PREFIX}.find_sagemaker_model_by_tag", return_value=None)
@patch(f"{PATCH_PREFIX}.validate_region")
class TestDeploySageMakerModelPackageArn(unittest.TestCase):
    """Tests for deploy() auto-detecting model package ARNs for SageMaker."""

    def _make_deployer(self, **kwargs):
        defaults = dict(region="us-east-1", model=Model.NOVA_MICRO)
        defaults.update(kwargs)
        return ForgeDeployer(**defaults)

    @patch(f"{PATCH_PREFIX}.create_sagemaker_endpoint")
    @patch(f"{PATCH_PREFIX}.create_sagemaker_model")
    @patch(f"{PATCH_PREFIX}._validate_sagemaker_instance_type_for_model_deployment")
    @patch(f"{PATCH_PREFIX}.create_sagemaker_execution_role")
    @patch("boto3.client")
    def test_model_package_arn_sets_model_package_name(
        self,
        mock_boto_client,
        mock_create_role,
        mock_validate_instance,
        mock_create_model,
        mock_create_endpoint,
        mock_validate_region,
        mock_find_by_tag,
    ):
        """A model package ARN should be auto-detected and passed as model_package_name."""
        mock_iam = MagicMock()

        def client_side_effect(service, **kwargs):
            if service == "iam":
                return mock_iam
            return MagicMock()

        mock_boto_client.side_effect = client_side_effect

        mock_create_role.return_value = {
            "Role": {"Arn": "arn:aws:iam::123456789012:role/SageMakerRole"}
        }
        mock_create_model.return_value = "arn:aws:sagemaker:us-east-1:123456789012:model/my-model"
        mock_create_endpoint.return_value = (
            "arn:aws:sagemaker:us-east-1:123456789012:endpoint/my-ep"
        )

        model_package_arn = "arn:aws:sagemaker:us-east-1:123456789012:model-package/my-package/1"
        deployer = self._make_deployer()
        result = deployer.deploy(
            model_artifact_path=model_package_arn,
            deploy_platform=DeployPlatform.SAGEMAKER,
        )

        self.assertIsInstance(result, DeploymentResult)
        self.assertEqual(result.endpoint.platform, DeployPlatform.SAGEMAKER)

        # model_package_name should be the ARN itself
        call_kwargs = mock_create_model.call_args.kwargs
        self.assertEqual(call_kwargs["model_package_name"], model_package_arn)
        # model_s3_location should be None when model_package_name is set
        self.assertIsNone(call_kwargs["model_s3_location"])

    @patch(f"{PATCH_PREFIX}.create_sagemaker_endpoint")
    @patch(f"{PATCH_PREFIX}.create_sagemaker_model")
    @patch(f"{PATCH_PREFIX}._validate_sagemaker_instance_type_for_model_deployment")
    @patch(f"{PATCH_PREFIX}.create_sagemaker_execution_role")
    @patch("boto3.client")
    def test_model_package_arn_skips_trailing_slash_normalization(
        self,
        mock_boto_client,
        mock_create_role,
        mock_validate_instance,
        mock_create_model,
        mock_create_endpoint,
        mock_validate_region,
        mock_find_by_tag,
    ):
        """Model package ARN should NOT get a trailing slash appended."""
        mock_iam = MagicMock()

        def client_side_effect(service, **kwargs):
            if service == "iam":
                return mock_iam
            return MagicMock()

        mock_boto_client.side_effect = client_side_effect

        mock_create_role.return_value = {
            "Role": {"Arn": "arn:aws:iam::123456789012:role/SageMakerRole"}
        }
        mock_create_model.return_value = "arn:aws:sagemaker:us-east-1:123456789012:model/my-model"
        mock_create_endpoint.return_value = (
            "arn:aws:sagemaker:us-east-1:123456789012:endpoint/my-ep"
        )

        model_package_arn = "arn:aws:sagemaker:us-west-2:123456789012:model-package/nova-pkg"
        deployer = self._make_deployer()
        deployer.deploy(
            model_artifact_path=model_package_arn,
            deploy_platform=DeployPlatform.SAGEMAKER,
        )

        # The artifact_path passed to _deploy_to_sagemaker should be the ARN unchanged
        # (no trailing slash appended)
        call_kwargs = mock_create_model.call_args.kwargs
        self.assertFalse(
            call_kwargs.get("model_package_name", "").endswith("/"),
            "Model package ARN should not have a trailing slash",
        )
        self.assertEqual(call_kwargs["model_package_name"], model_package_arn)

    @patch(f"{PATCH_PREFIX}.create_sagemaker_endpoint")
    @patch(f"{PATCH_PREFIX}.create_sagemaker_model")
    @patch(f"{PATCH_PREFIX}._validate_sagemaker_instance_type_for_model_deployment")
    @patch(f"{PATCH_PREFIX}.create_sagemaker_execution_role")
    @patch("boto3.client")
    def test_s3_path_does_not_set_model_package_name(
        self,
        mock_boto_client,
        mock_create_role,
        mock_validate_instance,
        mock_create_model,
        mock_create_endpoint,
        mock_validate_region,
        mock_find_by_tag,
    ):
        """A regular S3 path should NOT set model_package_name (remains None)."""
        mock_iam = MagicMock()

        def client_side_effect(service, **kwargs):
            if service == "iam":
                return mock_iam
            return MagicMock()

        mock_boto_client.side_effect = client_side_effect

        mock_create_role.return_value = {
            "Role": {"Arn": "arn:aws:iam::123456789012:role/SageMakerRole"}
        }
        mock_create_model.return_value = "arn:aws:sagemaker:us-east-1:123456789012:model/my-model"
        mock_create_endpoint.return_value = (
            "arn:aws:sagemaker:us-east-1:123456789012:endpoint/my-ep"
        )

        deployer = self._make_deployer()
        deployer.deploy(
            model_artifact_path="s3://bucket/model/path",
            deploy_platform=DeployPlatform.SAGEMAKER,
        )

        call_kwargs = mock_create_model.call_args.kwargs
        self.assertIsNone(call_kwargs.get("model_package_name"))
        # S3 path should get trailing slash normalization
        self.assertEqual(call_kwargs["model_s3_location"], "s3://bucket/model/path/")


@patch(f"{PATCH_PREFIX}.find_bedrock_model_by_tag", return_value=None)
@patch(f"{PATCH_PREFIX}.validate_region")
class TestCreateCustomModel(unittest.TestCase):
    """Tests for create_custom_model() validation and data source handling."""

    def _make_deployer(self, **kwargs):
        defaults = dict(region="us-east-1", model=Model.NOVA_MICRO)
        defaults.update(kwargs)
        return ForgeDeployer(**defaults)

    @patch(f"{PATCH_PREFIX}.monitor_model_create")
    @patch(f"{PATCH_PREFIX}.create_bedrock_execution_role")
    @patch("boto3.client")
    def test_custom_model_data_source_happy_path(
        self,
        mock_boto_client,
        mock_create_role,
        mock_monitor,
        mock_validate_region,
        mock_find_by_tag,
    ):
        """Passing custom_model_data_source alone should use customModelDataSource in API call."""
        mock_bedrock = MagicMock()
        mock_iam = MagicMock()

        def client_side_effect(service, **kwargs):
            if service == "bedrock":
                return mock_bedrock
            if service == "iam":
                return mock_iam
            return MagicMock()

        mock_boto_client.side_effect = client_side_effect

        mock_create_role.return_value = {
            "Role": {"Arn": "arn:aws:iam::123456789012:role/BedrockRole"}
        }
        model_arn = "arn:aws:bedrock:us-east-1:123456789012:custom-model/my-model"
        mock_bedrock.create_custom_model.return_value = {"modelArn": model_arn}

        data_source = {
            "modelPackageArnDataSource": {
                "modelPackageArn": "arn:aws:sagemaker:us-east-1:123456789012:model-package/pkg/1"
            }
        }

        deployer = self._make_deployer()
        result = deployer.create_custom_model(custom_model_data_source=data_source)

        self.assertEqual(result.model_arn, model_arn)
        create_kwargs = mock_bedrock.create_custom_model.call_args[1]
        # Should use customModelDataSource, not modelSourceConfig
        self.assertEqual(create_kwargs["customModelDataSource"], data_source)
        self.assertNotIn("modelSourceConfig", create_kwargs)

    def test_both_model_artifact_and_data_source_raises(
        self, mock_validate_region, mock_find_by_tag
    ):
        """Passing both model_artifact_path and custom_model_data_source should raise ValueError."""
        deployer = self._make_deployer()
        with self.assertRaises(ValueError) as ctx:
            deployer.create_custom_model(
                model_artifact_path="s3://bucket/model",
                custom_model_data_source={"modelPackageArnDataSource": {"modelPackageArn": "arn"}},
            )
        self.assertIn("not both", str(ctx.exception))

    def test_neither_model_artifact_nor_data_source_raises(
        self, mock_validate_region, mock_find_by_tag
    ):
        """Passing neither model_artifact_path nor custom_model_data_source should raise ValueError."""
        deployer = self._make_deployer()
        with self.assertRaises(ValueError) as ctx:
            deployer.create_custom_model()
        self.assertIn("must be provided", str(ctx.exception))

    @patch(f"{PATCH_PREFIX}.monitor_model_create")
    @patch(f"{PATCH_PREFIX}.create_bedrock_execution_role")
    @patch("boto3.client")
    def test_escrow_path_extracted_from_model_package_arn_data_source(
        self,
        mock_boto_client,
        mock_create_role,
        mock_monitor,
        mock_validate_region,
        mock_find_by_tag,
    ):
        """Escrow path should be extracted from modelPackageArnDataSource.modelPackageArn."""
        mock_bedrock = MagicMock()
        mock_iam = MagicMock()

        def client_side_effect(service, **kwargs):
            if service == "bedrock":
                return mock_bedrock
            if service == "iam":
                return mock_iam
            return MagicMock()

        mock_boto_client.side_effect = client_side_effect

        mock_create_role.return_value = {
            "Role": {"Arn": "arn:aws:iam::123456789012:role/BedrockRole"}
        }
        model_arn = "arn:aws:bedrock:us-east-1:123456789012:custom-model/my-model"
        mock_bedrock.create_custom_model.return_value = {"modelArn": model_arn}

        package_arn = "arn:aws:sagemaker:us-east-1:123456789012:model-package/my-pkg/1"
        data_source = {"modelPackageArnDataSource": {"modelPackageArn": package_arn}}

        deployer = self._make_deployer()
        result = deployer.create_custom_model(custom_model_data_source=data_source)

        # The escrow_uri on the result should be the extracted model package ARN
        self.assertEqual(result.escrow_uri, package_arn)

    @patch(f"{PATCH_PREFIX}.monitor_model_create")
    @patch(f"{PATCH_PREFIX}.create_bedrock_execution_role")
    @patch("boto3.client")
    def test_escrow_path_fallback_for_unknown_data_source_structure(
        self,
        mock_boto_client,
        mock_create_role,
        mock_monitor,
        mock_validate_region,
        mock_find_by_tag,
    ):
        """When modelPackageArnDataSource is absent, a warning is emitted and no escrow tag is added."""
        mock_bedrock = MagicMock()
        mock_iam = MagicMock()

        def client_side_effect(service, **kwargs):
            if service == "bedrock":
                return mock_bedrock
            if service == "iam":
                return mock_iam
            return MagicMock()

        mock_boto_client.side_effect = client_side_effect

        mock_create_role.return_value = {
            "Role": {"Arn": "arn:aws:iam::123456789012:role/BedrockRole"}
        }
        model_arn = "arn:aws:bedrock:us-east-1:123456789012:custom-model/my-model"
        mock_bedrock.create_custom_model.return_value = {"modelArn": model_arn}

        data_source = {"someOtherSource": {"key": "value"}}

        deployer = self._make_deployer()
        with self.assertLogs("nova_forge_sdk", level="WARNING") as log:
            result = deployer.create_custom_model(custom_model_data_source=data_source)

        # Warning should mention inability to extract tag-safe identifier
        warning_found = any("tag-safe identifier" in msg for msg in log.output)
        self.assertTrue(warning_found, f"Expected warning not found in: {log.output}")

        # escrow_uri should be empty since no tag-safe value was available
        self.assertEqual(result.escrow_uri, "")

        # find_published_model should not have been called (no escrow_path)
        mock_find_by_tag.assert_not_called()

        # No escrow tag should be in the create_custom_model call
        call_kwargs = mock_bedrock.create_custom_model.call_args[1]
        model_tags = call_kwargs.get("modelTags", [])
        escrow_tags = [t for t in model_tags if t.get("key") == ESCROW_URI_TAG_KEY]
        self.assertEqual(len(escrow_tags), 0)


if __name__ == "__main__":
    unittest.main()
