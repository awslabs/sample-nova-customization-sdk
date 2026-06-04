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
from unittest.mock import MagicMock, Mock, patch

from amzn_nova_forge.core.enums import (
    EvaluationTask,
    Model,
    Platform,
    TrainingMethod,
)
from amzn_nova_forge.core.types import ValidationConfig
from amzn_nova_forge.manager.runtime_manager import (
    SMHPRuntimeManager,
    SMTJRuntimeManager,
)
from amzn_nova_forge.validation.validator import (
    CLUSTER_NAME_REGEX,
    JOB_NAME_REGEX,
    NAMESPACE_REGEX,
    Validator,
    validate_rft_lambda_name,
)


class TestValidator(unittest.TestCase):
    def setUp(self):
        self.mock_smhp_infra = Mock(spec=SMHPRuntimeManager)
        self.mock_smhp_infra.cluster_name = "test-cluster"
        self.mock_smhp_infra.instance_type = "ml.p5.48xlarge"
        self.mock_smhp_infra.instance_count = 4
        self.mock_smhp_infra.region = "us-east-1"

        self.mock_smtj_infra = Mock(spec=SMTJRuntimeManager)
        self.mock_smtj_infra.execution_role = "arn:aws:iam::123456789012:role/TestExecutionRole"
        self.mock_smtj_infra.instance_type = "ml.p5.48xlarge"
        self.mock_smtj_infra.region = "us-east-1"

    @patch("amzn_nova_forge.validation.validator.get_cluster_instance_info")
    @patch("amzn_nova_forge.validation.validator.boto3.client")
    @patch("sagemaker.core.helper.session_helper.get_execution_role")
    @patch(
        "amzn_nova_forge.manager.runtime_manager.SMHPRuntimeManager.required_calling_role_permissions"
    )
    def test_validate_smhp_infrastructure_success(
        self,
        mock_required_permissions,
        mock_get_execution_role,
        mock_boto3_client,
        mock_get_cluster_info,
    ):
        # Mock successful cluster info
        mock_get_cluster_info.return_value = {
            "normal_instance_groups": [],
            "restricted_instance_groups": [
                {
                    "instance_group_name": "worker-group",
                    "instance_type": "ml.p5.48xlarge",
                    "current_count": 4,
                    "target_count": 4,
                    "status": "InService",
                }
            ],
        }

        # Mock SageMaker execution role
        mock_get_execution_role.return_value = (
            "arn:aws:iam::123456789012:role/SageMakerExecutionRole"
        )

        # Mock successful IAM permissions
        mock_sagemaker_client = Mock()
        mock_iam_client = Mock()
        mock_iam_client.get_role.return_value = {
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

        # Mock STS client for calling role permissions
        mock_sts_client = Mock()
        mock_sts_client.get_caller_identity.return_value = {
            "Account": "123456789012",
            "Arn": "arn:aws:sts::123456789012:assumed-role/TestRole/session",
        }

        # Mock IAM simulation for calling role permissions
        mock_iam_client.simulate_principal_policy.return_value = {
            "EvaluationResults": [{"EvalDecision": "allowed"}]
        }

        def mock_client_factory(service, **kwargs):
            return {
                "sagemaker": mock_sagemaker_client,
                "iam": mock_iam_client,
                "sts": mock_sts_client,
            }[service]

        mock_boto3_client.side_effect = mock_client_factory

        try:
            with patch("amzn_nova_forge.manager.runtime_manager.SMHPRuntimeManager.setup"):
                smhp_infra = SMHPRuntimeManager("ml.p5.48xlarge", 4, "test-cluster", "kubeflow")

                Validator.validate(
                    platform=Platform.SMHP,
                    method=TrainingMethod.SFT_LORA,
                    infra=smhp_infra,
                    recipe={},
                    overrides_template={},
                    data_s3_path="s3://test-bucket/data.jsonl",
                    output_s3_path="s3://test-bucket/output/",
                )
        except ValueError:
            self.fail("validate() raised ValueError unexpectedly!")

    @patch("amzn_nova_forge.validation.validator.get_cluster_instance_info")
    @patch("amzn_nova_forge.validation.validator.boto3.client")
    @patch("sagemaker.core.helper.session_helper.get_execution_role")
    def test_validate_smhp_missing_instance_type(
        self, mock_get_execution_role, mock_boto3_client, mock_get_cluster_info
    ):
        # Mock cluster with different instance type
        mock_get_cluster_info.return_value = {
            "normal_instance_groups": [],
            "restricted_instance_groups": [
                {
                    "instance_group_name": "worker-group-1",
                    "instance_type": "ml.g5.12xlarge",
                    "current_count": 4,
                    "target_count": 4,
                    "status": "InService",
                },
                {
                    "instance_group_name": "worker-group-2",
                    "instance_type": "ml.g5.24xlarge",
                    "current_count": 2,
                    "target_count": 2,
                    "status": "InService",
                },
            ],
        }

        # Mock SageMaker execution role
        mock_get_execution_role.return_value = (
            "arn:aws:iam::123456789012:role/SageMakerExecutionRole"
        )

        # Mock successful IAM permissions and SageMaker cluster access
        mock_sagemaker_client = Mock()
        # TODO: Get rid of this, the above mock is sufficient.
        mock_sagemaker_client.describe_cluster.return_value = {
            "ClusterName": "test-cluster",
            "ClusterStatus": "InService",
            "InstanceGroups": [
                {
                    "InstanceGroupName": "worker-group-1",
                    "InstanceType": "ml.g5.12xlarge",
                    "CurrentCount": 4,
                    "TargetCount": 4,
                    "Status": "InService",
                },
                {
                    "InstanceGroupName": "worker-group-2",
                    "InstanceType": "ml.g5.24xlarge",
                    "CurrentCount": 2,
                    "TargetCount": 2,
                    "Status": "InService",
                },
            ],
        }
        mock_iam_client = Mock()
        mock_iam_client.get_role.return_value = {
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

        mock_boto3_client.side_effect = lambda service, **kwargs: {
            "sagemaker": mock_sagemaker_client,
            "iam": mock_iam_client,
        }[service]

        with self.assertRaises(ValueError) as context:
            with patch("amzn_nova_forge.manager.runtime_manager.SMHPRuntimeManager.setup"):
                smhp_infra = SMHPRuntimeManager("ml.p5.48xlarge", 4, "test-cluster", "kubeflow")

                Validator.validate(
                    platform=Platform.SMHP,
                    method=TrainingMethod.SFT_LORA,
                    infra=smhp_infra,
                    recipe={},
                    overrides_template={},
                )

        self.assertIn("Instance type 'ml.p5.48xlarge' not available", str(context.exception))
        self.assertIn(
            "Available types: ['ml.g5.12xlarge', 'ml.g5.24xlarge']",
            str(context.exception),
        )

    @patch("amzn_nova_forge.validation.validator.boto3.client")
    def test_validate_iam_permissions_directly(self, mock_boto3_client):
        """Test IAM validation method directly."""
        # Mock IAM client
        mock_iam_client = Mock()
        mock_iam_client.get_role.return_value = {
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
        mock_iam_client.list_role_policies.return_value = {"PolicyNames": ["S3Policy"]}
        mock_iam_client.get_role_policy.return_value = {
            "PolicyDocument": {
                "Statement": [
                    {
                        "Effect": "Allow",
                        "Action": ["s3:GetObject", "s3:PutObject", "s3:ListBucket"],
                        "Resource": [
                            "arn:aws:s3:::test-bucket",
                            "arn:aws:s3:::test-bucket/*",
                        ],
                    }
                ]
            }
        }
        mock_iam_client.list_attached_role_policies.return_value = {"AttachedPolicies": []}

        # Mock STS client to return same account (making it same-account role)
        mock_sts_client = Mock()
        mock_sts_client.get_caller_identity.return_value = {
            "Account": "123456789012",
            "Arn": "arn:aws:sts::123456789012:assumed-role/TestRole/session",
        }

        # Mock IAM simulation for calling role permissions
        mock_iam_client.simulate_principal_policy.return_value = {
            "EvaluationResults": [{"EvalDecision": "allowed"}]
        }

        def mock_client_factory(service_name, **kwargs):
            if service_name == "iam":
                return mock_iam_client
            elif service_name == "sts":
                return mock_sts_client
            return Mock()

        mock_boto3_client.side_effect = mock_client_factory

        with patch("sagemaker.core.helper.session_helper.get_execution_role") as mock_get_role:
            with patch(
                "amzn_nova_forge.manager.runtime_manager.SMTJRuntimeManager.required_calling_role_permissions",
                return_value=[],
            ):
                with patch("amzn_nova_forge.manager.runtime_manager.SMTJRuntimeManager.setup"):
                    mock_get_role.return_value = "arn:aws:iam::123456789012:role/ValidRole"

                    smtj_infra = SMTJRuntimeManager(
                        "ml.p5.48xlarge", 1, "arn:aws:iam::123456789012:role/ValidRole"
                    )
                    smtj_infra.execution_role = "arn:aws:iam::123456789012:role/ValidRole"

                errors = []
                Validator._validate_iam_permissions(
                    errors,
                    smtj_infra,
                    data_s3_path="s3://test-bucket/data.jsonl",
                    output_s3_path="s3://test-bucket/output/",
                )

                # Should not add any errors
                self.assertEqual(len(errors), 0)

    @patch("amzn_nova_forge.validation.validator.boto3.client")
    def test_validate_iam_permissions_invalid_trust_policy(self, mock_boto3_client):
        """Test IAM validation with invalid trust policy."""
        # Mock IAM client with invalid trust policy
        mock_iam_client = Mock()
        mock_iam_client.get_role.return_value = {
            "Role": {
                "AssumeRolePolicyDocument": {
                    "Statement": [
                        {
                            "Effect": "Allow",
                            "Principal": {"Service": "ec2.amazonaws.com"},  # Wrong service
                            "Action": "sts:AssumeRole",
                        }
                    ]
                }
            }
        }
        mock_iam_client.list_role_policies.return_value = {"PolicyNames": []}
        mock_iam_client.list_attached_role_policies.return_value = {"AttachedPolicies": []}

        # Mock STS client to return same account (making it same-account role)
        mock_sts_client = Mock()
        mock_sts_client.get_caller_identity.return_value = {
            "Account": "123456789012",
            "Arn": "arn:aws:sts::123456789012:assumed-role/TestRole/session",
        }

        # Mock IAM simulation for calling role permissions
        mock_iam_client.simulate_principal_policy.return_value = {
            "EvaluationResults": [{"EvalDecision": "allowed"}]
        }

        def mock_client_factory(service_name, **kwargs):
            if service_name == "iam":
                return mock_iam_client
            elif service_name == "sts":
                return mock_sts_client
            return Mock()

        mock_boto3_client.side_effect = mock_client_factory

        with patch("sagemaker.core.helper.session_helper.get_execution_role") as mock_get_role:
            with patch(
                "amzn_nova_forge.manager.runtime_manager.SMTJRuntimeManager.required_calling_role_permissions",
                return_value=[],
            ):
                with patch("amzn_nova_forge.manager.runtime_manager.SMTJRuntimeManager.setup"):
                    mock_get_role.return_value = "arn:aws:iam::123456789012:role/InvalidRole"

                    smtj_infra = SMTJRuntimeManager(
                        "ml.p5.48xlarge",
                        1,
                        "arn:aws:iam::123456789012:role/InvalidRole",
                    )
                    smtj_infra.execution_role = "arn:aws:iam::123456789012:role/InvalidRole"

                errors = []
                Validator._validate_iam_permissions(
                    errors,
                    smtj_infra,
                    data_s3_path="s3://test-bucket/data.jsonl",
                    output_s3_path="s3://test-bucket/output/",
                )

                # Should add error about trust policy AND missing permissions
                self.assertEqual(len(errors), 2)
            # Check that both trust policy and permissions errors are present
            error_text = " ".join(errors)
            self.assertIn("does not trust sagemaker.amazonaws.com service", error_text)
            self.assertIn("missing required permissions", error_text)

    @patch("amzn_nova_forge.validation.validator.boto3.client")
    def test_validate_iam_permissions_cross_account_success(self, mock_boto3_client):
        """Test IAM validation for cross-account role with successful assumption."""
        # Mock STS client for cross-account detection (different accounts)
        mock_sts_client = Mock()
        mock_sts_client.get_caller_identity.return_value = {
            "Account": "111111111111",
            "Arn": "arn:aws:sts::111111111111:assumed-role/TestRole/session",
        }

        # Mock successful assume role
        mock_sts_client.assume_role.return_value = {
            "Credentials": {
                "AccessKeyId": "test-key",
                "SecretAccessKey": "test-secret",
                "SessionToken": "test-token",
            }
        }

        # Mock IAM client with assumed credentials
        mock_iam_client = Mock()
        mock_iam_client.get_role.return_value = {
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
        mock_iam_client.list_role_policies.return_value = {"PolicyNames": ["S3Policy"]}
        mock_iam_client.get_role_policy.return_value = {
            "PolicyDocument": {
                "Statement": [
                    {
                        "Effect": "Allow",
                        "Action": ["s3:GetObject", "s3:PutObject", "s3:ListBucket"],
                        "Resource": [
                            "arn:aws:s3:::cross-account-bucket",
                            "arn:aws:s3:::cross-account-bucket/*",
                        ],
                    }
                ]
            }
        }
        mock_iam_client.list_attached_role_policies.return_value = {"AttachedPolicies": []}

        # Mock IAM simulation for calling role permissions
        mock_iam_client.simulate_principal_policy.return_value = {
            "EvaluationResults": [{"EvalDecision": "allowed"}]
        }

        def mock_client_factory(service_name, **kwargs):
            if service_name == "sts":
                return mock_sts_client
            elif service_name == "iam":
                return mock_iam_client
            return Mock()

        mock_boto3_client.side_effect = mock_client_factory

        with patch("sagemaker.core.helper.session_helper.get_execution_role") as mock_get_role:
            with patch(
                "amzn_nova_forge.manager.runtime_manager.SMTJRuntimeManager.required_calling_role_permissions",
                return_value=[],
            ):
                with patch("amzn_nova_forge.manager.runtime_manager.SMTJRuntimeManager.setup"):
                    mock_get_role.return_value = "arn:aws:iam::222222222222:role/CrossAccountRole"

                    smtj_infra = SMTJRuntimeManager(
                        "ml.p5.48xlarge",
                        1,
                        "arn:aws:iam::222222222222:role/CrossAccountRole",
                    )
                    smtj_infra.execution_role = "arn:aws:iam::222222222222:role/CrossAccountRole"

                errors = []
                Validator._validate_iam_permissions(
                    errors,
                    smtj_infra,
                    data_s3_path="s3://test-bucket/data.jsonl",
                    output_s3_path="s3://test-bucket/output/",
                )

                # Should not add any errors for successful cross-account validation
                self.assertEqual(len(errors), 0)

    @patch("amzn_nova_forge.validation.validator.boto3.client")
    def test_validate_iam_permissions_cross_account_assume_fail(self, mock_boto3_client):
        """Test IAM validation for cross-account role when assume role fails."""
        # Mock STS client for cross-account detection (different accounts)
        mock_sts_client = Mock()
        mock_sts_client.get_caller_identity.return_value = {
            "Account": "111111111111",
            "Arn": "arn:aws:sts::111111111111:assumed-role/TestRole/session",
        }

        # Mock failed assume role
        mock_sts_client.assume_role.side_effect = Exception("AccessDenied")

        # Mock IAM client for calling role permissions
        mock_iam_client = Mock()
        mock_iam_client.simulate_principal_policy.return_value = {
            "EvaluationResults": [{"EvalDecision": "allowed"}]
        }

        def mock_client_factory(service_name, **kwargs):
            if service_name == "sts":
                return mock_sts_client
            elif service_name == "iam":
                return mock_iam_client
            return Mock()

        mock_boto3_client.side_effect = mock_client_factory

        with patch("sagemaker.core.helper.session_helper.get_execution_role") as mock_get_role:
            with patch(
                "amzn_nova_forge.manager.runtime_manager.SMTJRuntimeManager.required_calling_role_permissions",
                return_value=[],
            ):
                with patch("amzn_nova_forge.manager.runtime_manager.SMTJRuntimeManager.setup"):
                    cross_account_role = "arn:aws:iam::222222222222:role/CrossAccountRole"
                    mock_get_role.return_value = cross_account_role

                    smtj_infra = SMTJRuntimeManager(
                        "ml.p5.48xlarge", 1, self.mock_smtj_infra.execution_role
                    )
                    smtj_infra.execution_role = cross_account_role

                    errors = []
                    Validator._validate_iam_permissions(
                        errors,
                        smtj_infra,
                        data_s3_path="s3://test-bucket/data.jsonl",
                        output_s3_path="s3://test-bucket/output/",
                    )

                    # Should not add any errors - silently skip validation for cross-account
                    self.assertEqual(len(errors), 0)

    @patch("amzn_nova_forge.validation.validator.boto3.client")
    def test_validate_iam_permissions_cross_account_limited_permissions(self, mock_boto3_client):
        """Test IAM validation for cross-account role with limited IAM permissions."""
        # Mock STS client for cross-account detection (different accounts)
        mock_sts_client = Mock()
        mock_sts_client.get_caller_identity.return_value = {
            "Account": "111111111111",
            "Arn": "arn:aws:sts::111111111111:assumed-role/TestRole/session",
        }

        # Mock successful assume role
        mock_sts_client.assume_role.return_value = {
            "Credentials": {
                "AccessKeyId": "test-key",
                "SecretAccessKey": "test-secret",
                "SessionToken": "test-token",
            }
        }

        # Mock IAM client that can get role but not policies
        mock_iam_client = Mock()
        mock_iam_client.get_role.return_value = {
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
        # Fail on policy operations
        mock_iam_client.list_role_policies.side_effect = Exception("AccessDenied")

        # Mock IAM simulation for calling role permissions
        mock_iam_client.simulate_principal_policy.return_value = {
            "EvaluationResults": [{"EvalDecision": "allowed"}]
        }

        def mock_client_factory(service_name, **kwargs):
            if service_name == "sts":
                return mock_sts_client
            elif service_name == "iam":
                return mock_iam_client
            return Mock()

        mock_boto3_client.side_effect = mock_client_factory

        with patch("sagemaker.core.helper.session_helper.get_execution_role") as mock_get_role:
            with patch(
                "amzn_nova_forge.manager.runtime_manager.SMTJRuntimeManager.required_calling_role_permissions",
                return_value=[],
            ):
                with patch("amzn_nova_forge.manager.runtime_manager.SMTJRuntimeManager.setup"):
                    mock_get_role.return_value = "arn:aws:iam::222222222222:role/CrossAccountRole"

                    smtj_infra = SMTJRuntimeManager(
                        "ml.p5.48xlarge",
                        1,
                        "arn:aws:iam::222222222222:role/CrossAccountRole",
                    )
                    smtj_infra.execution_role = "arn:aws:iam::222222222222:role/CrossAccountRole"

                    errors = []
                    Validator._validate_iam_permissions(
                        errors,
                        smtj_infra,
                        data_s3_path="s3://test-bucket/data.jsonl",
                        output_s3_path="s3://test-bucket/output/",
                    )

                # Should not add any errors - validates trust policy only
                self.assertEqual(len(errors), 0)

    @patch("amzn_nova_forge.validation.validator.get_cluster_instance_info")
    @patch("amzn_nova_forge.validation.validator.boto3.client")
    def test_validate_infrastructure_directly(self, mock_boto3_client, mock_get_cluster_info):
        """Test infrastructure validation method directly."""
        # Mock successful cluster info
        mock_get_cluster_info.return_value = {
            "normal_instance_groups": [],
            "restricted_instance_groups": [
                {
                    "instance_group_name": "worker-group",
                    "instance_type": "ml.p5.48xlarge",
                    "current_count": 4,
                    "target_count": 4,
                    "status": "InService",
                }
            ],
        }

        mock_sagemaker_client = Mock()
        mock_boto3_client.return_value = mock_sagemaker_client

        errors = []
        Validator._validate_infrastructure(self.mock_smhp_infra, errors)

        # Should not add any errors
        self.assertEqual(len(errors), 0)

    @patch("amzn_nova_forge.validation.validator.get_cluster_instance_info")
    @patch("amzn_nova_forge.validation.validator.boto3.client")
    def test_validate_infrastructure_insufficient_capacity(
        self, mock_boto3_client, mock_get_cluster_info
    ):
        """Test infrastructure validation with insufficient capacity."""
        # Mock cluster with insufficient capacity
        mock_get_cluster_info.return_value = {
            "normal_instance_groups": [],
            "restricted_instance_groups": [
                {
                    "instance_group_name": "worker-group",
                    "instance_type": "ml.p5.48xlarge",
                    "current_count": 1,
                    "target_count": 4,
                    "status": "InService",
                }
            ],
        }

        mock_sagemaker_client = Mock()
        mock_boto3_client.return_value = mock_sagemaker_client

        # Update mock to require more instances than available
        self.mock_smhp_infra.instance_count = 2

        errors = []
        Validator._validate_infrastructure(self.mock_smhp_infra, errors)

        # Should add error about insufficient capacity
        self.assertEqual(len(errors), 1)
        self.assertIn("Insufficient capacity", errors[0])
        self.assertIn("Required: 2, Maximum available: 1", errors[0])


class TestValidationConfig(unittest.TestCase):
    """Test validation configuration functionality."""

    def test_get_validation_config_defaults(self):
        """Test that default validation config is correct."""
        config = ValidationConfig()
        self.assertTrue(config.iam)
        self.assertTrue(config.infra)
        self.assertTrue(config.recipe)

    @patch("amzn_nova_forge.validation.validator.boto3.client")
    def test_validation_config_recipe_disabled(self, mock_boto3_client):
        """Test that recipe validation is skipped when disabled."""
        mock_infra = Mock(spec=SMTJRuntimeManager)
        mock_infra.instance_type = "ml.g5.12xlarge"
        mock_infra.region = "us-east-1"

        # This overrides_template would normally trigger a recipe validation error
        # (required field missing from recipe)
        overrides_template = {"max_steps": {"required": True, "type": "int"}}

        # Should not raise even though recipe is empty and max_steps is required
        Validator.validate(
            platform=Platform.SMTJ,
            method=TrainingMethod.SFT_LORA,
            infra=mock_infra,
            recipe={},
            overrides_template=overrides_template,
            validation_config=ValidationConfig(iam=False, infra=False, recipe=False),
        )

    @patch("amzn_nova_forge.validation.validator.boto3.client")
    def test_validation_config_iam_disabled(self, mock_boto3_client):
        """Test that IAM validation is skipped when disabled."""
        mock_infra = Mock(spec=SMTJRuntimeManager)
        mock_infra.instance_type = "ml.g5.12xlarge"
        mock_infra.region = "us-east-1"

        try:
            Validator.validate(
                platform=Platform.SMTJ,
                infra=mock_infra,
                method=TrainingMethod.SFT_LORA,
                recipe={},
                overrides_template={},
                validation_config=ValidationConfig(iam=False, infra=False),
            )
        except ValueError as e:
            # Should not fail due to IAM validation
            self.assertNotIn("Failed to validate IAM permissions", str(e))
            self.assertNotIn("Failed to get SageMaker execution role", str(e))

    def test_validation_config_infra_disabled(self):
        """Test that infrastructure validation is skipped when disabled."""
        # Create a mock infra object that would normally cause validation errors
        mock_infra = Mock(spec=SMHPRuntimeManager)
        mock_infra.cluster_name = "non-existent-cluster"
        mock_infra.instance_type = "ml.g5.12xlarge"
        mock_infra.region = "us-east-1"

        try:
            Validator.validate(
                platform=Platform.SMHP,
                method=TrainingMethod.SFT_LORA,
                infra=mock_infra,
                recipe={},
                overrides_template={},
                validation_config=ValidationConfig(iam=False, infra=False),
            )
        except ValueError as e:
            # Should not fail due to infrastructure validation
            self.assertNotIn("not available in cluster", str(e))
            self.assertNotIn("Failed to validate cluster", str(e))

    @patch("amzn_nova_forge.validation.validator.boto3.client")
    def test_validation_config_partial_merge_with_defaults(self, mock_boto3_client):
        """Test that partial validation config is merged with defaults."""
        mock_infra = Mock(spec=SMTJRuntimeManager)
        mock_infra.instance_type = "ml.g5.12xlarge"
        mock_infra.region = "us-east-1"

        try:
            Validator.validate(
                platform=Platform.SMTJ,
                method=TrainingMethod.SFT_LORA,
                infra=mock_infra,
                recipe={},
                overrides_template={},
                validation_config=ValidationConfig(
                    iam=False
                ),  # infra not specified, should default to True
            )
        except ValueError as e:
            # Should not fail due to IAM validation (disabled)
            self.assertNotIn("Failed to validate IAM permissions", str(e))
            self.assertNotIn("Failed to get SageMaker execution role", str(e))


class TestRFTValidation(unittest.TestCase):
    """Test RFT validator integration with Validator."""

    @patch("amzn_nova_forge.validation.validator.boto3.client")
    def test_rft_validator_with_lambda_arn(self, mock_boto3_client):
        """Test RFT validator with required lambda ARN parameter."""
        mock_infra = Mock(spec=SMTJRuntimeManager)
        mock_infra.instance_type = "ml.p5.48xlarge"
        mock_infra.region = "us-east-1"

        try:
            Validator.validate(
                platform=Platform.SMTJ,
                method=TrainingMethod.RFT_LORA,
                infra=mock_infra,
                recipe={},
                overrides_template={},
                rft_lambda_arn="arn:aws:lambda:us-west-2:123456789012:function:test-function",
                validation_config=ValidationConfig(iam=False, infra=False),
            )
        except ValueError:
            self.fail("RFT validate() raised ValueError unexpectedly!")

    @patch("amzn_nova_forge.validation.validator.boto3.client")
    def test_rft_validator_with_invalid_lambda_arn(self, mock_boto3_client):
        """Test RFT validator fails with invalid lambda ARN."""
        mock_infra = Mock(spec=SMTJRuntimeManager)
        mock_infra.instance_type = "ml.p5.48xlarge"
        mock_infra.region = "us-east-1"

        with self.assertRaises(ValueError) as context:
            Validator.validate(
                platform=Platform.SMTJ,
                method=TrainingMethod.RFT_LORA,
                infra=mock_infra,
                recipe={},
                overrides_template={},
                validation_config=ValidationConfig(iam=False, infra=False),
                rft_lambda_arn="not a lambda arn",
            )

        self.assertIn("must be a valid Lambda function ARN", str(context.exception))

    @patch("amzn_nova_forge.validation.validator.boto3.client")
    def test_rft_validator_without_lambda_arn(self, mock_boto3_client):
        """Test RFT validator fails without lambda ARN."""
        mock_infra = Mock(spec=SMTJRuntimeManager)
        mock_infra.instance_type = "ml.p5.48xlarge"
        mock_infra.region = "us-east-1"

        with self.assertRaises(ValueError) as context:
            Validator.validate(
                platform=Platform.SMTJ,
                method=TrainingMethod.RFT_LORA,
                infra=mock_infra,
                recipe={},
                overrides_template={},
                validation_config=ValidationConfig(iam=False, infra=False),
            )

        self.assertIn(
            "Either 'rft_lambda_arn' or 'rft_lambda_source' is required",
            str(context.exception),
        )


class TestEvaluationValidation(unittest.TestCase):
    """Test evaluation-specific validation."""

    @patch("amzn_nova_forge.validation.validator.boto3.client")
    def test_evaluation_with_valid_task(self, mock_boto3_client):
        """Test evaluation with valid task."""
        mock_infra = Mock(spec=SMTJRuntimeManager)
        mock_infra.instance_type = "ml.p5.48xlarge"
        mock_infra.region = "us-east-1"

        try:
            Validator.validate(
                platform=Platform.SMTJ,
                method=TrainingMethod.EVALUATION,
                infra=mock_infra,
                recipe={},
                overrides_template={},
                eval_task=EvaluationTask.MMLU,
                validation_config=ValidationConfig(iam=False, infra=False),
            )
        except ValueError:
            self.fail("Evaluation validate() raised ValueError unexpectedly!")

    @patch("amzn_nova_forge.validation.validator.boto3.client")
    def test_evaluation_with_valid_byod_task(self, mock_boto3_client):
        """Test evaluation with valid task."""
        mock_infra = Mock(spec=SMTJRuntimeManager)
        mock_infra.instance_type = "ml.p5.48xlarge"
        mock_infra.region = "us-east-1"

        try:
            Validator.validate(
                platform=Platform.SMTJ,
                method=TrainingMethod.EVALUATION,
                infra=mock_infra,
                recipe={},
                overrides_template={},
                eval_task=EvaluationTask.GEN_QA,
                data_s3_path="data_s3_path",
                validation_config=ValidationConfig(iam=False, infra=False),
            )
        except ValueError:
            self.fail("Evaluation validate() raised ValueError unexpectedly!")

    @patch("amzn_nova_forge.validation.validator.boto3.client")
    def test_eval_validator_with_invalid_byod_task(self, mock_boto3_client):
        """Test eval validator fails with invalid BYOD eval task"""
        mock_infra = Mock(spec=SMTJRuntimeManager)
        mock_infra.instance_type = "ml.p5.48xlarge"
        mock_infra.region = "us-east-1"

        with self.assertRaises(ValueError) as context:
            Validator.validate(
                platform=Platform.SMTJ,
                method=TrainingMethod.EVALUATION,
                infra=mock_infra,
                recipe={},
                overrides_template={},
                eval_task=EvaluationTask.MMLU,
                data_s3_path="data_s3_path",
                validation_config=ValidationConfig(iam=False, infra=False),
            )

        self.assertIn(
            "BYOD evaluation must use one of the following eval tasks",
            str(context.exception),
        )

    @patch("amzn_nova_forge.validation.validator.boto3.client")
    def test_evaluation_with_valid_subtask(self, mock_boto3_client):
        """Test evaluation with valid subtask."""
        mock_infra = Mock(spec=SMTJRuntimeManager)
        mock_infra.instance_type = "ml.p5.48xlarge"
        mock_infra.region = "us-east-1"

        try:
            Validator.validate(
                platform=Platform.SMTJ,
                method=TrainingMethod.EVALUATION,
                infra=mock_infra,
                recipe={},
                overrides_template={},
                eval_task=EvaluationTask.MMLU,
                subtask="abstract_algebra",
                validation_config=ValidationConfig(iam=False, infra=False),
            )
        except ValueError:
            self.fail("Evaluation validate() raised ValueError unexpectedly!")

    @patch("amzn_nova_forge.validation.validator.boto3.client")
    def test_eval_validator_with_invalid_subtask(self, mock_boto3_client):
        """Test eval validator fails with invalid subtask"""
        mock_infra = Mock(spec=SMTJRuntimeManager)
        mock_infra.instance_type = "ml.p5.48xlarge"
        mock_infra.region = "us-east-1"

        with self.assertRaises(ValueError) as context:
            Validator.validate(
                platform=Platform.SMTJ,
                method=TrainingMethod.EVALUATION,
                infra=mock_infra,
                recipe={},
                overrides_template={},
                eval_task=EvaluationTask.MMLU,
                subtask="invalid",
                validation_config=ValidationConfig(iam=False, infra=False),
            )

        self.assertIn("Invalid subtask", str(context.exception))

    @patch("amzn_nova_forge.validation.validator.boto3.client")
    def test_eval_validator_with_unsupported_subtask(self, mock_boto3_client):
        """Test eval validator fails with unsupported subtask"""
        mock_infra = Mock(spec=SMTJRuntimeManager)
        mock_infra.instance_type = "ml.p5.48xlarge"
        mock_infra.region = "us-east-1"

        with self.assertRaises(ValueError) as context:
            Validator.validate(
                platform=Platform.SMTJ,
                method=TrainingMethod.EVALUATION,
                infra=mock_infra,
                recipe={},
                overrides_template={},
                eval_task=EvaluationTask.GEN_QA,
                subtask="abstract_algebra",
                validation_config=ValidationConfig(iam=False, infra=False),
            )

        self.assertIn("does not support subtasks", str(context.exception))

    @patch("amzn_nova_forge.validation.validator.boto3.client")
    def test_evaluation_with_valid_processor_config(self, mock_boto3_client):
        """Test evaluation with valid processor_config."""
        mock_infra = Mock(spec=SMTJRuntimeManager)
        mock_infra.instance_type = "ml.p5.48xlarge"
        mock_infra.region = "us-east-1"

        try:
            Validator.validate(
                platform=Platform.SMTJ,
                method=TrainingMethod.EVALUATION,
                infra=mock_infra,
                recipe={},
                overrides_template={},
                eval_task=EvaluationTask.GEN_QA,
                processor_config={
                    "lambda_arn": "arn:aws:lambda:us-east-1:123456789012:function:my-function"
                },
                validation_config=ValidationConfig(iam=False, infra=False),
            )
        except ValueError:
            self.fail("Evaluation validate() raised ValueError unexpectedly!")

    @patch("amzn_nova_forge.validation.validator.boto3.client")
    def test_eval_validator_with_processor_config_but_not_needed(self, mock_boto3_client):
        """Test eval validator fails when processor config is provided but not needed"""
        mock_infra = Mock(spec=SMTJRuntimeManager)
        mock_infra.instance_type = "ml.p5.48xlarge"
        mock_infra.region = "us-east-1"

        with self.assertRaises(ValueError) as context:
            Validator.validate(
                platform=Platform.SMTJ,
                method=TrainingMethod.EVALUATION,
                infra=mock_infra,
                recipe={},
                overrides_template={},
                eval_task=EvaluationTask.MMLU,
                processor_config={"lambda_arn": "my lambda"},
                validation_config=ValidationConfig(iam=False, infra=False),
            )

        self.assertIn("processor_config is only supported for gen_qa task", str(context.exception))

    @patch("amzn_nova_forge.validation.validator.boto3.client")
    def test_eval_validator_with_processor_config_missing_lambda_arn(self, mock_boto3_client):
        """Test eval validator fails when processor config is provided but is missing lambda_arn"""
        mock_infra = Mock(spec=SMTJRuntimeManager)
        mock_infra.instance_type = "ml.p5.48xlarge"
        mock_infra.region = "us-east-1"

        with self.assertRaises(ValueError) as context:
            Validator.validate(
                platform=Platform.SMTJ,
                method=TrainingMethod.EVALUATION,
                infra=mock_infra,
                recipe={},
                overrides_template={},
                eval_task=EvaluationTask.GEN_QA,
                processor_config={"lambda_task": "lambda_task"},
                validation_config=ValidationConfig(iam=False, infra=False),
            )

        self.assertIn("processor_config must contain a lambda_arn", str(context.exception))

    @patch("amzn_nova_forge.validation.validator.boto3.client")
    def test_eval_validator_with_rl_env_config_missing_reward_lambda_arn(self, mock_boto3_client):
        """Test eval validator fails when rl_env_config is provided but is missing reward_lambda_arn"""
        mock_infra = Mock(spec=SMTJRuntimeManager)
        mock_infra.instance_type = "ml.p5.48xlarge"
        mock_infra.region = "us-east-1"

        with self.assertRaises(ValueError) as context:
            Validator.validate(
                platform=Platform.SMTJ,
                method=TrainingMethod.EVALUATION,
                infra=mock_infra,
                recipe={},
                overrides_template={},
                eval_task=EvaluationTask.RFT_EVAL,
                rl_env_config={"something": "something"},
                validation_config=ValidationConfig(iam=False, infra=False),
            )

        self.assertIn("rl_env must contain a reward_lambda_arn", str(context.exception))

    @patch("amzn_nova_forge.validation.validator.boto3.client")
    def test_eval_validator_with_rl_env_config_invalid_task(self, mock_boto3_client):
        """Test eval validator fails when rl_env_config is provided but for invalid task"""
        mock_infra = Mock(spec=SMTJRuntimeManager)
        mock_infra.instance_type = "ml.p5.48xlarge"
        mock_infra.region = "us-east-1"

        with self.assertRaises(ValueError) as context:
            Validator.validate(
                platform=Platform.SMTJ,
                method=TrainingMethod.EVALUATION,
                infra=mock_infra,
                recipe={},
                overrides_template={},
                eval_task=EvaluationTask.MMLU,
                rl_env_config={"reward_lambda_arn": "reward_lambda_arn"},
                validation_config=ValidationConfig(iam=False, infra=False),
            )

        self.assertIn(
            "rl_env_config is only supported for rft_eval and rft_multiturn_eval task",
            str(context.exception),
        )

    @patch("amzn_nova_forge.validation.validator.boto3.client")
    def test_eval_validator_with_valid_rl_env_config(self, mock_boto3_client):
        """Test eval validator succeeds when rl_env_config is provided"""
        mock_infra = Mock(spec=SMTJRuntimeManager)
        mock_infra.instance_type = "ml.p5.48xlarge"
        mock_infra.region = "us-east-1"

        try:
            Validator.validate(
                platform=Platform.SMTJ,
                method=TrainingMethod.EVALUATION,
                infra=mock_infra,
                recipe={},
                overrides_template={},
                eval_task=EvaluationTask.RFT_EVAL,
                rl_env_config={
                    "reward_lambda_arn": "arn:aws:lambda:us-east-1:123456789012:function:my-function"
                },
                validation_config=ValidationConfig(iam=False, infra=False),
            )
        except ValueError:
            self.fail("Evaluation validate() raised ValueError unexpectedly!")


class TestRecipeValidation(unittest.TestCase):
    """Test cases for recipe validation logic in Validator._validate_recipe method"""

    def setUp(self):
        self.mock_infra = Mock(spec=SMTJRuntimeManager)
        self.mock_infra.instance_type = "ml.p5.48xlarge"
        self.mock_infra.region = "us-east-1"

    def test_validate_recipe_namespace_key_is_skipped(self):
        """Test that 'namespace' key is skipped during validation"""
        recipe = {}
        overrides_template = {
            "namespace": {"type": "string", "required": True, "default": "kubeflow"}
        }

        errors = []
        Validator._validate_recipe(
            recipe=recipe,
            overrides_template=overrides_template,
            instance_type="ml.p5.48xlarge",
            errors=errors,
            method=TrainingMethod.SFT_LORA,
        )

        self.assertEqual(len(errors), 0)

    def test_validate_recipe_instance_type_valid(self):
        """Test instance_type validation with valid value"""
        recipe = {}

        overrides_template = {"instance_type": {"enum": ["ml.g5.48xlarge", "ml.p5.48xlarge"]}}

        errors = []
        Validator._validate_recipe(
            recipe=recipe,
            overrides_template=overrides_template,
            instance_type="ml.p5.48xlarge",
            errors=errors,
            method=TrainingMethod.SFT_LORA,
        )

        self.assertEqual(len(errors), 0)

    def test_validate_recipe_instance_type_invalid(self):
        """Test instance_type validation with invalid value"""
        recipe = {}

        overrides_template = {"instance_type": {"enum": ["ml.g5.48xlarge", "ml.p5.48xlarge"]}}

        errors = []
        Validator._validate_recipe(
            recipe=recipe,
            overrides_template=overrides_template,
            instance_type="ml.g5.12xlarge",
            errors=errors,
            method=TrainingMethod.SFT_LORA,
        )

        self.assertEqual(len(errors), 1)
        self.assertIn("Instance type 'ml.g5.12xlarge' is not supported", errors[0])
        self.assertIn("ml.g5.48xlarge", errors[0])
        self.assertIn("ml.p5.48xlarge", errors[0])

    def test_validate_recipe_instance_type_without_enum(self):
        """Test instance_type validation when no instance types are specified"""
        recipe = {}

        errors = []
        Validator._validate_recipe(
            recipe=recipe,
            overrides_template={},
            instance_type="ml.g5.12xlarge",
            errors=errors,
            method=TrainingMethod.SFT_LORA,
        )

        self.assertEqual(len(errors), 0)

    def test_validate_recipe_required_validation_missing(self):
        """Test type validation fails with missing required field"""
        recipe = {}
        overrides_template = {"max_steps": {"required": True}}

        errors = []
        Validator._validate_recipe(
            recipe=recipe,
            overrides_template=overrides_template,
            instance_type="ml.p5.48xlarge",
            errors=errors,
            method=TrainingMethod.SFT_LORA,
        )

        self.assertEqual(len(errors), 1)
        self.assertIn("'max_steps' is required, but was not found in your recipe", errors[0])

    def test_validate_recipe_type_validation_string(self):
        """Test type validation for string type"""
        recipe = {"run": {"name": "my-training-run"}}
        overrides_template = {"name": {"type": "string"}}

        errors = []
        Validator._validate_recipe(
            recipe=recipe,
            overrides_template=overrides_template,
            instance_type="ml.p5.48xlarge",
            errors=errors,
            method=TrainingMethod.SFT_LORA,
        )

        self.assertEqual(len(errors), 0)

    def test_validate_recipe_type_validation_integer(self):
        """Test type validation for integer type"""
        recipe = {"training_config": {"max_steps": 100}}
        overrides_template = {"max_steps": {"type": "integer"}}

        errors = []
        Validator._validate_recipe(
            recipe=recipe,
            overrides_template=overrides_template,
            instance_type="ml.p5.48xlarge",
            errors=errors,
            method=TrainingMethod.SFT_LORA,
        )

        self.assertEqual(len(errors), 0)

    def test_validate_recipe_type_validation_boolean(self):
        """Test type validation for boolean type"""
        recipe = {"training_config": {"reasoning_enabled": True}}
        overrides_template = {"reasoning_enabled": {"type": "boolean"}}

        errors = []
        Validator._validate_recipe(
            recipe=recipe,
            overrides_template=overrides_template,
            instance_type="ml.p5.48xlarge",
            errors=errors,
            method=TrainingMethod.SFT_LORA,
        )

        self.assertEqual(len(errors), 0)

    def test_validate_recipe_type_validation_wrong_type(self):
        """Test type validation fails with wrong type"""
        recipe = {"training_config": {"max_steps": "100"}}
        overrides_template = {"max_steps": {"type": "integer"}}

        errors = []
        Validator._validate_recipe(
            recipe=recipe,
            overrides_template=overrides_template,
            instance_type="ml.p5.48xlarge",
            errors=errors,
            method=TrainingMethod.SFT_LORA,
        )

        self.assertEqual(len(errors), 1)
        self.assertIn("'max_steps' expects integer", errors[0])
        self.assertIn("You provided str", errors[0])

    def test_validate_recipe_unknown_type_in_metadata(self):
        """Test validation handles unknown type in metadata gracefully"""
        recipe = {"run": {"custom_field": "value"}}
        overrides_template = {"custom_field": {"type": "unknown_type"}}

        errors = []
        Validator._validate_recipe(
            recipe=recipe,
            overrides_template=overrides_template,
            instance_type="ml.p5.48xlarge",
            errors=errors,
            method=TrainingMethod.SFT_LORA,
        )

        self.assertEqual(len(errors), 0)

    def test_validate_recipe_enum_validation_valid(self):
        """Test enum validation with valid value"""
        recipe = {"run": {"replicas": 4}}
        overrides_template = {"replicas": {"enum": [4, 8, 16, 32]}}

        errors = []
        Validator._validate_recipe(
            recipe=recipe,
            overrides_template=overrides_template,
            instance_type="ml.p5.48xlarge",
            errors=errors,
            method=TrainingMethod.SFT_LORA,
        )

        self.assertEqual(len(errors), 0)

    def test_validate_recipe_enum_validation_invalid(self):
        """Test enum validation with invalid value"""
        recipe = {"run": {"replicas": 2}}
        overrides_template = {"replicas": {"type": "integer", "enum": [4, 8, 16, 32]}}

        errors = []
        Validator._validate_recipe(
            recipe=recipe,
            overrides_template=overrides_template,
            instance_type="ml.p5.48xlarge",
            errors=errors,
            method=TrainingMethod.SFT_LORA,
        )

        self.assertEqual(len(errors), 1)
        self.assertIn("'replicas' must be one of [4, 8, 16, 32]", errors[0])
        self.assertIn("You provided 2", errors[0])

    def test_validate_recipe_min_validation_valid(self):
        """Test minimum value validation with valid value"""
        recipe = {"training_config": {"max_steps": 100}}
        overrides_template = {"max_steps": {"type": "integer", "min": 4}}

        errors = []
        Validator._validate_recipe(
            recipe=recipe,
            overrides_template=overrides_template,
            instance_type="ml.p5.48xlarge",
            errors=errors,
            method=TrainingMethod.SFT_LORA,
        )

        self.assertEqual(len(errors), 0)

    def test_validate_recipe_min_validation_invalid(self):
        """Test minimum value validation with invalid value"""
        recipe = {"training_config": {"max_steps": 2}}
        overrides_template = {"max_steps": {"type": "integer", "min": 4}}

        errors = []
        Validator._validate_recipe(
            recipe=recipe,
            overrides_template=overrides_template,
            instance_type="ml.p5.48xlarge",
            errors=errors,
            method=TrainingMethod.SFT_LORA,
        )

        self.assertEqual(len(errors), 1)
        self.assertIn("'max_steps' must be at least 4", errors[0])
        self.assertIn("You provided 2", errors[0])

    def test_validate_recipe_min_validation_exact_boundary(self):
        """Test minimum value validation at exact boundary"""
        recipe = {"training_config": {"max_steps": 4}}
        overrides_template = {"max_steps": {"type": "integer", "min": 4}}

        errors = []
        Validator._validate_recipe(
            recipe=recipe,
            overrides_template=overrides_template,
            instance_type="ml.p5.48xlarge",
            errors=errors,
            method=TrainingMethod.SFT_LORA,
        )

        self.assertEqual(len(errors), 0)

    def test_validate_recipe_max_validation_valid(self):
        """Test maximum value validation with valid value"""
        recipe = {"training_config": {"max_steps": 100}}
        overrides_template = {"max_steps": {"type": "integer", "max": 500}}

        errors = []
        Validator._validate_recipe(
            recipe=recipe,
            overrides_template=overrides_template,
            instance_type="ml.p5.48xlarge",
            errors=errors,
            method=TrainingMethod.SFT_LORA,
        )

        self.assertEqual(len(errors), 0)

    def test_validate_recipe_max_validation_invalid(self):
        """Test maximum value validation with invalid value"""
        recipe = {"training_config": {"max_steps": 2}}
        overrides_template = {"max_steps": {"type": "integer", "max": 1}}

        errors = []
        Validator._validate_recipe(
            recipe=recipe,
            overrides_template=overrides_template,
            instance_type="ml.p5.48xlarge",
            errors=errors,
            method=TrainingMethod.SFT_LORA,
        )

        self.assertEqual(len(errors), 1)
        self.assertIn("'max_steps' must be no greater than 1", errors[0])
        self.assertIn("You provided 2", errors[0])

    def test_validate_recipe_max_validation_exact_boundary(self):
        """Test maximum value validation at exact boundary"""
        recipe = {"training_config": {"max_steps": 4}}
        overrides_template = {"max_steps": {"type": "integer", "max": 4}}

        errors = []
        Validator._validate_recipe(
            recipe=recipe,
            overrides_template=overrides_template,
            instance_type="ml.p5.48xlarge",
            errors=errors,
            method=TrainingMethod.SFT_LORA,
        )

        self.assertEqual(len(errors), 0)

    def test_validate_recipe_multiple_constraints_all_valid(self):
        """Test recipe with multiple constraints that are all valid"""
        recipe = {
            "run": {"replicas": 8, "name": "my-run"},
            "training_config": {"max_steps": 100, "reasoning_enabled": True},
        }
        overrides_template = {
            "replicas": {"type": "integer", "enum": [4, 8, 16, 32]},
            "name": {"type": "string"},
            "max_steps": {"type": "integer", "min": 4, "max": 100000},
            "reasoning_enabled": {"type": "boolean"},
        }

        errors = []
        Validator._validate_recipe(
            recipe=recipe,
            overrides_template=overrides_template,
            instance_type="ml.p5.48xlarge",
            errors=errors,
            method=TrainingMethod.SFT_LORA,
        )

        self.assertEqual(len(errors), 0)

    def test_validate_recipe_multiple_constraints_multiple_violations(self):
        """Test recipe with multiple constraint violations"""
        recipe = {
            "run": {"replicas": 3, "name": 123},
            "training_config": {"max_steps": 2},
        }
        overrides_template = {
            "replicas": {"type": "integer", "enum": [4, 8, 16, 32]},
            "name": {"type": "string"},
            "max_steps": {"type": "integer", "min": 4},
            "lr": {"required": True},
        }

        errors = []
        Validator._validate_recipe(
            recipe=recipe,
            overrides_template=overrides_template,
            instance_type="ml.p5.48xlarge",
            errors=errors,
            method=TrainingMethod.SFT_LORA,
        )

        # Should have 4 errors: replicas enum, name type, max_steps min, missing lr
        self.assertEqual(len(errors), 4)
        error_text = " ".join(errors)
        self.assertIn("replicas", error_text)
        self.assertIn("name", error_text)
        self.assertIn("max_steps", error_text)
        self.assertIn("lr", error_text)

    def test_validate_recipe_nested_recipe_structure(self):
        """Test validation works with nested recipe structure"""
        recipe = {
            "run": {"replicas": 4},
            "training_config": {
                "max_steps": 100,
                "optim_config": {"lr": 0.00001, "weight_decay": 0.0},
            },
        }
        overrides_template = {
            "replicas": {"type": "integer", "enum": [4, 8, 16, 32]},
            "max_steps": {"type": "integer", "min": 4, "max": 100000},
            "lr": {"type": "float", "min": 0, "max": 1},
        }

        errors = []
        Validator._validate_recipe(
            recipe=recipe,
            overrides_template=overrides_template,
            instance_type="ml.p5.48xlarge",
            errors=errors,
            method=TrainingMethod.SFT_LORA,
        )

        self.assertEqual(len(errors), 0)

    def test_validate_recipe_type_validation_prevents_further_checks(self):
        """Test that type validation failure prevents enum/min/max checks"""
        recipe = {"training_config": {"max_steps": "not_a_number"}}
        overrides_template = {
            "max_steps": {"type": "integer", "min": 4, "max": 100000, "enum": [4, 8]}
        }

        errors = []
        Validator._validate_recipe(
            recipe=recipe,
            overrides_template=overrides_template,
            instance_type="ml.p5.48xlarge",
            errors=errors,
            method=TrainingMethod.SFT_LORA,
        )

        # Should only have type error, not enum/min/max errors (due to continue statement)
        self.assertEqual(len(errors), 1)
        self.assertIn("expects integer", errors[0])
        self.assertNotIn("must be one of", errors[0])
        self.assertNotIn("must be at least", errors[0])

    def test_validate_recipe_with_real_world_template(self):
        """Test validation with realistic overrides template"""
        recipe = {
            "run": {
                "name": "my-full-rank-sft-run",
                "data_s3_path": "s3://my-bucket/train.jsonl",
                "output_s3_path": "s3://my-bucket/output/",
                "replicas": 4,
            },
            "training_config": {
                "max_steps": 100,
                "global_batch_size": 64,
                "reasoning_enabled": True,
                "max_length": 8192,
            },
        }

        overrides_template = {
            "replicas": {"type": "integer", "enum": [4, 8, 16, 32], "default": 4},
            "name": {"type": "string", "required": True},
            "data_s3_path": {"type": "string", "required": True},
            "output_s3_path": {"type": "string", "required": True},
            "global_batch_size": {"type": "integer", "enum": [32, 64, 128, 256]},
            "reasoning_enabled": {"type": "boolean", "default": True},
            "max_steps": {"type": "integer", "min": 4, "max": 100000, "default": 100},
            "max_context_length": {"type": "integer", "min": 1, "max": 131072},
            "instance_type": {
                "type": "string",
                "enum": ["ml.p5.48xlarge", "ml.p5en.48xlarge"],
            },
            "namespace": {"type": "string", "default": "kubeflow"},
        }

        errors = []
        Validator._validate_recipe(
            recipe=recipe,
            overrides_template=overrides_template,
            instance_type="ml.p5.48xlarge",
            errors=errors,
            method=TrainingMethod.SFT_LORA,
        )

        self.assertEqual(len(errors), 0)

    def test_validate_recipe_boolean_type_with_integer(self):
        """Test that boolean type validation catches integer values"""
        recipe = {"training_config": {"reasoning_enabled": 1}}
        overrides_template = {"reasoning_enabled": {"type": "boolean"}}

        errors = []
        Validator._validate_recipe(
            recipe=recipe,
            overrides_template=overrides_template,
            instance_type="ml.p5.48xlarge",
            errors=errors,
            method=TrainingMethod.SFT_LORA,
        )

        # Note: In Python, bool is a subclass of int, so isinstance(1, bool) is False
        # but isinstance(True, int) is True. This test verifies the validation catches this.
        self.assertEqual(len(errors), 1)
        self.assertIn("expects boolean", errors[0])

    def test_validate_recipe_empty_overrides_template(self):
        """Test validation with empty overrides template."""
        recipe = {
            "run": {"name": "my-run"},
            "training_config": {"max_steps": 100},
        }
        overrides_template = {}

        errors = []
        Validator._validate_recipe(
            recipe=recipe,
            overrides_template=overrides_template,
            instance_type="ml.p5.48xlarge",
            errors=errors,
            method=TrainingMethod.SFT_LORA,
        )

        self.assertEqual(len(errors), 0)

    @patch("amzn_nova_forge.validation.validator.boto3.client")
    def test_validate_recipe_integration_with_full_validate_method(self, mock_boto3_client):
        """Test recipe validation as part of full validate() method"""
        recipe = {"training_config": {"max_steps": 2}}  # Below minimum
        overrides_template = {"max_steps": {"type": "integer", "min": 4}}

        with self.assertRaises(ValueError) as context:
            Validator.validate(
                platform=Platform.SMTJ,
                method=TrainingMethod.SFT_LORA,
                infra=self.mock_infra,
                recipe=recipe,
                overrides_template=overrides_template,
                validation_config=ValidationConfig(iam=False, infra=False),
            )

        self.assertIn("'max_steps' must be at least 4", str(context.exception))

    def test_validate_recipe_save_steps_less_than_max_steps_valid(self):
        """Test validation passes when save_steps < max_steps."""
        recipe = {"training_config": {"max_steps": 1000, "save_steps": 100}}
        overrides_template = {
            "max_steps": {"type": "integer"},
            "save_steps": {"type": "integer"},
        }

        errors = []
        Validator._validate_recipe(
            recipe=recipe,
            overrides_template=overrides_template,
            instance_type="ml.p5.48xlarge",
            errors=errors,
            method=TrainingMethod.SFT_LORA,
        )

        self.assertEqual(len(errors), 0)

    def test_validate_recipe_save_steps_equal_to_max_steps_valid(self):
        """Test validation passes when save_steps == max_steps."""
        recipe = {"training_config": {"max_steps": 1000, "save_steps": 1000}}
        overrides_template = {
            "max_steps": {"type": "integer"},
            "save_steps": {"type": "integer"},
        }

        errors = []
        Validator._validate_recipe(
            recipe=recipe,
            overrides_template=overrides_template,
            instance_type="ml.p5.48xlarge",
            errors=errors,
            method=TrainingMethod.SFT_LORA,
        )

        self.assertEqual(len(errors), 0)

    def test_validate_recipe_save_steps_greater_than_max_steps_invalid(self):
        """Test validation fails when save_steps > max_steps."""
        recipe = {"training_config": {"max_steps": 100, "save_steps": 200}}
        overrides_template = {
            "max_steps": {"type": "integer"},
            "save_steps": {"type": "integer"},
        }

        errors = []
        Validator._validate_recipe(
            recipe=recipe,
            overrides_template=overrides_template,
            instance_type="ml.p5.48xlarge",
            errors=errors,
            method=TrainingMethod.SFT_LORA,
        )

        self.assertEqual(len(errors), 1)
        self.assertIn(
            "'save_steps' (200) must be less than or equal to 'max_steps' (100)",
            errors[0],
        )

    def test_validate_recipe_save_steps_validation_skipped_when_missing(self):
        """Test save_steps validation is skipped when either field is missing."""
        # Only max_steps present
        recipe1 = {"training_config": {"max_steps": 100}}
        overrides_template = {"max_steps": {"type": "integer"}}

        errors1 = []
        Validator._validate_recipe(
            recipe=recipe1,
            overrides_template=overrides_template,
            instance_type="ml.p5.48xlarge",
            errors=errors1,
            method=TrainingMethod.SFT_LORA,
        )

        self.assertEqual(len(errors1), 0)

        # Only save_steps present
        recipe2 = {"training_config": {"save_steps": 100}}
        overrides_template2 = {"save_steps": {"type": "integer"}}

        errors2 = []
        Validator._validate_recipe(
            recipe=recipe2,
            overrides_template=overrides_template2,
            instance_type="ml.p5.48xlarge",
            errors=errors2,
            method=TrainingMethod.SFT_LORA,
        )

        self.assertEqual(len(errors2), 0)


class TestCallingRolePermissionsValidation(unittest.TestCase):
    """Test cases for _validate_calling_role_permissions method"""

    @patch("boto3.client")
    def test_validate_calling_role_permissions_success(self, mock_boto3_client):
        """Test successful permission validation"""
        # Mock clients
        mock_iam_client = MagicMock()
        mock_sts_client = MagicMock()

        mock_boto3_client.side_effect = lambda service, **kwargs: {
            "iam": mock_iam_client,
            "sts": mock_sts_client,
        }[service]

        # Mock STS response
        mock_sts_client.get_caller_identity.return_value = {
            "Arn": "arn:aws:sts::123456789012:assumed-role/TestRole/session",
            "Account": "123456789012",
        }

        # Mock IAM simulation response - permission allowed
        mock_iam_client.simulate_principal_policy.return_value = {
            "EvaluationResults": [{"EvalDecision": "allowed"}]
        }

        errors = []
        required_permissions = [
            ("sagemaker:CreateTrainingJob", "*"),
            ("iam:PassRole", "*"),
        ]

        Validator._validate_calling_role_permissions(
            errors, required_permissions, None, "us-east-1"
        )

        # Should have no errors
        self.assertEqual(len(errors), 0)

        # Verify correct calls were made
        mock_sts_client.get_caller_identity.assert_called_once()
        self.assertEqual(
            mock_iam_client.simulate_principal_policy.call_count, 3
        )  # 1 test + 2 permissions

    @patch("boto3.client")
    def test_validate_calling_role_permissions_denied(self, mock_boto3_client):
        """Test permission validation with denied permissions"""
        # Mock clients
        mock_iam_client = MagicMock()
        mock_sts_client = MagicMock()

        mock_boto3_client.side_effect = lambda service, **kwargs: {
            "iam": mock_iam_client,
            "sts": mock_sts_client,
        }[service]

        # Mock STS response
        mock_sts_client.get_caller_identity.return_value = {
            "Arn": "arn:aws:sts::123456789012:assumed-role/TestRole/session",
            "Account": "123456789012",
        }

        # Mock IAM simulation response - permission denied
        mock_iam_client.simulate_principal_policy.return_value = {
            "EvaluationResults": [{"EvalDecision": "implicitDeny"}]
        }

        errors = []
        required_permissions = [("route53:CreateHostedZone", "*")]

        Validator._validate_calling_role_permissions(
            errors, required_permissions, None, "us-east-1"
        )

        # Should have one error
        self.assertEqual(len(errors), 1)
        self.assertIn(
            "Missing required calling role permission: route53:CreateHostedZone",
            errors[0],
        )

    @patch("boto3.client")
    def test_validate_calling_role_permissions_direct_role_arn(self, mock_boto3_client):
        """Test with direct role ARN (not assumed role)"""
        # Mock clients
        mock_iam_client = MagicMock()
        mock_sts_client = MagicMock()

        mock_boto3_client.side_effect = lambda service, **kwargs: {
            "iam": mock_iam_client,
            "sts": mock_sts_client,
        }[service]

        # Mock STS response with direct role ARN
        mock_sts_client.get_caller_identity.return_value = {
            "Arn": "arn:aws:iam::123456789012:role/DirectRole",
            "Account": "123456789012",
        }

        # Mock IAM simulation response
        mock_iam_client.simulate_principal_policy.return_value = {
            "EvaluationResults": [{"EvalDecision": "allowed"}]
        }

        errors = []
        required_permissions = [("sagemaker:ListClusters", "*")]

        Validator._validate_calling_role_permissions(
            errors, required_permissions, None, "us-east-1"
        )

        # Should use the direct ARN for simulation
        mock_iam_client.simulate_principal_policy.assert_called_with(
            PolicySourceArn="arn:aws:iam::123456789012:role/DirectRole",
            ActionNames=["sagemaker:ListClusters"],
            ResourceArns=["*"],
        )

    @patch("boto3.client")
    def test_validate_calling_role_permissions_simulation_error(self, mock_boto3_client):
        """Test handling of IAM simulation errors"""
        # Mock clients
        mock_iam_client = MagicMock()
        mock_sts_client = MagicMock()

        mock_boto3_client.side_effect = lambda service, **kwargs: {
            "iam": mock_iam_client,
            "sts": mock_sts_client,
        }[service]

        # Mock STS response
        mock_sts_client.get_caller_identity.return_value = {
            "Arn": "arn:aws:sts::123456789012:assumed-role/TestRole/session",
            "Account": "123456789012",
        }

        # Mock IAM simulation error
        mock_iam_client.simulate_principal_policy.side_effect = Exception("Simulation failed")

        errors = []
        required_permissions = [("sagemaker:CreateTrainingJob", "*")]

        Validator._validate_calling_role_permissions(
            errors, required_permissions, None, "us-east-1"
        )

        # Should have error about simulation failure
        self.assertEqual(len(errors), 1)
        self.assertIn(
            "Cannot run iam:SimulatePrincipalPolicy to validate calling role permissions: Simulation failed",
            errors[0],
        )

    @patch("boto3.client")
    def test_validate_calling_role_permissions_sts_error(self, mock_boto3_client):
        """Test handling of STS errors"""
        # Mock clients
        mock_iam_client = MagicMock()
        mock_sts_client = MagicMock()

        mock_boto3_client.side_effect = lambda service, **kwargs: {
            "iam": mock_iam_client,
            "sts": mock_sts_client,
        }[service]

        # Mock STS error
        mock_sts_client.get_caller_identity.side_effect = Exception("STS failed")

        errors = []
        required_permissions = [("sagemaker:CreateTrainingJob", "*")]

        Validator._validate_calling_role_permissions(
            errors, required_permissions, None, "us-east-1"
        )

        # Should have error about calling role validation failure
        self.assertEqual(len(errors), 1)
        self.assertIn("Failed to validate calling role permissions: STS failed", errors[0])


class TestPermissionValidationMethods(unittest.TestCase):
    """Test cases for permission validation helper methods and formats"""

    def test_check_policy_json_permissions_grants_required_actions(self):
        """Test successful policy JSON permission checking"""
        policies = [
            {
                "Statement": [
                    {
                        "Effect": "Allow",
                        "Action": ["s3:GetObject", "s3:PutObject"],
                        "Resource": "*",
                    }
                ]
            }
        ]

        required_permissions = ["s3:GetObject", "s3:PutObject"]
        missing = Validator._check_policy_json_permissions(policies, required_permissions)

        self.assertEqual(len(missing), 0)

    def test_check_policy_json_permissions_supports_wildcards(self):
        """Test policy JSON permission checking with wildcards"""
        policies = [{"Statement": [{"Effect": "Allow", "Action": "s3:*", "Resource": "*"}]}]

        required_permissions = ["s3:GetObject", "s3:PutObject"]
        missing = Validator._check_policy_json_permissions(policies, required_permissions)

        self.assertEqual(len(missing), 0)

    def test_check_policy_json_permissions_supports_action_prefix_wildcards(self):
        """Test policy JSON permission checking with action prefix wildcards like iam:Get*"""
        policies = [{"Statement": [{"Effect": "Allow", "Action": "iam:Get*", "Resource": "*"}]}]

        required_permissions = ["iam:GetRole", "iam:GetPolicy", "iam:GetUser"]
        missing = Validator._check_policy_json_permissions(policies, required_permissions)

        # Should now support prefix wildcards like iam:Get*
        self.assertEqual(len(missing), 0)

    def test_check_policy_json_permissions_action_prefix_wildcards_no_false_positives(
        self,
    ):
        """Test that action prefix wildcards don't match unrelated actions"""
        policies = [{"Statement": [{"Effect": "Allow", "Action": "iam:Get*", "Resource": "*"}]}]

        required_permissions = ["iam:CreateRole", "s3:GetObject", "iam:PutRole"]
        missing = Validator._check_policy_json_permissions(policies, required_permissions)

        # iam:Get* should not match iam:CreateRole, s3:GetObject, or iam:PutRole
        self.assertEqual(len(missing), 3)
        self.assertIn("iam:CreateRole", missing)
        self.assertIn("s3:GetObject", missing)
        self.assertIn("iam:PutRole", missing)

    def test_check_policy_json_permissions_supports_infix_wildcards(self):
        """Test policy JSON permission checking with infix wildcards like s3:*Object"""
        policies = [{"Statement": [{"Effect": "Allow", "Action": "s3:*Object", "Resource": "*"}]}]

        required_permissions = ["s3:GetObject", "s3:PutObject", "s3:DeleteObject"]
        missing = Validator._check_policy_json_permissions(policies, required_permissions)

        # Should support infix wildcards like s3:*Object
        self.assertEqual(len(missing), 0)

    def test_check_policy_json_permissions_infix_wildcards_no_false_positives(self):
        """Test that infix wildcards don't match unrelated actions"""
        policies = [{"Statement": [{"Effect": "Allow", "Action": "s3:*Object", "Resource": "*"}]}]

        required_permissions = ["s3:ListBucket", "s3:GetBucketLocation", "iam:GetRole"]
        missing = Validator._check_policy_json_permissions(policies, required_permissions)

        # s3:*Object should not match s3:ListBucket, s3:GetBucketLocation, or iam:GetRole
        self.assertEqual(len(missing), 3)
        self.assertIn("s3:ListBucket", missing)
        self.assertIn("s3:GetBucketLocation", missing)
        self.assertIn("iam:GetRole", missing)

    def test_check_policy_json_permissions_identifies_missing_actions(self):
        """Test policy JSON permission checking with missing permissions"""
        policies = [
            {"Statement": [{"Effect": "Allow", "Action": ["s3:GetObject"], "Resource": "*"}]}
        ]

        required_permissions = ["s3:GetObject", "s3:PutObject", "iam:PassRole"]
        missing = Validator._check_policy_json_permissions(policies, required_permissions)

        self.assertEqual(len(missing), 2)
        self.assertIn("s3:PutObject", missing)
        self.assertIn("iam:PassRole", missing)

    @patch("boto3.client")
    def test_validate_permissions_with_specific_resource_arn(self, mock_boto3_client):
        """Test permission validation using SimulatePrincipalPolicy with specific resource ARN"""
        mock_iam_client = MagicMock()
        mock_sts_client = MagicMock()

        mock_boto3_client.side_effect = lambda service, **kwargs: {
            "iam": mock_iam_client,
            "sts": mock_sts_client,
        }[service]

        mock_sts_client.get_caller_identity.return_value = {
            "Arn": "arn:aws:sts::123456789012:assumed-role/TestRole/session",
            "Account": "123456789012",
        }

        mock_iam_client.simulate_principal_policy.return_value = {
            "EvaluationResults": [{"EvalDecision": "allowed"}]
        }

        errors = []
        required_permissions = [
            ("sagemaker:CreateTrainingJob", "arn:aws:sagemaker:*:*:training-job/*")
        ]

        Validator._validate_calling_role_permissions(
            errors, required_permissions, None, "us-east-1"
        )

        self.assertEqual(len(errors), 0)
        mock_iam_client.simulate_principal_policy.assert_called_with(
            PolicySourceArn="arn:aws:iam::123456789012:role/TestRole",
            ActionNames=["sagemaker:CreateTrainingJob"],
            ResourceArns=["arn:aws:sagemaker:*:*:training-job/*"],
        )

    @patch("boto3.client")
    def test_validate_permissions_with_dynamic_resource_generation(self, mock_boto3_client):
        """Test permission validation using lambda functions to generate resource ARNs"""
        mock_iam_client = MagicMock()
        mock_sts_client = MagicMock()

        mock_boto3_client.side_effect = lambda service, **kwargs: {
            "iam": mock_iam_client,
            "sts": mock_sts_client,
        }[service]

        mock_sts_client.get_caller_identity.return_value = {
            "Arn": "arn:aws:sts::123456789012:assumed-role/TestRole/session",
            "Account": "123456789012",
        }

        mock_iam_client.simulate_principal_policy.return_value = {
            "EvaluationResults": [{"EvalDecision": "allowed"}]
        }

        # Create mock infra
        mock_infra = MagicMock()
        mock_infra.region = "us-west-2"
        mock_infra.cluster_name = "test-cluster"

        errors = []
        required_permissions = [
            (
                "sagemaker:DescribeCluster",
                lambda infra: f"arn:aws:sagemaker:{infra.region}:*:cluster/{infra.cluster_name}",
            )
        ]

        Validator._validate_calling_role_permissions(
            errors, required_permissions, mock_infra, "us-east-1"
        )

        self.assertEqual(len(errors), 0)
        mock_iam_client.simulate_principal_policy.assert_called_with(
            PolicySourceArn="arn:aws:iam::123456789012:role/TestRole",
            ActionNames=["sagemaker:DescribeCluster"],
            ResourceArns=["arn:aws:sagemaker:us-west-2:*:cluster/test-cluster"],
        )

    @patch("boto3.client")
    def test_validate_permissions_using_policy_document_parsing(self, mock_boto3_client):
        """Test permission validation using JSON policy document parsing for simple strings"""
        mock_iam_client = MagicMock()
        mock_sts_client = MagicMock()

        mock_boto3_client.side_effect = lambda service, **kwargs: {
            "iam": mock_iam_client,
            "sts": mock_sts_client,
        }[service]

        mock_sts_client.get_caller_identity.return_value = {
            "Arn": "arn:aws:sts::123456789012:assumed-role/TestRole/session",
            "Account": "123456789012",
        }

        # Mock policy retrieval for JSON parsing
        mock_iam_client.list_role_policies.return_value = {"PolicyNames": ["InlinePolicy"]}
        mock_iam_client.get_role_policy.return_value = {
            "PolicyDocument": {
                "Statement": [
                    {
                        "Effect": "Allow",
                        "Action": "iam:PassRole",
                        "Resource": "arn:aws:iam::123456789012:role/specific-role",
                    }
                ]
            }
        }
        mock_iam_client.list_attached_role_policies.return_value = {"AttachedPolicies": []}

        errors = []
        required_permissions = ["iam:PassRole"]  # Simple string - should use JSON parsing

        Validator._validate_calling_role_permissions(
            errors, required_permissions, None, "us-east-1"
        )

        self.assertEqual(len(errors), 0)
        # Should call policy retrieval methods for JSON parsing
        mock_iam_client.list_role_policies.assert_called_with(RoleName="TestRole")
        mock_iam_client.get_role_policy.assert_called_with(
            RoleName="TestRole", PolicyName="InlinePolicy"
        )

    @patch("amzn_nova_forge.validation.validator.boto3.client")
    def test_validate_permissions_with_insufficient_resource_permissions(self, mock_boto3_client):
        """Test that validation fails when policy has specific resources but not the required ones"""
        mock_iam_client = Mock()
        mock_sts_client = Mock()

        mock_boto3_client.side_effect = lambda service, **kwargs: {
            "iam": mock_iam_client,
            "sts": mock_sts_client,
        }[service]

        mock_sts_client.get_caller_identity.return_value = {
            "Arn": "arn:aws:sts::123456789012:assumed-role/TestRole/session",
            "Account": "123456789012",
        }

        # Mock policy with specific resources that don't match what's needed
        mock_iam_client.list_role_policies.return_value = {"PolicyNames": ["S3Policy"]}
        mock_iam_client.get_role_policy.return_value = {
            "PolicyDocument": {
                "Statement": [
                    {
                        "Effect": "Allow",
                        "Action": ["s3:GetObject", "s3:ListBucket"],
                        "Resource": [
                            "arn:aws:s3:::wrong-bucket",
                            "arn:aws:s3:::wrong-bucket/*",
                        ],
                    }
                ]
            }
        }
        mock_iam_client.list_attached_role_policies.return_value = {"AttachedPolicies": []}

        # Mock IAM simulation that will be called for resource-specific permissions
        mock_iam_client.simulate_principal_policy.return_value = {
            "EvaluationResults": [{"EvalDecision": "denied"}]
        }

        errors = []
        # Test both JSON parsing (string) and IAM simulation (tuple) permissions
        required_permissions = [
            "iam:PassRole",  # Will use JSON parsing and should fail
            (
                "s3:GetObject",
                "arn:aws:s3:::required-bucket/*",
            ),  # Will use IAM simulation and should fail
        ]

        Validator._validate_calling_role_permissions(
            errors, required_permissions, None, "us-east-1"
        )

        # Should have 2 errors - one from JSON parsing, one from IAM simulation
        self.assertEqual(len(errors), 2)
        self.assertIn("Missing required calling role permission: iam:PassRole", errors[0])
        self.assertIn(
            "Missing required calling role permission: s3:GetObject on arn:aws:s3:::required-bucket/*",
            errors[1],
        )

    @patch("amzn_nova_forge.validation.validator.boto3.client")
    def test_validate_permissions_with_multiple_s3_resources(self, mock_boto3_client):
        """Test that validation generates separate permissions for multiple S3 resources"""
        mock_iam_client = Mock()
        mock_sts_client = Mock()
        mock_sagemaker_client = Mock()

        mock_boto3_client.side_effect = lambda service, **kwargs: {
            "iam": mock_iam_client,
            "sts": mock_sts_client,
            "sagemaker": mock_sagemaker_client,
        }[service]

        mock_sts_client.get_caller_identity.return_value = {
            "Arn": "arn:aws:sts::123456789012:assumed-role/TestRole/session",
            "Account": "123456789012",
        }

        # Mock successful IAM simulation for all resources
        mock_iam_client.simulate_principal_policy.return_value = {
            "EvaluationResults": [
                {
                    "EvalActionName": "s3:GetObject",
                    "EvalResourceName": "arn:aws:s3:::bucket1/data/*",
                    "EvalDecision": "allowed",
                }
            ]
        }

        errors = []

        # Test SMTJ with multiple buckets
        required_permissions = SMTJRuntimeManager.required_calling_role_permissions(
            "s3://bucket1/data/train.jsonl", "s3://bucket2/output/models/"
        )

        # Should have separate permissions for each bucket and object path
        s3_permissions = [
            p for p in required_permissions if isinstance(p, tuple) and p[0].startswith("s3:")
        ]

        # Should have 7 S3 permissions: 2 buckets × 2 bucket perms + 1 data path read + 2 output path read/write
        self.assertEqual(len(s3_permissions), 7)

        # Check that we have permissions for both buckets
        bucket_permissions = [
            p for p in s3_permissions if p[0] in ["s3:CreateBucket", "s3:ListBucket"]
        ]
        bucket_arns = [p[1] for p in bucket_permissions]
        self.assertIn("arn:aws:s3:::bucket1", bucket_arns)
        self.assertIn("arn:aws:s3:::bucket2", bucket_arns)

        # Check data_s3_path permissions (read-only)
        input_permissions = [p for p in s3_permissions if "bucket1/data" in p[1]]
        self.assertEqual(len(input_permissions), 1)  # Only GetObject
        self.assertEqual(input_permissions[0][0], "s3:GetObject")
        self.assertEqual(input_permissions[0][1], "arn:aws:s3:::bucket1/data/train.jsonl*")

        # Check output_s3_path permissions (read-write)
        output_permissions = [p for p in s3_permissions if "bucket2/output" in p[1]]
        self.assertEqual(len(output_permissions), 2)  # GetObject and PutObject
        output_actions = [p[0] for p in output_permissions]
        self.assertIn("s3:GetObject", output_actions)
        self.assertIn("s3:PutObject", output_actions)

    @patch("amzn_nova_forge.validation.validator.boto3.client")
    def test_validate_permissions_fails_with_missing_specific_bucket_access(
        self, mock_boto3_client
    ):
        """Test that validation fails when role lacks access to specific buckets"""
        mock_iam_client = Mock()
        mock_sts_client = Mock()

        mock_boto3_client.side_effect = lambda service, **kwargs: {
            "iam": mock_iam_client,
            "sts": mock_sts_client,
        }[service]

        mock_sts_client.get_caller_identity.return_value = {
            "Arn": "arn:aws:sts::123456789012:assumed-role/TestRole/session",
            "Account": "123456789012",
        }

        # Mock IAM simulation that denies access to bucket2
        def mock_simulate_policy(PolicySourceArn, ActionNames, ResourceArns):
            results = []
            for action in ActionNames:
                for resource in ResourceArns:
                    if "bucket2" in resource:
                        # Deny access to bucket2
                        results.append(
                            {
                                "EvalActionName": action,
                                "EvalResourceName": resource,
                                "EvalDecision": "implicitDeny",
                            }
                        )
                    else:
                        # Allow access to bucket1
                        results.append(
                            {
                                "EvalActionName": action,
                                "EvalResourceName": resource,
                                "EvalDecision": "allowed",
                            }
                        )
            return {"EvaluationResults": results}

        mock_iam_client.simulate_principal_policy.side_effect = mock_simulate_policy

        errors = []
        required_permissions = [
            ("s3:GetObject", "arn:aws:s3:::bucket1/data/*"),
            ("s3:GetObject", "arn:aws:s3:::bucket2/output/*"),  # This should fail
        ]

        Validator._validate_calling_role_permissions(
            errors, required_permissions, None, "us-east-1"
        )

        # Should have 1 error for bucket2 access
        self.assertEqual(len(errors), 1)
        self.assertIn("bucket2", errors[0])
        self.assertIn("s3:GetObject", errors[0])

    @patch("amzn_nova_forge.validation.validator.boto3.client")
    def test_validate_permissions_fails_with_bucket_access_but_missing_object_access(
        self, mock_boto3_client
    ):
        """Test validation fails when we have bucket access but not specific object path access"""
        mock_iam_client = Mock()
        mock_sts_client = Mock()

        mock_boto3_client.side_effect = lambda service, **kwargs: {
            "iam": mock_iam_client,
            "sts": mock_sts_client,
        }[service]

        mock_sts_client.get_caller_identity.return_value = {
            "Arn": "arn:aws:sts::123456789012:assumed-role/TestRole/session",
            "Account": "123456789012",
        }

        def mock_simulate_policy(PolicySourceArn, ActionNames, ResourceArns, **kwargs):
            results = []
            for action in ActionNames:
                for resource in ResourceArns:
                    # Allow bucket-level operations but deny specific object path access
                    if "s3:ListBucket" in action and resource == "arn:aws:s3:::test-bucket":
                        decision = "allowed"
                    elif "s3:GetObject" in action and "specific/path" in resource:
                        decision = "denied"  # Deny access to specific object path
                    else:
                        decision = "allowed"

                    results.append(
                        {
                            "EvalActionName": action,
                            "EvalResourceName": resource,
                            "EvalDecision": decision,
                        }
                    )
            return {"EvaluationResults": results}

        mock_iam_client.simulate_principal_policy.side_effect = mock_simulate_policy

        errors = []
        required_permissions = [
            ("s3:ListBucket", "arn:aws:s3:::test-bucket"),  # This should pass
            (
                "s3:GetObject",
                "arn:aws:s3:::test-bucket/specific/path/*",
            ),  # This should fail
        ]

        Validator._validate_calling_role_permissions(
            errors, required_permissions, None, "us-east-1"
        )

        # Should have 1 error for the specific object path access
        self.assertEqual(len(errors), 1)
        self.assertIn("test-bucket/specific/path", errors[0])
        self.assertIn("s3:GetObject", errors[0])

    def test_validate_permissions_handles_lambda_evaluation_errors(self):
        """Test error handling when resource lambda functions fail during evaluation"""
        mock_infra = MagicMock()

        def failing_lambda(infra):
            raise Exception("Lambda evaluation failed")

        errors = []
        required_permissions = [("sagemaker:DescribeCluster", failing_lambda)]

        # This should handle the lambda failure gracefully
        with patch("boto3.client") as mock_boto3_client:
            mock_iam_client = MagicMock()
            mock_sts_client = MagicMock()

            mock_boto3_client.side_effect = lambda service, **kwargs: {
                "iam": mock_iam_client,
                "sts": mock_sts_client,
            }[service]

            mock_sts_client.get_caller_identity.return_value = {
                "Arn": "arn:aws:sts::123456789012:assumed-role/TestRole/session",
                "Account": "123456789012",
            }

            Validator._validate_calling_role_permissions(
                errors, required_permissions, mock_infra, "us-east-1"
            )

            self.assertEqual(len(errors), 1)
            self.assertIn("Failed to evaluate resource lambda", errors[0])

    @patch("sagemaker.core.helper.session_helper.get_execution_role")
    @patch("subprocess.run")
    def test_runtime_managers_define_mixed_permission_validation_types(
        self, mock_run, mock_get_execution_role
    ):
        """Test that runtime managers correctly define permissions using multiple validation approaches"""
        from amzn_nova_forge.manager.runtime_manager import (
            SMHPRuntimeManager,
            SMTJRuntimeManager,
        )

        # Mock subprocess for SMHP
        mock_run.return_value.stdout = ""
        mock_run.return_value.stderr = ""

        # Mock sagemaker execution role for SMTJ
        mock_get_execution_role.return_value = "arn:aws:iam::123456789012:role/test-role"

        # Test SMHP permissions
        smhp_permissions = SMHPRuntimeManager.required_calling_role_permissions()
        self.assertIsInstance(smhp_permissions, list)
        self.assertGreater(len(smhp_permissions), 0)

        # Should have tuple-specified calling role permissions
        has_tuples = any(isinstance(p, tuple) for p in smhp_permissions)
        self.assertTrue(has_tuples, "SMHP should have tuple permissions")

        # Test SMTJ permissions
        smtj_permissions = SMTJRuntimeManager.required_calling_role_permissions()
        self.assertIsInstance(smtj_permissions, list)
        self.assertGreater(len(smtj_permissions), 0)

        # Should have mixed types
        has_tuples = any(isinstance(p, tuple) for p in smtj_permissions)
        has_strings = any(isinstance(p, str) for p in smtj_permissions)
        self.assertTrue(has_tuples, "SMTJ should have tuple permissions")
        self.assertTrue(has_strings, "SMTJ should have string permissions")

    @patch("subprocess.run")
    @patch("amzn_nova_forge.manager.runtime_manager._get_caller_account_id")
    def test_smhp_required_permissions_uses_caller_account_id(self, mock_get_account_id, mock_run):
        """Happy path: account ID from STS appears in generated ARNs."""
        mock_run.return_value.stdout = ""
        mock_run.return_value.stderr = ""
        mock_get_account_id.return_value = "123456789012"

        mock_infra = MagicMock()
        mock_infra.region = "us-east-1"
        mock_infra.cluster_name = "forge-poc-dev"

        permissions = SMHPRuntimeManager.required_calling_role_permissions()
        lambda_perms = [(k, v) for k, v in permissions if callable(v)]

        self.assertGreater(len(lambda_perms), 0, "Should have lambda-based permissions")
        for action, resource_fn in lambda_perms:
            arn = resource_fn(mock_infra)
            self.assertIn(
                "123456789012",
                arn,
                f"{action} ARN should contain caller account ID, got: {arn}",
            )

    @patch("subprocess.run")
    @patch("amzn_nova_forge.manager.runtime_manager._get_caller_account_id")
    def test_smhp_required_permissions_falls_back_to_wildcard_on_sts_error(
        self, mock_get_account_id, mock_run
    ):
        """Fallback path: wildcard used when STS call raises an exception."""
        mock_run.return_value.stdout = ""
        mock_run.return_value.stderr = ""
        mock_get_account_id.return_value = "*"

        mock_infra = MagicMock()
        mock_infra.region = "us-west-2"
        mock_infra.cluster_name = "my-cluster"

        permissions = SMHPRuntimeManager.required_calling_role_permissions()
        lambda_perms = [(k, v) for k, v in permissions if callable(v)]

        self.assertGreater(len(lambda_perms), 0, "Should have lambda-based permissions")
        for action, resource_fn in lambda_perms:
            arn = resource_fn(mock_infra)
            account_field = arn.split(":")[4]
            self.assertEqual(
                account_field,
                "*",
                f"{action} ARN account field should be wildcard on STS error, got: {arn}",
            )

    @patch("amzn_nova_forge.validation.validator.boto3.client")
    def test_validate_calling_role_permissions_with_action_prefix_wildcards(
        self, mock_boto3_client
    ):
        """Test that validation correctly handles action prefix wildcards like iam:Get* in policies"""
        mock_iam_client = Mock()
        mock_sts_client = Mock()

        mock_boto3_client.side_effect = lambda service, **kwargs: {
            "iam": mock_iam_client,
            "sts": mock_sts_client,
        }[service]

        mock_sts_client.get_caller_identity.return_value = {
            "Arn": "arn:aws:sts::123456789012:assumed-role/TestRole/session",
            "Account": "123456789012",
        }

        # Mock policy with action prefix wildcard
        mock_iam_client.list_role_policies.return_value = {"PolicyNames": ["IAMPolicy"]}
        mock_iam_client.get_role_policy.return_value = {
            "PolicyDocument": {
                "Statement": [
                    {
                        "Effect": "Allow",
                        "Action": "iam:Get*",  # Action prefix wildcard
                        "Resource": "*",
                    }
                ]
            }
        }
        mock_iam_client.list_attached_role_policies.return_value = {"AttachedPolicies": []}

        errors = []
        required_permissions = ["iam:GetRole"]  # Should be covered by iam:Get*

        Validator._validate_calling_role_permissions(
            errors, required_permissions, None, "us-east-1"
        )

        # Should now support prefix wildcards like iam:Get*
        self.assertEqual(len(errors), 0)

    def test_validate_job_name_raises_exception(self):
        with self.assertRaises(ValueError) as context:
            Validator.validate_job_name("bad_job_name")

        self.assertEqual(
            str(context.exception),
            f"Job name must fit pattern {JOB_NAME_REGEX.pattern}",
        )

    def test_validate_job_name_does_not_raise_exception(self):
        Validator.validate_job_name("good-job-name")

    def test_validate_job_name_rejects_leading_hyphen(self):
        with self.assertRaises(ValueError):
            Validator.validate_job_name("-leading-hyphen")

    def test_validate_job_name_rejects_trailing_hyphen(self):
        with self.assertRaises(ValueError):
            Validator.validate_job_name("trailing-hyphen-")

    def test_validate_job_name_rejects_over_63_chars(self):
        with self.assertRaises(ValueError):
            Validator.validate_job_name("a" * 64)

    def test_validate_job_name_accepts_exactly_63_chars(self):
        Validator.validate_job_name("a" * 63)

    def test_validate_job_name_accepts_single_char(self):
        Validator.validate_job_name("a")

    def test_validate_job_name_rejects_empty_string(self):
        with self.assertRaises(ValueError):
            Validator.validate_job_name("")

    def test_validate_namespace_raises_exception(self):
        with self.assertRaises(ValueError) as context:
            Validator.validate_namespace("!bad_namespace")

        self.assertEqual(
            str(context.exception),
            f"Namespace must fit pattern {NAMESPACE_REGEX.pattern}",
        )

    def test_validate_namespace_does_not_raise_exception(self):
        Validator.validate_namespace("good-namespace")

    def test_validate_cluster_name_raises_exception(self):
        with self.assertRaises(ValueError) as context:
            Validator.validate_cluster_name("!bad_cluster_name")

        self.assertEqual(
            str(context.exception),
            f"Cluster name must fit pattern {CLUSTER_NAME_REGEX.pattern}",
        )

    def test_validate_cluster_name_does_not_raise_exception(self):
        Validator.validate_cluster_name("good_cluster-name")


class TestCrossAccountRegionPropagation(unittest.TestCase):
    """Test that _is_cross_account_role propagates region to boto3 clients."""

    @patch("amzn_nova_forge.validation.validator.boto3.client")
    def test_is_cross_account_role_passes_region_to_sts_client(self, mock_boto3_client):
        """Verify _is_cross_account_role passes region to boto3.client('sts', region_name=...)."""
        mock_sts = Mock()
        mock_sts.get_caller_identity.return_value = {"Account": "12345679012"}
        mock_boto3_client.return_value = mock_sts

        Validator._is_cross_account_role("arn:aws:iam::123456789012:role/test", region="eu-west-1")

        mock_boto3_client.assert_called_once_with("sts", region_name="eu-west-1")


class TestBedrockRegionValidation(unittest.TestCase):
    """Test cases for Bedrock region validation"""

    @patch("amzn_nova_forge.validation.validator.boto3.client")
    def test_bedrock_region_validation_rejects_non_us_east_1(self, mock_boto3_client):
        """Test that Bedrock validation rejects regions other than us-east-1"""
        from amzn_nova_forge.manager.runtime_manager import (
            BedrockRuntimeManager,
        )

        # Mock Bedrock runtime manager with us-west-2 region
        mock_bedrock_infra = Mock(spec=BedrockRuntimeManager)
        mock_bedrock_infra.region = "us-west-2"
        mock_bedrock_infra.instance_type = None
        mock_bedrock_infra.execution_role = "arn:aws:iam::123456789012:role/BedrockRole"

        # Attempt validation - should fail due to region
        with self.assertRaises(ValueError) as context:
            Validator.validate(
                platform=Platform.BEDROCK,
                method=TrainingMethod.SFT_LORA,
                infra=mock_bedrock_infra,
                recipe={},
                overrides_template={},
                validation_config=ValidationConfig(
                    iam=False,
                    infra=False,
                ),  # Disable other validations
            )

        error_message = str(context.exception)
        self.assertIn("us-west-2", error_message)
        self.assertIn("not supported", error_message)
        self.assertIn(
            "https://docs.aws.amazon.com/bedrock/latest/userguide/custom-model-fine-tuning.html",
            error_message,
        )

    @patch("amzn_nova_forge.validation.validator.boto3.client")
    def test_bedrock_region_validation_accepts_us_east_1(self, mock_boto3_client):
        """Test that Bedrock validation accepts us-east-1 region"""
        from amzn_nova_forge.manager.runtime_manager import (
            BedrockRuntimeManager,
        )

        # Mock Bedrock runtime manager with us-east-1 region
        mock_bedrock_infra = Mock(spec=BedrockRuntimeManager)
        mock_bedrock_infra.region = "us-east-1"
        mock_bedrock_infra.instance_type = None
        mock_bedrock_infra.execution_role = "arn:aws:iam::123456789012:role/BedrockRole"

        # Mock IAM/STS clients for permission validation
        mock_iam_client = Mock()
        mock_sts_client = Mock()

        mock_boto3_client.side_effect = lambda service, **kwargs: {
            "iam": mock_iam_client,
            "sts": mock_sts_client,
        }[service]

        mock_sts_client.get_caller_identity.return_value = {
            "Arn": "arn:aws:sts::123456789012:assumed-role/TestRole/session",
            "Account": "123456789012",
        }

        mock_iam_client.simulate_principal_policy.return_value = {
            "EvaluationResults": [{"EvalDecision": "allowed"}]
        }

        # Attempt validation - should succeed
        try:
            Validator.validate(
                platform=Platform.BEDROCK,
                method=TrainingMethod.SFT_LORA,
                infra=mock_bedrock_infra,
                recipe={},
                overrides_template={},
            )
        except ValueError as e:
            # Should not raise ValueError for region
            if "region" in str(e).lower() or "us-west-2" in str(e):
                self.fail(f"Validation failed with region error: {e}")

    @patch("amzn_nova_forge.validation.validator.boto3.client")
    def test_bedrock_region_validation_runs_even_with_iam_disabled(self, mock_boto3_client):
        """Test that region validation runs even when IAM validation is disabled"""
        from amzn_nova_forge.manager.runtime_manager import (
            BedrockRuntimeManager,
        )

        # Mock Bedrock runtime manager with invalid region
        mock_bedrock_infra = Mock(spec=BedrockRuntimeManager)
        mock_bedrock_infra.region = "eu-west-1"
        mock_bedrock_infra.instance_type = None
        mock_bedrock_infra.execution_role = "arn:aws:iam::123456789012:role/BedrockRole"

        # Attempt validation with IAM disabled - should still fail due to region
        with self.assertRaises(ValueError) as context:
            Validator.validate(
                platform=Platform.BEDROCK,
                method=TrainingMethod.SFT_LORA,
                infra=mock_bedrock_infra,
                recipe={},
                overrides_template={},
                validation_config=ValidationConfig(iam=False),  # IAM validation disabled
            )

        error_message = str(context.exception)
        self.assertIn("eu-west-1", error_message)
        self.assertIn("not supported", error_message)

    def test_smhp_name_without_sagemaker_raises(self):
        with self.assertRaises(ValueError) as ctx:
            validate_rft_lambda_name("my-reward-fn", Platform.SMHP)
        self.assertIn("SageMaker", str(ctx.exception))

    def test_smhp_name_with_sagemaker_passes(self):
        # Should not raise
        validate_rft_lambda_name("my-SageMaker-reward-fn", Platform.SMHP)

    def test_smhp_name_lowercase_sagemaker_raises(self):
        # 'sagemaker' without exact casing must fail
        with self.assertRaises(ValueError):
            validate_rft_lambda_name("my-sagemaker-reward-fn", Platform.SMHP)

    def test_smtj_any_name_passes(self):
        # SMTJ has no naming restriction
        validate_rft_lambda_name("my-reward-fn", Platform.SMTJ)
        validate_rft_lambda_name("sagemaker-reward", Platform.SMTJ)
        validate_rft_lambda_name("SageMaker-reward", Platform.SMTJ)


class TestValidatorNoneTypeHandling(unittest.TestCase):
    """Tests for the None value handling fix in _validate_recipe."""

    def setUp(self):
        self.mock_infra = Mock(spec=SMHPRuntimeManager)
        self.mock_infra.cluster_name = "test-cluster"
        self.mock_infra.instance_type = "ml.p5.48xlarge"
        self.mock_infra.instance_count = 4
        self.mock_infra.region = "us-east-1"

    def test_none_value_optional_field_does_not_raise(self):
        """Optional fields with None value (e.g. new hub content fields) must not raise."""
        recipe = {
            "model_package_group": None,
            "val_check_interval": None,
            "max_steps": 100,
        }
        overrides_template = {
            "model_package_group": {"type": "string"},  # optional, no required=True
            "val_check_interval": {"type": "integer"},  # optional, no required=True
            "max_steps": {"type": "integer", "default": 100},
        }
        errors = []
        Validator._validate_recipe(
            recipe=recipe,
            overrides_template=overrides_template,
            instance_type="ml.p5.48xlarge",
            errors=errors,
            method=TrainingMethod.SFT_LORA,
        )
        self.assertEqual(errors, [])

    def test_none_value_required_field_still_raises(self):
        """Required fields with None value must still produce a type error."""
        recipe = {"max_steps": None}
        overrides_template = {
            "max_steps": {"type": "integer", "required": True},
        }
        errors = []
        Validator._validate_recipe(
            recipe=recipe,
            overrides_template=overrides_template,
            instance_type="ml.p5.48xlarge",
            errors=errors,
            method=TrainingMethod.SFT_LORA,
        )
        self.assertTrue(
            any("expects integer" in e and "NoneType" in e for e in errors),
            f"Expected type error for required None field, got: {errors}",
        )

    def test_valid_string_field_passes(self):
        """A properly set string field must not produce errors."""
        recipe = {"model_package_group": "my-group"}
        overrides_template = {"model_package_group": {"type": "string"}}
        errors = []
        Validator._validate_recipe(
            recipe=recipe,
            overrides_template=overrides_template,
            instance_type="ml.p5.48xlarge",
            errors=errors,
            method=TrainingMethod.SFT_LORA,
        )
        self.assertEqual(errors, [])

    def test_wrong_type_still_raises(self):
        """A field with wrong type (not None) must still produce a type error."""
        recipe = {"max_steps": "not-an-int"}
        overrides_template = {"max_steps": {"type": "integer"}}
        errors = []
        Validator._validate_recipe(
            recipe=recipe,
            overrides_template=overrides_template,
            instance_type="ml.p5.48xlarge",
            errors=errors,
            method=TrainingMethod.SFT_LORA,
        )
        self.assertTrue(
            any("expects integer" in e for e in errors),
            f"Expected type error, got: {errors}",
        )

    def test_none_value_accepted_when_not_required(self):
        """None value must pass validation when the field is not required (e.g. reasoning_effort)."""
        overrides_template = {
            "reasoning_effort": {
                "type": "str",
                "required": False,
                "enum": ["low", "medium", "high"],
                "default": "medium",
            },
        }
        recipe = {
            "training_config": {
                "reasoning_effort": None,
            },
        }
        errors = []
        Validator._validate_recipe(
            recipe=recipe,
            overrides_template=overrides_template,
            instance_type="ml.p4d.24xlarge",
            errors=errors,
            method=TrainingMethod.RFT_LORA,
            platform=Platform.SMTJ,
            rft_lambda_arn="arn:aws:lambda:us-west-2:123456789012:function:reward-fn",
        )
        self.assertEqual(errors, [])

    def test_none_value_required_str_field_without_null_enum_still_raises(self):
        """None on a required str field must still error when null is NOT in the enum."""
        overrides_template = {
            "reasoning_effort": {
                "type": "str",
                "required": True,
                "enum": ["low", "medium", "high"],
                "default": "medium",
            },
        }
        recipe = {
            "training_config": {
                "reasoning_effort": None,
            },
        }
        errors = []
        Validator._validate_recipe(
            recipe=recipe,
            overrides_template=overrides_template,
            instance_type="ml.p4d.24xlarge",
            errors=errors,
            method=TrainingMethod.RFT_LORA,
            platform=Platform.SMTJ,
            rft_lambda_arn="arn:aws:lambda:us-west-2:123456789012:function:reward-fn",
        )
        self.assertEqual(len(errors), 1)
        self.assertIn("expects str", errors[0])
        self.assertIn("NoneType", errors[0])


class TestMTRLServerlessValidation(unittest.TestCase):
    """Tests for MTRL serverless validation logic in Validator._validate_recipe."""

    def setUp(self):
        self.mock_infra = Mock(spec=SMTJRuntimeManager)
        self.mock_infra.instance_type = "ml.p5.48xlarge"
        self.mock_infra.region = "us-east-1"

    def test_mtrl_serverless_skips_override_constraint_validation(self):
        """MTRL serverless jobs skip min/max/enum/type validation for most keys."""
        recipe = {
            "training_config": {
                "global_batch_size": 2,  # Would fail min=16 constraint for single-turn RFT
                "max_steps": 5,
            },
            "run": {
                "output_s3_path": "s3://bucket/output/",
                "data_s3_path": "s3://bucket/data/",
            },
        }
        overrides_template = {
            "global_batch_size": {"type": "integer", "min": 16, "max": 256},
            "max_steps": {"type": "integer", "min": 10, "max": 100000},
            "output_s3_path": {"type": "string", "required": True},
            "data_s3_path": {"type": "string", "required": True},
        }

        errors = []
        Validator._validate_recipe(
            recipe=recipe,
            overrides_template=overrides_template,
            instance_type=None,
            errors=errors,
            method=TrainingMethod.RFT_MULTITURN_LORA,
            platform=Platform.SMTJServerless,
        )

        # Should have NO errors because MTRL serverless skips constraint validation
        self.assertEqual(len(errors), 0)

    def test_non_mtrl_serverless_still_validates_constraints(self):
        """Non-MTRL methods on SMTJServerless should still validate constraints."""
        recipe = {"training_config": {"max_steps": 2}}
        overrides_template = {"max_steps": {"type": "integer", "min": 4}}

        errors = []
        Validator._validate_recipe(
            recipe=recipe,
            overrides_template=overrides_template,
            instance_type=None,
            errors=errors,
            method=TrainingMethod.SFT_LORA,
            platform=Platform.SMTJServerless,
        )

        # Should have error because SFT_LORA still validates
        self.assertEqual(len(errors), 1)
        self.assertIn("must be at least 4", errors[0])

    def test_mtrl_serverless_still_validates_output_s3_path(self):
        """MTRL serverless should still validate output_s3_path, data_s3_path, model_type."""
        recipe = {"run": {"output_s3_path": 12345}}  # Wrong type for output_s3_path
        overrides_template = {
            "output_s3_path": {"type": "string", "required": True},
        }

        errors = []
        Validator._validate_recipe(
            recipe=recipe,
            overrides_template=overrides_template,
            instance_type=None,
            errors=errors,
            method=TrainingMethod.RFT_MULTITURN_LORA,
            platform=Platform.SMTJServerless,
        )

        # output_s3_path is NOT skipped, so type validation should still catch this
        self.assertEqual(len(errors), 1)
        self.assertIn("output_s3_path", errors[0])

    @patch("amzn_nova_forge.validation.validator.boto3.client")
    def test_mtrl_serverless_accepts_agent_core_arn_no_lambda_required(self, mock_boto3_client):
        """RFT_MULTITURN_LORA on SMTJServerless does not require rft_lambda_arn."""
        mock_infra = Mock(spec=SMTJRuntimeManager)
        mock_infra.instance_type = None
        mock_infra.region = "us-east-1"

        # Should NOT raise — MTRL on SMTJServerless uses agent_core_arn, no lambda required
        try:
            Validator.validate(
                platform=Platform.SMTJServerless,
                method=TrainingMethod.RFT_MULTITURN_LORA,
                infra=mock_infra,
                recipe={},
                overrides_template={},
                validation_config=ValidationConfig(iam=False, infra=False, recipe=False),
            )
        except ValueError as e:
            self.fail(f"MTRL serverless validation should not require lambda ARN, but raised: {e}")

    @patch("amzn_nova_forge.validation.validator.boto3.client")
    def test_non_mtrl_method_does_not_skip_validation(self, mock_boto3_client):
        """Non-MTRL methods on serverless do NOT get the _is_mtrl_serverless skip."""
        mock_infra = Mock(spec=SMTJRuntimeManager)
        mock_infra.instance_type = "ml.p5.48xlarge"
        mock_infra.region = "us-east-1"

        # SFT_LORA on serverless should still validate normally (no skip)
        # This should not raise because recipe={} with no overrides is valid
        Validator.validate(
            platform=Platform.SMTJServerless,
            method=TrainingMethod.SFT_LORA,
            infra=mock_infra,
            recipe={},
            overrides_template={},
            validation_config=ValidationConfig(iam=False, infra=False, recipe=False),
        )


class TestValidationDataS3Path(unittest.TestCase):
    """Tests for validation_data_s3_path preflight validation."""

    @patch("amzn_nova_forge.validation.validator.boto3.client")
    def test_preflight_validates_validation_data_s3_path(self, mock_boto3_client):
        """Assert validation runs for validation_data_s3_path when set."""
        mock_infra = Mock(spec=SMTJRuntimeManager)
        mock_infra.instance_type = "ml.p5.48xlarge"
        mock_infra.region = "us-east-1"

        with self.assertRaises(ValueError) as context:
            Validator.validate(
                platform=Platform.SMTJ,
                method=TrainingMethod.SFT_LORA,
                infra=mock_infra,
                recipe={},
                overrides_template={},
                validation_data_s3_path="not-an-s3-path",
                validation_config=ValidationConfig(iam=False, infra=False),
            )
        self.assertIn("Invalid S3 path for validation_data_s3_path", str(context.exception))

    @patch("amzn_nova_forge.validation.validator.boto3.client")
    def test_preflight_skips_validation_data_when_none(self, mock_boto3_client):
        """Assert no validation error when validation_data_s3_path is None."""
        mock_infra = Mock(spec=SMTJRuntimeManager)
        mock_infra.instance_type = "ml.p5.48xlarge"
        mock_infra.region = "us-east-1"

        try:
            Validator.validate(
                platform=Platform.SMTJ,
                method=TrainingMethod.SFT_LORA,
                infra=mock_infra,
                recipe={},
                overrides_template={},
                validation_data_s3_path=None,
                validation_config=ValidationConfig(iam=False, infra=False),
            )
        except ValueError as e:
            self.assertNotIn("validation_data_s3_path", str(e))


if __name__ == "__main__":
    unittest.main()
