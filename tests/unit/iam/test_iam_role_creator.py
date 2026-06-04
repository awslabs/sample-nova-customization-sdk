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
import unittest
from unittest.mock import MagicMock, patch

from amzn_nova_forge.iam.iam_role_creator import (
    _attach_policies,
    create_bedrock_batch_inference_execution_role,
    create_bedrock_execution_role,
    create_sagemaker_execution_role,
    create_sagemaker_invoke_role,
    create_smtj_dataprep_execution_role,
)


class TestAttachPolicies(unittest.TestCase):
    def _make_iam_client(self):
        mock_iam = MagicMock()
        mock_iam.exceptions.EntityAlreadyExistsException = type(
            "EntityAlreadyExistsException", (Exception,), {}
        )
        return mock_iam

    def test_policy_already_attached_is_skipped(self):
        """Policy already attached to role → no create or attach calls."""
        mock_iam = self._make_iam_client()
        mock_iam.list_attached_role_policies.return_value = {
            "AttachedPolicies": [{"PolicyName": "my-roleMy_Policy"}]
        }
        policies = {"my_policy": {"Version": "2012-10-17", "Statement": []}}

        _attach_policies(mock_iam, "123456789012", "my-role", ["my_policy"], policies)

        mock_iam.create_policy.assert_not_called()
        mock_iam.attach_role_policy.assert_not_called()

    def test_policy_already_attached_case_insensitive(self):
        """Policy attached with different casing (e.g. Sagemaker vs SageMaker) → still skipped."""
        mock_iam = self._make_iam_client()
        mock_iam.list_attached_role_policies.return_value = {
            "AttachedPolicies": [{"PolicyName": "my-rolemy_policy"}]
        }
        policies = {"my_policy": {"Version": "2012-10-17", "Statement": []}}

        _attach_policies(mock_iam, "123456789012", "my-role", ["my_policy"], policies)

        mock_iam.create_policy.assert_not_called()
        mock_iam.attach_role_policy.assert_not_called()

    def test_policy_not_attached_is_created_and_attached(self):
        """Policy doesn't exist → created and attached."""
        mock_iam = self._make_iam_client()
        mock_iam.list_attached_role_policies.return_value = {"AttachedPolicies": []}
        mock_iam.create_policy.return_value = {
            "Policy": {"Arn": "arn:aws:iam::123456789012:policy/my-roleMy_Policy"}
        }
        policies = {"my_policy": {"Version": "2012-10-17", "Statement": []}}

        _attach_policies(mock_iam, "123456789012", "my-role", ["my_policy"], policies)

        mock_iam.create_policy.assert_called_once_with(
            PolicyName="my-roleMy_Policy",
            PolicyDocument=json.dumps(policies["my_policy"]),
        )
        mock_iam.attach_role_policy.assert_called_once_with(
            RoleName="my-role",
            PolicyArn="arn:aws:iam::123456789012:policy/my-roleMy_Policy",
        )

    def test_policy_exists_in_account_but_not_attached(self):
        """Policy exists in account (EntityAlreadyExistsException) → fetched and attached."""
        mock_iam = self._make_iam_client()
        mock_iam.list_attached_role_policies.return_value = {"AttachedPolicies": []}
        mock_iam.create_policy.side_effect = mock_iam.exceptions.EntityAlreadyExistsException(
            "already exists"
        )
        mock_iam.get_policy.return_value = {
            "Policy": {"Arn": "arn:aws:iam::123456789012:policy/my-roleMy_Policy"}
        }
        policies = {"my_policy": {"Version": "2012-10-17", "Statement": []}}

        _attach_policies(mock_iam, "123456789012", "my-role", ["my_policy"], policies)

        mock_iam.get_policy.assert_called_once_with(
            PolicyArn="arn:aws:iam::123456789012:policy/my-roleMy_Policy"
        )
        mock_iam.attach_role_policy.assert_called_once_with(
            RoleName="my-role",
            PolicyArn="arn:aws:iam::123456789012:policy/my-roleMy_Policy",
        )

    def test_create_policy_failure_raises_exception(self):
        """Unexpected error during create_policy → raises Exception."""
        mock_iam = self._make_iam_client()
        mock_iam.list_attached_role_policies.return_value = {"AttachedPolicies": []}
        mock_iam.create_policy.side_effect = Exception("some AWS error")
        policies = {"my_policy": {"Version": "2012-10-17", "Statement": []}}

        with self.assertRaises(Exception) as ctx:
            _attach_policies(mock_iam, "123456789012", "my-role", ["my_policy"], policies)

        self.assertIn("Failed to create or attach policy my_policy", str(ctx.exception))


class TestIAMRoleCreator(unittest.TestCase):
    @patch("boto3.client")
    def test_create_bedrock_execution_role_with_wildcards(self, mock_boto_client):
        mock_sts = MagicMock()
        mock_sts.get_caller_identity.return_value = {"Account": "123456789012"}
        mock_boto_client.return_value = mock_sts

        role_name = "role_name"
        mock_iam_client = MagicMock()
        mock_iam_client.exceptions.NoSuchEntityException = type(
            "NoSuchEntityException", (Exception,), {}
        )
        mock_iam_client.exceptions.EntityAlreadyExistsException = type(
            "EntityAlreadyExistsException", (Exception,), {}
        )
        mock_iam_client.get_role.side_effect = mock_iam_client.exceptions.NoSuchEntityException(
            "Role not found"
        )
        mock_iam_client.create_policy.return_value = {
            "Policy": {"Arn": "arn:aws:iam::123456789012:policy/foo"}
        }

        create_bedrock_execution_role(mock_iam_client, role_name)

        mock_iam_client.create_policy.assert_any_call(
            PolicyName=f"{role_name}Bedrock_Policy",
            PolicyDocument=json.dumps(
                {
                    "Version": "2012-10-17",
                    "Statement": [
                        {
                            "Effect": "Allow",
                            "Action": [
                                "bedrock:CreateCustomModelDeployment",
                                "bedrock:CreateCustomModel",
                                "bedrock:CreateProvisionedModelThroughput",
                                "bedrock:GetCustomModel",
                                "bedrock:GetCustomModelDeployment",
                            ],
                            "Resource": "*",
                        }
                    ],
                }
            ),
        )

        mock_iam_client.create_policy.assert_any_call(
            PolicyName=f"{role_name}S3_Read_Policy",
            PolicyDocument=json.dumps(
                {
                    "Version": "2012-10-17",
                    "Statement": [
                        {
                            "Effect": "Allow",
                            "Action": ["s3:GetObject", "s3:ListBucket"],
                            "Resource": "*",
                        }
                    ],
                }
            ),
        )

    @patch("boto3.client")
    def test_create_bedrock_execution_role_with_scoped_resources(self, mock_boto_client):
        account_id = "123456789012"
        mock_sts = MagicMock()
        mock_sts.get_caller_identity.return_value = {"Account": account_id}
        mock_boto_client.return_value = mock_sts

        role_name = "role_name"
        scoped_resource = "resource"
        mock_iam_client = MagicMock()
        mock_iam_client.exceptions.NoSuchEntityException = type(
            "NoSuchEntityException", (Exception,), {}
        )
        mock_iam_client.exceptions.EntityAlreadyExistsException = type(
            "EntityAlreadyExistsException", (Exception,), {}
        )
        mock_iam_client.get_role.side_effect = mock_iam_client.exceptions.NoSuchEntityException(
            "Role not found"
        )
        mock_iam_client.create_policy.return_value = {
            "Policy": {"Arn": f"arn:aws:iam::{account_id}:policy/foo"}
        }

        create_bedrock_execution_role(mock_iam_client, role_name, scoped_resource, scoped_resource)

        mock_iam_client.create_policy.assert_any_call(
            PolicyName=f"{role_name}Bedrock_Policy",
            PolicyDocument=json.dumps(
                {
                    "Version": "2012-10-17",
                    "Statement": [
                        {
                            "Effect": "Allow",
                            "Action": [
                                "bedrock:CreateCustomModelDeployment",
                                "bedrock:CreateCustomModel",
                                "bedrock:CreateProvisionedModelThroughput",
                                "bedrock:GetCustomModel",
                                "bedrock:GetCustomModelDeployment",
                            ],
                            "Resource": f"arn:aws:bedrock:*:*:custom-model/{scoped_resource}*",
                        }
                    ],
                }
            ),
        )

        mock_iam_client.create_policy.assert_any_call(
            PolicyName=f"{role_name}S3_Read_Policy",
            PolicyDocument=json.dumps(
                {
                    "Version": "2012-10-17",
                    "Statement": [
                        {
                            "Effect": "Allow",
                            "Action": ["s3:GetObject", "s3:ListBucket"],
                            "Resource": [
                                f"arn:aws:s3:::{scoped_resource}*",
                                f"arn:aws:s3:::{scoped_resource}*/*",
                                f"arn:aws:s3:::customer-escrow-{account_id}*",
                                f"arn:aws:s3:::customer-escrow-{account_id}*/*",
                            ],
                        }
                    ],
                }
            ),
        )

    @patch("boto3.client")
    def test_create_sagemaker_execution_role_with_wildcards(self, mock_boto_client):
        mock_sts = MagicMock()
        mock_sts.get_caller_identity.return_value = {"Account": "123456789012"}
        mock_boto_client.return_value = mock_sts

        role_name = "role_name"
        mock_iam_client = MagicMock()
        mock_iam_client.get_role.side_effect = mock_iam_client.exceptions.NoSuchEntityException(
            "Role not found"
        )
        mock_iam_client.create_policy.return_value = {
            "Policy": {"Arn": "arn:aws:iam::123456789012:policy/foo"}
        }

        create_sagemaker_execution_role(mock_iam_client, role_name)

        mock_iam_client.create_policy.assert_any_call(
            PolicyName=f"{role_name}Kms_Policy",
            PolicyDocument=json.dumps(
                {
                    "Version": "2012-10-17",
                    "Statement": [
                        {
                            "Effect": "Allow",
                            "Action": [
                                "kms:Encrypt",
                            ],
                            "Resource": "*",
                        }
                    ],
                }
            ),
        )

        mock_iam_client.create_policy.assert_any_call(
            PolicyName=f"{role_name}S3_Read_Policy",
            PolicyDocument=json.dumps(
                {
                    "Version": "2012-10-17",
                    "Statement": [
                        {
                            "Effect": "Allow",
                            "Action": ["s3:GetObject", "s3:ListBucket"],
                            "Resource": "*",
                        }
                    ],
                }
            ),
        )

    @patch("boto3.client")
    def test_create_sagemaker_execution_role_with_scoped_resources(self, mock_boto_client):
        account_id = "123456789012"
        mock_sts = MagicMock()
        mock_sts.get_caller_identity.return_value = {"Account": account_id}
        mock_boto_client.return_value = mock_sts

        role_name = "role_name"
        scoped_resource = "resource"
        mock_iam_client = MagicMock()
        mock_iam_client.get_role.side_effect = mock_iam_client.exceptions.NoSuchEntityException(
            "Role not found"
        )
        mock_iam_client.create_policy.return_value = {
            "Policy": {"Arn": f"arn:aws:iam::{account_id}:policy/foo"}
        }

        create_sagemaker_execution_role(
            iam_client=mock_iam_client,
            role_name=role_name,
            s3_resource=scoped_resource,
            kms_resource=scoped_resource,
        )

        mock_iam_client.create_policy.assert_any_call(
            PolicyName=f"{role_name}Kms_Policy",
            PolicyDocument=json.dumps(
                {
                    "Version": "2012-10-17",
                    "Statement": [
                        {
                            "Effect": "Allow",
                            "Action": [
                                "kms:Encrypt",
                            ],
                            "Resource": f"arn:aws:kms:*:*:key/{scoped_resource}*",
                        }
                    ],
                }
            ),
        )

        mock_iam_client.create_policy.assert_any_call(
            PolicyName=f"{role_name}S3_Read_Policy",
            PolicyDocument=json.dumps(
                {
                    "Version": "2012-10-17",
                    "Statement": [
                        {
                            "Effect": "Allow",
                            "Action": ["s3:GetObject", "s3:ListBucket"],
                            "Resource": [
                                f"arn:aws:s3:::{scoped_resource}*",
                                f"arn:aws:s3:::{scoped_resource}*/*",
                                f"arn:aws:s3:::customer-escrow-{account_id}*",
                                f"arn:aws:s3:::customer-escrow-{account_id}*/*",
                            ],
                        }
                    ],
                }
            ),
        )


class TestCreateSagemakerInvokeRole(unittest.TestCase):
    """Tests for create_sagemaker_invoke_role."""

    ACCOUNT_ID = "123456789012"
    ROLE_NAME = "GlueDataPrepRole"

    def _make_iam_client(self):
        mock_iam = MagicMock()
        mock_iam.exceptions.NoSuchEntityException = type("NoSuchEntityException", (Exception,), {})
        mock_iam.exceptions.EntityAlreadyExistsException = type(
            "EntityAlreadyExistsException", (Exception,), {}
        )
        mock_iam.exceptions.LimitExceededException = type(
            "LimitExceededException", (Exception,), {}
        )
        return mock_iam

    def _make_sts_mock(self, mock_boto_client):
        mock_sts = MagicMock()
        mock_sts.get_caller_identity.return_value = {"Account": self.ACCOUNT_ID}
        mock_boto_client.return_value = mock_sts
        return mock_sts

    @patch("boto3.client")
    def test_creates_role_when_not_exists(self, mock_boto_client):
        """Role doesn't exist → create_role is called, result returned."""
        self._make_sts_mock(mock_boto_client)
        mock_iam = self._make_iam_client()
        mock_iam.get_role.side_effect = mock_iam.exceptions.NoSuchEntityException("Role not found")
        expected_response = {
            "Role": {"Arn": f"arn:aws:iam::{self.ACCOUNT_ID}:role/{self.ROLE_NAME}"}
        }
        mock_iam.create_role.return_value = expected_response
        mock_iam.create_policy.return_value = {
            "Policy": {"Arn": f"arn:aws:iam::{self.ACCOUNT_ID}:policy/foo"}
        }

        result = create_sagemaker_invoke_role(mock_iam, self.ROLE_NAME)

        mock_iam.create_role.assert_called_once()
        call_kwargs = mock_iam.create_role.call_args[1]
        self.assertEqual(call_kwargs["RoleName"], self.ROLE_NAME)
        trust = json.loads(call_kwargs["AssumeRolePolicyDocument"])
        self.assertEqual(trust["Statement"][0]["Principal"], {"Service": "glue.amazonaws.com"})
        self.assertEqual(result, expected_response)

    @patch("boto3.client")
    def test_returns_existing_role_when_already_exists(self, mock_boto_client):
        """Role already exists → get_role result returned, create_role not called."""
        self._make_sts_mock(mock_boto_client)
        mock_iam = self._make_iam_client()
        existing_role = {"Role": {"Arn": f"arn:aws:iam::{self.ACCOUNT_ID}:role/{self.ROLE_NAME}"}}
        mock_iam.get_role.return_value = existing_role
        mock_iam.create_policy.return_value = {
            "Policy": {"Arn": f"arn:aws:iam::{self.ACCOUNT_ID}:policy/foo"}
        }

        result = create_sagemaker_invoke_role(mock_iam, self.ROLE_NAME)

        mock_iam.create_role.assert_not_called()
        self.assertEqual(result, existing_role)
        # Policies should still be created/attached
        self.assertEqual(mock_iam.create_policy.call_count, 3)
        self.assertEqual(mock_iam.attach_role_policy.call_count, 3)

    @patch("boto3.client")
    def test_scoped_s3_and_glue_resources(self, mock_boto_client):
        """Non-wildcard s3_resource and glue_job_resource produce scoped ARNs."""
        self._make_sts_mock(mock_boto_client)
        mock_iam = self._make_iam_client()
        mock_iam.get_role.side_effect = mock_iam.exceptions.NoSuchEntityException("Role not found")
        mock_iam.create_role.return_value = {
            "Role": {"Arn": f"arn:aws:iam::{self.ACCOUNT_ID}:role/{self.ROLE_NAME}"}
        }
        mock_iam.create_policy.return_value = {
            "Policy": {"Arn": f"arn:aws:iam::{self.ACCOUNT_ID}:policy/foo"}
        }

        create_sagemaker_invoke_role(
            mock_iam,
            self.ROLE_NAME,
            s3_resource="my-bucket",
            glue_job_resource="my-job",
        )

        # Inspect the policy documents passed to create_policy
        policy_calls = {
            call[1]["PolicyName"]: json.loads(call[1]["PolicyDocument"])
            for call in mock_iam.create_policy.call_args_list
        }

        # Glue base policy should be scoped to the specific job
        glue_doc = policy_calls[f"{self.ROLE_NAME}Glue_Base_Policy"]
        self.assertEqual(
            glue_doc["Statement"][0]["Resource"],
            [f"arn:aws:glue:*:{self.ACCOUNT_ID}:job/my-job"],
        )

        # S3 policy should be scoped to the specific bucket
        s3_doc = policy_calls[f"{self.ROLE_NAME}Glue_S3_Policy"]
        self.assertEqual(
            s3_doc["Statement"][0]["Resource"],
            ["arn:aws:s3:::my-bucket", "arn:aws:s3:::my-bucket/*"],
        )

    @patch("boto3.client")
    def test_wildcard_resources(self, mock_boto_client):
        """Wildcard s3_resource and glue_job_resource produce '*' resources."""
        self._make_sts_mock(mock_boto_client)
        mock_iam = self._make_iam_client()
        mock_iam.get_role.side_effect = mock_iam.exceptions.NoSuchEntityException("Role not found")
        mock_iam.create_role.return_value = {
            "Role": {"Arn": f"arn:aws:iam::{self.ACCOUNT_ID}:role/{self.ROLE_NAME}"}
        }
        mock_iam.create_policy.return_value = {
            "Policy": {"Arn": f"arn:aws:iam::{self.ACCOUNT_ID}:policy/foo"}
        }

        create_sagemaker_invoke_role(mock_iam, self.ROLE_NAME)

        policy_calls = {
            call[1]["PolicyName"]: json.loads(call[1]["PolicyDocument"])
            for call in mock_iam.create_policy.call_args_list
        }

        glue_doc = policy_calls[f"{self.ROLE_NAME}Glue_Base_Policy"]
        self.assertEqual(glue_doc["Statement"][0]["Resource"], "*")

        s3_doc = policy_calls[f"{self.ROLE_NAME}Glue_S3_Policy"]
        self.assertEqual(s3_doc["Statement"][0]["Resource"], "*")

    @patch("boto3.client")
    def test_trust_principal_override(self, mock_boto_client):
        """Custom trust_principal replaces the default glue.amazonaws.com."""
        self._make_sts_mock(mock_boto_client)
        mock_iam = self._make_iam_client()
        mock_iam.get_role.side_effect = mock_iam.exceptions.NoSuchEntityException("Role not found")
        mock_iam.create_role.return_value = {
            "Role": {"Arn": f"arn:aws:iam::{self.ACCOUNT_ID}:role/{self.ROLE_NAME}"}
        }
        mock_iam.create_policy.return_value = {
            "Policy": {"Arn": f"arn:aws:iam::{self.ACCOUNT_ID}:policy/foo"}
        }

        custom_principal = {"AWS": f"arn:aws:iam::{self.ACCOUNT_ID}:root"}
        create_sagemaker_invoke_role(mock_iam, self.ROLE_NAME, trust_principal=custom_principal)

        trust_doc = json.loads(mock_iam.create_role.call_args[1]["AssumeRolePolicyDocument"])
        self.assertEqual(trust_doc["Statement"][0]["Principal"], custom_principal)

    @patch("boto3.client")
    def test_policy_already_exists_updates_and_attaches(self, mock_boto_client):
        """EntityAlreadyExistsException → get existing policy, update doc, attach."""
        self._make_sts_mock(mock_boto_client)
        mock_iam = self._make_iam_client()
        mock_iam.get_role.return_value = {
            "Role": {"Arn": f"arn:aws:iam::{self.ACCOUNT_ID}:role/{self.ROLE_NAME}"}
        }

        # All three create_policy calls hit EntityAlreadyExistsException
        mock_iam.create_policy.side_effect = mock_iam.exceptions.EntityAlreadyExistsException(
            "already exists"
        )
        mock_iam.get_policy.return_value = {
            "Policy": {"Arn": f"arn:aws:iam::{self.ACCOUNT_ID}:policy/existing"}
        }

        create_sagemaker_invoke_role(mock_iam, self.ROLE_NAME)

        # get_policy called once per policy (3 policies)
        self.assertEqual(mock_iam.get_policy.call_count, 3)
        # create_policy_version called once per policy (update)
        self.assertEqual(mock_iam.create_policy_version.call_count, 3)
        # attach_role_policy called once per policy
        self.assertEqual(mock_iam.attach_role_policy.call_count, 3)

    @patch("boto3.client")
    def test_unexpected_get_role_error_raises(self, mock_boto_client):
        """Non-NoSuchEntityException from get_role → raises with descriptive message."""
        self._make_sts_mock(mock_boto_client)
        mock_iam = self._make_iam_client()
        mock_iam.get_role.side_effect = Exception("AccessDenied")

        with self.assertRaises(Exception) as ctx:
            create_sagemaker_invoke_role(mock_iam, self.ROLE_NAME)

        self.assertIn("Failed to create the Glue data prep role", str(ctx.exception))

    @patch("boto3.client")
    def test_unexpected_policy_error_raises(self, mock_boto_client):
        """Non-EntityAlreadyExistsException from create_policy → raises."""
        self._make_sts_mock(mock_boto_client)
        mock_iam = self._make_iam_client()
        mock_iam.get_role.return_value = {
            "Role": {"Arn": f"arn:aws:iam::{self.ACCOUNT_ID}:role/{self.ROLE_NAME}"}
        }
        mock_iam.create_policy.side_effect = Exception("some AWS error")

        with self.assertRaises(Exception) as ctx:
            create_sagemaker_invoke_role(mock_iam, self.ROLE_NAME)

        self.assertIn("Failed to create or attach policy", str(ctx.exception))

    @patch("boto3.client")
    def test_all_three_policies_created_and_attached(self, mock_boto_client):
        """Exactly glue_base, glue_s3, and glue_logs policies are created."""
        self._make_sts_mock(mock_boto_client)
        mock_iam = self._make_iam_client()
        mock_iam.get_role.side_effect = mock_iam.exceptions.NoSuchEntityException("Role not found")
        mock_iam.create_role.return_value = {
            "Role": {"Arn": f"arn:aws:iam::{self.ACCOUNT_ID}:role/{self.ROLE_NAME}"}
        }
        mock_iam.create_policy.return_value = {
            "Policy": {"Arn": f"arn:aws:iam::{self.ACCOUNT_ID}:policy/foo"}
        }

        create_sagemaker_invoke_role(mock_iam, self.ROLE_NAME)

        created_policy_names = [
            call[1]["PolicyName"] for call in mock_iam.create_policy.call_args_list
        ]
        self.assertEqual(
            created_policy_names,
            [
                f"{self.ROLE_NAME}Glue_Base_Policy",
                f"{self.ROLE_NAME}Glue_S3_Policy",
                f"{self.ROLE_NAME}Glue_Logs_Policy",
            ],
        )

    @patch("boto3.client")
    def test_create_smtj_dataprep_execution_role_with_wildcards(self, mock_boto_client):
        mock_sts = MagicMock()
        mock_sts.get_caller_identity.return_value = {"Account": "123456789012"}
        mock_boto_client.return_value = mock_sts

        role_name = "SmtjDataPrepRole"
        mock_iam_client = MagicMock()
        mock_iam_client.exceptions.NoSuchEntityException = type(
            "NoSuchEntityException", (Exception,), {}
        )
        mock_iam_client.exceptions.EntityAlreadyExistsException = type(
            "EntityAlreadyExistsException", (Exception,), {}
        )
        mock_iam_client.get_role.side_effect = mock_iam_client.exceptions.NoSuchEntityException(
            "Role not found"
        )
        mock_iam_client.create_policy.return_value = {
            "Policy": {"Arn": "arn:aws:iam::123456789012:policy/foo"}
        }

        create_smtj_dataprep_execution_role(mock_iam_client, role_name)

        # Verify trust policy allows sagemaker.amazonaws.com
        mock_iam_client.create_role.assert_called_once()
        trust_doc = json.loads(mock_iam_client.create_role.call_args[1]["AssumeRolePolicyDocument"])
        self.assertEqual(
            trust_doc["Statement"][0]["Principal"]["Service"],
            "sagemaker.amazonaws.com",
        )

        # Verify S3 policy includes PutObject
        mock_iam_client.create_policy.assert_any_call(
            PolicyName=f"{role_name}S3_Policy",
            PolicyDocument=json.dumps(
                {
                    "Version": "2012-10-17",
                    "Statement": [
                        {
                            "Effect": "Allow",
                            "Action": [
                                "s3:GetObject",
                                "s3:PutObject",
                                "s3:ListBucket",
                            ],
                            "Resource": "*",
                        }
                    ],
                }
            ),
        )

        # Verify ECR policy has 2 statements: GetAuthorizationToken (Resource: *)
        # and pull actions (Resource: * when no scoping)
        mock_iam_client.create_policy.assert_any_call(
            PolicyName=f"{role_name}Ecr_Read_Policy",
            PolicyDocument=json.dumps(
                {
                    "Version": "2012-10-17",
                    "Statement": [
                        {
                            "Effect": "Allow",
                            "Action": ["ecr:GetAuthorizationToken"],
                            "Resource": "*",
                        },
                        {
                            "Effect": "Allow",
                            "Action": [
                                "ecr:BatchCheckLayerAvailability",
                                "ecr:GetDownloadUrlForLayer",
                                "ecr:BatchGetImage",
                            ],
                            "Resource": "*",
                        },
                    ],
                }
            ),
        )

    @patch("boto3.client")
    def test_create_smtj_dataprep_execution_role_with_scoped_resources(self, mock_boto_client):
        account_id = "123456789012"
        mock_sts = MagicMock()
        mock_sts.get_caller_identity.return_value = {"Account": account_id}
        mock_boto_client.return_value = mock_sts

        role_name = "SmtjDataPrepRole"
        s3_resource = "my-data-bucket"
        ecr_resource = "my-dataprep-repo"
        mock_iam_client = MagicMock()
        mock_iam_client.exceptions.NoSuchEntityException = type(
            "NoSuchEntityException", (Exception,), {}
        )
        mock_iam_client.exceptions.EntityAlreadyExistsException = type(
            "EntityAlreadyExistsException", (Exception,), {}
        )
        mock_iam_client.get_role.side_effect = mock_iam_client.exceptions.NoSuchEntityException(
            "Role not found"
        )
        mock_iam_client.create_policy.return_value = {
            "Policy": {"Arn": f"arn:aws:iam::{account_id}:policy/foo"}
        }

        create_smtj_dataprep_execution_role(
            mock_iam_client,
            role_name,
            s3_resource=s3_resource,
            ecr_resource=ecr_resource,
        )

        # Verify scoped S3 policy
        mock_iam_client.create_policy.assert_any_call(
            PolicyName=f"{role_name}S3_Policy",
            PolicyDocument=json.dumps(
                {
                    "Version": "2012-10-17",
                    "Statement": [
                        {
                            "Effect": "Allow",
                            "Action": [
                                "s3:GetObject",
                                "s3:PutObject",
                                "s3:ListBucket",
                            ],
                            "Resource": [
                                f"arn:aws:s3:::{s3_resource}",
                                f"arn:aws:s3:::{s3_resource}/*",
                            ],
                        }
                    ],
                }
            ),
        )

        # Verify scoped ECR policy: GetAuthorizationToken stays Resource: *,
        # pull actions scoped to the specific repository
        mock_iam_client.create_policy.assert_any_call(
            PolicyName=f"{role_name}Ecr_Read_Policy",
            PolicyDocument=json.dumps(
                {
                    "Version": "2012-10-17",
                    "Statement": [
                        {
                            "Effect": "Allow",
                            "Action": ["ecr:GetAuthorizationToken"],
                            "Resource": "*",
                        },
                        {
                            "Effect": "Allow",
                            "Action": [
                                "ecr:BatchCheckLayerAvailability",
                                "ecr:GetDownloadUrlForLayer",
                                "ecr:BatchGetImage",
                            ],
                            "Resource": f"arn:aws:ecr:*:{account_id}:repository/{ecr_resource}",
                        },
                    ],
                }
            ),
        )


class TestCreateBedrockBatchInferenceRole(unittest.TestCase):
    ACCOUNT_ID = "123456789012"
    ROLE_NAME = "BedrockBatchInferenceRole"

    def _make_iam_client(self):
        mock_iam = MagicMock()
        mock_iam.exceptions.NoSuchEntityException = type("NoSuchEntityException", (Exception,), {})
        mock_iam.exceptions.EntityAlreadyExistsException = type(
            "EntityAlreadyExistsException", (Exception,), {}
        )
        mock_iam.create_policy.return_value = {
            "Policy": {"Arn": f"arn:aws:iam::{self.ACCOUNT_ID}:policy/foo"}
        }
        return mock_iam

    def _make_sts_mock(self, mock_boto_client):
        mock_sts = MagicMock()
        mock_sts.get_caller_identity.return_value = {"Account": self.ACCOUNT_ID}
        mock_boto_client.return_value = mock_sts

    @patch("boto3.client")
    def test_wildcard_s3_resource(self, mock_boto_client):
        """Default s3_resource='*' produces wildcard resource on both policies."""
        self._make_sts_mock(mock_boto_client)
        mock_iam = self._make_iam_client()
        mock_iam.get_role.side_effect = mock_iam.exceptions.NoSuchEntityException("Role not found")
        mock_iam.create_role.return_value = {
            "Role": {"Arn": f"arn:aws:iam::{self.ACCOUNT_ID}:role/{self.ROLE_NAME}"}
        }

        create_bedrock_batch_inference_execution_role(mock_iam, self.ROLE_NAME)

        mock_iam.create_policy.assert_any_call(
            PolicyName=f"{self.ROLE_NAME}Bedrock_Policy",
            PolicyDocument=json.dumps(
                {
                    "Version": "2012-10-17",
                    "Statement": [
                        {
                            "Effect": "Allow",
                            "Action": [
                                "bedrock:CreateModelInvocationJob",
                                "bedrock:GetModelInvocationJob",
                            ],
                            "Resource": "*",
                        }
                    ],
                }
            ),
        )

        mock_iam.create_policy.assert_any_call(
            PolicyName=f"{self.ROLE_NAME}S3_Read_Write_Policy",
            PolicyDocument=json.dumps(
                {
                    "Version": "2012-10-17",
                    "Statement": [
                        {
                            "Effect": "Allow",
                            "Action": ["s3:GetObject", "s3:ListBucket", "s3:PutObject"],
                            "Resource": "*",
                        }
                    ],
                }
            ),
        )

    @patch("boto3.client")
    def test_scoped_s3_resource(self, mock_boto_client):
        """Scoped s3_resource produces bucket ARN patterns."""
        self._make_sts_mock(mock_boto_client)
        mock_iam = self._make_iam_client()
        mock_iam.get_role.side_effect = mock_iam.exceptions.NoSuchEntityException("Role not found")
        mock_iam.create_role.return_value = {
            "Role": {"Arn": f"arn:aws:iam::{self.ACCOUNT_ID}:role/{self.ROLE_NAME}"}
        }

        create_bedrock_batch_inference_execution_role(
            mock_iam, self.ROLE_NAME, s3_resource="my-bucket"
        )

        mock_iam.create_policy.assert_any_call(
            PolicyName=f"{self.ROLE_NAME}S3_Read_Write_Policy",
            PolicyDocument=json.dumps(
                {
                    "Version": "2012-10-17",
                    "Statement": [
                        {
                            "Effect": "Allow",
                            "Action": ["s3:GetObject", "s3:ListBucket", "s3:PutObject"],
                            "Resource": [
                                "arn:aws:s3:::my-bucket",
                                "arn:aws:s3:::my-bucket/*",
                            ],
                        }
                    ],
                }
            ),
        )

    @patch("boto3.client")
    def test_role_already_exists_updates_policies(self, mock_boto_client):
        """When role exists, skips create_role and updates policies idempotently."""
        self._make_sts_mock(mock_boto_client)
        mock_iam = self._make_iam_client()
        mock_iam.get_role.return_value = {
            "Role": {"Arn": f"arn:aws:iam::{self.ACCOUNT_ID}:role/{self.ROLE_NAME}"}
        }

        create_bedrock_batch_inference_execution_role(mock_iam, self.ROLE_NAME)

        mock_iam.create_role.assert_not_called()
        self.assertEqual(mock_iam.create_policy.call_count, 2)

    @patch("boto3.client")
    def test_existing_policy_updates_and_reattaches(self, mock_boto_client):
        """When policy already exists, updates document and reattaches."""
        self._make_sts_mock(mock_boto_client)
        mock_iam = self._make_iam_client()
        mock_iam.get_role.side_effect = mock_iam.exceptions.NoSuchEntityException("Role not found")
        mock_iam.create_role.return_value = {
            "Role": {"Arn": f"arn:aws:iam::{self.ACCOUNT_ID}:role/{self.ROLE_NAME}"}
        }
        # First policy creates fine, second already exists
        mock_iam.create_policy.side_effect = [
            {"Policy": {"Arn": f"arn:aws:iam::{self.ACCOUNT_ID}:policy/bedrock"}},
            mock_iam.exceptions.EntityAlreadyExistsException("exists"),
        ]
        mock_iam.get_policy.return_value = {
            "Policy": {
                "Arn": f"arn:aws:iam::{self.ACCOUNT_ID}:policy/{self.ROLE_NAME}S3_Read_Write_Policy"
            }
        }
        mock_iam.list_policy_versions.return_value = {"Versions": []}

        create_bedrock_batch_inference_execution_role(mock_iam, self.ROLE_NAME)

        # Should have called attach_role_policy twice (once per policy)
        self.assertEqual(mock_iam.attach_role_policy.call_count, 2)
        # Should have called get_policy for the existing one
        mock_iam.get_policy.assert_called_once()

    @patch("boto3.client")
    def test_creates_exactly_two_policies(self, mock_boto_client):
        """Exactly bedrock_policy and s3_read_write_policy are created."""
        self._make_sts_mock(mock_boto_client)
        mock_iam = self._make_iam_client()
        mock_iam.get_role.side_effect = mock_iam.exceptions.NoSuchEntityException("Role not found")
        mock_iam.create_role.return_value = {
            "Role": {"Arn": f"arn:aws:iam::{self.ACCOUNT_ID}:role/{self.ROLE_NAME}"}
        }

        create_bedrock_batch_inference_execution_role(mock_iam, self.ROLE_NAME)

        created_policy_names = [
            call[1]["PolicyName"] for call in mock_iam.create_policy.call_args_list
        ]
        self.assertEqual(
            created_policy_names,
            [
                f"{self.ROLE_NAME}Bedrock_Policy",
                f"{self.ROLE_NAME}S3_Read_Write_Policy",
            ],
        )


class TestSagemakerModelPackagePolicy(unittest.TestCase):
    """Tests that sagemaker_model_package_policy is created and attached by both role creators."""

    @classmethod
    def setUpClass(cls):
        from importlib import resources as importlib_resources

        with (
            importlib_resources.files("amzn_nova_forge.iam")
            .joinpath("sagemaker_policies.json")
            .open() as f
        ):
            sagemaker_policies = json.load(f)
        with (
            importlib_resources.files("amzn_nova_forge.iam")
            .joinpath("bedrock_policies.json")
            .open() as f
        ):
            bedrock_policies = json.load(f)

        cls.sagemaker_model_package_policy = sagemaker_policies["sagemaker_model_package_policy"]
        cls.bedrock_model_package_policy = bedrock_policies["sagemaker_model_package_policy"]

    @patch("boto3.client")
    def test_bedrock_role_creates_sagemaker_model_package_policy(self, mock_boto_client):
        """create_bedrock_execution_role creates and attaches sagemaker_model_package_policy."""
        mock_sts = MagicMock()
        mock_sts.get_caller_identity.return_value = {"Account": "123456789012"}
        mock_boto_client.return_value = mock_sts

        role_name = "test-bedrock-role"
        mock_iam = MagicMock()
        mock_iam.exceptions.NoSuchEntityException = type("NoSuchEntityException", (Exception,), {})
        mock_iam.exceptions.EntityAlreadyExistsException = type(
            "EntityAlreadyExistsException", (Exception,), {}
        )
        mock_iam.get_role.side_effect = mock_iam.exceptions.NoSuchEntityException("not found")
        mock_iam.create_policy.return_value = {
            "Policy": {"Arn": "arn:aws:iam::123456789012:policy/foo"}
        }

        create_bedrock_execution_role(iam_client=mock_iam, role_name=role_name)

        mock_iam.create_policy.assert_any_call(
            PolicyName=f"{role_name}Sagemaker_Model_Package_Policy",
            PolicyDocument=json.dumps(self.bedrock_model_package_policy),
        )
        mock_iam.attach_role_policy.assert_any_call(
            RoleName=role_name,
            PolicyArn="arn:aws:iam::123456789012:policy/foo",
        )

    @patch("boto3.client")
    def test_sagemaker_role_creates_sagemaker_model_package_policy(self, mock_boto_client):
        """create_sagemaker_execution_role creates and attaches sagemaker_model_package_policy."""
        mock_sts = MagicMock()
        mock_sts.get_caller_identity.return_value = {"Account": "123456789012"}
        mock_boto_client.return_value = mock_sts

        role_name = "test-sagemaker-role"
        mock_iam = MagicMock()
        mock_iam.exceptions.NoSuchEntityException = type("NoSuchEntityException", (Exception,), {})
        mock_iam.exceptions.EntityAlreadyExistsException = type(
            "EntityAlreadyExistsException", (Exception,), {}
        )
        mock_iam.get_role.side_effect = mock_iam.exceptions.NoSuchEntityException("not found")
        mock_iam.create_policy.return_value = {
            "Policy": {"Arn": "arn:aws:iam::123456789012:policy/foo"}
        }

        create_sagemaker_execution_role(iam_client=mock_iam, role_name=role_name)

        mock_iam.create_policy.assert_any_call(
            PolicyName=f"{role_name}Sagemaker_Model_Package_Policy",
            PolicyDocument=json.dumps(self.sagemaker_model_package_policy),
        )
        mock_iam.attach_role_policy.assert_any_call(
            RoleName=role_name,
            PolicyArn="arn:aws:iam::123456789012:policy/foo",
        )

    @patch("boto3.client")
    def test_bedrock_role_handles_existing_model_package_policy(self, mock_boto_client):
        """create_bedrock_execution_role handles EntityAlreadyExistsException for sagemaker_model_package_policy."""
        account_id = "123456789012"
        mock_sts = MagicMock()
        mock_sts.get_caller_identity.return_value = {"Account": account_id}
        mock_boto_client.return_value = mock_sts

        role_name = "test-bedrock-role"
        mock_iam = MagicMock()
        mock_iam.exceptions.NoSuchEntityException = type("NoSuchEntityException", (Exception,), {})
        mock_iam.exceptions.EntityAlreadyExistsException = type(
            "EntityAlreadyExistsException", (Exception,), {}
        )
        mock_iam.get_role.side_effect = mock_iam.exceptions.NoSuchEntityException("not found")

        # First two policies succeed, sagemaker_model_package_policy already exists
        existing_policy_arn = (
            f"arn:aws:iam::{account_id}:policy/{role_name}Sagemaker_Model_Package_Policy"
        )
        mock_iam.create_policy.side_effect = [
            {"Policy": {"Arn": f"arn:aws:iam::{account_id}:policy/{role_name}Bedrock_Policy"}},
            {"Policy": {"Arn": f"arn:aws:iam::{account_id}:policy/{role_name}S3_Read_Policy"}},
            mock_iam.exceptions.EntityAlreadyExistsException("already exists"),
        ]
        mock_iam.get_policy.return_value = {"Policy": {"Arn": existing_policy_arn}}

        create_bedrock_execution_role(iam_client=mock_iam, role_name=role_name)

        mock_iam.get_policy.assert_called_with(PolicyArn=existing_policy_arn)
        mock_iam.attach_role_policy.assert_any_call(
            RoleName=role_name,
            PolicyArn=existing_policy_arn,
        )

    @patch("boto3.client")
    def test_sagemaker_role_handles_existing_model_package_policy(self, mock_boto_client):
        """create_sagemaker_execution_role handles EntityAlreadyExistsException for sagemaker_model_package_policy."""
        account_id = "123456789012"
        mock_sts = MagicMock()
        mock_sts.get_caller_identity.return_value = {"Account": account_id}
        mock_boto_client.return_value = mock_sts

        role_name = "test-sagemaker-role"
        mock_iam = MagicMock()
        mock_iam.exceptions.NoSuchEntityException = type("NoSuchEntityException", (Exception,), {})
        mock_iam.exceptions.EntityAlreadyExistsException = type(
            "EntityAlreadyExistsException", (Exception,), {}
        )
        mock_iam.get_role.side_effect = mock_iam.exceptions.NoSuchEntityException("not found")

        # First 8 policies succeed, sagemaker_model_package_policy (9th) already exists
        existing_policy_arn = (
            f"arn:aws:iam::{account_id}:policy/{role_name}Sagemaker_Model_Package_Policy"
        )
        successful_result = {"Policy": {"Arn": f"arn:aws:iam::{account_id}:policy/foo"}}
        mock_iam.create_policy.side_effect = [
            successful_result,  # cloudwatch_ec2_ec2_policy
            successful_result,  # cloudwatch_metric_policy
            successful_result,  # cloudwatch_logstream_policy
            successful_result,  # cloudwatch_loggroup_policy
            successful_result,  # ecr_read_policy
            successful_result,  # s3_read_policy
            successful_result,  # kms_policy
            successful_result,  # ec2_policy
            mock_iam.exceptions.EntityAlreadyExistsException("already exists"),
        ]
        mock_iam.get_policy.return_value = {"Policy": {"Arn": existing_policy_arn}}

        create_sagemaker_execution_role(iam_client=mock_iam, role_name=role_name)

        mock_iam.get_policy.assert_called_with(PolicyArn=existing_policy_arn)
        mock_iam.attach_role_policy.assert_any_call(
            RoleName=role_name,
            PolicyArn=existing_policy_arn,
        )


class TestRegionPropagation(unittest.TestCase):
    """Tests that the region parameter is propagated to the STS client."""

    @patch("boto3.client")
    def test_create_bedrock_execution_role_propagates_region(self, mock_boto_client):
        mock_sts = MagicMock()
        mock_sts.get_caller_identity.return_value = {"Account": "123456789012"}
        mock_boto_client.return_value = mock_sts

        mock_iam = MagicMock()
        mock_iam.exceptions.NoSuchEntityException = type("NoSuchEntityException", (Exception,), {})
        mock_iam.exceptions.EntityAlreadyExistsException = type(
            "EntityAlreadyExistsException", (Exception,), {}
        )
        mock_iam.get_role.side_effect = mock_iam.exceptions.NoSuchEntityException("not found")
        mock_iam.create_policy.return_value = {
            "Policy": {"Arn": "arn:aws:iam::123456789012:policy/foo"}
        }

        create_bedrock_execution_role(
            iam_client=mock_iam, role_name="test-role", region="eu-west-1"
        )

        mock_boto_client.assert_called_once_with("sts", region_name="eu-west-1")


if __name__ == "__main__":
    unittest.main()
