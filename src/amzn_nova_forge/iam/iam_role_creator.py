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
from importlib import resources
from typing import Any, Dict, Optional, Set

import boto3

from amzn_nova_forge.telemetry import Feature, _telemetry_emitter
from amzn_nova_forge.util.logging import logger


def _get_attached_policy_names(iam_client, role_name: str) -> Set[str]:
    """Return the set of policy names already attached to a role."""
    attached = iam_client.list_attached_role_policies(RoleName=role_name)
    return {p["PolicyName"].lower() for p in attached.get("AttachedPolicies", [])}


def _attach_policies(
    iam_client,
    account_id: str,
    role_name: str,
    policy_names: list,
    policies: Dict,
) -> None:
    """
    Create and attach a list of IAM policies to a role, idempotently.
    Skips policies already attached. Attaches without recreating if they already exist in the account.
    """
    already_attached = _get_attached_policy_names(iam_client, role_name)

    for policy_name in policy_names:
        managed_policy_name = f"{role_name}{policy_name.title()}"
        if managed_policy_name.lower() in already_attached:
            logger.info(f"Policy {policy_name} is already attached to {role_name}, skipping.")
            continue
        try:
            policy_arn = iam_client.create_policy(
                PolicyName=managed_policy_name,
                PolicyDocument=json.dumps(policies[policy_name]),
            )["Policy"]["Arn"]
            logger.info(
                f"Creating {policy_name} with the following permissions {json.dumps(policies[policy_name])}."
            )
            iam_client.attach_role_policy(RoleName=role_name, PolicyArn=policy_arn)
        except iam_client.exceptions.EntityAlreadyExistsException:
            # Policy exists in the account but isn't attached yet — attach it.
            logger.info(f"Policy {policy_name} already exists, attaching to {role_name}.")
            policy_arn = iam_client.get_policy(
                PolicyArn=f"arn:aws:iam::{account_id}:policy/{managed_policy_name}"
            )["Policy"]["Arn"]
            iam_client.attach_role_policy(RoleName=role_name, PolicyArn=policy_arn)
        except Exception as e:
            raise RuntimeError(f"Failed to create or attach policy {policy_name}: {str(e)}") from e


def _update_policy_document(iam_client, policy_arn: str, policy_document: str) -> None:
    """Update a managed IAM policy by creating a new version and removing the oldest non-default version if at the limit."""
    try:
        iam_client.create_policy_version(
            PolicyArn=policy_arn,
            PolicyDocument=policy_document,
            SetAsDefault=True,
        )
        logger.info("Updated policy %s with new document.", policy_arn)
    except iam_client.exceptions.LimitExceededException:
        # IAM allows max 5 versions; delete the oldest non-default version and retry.
        versions = iam_client.list_policy_versions(PolicyArn=policy_arn)["Versions"]
        non_default = [v for v in versions if not v["IsDefaultVersion"]]
        non_default.sort(key=lambda v: v["CreateDate"])
        if non_default:
            iam_client.delete_policy_version(
                PolicyArn=policy_arn,
                VersionId=non_default[0]["VersionId"],
            )
            iam_client.create_policy_version(
                PolicyArn=policy_arn,
                PolicyDocument=policy_document,
                SetAsDefault=True,
            )
            logger.info(
                "Updated policy %s (removed oldest version to make room).",
                policy_arn,
            )


@_telemetry_emitter(Feature.INFRA, "create_bedrock_execution_role")
def create_bedrock_execution_role(
    iam_client,
    role_name: str,
    bedrock_resource: str = "*",
    s3_resource: str = "*",
    region: Optional[str] = None,
) -> Dict:
    """
    Creates a new IAM Role that allows for Bedrock model creation and deployment.
    If the role already exists, ensures required policies are attached (idempotent).

    Args:
        iam_client: The boto3 client to use when creating the role.
        role_name: The name of the role to create.
        bedrock_resource: Optional name of the bedrock resources that IAM role should have
            restricted create and get access to. Defaults to "*" (all resources).
        s3_resource: Optional name of additional s3 resources that IAM role should have
            restricted read access to such as the training output bucket. Defaults to "*".

    Returns:
        Dict: The IAM role response containing role details

    Raises:
        Exception: If it fails at creating the new role.
    """
    sts_client = boto3.client("sts", region_name=region)

    with resources.files("amzn_nova_forge.iam").joinpath("bedrock_policies.json").open() as f:
        policies = json.load(f)

    # Create a new execution role for creating and deploying the models.
    try:
        # Checks if the role exists already.
        bedrock_execution_role = iam_client.get_role(RoleName=role_name)
    except iam_client.exceptions.NoSuchEntityException:
        logger.info(f"The {role_name} role doesn't exist. Creating it now...")
        bedrock_execution_role = iam_client.create_role(
            RoleName=role_name,
            AssumeRolePolicyDocument=json.dumps(policies["trust_policy"]),
            Description="This role allows for models to be created and deployed.",
        )
    except Exception as e:
        raise RuntimeError(
            f"Failed to create the Bedrock execution role {role_name}: {str(e)}"
        ) from e

    if bedrock_resource != "*":
        policies["bedrock_policy"]["Statement"][0]["Resource"] = (
            f"arn:aws:bedrock:*:*:custom-model/{bedrock_resource}*"
        )

    else:
        policies["bedrock_policy"]["Statement"][0]["Resource"] = "*"

    # S3 resources needed are the escrow bucket and the training output bucket
    account_id = sts_client.get_caller_identity()["Account"]
    if s3_resource != "*":
        policies["s3_read_policy"]["Statement"][0]["Resource"] = [
            f"arn:aws:s3:::{s3_resource}*",
            f"arn:aws:s3:::{s3_resource}*/*",
            f"arn:aws:s3:::customer-escrow-{account_id}*",
            f"arn:aws:s3:::customer-escrow-{account_id}*/*",
        ]
    else:
        policies["s3_read_policy"]["Statement"][0]["Resource"] = "*"

    # Create and attach policies
    for policy_name in ["bedrock_policy", "s3_read_policy", "sagemaker_model_package_policy"]:
        try:
            policy_arn = iam_client.create_policy(
                PolicyName=f"{role_name}{policy_name.title()}",
                PolicyDocument=json.dumps(policies[policy_name]),
            )["Policy"]["Arn"]

            logger.info(
                f"Creating {policy_name} with the following permissions {json.dumps(policies[policy_name])}."
            )

            iam_client.attach_role_policy(RoleName=role_name, PolicyArn=policy_arn)
        except iam_client.exceptions.EntityAlreadyExistsException:
            # If the policy already exists, get its ARN, update it, and attach it to the role.
            logger.info(
                f"The {policy_name} already exists in your account, updating and attaching it to the role now."
            )
            policy_arn = iam_client.get_policy(
                PolicyArn=f"arn:aws:iam::{account_id}:policy/{role_name}{policy_name.title()}"
            )["Policy"]["Arn"]

            _update_policy_document(iam_client, policy_arn, json.dumps(policies[policy_name]))
            iam_client.attach_role_policy(RoleName=role_name, PolicyArn=policy_arn)
        except Exception as e:
            raise RuntimeError(f"Failed to create or attach policy {policy_name}: {str(e)}") from e
    return bedrock_execution_role


@_telemetry_emitter(Feature.INFRA, "create_sagemaker_execution_role")
def create_sagemaker_execution_role(
    iam_client,
    role_name: str,
    s3_resource: str = "*",
    kms_resource: str = "*",
    ec2_condition: Optional[Dict[str, Any]] = None,
    cloudwatch_metric_condition: Optional[Dict[str, Any]] = None,
    cloudwatch_logstream_resource: str = "*",
    cloudwatch_loggroup_resource: str = "*",
    region: Optional[str] = None,
) -> Dict:
    """
    Creates a new IAM Role that allows for SageMaker model creation and deployment.
    If the role already exists, ensures required policies are attached (idempotent).

    Args:
        iam_client: The boto3 client to use when creating the role.
        role_name: The name of the role to create.
        s3_resource: Optional name of additional s3 resources that IAM role should have
            restricted read access to such as the training output bucket. Defaults to "*".
        kms_resource: Optional name of KMS resource that IAM role should have restricted access to.
        ec2_condition: Optional condition for the EC2 resource to limit access.
        cloudwatch_metric_condition: Optional condition for CloudWatch metric policy.
        cloudwatch_logstream_resource: Optional log stream resource name.
        cloudwatch_loggroup_resource: Optional log group resource name.

    Returns:
        Dict: The IAM role response containing role details

    Raises:
        Exception: If it fails at creating the new role.
    """
    sts_client = boto3.client("sts", region_name=region)

    with resources.files("amzn_nova_forge.iam").joinpath("sagemaker_policies.json").open() as f:
        policies = json.load(f)

    # Create a new execution role for creating and deploying the models.
    try:
        # Checks if the role exists already.
        sagemaker_execution_role = iam_client.get_role(RoleName=role_name)
    except iam_client.exceptions.NoSuchEntityException:
        logger.info(f"The {role_name} role doesn't exist. Creating it now...")
        sagemaker_execution_role = iam_client.create_role(
            RoleName=role_name,
            AssumeRolePolicyDocument=json.dumps(policies["trust_policy"]),
            Description="This role allows for models to be created and deployed to SageMaker.",
        )
    except Exception as e:
        raise RuntimeError(
            f"Failed to create the SageMaker execution role {role_name}: {str(e)}"
        ) from e

    # Allow KMS access
    policies["kms_policy"]["Statement"][0]["Resource"] = (
        f"arn:aws:kms:*:*:key/{kms_resource}*" if kms_resource != "*" else "*"
    )

    # Add condition to EC2 policy
    if ec2_condition is not None:
        policies["ec2_policy"]["Statement"][0]["Condition"] = ec2_condition

    if cloudwatch_metric_condition is not None:
        policies["cloudwatch_metric_policy"]["Statement"][0]["Condition"] = (
            cloudwatch_metric_condition
        )

    policies["cloudwatch_logstream_policy"]["Statement"][0]["Resource"] = (
        f"arn:aws:logs:*:*:log-group:{cloudwatch_loggroup_resource}:log-stream:{cloudwatch_logstream_resource}"
        if cloudwatch_logstream_resource != "*"
        else "*"
    )

    policies["cloudwatch_loggroup_policy"]["Statement"][0]["Resource"] = (
        f"arn:aws:logs:*:*:log-group:{cloudwatch_loggroup_resource}*"
        if cloudwatch_loggroup_resource != "*"
        else "*"
    )

    # S3 resources needed are the escrow bucket and the training output bucket
    account_id = sts_client.get_caller_identity()["Account"]
    if s3_resource != "*":
        policies["s3_read_policy"]["Statement"][0]["Resource"] = [
            f"arn:aws:s3:::{s3_resource}*",
            f"arn:aws:s3:::{s3_resource}*/*",
            f"arn:aws:s3:::customer-escrow-{account_id}*",
            f"arn:aws:s3:::customer-escrow-{account_id}*/*",
        ]
    else:
        policies["s3_read_policy"]["Statement"][0]["Resource"] = "*"

    # Create and attach policies
    for policy_name in [
        "cloudwatch_ec2_ec2_policy",
        "cloudwatch_metric_policy",
        "cloudwatch_logstream_policy",
        "cloudwatch_loggroup_policy",
        "ecr_read_policy",
        "s3_read_policy",
        "kms_policy",
        "ec2_policy",
        "sagemaker_model_package_policy",
    ]:
        try:
            logger.info(f"{json.dumps(policies[policy_name])}.")

            policy_arn = iam_client.create_policy(
                PolicyName=f"{role_name}{policy_name.title()}",
                PolicyDocument=json.dumps(policies[policy_name]),
            )["Policy"]["Arn"]

            logger.info(
                f"Creating {policy_name} with the following permissions {json.dumps(policies[policy_name])}."
            )

            iam_client.attach_role_policy(RoleName=role_name, PolicyArn=policy_arn)
        except iam_client.exceptions.EntityAlreadyExistsException:
            # If the policy already exists, get its ARN, update it, and attach it to the role.
            logger.info(
                f"The {policy_name} already exists in your account, updating and attaching it to the role now."
            )
            policy_arn = iam_client.get_policy(
                PolicyArn=f"arn:aws:iam::{account_id}:policy/{role_name}{policy_name.title()}"
            )["Policy"]["Arn"]

            _update_policy_document(iam_client, policy_arn, json.dumps(policies[policy_name]))
            iam_client.attach_role_policy(RoleName=role_name, PolicyArn=policy_arn)
        except Exception as e:
            raise RuntimeError(f"Failed to create or attach policy {policy_name}: {str(e)}") from e
    return sagemaker_execution_role


@_telemetry_emitter(Feature.INFRA, "create_smtj_dataprep_execution_role")
def create_smtj_dataprep_execution_role(
    iam_client,
    role_name: str,
    s3_resource: str = "*",
    ecr_resource: str = "*",
    region: Optional[str] = None,
) -> Dict:
    """Creates a new IAM Role for SageMaker Training Job data preparation pipelines.

    The trust policy allows ``sagemaker.amazonaws.com`` to assume this role.

    Attached policies:

    - **s3_policy** — S3 read/write for input data, output data, and artifact bucket
    - **ecr_read_policy** — ECR pull access for the data prep Docker image
    - **cloudwatch_logs_policy** — CloudWatch Logs for ``/aws/sagemaker/TrainingJobs``
    - **cloudwatch_metric_policy** — CloudWatch PutMetricData
    - **ec2_policy** — EC2 network interface management for VPC mode

    Args:
        iam_client: The boto3 client to use when creating the role.
        role_name: The name of the role to create.
        s3_resource: S3 bucket name to scope read/write access.
            Use ``"*"`` (default) for all buckets.
        ecr_resource: ECR repository name to scope image pull access.
            Use ``"*"`` (default) for all repositories.

    Returns:
        Dict: The IAM role response containing role details.

    Raises:
        Exception: If it fails at creating the new role.
    """
    sts_client = boto3.client("sts", region_name=region)
    with resources.files("amzn_nova_forge.iam").joinpath("smtj_dataprep_policies.json").open() as f:
        policies = json.load(f)

    # Create or retrieve the role
    try:
        smtj_dataprep_role = iam_client.get_role(RoleName=role_name)
    except iam_client.exceptions.NoSuchEntityException:
        logger.info(f"The {role_name} role doesn't exist. Creating it now...")
        smtj_dataprep_role = iam_client.create_role(
            RoleName=role_name,
            AssumeRolePolicyDocument=json.dumps(policies["trust_policy"]),
            Description="This role allows SageMaker Training Jobs to run data preparation pipelines.",
        )
    except Exception as e:
        raise RuntimeError(
            f"Failed to create the SMTJ data prep role '{role_name}': {str(e)}. "
            "Your IAM identity needs iam:CreateRole and iam:CreatePolicy permissions, "
            "or you can create the role manually and pass it via the execution_role_name parameter."
        ) from e

    # Scope S3 resource
    if s3_resource != "*":
        policies["s3_policy"]["Statement"][0]["Resource"] = [
            f"arn:aws:s3:::{s3_resource}",
            f"arn:aws:s3:::{s3_resource}/*",
        ]
    else:
        policies["s3_policy"]["Statement"][0]["Resource"] = "*"

    # Scope ECR resource
    if ecr_resource != "*":
        account_id = sts_client.get_caller_identity()["Account"]
        policies["ecr_read_policy"]["Statement"][1]["Resource"] = (
            f"arn:aws:ecr:*:{account_id}:repository/{ecr_resource}"
        )
    else:
        policies["ecr_read_policy"]["Statement"][1]["Resource"] = "*"

    # Create and attach all policies
    policy_names = [
        "s3_policy",
        "ecr_read_policy",
        "cloudwatch_logs_policy",
        "cloudwatch_metric_policy",
        "ec2_policy",
    ]
    for policy_name in policy_names:
        try:
            policy_arn = iam_client.create_policy(
                PolicyName=f"{role_name}{policy_name.title()}",
                PolicyDocument=json.dumps(policies[policy_name]),
            )["Policy"]["Arn"]

            logger.info(
                f"Creating {policy_name} with the following permissions {json.dumps(policies[policy_name])}."
            )

            iam_client.attach_role_policy(RoleName=role_name, PolicyArn=policy_arn)
        except iam_client.exceptions.EntityAlreadyExistsException:
            logger.info(
                f"The {policy_name} already exists in your account, updating and attaching it to the role now."
            )
            policy_arn = iam_client.get_policy(
                PolicyArn=f"arn:aws:iam::{sts_client.get_caller_identity()['Account']}:policy/{role_name}{policy_name.title()}"
            )["Policy"]["Arn"]

            _update_policy_document(iam_client, policy_arn, json.dumps(policies[policy_name]))
            iam_client.attach_role_policy(RoleName=role_name, PolicyArn=policy_arn)
        except Exception as e:
            raise RuntimeError(f"Failed to create or attach policy {policy_name}: {str(e)}") from e
    return smtj_dataprep_role


@_telemetry_emitter(Feature.INFRA, "create_sagemaker_invoke_role")
def create_sagemaker_invoke_role(
    iam_client,
    role_name: str,
    s3_resource: str = "*",
    glue_job_resource: str = "*",
    trust_principal: Optional[Dict[str, Any]] = None,
    region: Optional[str] = None,
) -> Dict:
    """
    Creates a new IAM Role for AWS Glue data preparation jobs.

    The trust policy allows ``glue.amazonaws.com`` to assume this role.

    Attached policies:

    - **glue_base_policy** — GetJob, GetJobRun, StartJobRun, BatchStopJobRun
    - **glue_s3_policy** — S3 read/write for Glue job scripts and data
    - **glue_logs_policy** — CloudWatch Logs for ``/aws-glue/*`` log groups

    Args:
        iam_client: The boto3 client to use when creating the role.
        role_name: The name of the role to create.
        s3_resource: S3 bucket name to scope read/write access for Glue
            job data.  Use ``"*"`` (default) for all buckets.
        glue_job_resource: Glue job name to scope access.  Use ``"*"``
            (default) for all jobs.
        trust_principal: Optional override for the trust policy principal.
            Defaults to ``{"Service": "glue.amazonaws.com"}``.

    Returns:
        Dict: The IAM role response containing role details.

    Raises:
        Exception: If it fails at creating the new role.
    """
    sts_client = boto3.client("sts", region_name=region)
    with resources.files("amzn_nova_forge.iam").joinpath("glue_policies.json").open() as f:
        policies = json.load(f)

    # Allow caller to override the trust principal
    if trust_principal is not None:
        policies["trust_policy"]["Statement"][0]["Principal"] = trust_principal

    # Create or retrieve the role
    try:
        sagemaker_invoke_role = iam_client.get_role(RoleName=role_name)
    except iam_client.exceptions.NoSuchEntityException:
        logger.info(f"The {role_name} role doesn't exist. Creating it now...")
        sagemaker_invoke_role = iam_client.create_role(
            RoleName=role_name,
            AssumeRolePolicyDocument=json.dumps(policies["trust_policy"]),
            Description="This role allows Glue jobs to run data preparation pipelines.",
        )
    except Exception as e:
        raise RuntimeError(
            f"Failed to create the Glue data prep role '{role_name}': {str(e)}. "
            "Your IAM identity needs iam:CreateRole and iam:CreatePolicy permissions, "
            "or you can create the role manually and pass it via the glue_role_name parameter."
        ) from e

    account_id = sts_client.get_caller_identity()["Account"]

    # Scope Glue job resource
    if glue_job_resource != "*":
        policies["glue_base_policy"]["Statement"][0]["Resource"] = [
            f"arn:aws:glue:*:{account_id}:job/{glue_job_resource}",
        ]
    else:
        policies["glue_base_policy"]["Statement"][0]["Resource"] = "*"

    # Scope S3 resource for Glue data access
    if s3_resource != "*":
        policies["glue_s3_policy"]["Statement"][0]["Resource"] = [
            f"arn:aws:s3:::{s3_resource}",
            f"arn:aws:s3:::{s3_resource}/*",
        ]
    else:
        policies["glue_s3_policy"]["Statement"][0]["Resource"] = "*"

    # Create and attach all policies
    policy_names = [
        "glue_base_policy",
        "glue_s3_policy",
        "glue_logs_policy",
    ]
    for policy_name in policy_names:
        try:
            policy_arn = iam_client.create_policy(
                PolicyName=f"{role_name}{policy_name.title()}",
                PolicyDocument=json.dumps(policies[policy_name]),
            )["Policy"]["Arn"]

            logger.info(
                f"Creating {policy_name} with the following permissions {json.dumps(policies[policy_name])}."
            )

            iam_client.attach_role_policy(RoleName=role_name, PolicyArn=policy_arn)
        except iam_client.exceptions.EntityAlreadyExistsException:
            logger.info(
                f"The {policy_name} already exists in your account, updating and attaching it to the role now."
            )
            policy_arn = iam_client.get_policy(
                PolicyArn=f"arn:aws:iam::{account_id}:policy/{role_name}{policy_name.title()}"
            )["Policy"]["Arn"]

            _update_policy_document(iam_client, policy_arn, json.dumps(policies[policy_name]))
            iam_client.attach_role_policy(RoleName=role_name, PolicyArn=policy_arn)
        except Exception as e:
            raise RuntimeError(f"Failed to create or attach policy {policy_name}: {str(e)}") from e
    return sagemaker_invoke_role


@_telemetry_emitter(Feature.INFRA, "create_bedrock_batch_inference_execution_role")
def create_bedrock_batch_inference_execution_role(
    iam_client,
    role_name: str,
    s3_resource: str = "*",
    region: Optional[str] = None,
) -> Dict:
    """
    Creates a new IAM Role for Bedrock batch inference operations.
    If the role already exists, ensures required policies are attached (idempotent).

    Args:
        iam_client: The boto3 client to use when creating the role.
        role_name: The name of the role to create.
        s3_resource: Optional name of additional s3 resources that IAM role should have
            restricted read/write access to. Defaults to "*" (all resources).

    Returns:
        Dict: The IAM role response containing role details

    Raises:
        Exception: If it fails at creating the new role.
    """
    sts_client = boto3.client("sts", region_name=region)

    with (
        resources.files("amzn_nova_forge.iam")
        .joinpath("bedrock_batch_inference_policies.json")
        .open() as f
    ):
        policies = json.load(f)

    # Create or retrieve the role
    try:
        bedrock_execution_role = iam_client.get_role(RoleName=role_name)
    except iam_client.exceptions.NoSuchEntityException:
        logger.info(f"The {role_name} role doesn't exist. Creating it now...")
        bedrock_execution_role = iam_client.create_role(
            RoleName=role_name,
            AssumeRolePolicyDocument=json.dumps(policies["trust_policy"]),
            Description="This role allows for Bedrock batch inference operations.",
        )
    except Exception as e:
        raise RuntimeError(
            f"Failed to create the Bedrock batch inference execution role '{role_name}': {str(e)}. "
            "Your IAM identity needs iam:CreateRole and iam:CreatePolicy permissions, "
            "or you can create the role manually and pass it via the execution_role_name parameter."
        ) from e

    # Batch inference uses wildcard resource scope
    policies["bedrock_policy"]["Statement"][0]["Resource"] = "*"

    # Scope S3 resources to the user-supplied bucket
    account_id = sts_client.get_caller_identity()["Account"]
    if s3_resource != "*":
        policies["s3_read_write_policy"]["Statement"][0]["Resource"] = [
            f"arn:aws:s3:::{s3_resource}",
            f"arn:aws:s3:::{s3_resource}/*",
        ]
    else:
        policies["s3_read_write_policy"]["Statement"][0]["Resource"] = "*"

    # Create and attach policies
    for policy_name in ["bedrock_policy", "s3_read_write_policy"]:
        try:
            policy_arn = iam_client.create_policy(
                PolicyName=f"{role_name}{policy_name.title()}",
                PolicyDocument=json.dumps(policies[policy_name]),
            )["Policy"]["Arn"]

            logger.info(
                f"Creating {policy_name} with the following permissions {json.dumps(policies[policy_name])}."
            )

            iam_client.attach_role_policy(RoleName=role_name, PolicyArn=policy_arn)
        except iam_client.exceptions.EntityAlreadyExistsException:
            logger.info(
                f"The {policy_name} already exists in your account, updating and attaching it to the role now."
            )
            policy_arn = iam_client.get_policy(
                PolicyArn=f"arn:aws:iam::{account_id}:policy/{role_name}{policy_name.title()}"
            )["Policy"]["Arn"]

            _update_policy_document(iam_client, policy_arn, json.dumps(policies[policy_name]))
            iam_client.attach_role_policy(RoleName=role_name, PolicyArn=policy_arn)
        except Exception as e:
            raise Exception(f"Failed to create or attach policy {policy_name}: {str(e)}")
    return bedrock_execution_role
