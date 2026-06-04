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
"""
Helper functions for Sagemaker management.
"""

import json
import re
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

import boto3
from botocore.client import BaseClient
from botocore.exceptions import ClientError

from amzn_nova_forge.core.constants import (
    ESCROW_URI_TAG_KEY,
    REGION_TO_ESCROW_ACCOUNT_MAPPING,
    SUPPORTED_SMI_CONFIGS,
    _escrow_tag_value,
)
from amzn_nova_forge.core.enums import DeploymentMode, DeployPlatform, Model, Platform
from amzn_nova_forge.core.result.inference_result import (
    SingleInferenceResult,
)
from amzn_nova_forge.core.runtime import RuntimeManager
from amzn_nova_forge.core.types import (
    _IC_MIN_COMPUTE_REQUIREMENTS,
    DeploymentResult,
    EndpointInfo,
    InferenceComponentConfig,
    ModelArtifacts,
    validate_inference_component_resources,
)
from amzn_nova_forge.telemetry.constants import Feature
from amzn_nova_forge.telemetry.telemetry_logging import _telemetry_emitter
from amzn_nova_forge.validation.endpoint_validator import (
    validate_s3_uri_prefix,
)

from .logging import logger

SAGEMAKER_EXECUTION_ROLE_NAME = "SageMakerDeployModelExecutionRole"


def register_lambda_as_hub_content(
    lambda_arn: str,
    hub_name: str,
    sagemaker_client: Any,
    evaluator_name: Optional[str] = None,
) -> str:
    """Register a Lambda ARN as a JsonDoc hub-content and return the hub-content ARN.

    The serverless API's EvaluatorArn field only accepts hub-content ARNs, not Lambda ARNs
    directly. This wraps the Lambda ARN in a JsonDoc document inside a private hub,
    creating the hub if it doesn't exist.

    The hub-content is upserted — if a document with the same name already exists at the
    same version, the existing ARN is returned so repeated train() calls are idempotent.

    Args:
        lambda_arn: A valid Lambda function ARN.
        hub_name: Name of the private hub to register into.
        sagemaker_client: Boto3 SageMaker client.
        evaluator_name: Optional human-readable name for the hub-content entry.
            Defaults to the Lambda function name derived from the ARN.

    Returns:
        The hub-content ARN that can be passed as EvaluatorArn.
    """
    # Use provided name or derive from Lambda function name
    if evaluator_name:
        content_name = re.sub(r"[^a-zA-Z0-9-]", "-", evaluator_name)[:63]
    else:
        content_name = re.sub(r"[^a-zA-Z0-9-]", "-", lambda_arn.split(":")[-1])[:63]
    content_version = "0.0.1"
    document = json.dumps(
        {
            "SubType": "AWS/Evaluator",
            "JsonContent": json.dumps(
                {
                    "EvaluatorType": "RewardFunction",
                    "Reference": lambda_arn,
                }
            ),
        }
    )

    # Ensure the hub exists
    try:
        sagemaker_client.describe_hub(HubName=hub_name)
    except sagemaker_client.exceptions.ResourceNotFound:
        logger.info(f"Creating private hub '{hub_name}' for reward function registration.")
        try:
            sagemaker_client.create_hub(
                HubName=hub_name,
                HubDescription="Private hub for Nova Forge serverless reward functions",
            )
        except sagemaker_client.exceptions.ResourceInUse:
            logger.info(f"Hub '{hub_name}' was created concurrently; proceeding.")

    # Upsert the JsonDoc hub-content
    try:
        resp = sagemaker_client.import_hub_content(
            HubName=hub_name,
            HubContentName=content_name,
            HubContentType="JsonDoc",
            HubContentVersion=content_version,
            DocumentSchemaVersion="2.0.0",
            HubContentDocument=document,
        )
        hub_content_arn = resp["HubContentArn"]
        logger.info(f"Registered Lambda as hub-content: {hub_content_arn}")
    except sagemaker_client.exceptions.ResourceInUse:
        # Version already exists — check if it still points to the same Lambda ARN.
        # If the user updated their Lambda to a different ARN, register a new version.
        # Retry up to 10 times, bumping the patch version on each ResourceInUse.
        major, minor, patch = content_version.split(".")
        hub_content_arn = None
        for attempt in range(10):
            bump_version = f"{major}.{minor}.{int(patch) + attempt}"
            existing = sagemaker_client.describe_hub_content(
                HubName=hub_name,
                HubContentName=content_name,
                HubContentType="JsonDoc",
                HubContentVersion=bump_version,
            )
            existing_doc = json.loads(existing["HubContentDocument"])
            existing_ref = json.loads(existing_doc.get("JsonContent", "{}")).get("Reference")

            if existing_ref == lambda_arn:
                hub_content_arn = existing["HubContentArn"]
                logger.info(f"Reusing existing hub-content: {hub_content_arn}")
                break

            # Lambda ARN changed — try the next version
            next_version = f"{major}.{minor}.{int(patch) + attempt + 1}"
            logger.info(f"Lambda ARN changed (was {existing_ref}), trying version {next_version}.")
            try:
                resp = sagemaker_client.import_hub_content(
                    HubName=hub_name,
                    HubContentName=content_name,
                    HubContentType="JsonDoc",
                    HubContentVersion=next_version,
                    DocumentSchemaVersion="2.0.0",
                    HubContentDocument=document,
                )
                hub_content_arn = resp["HubContentArn"]
                logger.info(f"Registered updated Lambda as hub-content: {hub_content_arn}")
                break
            except sagemaker_client.exceptions.ResourceInUse:
                # Another version already exists — keep bumping
                continue

        if hub_content_arn is None:
            raise RuntimeError(
                f"Could not register Lambda ARN as hub-content after 10 retries "
                f"(all versions 0.0.1–0.0.{int(patch) + 10} are in use)."
            )

    return hub_content_arn


def extract_lambda_arn_from_hub_content(
    hub_content_arn: str,
    sagemaker_client: Any,
) -> Optional[str]:
    """Extract the Lambda ARN stored inside a JsonDoc hub-content evaluator.

    Args:
        hub_content_arn: A SageMaker hub-content ARN.
        sagemaker_client: Boto3 SageMaker client.

    Returns:
        The Lambda ARN if found, or None if extraction fails.
    """
    try:
        # ARN: arn:aws:sagemaker:region:account:hub-content/hub/type/name/version
        resource = hub_content_arn.split(":", 5)[5]  # hub-content/hub/type/name/version
        _, hub_name, _, content_name, content_version = resource.split("/")
        resp = sagemaker_client.describe_hub_content(
            HubName=hub_name,
            HubContentName=content_name,
            HubContentType="JsonDoc",
            HubContentVersion=content_version,
        )
        doc = json.loads(resp["HubContentDocument"])
        inner = json.loads(doc.get("JsonContent", "{}"))
        return inner.get("Reference")
    except Exception as e:
        logger.warning(f"Could not extract Lambda ARN from hub-content '{hub_content_arn}': {e}")
        return None


def _get_sagemaker_inference_image(region: str) -> str:
    if region not in REGION_TO_ESCROW_ACCOUNT_MAPPING:
        raise ValueError(
            f"Unsupported region: {region}. Supported regions are: {list(REGION_TO_ESCROW_ACCOUNT_MAPPING.keys())}"
        )

    return f"{REGION_TO_ESCROW_ACCOUNT_MAPPING[region]}.dkr.ecr.{region}.amazonaws.com/nova-inference-repo:SM-Inference-latest"


@_telemetry_emitter(Feature.TRAINING, "get_model_artifacts")
def get_model_artifacts(
    job_name: str,
    infra: RuntimeManager,
    output_s3_path: Optional[str] = None,
    region: Optional[str] = None,
) -> ModelArtifacts:
    """
    Retrieve model artifacts for a job

    Args:
        job_name: Name of the job
        infra: Infrastructure of the job
        output_s3_path: Output S3 path of the job (required for HyperPod)

    Returns:
        ModelArtifacts: Model artifact S3 paths

    Raises:
        Exception: If unable to obtain job artifact information
    """
    # Use the infra's sagemaker client if available — it may be configured with a custom
    # endpoint (e.g. gamma) that the job was submitted to.
    sagemaker_client = getattr(infra, "sagemaker_client", None) or boto3.client(
        "sagemaker", region_name=region
    )

    if infra.platform in (Platform.SMTJ, Platform.SMTJServerless):
        try:
            response = sagemaker_client.describe_training_job(TrainingJobName=job_name)
        except ClientError as e:
            error_code = e.response["Error"]["Code"]
            if error_code not in ("ValidationException", "ResourceNotFound"):
                raise
            if infra.platform != Platform.SMTJServerless:
                raise
            # MTRL jobs are AgentRFT jobs, not standard SageMaker training jobs
            from sagemaker.train.agent_rft_job import AgentRFTJob

            session = boto3.Session(region_name=region)
            rft_job = AgentRFTJob.get(job_name, session=session)
            return ModelArtifacts(
                checkpoint_s3_path=None,
                output_s3_path=rft_job.s3_output_path,
                output_model_arn=rft_job.output_model_package_arn,
            )
        # Serverless jobs populate OutputModelPackageArn; use it to get the checkpoint S3 URI
        # SMTJ jobs use CheckpointConfig.S3Uri
        checkpoint_s3_path = None
        model_package_arn = response.get("OutputModelPackageArn")
        if model_package_arn:
            # For serverless, get the S3 checkpoint URI from the model package directly
            try:
                pkg = sagemaker_client.describe_model_package(ModelPackageName=model_package_arn)
                checkpoint_s3_path = (
                    pkg.get("InferenceSpecification", {})
                    .get("Containers", [{}])[0]
                    .get("ModelDataSource", {})
                    .get("S3DataSource", {})
                    .get("S3Uri")
                )
            except Exception as e:
                logger.warning(
                    "Failed to extract checkpoint path for serverless job '%s': %s",
                    job_name,
                    e,
                )
        if (
            not checkpoint_s3_path
            and "CheckpointConfig" in response
            and response["CheckpointConfig"]
        ):
            checkpoint_s3_path = response["CheckpointConfig"]["S3Uri"]
        return ModelArtifacts(
            checkpoint_s3_path=checkpoint_s3_path,
            output_s3_path=response["OutputDataConfig"]["S3OutputPath"],
            output_model_arn=model_package_arn,
        )
    elif infra.platform == Platform.SMHP:
        if not output_s3_path:
            raise ValueError("output_s3_path is required for HyperPod jobs")
        try:
            cluster_name = infra.cluster_name  # type: ignore[attr-defined]
        except AttributeError:
            raise ValueError("SMHPRuntimeManager requires cluster_name for get_model_artifacts")
        response = sagemaker_client.describe_cluster(ClusterName=cluster_name)
        rigs = response.get("RestrictedInstanceGroups", [])

        # If there's only one RIG in the cluster, we know that the job had to be submitted to that RIG
        checkpoint_s3_path = None
        if len(rigs) == 1:
            checkpoint_s3_path = rigs[0].get("EnvironmentConfig", {}).get("S3OutputPath")

        return ModelArtifacts(
            checkpoint_s3_path=checkpoint_s3_path,
            output_s3_path=output_s3_path,
        )
    else:
        raise ValueError(f"Unsupported platform: {infra.platform}")


def get_cluster_instance_info(
    cluster_name: str, region: Optional[str] = None
) -> Dict[str, List[Dict[str, Any]]]:
    """
    Get instance types and counts from a HyperPod cluster.

    Args:
        cluster_name: Name of the HyperPod cluster
        region: AWS region (optional, uses default session region if not provided)

    Returns:
        Dict with 'normal_instance_groups' and 'restricted_instance_groups' keys

    Raises:
        Exception: If unable to describe the cluster
    """
    sagemaker_client = boto3.client("sagemaker", region_name=region)

    try:
        response = sagemaker_client.describe_cluster(ClusterName=cluster_name)

        normal_instance_groups = []
        restricted_instance_groups = []

        # Process normal instance groups
        for group in response.get("InstanceGroups", []):
            group_info = {
                "instance_group_name": group["InstanceGroupName"],
                "instance_type": group["InstanceType"],
                "current_count": group["CurrentCount"],
                "target_count": group["TargetCount"],
                "status": group["Status"],
            }
            normal_instance_groups.append(group_info)

        # Process restricted instance groups
        for group in response.get("RestrictedInstanceGroups", []):
            group_info = {
                "instance_group_name": group["InstanceGroupName"],
                "instance_type": group["InstanceType"],
                "current_count": group["CurrentCount"],
                "target_count": group["TargetCount"],
                "status": group["Status"],
            }
            restricted_instance_groups.append(group_info)

        return {
            "normal_instance_groups": normal_instance_groups,
            "restricted_instance_groups": restricted_instance_groups,
        }

    except Exception as e:
        raise RuntimeError(f"Failed to get cluster instance info for {cluster_name}: {str(e)}")


def _monitor_endpoint_creation(sagemaker_client: BaseClient, endpoint_name: str) -> str:
    """
    Monitors the status of a custom endpoint creation in SageMaker.

    Args:
        sagemaker_client: The boto3 sagemaker client used in the script
        endpoint_name: The name of the model endpoint.

    Returns:
        str: Final status of the model ('INSERVICE' or raises exception)
    """
    start_time = datetime.now(timezone.utc)

    while True:
        try:
            response = sagemaker_client.describe_endpoint(EndpointName=endpoint_name)
            status = response["EndpointStatus"]

            elapsed_time = datetime.now(timezone.utc) - start_time

            logger.info(f"Status: {status} | Elapsed: {elapsed_time}")

            if status.upper() == "INSERVICE":
                logger.info(
                    f"\n\nSUCCESS! Endpoint creation is complete! '{endpoint_name}' is now INSERVICE!"
                )
                logger.info(f"Total time elapsed: {elapsed_time}")
                return status
            elif status.upper() in ["FAILED"]:
                error_msg = f"\n\nERROR! Endpoint '{endpoint_name}' status is: {status}\n"
                logger.error(f"{error_msg}\nPlease check the AWS console for more details.\n")
                raise Exception(error_msg)
        except Exception as e:
            logger.error(f"Error checking status: {str(e)}\n")
            raise
        time.sleep(60)  # Sleep for a minute.


def _validate_sagemaker_instance_type_for_model_deployment(
    instance_type: str,
    model: Model,
    context_length: Optional[str] = None,
    max_concurrency: Optional[str] = None,
) -> None:
    """
    Validation method that checks the instance_type and if it is compatible with the desired model.
    Validates CONTEXT_LENGTH and MAX_CONCURRENCY against supported SMI configurations when both values are provided.

    Args:
        instance_type: instance type
        model: Model (enum)
        context_length: Optional CONTEXT_LENGTH value to validate
        max_concurrency: Optional MAX_CONCURRENCY value to validate

    Raises:
        ValueError: If validation fails

    """
    # Check if the model and instance combination is supported
    config_key = (model, instance_type)
    if config_key not in SUPPORTED_SMI_CONFIGS:
        # Collect all supported instance types for this model for error message
        supported_instances = [inst for (m, inst) in SUPPORTED_SMI_CONFIGS.keys() if m == model]
        if not supported_instances:
            raise ValueError(
                f"No supported instance types found for {model}. "
                f"Please check SUPPORTED_SMI_CONFIGS in constants.py"
            )
        raise ValueError(
            f"{instance_type} is not in the supported instances list for {model}: "
            f"{sorted(supported_instances)}"
        )

    # If context_length and max_concurrency are provided, validate SMI config bounds
    if context_length is not None and max_concurrency is not None:
        tiers = SUPPORTED_SMI_CONFIGS[config_key]
        context_length_val = int(context_length)
        max_concurrency_val = int(max_concurrency)

        for tier_context, tier_concurrency in tiers:
            if context_length_val <= tier_context and max_concurrency_val <= tier_concurrency:
                return

        # If no tier matches, raise an error with available options
        raise ValueError(
            f"CONTEXT_LENGTH={context_length} and MAX_CONCURRENCY={max_concurrency} "
            f"is not a supported configuration for {model.name} on {instance_type}. "
            f"Available tiers (max_context_length, max_concurrency): {tiers}"
        )


def create_sagemaker_model(
    region: str,
    model_name: str,
    sagemaker_execution_role_arn: str,
    sagemaker_client: BaseClient,
    model_s3_location: Optional[str] = None,
    environment: Dict[str, Any] = {},  # noqa: B006 - never mutated
    network_isolation: bool = True,
    deployment_mode: DeploymentMode = DeploymentMode.FAIL_IF_EXISTS,
    tags: Optional[List[Dict[str, str]]] = None,
    model_package_name: Optional[str] = None,
) -> str:
    """Create a SageMaker model resource.

    Supports two mutually exclusive model source modes:
    1. S3 data source: provide ``model_s3_location``
    2. Model package: provide ``model_package_name`` (name or ARN of a SageMaker Model Package).

    Args:
        region: AWS region
        model_name: Name of the SageMaker model
        model_s3_location: S3 URI where model artifacts are stored.
            Required when model_package_name is not provided.
        sagemaker_execution_role_arn: IAM role ARN for SageMaker execution
        sagemaker_client: SageMaker client
        environment: Environment variables for the model
        network_isolation: Enable network isolation
        deployment_mode: How to handle existing model
        tags: Optional resource tags
        model_package_name: Name or ARN of a SageMaker Model Package to use as the model source.

    Returns:
        str: Model ARN

    Raises:
        ValueError: If neither model_s3_location nor model_package_name is provided, or if both are provided.
        Exception: If model already exists (FAIL_IF_EXISTS) or creation fails
    """
    if model_package_name and model_s3_location:
        raise ValueError(
            "Only one of model_s3_location or model_package_name may be provided, not both."
        )
    if not model_package_name and not model_s3_location:
        raise ValueError("Either model_s3_location or model_package_name must be provided.")

    if model_s3_location:
        validate_s3_uri_prefix(s3_uri=model_s3_location)

    if deployment_mode in [
        DeploymentMode.FAIL_IF_EXISTS,
        DeploymentMode.UPDATE_IF_EXISTS,
    ]:
        try:
            sagemaker_client.describe_model(ModelName=model_name)
            raise Exception(f"Model '{model_name}' already exists.")
        except ClientError as e:
            if e.response["Error"]["Code"] != "ValidationException":
                raise

    logger.info(f"Creating model: {model_name}...")

    # Build PrimaryContainer based on model source
    primary_container: Dict[str, Any] = {
        "Image": _get_sagemaker_inference_image(region),
        "Environment": environment,
    }

    if model_package_name:
        primary_container["ModelPackageName"] = model_package_name
    else:
        primary_container["ModelDataSource"] = {
            "S3DataSource": {
                "S3Uri": model_s3_location,
                "S3DataType": "S3Prefix",
                "CompressionType": "None",
            }
        }

    create_kwargs = {
        "ModelName": model_name,
        "PrimaryContainer": primary_container,
        "ExecutionRoleArn": sagemaker_execution_role_arn,
        "EnableNetworkIsolation": network_isolation,
    }
    if tags:
        create_kwargs["Tags"] = tags
    model_response = sagemaker_client.create_model(**create_kwargs)
    logger.info(f"Model created successfully: {model_response['ModelArn']}")
    return model_response["ModelArn"]


def find_sagemaker_model_by_tag(escrow_uri: str, sagemaker_client: BaseClient) -> Optional[str]:
    """Find an existing SageMaker model tagged with the given escrow URI.

    Uses ResourceGroupsTaggingAPI for efficient tag-based lookup (single API call).
    Returns model ARN or None. Catches permission errors gracefully.
    """
    tag_value = _escrow_tag_value(escrow_uri)
    try:
        tagging_client = boto3.client(
            "resourcegroupstaggingapi",
            region_name=sagemaker_client.meta.region_name,
        )
        response = tagging_client.get_resources(
            TagFilters=[{"Key": ESCROW_URI_TAG_KEY, "Values": [tag_value]}],
            ResourceTypeFilters=["sagemaker:model"],
        )
        results = response.get("ResourceTagMappingList", [])
        if len(results) == 1:
            return results[0]["ResourceARN"]
        if len(results) > 1:
            # Multiple models share the same tag (e.g. from repeated test runs).
            # Return the most recently created one to avoid stale dedup hits.
            def _creation_time(arn: str) -> Any:
                try:
                    model_name = arn.split("/")[-1]
                    return sagemaker_client.describe_model(ModelName=model_name)["CreationTime"]
                except Exception:
                    return datetime.min.replace(tzinfo=timezone.utc)

            return max((r["ResourceARN"] for r in results), key=_creation_time)
    except ClientError as e:
        logger.warning(
            f"Could not search SageMaker models by tag (may lack tag:GetResources permission): {e}"
        )
    except Exception as e:
        logger.warning(f"Unexpected error searching SageMaker models: {e}")
    return None


def create_sagemaker_endpoint(
    model_name: str,
    endpoint_config_name: str,
    endpoint_name: str,
    instance_type: str,
    sagemaker_client: BaseClient,
    initial_instance_count: int = 1,
    deployment_mode: DeploymentMode = DeploymentMode.FAIL_IF_EXISTS,
    inference_component_configs: List[InferenceComponentConfig] = [],  # noqa: B006 - never mutated
    execution_role_arn: Optional[str] = None,
) -> str:
    """Create a SageMaker endpoint config and endpoint.

    When inference_component_configs is provided, creates an IC-compatible endpoint
    (no ModelName in ProductionVariants, uses RoutingConfig) and then creates the
    inference component(s) after the endpoint reaches InService.

    Args:
        model_name: Name of the existing SageMaker model (used in ProductionVariants for standard
            endpoints, or in InferenceComponent Specification when inference_component_configs is provided)
        endpoint_config_name: Name for the endpoint configuration
        endpoint_name: Name for the endpoint
        instance_type: EC2 instance type
        sagemaker_client: SageMaker client
        initial_instance_count: Number of instances
        deployment_mode: How to handle existing resources
        inference_component_configs: List of configs for creating inference components.
            When provided, the endpoint is created in IC-compatible mode.
        execution_role_arn: IAM execution role ARN. Required when inference_component_configs is provided.

    Returns:
        str: Endpoint ARN

    Raises:
        Exception: If resources exist (FAIL_IF_EXISTS) or creation fails
    """
    if deployment_mode in [
        DeploymentMode.FAIL_IF_EXISTS,
        DeploymentMode.UPDATE_IF_EXISTS,
    ]:
        try:
            sagemaker_client.describe_endpoint_config(EndpointConfigName=endpoint_config_name)
            raise Exception(f"Endpoint configuration '{endpoint_config_name}' already exists.")
        except ClientError as e:
            if e.response["Error"]["Code"] != "ValidationException":
                raise

        try:
            sagemaker_client.describe_endpoint(EndpointName=endpoint_name)
            raise Exception(f"Endpoint '{endpoint_name}' already exists.")
        except ClientError as e:
            if e.response["Error"]["Code"] != "ValidationException":
                raise

    # Build endpoint config based on whether we're using inference components
    if inference_component_configs:
        if not execution_role_arn:
            raise ValueError(
                "execution_role_arn is required when creating an inference component endpoint."
            )
        # IC-enabled endpoints support a single production variant; all ICs must
        # target the same variant.
        variant_names = {ic.variant_name for ic in inference_component_configs}
        if len(variant_names) > 1:
            raise ValueError(
                f"All inference component configs must use the same variant_name when "
                f"deploying to a single endpoint, but found: {variant_names}"
            )
        variant_name = inference_component_configs[0].variant_name
        production_variant = {
            "VariantName": variant_name,
            "InstanceType": instance_type,
            "InitialInstanceCount": initial_instance_count,
            "RoutingConfig": {"RoutingStrategy": "LEAST_OUTSTANDING_REQUESTS"},
        }
        config_kwargs: Dict[str, Any] = {
            "EndpointConfigName": endpoint_config_name,
            "ExecutionRoleArn": execution_role_arn,
            "ProductionVariants": [production_variant],
            "Tags": [{"Key": "sagemaker:nova-inference-component", "Value": "true"}],
        }
    else:
        production_variant = {
            "VariantName": "primary",
            "ModelName": model_name,
            "InitialInstanceCount": initial_instance_count,
            "InstanceType": instance_type,
        }
        config_kwargs = {
            "EndpointConfigName": endpoint_config_name,
            "ProductionVariants": [production_variant],
        }

    logger.info(f"Creating endpoint configuration: {endpoint_config_name}...")
    config_response = sagemaker_client.create_endpoint_config(**config_kwargs)
    logger.info(
        f"Endpoint configuration created successfully: {config_response['EndpointConfigArn']}"
    )

    logger.info(f"Creating endpoint: {endpoint_name}...")
    endpoint_response = sagemaker_client.create_endpoint(
        EndpointName=endpoint_name, EndpointConfigName=endpoint_config_name
    )

    logger.info("Waiting for endpoint creation to complete. This can take ~10 minutes...")
    try:
        _monitor_endpoint_creation(sagemaker_client, endpoint_name)
    except Exception as e:
        raise Exception(f"Failed to create deployment {endpoint_name}: {e}")

    # If inference component configs are provided, create ICs after endpoint is InService
    if inference_component_configs:
        for ic_config in inference_component_configs:
            logger.info(f"Creating inference component: {ic_config.inference_component_name}...")
            ic_response = sagemaker_client.create_inference_component(
                InferenceComponentName=ic_config.inference_component_name,
                EndpointName=endpoint_name,
                VariantName=ic_config.variant_name,
                Specification={
                    "ModelName": model_name,
                    "ComputeResourceRequirements": {
                        "NumberOfCpuCoresRequired": ic_config.num_cpus,
                        "NumberOfAcceleratorDevicesRequired": ic_config.num_accelerators,
                        "MinMemoryRequiredInMb": ic_config.min_memory_in_mb,
                    },
                },
                RuntimeConfig={
                    "CopyCount": ic_config.copy_count,
                },
            )
            ic_arn = ic_response["InferenceComponentArn"]
            logger.info(f"Triggered inference component creation: {ic_arn}")

    return endpoint_response["EndpointArn"]


def invoke_sagemaker_inference(
    request_body: Dict[str, Any],
    endpoint_name: str,
    sagemaker_client: BaseClient,
    inference_component_name: Optional[str] = None,
) -> SingleInferenceResult:
    """
     Invoke Sagemaker inference and return result

    Args:
        request_body (Dict[str, Any]): The payload to send to the inference endpoint.
        endpoint_name (str): Name of the SageMaker inference endpoint.
        sagemaker_client (BaseClient): Sagemaker client
        inference_component_name (Optional[str]): Optional inference component to target.
            When provided, adds InferenceComponentName to the API call.

    Returns:
        - Generator[str, None, None] for streaming responses
        - str for non-streaming responses
    """
    current_time = datetime.now(timezone.utc)

    body = json.dumps(request_body)
    is_streaming = request_body.get("stream", False)

    try:
        logger.info(f"Invoking endpoint ({'streaming' if is_streaming else 'non-streaming'})...")

        if is_streaming:
            stream_kwargs: Dict[str, Any] = {
                "EndpointName": endpoint_name,
                "ContentType": "application/json",
                "Body": body,
            }
            if inference_component_name is not None:
                stream_kwargs["InferenceComponentName"] = inference_component_name

            response = sagemaker_client.invoke_endpoint_with_response_stream(**stream_kwargs)

            event_stream = response["Body"]

            def stream_generator():
                for event in event_stream:
                    if "PayloadPart" in event:
                        chunk = event["PayloadPart"]
                        if "Bytes" in chunk:
                            data = chunk["Bytes"].decode()
                            yield data

            return SingleInferenceResult(
                job_id=response["ResponseMetadata"]["RequestId"],
                inference_output_path="",
                started_time=current_time,
                streaming_response=stream_generator(),
                nonstreaming_response=None,
            )
        else:
            invoke_kwargs: Dict[str, Any] = {
                "EndpointName": endpoint_name,
                "ContentType": "application/json",
                "Accept": "application/json",
                "Body": body,
            }
            if inference_component_name is not None:
                invoke_kwargs["InferenceComponentName"] = inference_component_name

            response = sagemaker_client.invoke_endpoint(**invoke_kwargs)

            body_content = json.loads(response["Body"].read().decode("utf-8"))

            return SingleInferenceResult(
                job_id=body_content["id"],
                inference_output_path="",
                started_time=datetime.fromtimestamp(body_content["created"]),
                streaming_response=None,
                nonstreaming_response=body_content["choices"],
            )

    except Exception as e:
        raise Exception(f"Error invoking endpoint {endpoint_name}: {str(e)}")


def create_inference_component(
    inference_component_name: str,
    endpoint_name: str,
    variant_name: str,
    model_name: str,
    num_cpus: int,
    num_accelerators: int,
    min_memory_in_mb: int,
    copy_count: int,
    sagemaker_client: BaseClient,
    region: Optional[str] = None,
) -> DeploymentResult:
    """Create a SageMaker inference component on an existing endpoint.

    Validates the target endpoint is InService and IC-compatible, then calls
    CreateInferenceComponent using the specified SageMaker model name and returns
    a DeploymentResult immediately without waiting for the component to become active.

    Args:
        inference_component_name: Unique name for the inference component.
        endpoint_name: Name of the existing SageMaker endpoint (must be InService).
        variant_name: Production variant name on the endpoint.
        model_name: Name of the existing SageMaker model to use.
        num_cpus: Number of vCPUs to allocate.
        num_accelerators: Number of accelerators (GPUs) to allocate.
        min_memory_in_mb: Minimum memory in MB to allocate.
        copy_count: Number of model copies to deploy.
        sagemaker_client: Boto3 SageMaker client.
        region: Optional AWS region name. When provided, stored in the returned EndpointInfo.

    Returns:
        DeploymentResult: Contains endpoint info with the inference component ARN
            as the URI. Use .status to check current deployment state.

    Raises:
        ValueError: If required parameters are missing or invalid.
        Exception: If the endpoint does not exist, is not InService,
                   or the CreateInferenceComponent API call fails.
    """
    # Validate required string parameters
    required_str_params = {
        "inference_component_name": inference_component_name,
        "endpoint_name": endpoint_name,
        "variant_name": variant_name,
        "model_name": model_name,
    }
    for param_name, param_value in required_str_params.items():
        if not param_value:
            raise ValueError(
                f"Parameter '{param_name}' is required for inference component creation"
            )

    # Validate endpoint exists and is InService
    try:
        endpoint_response = sagemaker_client.describe_endpoint(EndpointName=endpoint_name)
    except ClientError as e:
        error_code = e.response["Error"]["Code"]
        if error_code in ("ValidationException", "ResourceNotFound"):
            raise Exception(f"Endpoint '{endpoint_name}' not found") from e
        raise

    endpoint_status = endpoint_response["EndpointStatus"]
    if endpoint_status != "InService":
        raise Exception(
            f"Endpoint '{endpoint_name}' is not InService (current status: {endpoint_status})"
        )

    # Validate endpoint config is IC-compatible (no ModelName, has RoutingConfig)
    endpoint_config_name = endpoint_response.get("EndpointConfigName")
    if endpoint_config_name:
        try:
            config_response = sagemaker_client.describe_endpoint_config(
                EndpointConfigName=endpoint_config_name
            )
            variants = config_response.get("ProductionVariants", [])
            for variant in variants:
                if "ModelName" in variant:
                    raise Exception(
                        f"Endpoint '{endpoint_name}' is not configured for inference components. "
                        f"The endpoint config '{endpoint_config_name}' has 'ModelName' set on "
                        f"variant '{variant.get('VariantName')}'. Inference components require an "
                        f"endpoint config without ModelName and with RoutingConfig set to "
                        f"LEAST_OUTSTANDING_REQUESTS."
                    )
                routing_config = variant.get("RoutingConfig", {})
                routing_strategy = routing_config.get("RoutingStrategy")
                if routing_strategy != "LEAST_OUTSTANDING_REQUESTS":
                    raise Exception(
                        f"Endpoint '{endpoint_name}' is not configured for inference components. "
                        f"The endpoint config '{endpoint_config_name}' variant "
                        f"'{variant.get('VariantName')}' has RoutingStrategy "
                        f"'{routing_strategy}' but inference components require "
                        f"'LEAST_OUTSTANDING_REQUESTS'."
                    )
        except ClientError as e:
            logger.warning(
                f"Could not validate endpoint config '{endpoint_config_name}': {e}. "
                f"Proceeding with inference component creation."
            )

    create_request = {
        "InferenceComponentName": inference_component_name,
        "EndpointName": endpoint_name,
        "VariantName": variant_name,
        "Specification": {
            "ModelName": model_name,
            "ComputeResourceRequirements": {
                "NumberOfCpuCoresRequired": num_cpus,
                "NumberOfAcceleratorDevicesRequired": num_accelerators,
                "MinMemoryRequiredInMb": min_memory_in_mb,
            },
        },
        "RuntimeConfig": {
            "CopyCount": copy_count,
        },
    }

    try:
        response = sagemaker_client.create_inference_component(**create_request)
    except ClientError as e:
        error_code = e.response["Error"]["Code"]
        if error_code == "ResourceInUse":
            raise Exception(f"Inference component '{inference_component_name}' already exists")
        raise Exception(f"Failed to create inference component '{inference_component_name}': {e}")

    inference_component_arn = response["InferenceComponentArn"]

    return DeploymentResult(
        endpoint=EndpointInfo(
            platform=DeployPlatform.SAGEMAKER,
            endpoint_name=endpoint_name,
            uri=inference_component_arn,
            model_artifact_path=model_name,
            region=region,
        ),
        created_at=datetime.now(timezone.utc),
    )


def monitor_inference_component(
    inference_component_name: str,
    sagemaker_client: BaseClient,
) -> str:
    """Monitor an inference component until it reaches a terminal state.

    Polls DescribeInferenceComponent every 60 seconds until the status is
    InService (success) or Failed (raises exception).  Times out after 1 hour
    if the component remains in a non-terminal state.

    Args:
        inference_component_name: Name of the inference component to monitor.
        sagemaker_client: Boto3 SageMaker client.

    Returns:
        str: Final status ("InService").

    Raises:
        Exception: If the component reaches Failed status, times out, or the API call errors.
    """
    MAX_WAIT_SECONDS = 3600  # 1 hour
    POLL_INTERVAL_SECONDS = 60
    start_time = datetime.now(timezone.utc)

    while True:
        try:
            response = sagemaker_client.describe_inference_component(
                InferenceComponentName=inference_component_name
            )
        except ClientError as e:
            raise Exception(
                f"Error describing inference component '{inference_component_name}': "
                f"{e.response['Error']['Code']} - {e.response['Error']['Message']}"
            )

        status = response["InferenceComponentStatus"]
        elapsed = (datetime.now(timezone.utc) - start_time).total_seconds()

        logger.info(
            f"Inference component '{inference_component_name}' status: {status} | "
            f"Elapsed: {timedelta(seconds=int(elapsed))}"
        )

        if status == "InService":
            logger.info(
                f"Inference component '{inference_component_name}' is now InService. "
                f"Total time elapsed: {timedelta(seconds=int(elapsed))}"
            )
            return "InService"
        elif status == "Failed":
            raise Exception(f"Inference component '{inference_component_name}' failed to deploy")

        if elapsed > MAX_WAIT_SECONDS:
            raise Exception(
                f"Inference component '{inference_component_name}' did not reach a terminal state "
                f"within {MAX_WAIT_SECONDS} seconds (last status: {status})"
            )

        time.sleep(POLL_INTERVAL_SECONDS)


def check_sagemaker_deployment_status(
    deployment_arn: str, region: Optional[str] = None
) -> Optional[str]:
    """Check the current status of a SageMaker deployment (endpoint or inference component).

    For inference component ARNs (containing "inference-component/"), calls
    DescribeInferenceComponent and returns InferenceComponentStatus.
    For endpoint ARNs, calls DescribeEndpoint and returns EndpointStatus.

    Args:
        deployment_arn: The ARN of the SageMaker deployment to check.
        region: Optional AWS region for the client.

    Returns:
        str: Current status of the deployment.

    Raises:
        Exception: If unable to check deployment status.
    """
    sagemaker_client = boto3.client("sagemaker", region_name=region)
    try:
        if "inference-component/" in deployment_arn:
            component_name = deployment_arn.split("/")[-1]
            response = sagemaker_client.describe_inference_component(
                InferenceComponentName=component_name
            )
            return response["InferenceComponentStatus"]
        else:
            endpoint_name = deployment_arn.split("/")[-1]
            response = sagemaker_client.describe_endpoint(EndpointName=endpoint_name)
            return response["EndpointStatus"]
    except Exception as e:
        raise Exception(f"Failed to check deployment status: {e}")


DeploymentResult._register_sagemaker_status_checker(check_sagemaker_deployment_status)
