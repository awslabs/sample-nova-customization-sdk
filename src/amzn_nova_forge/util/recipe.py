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
import re
import shutil
import textwrap
from pathlib import Path
from typing import Any, Dict, List, Optional

import boto3
import yaml
from botocore.exceptions import ClientError

from amzn_nova_forge.core.constants import (
    REGION_TO_ESCROW_ACCOUNT_MAPPING,
    SUPPORTED_DATAMIXING_METHODS,
)
from amzn_nova_forge.core.enums import (
    EvaluationTask,
    Model,
    Platform,
    TrainingMethod,
    Version,
)
from amzn_nova_forge.rft_multiturn import RFTMultiturnInfrastructure
from amzn_nova_forge.util.hub_util import get_hub_content
from amzn_nova_forge.util.logging import logger

S3_URI_REGEX = re.compile(r"^s3://([a-zA-Z0-9.\-_]+)/(.+)$")
S3_ACCESS_POINT_REGEX = re.compile(r"^arn:aws:s3:([^:]+):([^:]+):accesspoint/([^/]+)(?:/(.+))?$")


class FileLoadError(Exception):
    """Custom exception for file loading errors."""

    pass


class RecipePath:
    """Container for recipe paths. Allows automatically deleting temporary recipe directories."""

    roots: List[str] = []

    def __init__(self, path: str, root: Optional[str] = None, temp: bool = False):
        self.path = path
        self.root = root
        self.temp = temp

        if temp and root is not None:
            self.roots.append(root)

    @staticmethod
    def delete_temp_dir(directory):
        try:
            shutil.rmtree(directory)
        except Exception as e:
            logger.warning(f"Failed to delete temporary directory {directory}\nError: {e}")

    def close(self):
        if self.temp:
            RecipePath.delete_temp_dir(self.root)

    def close_all(self):
        for path in RecipePath.roots:
            RecipePath.delete_temp_dir(path)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()


def _parse_s3_uri(uri: str) -> tuple[str, str] | None:
    """Parse S3 URI into (bucket, key) tuple, or None if the URI is invalid."""
    match = S3_URI_REGEX.match(uri)
    if not match:
        return None
    bucket, key = match.groups()
    return (bucket, key)


def _validate_extension(path: str, extension: str) -> None:
    """
    Validate that the given path has the required file extension.

    Args:
        path: File path or S3 URI
        extension: Extension (e.g., '.yaml')

    Raises:
        FileLoadError: If extension doesn't match
    """
    if not path.lower().endswith(extension.lower()):
        raise FileLoadError(f"File must have {extension} extension: {path}")


def load_file_content(
    file_path: str,
    extension: Optional[str] = None,
    encoding: Optional[str] = "utf-8",
    region: Optional[str] = None,
):
    """
    Stream file content line by line from S3 or local filesystem.
    This is a generator that yields lines lazily without loading the entire file into memory.

    Args:
        file_path: Path to file (either local path or S3 URI)
        extension: Optional file extension to validate
        encoding: Optional encoding format (defaults to utf-8)

    Yields:
        Lines from the file

    Raises:
        FileLoadError: If file cannot be loaded
    """
    # Validate extension
    if extension is not None:
        _validate_extension(file_path, extension)

    # Try S3 first
    s3_parts = _parse_s3_uri(file_path)
    if s3_parts:
        bucket, key = s3_parts
        try:
            s3 = boto3.client("s3", region_name=region)
            response = s3.get_object(Bucket=bucket, Key=key)
            # Stream from S3 using iter_lines
            for line in response["Body"].iter_lines():
                yield line.decode(encoding)
        except ClientError as e:
            raise FileLoadError(f"Failed to load S3 file {file_path}: {e}")
    else:
        # Try local filesystem
        try:
            path = Path(file_path)
            with open(path, "r", encoding=encoding) as f:
                for line in f:
                    yield line.rstrip("\n\r")
        except FileNotFoundError:
            raise FileLoadError(f"File not found: {file_path}")
        except OSError as e:
            raise FileLoadError(f"Failed to read file {file_path}: {e}")


def load_file_as_string(
    file_path: str, extension: Optional[str] = None, encoding: Optional[str] = "utf-8"
) -> str:
    """
    Load entire file content as a string from S3 or local filesystem.
    Use this for files that need to be fully parsed (e.g., YAML, JSON).
    For line-by-line processing, use load_file_content() instead.

    Args:
        file_path: Path to file (either local path or S3 URI)
        extension: Optional file extension to validate
        encoding: Optional encoding format (defaults to utf-8)

    Returns:
        File content as string

    Raises:
        FileLoadError: If file cannot be loaded
    """
    lines = load_file_content(file_path, extension, encoding)
    return "\n".join(lines)


def get_hub_recipe_metadata(
    model: Model,
    method: TrainingMethod,
    platform: Platform,
    region: str,
    instance_type: Optional[str],
    task: Optional[EvaluationTask] = None,
    data_mixing: bool = False,
    is_multimodal: bool = False,
    hub_content_version: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Extract a single recipe's metadata from a SageMaker DescribeHubContent response

    Args:
        model: Model to fetch recipe metadata for
        method: Training method to fetch recipe metadata for
        platform: Training platform to fetch recipe metadata for
        region: AWS region
        instance_type: Instance type to fetch recipe metadata for (None for Bedrock)
        task: Evaluation task (only required for evaluation)
        hub_content_version: Optional version of the hub content to retrieve

    Returns:
        Dict containing raw recipe metadata. Example:
        {
            "DisplayName": "Nova Lite V2 LoRA RLVR SMTJ training on GPU",
            "Name": "nova_lite_v2_smtj_p5_p5en_gpu_lora_rft",
            "RecipeFilePath": "recipes/fine-tuning/nova/nova_2_0/nova_lite/RFT/nova_lite_v2_smtj_p5_p5en_gpu_lora_rft.yaml",
            "CustomizationTechnique": "RLVR",
            "InstanceCount": 4,
            "Type": "FineTuning",
            "Versions": [
              "1.0"
            ],
            "Hardware": "GPU",
            "SupportedInstanceTypes": [
              "ml.p5.48xlarge",
              "ml.p5en.48xlarge"
            ],
            "Peft": "LORA",
            "SequenceLength": "8K",
            "ServerlessMeteringType": "Hourly",
            "SmtjRecipeTemplateS3Uri": "s3://jumpstart-cache-prod-us-east-1/recipes/nova_lite_v2_smtj_p5_p5en_gpu_lora_rft_payload_template_sm_jobs_v1.0.20.yaml",
            "SmtjOverrideParamsS3Uri": "s3://jumpstart-cache-prod-us-east-1/recipes/nova_lite_v2_smtj_p5_p5en_gpu_lora_rft_override_params_sm_jobs_v1.0.20.json",
            "SmtjImageUri": "708977205387.dkr.ecr.us-east-1.amazonaws.com/nova-fine-tune-repo:SM-TJ-RFT-V2-latest"
      }
    """
    # Handle Bedrock platform - return local template paths
    if platform == Platform.BEDROCK:
        # Map training method to template name
        if method == TrainingMethod.SFT_LORA:
            template_name = "sft_lora"
        elif method == TrainingMethod.RFT_LORA:
            template_name = "rft_lora"
        else:
            raise ValueError(
                f"Bedrock only supports SFT_LORA and RFT_LORA. "
                f"Method '{method.value}' is not supported."
            )

        # Get the path to the templates directory
        import amzn_nova_forge

        sdk_path = os.path.dirname(amzn_nova_forge.__file__)

        return {
            "InstanceCount": 1,
            "SupportedInstanceTypes": [],  # Bedrock manages infrastructure
            "RecipeTemplatePath": os.path.join(
                sdk_path,
                "recipe",
                "templates",
                "bedrock",
                "recipe",
                f"{template_name}.yaml",
            ),
            "OverrideParamsPath": os.path.join(
                sdk_path,
                "recipe",
                "templates",
                "bedrock",
                "override",
                f"{template_name}.json",
            ),
            "Platform": platform.value,
            "Model": model.value,
            "Method": method.value,
            "ImageUri": "bedrock-managed",  # Bedrock manages its own infrastructure
        }

    # instance_type is required for SMHP and SMTJ
    if instance_type is None and platform != Platform.SMTJServerless:
        raise ValueError(f"instance_type is required for platform {platform.value}. ")

    hub_content = get_hub_content(
        hub_name="SageMakerPublicHub",
        hub_content_name=model.hub_content_name,
        hub_content_type="Model",
        region=region,
        hub_content_version=hub_content_version,
    )

    document = hub_content.get("HubContentDocument", {})
    recipe_collection = document.get("RecipeCollection", [])

    # Filter out Forge recipes
    if not data_mixing:
        recipe_collection = [
            r for r in recipe_collection if r.get("IsSubscriptionModel") is not True
        ]

    # Filter recipes for training method (SageMaker stores "RFT" as "RLVR")
    METHOD_FILTER = {
        TrainingMethod.CPT: ("CustomizationTechnique", "CPT"),
        TrainingMethod.DPO_LORA: ("CustomizationTechnique", "DPO"),
        TrainingMethod.DPO_FULL: ("CustomizationTechnique", "DPO"),
        TrainingMethod.RFT_LORA: ("CustomizationTechnique", "RLVR"),
        TrainingMethod.RFT_FULL: ("CustomizationTechnique", "RLVR"),
        TrainingMethod.SFT_LORA: ("CustomizationTechnique", "SFT"),
        TrainingMethod.SFT_FULL: ("CustomizationTechnique", "SFT"),
        TrainingMethod.EVALUATION: ("Type", "Evaluation"),
    }
    key, value = METHOD_FILTER[method]
    recipe_collection = [r for r in recipe_collection if r.get(key) == value]
    if not recipe_collection:
        raise ValueError(f"{method.name} is not supported for {model.name}")

    # Filter recipes for training platform
    if platform == Platform.SMTJ or platform == Platform.SMTJServerless:
        recipe_collection = [r for r in recipe_collection if r.get("SmtjRecipeTemplateS3Uri")]
    else:
        recipe_collection = [r for r in recipe_collection if r.get("HpEksPayloadTemplateS3Uri")]
    if not recipe_collection:
        raise ValueError(f"{method.name} is not supported on {platform.name}")

    # For methods with data mixing enabled, look for recipes with "text_with_datamix" or "mm_with_datamix" in the Name
    if data_mixing and method in SUPPORTED_DATAMIXING_METHODS:
        # For SFT methods that support multimodal, check if data is multimodal
        datamix_suffix = "mm_with_datamix" if is_multimodal else "text_with_datamix"
        datamix_recipes = [
            r for r in recipe_collection if datamix_suffix in r.get("Name", "").lower()
        ]
        if datamix_recipes:
            recipe_collection = datamix_recipes
        else:
            # If no datamix recipes found, log warning and continue with regular recipes
            multimodal_log = "multimodal" if is_multimodal else "text"
            logger.warning(
                f"Data mixing is not supported for {method.name} with {multimodal_log} data. "
                "Using standard recipe instead."
            )

    # Filter recipes for training type (i.e. evaluation task, full/lora, etc.)
    if method == TrainingMethod.EVALUATION:
        if task is None:
            raise ValueError("'eval_task' is a required parameter when calling evaluate().")
        if task == EvaluationTask.GEN_QA:
            recipe_collection = [
                r
                for r in recipe_collection
                if "bring your own dataset" in r.get("DisplayName").lower()
            ]
        elif task in [
            EvaluationTask.LLM_JUDGE,
            EvaluationTask.RUBRIC_LLM_JUDGE,
            EvaluationTask.RFT_EVAL,
        ]:
            base_name = f"{task.value}_{model.version.name.lower()}"
            import amzn_nova_forge

            image_prefix = (
                f"{REGION_TO_ESCROW_ACCOUNT_MAPPING[region]}"
                f".dkr.ecr.{region}.amazonaws.com/nova-evaluation-repo:"
            )
            image_infix = "SM-HP-" if platform == Platform.SMHP else "SM-TJ-"
            image_suffix = "Eval-V2-latest" if model.version == Version.TWO else "Eval-latest"

            sdk_path = os.path.dirname(amzn_nova_forge.__file__)
            return {
                "InstanceCount": 1,
                "SupportedInstanceTypes": ["ml.p5.48xlarge"],
                "RecipeTemplatePath": os.path.join(
                    sdk_path, "recipe", "templates", "recipe", f"{base_name}.yaml"
                ),
                "OverrideParamsPath": os.path.join(
                    sdk_path, "recipe", "templates", "override", f"{base_name}.json"
                ),
                "EvaluationTask": task.value,
                "Platform": platform.value,
                "Model": model.value,
                "ImageUri": image_prefix + image_infix + image_suffix,
            }
        else:
            recipe_collection = [
                r
                for r in recipe_collection
                if "general text benchmark" in r.get("DisplayName").lower()
            ]
    elif method.value.lower().endswith("lora"):
        recipe_collection = [r for r in recipe_collection if r.get("Peft")]
    elif method.value.lower().endswith("full"):
        recipe_collection = [r for r in recipe_collection if not r.get("Peft")]

    # If multiple recipes still remain, filter by instance type
    if len(recipe_collection) > 1:
        recipe_collection = [
            r
            for r in recipe_collection
            if r.get("SupportedInstanceTypes") and instance_type in r.get("SupportedInstanceTypes")
        ]

    if recipe_collection:
        return recipe_collection[0]
    else:
        raise ValueError(f"{method.name} using {instance_type} is not supported on {platform.name}")


def _get_aws_account_id(region: Optional[str] = None) -> str:
    """
    Get the AWS account ID from current credentials.

    Returns:
        AWS account ID string
    """
    try:
        sts = boto3.client("sts", region_name=region)
        response = sts.get_caller_identity()
        return response["Account"]
    except Exception as e:
        raise ValueError(f"Failed to get AWS account ID: {e}")


def _replace_customer_id_placeholder(uri: str, current_account: str) -> str:
    """
    Replace {customer_id} placeholder in URI with actual AWS account ID.

    Args:
        uri: URI potentially containing {customer_id} placeholder

    Returns:
        URI with placeholder replaced
    """
    if "{customer_id}" in uri:
        uri = uri.replace("{customer_id}", current_account)
    return uri


def _download_from_s3_or_access_point(uri: str, region: Optional[str] = None) -> bytes:
    """
    Download content from S3 URI or S3 Access Point ARN.
    Handles {customer_id} placeholders in URIs.

    Args:
        uri: S3 URI (s3://bucket/key) or Access Point ARN with optional {customer_id} placeholder
        region: AWS region (optional)

    Returns:
        Content as bytes

    Raises:
        ValueError: If URI format is invalid or download fails
    """
    # Replace {customer_id} placeholder if present

    s3 = boto3.client("s3", region_name=region)

    current_account = _get_aws_account_id(region=region)
    formatted_uri = _replace_customer_id_placeholder(uri, current_account)
    # Check if this is an access point ARN (with or without s3:// prefix)
    arn_to_check = (
        formatted_uri[5:] if formatted_uri.startswith("s3://arn:aws:s3:") else formatted_uri
    )
    access_point_match = S3_ACCESS_POINT_REGEX.match(arn_to_check)

    if access_point_match:
        arn_region, account, access_point_name, key = access_point_match.groups()

        # For access points, we need to use the full ARN as the bucket
        if not key:
            raise ValueError(f"S3 Access Point ARN must include key: {uri}")

        bucket_arn = f"arn:aws:s3:{arn_region}:{account}:accesspoint/{access_point_name}"

        try:
            response = s3.get_object(Bucket=bucket_arn, Key=key)
            return response["Body"].read()
        except Exception as e:
            raise ValueError(
                f"Failed to download from S3 Access Point {e}"
                f"\nVerify if account {current_account} has Forge subscription. Refer: https://docs.aws.amazon.com/sagemaker/latest/dg/nova-forge.html#nova-forge-prereq-access"
                f"\nAlso ensure your IAM role has the right 's3:GetObject'. Refer DataMixingForgeRecipes permission in docs/user-guides/iam_setup.md"
                f" or set data_mixing = False"
            )

    # Regular S3 URI
    s3_match = S3_URI_REGEX.match(formatted_uri)
    if s3_match:
        bucket, key = s3_match.groups()
        try:
            response = s3.get_object(Bucket=bucket, Key=key)
            return response["Body"].read()
        except ClientError as e:
            raise FileLoadError(f"Failed to load S3 file {formatted_uri}: {e}")

    raise ValueError(f"Invalid S3 URI or Access Point ARN format: {uri}")


def download_templates_from_s3(
    recipe_metadata: Dict[str, Any], platform: Platform, method: TrainingMethod
) -> tuple:
    """
    Download recipe and overrides templates from Jump Start S3 buckets or load from local files

    Args:
        recipe_metadata: Dict of recipe metadata fetched from SM DescribeHubContent API
        platform: Platform for training (SMTJ, SMHP, or BEDROCK)
        method: Training method

    Returns:
        tuple: (recipe template, overrides template, image_uri)
    """
    # Check if this recipe uses local templates (Bedrock or certain evaluation tasks)
    if "RecipeTemplatePath" in recipe_metadata and "OverrideParamsPath" in recipe_metadata:
        return download_templates_from_local(recipe_metadata)

    image_uri = None

    if platform == Platform.SMTJ or platform == Platform.SMTJServerless:
        recipe_template_s3_uri = recipe_metadata.get("SmtjRecipeTemplateS3Uri")
        overrides_template_s3_uri = recipe_metadata.get("SmtjOverrideParamsS3Uri")
        image_uri = recipe_metadata.get("SmtjImageUri")
    else:
        recipe_template_s3_uri = recipe_metadata.get("HpEksPayloadTemplateS3Uri")
        overrides_template_s3_uri = recipe_metadata.get("HpEksOverrideParamsS3Uri")

    if recipe_template_s3_uri is None or overrides_template_s3_uri is None:
        raise ValueError("Unable to find recipe")

    recipe_template_content = _download_from_s3_or_access_point(recipe_template_s3_uri)
    overrides_template_content = _download_from_s3_or_access_point(overrides_template_s3_uri)

    recipe_template_raw = recipe_template_content.decode("utf-8")
    recipe_template = recipe_template_raw

    # SMHP recipe template includes additional information that we can exclude
    if platform == Platform.SMHP:
        if "training-config.yaml" in recipe_template:
            # Extract recipe template via the first occurrence of training-config.yaml
            recipe_pattern = (
                r"# Source: .*/training-config\.yaml.*?config\.yaml: \|-\n(.*?)(?=---|\Z)"
            )
            recipe_match = re.search(recipe_pattern, recipe_template, re.DOTALL)
            if recipe_match:
                # Extract just the config content after "config.yaml: |-"
                recipe_template = textwrap.dedent(recipe_match.group(1)).strip()

                # Remove extra line from RFT recipes, and normalize spacing
                if method in [TrainingMethod.RFT_FULL, TrainingMethod.RFT_LORA]:
                    recipe_template = re.sub(
                        r"^\s*task_type:.*$", "", recipe_template, flags=re.MULTILINE
                    )

                recipe_template = textwrap.dedent(recipe_template)
            else:
                raise ValueError(
                    "Unable to generate HyperPod recipe. Please raise an issue if the error persists: https://github.com/aws/nova-forge-sdk/issues"
                )

            # Extract training image URI
            image_pattern = r"name:\s*pytorch\s*\n\s*image:\s*(.+?)(?:\s|$)"
            image_match = re.search(
                image_pattern,
                recipe_template_raw,
                re.MULTILINE,
            )
            if image_match:
                image_uri = image_match.group(1).strip()
            else:
                raise ValueError(
                    "Unable to generate image URI. Please raise an issue if the error persists: https://github.com/aws/nova-forge-sdk/issues"
                )
        else:
            raise ValueError(
                "Unable to generate HyperPod recipe. Please raise an issue if the error persists: https://github.com/aws/nova-forge-sdk/issues"
            )

    if image_uri is None:
        raise ValueError(f"SDK does not yet support '{method.value}' on '{platform.value}'")

    recipe_template_dict = yaml.safe_load(recipe_template)
    overrides_template_dict = json.loads(overrides_template_content)

    return recipe_template_dict, overrides_template_dict, image_uri


def download_templates_from_local(recipe_metadata: Dict[str, Any]) -> tuple:
    """
    Download recipe and overrides templates from a local path

    Args:
        recipe_metadata: Dict of recipe metadata

    Returns:
        tuple: (recipe template, overrides template)
    """
    recipe_template_path = recipe_metadata["RecipeTemplatePath"]
    overrides_template_path = recipe_metadata["OverrideParamsPath"]
    image_uri = recipe_metadata["ImageUri"]

    try:
        with open(recipe_template_path, "r") as file:
            recipe_template_dict = yaml.safe_load(file)
        with open(overrides_template_path, "r") as file:
            overrides_template_dict = json.load(file)
    except:
        raise ValueError(
            f"'{recipe_metadata['EvaluationTask']}' is not supported on {recipe_metadata['Platform']} for {recipe_metadata['Model']}"
        )

    return recipe_template_dict, overrides_template_dict, image_uri


def _build_rft_overrides_from_recipe(recipe_str: str, method: TrainingMethod) -> Dict[str, Any]:
    """
    Build overrides template from recipe YAML for RFT multiturn.
    Extracts all leaf fields and uses their values as defaults.

    Args:
        recipe_str: Recipe YAML as string

    Returns:
        Overrides template dict
    """
    import yaml

    recipe_dict = yaml.safe_load(recipe_str)
    overrides_template = {}

    def extract_fields(obj: Any):
        """Recursively extract all leaf fields"""
        if isinstance(obj, dict):
            for key, value in obj.items():
                if isinstance(value, dict):
                    extract_fields(value)
                else:
                    overrides_template[key] = {
                        "default": value,
                        "type": None,  # using None to avoid float v/s int conflicts (especially for temperature)
                        "required": False,
                    }
                    if key == "replicas":
                        # List values from https://docs.aws.amazon.com/sagemaker/latest/dg/nova-reinforcement-fine-tuning.html#nova-rft-creating-jobs
                        if method in [
                            TrainingMethod.RFT_MULTITURN_FULL,
                            TrainingMethod.RFT_MULTITURN_LORA,
                        ]:
                            overrides_template[key]["enum"] = [2, 4, 6, 8]
                        else:  # for eval
                            overrides_template[key]["enum"] = [1, 2, 4, 8, 16]

    extract_fields(recipe_dict)
    return overrides_template


def _get_smhp_replicas_enum(
    model: Model,
    method: TrainingMethod,
    region: str,
    instance_type: str,
    eval_task: Optional[EvaluationTask] = None,
    data_mixing_enabled: bool = False,
    is_multimodal: bool = False,
    hub_content_version: Optional[str] = None,
) -> Optional[List[int]]:
    """
    Fetch the SMHP overrides template for the given model/method/instance_type and
    return the replicas enum (valid instance counts).

    SMTJ hub content does not include a replicas enum in its overrides template, but
    the SMHP recipe for the same configuration does. This function retrieves that enum
    so it can be applied to SMTJ recipe validation.

    Args:
        model: The Nova model
        method: The training method
        region: AWS region
        instance_type: Instance type
        eval_task: Optional evaluation task
        data_mixing_enabled: Whether data mixing is enabled
        is_multimodal: Whether the dataset is multimodal
        hub_content_version: Optional version of the hub content to retrieve

    Returns:
        List of valid instance counts, or None if unavailable
    """
    try:
        smhp_metadata = get_hub_recipe_metadata(
            model=model,
            method=method,
            platform=Platform.SMHP,
            region=region,
            instance_type=instance_type,
            task=eval_task,
            data_mixing=data_mixing_enabled,
            is_multimodal=is_multimodal,
            hub_content_version=hub_content_version,
        )
        _, smhp_overrides, _ = download_templates_from_s3(
            recipe_metadata=smhp_metadata,
            platform=Platform.SMHP,
            method=method,
        )
        replicas_meta = smhp_overrides.get("replicas", {})
        enum_val = replicas_meta.get("enum")
        if isinstance(enum_val, list) and enum_val:
            return enum_val
    except Exception as e:
        logger.warning(
            f"Could not fetch valid instance counts for {model.name}/{method.name}/{instance_type}: {e}. "
            "Instance count validation will be skipped. "
            "For valid instance counts, refer to: "
            "Nova 1: https://docs.aws.amazon.com/nova/latest/userguide/nova-fine-tune-1.html | "
            "Nova 2: https://docs.aws.amazon.com/nova/latest/nova2-userguide/nova-fine-tune-2.html"
        )
    return None


def load_recipe_templates(
    model: Model,
    method: TrainingMethod,
    platform: Platform,
    region: str,
    instance_type: Optional[str],
    data_mixing_enabled: bool = False,
    eval_task: Optional[EvaluationTask] = None,
    rft_multiturn_infra: Optional[RFTMultiturnInfrastructure] = None,
    image_uri_override: Optional[str] = None,
    is_multimodal: bool = False,
    hub_content_version: Optional[str] = None,
) -> tuple:
    """
    Load recipe metadata and templates for Nova model customization.

    This function handles the logic to get recipe metadata and download the appropriate
    recipe and overrides templates based on the training method and task.

    Args:
        model: The Nova model to be trained
        method: The fine-tuning method
        platform: Training platform (SMTJ or SMHP)
        region: AWS region
        instance_type: Instance type to fetch recipe metadata for (None for Bedrock)
        data_mixing_enabled: Whether data mixing is enabled
        eval_task: Optional evaluation task (only for evaluation methods)
        image_uri_override: Optional custom ECR image URI to override default
        is_multimodal: Whether the dataset contains multimodal data (image/video)
        hub_content_version: Optional version of the hub content to retrieve

    Returns:
        tuple: (recipe_metadata, recipe_template, overrides_template, image_uri)

    Raises:
        Exception: If recipe configuration cannot be loaded
    """
    # Handle RFT Multiturn methods and eval
    is_rft_multiturn_training = method in [
        TrainingMethod.RFT_MULTITURN_LORA,
        TrainingMethod.RFT_MULTITURN_FULL,
    ]
    is_rft_multiturn_eval = (
        method == TrainingMethod.EVALUATION
        and eval_task == EvaluationTask.RFT_MULTITURN_EVAL
        and rft_multiturn_infra
    )

    if (is_rft_multiturn_training or is_rft_multiturn_eval) and platform != Platform.SMTJServerless:
        if not rft_multiturn_infra:
            raise ValueError(
                f"rft_multiturn_infra is required for RFT multiturn {'training' if is_rft_multiturn_training else 'evaluation'}"
            )
        if platform != Platform.SMHP:
            raise ValueError(
                f"RFT multiturn {'training' if is_rft_multiturn_training else 'eval'} is only supported on HyperPod (SMHP) or SMTJServerless"
            )

        # RFT multiturn requires instance_type (only supported on SMHP)
        if instance_type is None:
            raise ValueError(
                "instance_type is required for RFT multiturn (only supported on SMHP, not Bedrock)"
            )

        # Determine base method for metadata
        if is_rft_multiturn_training:
            base_method = (
                TrainingMethod.RFT_LORA
                if method == TrainingMethod.RFT_MULTITURN_LORA
                else TrainingMethod.RFT_FULL
            )
            base_task = eval_task
        else:
            base_method = method
            base_task = EvaluationTask.RFT_EVAL

        recipe_metadata = get_hub_recipe_metadata(
            model=model,
            method=base_method,
            platform=platform,
            region=region,
            instance_type=instance_type,
            task=base_task,
            data_mixing=data_mixing_enabled,
            is_multimodal=is_multimodal,
            hub_content_version=hub_content_version,
        )

        recipe_path = rft_multiturn_infra.get_recipe_path(
            method if is_rft_multiturn_training else eval_task
        )
        with open(recipe_path, "r") as f:
            recipe_template_str = f.read()

        import yaml

        recipe_template = yaml.safe_load(recipe_template_str)
        overrides_template = _build_rft_overrides_from_recipe(recipe_template_str, method)

        # Apply RFT multiturn infrastructure overrides
        rft_overrides = rft_multiturn_infra.get_recipe_overrides()
        for key, value in rft_overrides.items():
            if key in overrides_template:
                overrides_template[key]["default"] = value

        if is_rft_multiturn_training:
            _, _, image_uri = download_templates_from_s3(
                recipe_metadata=recipe_metadata, platform=platform, method=base_method
            )
        elif is_rft_multiturn_eval:
            _, _, image_uri = download_templates_from_local(recipe_metadata=recipe_metadata)

        # Override image URI if provided
        if image_uri_override:
            from amzn_nova_forge.validation.validator import Validator

            Validator.validate_ecr_image_uri(image_uri_override)
            image_uri = image_uri_override

        return recipe_metadata, recipe_template, overrides_template, image_uri

    # For non-Bedrock platforms, instance_type is required
    if platform not in [Platform.BEDROCK, Platform.SMTJServerless] and instance_type is None:
        raise ValueError(f"instance_type is required for platform {platform.value}.")

    # Get recipe metadata
    # SMTJServerless has no instance type — use p5 as default to select recipe
    # (only hyperparameters are used from the recipe, not compute config)
    resolved_instance_type = (
        "ml.p5.48xlarge" if platform == Platform.SMTJServerless else instance_type
    )
    recipe_metadata = get_hub_recipe_metadata(
        model=model,
        method=method,
        platform=platform,
        region=region,
        instance_type=resolved_instance_type,
        task=eval_task,
        data_mixing=data_mixing_enabled,
        is_multimodal=is_multimodal,
        hub_content_version=hub_content_version,
    )

    # Download recipe and overrides templates
    # download_templates_from_s3() automatically handles both S3 and local templates
    # based on the presence of RecipeTemplatePath/OverrideParamsPath in recipe_metadata
    recipe_template, overrides_template, image_uri = download_templates_from_s3(
        recipe_metadata=recipe_metadata, platform=platform, method=method
    )

    # For SMTJ, the hub content overrides template does not include a replicas enum
    # (valid instance counts). Fetch the SMHP overrides template for the same
    # model/method/instance_type and copy its replicas.enum so that invalid instance
    # counts are caught during recipe validation.
    # Skip for EVALUATION — neither SMTJ nor SMHP eval overrides templates contain a
    # replicas enum and we use 1 by default.
    if (
        platform == Platform.SMTJ
        and instance_type is not None
        and method != TrainingMethod.EVALUATION
    ):
        smhp_replicas_enum = _get_smhp_replicas_enum(
            model=model,
            method=method,
            region=region,
            instance_type=instance_type,
            eval_task=eval_task,
            data_mixing_enabled=data_mixing_enabled,
            is_multimodal=is_multimodal,
            hub_content_version=hub_content_version,
        )
        if smhp_replicas_enum:
            overrides_template.setdefault("replicas", {})["enum"] = smhp_replicas_enum

    # Override image URI if provided
    if image_uri_override:
        from amzn_nova_forge.validation.validator import Validator

        Validator.validate_ecr_image_uri(image_uri_override)
        image_uri = image_uri_override

    return recipe_metadata, recipe_template, overrides_template, image_uri
