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
import time
import unittest
from datetime import datetime
from unittest.mock import MagicMock, patch

from botocore.exceptions import ClientError

from amzn_nova_forge.core.enums import DeploymentMode, DeployPlatform, Model, Platform
from amzn_nova_forge.core.result.inference_result import (
    SingleInferenceResult,
)
from amzn_nova_forge.core.types import DeploymentResult, EndpointInfo, ModelArtifacts
from amzn_nova_forge.manager.runtime_manager import (
    RuntimeManager,
    SMHPRuntimeManager,
    SMTJRuntimeManager,
    SMTJServerlessRuntimeManager,
)
from amzn_nova_forge.util.sagemaker import (
    _IC_MIN_COMPUTE_REQUIREMENTS,
    InferenceComponentConfig,
    _get_sagemaker_inference_image,
    _validate_sagemaker_instance_type_for_model_deployment,
    create_inference_component,
    create_sagemaker_endpoint,
    create_sagemaker_model,
    get_model_artifacts,
    invoke_sagemaker_inference,
    monitor_inference_component,
    validate_inference_component_resources,
)

IC_NAME = "my-ic-component"
IC_ENDPOINT_NAME = "my-endpoint"
IC_VARIANT_NAME = "AllTraffic"
IC_IMAGE_URI = "123456789012.dkr.ecr.us-east-1.amazonaws.com/my-repo:latest"
IC_MODEL_ARTIFACT_PATH = "s3://my-bucket/model/artifacts"
IC_ENVIRONMENT = {"MODEL_NAME": "nova-micro"}
IC_NUM_CPUS = 4
IC_NUM_ACCELERATORS = 1
IC_MIN_MEMORY_IN_MB = 8192
IC_COPY_COUNT = 1

DEFAULT_IC_PARAMS = {
    "inference_component_name": IC_NAME,
    "endpoint_name": IC_ENDPOINT_NAME,
    "variant_name": IC_VARIANT_NAME,
    "model_name": "my-sagemaker-model",
    "num_cpus": IC_NUM_CPUS,
    "num_accelerators": IC_NUM_ACCELERATORS,
    "min_memory_in_mb": IC_MIN_MEMORY_IN_MB,
    "copy_count": IC_COPY_COUNT,
}


class TestSagemaker(unittest.TestCase):
    def setUp(self):
        # Common test data
        self.endpoint_name = "test-endpoint"
        self.request_body = {"inputs": "Test prompt", "stream": False}
        self.region = "us-east-1"
        self.model_name = "test-model"
        self.model_s3_location = "s3://test-bucket/model/"
        self.sagemaker_execution_role_arn = "arn:aws:iam::123456789012:role/test-role"
        self.endpoint_config_name = "test-endpoint-config"
        self.endpoint_name = "test-endpoint"

    def test_create_sagemaker_model_success(self):
        mock_client = MagicMock()
        mock_client.describe_model.side_effect = ClientError(
            {"Error": {"Code": "ValidationException"}}, ""
        )
        mock_client.create_model.return_value = {
            "ModelArn": "arn:aws:sagemaker:us-east-1:123456789012:model/test-model"
        }

        result = create_sagemaker_model(
            region=self.region,
            model_name=self.model_name,
            model_s3_location=self.model_s3_location,
            sagemaker_execution_role_arn=self.sagemaker_execution_role_arn,
            sagemaker_client=mock_client,
        )

        self.assertEqual(result, "arn:aws:sagemaker:us-east-1:123456789012:model/test-model")
        mock_client.create_model.assert_called_once()

    def test_create_sagemaker_model_already_exists(self):
        mock_client = MagicMock()
        mock_client.describe_model.return_value = {"ModelName": "test-model"}

        with self.assertRaises(Exception) as ctx:
            create_sagemaker_model(
                region=self.region,
                model_name=self.model_name,
                model_s3_location=self.model_s3_location,
                sagemaker_execution_role_arn=self.sagemaker_execution_role_arn,
                sagemaker_client=mock_client,
            )
        self.assertIn("already exists", str(ctx.exception))

    def test_create_sagemaker_model_invalid_s3(self):
        with self.assertRaises(ValueError):
            create_sagemaker_model(
                region=self.region,
                model_name=self.model_name,
                model_s3_location="invalid-s3-uri",
                sagemaker_execution_role_arn=self.sagemaker_execution_role_arn,
                sagemaker_client=MagicMock(),
            )

    def test_create_sagemaker_model_with_model_package_name(self):
        """Model package name is passed as ModelPackageName in PrimaryContainer."""
        mock_client = MagicMock()
        mock_client.describe_model.side_effect = ClientError(
            {"Error": {"Code": "ValidationException"}}, ""
        )
        mock_client.create_model.return_value = {
            "ModelArn": "arn:aws:sagemaker:us-east-1:123456789012:model/test-model"
        }

        model_package_arn = "arn:aws:sagemaker:us-east-1:123456789012:model-package/my-group/1"
        result = create_sagemaker_model(
            region=self.region,
            model_name=self.model_name,
            sagemaker_execution_role_arn=self.sagemaker_execution_role_arn,
            sagemaker_client=mock_client,
            model_package_name=model_package_arn,
        )

        self.assertEqual(result, "arn:aws:sagemaker:us-east-1:123456789012:model/test-model")
        call_kwargs = mock_client.create_model.call_args[1]
        primary_container = call_kwargs["PrimaryContainer"]
        self.assertEqual(primary_container["ModelPackageName"], model_package_arn)
        self.assertNotIn("ModelDataSource", primary_container)
        # Image should still be present
        self.assertIn("Image", primary_container)

    def test_create_sagemaker_model_both_s3_and_package_raises(self):
        """Providing both model_s3_location and model_package_name raises ValueError."""
        with self.assertRaises(ValueError) as ctx:
            create_sagemaker_model(
                region=self.region,
                model_name=self.model_name,
                model_s3_location=self.model_s3_location,
                sagemaker_execution_role_arn=self.sagemaker_execution_role_arn,
                sagemaker_client=MagicMock(),
                model_package_name="arn:aws:sagemaker:us-east-1:123456789012:model-package/group/1",
            )
        self.assertIn("Only one of", str(ctx.exception))

    def test_create_sagemaker_model_neither_s3_nor_package_raises(self):
        """Providing neither model_s3_location nor model_package_name raises ValueError."""
        with self.assertRaises(ValueError) as ctx:
            create_sagemaker_model(
                region=self.region,
                model_name=self.model_name,
                sagemaker_execution_role_arn=self.sagemaker_execution_role_arn,
                sagemaker_client=MagicMock(),
            )
        self.assertIn("must be provided", str(ctx.exception))

    @patch("amzn_nova_forge.util.sagemaker._monitor_endpoint_creation")
    def test_create_sagemaker_endpoint_success(self, mock_monitor):
        mock_client = MagicMock()
        mock_client.describe_endpoint_config.side_effect = ClientError(
            {"Error": {"Code": "ValidationException"}}, ""
        )
        mock_client.describe_endpoint.side_effect = ClientError(
            {"Error": {"Code": "ValidationException"}}, ""
        )
        mock_client.create_endpoint_config.return_value = {"EndpointConfigArn": "test-config-arn"}
        mock_client.create_endpoint.return_value = {"EndpointArn": "test-endpoint-arn"}
        mock_monitor.return_value = "InService"

        result = create_sagemaker_endpoint(
            model_name=self.model_name,
            endpoint_config_name=self.endpoint_config_name,
            endpoint_name=self.endpoint_name,
            instance_type="ml.g5.4xlarge",
            sagemaker_client=mock_client,
        )

        self.assertEqual(result, "test-endpoint-arn")
        mock_client.create_endpoint_config.assert_called_once()
        mock_client.create_endpoint.assert_called_once()

    def test_create_sagemaker_endpoint_already_exists(self):
        mock_client = MagicMock()
        mock_client.describe_endpoint_config.side_effect = ClientError(
            {"Error": {"Code": "ValidationException"}}, ""
        )
        mock_client.describe_endpoint.return_value = {"EndpointName": "test-endpoint"}

        with self.assertRaises(Exception) as ctx:
            create_sagemaker_endpoint(
                model_name=self.model_name,
                endpoint_config_name=self.endpoint_config_name,
                endpoint_name=self.endpoint_name,
                instance_type="ml.g5.4xlarge",
                sagemaker_client=mock_client,
            )
        self.assertIn("already exists", str(ctx.exception))

    @patch("amzn_nova_forge.util.sagemaker._monitor_endpoint_creation")
    def test_create_sagemaker_endpoint_with_ic_returns_endpoint_arn(self, mock_monitor):
        mock_client = MagicMock()
        mock_client.describe_endpoint_config.side_effect = ClientError(
            {"Error": {"Code": "ValidationException"}}, ""
        )
        mock_client.describe_endpoint.side_effect = ClientError(
            {"Error": {"Code": "ValidationException"}}, ""
        )
        mock_client.create_endpoint_config.return_value = {"EndpointConfigArn": "test-config-arn"}
        mock_client.create_endpoint.return_value = {
            "EndpointArn": "arn:aws:sagemaker:us-east-1:123456789012:endpoint/test-endpoint"
        }
        mock_client.create_inference_component.return_value = {
            "InferenceComponentArn": "arn:aws:sagemaker:us-east-1:123456789012:inference-component/test-ic"
        }
        mock_monitor.return_value = "InService"

        ic_config = InferenceComponentConfig(
            inference_component_name="test-ic",
            num_cpus=15,
            num_accelerators=4,
            min_memory_in_mb=25000,
        )

        result = create_sagemaker_endpoint(
            model_name=self.model_name,
            endpoint_config_name=self.endpoint_config_name,
            endpoint_name=self.endpoint_name,
            instance_type="ml.g5.4xlarge",
            sagemaker_client=mock_client,
            inference_component_configs=[ic_config],
            execution_role_arn="arn:aws:iam::123456789012:role/test-role",
        )

        self.assertEqual(result, "arn:aws:sagemaker:us-east-1:123456789012:endpoint/test-endpoint")
        mock_client.create_inference_component.assert_called_once()

    def test_create_sagemaker_endpoint_ic_without_execution_role_raises(self):
        """Providing inference_component_configs without execution_role_arn must raise ValueError."""
        mock_client = MagicMock()
        mock_client.describe_endpoint_config.side_effect = ClientError(
            {"Error": {"Code": "ValidationException"}}, ""
        )
        mock_client.describe_endpoint.side_effect = ClientError(
            {"Error": {"Code": "ValidationException"}}, ""
        )

        ic_config = InferenceComponentConfig(
            inference_component_name="test-ic",
            num_cpus=15,
            num_accelerators=4,
            min_memory_in_mb=25000,
        )

        with self.assertRaises(ValueError) as ctx:
            create_sagemaker_endpoint(
                model_name=self.model_name,
                endpoint_config_name=self.endpoint_config_name,
                endpoint_name=self.endpoint_name,
                instance_type="ml.g5.4xlarge",
                sagemaker_client=mock_client,
                inference_component_configs=[ic_config],
                # execution_role_arn intentionally omitted
            )

        self.assertIn("execution_role_arn", str(ctx.exception))
        mock_client.create_endpoint_config.assert_not_called()
        mock_client.create_endpoint.assert_not_called()

    def test_get_sagemaker_inference_image_unsupported_region(self):
        with self.assertRaises(ValueError):
            _get_sagemaker_inference_image(region="unsupported_region")

    def test_get_sagemaker_inference_image_supported_region(self):
        self.assertEqual(
            _get_sagemaker_inference_image(region="us-east-1"),
            "708977205387.dkr.ecr.us-east-1.amazonaws.com/nova-inference-repo:SM-Inference-latest",
        )

    @patch("amzn_nova_forge.util.sagemaker.boto3.client")
    def test_get_model_artifacts_smtj(self, mock_boto_client):
        job_name = "test-training-job"
        checkpoint_s3_uri = "s3://my-bucket/checkpoints/"
        output_s3_path = "s3://my-bucket/output/"

        mock_sagemaker = MagicMock()
        mock_sagemaker.describe_training_job.return_value = {
            "CheckpointConfig": {"S3Uri": checkpoint_s3_uri},
            "OutputDataConfig": {"S3OutputPath": output_s3_path},
        }
        mock_boto_client.return_value = mock_sagemaker

        infra = MagicMock(spec=SMTJRuntimeManager)
        infra.platform = Platform.SMTJ

        result = get_model_artifacts(
            job_name=job_name,
            infra=infra,
            output_s3_path=output_s3_path,
        )

        self.assertIsInstance(result, ModelArtifacts)
        self.assertEqual(result.checkpoint_s3_path, checkpoint_s3_uri)
        self.assertEqual(result.output_s3_path, output_s3_path)
        mock_sagemaker.describe_training_job.assert_called_once_with(TrainingJobName=job_name)

    @patch("amzn_nova_forge.util.sagemaker.boto3.client")
    def test_get_model_artifacts_smhp_single_rig(self, mock_boto_client):
        job_name = "test-hyperpod-job"
        cluster_name = "test-cluster"
        checkpoint_s3_path = "s3://my-bucket/hyperpod-checkpoints/"
        output_s3_path = "s3://my-bucket/hyperpod-output/"

        mock_sagemaker = MagicMock()
        mock_sagemaker.describe_cluster.return_value = {
            "RestrictedInstanceGroups": [
                {
                    "EnvironmentConfig": {
                        "S3OutputPath": checkpoint_s3_path,
                    }
                }
            ]
        }
        mock_boto_client.return_value = mock_sagemaker

        infra = MagicMock(spec=SMHPRuntimeManager)
        infra.platform = Platform.SMHP
        infra.cluster_name = cluster_name

        result = get_model_artifacts(
            job_name=job_name,
            infra=infra,
            output_s3_path=output_s3_path,
        )

        self.assertIsInstance(result, ModelArtifacts)
        self.assertEqual(result.checkpoint_s3_path, checkpoint_s3_path)
        self.assertEqual(result.output_s3_path, output_s3_path)
        mock_sagemaker.describe_cluster.assert_called_once_with(ClusterName=cluster_name)

    @patch("amzn_nova_forge.util.sagemaker.boto3.client")
    def test_get_model_artifacts_smhp_multiple_rigs(self, mock_boto_client):
        job_name = "test-hyperpod-job"
        cluster_name = "test-cluster"
        output_s3_path = "s3://my-bucket/hyperpod-output/"

        mock_sagemaker = MagicMock()
        mock_sagemaker.describe_cluster.return_value = {
            "RestrictedInstanceGroups": [
                {"EnvironmentConfig": {"S3OutputPath": "s3://bucket1/path1/"}},
                {"EnvironmentConfig": {"S3OutputPath": "s3://bucket2/path2/"}},
            ]
        }
        mock_boto_client.return_value = mock_sagemaker

        infra = MagicMock(spec=SMHPRuntimeManager)
        infra.platform = Platform.SMHP
        infra.cluster_name = cluster_name

        result = get_model_artifacts(
            job_name=job_name,
            infra=infra,
            output_s3_path=output_s3_path,
        )

        self.assertIsInstance(result, ModelArtifacts)
        self.assertIsNone(result.checkpoint_s3_path)
        self.assertEqual(result.output_s3_path, output_s3_path)

    @patch("amzn_nova_forge.util.sagemaker.boto3.client")
    def test_get_model_artifacts_smhp_no_rigs(self, mock_boto_client):
        job_name = "test-hyperpod-job"
        cluster_name = "test-cluster"
        output_s3_path = "s3://my-bucket/hyperpod-output/"

        mock_sagemaker = MagicMock()
        mock_sagemaker.describe_cluster.return_value = {"RestrictedInstanceGroups": []}
        mock_boto_client.return_value = mock_sagemaker

        infra = MagicMock(spec=SMHPRuntimeManager)
        infra.platform = Platform.SMHP
        infra.cluster_name = cluster_name

        result = get_model_artifacts(
            job_name=job_name,
            infra=infra,
            output_s3_path=output_s3_path,
        )

        self.assertIsInstance(result, ModelArtifacts)
        self.assertIsNone(result.checkpoint_s3_path)
        self.assertEqual(result.output_s3_path, output_s3_path)

    @patch("amzn_nova_forge.util.sagemaker.boto3.client")
    def test_get_model_artifacts_unsupported_platform(self, mock_boto_client):
        mock_sagemaker = MagicMock()
        mock_boto_client.return_value = mock_sagemaker

        job_name = "test-job"
        output_s3_path = "s3://my-bucket/output/"

        infra = MagicMock(spec=RuntimeManager)

        with self.assertRaises(ValueError) as context:
            get_model_artifacts(
                job_name=job_name,
                infra=infra,
                output_s3_path=output_s3_path,
            )

        self.assertIn("Unsupported platform", str(context.exception))

    @patch("amzn_nova_forge.util.sagemaker.boto3.client")
    def test_get_model_artifacts_smtj_client_error(self, mock_boto_client):
        job_name = "test-training-job"
        output_s3_path = "s3://my-bucket/output/"

        mock_sagemaker = MagicMock()
        mock_sagemaker.describe_training_job.side_effect = ClientError(
            error_response={"Error": {"Code": "ResourceNotFound", "Message": "Job not found"}},
            operation_name="DescribeTrainingJob",
        )
        mock_boto_client.return_value = mock_sagemaker

        infra = MagicMock(spec=SMTJRuntimeManager)
        infra.platform = Platform.SMTJ

        with self.assertRaises(ClientError):
            get_model_artifacts(
                job_name=job_name,
                infra=infra,
                output_s3_path=output_s3_path,
            )

    @patch("amzn_nova_forge.util.sagemaker.boto3.client")
    def test_get_model_artifacts_smhp_client_error(self, mock_boto_client):
        job_name = "test-hyperpod-job"
        cluster_name = "test-cluster"
        output_s3_path = "s3://my-bucket/output/"

        mock_sagemaker = MagicMock()
        mock_sagemaker.describe_cluster.side_effect = ClientError(
            error_response={"Error": {"Code": "ResourceNotFound", "Message": "Cluster not found"}},
            operation_name="DescribeCluster",
        )
        mock_boto_client.return_value = mock_sagemaker

        infra = MagicMock(spec=SMHPRuntimeManager)
        infra.platform = Platform.SMHP
        infra.cluster_name = cluster_name

        with self.assertRaises(ClientError):
            get_model_artifacts(
                job_name=job_name,
                infra=infra,
                output_s3_path=output_s3_path,
            )

    @patch("amzn_nova_forge.util.sagemaker.boto3.client")
    def test_get_model_artifacts_serverless_with_model_package(self, mock_boto_client):
        """Serverless jobs return output_model_arn and checkpoint_s3_path from model package."""
        job_name = "test-serverless-job"
        model_package_arn = "arn:aws:sagemaker:us-east-1:123456789012:model-package/group/1"
        output_s3_path = "s3://my-bucket/output/"
        checkpoint_s3_path = "s3://customer-escrow/job/384/"

        mock_sagemaker = MagicMock()
        mock_sagemaker.describe_training_job.return_value = {
            "OutputModelPackageArn": model_package_arn,
            "OutputDataConfig": {"S3OutputPath": output_s3_path},
        }
        mock_sagemaker.describe_model_package.return_value = {
            "InferenceSpecification": {
                "Containers": [{"ModelDataSource": {"S3DataSource": {"S3Uri": checkpoint_s3_path}}}]
            }
        }
        mock_boto_client.return_value = mock_sagemaker

        infra = MagicMock(spec=SMTJServerlessRuntimeManager)
        infra.platform = Platform.SMTJServerless

        result = get_model_artifacts(job_name=job_name, infra=infra, output_s3_path=output_s3_path)

        self.assertEqual(result.checkpoint_s3_path, checkpoint_s3_path)
        self.assertEqual(result.output_s3_path, output_s3_path)
        self.assertEqual(result.output_model_arn, model_package_arn)
        mock_sagemaker.describe_model_package.assert_called_once_with(
            ModelPackageName=model_package_arn
        )

    @patch("amzn_nova_forge.util.sagemaker.boto3.client")
    def test_get_model_artifacts_serverless_no_model_package(self, mock_boto_client):
        """Serverless job with no OutputModelPackageArn returns None checkpoint and arn."""
        job_name = "test-serverless-job"
        output_s3_path = "s3://my-bucket/output/"

        mock_sagemaker = MagicMock()
        mock_sagemaker.describe_training_job.return_value = {
            "OutputDataConfig": {"S3OutputPath": output_s3_path},
        }
        mock_boto_client.return_value = mock_sagemaker

        infra = MagicMock(spec=SMTJServerlessRuntimeManager)
        infra.platform = Platform.SMTJServerless

        result = get_model_artifacts(job_name=job_name, infra=infra, output_s3_path=output_s3_path)

        self.assertIsNone(result.checkpoint_s3_path)
        self.assertIsNone(result.output_model_arn)
        self.assertEqual(result.output_s3_path, output_s3_path)
        mock_sagemaker.describe_model_package.assert_not_called()

    def test_non_streaming_inference(self):
        """Test non-streaming inference invocation"""
        mock_sagemaker_client = MagicMock()
        mock_response = {
            "Body": MagicMock(
                read=lambda: json.dumps(
                    {
                        "id": "test-id",
                        "created": datetime.now().timestamp(),
                        "choices": ["test_result"],
                    }
                ).encode("utf-8")
            )
        }
        mock_sagemaker_client.invoke_endpoint.return_value = mock_response

        result = invoke_sagemaker_inference(
            request_body=self.request_body,
            endpoint_name=self.endpoint_name,
            sagemaker_client=mock_sagemaker_client,
        )

        mock_sagemaker_client.invoke_endpoint.assert_called_once()

        self.assertIsInstance(result, SingleInferenceResult)

        self.assertEqual(
            result.get(),
            {
                "inference_results": {
                    "is_streaming": False,
                    "response": ["test_result"],
                }
            },
        )

    def test_streaming_inference(self):
        """Test streaming inference invocation"""
        mock_sagemaker_client = MagicMock()

        mock_response_json = {
            "ResponseMetadata": {
                "RequestId": "test-request-id",
            },
            "Body": [
                {"PayloadPart": {"Bytes": b"Hello "}},
                {"PayloadPart": {"Bytes": b"world"}},
                {"SomeOtherEvent": "ignored"},
            ],
        }

        mock_sagemaker_client.invoke_endpoint_with_response_stream.return_value = mock_response_json

        streaming_request = self.request_body.copy()
        streaming_request["stream"] = True

        result = invoke_sagemaker_inference(
            request_body=streaming_request,
            endpoint_name=self.endpoint_name,
            sagemaker_client=mock_sagemaker_client,
        )

        self.assertIsInstance(result, SingleInferenceResult)

        self.assertIsNotNone(result._streaming_response)
        self.assertIsNone(result._nonstreaming_response)

        collected_results = list(result._streaming_response)

        mock_sagemaker_client.invoke_endpoint_with_response_stream.assert_called_once()
        self.assertEqual(collected_results, ["Hello ", "world"])

    def test_streaming_inference_empty_response(self):
        """Test streaming inference with empty response"""
        mock_sagemaker_client = MagicMock()

        mock_response_json = {
            "ResponseMetadata": {
                "RequestId": "test-request-id",
            },
            "Body": [],
        }

        mock_sagemaker_client.invoke_endpoint_with_response_stream.return_value = mock_response_json

        streaming_request = self.request_body.copy()
        streaming_request["stream"] = True

        result = invoke_sagemaker_inference(
            request_body=streaming_request,
            endpoint_name=self.endpoint_name,
            sagemaker_client=mock_sagemaker_client,
        )

        self.assertIsInstance(result, SingleInferenceResult)

        collected_results = list(result._streaming_response)

        self.assertEqual(collected_results, [])

    def test_invoke_sagemaker_inference_error_handling(self):
        """Test handling of errors"""
        mock_sagemaker_client = MagicMock()
        mock_sagemaker_client.invoke_endpoint.side_effect = RuntimeError("Unexpected error")

        with self.assertRaises(Exception) as context:
            invoke_sagemaker_inference(
                request_body=self.request_body,
                endpoint_name=self.endpoint_name,
                sagemaker_client=mock_sagemaker_client,
            )

    def test_valid_instance_types(self):
        # Test valid instance types for each model (based on SUPPORTED_SMI_CONFIGS)
        valid_test_cases = [
            (Model.NOVA_MICRO, "ml.g5.12xlarge"),
            (Model.NOVA_MICRO, "ml.g5.24xlarge"),
            (Model.NOVA_MICRO, "ml.g6.12xlarge"),
            (Model.NOVA_MICRO, "ml.g6.24xlarge"),
            (Model.NOVA_MICRO, "ml.g6.48xlarge"),
            (Model.NOVA_MICRO, "ml.p5.48xlarge"),
            (Model.NOVA_LITE, "ml.g6.12xlarge"),
            (Model.NOVA_LITE, "ml.g6.24xlarge"),
            (Model.NOVA_LITE, "ml.g6.48xlarge"),
            (Model.NOVA_LITE, "ml.p5.48xlarge"),
            (Model.NOVA_LITE_2, "ml.g6.48xlarge"),
            (Model.NOVA_LITE_2, "ml.p5.48xlarge"),
            (Model.NOVA_PRO, "ml.p5.48xlarge"),
        ]

        for model, instance_type in valid_test_cases:
            _validate_sagemaker_instance_type_for_model_deployment(instance_type, model)

    def test_invalid_instance_types(self):
        # Test invalid instance types for each model
        invalid_test_cases = [
            (Model.NOVA_MICRO, "ml.fake-instance"),
            (Model.NOVA_LITE, "ml.m5.2xlarge"),
            (Model.NOVA_LITE_2, "ml.m5.2xlarge"),
            (Model.NOVA_PRO, "ml.m5.2xlarge"),
        ]

        for model, instance_type in invalid_test_cases:
            with self.assertRaises(ValueError):
                _validate_sagemaker_instance_type_for_model_deployment(instance_type, model)

    @patch("amzn_nova_forge.util.sagemaker.boto3.client")
    def test_get_model_artifacts_mtrl_fallback_on_validation_exception(self, mock_boto_client):
        """SMTJServerless jobs fall through to AgentRFTJob on ValidationException."""
        import sys
        import types

        mock_sagemaker = MagicMock()
        mock_sagemaker.describe_training_job.side_effect = ClientError(
            error_response={"Error": {"Code": "ValidationException", "Message": "not found"}},
            operation_name="DescribeTrainingJob",
        )
        mock_boto_client.return_value = mock_sagemaker

        infra = MagicMock()
        infra.platform = Platform.SMTJServerless
        infra.sagemaker_client = mock_sagemaker

        mock_rft_job = MagicMock()
        mock_rft_job.s3_output_path = "s3://bucket/output/"
        mock_rft_job.output_model_package_arn = (
            "arn:aws:sagemaker:us-east-1:123456789012:model-package/grp/1"
        )

        mock_agent_module = types.ModuleType("sagemaker.train.agent_rft_job")
        mock_agent_cls = MagicMock()
        mock_agent_cls.get.return_value = mock_rft_job
        mock_agent_module.AgentRFTJob = mock_agent_cls

        with patch.dict("sys.modules", {"sagemaker.train.agent_rft_job": mock_agent_module}):
            result = get_model_artifacts(
                job_name="mtrl-job-123",
                infra=infra,
                region="us-east-1",
            )

        self.assertEqual(
            result.output_model_arn, "arn:aws:sagemaker:us-east-1:123456789012:model-package/grp/1"
        )
        self.assertEqual(result.output_s3_path, "s3://bucket/output/")
        self.assertIsNone(result.checkpoint_s3_path)

    @patch("amzn_nova_forge.util.sagemaker.boto3.client")
    def test_get_model_artifacts_mtrl_no_output_s3_path_param_needed(self, mock_boto_client):
        """output_s3_path parameter is optional for MTRL — fetched from AgentRFTJob."""
        import sys
        import types

        mock_sagemaker = MagicMock()
        mock_sagemaker.describe_training_job.side_effect = ClientError(
            error_response={"Error": {"Code": "ValidationException", "Message": "not found"}},
            operation_name="DescribeTrainingJob",
        )
        mock_boto_client.return_value = mock_sagemaker

        infra = MagicMock()
        infra.platform = Platform.SMTJServerless
        infra.sagemaker_client = mock_sagemaker

        mock_rft_job = MagicMock()
        mock_rft_job.s3_output_path = "s3://from-api/output/"
        mock_rft_job.output_model_package_arn = None

        mock_agent_module = types.ModuleType("sagemaker.train.agent_rft_job")
        mock_agent_cls = MagicMock()
        mock_agent_cls.get.return_value = mock_rft_job
        mock_agent_module.AgentRFTJob = mock_agent_cls

        with patch.dict("sys.modules", {"sagemaker.train.agent_rft_job": mock_agent_module}):
            result = get_model_artifacts(
                job_name="mtrl-job-456",
                infra=infra,
            )

        self.assertEqual(result.output_s3_path, "s3://from-api/output/")
        self.assertIsNone(result.output_model_arn)

    @patch("amzn_nova_forge.util.sagemaker.boto3.client")
    def test_get_model_artifacts_smtj_does_not_fallback_to_agent_rft(self, mock_boto_client):
        """Platform.SMTJ should re-raise ResourceNotFound, not fallback to AgentRFT."""
        mock_sagemaker = MagicMock()
        mock_sagemaker.describe_training_job.side_effect = ClientError(
            error_response={"Error": {"Code": "ResourceNotFound", "Message": "not found"}},
            operation_name="DescribeTrainingJob",
        )
        mock_boto_client.return_value = mock_sagemaker

        infra = MagicMock()
        infra.platform = Platform.SMTJ
        infra.sagemaker_client = mock_sagemaker

        with self.assertRaises(ClientError):
            get_model_artifacts(
                job_name="smtj-job-123",
                infra=infra,
                output_s3_path="s3://bucket/output/",
            )

    def test_model_artifacts_all_fields_optional(self):
        """ModelArtifacts can be constructed with no arguments."""
        artifacts = ModelArtifacts()
        self.assertIsNone(artifacts.checkpoint_s3_path)
        self.assertIsNone(artifacts.output_s3_path)
        self.assertIsNone(artifacts.output_model_arn)


def _mock_client_for_successful_creation(ic_name="my-ic-component", endpoint_name="my-endpoint"):
    mock_client = MagicMock()
    mock_client.describe_endpoint.return_value = {
        "EndpointName": endpoint_name,
        "EndpointStatus": "InService",
        "EndpointConfigName": "my-endpoint-config",
    }
    mock_client.describe_endpoint_config.return_value = {
        "ProductionVariants": [
            {
                "VariantName": "AllTraffic",
                "RoutingConfig": {"RoutingStrategy": "LEAST_OUTSTANDING_REQUESTS"},
            }
        ]
    }
    mock_client.create_inference_component.return_value = {
        "InferenceComponentArn": (
            f"arn:aws:sagemaker:us-east-1:123456789012:inference-component/{ic_name}"
        ),
    }
    return mock_client


class TestCreateInferenceComponentNonBlocking(unittest.TestCase):
    def test_returns_deployment_result(self):
        params = {**DEFAULT_IC_PARAMS}
        mock_client = _mock_client_for_successful_creation()

        with patch("amzn_nova_forge.util.sagemaker.time.sleep") as mock_sleep:
            result = create_inference_component(**params, sagemaker_client=mock_client)

        self.assertIsInstance(result, DeploymentResult)
        mock_client.describe_inference_component.assert_not_called()
        mock_sleep.assert_not_called()

    def test_returns_deployment_result_various_names(self):
        names = ["component-alpha", "ic_beta_123", "X", "a-very-long-component-name-here"]
        for name in names:
            params = {**DEFAULT_IC_PARAMS}
            params["inference_component_name"] = name
            mock_client = _mock_client_for_successful_creation(ic_name=name)

            with patch("amzn_nova_forge.util.sagemaker.time.sleep") as mock_sleep:
                result = create_inference_component(**params, sagemaker_client=mock_client)

            self.assertIsInstance(result, DeploymentResult)
            mock_client.describe_inference_component.assert_not_called()
            mock_sleep.assert_not_called()


class TestCreateInferenceComponentValidation(unittest.TestCase):
    def _assert_invalid_param_raises(self, param_name, value):
        params = {**DEFAULT_IC_PARAMS, param_name: value}
        mock_client = MagicMock()

        with self.assertRaises(ValueError) as ctx:
            create_inference_component(**params, sagemaker_client=mock_client)

        self.assertIn(param_name, str(ctx.exception))
        mock_client.create_inference_component.assert_not_called()

    def test_empty_string_params_raise(self):
        for param in (
            "inference_component_name",
            "endpoint_name",
            "variant_name",
            "model_name",
        ):
            with self.subTest(param=param):
                self._assert_invalid_param_raises(param, "")

    def test_none_params_raise(self):
        for param in ("inference_component_name", "endpoint_name", "model_name"):
            with self.subTest(param=param):
                self._assert_invalid_param_raises(param, None)


class TestCreateInferenceComponentEndpointValidation(unittest.TestCase):
    def test_non_existent_endpoint_raises(self):
        params = {**DEFAULT_IC_PARAMS}
        params["endpoint_name"] = "missing-endpoint"
        mock_client = MagicMock()
        mock_client.describe_endpoint.side_effect = ClientError(
            {"Error": {"Code": "ValidationException", "Message": "Could not find endpoint"}},
            "DescribeEndpoint",
        )

        with self.assertRaises(Exception) as ctx:
            create_inference_component(**params, sagemaker_client=mock_client)

        self.assertIn("not found", str(ctx.exception).lower())
        self.assertIn("missing-endpoint", str(ctx.exception))
        mock_client.create_inference_component.assert_not_called()

    def test_non_inservice_endpoint_raises(self):
        params = {**DEFAULT_IC_PARAMS}
        status = "Creating"
        mock_client = MagicMock()
        mock_client.describe_endpoint.return_value = {
            "EndpointName": params["endpoint_name"],
            "EndpointStatus": status,
        }

        with self.assertRaises(Exception) as ctx:
            create_inference_component(**params, sagemaker_client=mock_client)

        self.assertIn(status, str(ctx.exception))
        mock_client.create_inference_component.assert_not_called()


class TestCreateInferenceComponentEndpointInfo(unittest.TestCase):
    def test_endpoint_info_fields(self):
        params = {**DEFAULT_IC_PARAMS}
        mock_client = _mock_client_for_successful_creation(
            ic_name=params["inference_component_name"],
            endpoint_name=params["endpoint_name"],
        )

        result = create_inference_component(**params, sagemaker_client=mock_client)

        self.assertEqual(result.endpoint.platform, DeployPlatform.SAGEMAKER)
        self.assertEqual(result.endpoint.endpoint_name, params["endpoint_name"])
        self.assertIn(params["inference_component_name"], result.endpoint.uri)
        self.assertEqual(result.endpoint.model_artifact_path, params["model_name"])

    def test_endpoint_info_with_different_params(self):
        params = {**DEFAULT_IC_PARAMS}
        params["inference_component_name"] = "custom-ic-name"
        params["endpoint_name"] = "custom-endpoint"
        params["model_name"] = "custom-sagemaker-model"
        mock_client = _mock_client_for_successful_creation(
            ic_name="custom-ic-name",
            endpoint_name="custom-endpoint",
        )

        result = create_inference_component(**params, sagemaker_client=mock_client)

        self.assertEqual(result.endpoint.platform, DeployPlatform.SAGEMAKER)
        self.assertEqual(result.endpoint.endpoint_name, "custom-endpoint")
        self.assertIn("custom-ic-name", result.endpoint.uri)
        self.assertEqual(result.endpoint.model_artifact_path, "custom-sagemaker-model")


class TestMonitorInferenceComponentPolling(unittest.TestCase):
    @patch("amzn_nova_forge.util.sagemaker.time.sleep")
    def test_polling_terminates_on_terminal_state(self, mock_sleep):
        cases = [
            # (side_effect, expected_calls, expected_sleeps, terminal_status, should_raise)
            (
                [{"InferenceComponentName": "my-ic", "InferenceComponentStatus": "InService"}],
                1,
                0,
                "InService",
                False,
            ),
            (
                [
                    {"InferenceComponentName": "my-ic", "InferenceComponentStatus": "Creating"},
                    {"InferenceComponentName": "my-ic", "InferenceComponentStatus": "Creating"},
                    {"InferenceComponentName": "my-ic", "InferenceComponentStatus": "InService"},
                ],
                3,
                2,
                "InService",
                False,
            ),
            (
                [
                    {"InferenceComponentName": "my-ic", "InferenceComponentStatus": "Creating"},
                    {"InferenceComponentName": "my-ic", "InferenceComponentStatus": "Failed"},
                ],
                2,
                1,
                None,
                True,
            ),
        ]

        mock_client = MagicMock()

        for side_effect, expected_calls, expected_sleeps, terminal_status, should_raise in cases:
            with self.subTest(side_effect=side_effect):
                mock_client.reset_mock()
                mock_sleep.reset_mock()
                mock_client.describe_inference_component.side_effect = side_effect

                if should_raise:
                    with self.assertRaises(Exception) as ctx:
                        monitor_inference_component("my-ic", mock_client)
                    self.assertIn("my-ic", str(ctx.exception))
                else:
                    result = monitor_inference_component("my-ic", mock_client)
                    self.assertEqual(result, terminal_status)

                self.assertEqual(
                    mock_client.describe_inference_component.call_count, expected_calls
                )
                self.assertEqual(mock_sleep.call_count, expected_sleeps)


class TestInvokeInferenceComponentName(unittest.TestCase):
    def test_non_streaming_includes_inference_component_name(self):
        mock_client = MagicMock()
        mock_client.invoke_endpoint.return_value = {
            "Body": MagicMock(
                read=MagicMock(
                    return_value=json.dumps(
                        {"id": "test-id", "created": 1234567890, "choices": []}
                    ).encode()
                )
            ),
            "ResponseMetadata": {"RequestId": "test-request-id"},
        }

        invoke_sagemaker_inference(
            request_body={"inputs": "test"},
            endpoint_name="my-endpoint",
            sagemaker_client=mock_client,
            inference_component_name="my-ic",
        )

        call_kwargs = mock_client.invoke_endpoint.call_args[1]
        self.assertIn("InferenceComponentName", call_kwargs)
        self.assertEqual(call_kwargs["InferenceComponentName"], "my-ic")

    def test_streaming_includes_inference_component_name(self):
        mock_client = MagicMock()
        mock_client.invoke_endpoint_with_response_stream.return_value = {
            "Body": [],
            "ResponseMetadata": {"RequestId": "test-request-id"},
        }

        invoke_sagemaker_inference(
            request_body={"inputs": "test", "stream": True},
            endpoint_name="my-endpoint",
            sagemaker_client=mock_client,
            inference_component_name="my-ic",
        )

        call_kwargs = mock_client.invoke_endpoint_with_response_stream.call_args[1]
        self.assertIn("InferenceComponentName", call_kwargs)
        self.assertEqual(call_kwargs["InferenceComponentName"], "my-ic")


class TestCheckDeploymentStatusSagemaker(unittest.TestCase):
    @patch("amzn_nova_forge.util.sagemaker.boto3.client")
    def test_inference_component_status_returned(self, mock_boto_client):
        from amzn_nova_forge.util.sagemaker import check_sagemaker_deployment_status

        mock_sm_client = MagicMock()
        mock_sm_client.describe_inference_component.return_value = {
            "InferenceComponentName": "my-ic",
            "InferenceComponentStatus": "InService",
        }
        mock_boto_client.return_value = mock_sm_client

        arn = "arn:aws:sagemaker:us-east-1:123456789012:inference-component/my-ic"
        status = check_sagemaker_deployment_status(arn)

        self.assertEqual(status, "InService")
        mock_boto_client.assert_any_call("sagemaker", region_name=None)
        mock_sm_client.describe_inference_component.assert_called_once_with(
            InferenceComponentName="my-ic"
        )

    @patch("amzn_nova_forge.util.sagemaker.boto3.client")
    def test_inference_component_creating_status(self, mock_boto_client):
        from amzn_nova_forge.util.sagemaker import check_sagemaker_deployment_status

        mock_sm_client = MagicMock()
        mock_sm_client.describe_inference_component.return_value = {
            "InferenceComponentName": "ic-creating",
            "InferenceComponentStatus": "Creating",
        }
        mock_boto_client.return_value = mock_sm_client

        arn = "arn:aws:sagemaker:us-east-1:123456789012:inference-component/ic-creating"
        status = check_sagemaker_deployment_status(arn)

        self.assertEqual(status, "Creating")

    @patch("amzn_nova_forge.util.sagemaker.boto3.client")
    def test_inference_component_failed_status(self, mock_boto_client):
        from amzn_nova_forge.util.sagemaker import check_sagemaker_deployment_status

        mock_sm_client = MagicMock()
        mock_sm_client.describe_inference_component.return_value = {
            "InferenceComponentName": "ic-failed",
            "InferenceComponentStatus": "Failed",
        }
        mock_boto_client.return_value = mock_sm_client

        arn = "arn:aws:sagemaker:us-east-1:123456789012:inference-component/ic-failed"
        status = check_sagemaker_deployment_status(arn)

        self.assertEqual(status, "Failed")

    @patch("amzn_nova_forge.util.sagemaker.boto3.client")
    def test_deployment_result_status_property(self, mock_boto_client):
        mock_sm_client = MagicMock()
        mock_sm_client.describe_inference_component.return_value = {
            "InferenceComponentName": "my-ic",
            "InferenceComponentStatus": "InService",
        }
        mock_boto_client.return_value = mock_sm_client

        result = DeploymentResult(
            endpoint=EndpointInfo(
                platform=DeployPlatform.SAGEMAKER,
                endpoint_name="my-endpoint",
                uri="arn:aws:sagemaker:us-east-1:123456789012:inference-component/my-ic",
                model_artifact_path="s3://bucket/model",
            ),
            created_at=datetime.now(),
        )

        status = result.status
        self.assertEqual(status, "InService")


class TestCheckDeploymentStatusEndpoint(unittest.TestCase):
    @patch("amzn_nova_forge.util.sagemaker.boto3.client")
    def test_returns_endpoint_status_in_service(self, mock_boto_client):
        from amzn_nova_forge.util.sagemaker import check_sagemaker_deployment_status

        mock_sm_client = MagicMock()
        mock_sm_client.describe_endpoint.return_value = {"EndpointStatus": "InService"}
        mock_boto_client.return_value = mock_sm_client

        arn = "arn:aws:sagemaker:us-east-1:123456789012:endpoint/my-endpoint"
        status = check_sagemaker_deployment_status(arn)

        self.assertEqual(status, "InService")
        mock_boto_client.assert_any_call("sagemaker", region_name=None)
        mock_sm_client.describe_endpoint.assert_called_once_with(EndpointName="my-endpoint")

    @patch("amzn_nova_forge.util.sagemaker.boto3.client")
    def test_returns_endpoint_status_creating(self, mock_boto_client):
        from amzn_nova_forge.util.sagemaker import check_sagemaker_deployment_status

        mock_sm_client = MagicMock()
        mock_sm_client.describe_endpoint.return_value = {"EndpointStatus": "Creating"}
        mock_boto_client.return_value = mock_sm_client

        arn = "arn:aws:sagemaker:us-east-1:123456789012:endpoint/my-endpoint"
        status = check_sagemaker_deployment_status(arn)

        self.assertEqual(status, "Creating")


class TestValidateInferenceComponentResources(unittest.TestCase):
    def test_sufficient_resources_passes(self):
        config = InferenceComponentConfig(
            inference_component_name="test-ic",
            num_cpus=15,
            num_accelerators=4,
            min_memory_in_mb=25000,
        )
        validate_inference_component_resources(config, Model.NOVA_MICRO)

    def test_resources_below_minimums_raises_valueerror(self):
        config = InferenceComponentConfig(
            inference_component_name="test-ic",
            num_cpus=4,
            num_accelerators=1,
            min_memory_in_mb=8192,
        )
        with self.assertRaises(ValueError):
            validate_inference_component_resources(config, Model.NOVA_MICRO)

    def test_model_not_in_requirements_passes(self):
        self.assertNotIn(Model.NOVA_PRO, _IC_MIN_COMPUTE_REQUIREMENTS)

        config = InferenceComponentConfig(
            inference_component_name="test-ic",
            num_cpus=1,
            num_accelerators=0,
            min_memory_in_mb=512,
        )
        validate_inference_component_resources(config, Model.NOVA_PRO)


if __name__ == "__main__":
    unittest.main()
