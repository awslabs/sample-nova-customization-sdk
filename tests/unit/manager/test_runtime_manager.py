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
import io
import json
import os
import subprocess
import tempfile
import unittest
import zipfile
from unittest.mock import MagicMock, patch

from amzn_nova_forge.core.enums import TrainingMethod
from amzn_nova_forge.manager.runtime_manager import (
    DataPrepJobConfig,
    JobConfig,
    SMHPRuntimeManager,
    SMTJRuntimeManager,
)
from amzn_nova_forge.recipe.recipe_builder import HYPERPOD_RECIPE_PATH
from amzn_nova_forge.validation.validator import is_lambda_arn, validate_lambda_arn


class TestSMTJRuntimeManager(unittest.TestCase):
    def setUp(self):
        self.mock_role = "arn:aws:iam::123456789012:role/SageMakerExecutionRole"
        self.instance_type = "ml.m5.xlarge"
        self.instance_count = 1

    @patch.object(SMTJRuntimeManager, "setup", return_value=None)
    def _create_manager(self, mock_setup):
        manager = SMTJRuntimeManager(self.instance_type, self.instance_count)
        manager.execution_role = self.mock_role
        manager.sagemaker_client = MagicMock()
        manager.sagemaker_session = MagicMock()
        manager.region = "us-east-1"
        return manager

    @patch.object(SMTJRuntimeManager, "setup", return_value=None)
    def test_initialization(self, mock_setup):
        manager = SMTJRuntimeManager(self.instance_type, self.instance_count)
        manager.execution_role = self.mock_role
        manager.region = "us-east-1"

        self.assertEqual(manager.instance_type, self.instance_type)
        self.assertEqual(manager.instance_count, self.instance_count)
        self.assertEqual(manager.region, "us-east-1")

    @patch.object(SMTJRuntimeManager, "setup", return_value=None)
    def test_instance_count_setter(self, mock_setup):
        manager = SMTJRuntimeManager(self.instance_type, self.instance_count)
        self.assertEqual(manager.instance_count, 1)

        new_instance_count = 4
        manager.instance_count = new_instance_count

        self.assertEqual(manager.instance_count, new_instance_count)

    @patch("amzn_nova_forge.manager.runtime_manager.Session")
    @patch("amzn_nova_forge.manager.runtime_manager.get_execution_role")
    @patch("amzn_nova_forge.manager.runtime_manager.boto3.client")
    @patch("amzn_nova_forge.manager.runtime_manager.boto3.session.Session")
    def test_setup(
        self,
        mock_boto_session_class,
        mock_boto_client,
        mock_get_execution_role,
        mock_sagemaker_session_class,
    ):
        mock_boto_session = MagicMock()
        mock_boto_session.region_name = None
        mock_boto_session_class.return_value = mock_boto_session

        mock_client = MagicMock()
        mock_boto_client.return_value = mock_client

        mock_role = "arn:aws:iam::123456789012:role/SageMakerExecutionRole"
        mock_get_execution_role.return_value = mock_role

        mock_sagemaker_session = MagicMock()
        mock_sagemaker_session_class.return_value = mock_sagemaker_session

        manager = SMTJRuntimeManager("ml.m5.xlarge", 1)

        self.assertEqual(manager.region, "us-east-1")
        self.assertEqual(manager.sagemaker_client, mock_client)
        self.assertEqual(manager.execution_role, mock_role)
        self.assertEqual(manager.sagemaker_session, mock_sagemaker_session)

        mock_boto_session_class.assert_called_once()
        mock_boto_client.assert_called_once_with("sagemaker", region_name="us-east-1")
        mock_get_execution_role.assert_called_once_with(use_default=True)
        mock_sagemaker_session_class.assert_called_once_with(
            boto_session=mock_boto_session, sagemaker_client=mock_client
        )

    @patch("amzn_nova_forge.manager.runtime_manager.boto3.client")
    @patch("amzn_nova_forge.manager.runtime_manager.ModelTrainer")
    @patch.object(SMTJRuntimeManager, "setup", return_value=None)
    def test_execute_success(self, mock_setup, mock_model_trainer_cls, mock_boto_client):
        manager = self._create_manager()

        mock_model_trainer = MagicMock()
        mock_model_trainer.with_tensorboard_output_config.return_value = mock_model_trainer
        mock_model_trainer_cls.from_recipe.return_value = mock_model_trainer

        manager.sagemaker_client.list_training_jobs.return_value = {
            "TrainingJobSummaries": [{"TrainingJobName": "test-job-suffix"}]
        }

        job_config = JobConfig(
            job_name="test-job",
            image_uri="123456789012.dkr.ecr.us-east-1.amazonaws.com/my-image:latest",
            recipe_path="/path/to/recipe",
            output_s3_path="s3://output-bucket/output",
            data_s3_path="s3://input-bucket/data",
            input_s3_data_type="data_type",
        )

        job_id = manager.execute(job_config)

        mock_model_trainer_cls.from_recipe.assert_called_once()
        mock_model_trainer.train.assert_called_once_with(wait=False, logs=False)
        self.assertEqual(job_id, "test-job-suffix")

    @patch("amzn_nova_forge.manager.runtime_manager.boto3.client")
    @patch("amzn_nova_forge.manager.runtime_manager.ModelTrainer")
    @patch.object(SMTJRuntimeManager, "setup", return_value=None)
    def test_execute_without_optional_params(
        self, mock_setup, mock_model_trainer_cls, mock_boto_client
    ):
        manager = self._create_manager()

        mock_model_trainer = MagicMock()
        mock_model_trainer.with_tensorboard_output_config.return_value = mock_model_trainer
        mock_model_trainer_cls.from_recipe.return_value = mock_model_trainer

        manager.sagemaker_client.list_training_jobs.return_value = {
            "TrainingJobSummaries": [{"TrainingJobName": "test-job-suffix"}]
        }

        job_config = JobConfig(
            job_name="test-job",
            image_uri="123456789012.dkr.ecr.us-east-1.amazonaws.com/my-image:latest",
            recipe_path="/path/to/recipe",
            output_s3_path="s3://output-bucket/output",
        )

        job_id = manager.execute(job_config)

        mock_model_trainer_cls.from_recipe.assert_called_once()
        call_kwargs = mock_model_trainer_cls.from_recipe.call_args.kwargs
        self.assertNotIn("input_data_config", call_kwargs)
        mock_model_trainer.train.assert_called_once_with(wait=False, logs=False)
        self.assertEqual(job_id, "test-job-suffix")

    @patch("amzn_nova_forge.manager.runtime_manager.boto3.client")
    @patch("amzn_nova_forge.manager.runtime_manager.ModelTrainer")
    @patch.object(SMTJRuntimeManager, "setup", return_value=None)
    def test_execute_eval_with_model_package_arn_passes_model_package_config(
        self, mock_setup, mock_model_trainer_cls, mock_boto_client
    ):
        """Eval job with a model package ARN should pass ModelPackageConfig to from_recipe."""
        manager = self._create_manager()

        mock_model_trainer = MagicMock()
        mock_model_trainer.with_tensorboard_output_config.return_value = mock_model_trainer
        mock_model_trainer_cls.from_recipe.return_value = mock_model_trainer

        manager.sagemaker_client.list_training_jobs.return_value = {
            "TrainingJobSummaries": [{"TrainingJobName": "eval-job-suffix"}]
        }
        manager.sagemaker_client.describe_model_package_group.return_value = {
            "ModelPackageGroupArn": "arn:aws:sagemaker:us-east-1:123456789012:model-package-group/my-group"
        }

        model_package_arn = "arn:aws:sagemaker:us-east-1:123456789012:model-package/my-group/3"
        job_config = JobConfig(
            job_name="eval-job",
            image_uri="123456789012.dkr.ecr.us-east-1.amazonaws.com/eval-image:latest",
            recipe_path="/path/to/recipe",
            output_s3_path="s3://output-bucket/eval-output",
            method=TrainingMethod.EVALUATION,
            model_name_or_path=model_package_arn,
        )

        job_id = manager.execute(job_config)

        call_kwargs = mock_model_trainer_cls.from_recipe.call_args.kwargs
        self.assertIn("model_package_config", call_kwargs)
        manager.sagemaker_client.describe_model_package_group.assert_called_once_with(
            ModelPackageGroupName="my-group"
        )
        self.assertEqual(job_id, "eval-job-suffix")

    @patch("amzn_nova_forge.manager.runtime_manager.boto3.client")
    @patch("amzn_nova_forge.manager.runtime_manager.ModelTrainer")
    @patch.object(SMTJRuntimeManager, "setup", return_value=None)
    def test_execute_eval_with_s3_path_omits_model_package_config(
        self, mock_setup, mock_model_trainer_cls, mock_boto_client
    ):
        """Eval job with an S3 path should not pass ModelPackageConfig."""
        manager = self._create_manager()

        mock_model_trainer = MagicMock()
        mock_model_trainer.with_tensorboard_output_config.return_value = mock_model_trainer
        mock_model_trainer_cls.from_recipe.return_value = mock_model_trainer

        manager.sagemaker_client.list_training_jobs.return_value = {
            "TrainingJobSummaries": [{"TrainingJobName": "eval-job-suffix"}]
        }

        job_config = JobConfig(
            job_name="eval-job",
            image_uri="123456789012.dkr.ecr.us-east-1.amazonaws.com/eval-image:latest",
            recipe_path="/path/to/recipe",
            output_s3_path="s3://output-bucket/eval-output",
            method=TrainingMethod.EVALUATION,
            model_name_or_path="s3://my-bucket/checkpoint/step_10",
        )

        job_id = manager.execute(job_config)

        call_kwargs = mock_model_trainer_cls.from_recipe.call_args.kwargs
        self.assertNotIn("model_package_config", call_kwargs)
        self.assertEqual(job_id, "eval-job-suffix")

    @patch("amzn_nova_forge.manager.runtime_manager.boto3.client")
    @patch("amzn_nova_forge.manager.runtime_manager.ModelTrainer")
    @patch.object(SMTJRuntimeManager, "setup", return_value=None)
    def test_execute_eval_with_nonexistent_model_package_group_raises(
        self, mock_setup, mock_model_trainer_cls, mock_boto_client
    ):
        """Eval job with a model package ARN pointing to a nonexistent group should raise."""
        from botocore.exceptions import ClientError

        manager = self._create_manager()

        mock_model_trainer = MagicMock()
        mock_model_trainer.with_tensorboard_output_config.return_value = mock_model_trainer
        mock_model_trainer_cls.from_recipe.return_value = mock_model_trainer

        manager.sagemaker_client.describe_model_package_group.side_effect = ClientError(
            {"Error": {"Code": "ValidationException", "Message": "not found"}},
            "DescribeModelPackageGroup",
        )

        model_package_arn = "arn:aws:sagemaker:us-east-1:123456789012:model-package/missing-group/1"
        job_config = JobConfig(
            job_name="eval-job",
            image_uri="123456789012.dkr.ecr.us-east-1.amazonaws.com/eval-image:latest",
            recipe_path="/path/to/recipe",
            output_s3_path="s3://output-bucket/eval-output",
            method=TrainingMethod.EVALUATION,
            model_name_or_path=model_package_arn,
        )

        with self.assertRaises(ValueError) as ctx:
            manager.execute(job_config)

        self.assertIn("does not exist", str(ctx.exception))

    @patch.object(SMTJRuntimeManager, "setup", return_value=None)
    def test_cleanup_success(self, mock_setup):
        manager = self._create_manager()
        mock_client = manager.sagemaker_client

        manager.cleanup("test-job")

        mock_client.stop_training_job.assert_called_once_with(TrainingJobName="test-job")
        mock_client.close.assert_called_once()

    @patch.object(SMTJRuntimeManager, "setup", return_value=None)
    def test_cleanup_handles_error(self, mock_setup):
        manager = self._create_manager()
        mock_client = manager.sagemaker_client
        mock_client.stop_training_job.side_effect = Exception("Cleanup failed")

        with self.assertRaises(Exception) as context:
            manager.cleanup("test-job")

        self.assertEqual(str(context.exception), "Cleanup failed")
        mock_client.stop_training_job.assert_called_once_with(TrainingJobName="test-job")

    @patch("amzn_nova_forge.manager.runtime_manager.ModelTrainer")
    @patch.object(SMTJRuntimeManager, "setup", return_value=None)
    def test_execute_handles_error(self, mock_setup, mock_model_trainer_cls):
        manager = self._create_manager()

        mock_model_trainer = MagicMock()
        mock_model_trainer.with_tensorboard_output_config.return_value = mock_model_trainer
        mock_model_trainer.train.side_effect = Exception("Training failed")
        mock_model_trainer_cls.from_recipe.return_value = mock_model_trainer

        job_config = JobConfig(
            job_name="test-job",
            image_uri="123456789012.dkr.ecr.us-east-1.amazonaws.com/my-image:latest",
            recipe_path="/path/to/recipe",
            output_s3_path="s3://output-bucket/output",
            data_s3_path="s3://input-bucket/data",
            input_s3_data_type="data_type",
        )

        with self.assertRaises(Exception) as context:
            manager.execute(job_config)

        self.assertEqual(str(context.exception), "Training failed")
        mock_model_trainer_cls.from_recipe.assert_called_once()
        mock_model_trainer.train.assert_called_once()

    def test_is_lambda_arn_true_for_valid_arn(self):
        self.assertTrue(is_lambda_arn("arn:aws:lambda:us-east-1:123456789012:function:my-fn"))

    def test_is_lambda_arn_false_for_file_path(self):
        self.assertFalse(is_lambda_arn("reward.py"))
        self.assertFalse(is_lambda_arn("/path/to/reward_fn.py"))

    def test_validate_lambda_arn_passes_for_valid_arn(self):
        validate_lambda_arn("arn:aws:lambda:us-east-1:123456789012:function:my-fn")

    def test_validate_lambda_arn_raises_for_invalid(self):
        with self.assertRaises(ValueError) as ctx:
            validate_lambda_arn("not-an-arn")
        self.assertIn("not a valid Lambda function ARN", str(ctx.exception))

    @patch.object(SMTJRuntimeManager, "setup", return_value=None)
    def test_rft_lambda_arn_set_immediately_when_arn_passed(self, mock_setup):
        arn = "arn:aws:lambda:us-east-1:123456789012:function:my-fn"
        mgr = SMTJRuntimeManager("ml.m5.xlarge", 1, rft_lambda=arn)
        self.assertEqual(mgr.rft_lambda_arn, arn)
        self.assertEqual(mgr.rft_lambda, arn)

    @patch.object(SMTJRuntimeManager, "setup", return_value=None)
    def test_rft_lambda_arn_none_when_file_path_passed(self, mock_setup):
        mgr = SMTJRuntimeManager("ml.m5.xlarge", 1, rft_lambda="reward.py")
        self.assertIsNone(mgr.rft_lambda_arn)
        self.assertEqual(mgr.rft_lambda, "reward.py")

    @patch.object(SMTJRuntimeManager, "setup", return_value=None)
    def test_rft_lambda_arn_none_when_not_set(self, mock_setup):
        mgr = SMTJRuntimeManager("ml.m5.xlarge", 1)
        self.assertIsNone(mgr.rft_lambda_arn)
        self.assertIsNone(mgr.rft_lambda)

    @patch.object(SMTJRuntimeManager, "setup", return_value=None)
    def test_rft_lambda_arn_setter(self, mock_setup):
        arn = "arn:aws:lambda:us-east-1:123456789012:function:my-fn"
        mgr = SMTJRuntimeManager("ml.m5.xlarge", 1)
        mgr.rft_lambda_arn = arn
        self.assertEqual(mgr.rft_lambda_arn, arn)

    def _mock_lambda_client(self, exists=False):
        arn = "arn:aws:lambda:us-east-1:123456789012:function:my-fn"
        client = MagicMock()
        # Make exceptions.ResourceNotFoundException a real exception class so
        # `except lambda_client.exceptions.ResourceNotFoundException` works
        client.exceptions.ResourceNotFoundException = type(
            "ResourceNotFoundException", (Exception,), {}
        )
        if exists:
            client.get_function.return_value = {"Configuration": {"FunctionArn": arn}}
            client.update_function_code.return_value = {"FunctionArn": arn}
        else:
            client.get_function.side_effect = client.exceptions.ResourceNotFoundException()
            client.create_function.return_value = {"FunctionArn": arn}
        client.get_waiter.return_value = MagicMock()
        return client

    @patch("amzn_nova_forge.manager.runtime_manager.boto3.client")
    def test_deploy_lambda_creates_new_function(self, mock_boto_client):
        mock_lambda = self._mock_lambda_client(exists=False)
        mock_boto_client.return_value = mock_lambda

        with tempfile.NamedTemporaryFile(suffix=".py", delete=False) as f:
            f.write(b"def lambda_handler(event, context): return {'reward': 1.0}")
            src = f.name

        try:
            mgr = self._create_manager()
            mgr.rft_lambda = src
            returned_arn = mgr.deploy_lambda(lambda_name="SageMaker-my-reward")

            mock_lambda.create_function.assert_called_once()
            call_kwargs = mock_lambda.create_function.call_args.kwargs
            self.assertEqual(call_kwargs["FunctionName"], "SageMaker-my-reward")
            self.assertEqual(call_kwargs["Role"], self.mock_role)
            self.assertEqual(returned_arn, "arn:aws:lambda:us-east-1:123456789012:function:my-fn")
            self.assertEqual(mgr.rft_lambda_arn, returned_arn)
        finally:
            os.unlink(src)

    @patch("amzn_nova_forge.manager.runtime_manager.boto3.client")
    def test_deploy_lambda_updates_existing_function(self, mock_boto_client):
        mock_lambda = self._mock_lambda_client(exists=True)
        mock_boto_client.return_value = mock_lambda

        with tempfile.NamedTemporaryFile(suffix=".py", delete=False) as f:
            f.write(b"def lambda_handler(event, context): return {'reward': 1.0}")
            src = f.name

        try:
            mgr = self._create_manager()
            mgr.rft_lambda = src
            mgr.deploy_lambda(lambda_name="SageMaker-my-reward")

            mock_lambda.update_function_code.assert_called_once()
            mock_lambda.create_function.assert_not_called()
        finally:
            os.unlink(src)

    @patch("amzn_nova_forge.manager.runtime_manager.boto3.client")
    def test_deploy_lambda_derives_name_from_filename(self, mock_boto_client):
        mock_lambda = self._mock_lambda_client(exists=False)
        mock_boto_client.return_value = mock_lambda

        with tempfile.NamedTemporaryFile(suffix=".py", prefix="my_reward_fn_", delete=False) as f:
            f.write(b"def lambda_handler(event, context): return {'reward': 1.0}")
            src = f.name

        try:
            mgr = self._create_manager()
            mgr.rft_lambda = src
            mgr.deploy_lambda()

            call_kwargs = mock_lambda.create_function.call_args.kwargs
            self.assertNotIn("_", call_kwargs["FunctionName"])
            self.assertNotIn(".py", call_kwargs["FunctionName"])
        finally:
            os.unlink(src)

    @patch("amzn_nova_forge.manager.runtime_manager.boto3.client")
    def test_deploy_lambda_packages_file_as_zip(self, mock_boto_client):
        mock_lambda = self._mock_lambda_client(exists=False)
        mock_boto_client.return_value = mock_lambda

        with tempfile.NamedTemporaryFile(suffix=".py", delete=False) as f:
            f.write(b"def lambda_handler(event, context): return {'reward': 1.0}")
            src = f.name

        try:
            mgr = self._create_manager()
            mgr.rft_lambda = src
            mgr.deploy_lambda(lambda_name="SageMaker-reward")

            zip_bytes = mock_lambda.create_function.call_args.kwargs["Code"]["ZipFile"]
            with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
                self.assertIn("lambda_function.py", zf.namelist())
        finally:
            os.unlink(src)

    @patch("amzn_nova_forge.manager.runtime_manager.boto3.client")
    def test_deploy_lambda_explicit_role_overrides_manager_role(self, mock_boto_client):
        mock_lambda = self._mock_lambda_client(exists=False)
        mock_boto_client.return_value = mock_lambda

        with tempfile.NamedTemporaryFile(suffix=".py", delete=False) as f:
            f.write(b"def lambda_handler(event, context): return {'reward': 1.0}")
            src = f.name

        try:
            mgr = self._create_manager()
            mgr.rft_lambda = src
            override_role = "arn:aws:iam::123456789012:role/OtherRole"
            mgr.deploy_lambda(lambda_name="SageMaker-reward", execution_role_arn=override_role)
            self.assertEqual(mock_lambda.create_function.call_args.kwargs["Role"], override_role)
        finally:
            os.unlink(src)

    def test_deploy_lambda_raises_when_no_source_resolvable(self):
        mgr = self._create_manager()
        mgr.rft_lambda = None
        with self.assertRaises(ValueError) as ctx:
            mgr.deploy_lambda()
        self.assertIn("rft_lambda must be set", str(ctx.exception))

    def test_deploy_lambda_raises_when_rft_lambda_is_arn(self):
        arn = "arn:aws:lambda:us-east-1:123456789012:function:my-fn"
        mgr = self._create_manager()
        mgr.rft_lambda = arn
        with self.assertRaises(ValueError) as ctx:
            mgr.deploy_lambda()
        self.assertIn("already a deployed Lambda ARN", str(ctx.exception))

    def test_deploy_lambda_raises_when_file_not_found(self):
        mgr = self._create_manager()
        mgr.rft_lambda = "/nonexistent/reward.py"
        with self.assertRaises(ValueError) as ctx:
            mgr.deploy_lambda(lambda_name="SageMaker-reward")
        self.assertIn("file not found", str(ctx.exception))

    def test_deploy_lambda_raises_when_no_execution_role(self):
        with tempfile.NamedTemporaryFile(suffix=".py", delete=False) as f:
            f.write(b"def lambda_handler(event, context): return {'reward': 1.0}")
            src = f.name

        try:
            mgr = self._create_manager()
            mgr.execution_role = None
            mgr.rft_lambda = src
            with self.assertRaises(ValueError) as ctx:
                mgr.deploy_lambda(lambda_name="SageMaker-reward")
            self.assertIn("execution_role_arn", str(ctx.exception))
        finally:
            os.unlink(src)

    @patch("amzn_nova_forge.validation.validator.verify_rft_lambda")
    @patch("amzn_nova_forge.manager.runtime_manager.boto3.client")
    def test_validate_lambda_uses_rft_lambda_arn_property(self, mock_boto_client, mock_verify):
        arn = "arn:aws:lambda:us-east-1:123456789012:function:my-fn"
        mgr = self._create_manager()
        mgr.rft_lambda_arn = arn
        mgr.validate_lambda(data_s3_path="s3://bucket/data.jsonl")

        mock_verify.assert_called_once()
        call_kwargs = mock_verify.call_args.kwargs
        self.assertEqual(call_kwargs["lambda_arn"], arn)
        self.assertEqual(call_kwargs["data_s3_path"], "s3://bucket/data.jsonl")

    @patch("amzn_nova_forge.validation.validator.verify_rft_lambda")
    @patch("amzn_nova_forge.manager.runtime_manager.boto3.client")
    def test_validate_lambda_uses_rft_lambda_when_arn(self, mock_boto_client, mock_verify):
        arn = "arn:aws:lambda:us-east-1:123456789012:function:my-fn"
        # Pass ARN via constructor so rft_lambda_arn is resolved immediately
        with patch.object(SMTJRuntimeManager, "setup", return_value=None):
            mgr = SMTJRuntimeManager("ml.m5.xlarge", 1, rft_lambda=arn)
            mgr.region = "us-east-1"
        mgr.validate_lambda(data_s3_path="s3://bucket/data.jsonl")

        self.assertEqual(mock_verify.call_args.kwargs["lambda_arn"], arn)

    @patch("amzn_nova_forge.validation.validator.verify_rft_lambda")
    @patch("amzn_nova_forge.manager.runtime_manager.boto3.client")
    def test_validate_lambda_passes_sample_count(self, mock_boto_client, mock_verify):
        arn = "arn:aws:lambda:us-east-1:123456789012:function:my-fn"
        mgr = self._create_manager()
        mgr.rft_lambda_arn = arn
        mgr.validate_lambda(data_s3_path="s3://bucket/data.jsonl", validation_samples=5)

        self.assertEqual(mock_verify.call_args.kwargs["sample_count"], 5)

    @patch("amzn_nova_forge.manager.runtime_manager.verify_reward_function")
    @patch("amzn_nova_forge.manager.runtime_manager.boto3.client")
    def test_validate_lambda_local_source_calls_verify_reward_function(
        self, mock_boto_client, mock_verify_reward
    ):
        mock_s3 = MagicMock()
        mock_s3.get_object.return_value = {
            "Body": MagicMock(
                iter_lines=lambda: [b'{"id": "1", "messages": [{"role": "user", "content": "hi"}]}']
            )
        }
        mock_boto_client.return_value = mock_s3

        with tempfile.NamedTemporaryFile(suffix=".py", delete=False) as f:
            f.write(b"def lambda_handler(event, context): return {'reward': 1.0}")
            src = f.name

        try:
            mgr = self._create_manager()
            mgr.rft_lambda = src
            mgr.validate_lambda(data_s3_path="s3://bucket/data.jsonl")

            mock_verify_reward.assert_called_once()
            self.assertEqual(mock_verify_reward.call_args.kwargs["reward_function"], src)
        finally:
            os.unlink(src)

    def test_validate_lambda_raises_when_nothing_resolvable(self):
        mgr = self._create_manager()
        mgr.rft_lambda = None
        with self.assertRaises(ValueError) as ctx:
            mgr.validate_lambda(data_s3_path="s3://bucket/data.jsonl")
        self.assertIn("Either lambda_arn or lambda_source must be provided", str(ctx.exception))

    @patch("amzn_nova_forge.manager.runtime_manager.verify_reward_function")
    @patch("amzn_nova_forge.manager.runtime_manager.boto3.client")
    def test_validate_lambda_local_raises_on_invalid_s3_path(
        self, mock_boto_client, mock_verify_reward
    ):
        with tempfile.NamedTemporaryFile(suffix=".py", delete=False) as f:
            f.write(b"def lambda_handler(event, context): return {'reward': 1.0}")
            src = f.name

        try:
            mgr = self._create_manager()
            mgr.rft_lambda = src
            with self.assertRaises(ValueError) as ctx:
                mgr.validate_lambda(data_s3_path="not-an-s3-path")
            self.assertIn("Invalid S3 path", str(ctx.exception))
        finally:
            os.unlink(src)

    @patch("amzn_nova_forge.manager.runtime_manager.verify_reward_function")
    @patch("amzn_nova_forge.manager.runtime_manager.boto3.client")
    def test_validate_lambda_local_raises_on_s3_read_failure(
        self, mock_boto_client, mock_verify_reward
    ):
        mock_s3 = MagicMock()
        mock_s3.get_object.side_effect = Exception("Access Denied")
        mock_boto_client.return_value = mock_s3

        with tempfile.NamedTemporaryFile(suffix=".py", delete=False) as f:
            f.write(b"def lambda_handler(event, context): return {'reward': 1.0}")
            src = f.name

        try:
            mgr = self._create_manager()
            mgr.rft_lambda = src
            with self.assertRaises(ValueError) as ctx:
                mgr.validate_lambda(data_s3_path="s3://bucket/data.jsonl")
            self.assertIn("Failed to read samples", str(ctx.exception))
        finally:
            os.unlink(src)

    @patch("amzn_nova_forge.manager.runtime_manager.boto3.client")
    @patch("amzn_nova_forge.manager.runtime_manager.ModelTrainer")
    @patch.object(SMTJRuntimeManager, "setup", return_value=None)
    def test_execute_passes_environment_to_trainer(
        self, mock_setup, mock_model_trainer_cls, mock_boto_client
    ):
        """When job_config.environment is set, it must be passed to ModelTrainer.from_recipe."""
        manager = self._create_manager()

        mock_model_trainer = MagicMock()
        mock_model_trainer.with_tensorboard_output_config.return_value = mock_model_trainer
        mock_model_trainer_cls.from_recipe.return_value = mock_model_trainer
        manager.sagemaker_client.list_training_jobs.return_value = {
            "TrainingJobSummaries": [{"TrainingJobName": "test-job-env"}]
        }

        env = {"HF_TOKEN": "hf_xxx", "HF_HUB_DOWNLOAD_TIMEOUT": "300"}
        job_config = JobConfig(
            job_name="test-job",
            image_uri="123456789012.dkr.ecr.us-east-1.amazonaws.com/my-image:latest",
            recipe_path="/path/to/recipe",
            output_s3_path="s3://output-bucket/output",
            data_s3_path="s3://input-bucket/data",
            input_s3_data_type="S3Prefix",
            environment=env,
        )

        manager.execute(job_config)

        call_kwargs = mock_model_trainer_cls.from_recipe.call_args.kwargs
        self.assertEqual(call_kwargs["environment"], env)


class TestSMTJValidationDataset(unittest.TestCase):
    """Tests for SFT validation dataset support in SMTJRuntimeManager."""

    def setUp(self):
        self.mock_role = "arn:aws:iam::123456789012:role/SageMakerExecutionRole"
        self.instance_type = "ml.m5.xlarge"
        self.instance_count = 1

    @patch.object(SMTJRuntimeManager, "setup", return_value=None)
    def _create_manager(self, mock_setup):
        manager = SMTJRuntimeManager(self.instance_type, self.instance_count)
        manager.execution_role = self.mock_role
        manager.sagemaker_client = MagicMock()
        manager.sagemaker_session = MagicMock()
        manager.region = "us-east-1"
        return manager

    @patch("amzn_nova_forge.manager.runtime_manager.InputData")
    @patch("amzn_nova_forge.manager.runtime_manager.boto3.client")
    @patch("amzn_nova_forge.manager.runtime_manager.ModelTrainer")
    @patch.object(SMTJRuntimeManager, "setup", return_value=None)
    def test_smtj_sft_with_validation_data(
        self, mock_setup, mock_model_trainer_cls, mock_boto_client, mock_input_data
    ):
        manager = self._create_manager()

        mock_model_trainer = MagicMock()
        mock_model_trainer.with_tensorboard_output_config.return_value = mock_model_trainer
        mock_model_trainer_cls.from_recipe.return_value = mock_model_trainer

        manager.sagemaker_client.list_training_jobs.return_value = {
            "TrainingJobSummaries": [{"TrainingJobName": "test-job-suffix"}]
        }

        job_config = JobConfig(
            job_name="test-job",
            image_uri="123456789012.dkr.ecr.us-east-1.amazonaws.com/my-image:latest",
            recipe_path="/path/to/recipe",
            output_s3_path="s3://output-bucket/output",
            data_s3_path="s3://input-bucket/train.jsonl",
            validation_data_s3_path="s3://input-bucket/val.jsonl",
            input_s3_data_type="S3Prefix",
        )

        manager.execute(job_config)

        # InputData should be called twice: once for "train", once for "validation"
        self.assertEqual(mock_input_data.call_count, 2)
        channel_names = [call.kwargs["channel_name"] for call in mock_input_data.call_args_list]
        self.assertIn("train", channel_names)
        self.assertIn("validation", channel_names)

        call_kwargs = mock_model_trainer_cls.from_recipe.call_args.kwargs
        self.assertEqual(len(call_kwargs["input_data_config"]), 2)

    @patch("amzn_nova_forge.manager.runtime_manager.InputData")
    @patch("amzn_nova_forge.manager.runtime_manager.boto3.client")
    @patch("amzn_nova_forge.manager.runtime_manager.ModelTrainer")
    @patch.object(SMTJRuntimeManager, "setup", return_value=None)
    def test_smtj_sft_without_validation_data(
        self, mock_setup, mock_model_trainer_cls, mock_boto_client, mock_input_data
    ):
        manager = self._create_manager()

        mock_model_trainer = MagicMock()
        mock_model_trainer.with_tensorboard_output_config.return_value = mock_model_trainer
        mock_model_trainer_cls.from_recipe.return_value = mock_model_trainer

        manager.sagemaker_client.list_training_jobs.return_value = {
            "TrainingJobSummaries": [{"TrainingJobName": "test-job-suffix"}]
        }

        job_config = JobConfig(
            job_name="test-job",
            image_uri="123456789012.dkr.ecr.us-east-1.amazonaws.com/my-image:latest",
            recipe_path="/path/to/recipe",
            output_s3_path="s3://output-bucket/output",
            data_s3_path="s3://input-bucket/train.jsonl",
            input_s3_data_type="S3Prefix",
        )

        manager.execute(job_config)

        # InputData should be called only once for "train"
        self.assertEqual(mock_input_data.call_count, 1)
        self.assertEqual(mock_input_data.call_args.kwargs["channel_name"], "train")

        call_kwargs = mock_model_trainer_cls.from_recipe.call_args.kwargs
        self.assertEqual(len(call_kwargs["input_data_config"]), 1)

    @patch("amzn_nova_forge.manager.runtime_manager.InputData")
    @patch("amzn_nova_forge.manager.runtime_manager.boto3.client")
    @patch("amzn_nova_forge.manager.runtime_manager.ModelTrainer")
    @patch.object(SMTJRuntimeManager, "setup", return_value=None)
    def test_smtj_hyperparameters_passed_to_model_trainer(
        self, mock_setup, mock_model_trainer_cls, mock_boto_client, mock_input_data
    ):
        manager = self._create_manager()

        mock_model_trainer = MagicMock()
        mock_model_trainer.with_tensorboard_output_config.return_value = mock_model_trainer
        mock_model_trainer_cls.from_recipe.return_value = mock_model_trainer

        manager.sagemaker_client.list_training_jobs.return_value = {
            "TrainingJobSummaries": [{"TrainingJobName": "test-job-suffix"}]
        }

        job_config = JobConfig(
            job_name="test-job",
            image_uri="123456789012.dkr.ecr.us-east-1.amazonaws.com/my-image:latest",
            recipe_path="/path/to/recipe",
            output_s3_path="s3://output-bucket/output",
            data_s3_path="s3://input-bucket/train.jsonl",
            input_s3_data_type="S3Prefix",
            trainer_config_hyperparameters={"val_check_interval": "500"},
        )

        manager.execute(job_config)

        call_kwargs = mock_model_trainer_cls.from_recipe.call_args.kwargs
        self.assertIn("hyperparameters", call_kwargs)
        self.assertEqual(call_kwargs["hyperparameters"]["val_check_interval"], "500")

    @patch("amzn_nova_forge.manager.runtime_manager.InputData")
    @patch("amzn_nova_forge.manager.runtime_manager.boto3.client")
    @patch("amzn_nova_forge.manager.runtime_manager.ModelTrainer")
    @patch.object(SMTJRuntimeManager, "setup", return_value=None)
    def test_smtj_no_hyperparameters_when_none(
        self, mock_setup, mock_model_trainer_cls, mock_boto_client, mock_input_data
    ):
        manager = self._create_manager()

        mock_model_trainer = MagicMock()
        mock_model_trainer.with_tensorboard_output_config.return_value = mock_model_trainer
        mock_model_trainer_cls.from_recipe.return_value = mock_model_trainer

        manager.sagemaker_client.list_training_jobs.return_value = {
            "TrainingJobSummaries": [{"TrainingJobName": "test-job-suffix"}]
        }

        job_config = JobConfig(
            job_name="test-job",
            image_uri="123456789012.dkr.ecr.us-east-1.amazonaws.com/my-image:latest",
            recipe_path="/path/to/recipe",
            output_s3_path="s3://output-bucket/output",
            data_s3_path="s3://input-bucket/train.jsonl",
            input_s3_data_type="S3Prefix",
            trainer_config_hyperparameters=None,
        )

        manager.execute(job_config)

        call_kwargs = mock_model_trainer_cls.from_recipe.call_args.kwargs
        self.assertNotIn("hyperparameters", call_kwargs)


class TestSMTJRuntimeManagerDataPrepDelegation(unittest.TestCase):
    """SMTJRuntimeManager.set_mode(DATA_PREP) flips execute/cleanup to the delegate."""

    @patch.object(SMTJRuntimeManager, "setup", return_value=None)
    @patch("amzn_nova_forge.manager.runtime_manager.SMTJDataPrepRuntimeManager")
    def test_set_mode_data_prep_creates_delegate(self, mock_delegate_cls, mock_setup):
        from amzn_nova_forge.manager.runtime_manager import SMTJRuntimeMode

        mock_delegate = MagicMock()
        mock_delegate_cls.return_value = mock_delegate

        mgr = SMTJRuntimeManager(
            instance_type="ml.m5.2xlarge",
            instance_count=1,
        )
        self.assertIsNone(mgr._data_prep_delegate)
        self.assertEqual(mgr._mode, SMTJRuntimeMode.TRAINING)

        mgr.set_mode(SMTJRuntimeMode.DATA_PREP)

        mock_delegate_cls.assert_called_once()
        call_kwargs = mock_delegate_cls.call_args.kwargs
        self.assertEqual(call_kwargs["instance_type"], "ml.m5.2xlarge")
        self.assertEqual(call_kwargs["instance_count"], 1)
        self.assertIs(mgr._data_prep_delegate, mock_delegate)
        self.assertEqual(mgr._mode, SMTJRuntimeMode.DATA_PREP)

    @patch.object(SMTJRuntimeManager, "setup", return_value=None)
    @patch("amzn_nova_forge.manager.runtime_manager.SMTJDataPrepRuntimeManager")
    def test_set_mode_data_prep_is_idempotent(self, mock_delegate_cls, mock_setup):
        from amzn_nova_forge.manager.runtime_manager import SMTJRuntimeMode

        mock_delegate_cls.return_value = MagicMock()

        mgr = SMTJRuntimeManager("ml.m5.2xlarge", 1)
        mgr.set_mode(SMTJRuntimeMode.DATA_PREP)
        mgr.set_mode(SMTJRuntimeMode.DATA_PREP)

        # Delegate should only be built once, no matter how many filter
        # ops call set_mode.
        mock_delegate_cls.assert_called_once()

    @patch.object(SMTJRuntimeManager, "setup", return_value=None)
    @patch("amzn_nova_forge.manager.runtime_manager.SMTJDataPrepRuntimeManager")
    def test_set_mode_rejects_dataprep_to_training_flip(self, mock_delegate_cls, mock_setup):
        from amzn_nova_forge.manager.runtime_manager import SMTJRuntimeMode

        mock_delegate_cls.return_value = MagicMock()

        mgr = SMTJRuntimeManager("ml.m5.2xlarge", 1)
        mgr.set_mode(SMTJRuntimeMode.DATA_PREP)

        with self.assertRaises(RuntimeError):
            mgr.set_mode(SMTJRuntimeMode.TRAINING)

    @patch.object(SMTJRuntimeManager, "setup", return_value=None)
    @patch("amzn_nova_forge.manager.runtime_manager.SMTJDataPrepRuntimeManager")
    def test_execute_delegates_after_set_mode_data_prep(self, mock_delegate_cls, mock_setup):
        from amzn_nova_forge.manager.runtime_manager import SMTJRuntimeMode

        mock_delegate = MagicMock()
        mock_delegate.execute.return_value = "job-123"
        mock_delegate_cls.return_value = mock_delegate

        mgr = SMTJRuntimeManager("ml.m5.2xlarge", 1)
        mgr.set_mode(SMTJRuntimeMode.DATA_PREP)

        config = JobConfig(job_name="j", image_uri="i", recipe_path="/r")
        result = mgr.execute(config)

        self.assertEqual(result, "job-123")
        mock_delegate.execute.assert_called_once_with(config)

    @patch.object(SMTJRuntimeManager, "setup", return_value=None)
    @patch("amzn_nova_forge.manager.runtime_manager.SMTJDataPrepRuntimeManager")
    def test_cleanup_delegates_after_set_mode_data_prep(self, mock_delegate_cls, mock_setup):
        from amzn_nova_forge.manager.runtime_manager import SMTJRuntimeMode

        mock_delegate = MagicMock()
        mock_delegate_cls.return_value = mock_delegate

        mgr = SMTJRuntimeManager("ml.m5.2xlarge", 1)
        mgr.set_mode(SMTJRuntimeMode.DATA_PREP)

        mgr.cleanup("nova-forge-dataprep-test")
        mock_delegate.cleanup.assert_called_once_with("nova-forge-dataprep-test")

    @patch.object(SMTJRuntimeManager, "setup", return_value=None)
    def test_training_mode_default_does_not_create_delegate(self, mock_setup):
        mgr = SMTJRuntimeManager("ml.m5.xlarge", 1)
        self.assertIsNone(mgr._data_prep_delegate)

    @patch.object(SMTJRuntimeManager, "setup", return_value=None)
    def test_execute_rejects_dataprep_config_when_in_training_mode(self, mock_setup):
        mgr = SMTJRuntimeManager("ml.m5.xlarge", 1)
        config = DataPrepJobConfig(
            job_name="",
            image_uri="",
            recipe_path="",
            data_s3_path="s3://b/in",
            output_s3_path="s3://b/out",
            extra_args={"pipeline_id": "p"},
        )

        with self.assertRaises(ValueError) as ctx:
            mgr.execute(config)
        self.assertIn("set_mode", str(ctx.exception))


class TestSMHPRuntimeManager(unittest.TestCase):
    def setUp(self):
        self.instance_type = "ml.m5.xlarge"
        self.instance_count = 1
        self.cluster_name = "test-cluster"
        self.namespace = "test-namespace"

    @patch(
        "amzn_nova_forge.manager.runtime_manager.get_execution_role",
        return_value="arn:aws:iam::123456789012:role/MockRole",
    )
    @patch("subprocess.run")
    def test_initialization(self, mock_run, mock_get_role):
        mock_run.return_value.stdout = ""
        mock_run.return_value.stderr = ""

        manager = SMHPRuntimeManager(
            self.instance_type, self.instance_count, self.cluster_name, self.namespace
        )

        mock_run.assert_called_once_with(
            [
                "hyperpod",
                "connect-cluster",
                "--cluster-name",
                self.cluster_name,
                "--namespace",
                self.namespace,
            ],
            capture_output=True,
            text=True,
            check=True,
        )
        self.assertEqual(manager.instance_type, self.instance_type)
        self.assertEqual(manager.instance_count, self.instance_count)
        self.assertEqual(manager.cluster_name, self.cluster_name)
        self.assertEqual(manager.namespace, self.namespace)

    @patch.object(SMHPRuntimeManager, "setup", return_value=None)
    def test_instance_count_setter(self, mock_setup):
        manager = SMHPRuntimeManager(
            self.instance_type, self.instance_count, self.cluster_name, self.namespace
        )
        self.assertEqual(manager.instance_count, 1)

        new_instance_count = 4
        manager.instance_count = new_instance_count

        self.assertEqual(manager.instance_count, new_instance_count)

    @patch(
        "amzn_nova_forge.manager.runtime_manager.get_execution_role",
        return_value="arn:aws:iam::123456789012:role/MockRole",
    )
    @patch("subprocess.run")
    def test_initialization_stderr_benign(self, mock_run, mock_get_role):
        mock_run.return_value.stderr = "NotOpenSSLWarning: urllib3 v2 only supports OpenSSL 1.1.1+"
        mock_run.return_value.stdout = ""

        # Should NOT raise — benign stderr is filtered by _check_hyperpod_stderr
        manager = SMHPRuntimeManager(
            self.instance_type,
            self.instance_count,
            self.cluster_name,
            self.namespace,
        )
        self.assertIsNotNone(manager)

    @patch(
        "amzn_nova_forge.manager.runtime_manager.get_execution_role",
        return_value="arn:aws:iam::123456789012:role/MockRole",
    )
    @patch("subprocess.run")
    def test_initialization_stderr_real_error(self, mock_run, mock_get_role):
        mock_run.return_value.stderr = "Cluster with name not found"
        mock_run.return_value.stdout = ""

        with self.assertRaises(RuntimeError) as ctx:
            SMHPRuntimeManager(
                self.instance_type,
                self.instance_count,
                self.cluster_name,
                self.namespace,
            )
        self.assertIn("Cluster with name not found", str(ctx.exception))

    @patch(
        "amzn_nova_forge.manager.runtime_manager.get_execution_role",
        return_value="arn:aws:iam::123456789012:role/MockRole",
    )
    @patch("subprocess.run")
    def test_initialization_subprocess_error(self, mock_run, mock_get_role):
        mock_run.side_effect = subprocess.CalledProcessError(
            1, "hyperpod", stderr="Error: cluster not found"
        )

        with self.assertRaises(subprocess.CalledProcessError):
            SMHPRuntimeManager(
                self.instance_type,
                self.instance_count,
                self.cluster_name,
                self.namespace,
            )

    @patch(
        "amzn_nova_forge.manager.runtime_manager.get_execution_role",
        return_value="arn:aws:iam::123456789012:role/MockRole",
    )
    @patch("subprocess.run")
    def test_execute_success(self, mock_run, mock_get_role):
        mock_run.return_value.stdout = "NAME: test-job-123"
        mock_run.return_value.stderr = ""

        manager = SMHPRuntimeManager(
            self.instance_type, self.instance_count, self.cluster_name, self.namespace
        )
        mock_run.reset_mock()  # Reset call count after initialization

        image_uri = "123456789012.dkr.ecr.us-east-1.amazonaws.com/my-image:latest"

        job_config = JobConfig(
            job_name="test-job",
            image_uri=image_uri,
            recipe_path=f"{HYPERPOD_RECIPE_PATH}/path/to/recipe.yaml",
            output_s3_path="s3://output-bucket/output",
            data_s3_path="s3://input-bucket/data",
        )

        job_id = manager.execute(job_config)

        override_parameters = json.dumps(
            {
                "instance_type": self.instance_type,
                "container": image_uri,
            }
        )

        mock_run.assert_called_once_with(
            [
                "hyperpod",
                "start-job",
                "--namespace",
                self.namespace,
                "--recipe",
                "path/to/recipe",  # HyperPod CLI prefix and .yaml should be removed
                "--override-parameters",
                override_parameters,
            ],
            capture_output=True,
            text=True,
            check=True,
        )
        self.assertEqual(job_id, "test-job-123")

    @patch(
        "amzn_nova_forge.manager.runtime_manager.get_execution_role",
        return_value="arn:aws:iam::123456789012:role/MockRole",
    )
    @patch("subprocess.run")
    def test_execute_missing_parameters(self, mock_run, mock_get_role):
        mock_run.return_value = MagicMock(stdout="", stderr="")

        manager = SMHPRuntimeManager(
            self.instance_type, self.instance_count, self.cluster_name, self.namespace
        )

        job_config = JobConfig(
            job_name="",
            image_uri="123456789012.dkr.ecr.us-east-1.amazonaws.com/my-image:latest",
            recipe_path=f"{HYPERPOD_RECIPE_PATH}/path/to/recipe.yaml",
            output_s3_path="s3://output-bucket/output",
            data_s3_path="s3://input-bucket/data",
        )

        with self.assertRaises(ValueError):
            manager.execute(job_config)

    @patch(
        "amzn_nova_forge.manager.runtime_manager.get_execution_role",
        return_value="arn:aws:iam::123456789012:role/MockRole",
    )
    @patch("subprocess.run")
    def test_execute_handles_error(self, mock_run, mock_get_role):
        mock_run.side_effect = [
            MagicMock(stdout="", stderr=""),
            Exception("Failed to start job"),
        ]

        manager = SMHPRuntimeManager(
            self.instance_type, self.instance_count, self.cluster_name, self.namespace
        )

        job_config = JobConfig(
            job_name="test-job",
            image_uri="123456789012.dkr.ecr.us-east-1.amazonaws.com/my-image:latest",
            recipe_path=f"{HYPERPOD_RECIPE_PATH}/path/to/recipe.yaml",
            output_s3_path="s3://output-bucket/output",
            data_s3_path="s3://input-bucket/data",
        )

        with self.assertRaises(Exception) as context:
            manager.execute(job_config)

        self.assertEqual(str(context.exception), "Failed to start job")

    @patch(
        "amzn_nova_forge.manager.runtime_manager.get_execution_role",
        return_value="arn:aws:iam::123456789012:role/MockRole",
    )
    @patch("subprocess.run")
    def test_cleanup_success(self, mock_run, mock_get_role):
        mock_run.return_value.stdout = ""
        mock_run.return_value.stderr = ""

        manager = SMHPRuntimeManager(
            self.instance_type, self.instance_count, self.cluster_name, self.namespace
        )
        mock_run.reset_mock()  # Reset call count after initialization

        manager.cleanup("test-job")

        mock_run.assert_called_once_with(
            [
                "hyperpod",
                "cancel-job",
                "--job-name",
                "test-job",
                "--namespace",
                self.namespace,
            ],
            capture_output=True,
            text=True,
            check=True,
        )

    @patch(
        "amzn_nova_forge.manager.runtime_manager.get_execution_role",
        return_value="arn:aws:iam::123456789012:role/MockRole",
    )
    @patch("subprocess.run")
    def test_cleanup_handles_error(self, mock_run, mock_get_role):
        mock_run.side_effect = [
            MagicMock(stdout="", stderr=""),
            Exception("Cleanup failed"),
        ]

        manager = SMHPRuntimeManager(
            self.instance_type, self.instance_count, self.cluster_name, self.namespace
        )

        with self.assertRaises(Exception) as context:
            manager.cleanup("test-job")

        self.assertEqual(str(context.exception), "Cleanup failed")

    @patch(
        "amzn_nova_forge.manager.runtime_manager.get_execution_role",
        return_value="arn:aws:iam::123456789012:role/MockRole",
    )
    @patch("subprocess.run")
    def test_cleanup_stderr_benign(self, mock_run, mock_get_role):
        mock_run.side_effect = [
            MagicMock(stdout="", stderr=""),
            MagicMock(
                stdout="",
                stderr="NotOpenSSLWarning: urllib3 v2 only supports OpenSSL 1.1.1+",
            ),
        ]

        manager = SMHPRuntimeManager(
            self.instance_type, self.instance_count, self.cluster_name, self.namespace
        )
        # Should not raise — benign stderr is filtered
        manager.cleanup("test-job")

    @patch(
        "amzn_nova_forge.manager.runtime_manager.get_execution_role",
        return_value="arn:aws:iam::123456789012:role/MockRole",
    )
    @patch("subprocess.run")
    def test_cleanup_stderr_real_error_logged(self, mock_run, mock_get_role):
        mock_run.side_effect = [
            MagicMock(stdout="", stderr=""),
            MagicMock(stdout="", stderr="Cluster with name not found"),
        ]

        manager = SMHPRuntimeManager(
            self.instance_type, self.instance_count, self.cluster_name, self.namespace
        )
        # Should NOT raise — cleanup is best-effort, real errors are logged
        with self.assertLogs(level="ERROR") as log:
            manager.cleanup("test-job")
        self.assertTrue(any("Failed to cleanup HyperPod job" in msg for msg in log.output))

    @patch.object(SMHPRuntimeManager, "setup", return_value=None)
    def test_rft_lambda_arn_set_immediately_when_arn_passed(self, mock_setup):
        arn = "arn:aws:lambda:us-east-1:123456789012:function:SageMaker-reward"
        mgr = SMHPRuntimeManager(
            self.instance_type,
            self.instance_count,
            self.cluster_name,
            self.namespace,
            rft_lambda=arn,
        )
        self.assertEqual(mgr.rft_lambda_arn, arn)

    @patch.object(SMHPRuntimeManager, "setup", return_value=None)
    def test_rft_lambda_arn_none_when_file_path_passed(self, mock_setup):
        mgr = SMHPRuntimeManager(
            self.instance_type,
            self.instance_count,
            self.cluster_name,
            self.namespace,
            rft_lambda="reward.py",
        )
        self.assertIsNone(mgr.rft_lambda_arn)
        self.assertEqual(mgr.rft_lambda, "reward.py")

    @patch("amzn_nova_forge.manager.runtime_manager.boto3.client")
    @patch.object(SMHPRuntimeManager, "setup", return_value=None)
    def test_deploy_lambda_smhp_requires_sagemaker_prefix(self, mock_setup, mock_boto_client):
        """SMHP platform validation requires 'SageMaker' prefix in lambda name."""
        with tempfile.NamedTemporaryFile(suffix=".py", delete=False) as f:
            f.write(b"def lambda_handler(event, context): return {'reward': 1.0}")
            src = f.name

        try:
            mgr = SMHPRuntimeManager(
                self.instance_type,
                self.instance_count,
                self.cluster_name,
                self.namespace,
                rft_lambda=src,
            )
            mgr.execution_role = "arn:aws:iam::123456789012:role/SageMakerExecutionRole"
            mgr.region = "us-east-1"
            with self.assertRaises(ValueError):
                mgr.deploy_lambda(lambda_name="my-reward-fn")
        finally:
            os.unlink(src)

    @patch("amzn_nova_forge.manager.runtime_manager.boto3.client")
    @patch.object(SMHPRuntimeManager, "setup", return_value=None)
    def test_scale_cluster_success(self, mock_setup, mock_boto_client):
        """Test successful cluster scaling with valid RIG and positive value."""
        mock_sagemaker = MagicMock()
        mock_boto_client.return_value = mock_sagemaker

        # Mock describe_cluster response
        mock_sagemaker.describe_cluster.return_value = {
            "ClusterName": self.cluster_name,
            "ClusterStatus": "InService",
            "RestrictedInstanceGroups": [
                {
                    "InstanceGroupName": "worker-group",
                    "InstanceType": "ml.p4d.24xlarge",
                    "CurrentCount": 4,
                    "TargetCount": 4,
                    "ExecutionRole": "arn:aws:iam::123456789012:role/ExecutionRole",
                    "OverrideVpcConfig": {
                        "SecurityGroupIds": ["sg-123"],
                        "Subnets": ["subnet-123"],
                    },
                    "EnvironmentConfig": {
                        "FSxLustreConfig": {
                            "SizeInGiB": 1200,
                            "PerUnitStorageThroughput": 125,
                        }
                    },
                }
            ],
        }

        # Mock update_cluster response
        mock_sagemaker.update_cluster.return_value = {
            "ClusterArn": "arn:aws:sagemaker:us-west-2:123456789012:cluster/test-cluster"
        }

        manager = SMHPRuntimeManager(
            self.instance_type,
            self.instance_count,
            self.cluster_name,
            self.namespace,
        )
        manager.region = "us-west-2"

        result = manager.scale_cluster(instance_group_name="worker-group", target_instance_count=8)

        # Verify the result
        self.assertEqual(result["InstanceGroupName"], "worker-group")
        self.assertEqual(result["InstanceType"], "ml.p4d.24xlarge")
        self.assertEqual(result["PreviousCount"], 4)
        self.assertEqual(result["TargetCount"], 8)
        self.assertIn("ClusterArn", result)

        # Verify update_cluster was called with correct parameters
        mock_sagemaker.update_cluster.assert_called_once()
        call_kwargs = mock_sagemaker.update_cluster.call_args.kwargs
        self.assertEqual(call_kwargs["ClusterName"], self.cluster_name)
        self.assertIn("RestrictedInstanceGroups", call_kwargs)

        rig_params = call_kwargs["RestrictedInstanceGroups"][0]
        self.assertEqual(rig_params["InstanceGroupName"], "worker-group")
        self.assertEqual(rig_params["InstanceCount"], 8)
        self.assertEqual(rig_params["InstanceType"], "ml.p4d.24xlarge")
        self.assertIn("OverrideVpcConfig", rig_params)
        self.assertIn("EnvironmentConfig", rig_params)

    @patch("amzn_nova_forge.manager.runtime_manager.boto3.client")
    @patch.object(SMHPRuntimeManager, "setup", return_value=None)
    def test_scale_cluster_invalid_instance_group(self, mock_setup, mock_boto_client):
        """Test scaling with invalid instance group name raises ValueError."""
        mock_sagemaker = MagicMock()
        mock_boto_client.return_value = mock_sagemaker

        # Mock describe_cluster response with different group names
        mock_sagemaker.describe_cluster.return_value = {
            "ClusterName": self.cluster_name,
            "ClusterStatus": "InService",
            "RestrictedInstanceGroups": [
                {
                    "InstanceGroupName": "worker-group",
                    "InstanceType": "ml.p4d.24xlarge",
                    "CurrentCount": 4,
                    "TargetCount": 4,
                    "ExecutionRole": "arn:aws:iam::123456789012:role/ExecutionRole",
                    "OverrideVpcConfig": {
                        "SecurityGroupIds": ["sg-123"],
                        "Subnets": ["subnet-123"],
                    },
                    "EnvironmentConfig": {
                        "FSxLustreConfig": {
                            "SizeInGiB": 1200,
                            "PerUnitStorageThroughput": 125,
                        }
                    },
                }
            ],
        }

        manager = SMHPRuntimeManager(
            self.instance_type,
            self.instance_count,
            self.cluster_name,
            self.namespace,
        )
        manager.region = "us-west-2"

        # Try to scale non-existent group
        with self.assertRaises(ValueError) as context:
            manager.scale_cluster(instance_group_name="invalid-group", target_instance_count=8)

        error_msg = str(context.exception)
        self.assertIn("not found", error_msg)
        self.assertIn("invalid-group", error_msg)
        self.assertIn("worker-group", error_msg)  # Should list available groups

    @patch("amzn_nova_forge.manager.runtime_manager.boto3.client")
    @patch.object(SMHPRuntimeManager, "setup", return_value=None)
    def test_scale_cluster_negative_count(self, mock_setup, mock_boto_client):
        """Test scaling with negative instance count raises ValueError."""
        manager = SMHPRuntimeManager(
            self.instance_type,
            self.instance_count,
            self.cluster_name,
            self.namespace,
        )
        manager.region = "us-west-2"

        with self.assertRaises(ValueError) as context:
            manager.scale_cluster(instance_group_name="worker-group", target_instance_count=-1)

        error_msg = str(context.exception)
        self.assertIn("non-negative", error_msg)

    @patch("amzn_nova_forge.manager.runtime_manager.boto3.client")
    @patch.object(SMHPRuntimeManager, "setup", return_value=None)
    def test_scale_cluster_without_fsx_config(self, mock_setup, mock_boto_client):
        """Test scaling cluster without FSxLustreConfig in EnvironmentConfig."""
        mock_sagemaker = MagicMock()
        mock_boto_client.return_value = mock_sagemaker

        # Mock describe_cluster response without FSxLustreConfig (EnvironmentConfig)
        mock_sagemaker.describe_cluster.return_value = {
            "ClusterName": self.cluster_name,
            "ClusterStatus": "InService",
            "RestrictedInstanceGroups": [
                {
                    "InstanceGroupName": "worker-group",
                    "InstanceType": "ml.p4d.24xlarge",
                    "CurrentCount": 2,
                    "TargetCount": 2,
                    "ExecutionRole": "arn:aws:iam::123456789012:role/ExecutionRole",
                    "OverrideVpcConfig": {
                        "SecurityGroupIds": ["sg-123"],
                        "Subnets": ["subnet-123"],
                    },
                    "EnvironmentConfig": {},
                }
            ],
        }

        mock_sagemaker.update_cluster.return_value = {
            "ClusterArn": "arn:aws:sagemaker:us-west-2:123456789012:cluster/test-cluster"
        }

        manager = SMHPRuntimeManager(
            self.instance_type,
            self.instance_count,
            self.cluster_name,
            self.namespace,
        )
        manager.region = "us-west-2"

        result = manager.scale_cluster(instance_group_name="worker-group", target_instance_count=5)

        self.assertEqual(result["TargetCount"], 5)

        # Verify EnvironmentConfig is empty in the update call
        call_kwargs = mock_sagemaker.update_cluster.call_args.kwargs
        rig_params = call_kwargs["RestrictedInstanceGroups"][0]
        self.assertEqual(rig_params["EnvironmentConfig"], {})

    @patch("amzn_nova_forge.manager.runtime_manager.boto3.client")
    @patch.object(SMHPRuntimeManager, "setup", return_value=None)
    def test_scale_cluster_preserves_all_instance_groups(self, mock_setup, mock_boto_client):
        """Test that all instance groups are preserved when scaling one group."""
        mock_sagemaker = MagicMock()
        mock_boto_client.return_value = mock_sagemaker
        # Mock describe_cluster response with multiple instance groups
        mock_sagemaker.describe_cluster.return_value = {
            "ClusterName": self.cluster_name,
            "ClusterStatus": "InService",
            "RestrictedInstanceGroups": [
                {
                    "InstanceGroupName": "worker-group",
                    "InstanceType": "ml.p4d.24xlarge",
                    "CurrentCount": 4,
                    "TargetCount": 4,
                    "ExecutionRole": "arn:aws:iam::123456789012:role/ExecutionRole",
                    "OverrideVpcConfig": {
                        "SecurityGroupIds": ["sg-123"],
                        "Subnets": ["subnet-123"],
                    },
                    "EnvironmentConfig": {
                        "FSxLustreConfig": {
                            "SizeInGiB": 1200,
                            "PerUnitStorageThroughput": 125,
                        }
                    },
                },
                {
                    "InstanceGroupName": "controller-group",
                    "InstanceType": "ml.m5.xlarge",
                    "CurrentCount": 1,
                    "TargetCount": 1,
                    "ExecutionRole": "arn:aws:iam::123456789012:role/ExecutionRole",
                    "OverrideVpcConfig": {
                        "SecurityGroupIds": ["sg-456"],
                        "Subnets": ["subnet-456"],
                    },
                    "EnvironmentConfig": {},
                },
            ],
        }

        mock_sagemaker.update_cluster.return_value = {
            "ClusterArn": "arn:aws:sagemaker:us-west-2:123456789012:cluster/test-cluster"
        }

        manager = SMHPRuntimeManager(
            self.instance_type,
            self.instance_count,
            self.cluster_name,
            self.namespace,
        )
        manager.region = "us-west-2"

        result = manager.scale_cluster(instance_group_name="worker-group", target_instance_count=8)

        # Verify both instance groups are included in the update call
        call_kwargs = mock_sagemaker.update_cluster.call_args.kwargs
        all_groups = call_kwargs["RestrictedInstanceGroups"]

        self.assertEqual(len(all_groups), 2, "All instance groups should be included")

        # Find the worker group and controller group
        worker_group = next(g for g in all_groups if g["InstanceGroupName"] == "worker-group")
        controller_group = next(
            g for g in all_groups if g["InstanceGroupName"] == "controller-group"
        )

        # Verify worker group was scaled
        self.assertEqual(worker_group["InstanceCount"], 8)
        self.assertEqual(worker_group["InstanceType"], "ml.p4d.24xlarge")

        # Verify controller group was NOT modified
        self.assertEqual(controller_group["InstanceCount"], 1)
        self.assertEqual(controller_group["InstanceType"], "ml.m5.xlarge")

        # Verify both have their OverrideVpcConfig preserved
        self.assertIn("OverrideVpcConfig", worker_group)
        self.assertIn("OverrideVpcConfig", controller_group)

    @patch("amzn_nova_forge.manager.runtime_manager.boto3.client")
    @patch.object(SMHPRuntimeManager, "setup", return_value=None)
    def test_scale_cluster_not_in_service_state(self, mock_setup, mock_boto_client):
        """Test that scaling fails when cluster is not in InService state."""
        mock_sagemaker = MagicMock()
        mock_boto_client.return_value = mock_sagemaker

        # Mock describe_cluster response with cluster in Updating state
        mock_sagemaker.describe_cluster.return_value = {
            "ClusterName": self.cluster_name,
            "ClusterStatus": "Updating",
            "RestrictedInstanceGroups": [
                {
                    "InstanceGroupName": "worker-group",
                    "InstanceType": "ml.p4d.24xlarge",
                    "CurrentCount": 4,
                    "TargetCount": 4,
                    "ExecutionRole": "arn:aws:iam::123456789012:role/ExecutionRole",
                    "OverrideVpcConfig": {
                        "SecurityGroupIds": ["sg-123"],
                        "Subnets": ["subnet-123"],
                    },
                    "EnvironmentConfig": {
                        "FSxLustreConfig": {
                            "SizeInGiB": 1200,
                            "PerUnitStorageThroughput": 125,
                        }
                    },
                }
            ],
        }

        manager = SMHPRuntimeManager(
            self.instance_type,
            self.instance_count,
            self.cluster_name,
            self.namespace,
        )
        manager.region = "us-west-2"

        # Try to scale cluster that's not in InService state
        with self.assertRaises(ValueError) as context:
            manager.scale_cluster(instance_group_name="worker-group", target_instance_count=8)

        error_msg = str(context.exception)
        self.assertIn("Updating", error_msg)
        self.assertIn("InService", error_msg)

        # Verify update_cluster was NOT called
        mock_sagemaker.update_cluster.assert_not_called()

    @patch("amzn_nova_forge.manager.runtime_manager.boto3.client")
    @patch.object(SMHPRuntimeManager, "setup", return_value=None)
    def test_get_instance_groups_success(self, mock_setup, mock_boto_client):
        """Test successfully retrieving instance groups from cluster."""
        mock_sagemaker = MagicMock()
        mock_boto_client.return_value = mock_sagemaker

        # Mock describe_cluster response with multiple instance groups
        mock_sagemaker.describe_cluster.return_value = {
            "ClusterName": self.cluster_name,
            "ClusterStatus": "InService",
            "RestrictedInstanceGroups": [
                {
                    "InstanceGroupName": "worker-group",
                    "InstanceType": "ml.p4d.24xlarge",
                    "CurrentCount": 4,
                    "TargetCount": 4,
                    "ExecutionRole": "arn:aws:iam::123456789012:role/ExecutionRole",
                    "OverrideVpcConfig": {
                        "SecurityGroupIds": ["sg-123"],
                        "Subnets": ["subnet-123"],
                    },
                    "EnvironmentConfig": {
                        "FSxLustreConfig": {
                            "SizeInGiB": 1200,
                            "PerUnitStorageThroughput": 125,
                        }
                    },
                },
                {
                    "InstanceGroupName": "controller-group",
                    "InstanceType": "ml.m5.xlarge",
                    "CurrentCount": 1,
                    "TargetCount": 1,
                    "ExecutionRole": "arn:aws:iam::123456789012:role/ExecutionRole",
                    "OverrideVpcConfig": {
                        "SecurityGroupIds": ["sg-456"],
                        "Subnets": ["subnet-456"],
                    },
                    "EnvironmentConfig": {},
                },
                {
                    "InstanceGroupName": "test-group",
                    "InstanceType": "ml.c5.xlarge",
                    "CurrentCount": 0,
                    "TargetCount": 0,
                    "ExecutionRole": "arn:aws:iam::123456789012:role/ExecutionRole",
                    "OverrideVpcConfig": {
                        "SecurityGroupIds": ["sg-789"],
                        "Subnets": ["subnet-789"],
                    },
                    "EnvironmentConfig": {},
                },
            ],
        }

        manager = SMHPRuntimeManager(
            self.instance_type,
            self.instance_count,
            self.cluster_name,
            self.namespace,
        )
        manager.region = "us-west-2"

        result = manager.get_instance_groups()

        # Verify the result contains only essential fields
        self.assertEqual(len(result), 3)

        # Check first group
        self.assertEqual(result[0]["InstanceGroupName"], "worker-group")
        self.assertEqual(result[0]["InstanceType"], "ml.p4d.24xlarge")
        self.assertEqual(result[0]["CurrentCount"], 4)
        self.assertEqual(len(result[0]), 3, "Should only have 3 fields")

        # Check second group
        self.assertEqual(result[1]["InstanceGroupName"], "controller-group")
        self.assertEqual(result[1]["InstanceType"], "ml.m5.xlarge")
        self.assertEqual(result[1]["CurrentCount"], 1)

        # Check third group (with 0 instances)
        self.assertEqual(result[2]["InstanceGroupName"], "test-group")
        self.assertEqual(result[2]["InstanceType"], "ml.c5.xlarge")
        self.assertEqual(result[2]["CurrentCount"], 0)

        # Verify no extra fields are included
        for group in result:
            self.assertNotIn("ExecutionRole", group)
            self.assertNotIn("OverrideVpcConfig", group)
            self.assertNotIn("EnvironmentConfig", group)
            self.assertNotIn("TargetCount", group)
            self.assertNotIn("Status", group)

    @patch("amzn_nova_forge.manager.runtime_manager.boto3.client")
    @patch.object(SMHPRuntimeManager, "setup", return_value=None)
    def test_get_instance_groups_empty_cluster(self, mock_setup, mock_boto_client):
        """Test retrieving instance groups from cluster with no RIGs."""
        mock_sagemaker = MagicMock()
        mock_boto_client.return_value = mock_sagemaker

        # Mock describe_cluster response with no instance groups
        mock_sagemaker.describe_cluster.return_value = {
            "ClusterName": self.cluster_name,
            "ClusterStatus": "InService",
            "RestrictedInstanceGroups": [],
        }

        manager = SMHPRuntimeManager(
            self.instance_type,
            self.instance_count,
            self.cluster_name,
            self.namespace,
        )
        manager.region = "us-west-2"

        result = manager.get_instance_groups()

        # Verify empty list is returned
        self.assertEqual(result, [])
        self.assertEqual(len(result), 0)

    @patch("amzn_nova_forge.manager.runtime_manager.boto3.client")
    @patch.object(SMHPRuntimeManager, "setup", return_value=None)
    def test_scale_cluster_preserves_instance_storage_configs(self, mock_setup, mock_boto_client):
        """Test that InstanceStorageConfigs are preserved during scaling."""
        mock_sagemaker = MagicMock()
        mock_boto_client.return_value = mock_sagemaker

        # Mock describe_cluster response with InstanceStorageConfigs
        mock_sagemaker.describe_cluster.return_value = {
            "ClusterName": self.cluster_name,
            "ClusterStatus": "InService",
            "RestrictedInstanceGroups": [
                {
                    "InstanceGroupName": "worker-group",
                    "InstanceType": "ml.p4d.24xlarge",
                    "CurrentCount": 4,
                    "TargetCount": 4,
                    "ExecutionRole": "arn:aws:iam::123456789012:role/ExecutionRole",
                    "InstanceStorageConfigs": [
                        {
                            "EbsVolumeConfig": {
                                "VolumeSizeInGB": 500,
                                "RootVolume": True,
                            }
                        },
                        {
                            "FsxLustreConfig": {
                                "DnsName": "fs-12345.fsx.us-west-2.amazonaws.com",
                                "MountName": "fsx",
                                "MountPath": "/fsx",
                            }
                        },
                    ],
                    "OverrideVpcConfig": {
                        "SecurityGroupIds": ["sg-123"],
                        "Subnets": ["subnet-123"],
                    },
                }
            ],
        }

        mock_sagemaker.update_cluster.return_value = {
            "ClusterArn": "arn:aws:sagemaker:us-west-2:123456789012:cluster/test-cluster"
        }

        manager = SMHPRuntimeManager(
            self.instance_type,
            self.instance_count,
            self.cluster_name,
            self.namespace,
        )
        manager.region = "us-west-2"

        result = manager.scale_cluster(instance_group_name="worker-group", target_instance_count=8)

        # Verify InstanceStorageConfigs are included in the update call
        call_kwargs = mock_sagemaker.update_cluster.call_args.kwargs
        rig_params = call_kwargs["RestrictedInstanceGroups"][0]

        self.assertIn("InstanceStorageConfigs", rig_params)
        self.assertEqual(len(rig_params["InstanceStorageConfigs"]), 2)

        # Verify EBS config is preserved
        ebs_config = rig_params["InstanceStorageConfigs"][0]
        self.assertIn("EbsVolumeConfig", ebs_config)
        self.assertEqual(ebs_config["EbsVolumeConfig"]["VolumeSizeInGB"], 500)

        # Verify FSx config is preserved
        fsx_config = rig_params["InstanceStorageConfigs"][1]
        self.assertIn("FsxLustreConfig", fsx_config)
        self.assertEqual(
            fsx_config["FsxLustreConfig"]["DnsName"],
            "fs-12345.fsx.us-west-2.amazonaws.com",
        )

    @patch("amzn_nova_forge.manager.runtime_manager.boto3.client")
    @patch.object(SMHPRuntimeManager, "setup", return_value=None)
    def test_scale_cluster_preserves_threads_per_core(self, mock_setup, mock_boto_client):
        """Test that ThreadsPerCore is preserved during scaling."""
        mock_sagemaker = MagicMock()
        mock_boto_client.return_value = mock_sagemaker

        # Mock describe_cluster response with ThreadsPerCore
        mock_sagemaker.describe_cluster.return_value = {
            "ClusterName": self.cluster_name,
            "ClusterStatus": "InService",
            "RestrictedInstanceGroups": [
                {
                    "InstanceGroupName": "worker-group",
                    "InstanceType": "ml.p4d.24xlarge",
                    "CurrentCount": 4,
                    "TargetCount": 4,
                    "ExecutionRole": "arn:aws:iam::123456789012:role/ExecutionRole",
                    "ThreadsPerCore": 1,
                    "OverrideVpcConfig": {
                        "SecurityGroupIds": ["sg-123"],
                        "Subnets": ["subnet-123"],
                    },
                }
            ],
        }

        mock_sagemaker.update_cluster.return_value = {
            "ClusterArn": "arn:aws:sagemaker:us-west-2:123456789012:cluster/test-cluster"
        }

        manager = SMHPRuntimeManager(
            self.instance_type,
            self.instance_count,
            self.cluster_name,
            self.namespace,
        )
        manager.region = "us-west-2"

        result = manager.scale_cluster(instance_group_name="worker-group", target_instance_count=8)

        # Verify ThreadsPerCore is included in the update call
        call_kwargs = mock_sagemaker.update_cluster.call_args.kwargs
        rig_params = call_kwargs["RestrictedInstanceGroups"][0]

        self.assertIn("ThreadsPerCore", rig_params)
        self.assertEqual(rig_params["ThreadsPerCore"], 1)

    @patch("amzn_nova_forge.manager.runtime_manager.boto3.client")
    @patch.object(SMHPRuntimeManager, "setup", return_value=None)
    def test_scale_cluster_preserves_deep_health_checks(self, mock_setup, mock_boto_client):
        """Test that OnStartDeepHealthChecks are preserved during scaling."""
        mock_sagemaker = MagicMock()
        mock_boto_client.return_value = mock_sagemaker

        # Mock describe_cluster response with OnStartDeepHealthChecks
        mock_sagemaker.describe_cluster.return_value = {
            "ClusterName": self.cluster_name,
            "ClusterStatus": "InService",
            "RestrictedInstanceGroups": [
                {
                    "InstanceGroupName": "worker-group",
                    "InstanceType": "ml.p4d.24xlarge",
                    "CurrentCount": 4,
                    "TargetCount": 4,
                    "ExecutionRole": "arn:aws:iam::123456789012:role/ExecutionRole",
                    "OnStartDeepHealthChecks": [
                        "InstanceStress",
                        "InstanceConnectivity",
                    ],
                    "OverrideVpcConfig": {
                        "SecurityGroupIds": ["sg-123"],
                        "Subnets": ["subnet-123"],
                    },
                }
            ],
        }

        mock_sagemaker.update_cluster.return_value = {
            "ClusterArn": "arn:aws:sagemaker:us-west-2:123456789012:cluster/test-cluster"
        }

        manager = SMHPRuntimeManager(
            self.instance_type,
            self.instance_count,
            self.cluster_name,
            self.namespace,
        )
        manager.region = "us-west-2"

        result = manager.scale_cluster(instance_group_name="worker-group", target_instance_count=8)

        # Verify OnStartDeepHealthChecks are included in the update call
        call_kwargs = mock_sagemaker.update_cluster.call_args.kwargs
        rig_params = call_kwargs["RestrictedInstanceGroups"][0]

        self.assertIn("OnStartDeepHealthChecks", rig_params)
        self.assertEqual(
            rig_params["OnStartDeepHealthChecks"],
            ["InstanceStress", "InstanceConnectivity"],
        )

    @patch("amzn_nova_forge.manager.runtime_manager.boto3.client")
    @patch.object(SMHPRuntimeManager, "setup", return_value=None)
    def test_scale_cluster_with_all_immutable_fields(self, mock_setup, mock_boto_client):
        """Test scaling with all immutable fields present (comprehensive test)."""
        mock_sagemaker = MagicMock()
        mock_boto_client.return_value = mock_sagemaker

        # Mock describe_cluster response with all possible immutable fields
        mock_sagemaker.describe_cluster.return_value = {
            "ClusterName": self.cluster_name,
            "ClusterStatus": "InService",
            "RestrictedInstanceGroups": [
                {
                    "InstanceGroupName": "worker-group",
                    "InstanceType": "ml.p4d.24xlarge",
                    "CurrentCount": 4,
                    "TargetCount": 4,
                    "ExecutionRole": "arn:aws:iam::123456789012:role/ExecutionRole",
                    "ThreadsPerCore": 2,
                    "InstanceStorageConfigs": [
                        {
                            "EbsVolumeConfig": {
                                "VolumeSizeInGB": 1000,
                                "VolumeKmsKeyId": "arn:aws:kms:us-west-2:123456789012:key/12345",
                                "RootVolume": False,
                            }
                        }
                    ],
                    "OnStartDeepHealthChecks": ["InstanceStress"],
                    "OverrideVpcConfig": {
                        "SecurityGroupIds": ["sg-123", "sg-456"],
                        "Subnets": ["subnet-123", "subnet-456"],
                    },
                    "EnvironmentConfig": {
                        "FSxLustreConfig": {
                            "SizeInGiB": 2400,
                            "PerUnitStorageThroughput": 250,
                        }
                    },
                    "TrainingPlanArn": "arn:aws:sagemaker:us-west-2:123456789012:training-plan/my-plan",
                    "ScheduledUpdateConfig": {
                        "ScheduleExpression": "cron(0 0 * * ? *)",
                    },
                }
            ],
        }

        mock_sagemaker.update_cluster.return_value = {
            "ClusterArn": "arn:aws:sagemaker:us-west-2:123456789012:cluster/test-cluster"
        }

        manager = SMHPRuntimeManager(
            self.instance_type,
            self.instance_count,
            self.cluster_name,
            self.namespace,
        )
        manager.region = "us-west-2"

        result = manager.scale_cluster(instance_group_name="worker-group", target_instance_count=10)

        # Verify all immutable fields are preserved in the update call
        call_kwargs = mock_sagemaker.update_cluster.call_args.kwargs
        rig_params = call_kwargs["RestrictedInstanceGroups"][0]

        # Verify only InstanceCount changed
        self.assertEqual(rig_params["InstanceCount"], 10)
        self.assertEqual(rig_params["InstanceType"], "ml.p4d.24xlarge")

        # Verify all immutable fields are preserved with original values
        self.assertEqual(rig_params["ThreadsPerCore"], 2)
        self.assertIn("InstanceStorageConfigs", rig_params)
        self.assertEqual(
            rig_params["InstanceStorageConfigs"][0]["EbsVolumeConfig"]["VolumeSizeInGB"],
            1000,
        )
        self.assertEqual(rig_params["OnStartDeepHealthChecks"], ["InstanceStress"])
        self.assertIn("OverrideVpcConfig", rig_params)
        self.assertEqual(len(rig_params["OverrideVpcConfig"]["SecurityGroupIds"]), 2)
        self.assertIn("EnvironmentConfig", rig_params)
        self.assertEqual(rig_params["EnvironmentConfig"]["FSxLustreConfig"]["SizeInGiB"], 2400)
        self.assertEqual(
            rig_params["TrainingPlanArn"],
            "arn:aws:sagemaker:us-west-2:123456789012:training-plan/my-plan",
        )
        self.assertIn("ScheduledUpdateConfig", rig_params)

    @patch("amzn_nova_forge.manager.runtime_manager.boto3.client")
    @patch.object(SMHPRuntimeManager, "setup", return_value=None)
    def test_scale_cluster_without_optional_fields(self, mock_setup, mock_boto_client):
        """Test scaling when optional immutable fields are not present."""
        mock_sagemaker = MagicMock()
        mock_boto_client.return_value = mock_sagemaker

        # Mock describe_cluster response with minimal fields (no optional immutable fields)
        mock_sagemaker.describe_cluster.return_value = {
            "ClusterName": self.cluster_name,
            "ClusterStatus": "InService",
            "RestrictedInstanceGroups": [
                {
                    "InstanceGroupName": "worker-group",
                    "InstanceType": "ml.p4d.24xlarge",
                    "CurrentCount": 4,
                    "TargetCount": 4,
                    "ExecutionRole": "arn:aws:iam::123456789012:role/ExecutionRole",
                    # No ThreadsPerCore, InstanceStorageConfigs, OnStartDeepHealthChecks, etc.
                }
            ],
        }

        mock_sagemaker.update_cluster.return_value = {
            "ClusterArn": "arn:aws:sagemaker:us-west-2:123456789012:cluster/test-cluster"
        }

        manager = SMHPRuntimeManager(
            self.instance_type,
            self.instance_count,
            self.cluster_name,
            self.namespace,
        )
        manager.region = "us-west-2"

        result = manager.scale_cluster(instance_group_name="worker-group", target_instance_count=6)

        # Verify update succeeds with minimal fields
        call_kwargs = mock_sagemaker.update_cluster.call_args.kwargs
        rig_params = call_kwargs["RestrictedInstanceGroups"][0]

        # Verify only required fields are present
        self.assertEqual(rig_params["InstanceCount"], 6)
        self.assertEqual(rig_params["InstanceType"], "ml.p4d.24xlarge")
        self.assertEqual(
            rig_params["ExecutionRole"],
            "arn:aws:iam::123456789012:role/ExecutionRole",
        )

        # Verify optional fields are NOT present
        self.assertNotIn("ThreadsPerCore", rig_params)
        self.assertNotIn("InstanceStorageConfigs", rig_params)
        self.assertNotIn("OnStartDeepHealthChecks", rig_params)
        self.assertNotIn("OverrideVpcConfig", rig_params)
        self.assertNotIn("EnvironmentConfig", rig_params)
        self.assertNotIn("TrainingPlanArn", rig_params)
        self.assertNotIn("ScheduledUpdateConfig", rig_params)


if __name__ == "__main__":
    unittest.main()
