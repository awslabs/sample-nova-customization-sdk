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
import tarfile
import tempfile
import unittest
from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

from amzn_nova_forge.core.enums import EvaluationTask, Platform
from amzn_nova_forge.core.result import BaseJobResult
from amzn_nova_forge.core.result.eval_result import (
    SMTJEvaluationResult,
)
from amzn_nova_forge.core.result.job_result import JobStatus


class TestSMTJEvaluationResult(unittest.TestCase):
    def setUp(self):
        self.job_id = "test-job-123"
        self.started_time = datetime.now()
        self.eval_task = EvaluationTask.MMLU
        self.eval_output_path = "s3://test-bucket/output/test-job/output.tar.gz"
        self.mock_sagemaker_client = Mock()
        self.mock_s3_client = Mock()

        self.result = SMTJEvaluationResult(
            job_id=self.job_id,
            started_time=self.started_time,
            eval_task=self.eval_task,
            eval_output_path=self.eval_output_path,
            sagemaker_client=self.mock_sagemaker_client,
            s3_client=self.mock_s3_client,
        )

    def tearDown(self):
        # Clean up cache dir
        if hasattr(self.result, "_cached_results_dir"):
            self.result.clean()

    def test_get_job_status_completed(self):
        self.mock_sagemaker_client.describe_training_job.return_value = {
            "TrainingJobStatus": "Completed"
        }

        status, raw_status = self.result.get_job_status()

        self.assertEqual(status, JobStatus.COMPLETED)
        self.assertEqual(raw_status, "Completed")

    def test_get_job_status_unknown_maps_to_failed(self):
        self.mock_sagemaker_client.describe_training_job.return_value = {
            "TrainingJobStatus": "Stopped"
        }

        status, raw_status = self.result.get_job_status()

        self.assertEqual(status, JobStatus.FAILED)
        self.assertEqual(raw_status, "Stopped")

    @patch("builtins.print")
    def test_show_in_progress(self, mock_print):
        self.mock_sagemaker_client.describe_training_job.return_value = {
            "TrainingJobStatus": "InProgress"
        }

        self.result.show()

        mock_print.assert_called_with(
            "Job 'test-job-123' still running in progress. Please wait until the job is completed."
        )

    @patch("builtins.print")
    def test_show_failed(self, mock_print):
        self.mock_sagemaker_client.describe_training_job.return_value = {
            "TrainingJobStatus": "Failed"
        }

        self.result.show()

        mock_print.assert_called_with(
            "Cannot show eval result. Job 'test-job-123' in Failed status."
        )

    @patch("builtins.print")
    def test_show_stopped(self, mock_print):
        self.mock_sagemaker_client.describe_training_job.return_value = {
            "TrainingJobStatus": "Stopped"
        }

        self.result.show()

        mock_print.assert_called_with(
            "Cannot show eval result. Job 'test-job-123' in Stopped status."
        )

    @patch("builtins.print")
    def test_show_completed_with_results(self, mock_print):
        # Mock SageMaker response
        self.mock_sagemaker_client.describe_training_job.return_value = {
            "TrainingJobStatus": "Completed"
        }

        # Create mock tar.gz file with results
        test_results = {"accuracy": 0.8542, "f1_score": 0.7891, "task_name": "mmlu"}

        with tempfile.TemporaryDirectory() as source_dir:
            with tempfile.TemporaryDirectory() as cache_dir:
                # Create tar.gz file in source directory
                results_dir = Path(source_dir) / "run_name" / "eval_results"
                results_dir.mkdir(parents=True)

                results_file = results_dir / "results_20241104.json"
                with open(results_file, "w") as f:
                    json.dump(test_results, f)

                tar_path = Path(source_dir) / "output.tar.gz"
                with tarfile.open(tar_path, "w:gz") as tar:
                    tar.add(results_dir.parent, arcname="run_name")

                # Mock S3 download to copy from source to cache directory
                def mock_download_file(bucket, key, local_path):
                    import shutil

                    shutil.copy2(tar_path, local_path)

                self.mock_s3_client.download_file.side_effect = mock_download_file

                # Mock mkdtemp to return cache_dir
                with patch("tempfile.mkdtemp", return_value=cache_dir):
                    self.result.show()

                # Verify print calls
                expected_calls = [
                    unittest.mock.call("Job 'test-job-123' completed successfully."),
                    unittest.mock.call(
                        "\nEvaluation Results for job_id=test-job-123, eval_task=mmlu:"
                    ),
                    unittest.mock.call(json.dumps(test_results, indent=2)),
                ]
                mock_print.assert_has_calls(expected_calls)

    @patch("builtins.print")
    def test_show_completed_no_results_file(self, mock_print):
        self.mock_sagemaker_client.describe_training_job.return_value = {
            "TrainingJobStatus": "Completed"
        }

        with tempfile.TemporaryDirectory() as source_dir:
            with tempfile.TemporaryDirectory() as cache_dir:
                # Create empty tar.gz file in source directory
                tar_path = Path(source_dir) / "output.tar.gz"
                with tarfile.open(tar_path, "w:gz") as tar:
                    pass  # Empty tar file

                def mock_download_file(bucket, key, local_path):
                    import shutil

                    shutil.copy2(tar_path, local_path)

                self.mock_s3_client.download_file.side_effect = mock_download_file

                with patch("tempfile.mkdtemp", return_value=cache_dir):
                    self.result.show()

                mock_print.assert_any_call(
                    "No evaluation results json file found for job test-job-123"
                )

    @patch("builtins.print")
    def test_show_completed_s3_error(self, mock_print):
        self.mock_sagemaker_client.describe_training_job.return_value = {
            "TrainingJobStatus": "Completed"
        }

        self.mock_s3_client.download_file.side_effect = Exception("S3 access denied")

        with self.assertRaises(Exception):
            self.result.show()

        mock_print.assert_any_call("Error retrieving evaluation results: S3 access denied")
        mock_print.assert_any_call(f"Results available at: {self.eval_output_path}")

    def test_caching_mechanism(self):
        """Test caching mechanism, second .show() call should not re-download the file"""
        self.mock_sagemaker_client.describe_training_job.return_value = {
            "TrainingJobStatus": "Completed"
        }

        test_results = {"accuracy": 0.85}

        with tempfile.TemporaryDirectory() as source_dir:
            with tempfile.TemporaryDirectory() as cache_dir:
                # Create test file in source dir
                results_dir = Path(source_dir) / "run_name" / "eval_results"
                results_dir.mkdir(parents=True)

                results_file = results_dir / "results_test.json"
                with open(results_file, "w") as f:
                    json.dump(test_results, f)

                tar_path = Path(source_dir) / "output.tar.gz"
                with tarfile.open(tar_path, "w:gz") as tar:
                    tar.add(results_dir.parent, arcname="run_name")

                def mock_download_file(bucket, key, local_path):
                    import shutil

                    shutil.copy2(tar_path, local_path)

                self.mock_s3_client.download_file.side_effect = mock_download_file

                with patch("tempfile.mkdtemp", return_value=cache_dir):
                    # First call, should call s3 to download file
                    self.result.show()
                    self.assertEqual(self.mock_s3_client.download_file.call_count, 1)

                    # Second call, should use cache
                    self.result.show()
                    self.assertEqual(
                        self.mock_s3_client.download_file.call_count, 1
                    )  # Should still 1, no increase

    def test_clean_method(self):
        """Test temp cache dir was cleaned"""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Simulate cache dir
            self.result._cached_results_dir = temp_dir
            test_file = Path(temp_dir) / "test.txt"
            test_file.write_text("test")

            self.assertTrue(test_file.exists())

            # Call clean method
            self.result.clean()

            # Assert cache dir has been cleaned
            self.assertFalse(Path(temp_dir).exists())
            self.assertIsNone(self.result._cached_results_dir)

    def test_job_status_caching_completed(self):
        # Return COMPLETED at first call
        self.mock_sagemaker_client.describe_training_job.return_value = {
            "TrainingJobStatus": "Completed"
        }

        # First call
        status1, raw_status1 = self.result.get_job_status()
        self.assertEqual(status1, JobStatus.COMPLETED)
        self.assertEqual(raw_status1, "Completed")
        self.assertEqual(self.mock_sagemaker_client.describe_training_job.call_count, 1)

        # Second call - should use cache rather than making api call
        status2, raw_status2 = self.result.get_job_status()
        self.assertEqual(status2, JobStatus.COMPLETED)
        self.assertEqual(raw_status2, "Completed")
        self.assertEqual(self.mock_sagemaker_client.describe_training_job.call_count, 1)  # Still 1

    def test_job_status_caching_in_progress_then_completed(self):
        """Test job status change from IN_PROGRESS to COMPLETED"""
        # First call return IN_PROGRESS
        self.mock_sagemaker_client.describe_training_job.return_value = {
            "TrainingJobStatus": "InProgress"
        }

        # First call
        status1, raw_status1 = self.result.get_job_status()
        self.assertEqual(status1, JobStatus.IN_PROGRESS)
        self.assertEqual(raw_status1, "InProgress")
        self.assertEqual(self.mock_sagemaker_client.describe_training_job.call_count, 1)

        # Change job status to COMPLETED
        self.mock_sagemaker_client.describe_training_job.return_value = {
            "TrainingJobStatus": "Completed"
        }

        # Second call - should call API to get new status
        status2, raw_status2 = self.result.get_job_status()
        self.assertEqual(status2, JobStatus.COMPLETED)
        self.assertEqual(raw_status2, "Completed")
        self.assertEqual(
            self.mock_sagemaker_client.describe_training_job.call_count, 2
        )  # Should increase to 2

        # Third call - should use cache now
        status3, raw_status3 = self.result.get_job_status()
        self.assertEqual(status3, JobStatus.COMPLETED)
        self.assertEqual(raw_status3, "Completed")
        self.assertEqual(self.mock_sagemaker_client.describe_training_job.call_count, 2)  # Still 2

    def test_job_status_caching_failed_no_cache(self):
        """Test FAILED status should be cached"""
        self.mock_sagemaker_client.describe_training_job.return_value = {
            "TrainingJobStatus": "Failed"
        }

        # First call
        status1, raw_status1 = self.result.get_job_status()
        self.assertEqual(status1, JobStatus.FAILED)
        self.assertEqual(raw_status1, "Failed")
        self.assertEqual(self.mock_sagemaker_client.describe_training_job.call_count, 1)

        # Second call - should use cache
        status2, raw_status2 = self.result.get_job_status()
        self.assertEqual(status2, JobStatus.FAILED)
        self.assertEqual(raw_status2, "Failed")
        self.assertEqual(self.mock_sagemaker_client.describe_training_job.call_count, 1)  # Still 1

    def test_upload_tensorboard_results_with_custom_path(self):
        """Test upload tensorboard results with custom S3 path"""
        with tempfile.TemporaryDirectory() as cache_dir:
            # Create tensorboard results directory and files
            tensorboard_dir = Path(cache_dir) / "tensorboard_results" / "eval"
            tensorboard_dir.mkdir(parents=True)
            (tensorboard_dir / "events.out.tfevents.123").write_text("tensorboard data")

            self.result._cached_results_dir = cache_dir

            self.result.upload_tensorboard_results("s3://custom-bucket/custom/path/")

            self.mock_s3_client.upload_file.assert_called_once_with(
                str(tensorboard_dir / "events.out.tfevents.123"),
                "custom-bucket",
                "custom/path/eval/events.out.tfevents.123",
            )

    def test_upload_tensorboard_results_default_path(self):
        """Test upload tensorboard results with default S3 path"""
        with tempfile.TemporaryDirectory() as cache_dir:
            tensorboard_dir = Path(cache_dir) / "tensorboard_results" / "eval"
            tensorboard_dir.mkdir(parents=True)
            (tensorboard_dir / "events.out.tfevents.123").write_text("tensorboard data")

            self.result._cached_results_dir = cache_dir

            self.result.upload_tensorboard_results()

            self.mock_s3_client.upload_file.assert_called_once_with(
                str(tensorboard_dir / "events.out.tfevents.123"),
                "test-bucket",
                "output/test-job/tensorboard_results/eval/events.out.tfevents.123",
            )

    def test_upload_tensorboard_results_no_directory(self):
        """Test upload tensorboard results when directory doesn't exist"""
        with tempfile.TemporaryDirectory() as cache_dir:
            self.result._cached_results_dir = cache_dir

            with patch("amzn_nova_forge.core.result.eval_result.logger") as mock_logger:
                self.result.upload_tensorboard_results()
                mock_logger.warning.assert_called_once()

    def test_upload_tensorboard_results_s3_error(self):
        """Test upload tensorboard results with S3 error"""
        self.mock_s3_client.upload_file.side_effect = Exception("S3 error")

        with tempfile.TemporaryDirectory() as cache_dir:
            tensorboard_dir = Path(cache_dir) / "tensorboard_results" / "eval"
            tensorboard_dir.mkdir(parents=True)
            (tensorboard_dir / "events.out.tfevents.123").write_text("tensorboard data")

            self.result._cached_results_dir = cache_dir

            with self.assertRaises(Exception):
                self.result.upload_tensorboard_results()

    def test_dump_with_default_filename(self):
        with patch("builtins.open", create=True) as mock_open:
            mock_file = MagicMock()
            mock_open.return_value.__enter__.return_value = mock_file

            with patch("json.dump") as mock_json_dump:
                self.result.dump()

                expected_filename = f"{self.job_id}_{Platform.SMTJ.value}.json"
                mock_open.assert_called_once_with(Path(expected_filename), "w")

                # Check that json.dump was called with data containing __class_name__
                args, kwargs = mock_json_dump.call_args
                data = args[0]
                self.assertEqual(data["__class_name__"], "SMTJEvaluationResult")
                self.assertEqual(data["job_id"], self.job_id)
                self.assertEqual(data["started_time"], self.started_time.isoformat())
                self.assertEqual(data["eval_task"], self.eval_task.value)
                self.assertEqual(data["eval_output_path"], self.eval_output_path)

    def test_dump_with_custom_filename(self):
        with patch("builtins.open", create=True) as mock_open:
            mock_file = MagicMock()
            mock_open.return_value.__enter__.return_value = mock_file

            with patch("json.dump") as mock_json_dump:
                self.result.dump(file_name="custom_eval_result.json")

                mock_open.assert_called_once_with(Path("custom_eval_result.json"), "w")

    def test_dump_creates_valid_json_file(self):
        """Test that dump method creates a JSON file with correct content"""
        with tempfile.TemporaryDirectory() as temp_dir:
            file_path = Path(temp_dir)

            result_path = self.result.dump(file_path=str(file_path))

            # Verify file was created
            self.assertTrue(result_path.exists())

            # Read and verify file content
            with open(result_path, "r") as f:
                data = json.load(f)

            # Verify required fields and all output
            expected_data = {
                "__class_name__": "SMTJEvaluationResult",
                "job_id": self.job_id,
                "started_time": self.started_time.isoformat(),
                "eval_task": self.eval_task.value,
                "eval_output_path": self.eval_output_path,
            }
            print(data)
            self.assertEqual(data["__class_name__"], "SMTJEvaluationResult")
            self.assertEqual(data["job_id"], self.job_id)
            self.assertEqual(data["started_time"], self.started_time.isoformat())
            self.assertEqual(data["eval_task"], self.eval_task.value)
            self.assertEqual(data["eval_output_path"], self.eval_output_path)
            self.assertEqual(data, expected_data)

    @patch("amzn_nova_forge.core.result.job_result.json.load")
    @patch("builtins.open")
    def test_load_with_class_name(self, mock_open, mock_json_load):
        test_data = {
            "__class_name__": "SMTJEvaluationResult",
            "job_id": self.job_id,
            "started_time": self.started_time.isoformat(),
            "eval_task": self.eval_task.value,
            "eval_output_path": self.eval_output_path,
        }

        mock_json_load.return_value = test_data
        mock_file = MagicMock()
        mock_open.return_value.__enter__.return_value = mock_file

        # Mock the constructor to avoid actual initialization
        with patch.object(SMTJEvaluationResult, "__init__", return_value=None) as mock_init:
            result = BaseJobResult.load("test.json")

            self.assertIsInstance(result, SMTJEvaluationResult)
            mock_open.assert_called_once_with("test.json", "r")
            mock_init.assert_called_once_with(
                job_id=self.job_id,
                started_time=self.started_time,
                eval_task=self.eval_task,
                eval_output_path=self.eval_output_path,
                region=None,
            )

    @patch("amzn_nova_forge.core.result.job_result.json.load")
    @patch("builtins.open")
    def test_load_without_class_name(self, mock_open, mock_json_load):
        test_data = {
            "job_id": self.job_id,
            "started_time": self.started_time.isoformat(),
            "eval_task": self.eval_task.value,
            "eval_output_path": self.eval_output_path,
        }

        mock_json_load.return_value = test_data
        mock_file = MagicMock()
        mock_open.return_value.__enter__.return_value = mock_file

        with self.assertRaises(ValueError) as context:
            BaseJobResult.load("test.json")

        self.assertIn("no class name found", str(context.exception))

    @patch("amzn_nova_forge.core.result.job_result.json.load")
    @patch("builtins.open")
    def test_load_with_bad_class_name(self, mock_open, mock_json_load):
        test_data = {
            "__class_name__": "BadClassName",
            "job_id": self.job_id,
            "started_time": self.started_time.isoformat(),
            "eval_task": self.eval_task.value,
            "eval_output_path": self.eval_output_path,
        }

        mock_json_load.return_value = test_data
        mock_file = MagicMock()
        mock_open.return_value.__enter__.return_value = mock_file

        with self.assertRaises(ValueError) as context:
            BaseJobResult.load("test.json")

        self.assertIn("not found or not a subclass of BaseJobResult", str(context.exception))

    @patch("amzn_nova_forge.core.result.job_result.json.load")
    @patch("builtins.open")
    def test_load_with_unknown_fields_succeed(self, mock_open, mock_json_load):
        test_data = {
            "__class_name__": "SMTJEvaluationResult",
            "job_id": self.job_id,
            "started_time": self.started_time.isoformat(),
            "eval_task": self.eval_task.value,
            "eval_output_path": self.eval_output_path,
            "unknown_fields": "unknown_value",
        }

        mock_json_load.return_value = test_data
        mock_file = MagicMock()
        mock_open.return_value.__enter__.return_value = mock_file

        # Mock the constructor to avoid actual initialization
        with patch.object(SMTJEvaluationResult, "__init__", return_value=None) as mock_init:
            result = BaseJobResult.load("test.json")

            self.assertIsInstance(result, SMTJEvaluationResult)
            mock_open.assert_called_once_with("test.json", "r")
            mock_init.assert_called_once_with(
                job_id=self.job_id,
                started_time=self.started_time,
                eval_task=self.eval_task,
                eval_output_path=self.eval_output_path,
                region=None,
            )

    def test_get_method_returns_dict(self):
        self.mock_sagemaker_client.describe_training_job.return_value = {
            "TrainingJobStatus": "Completed"
        }

        test_results = {"accuracy": 0.85, "f1_score": 0.78}

        with tempfile.TemporaryDirectory() as source_dir:
            with tempfile.TemporaryDirectory() as cache_dir:
                # Create test results file
                results_dir = Path(source_dir) / "run_name" / "eval_results"
                results_dir.mkdir(parents=True)

                results_file = results_dir / "results_test.json"
                with open(results_file, "w") as f:
                    json.dump(test_results, f)

                tar_path = Path(source_dir) / "output.tar.gz"
                with tarfile.open(tar_path, "w:gz") as tar:
                    tar.add(results_dir.parent, arcname="run_name")

                def mock_download_file(bucket, key, local_path):
                    import shutil

                    shutil.copy2(tar_path, local_path)

                self.mock_s3_client.download_file.side_effect = mock_download_file

                with patch("tempfile.mkdtemp", return_value=cache_dir):
                    result_dict = self.result.get()

                    self.assertEqual(result_dict, test_results)

    def test_platform_property(self):
        self.assertEqual(self.result._platform, Platform.SMTJ)

    @patch("builtins.print")
    def test_download_s3_directory(self, mock_print):
        """Test downloading from S3 directory instead of tar.gz file"""
        # Set up S3 directory path
        self.result.eval_output_path = "s3://test-bucket/output/test-job/eval-result/"

        self.mock_sagemaker_client.describe_training_job.return_value = {
            "TrainingJobStatus": "Completed"
        }

        test_results = {"accuracy": 0.85, "f1_score": 0.78}

        # Mock S3 list_objects_v2 paginator
        mock_paginator = Mock()
        mock_page = {
            "Contents": [
                {"Key": "output/test-job/eval-result/results_test.json"},
                {"Key": "output/test-job/eval-result/subdir/other_file.txt"},
            ]
        }
        mock_paginator.paginate.return_value = [mock_page]
        self.mock_s3_client.get_paginator.return_value = mock_paginator

        with tempfile.TemporaryDirectory() as cache_dir:
            # Mock download_file to create test files
            def mock_download_file(bucket, key, local_path):
                Path(local_path).parent.mkdir(parents=True, exist_ok=True)
                if key.endswith("results_test.json"):
                    with open(local_path, "w") as f:
                        json.dump(test_results, f)
                else:
                    Path(local_path).write_text("test content")

            self.mock_s3_client.download_file.side_effect = mock_download_file

            with patch("tempfile.mkdtemp", return_value=cache_dir):
                result_dict = self.result.get()

                # Verify S3 operations
                self.mock_s3_client.get_paginator.assert_called_once_with("list_objects_v2")
                mock_paginator.paginate.assert_called_once_with(
                    Bucket="test-bucket", Prefix="output/test-job/eval-result/"
                )

                # Verify files were downloaded
                self.assertEqual(self.mock_s3_client.download_file.call_count, 2)

                # Verify results
                self.assertEqual(result_dict, test_results)


class TestRegionPropagation(unittest.TestCase):
    @patch("amzn_nova_forge.core.result.eval_result.boto3")
    def test_smtj_evaluation_result_passes_region_to_sagemaker(self, mock_boto3):
        mock_boto3.client.return_value = Mock()

        SMTJEvaluationResult(
            job_id="test-job",
            started_time=datetime.now(),
            eval_task=EvaluationTask.MMLU,
            eval_output_path="s3://bucket/output",
            sagemaker_client=None,
            s3_client=Mock(),
            region="eu-west-1",
        )

        mock_boto3.client.assert_called_with("sagemaker", region_name="eu-west-1")


class TestSMTJEvaluationResultMTRL(unittest.TestCase):
    """Tests for MTRL eval pipeline status polling."""

    @patch("boto3.client")
    def test_mtrl_eval_uses_pipeline_status_manager(self, mock_boto3):
        mock_boto3.return_value = Mock()
        result = SMTJEvaluationResult(
            job_id="arn:aws:sagemaker:us-east-1:123456789012:pipeline/SagemakerEvaluation-MTRLEvaluation/execution/abc123",
            started_time=datetime.now(),
            eval_task=EvaluationTask.RFT_MULTITURN_EVAL,
            eval_output_path="s3://bucket/output/",
            s3_client=Mock(),
        )
        from amzn_nova_forge.core.result.job_result import MTRLStatusManager

        self.assertIsInstance(result._status_manager, MTRLStatusManager)

    @patch("boto3.client")
    def test_mtrl_eval_polls_pipeline_execution(self, mock_boto3):
        mock_sm_client = Mock()
        mock_sm_client.describe_pipeline_execution.return_value = {
            "PipelineExecutionStatus": "Executing"
        }
        mock_boto3.return_value = mock_sm_client

        result = SMTJEvaluationResult(
            job_id="arn:aws:sagemaker:us-east-1:123456789012:pipeline/Test/execution/xyz",
            started_time=datetime.now(),
            eval_task=EvaluationTask.RFT_MULTITURN_EVAL,
            eval_output_path="s3://bucket/output/",
            s3_client=Mock(),
        )

        status, raw = result.get_job_status()
        self.assertEqual(status, JobStatus.IN_PROGRESS)
        self.assertEqual(raw, "Executing")

    @patch("boto3.client")
    def test_mtrl_eval_pipeline_succeeded(self, mock_boto3):
        mock_sm_client = Mock()
        mock_sm_client.describe_pipeline_execution.return_value = {
            "PipelineExecutionStatus": "Succeeded"
        }
        mock_boto3.return_value = mock_sm_client

        result = SMTJEvaluationResult(
            job_id="arn:aws:sagemaker:us-east-1:123456789012:pipeline/Test/execution/xyz",
            started_time=datetime.now(),
            eval_task=EvaluationTask.RFT_MULTITURN_EVAL,
            eval_output_path="s3://bucket/output/",
            s3_client=Mock(),
        )

        status, raw = result.get_job_status()
        self.assertEqual(status, JobStatus.COMPLETED)
        self.assertEqual(raw, "Succeeded")

    @patch("boto3.client")
    def test_mtrl_eval_pipeline_failed(self, mock_boto3):
        mock_sm_client = Mock()
        mock_sm_client.describe_pipeline_execution.return_value = {
            "PipelineExecutionStatus": "Failed"
        }
        mock_boto3.return_value = mock_sm_client

        result = SMTJEvaluationResult(
            job_id="arn:aws:sagemaker:us-east-1:123456789012:pipeline/Test/execution/xyz",
            started_time=datetime.now(),
            eval_task=EvaluationTask.RFT_MULTITURN_EVAL,
            eval_output_path="s3://bucket/output/",
            s3_client=Mock(),
        )

        status, raw = result.get_job_status()
        self.assertEqual(status, JobStatus.FAILED)

    @patch("boto3.client")
    def test_non_mtrl_eval_uses_smtj_status_manager(self, mock_boto3):
        mock_boto3.return_value = Mock()
        result = SMTJEvaluationResult(
            job_id="some-training-job",
            started_time=datetime.now(),
            eval_task=EvaluationTask.MMLU,
            eval_output_path="s3://bucket/output/",
            sagemaker_client=Mock(),
            s3_client=Mock(),
        )
        from amzn_nova_forge.core.result.job_result import SMTJStatusManager

        self.assertIsInstance(result._status_manager, SMTJStatusManager)

    @patch("boto3.client")
    def test_mtrl_eval_extracts_region_from_arn(self, mock_boto3):
        mock_sm_client = Mock()
        mock_sm_client.describe_pipeline_execution.return_value = {
            "PipelineExecutionStatus": "Executing"
        }
        mock_boto3.return_value = mock_sm_client

        result = SMTJEvaluationResult(
            job_id="arn:aws:sagemaker:eu-west-1:123456789012:pipeline/Test/execution/xyz",
            started_time=datetime.now(),
            eval_task=EvaluationTask.RFT_MULTITURN_EVAL,
            eval_output_path="s3://bucket/output/",
            s3_client=Mock(),
            region="us-east-1",
        )

        result.get_job_status()
        mock_boto3.assert_any_call("sagemaker", region_name="eu-west-1")

    @patch("boto3.client")
    def test_mtrl_eval_resolve_start_time_pipeline(self, mock_boto3):
        from datetime import timezone

        mock_sm_client = Mock()
        mock_sm_client.describe_pipeline_execution.return_value = {
            "CreationTime": datetime(2026, 6, 3, 18, 0, 0, tzinfo=timezone.utc)
        }
        mock_boto3.return_value = mock_sm_client

        from amzn_nova_forge.core.result.job_result import MTRLStatusManager

        mgr = MTRLStatusManager(region="us-east-1")
        result = mgr.resolve_start_time(
            "arn:aws:sagemaker:us-east-1:123456789012:pipeline/Test/execution/xyz"
        )
        self.assertEqual(result, datetime(2026, 6, 3, 18, 0, 0, tzinfo=timezone.utc))

    @patch("boto3.client")
    def test_dump_pipeline_arn_with_job_name(self, mock_boto3):
        import os

        mock_boto3.return_value = Mock()
        result = SMTJEvaluationResult(
            job_id="arn:aws:sagemaker:us-east-1:123456789012:pipeline/Eval/execution/abc123",
            started_time=datetime.now(),
            eval_task=EvaluationTask.RFT_MULTITURN_EVAL,
            eval_output_path="s3://bucket/output/",
            s3_client=Mock(),
        )
        result._job_name = "my-eval"
        path = result.dump()
        self.assertEqual(str(path), "my-eval-abc123_SMTJServerless.json")
        os.remove(str(path))

    @patch("boto3.client")
    def test_dump_pipeline_arn_without_job_name(self, mock_boto3):
        import os

        mock_boto3.return_value = Mock()
        result = SMTJEvaluationResult(
            job_id="arn:aws:sagemaker:us-east-1:123456789012:pipeline/Eval/execution/xyz789",
            started_time=datetime.now(),
            eval_task=EvaluationTask.RFT_MULTITURN_EVAL,
            eval_output_path="s3://bucket/output/",
            s3_client=Mock(),
        )
        path = result.dump()
        self.assertEqual(str(path), "xyz789_SMTJServerless.json")
        os.remove(str(path))

    @patch("boto3.client")
    def test_dump_regular_job_id(self, mock_boto3):
        import os

        mock_boto3.return_value = Mock()
        result = SMTJEvaluationResult(
            job_id="my-training-job-123",
            started_time=datetime.now(),
            eval_task=EvaluationTask.MMLU,
            eval_output_path="s3://bucket/output/",
            sagemaker_client=Mock(),
            s3_client=Mock(),
        )
        path = result.dump()
        self.assertEqual(str(path), "my-training-job-123_SMTJ.json")
        os.remove(str(path))


if __name__ == "__main__":
    unittest.main()
