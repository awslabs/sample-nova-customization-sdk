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
import tempfile
import unittest
from datetime import datetime
from pathlib import Path
from unittest.mock import Mock, patch

from amzn_nova_forge.core.enums import (
    Model,
    Platform,
    TrainingMethod,
)
from amzn_nova_forge.core.result.job_result import (
    JobStatus,
    SMHPStatusManager,
    SMTJStatusManager,
)
from amzn_nova_forge.core.result.training_result import (
    BedrockTrainingResult,
    SMHPTrainingResult,
    SMTJTrainingResult,
)
from amzn_nova_forge.core.types import ModelArtifacts


class TestSMTJTrainResult(unittest.TestCase):
    def setUp(self):
        self.model_artifacts = ModelArtifacts(
            checkpoint_s3_path="s3://bucket/checkpoint/",
            output_s3_path="s3://bucket/output/",
        )
        self.mock_sagemaker_client = Mock()

    def test_init_with_default_client(self):
        """Test initialization with default SageMaker client"""
        with patch("boto3.client") as mock_boto3:
            result = SMTJTrainingResult(
                job_id="test-job-123",
                started_time=datetime(2024, 1, 1, 12, 0, 0),
                method=TrainingMethod.SFT_LORA,
                model_artifacts=self.model_artifacts,
                model_type=Model.NOVA_MICRO,
            )

            self.assertEqual(result.job_id, "test-job-123")
            self.assertEqual(result.method, TrainingMethod.SFT_LORA)
            self.assertEqual(result.model_artifacts, self.model_artifacts)
            self.assertEqual(result.platform, Platform.SMTJ)
            self.assertIsInstance(result.status_manager, SMTJStatusManager)
            self.assertEqual(result.model_type, Model.NOVA_MICRO)
            mock_boto3.assert_called_once_with("sagemaker", region_name=None)

    def test_init_with_custom_client(self):
        """Test initialization with custom SageMaker client"""
        result = SMTJTrainingResult(
            job_id="test-job-123",
            started_time=datetime(2024, 1, 1, 12, 0, 0),
            method=TrainingMethod.SFT_FULL,
            model_artifacts=self.model_artifacts,
            sagemaker_client=self.mock_sagemaker_client,
            model_type=Model.NOVA_MICRO,
        )

        self.assertEqual(result._sagemaker_client, self.mock_sagemaker_client)
        self.assertEqual(result.method, TrainingMethod.SFT_FULL)

    def test_create_status_manager(self):
        """Test status manager creation"""
        result = SMTJTrainingResult(
            job_id="test-job-123",
            started_time=datetime(2024, 1, 1, 12, 0, 0),
            method=TrainingMethod.SFT_LORA,
            model_artifacts=self.model_artifacts,
            sagemaker_client=self.mock_sagemaker_client,
            model_type=Model.NOVA_MICRO,
        )

        status_manager = result._create_status_manager()
        self.assertIsInstance(status_manager, SMTJStatusManager)
        self.assertEqual(status_manager._sagemaker_client, self.mock_sagemaker_client)

    def test_to_dict(self):
        """Test dictionary conversion"""
        started_time = datetime(2024, 1, 1, 12, 0, 0)
        result = SMTJTrainingResult(
            job_id="test-job-123",
            started_time=started_time,
            method=TrainingMethod.RFT_LORA,
            model_artifacts=self.model_artifacts,
            sagemaker_client=self.mock_sagemaker_client,
            model_type=Model.NOVA_MICRO,
        )

        result_dict = result._to_dict()
        expected = {
            "job_id": "test-job-123",
            "started_time": started_time.isoformat(),
            "method": TrainingMethod.RFT_LORA.value,
            "model_artifacts": {
                "checkpoint_s3_path": "s3://bucket/checkpoint/",
                "output_s3_path": "s3://bucket/output/",
                "output_model_arn": None,
            },
            "model_type": Model.NOVA_MICRO.name,
        }

        self.assertEqual(result_dict, expected)

    def test_from_dict(self):
        """Test object creation from dictionary"""
        data = {
            "job_id": "test-job-456",
            "started_time": "2024-01-01T12:00:00",
            "method": "sft_lora",
            "model_artifacts": {
                "checkpoint_s3_path": "s3://bucket/checkpoint/",
                "output_s3_path": "s3://bucket/output/",
            },
            "model_type": "NOVA_MICRO",
        }

        with patch("boto3.client"):
            result = SMTJTrainingResult._from_dict(data)

            self.assertEqual(result.job_id, "test-job-456")
            self.assertEqual(result.started_time, datetime(2024, 1, 1, 12, 0, 0))
            self.assertEqual(result.method, TrainingMethod.SFT_LORA)
            self.assertEqual(result.model_artifacts.checkpoint_s3_path, "s3://bucket/checkpoint/")
            self.assertEqual(result.model_artifacts.output_s3_path, "s3://bucket/output/")

    def test_get_method(self):
        """Test get method returns dictionary"""
        result = SMTJTrainingResult(
            job_id="test-job-123",
            started_time=datetime(2024, 1, 1, 12, 0, 0),
            method=TrainingMethod.SFT_LORA,
            model_artifacts=self.model_artifacts,
            sagemaker_client=self.mock_sagemaker_client,
            model_type=Model.NOVA_MICRO,
        )

        result_data = result.get()
        self.assertIsInstance(result_data, dict)
        self.assertEqual(result_data["job_id"], "test-job-123")

    def test_show_method(self):
        """Test show method prints result"""
        result = SMTJTrainingResult(
            job_id="test-job-123",
            started_time=datetime(2024, 1, 1, 12, 0, 0),
            method=TrainingMethod.SFT_LORA,
            model_artifacts=self.model_artifacts,
            sagemaker_client=self.mock_sagemaker_client,
            model_type=Model.NOVA_MICRO,
        )

        with patch("builtins.print") as mock_print:
            result.show()
            mock_print.assert_called_once()
            printed_args = str(mock_print.call_args)
            self.assertIn("test-job-123", printed_args)

    def test_dump_and_load_roundtrip(self):
        """Test dump and load roundtrip"""
        original_result = SMTJTrainingResult(
            job_id="test-job-123",
            started_time=datetime(2024, 1, 1, 12, 0, 0),
            method=TrainingMethod.SFT_LORA,
            model_artifacts=self.model_artifacts,
            sagemaker_client=self.mock_sagemaker_client,
            model_type=Model.NOVA_MICRO,
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            file_path = Path(temp_dir)
            original_result.dump(str(file_path))

            expected_file = (
                file_path / f"{original_result.job_id}_{original_result.platform.value}.json"
            )
            self.assertTrue(expected_file.exists())

            with patch("boto3.client"):
                loaded_result = SMTJTrainingResult.load(str(expected_file))

                self.assertEqual(loaded_result.job_id, original_result.job_id)
                self.assertEqual(loaded_result.method, original_result.method)
                self.assertEqual(
                    loaded_result.model_artifacts.output_s3_path,
                    original_result.model_artifacts.output_s3_path,
                )


class TestSMHPTrainResult(unittest.TestCase):
    def setUp(self):
        self.model_artifacts = ModelArtifacts(
            checkpoint_s3_path="s3://bucket/checkpoint/",
            output_s3_path="s3://bucket/output/",
        )

    def test_init_with_default_namespace(self):
        """Test initialization with default namespace"""
        result = SMHPTrainingResult(
            job_id="test-job-123",
            started_time=datetime(2024, 1, 1, 12, 0, 0),
            method=TrainingMethod.SFT_LORA,
            model_artifacts=self.model_artifacts,
            cluster_name="test-cluster",
            model_type=Model.NOVA_MICRO,
        )

        self.assertEqual(result.job_id, "test-job-123")
        self.assertEqual(result.cluster_name, "test-cluster")
        self.assertEqual(result.namespace, "kubeflow")
        self.assertEqual(result.platform, Platform.SMHP)
        self.assertIsInstance(result.status_manager, SMHPStatusManager)
        self.assertEqual(result.model_type, Model.NOVA_MICRO)

    def test_init_with_custom_namespace(self):
        """Test initialization with custom namespace"""
        result = SMHPTrainingResult(
            job_id="test-job-123",
            started_time=datetime(2024, 1, 1, 12, 0, 0),
            method=TrainingMethod.RFT_FULL,
            model_artifacts=self.model_artifacts,
            cluster_name="test-cluster",
            namespace="custom-namespace",
            model_type=Model.NOVA_MICRO,
        )

        self.assertEqual(result.namespace, "custom-namespace")
        self.assertEqual(result.method, TrainingMethod.RFT_FULL)

    def test_create_status_manager(self):
        """Test status manager creation"""
        result = SMHPTrainingResult(
            job_id="test-job-123",
            started_time=datetime(2024, 1, 1, 12, 0, 0),
            method=TrainingMethod.SFT_LORA,
            model_artifacts=self.model_artifacts,
            cluster_name="test-cluster",
            namespace="test-namespace",
            model_type=Model.NOVA_MICRO,
        )

        status_manager = result._create_status_manager()
        self.assertIsInstance(status_manager, SMHPStatusManager)
        self.assertEqual(status_manager.cluster_name, "test-cluster")
        self.assertEqual(status_manager.namespace, "test-namespace")

    def test_to_dict(self):
        """Test dictionary conversion"""
        started_time = datetime(2024, 1, 1, 12, 0, 0)
        result = SMHPTrainingResult(
            job_id="test-job-123",
            started_time=started_time,
            method=TrainingMethod.EVALUATION,
            model_artifacts=self.model_artifacts,
            cluster_name="test-cluster",
            namespace="test-namespace",
            model_type=Model.NOVA_MICRO,
        )

        result_dict = result._to_dict()
        expected = {
            "job_id": "test-job-123",
            "started_time": started_time.isoformat(),
            "method": TrainingMethod.EVALUATION.value,
            "model_artifacts": {
                "checkpoint_s3_path": "s3://bucket/checkpoint/",
                "output_s3_path": "s3://bucket/output/",
                "output_model_arn": None,
            },
            "cluster_name": "test-cluster",
            "namespace": "test-namespace",
            "model_type": Model.NOVA_MICRO.name,
        }

        self.assertEqual(result_dict, expected)

    def test_from_dict(self):
        """Test object creation from dictionary"""
        data = {
            "job_id": "test-job-456",
            "started_time": "2024-01-01T12:00:00",
            "method": "rft_lora",
            "model_artifacts": {
                "checkpoint_s3_path": "s3://bucket/checkpoint/",
                "output_s3_path": "s3://bucket/output/",
            },
            "cluster_name": "test-cluster",
            "namespace": "test-namespace",
            "model_type": "NOVA_MICRO",
        }

        result = SMHPTrainingResult._from_dict(data)

        self.assertEqual(result.job_id, "test-job-456")
        self.assertEqual(result.started_time, datetime(2024, 1, 1, 12, 0, 0))
        self.assertEqual(result.method, TrainingMethod.RFT_LORA)
        self.assertEqual(result.cluster_name, "test-cluster")
        self.assertEqual(result.namespace, "test-namespace")
        self.assertEqual(result.model_artifacts.checkpoint_s3_path, "s3://bucket/checkpoint/")
        self.assertEqual(result.model_type, Model.NOVA_MICRO)

    def test_get_method(self):
        """Test get method returns dictionary"""
        result = SMHPTrainingResult(
            job_id="test-job-123",
            started_time=datetime(2024, 1, 1, 12, 0, 0),
            method=TrainingMethod.SFT_LORA,
            model_artifacts=self.model_artifacts,
            cluster_name="test-cluster",
            model_type=Model.NOVA_MICRO,
        )

        result_data = result.get()
        self.assertIsInstance(result_data, dict)
        self.assertEqual(result_data["job_id"], "test-job-123")
        self.assertEqual(result_data["cluster_name"], "test-cluster")

    def test_show_method(self):
        """Test show method prints result"""
        result = SMHPTrainingResult(
            job_id="test-job-123",
            started_time=datetime(2024, 1, 1, 12, 0, 0),
            method=TrainingMethod.SFT_LORA,
            model_artifacts=self.model_artifacts,
            cluster_name="test-cluster",
            model_type=Model.NOVA_MICRO,
        )

        with patch("builtins.print") as mock_print:
            result.show()
            mock_print.assert_called_once()
            printed_args = str(mock_print.call_args)
            self.assertIn("test-job-123", printed_args)

    def test_dump_and_load_roundtrip(self):
        """Test dump and load roundtrip"""
        original_result = SMHPTrainingResult(
            job_id="test-job-456",
            started_time=datetime(2024, 1, 1, 12, 0, 0),
            method=TrainingMethod.RFT_FULL,
            model_artifacts=self.model_artifacts,
            cluster_name="test-cluster",
            namespace="test-namespace",
            model_type=Model.NOVA_MICRO,
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            file_path = Path(temp_dir)
            original_result.dump(str(file_path))

            expected_file = (
                file_path / f"{original_result.job_id}_{original_result.platform.value}.json"
            )
            self.assertTrue(expected_file.exists())

            loaded_result = SMHPTrainingResult.load(str(expected_file))

            self.assertEqual(loaded_result.job_id, original_result.job_id)
            self.assertEqual(loaded_result.cluster_name, original_result.cluster_name)
            self.assertEqual(loaded_result.namespace, original_result.namespace)
            self.assertEqual(loaded_result.method, original_result.method)

    def test_all_training_methods(self):
        """Test all training method enums work correctly"""
        methods = [
            TrainingMethod.SFT_LORA,
            TrainingMethod.SFT_FULL,
            TrainingMethod.RFT_FULL,
            TrainingMethod.RFT_LORA,
            TrainingMethod.EVALUATION,
        ]

        for method in methods:
            result = SMHPTrainingResult(
                job_id=f"test-job-{method.value}",
                started_time=datetime(2024, 1, 1, 12, 0, 0),
                method=method,
                model_artifacts=self.model_artifacts,
                cluster_name="test-cluster",
                model_type=Model.NOVA_MICRO,
            )

            self.assertEqual(result.method, method)
            result_dict = result._to_dict()
            self.assertEqual(result_dict["method"], method.value)


class TestRegionPropagation(unittest.TestCase):
    def setUp(self):
        self.model_artifacts = ModelArtifacts(
            checkpoint_s3_path="s3://bucket/checkpoint/",
            output_s3_path="s3://bucket/output/",
        )

    def test_smtj_training_result_passes_region(self):
        """Test SMTJTrainingResult passes region to boto3 client"""
        with patch("boto3.client") as mock_boto3:
            SMTJTrainingResult(
                job_id="test",
                started_time=datetime(2024, 1, 1, 12, 0, 0),
                method=TrainingMethod.SFT_LORA,
                model_artifacts=self.model_artifacts,
                model_type=Model.NOVA_MICRO,
                sagemaker_client=None,
                region="eu-west-1",
            )
            mock_boto3.assert_called_once_with("sagemaker", region_name="eu-west-1")

    def test_bedrock_training_result_passes_region(self):
        """Test BedrockTrainingResult passes region to boto3 client"""
        with patch("boto3.client") as mock_boto3:
            BedrockTrainingResult(
                job_id="test",
                started_time=datetime(2024, 1, 1, 12, 0, 0),
                method=TrainingMethod.SFT_LORA,
                model_artifacts=self.model_artifacts,
                model_type=Model.NOVA_MICRO,
                bedrock_client=None,
                region="eu-west-1",
            )
            mock_boto3.assert_called_once_with("bedrock", region_name="eu-west-1")


class TestSMTJTrainingResultMTRL(unittest.TestCase):
    """Tests for MTRL-specific methods on SMTJTrainingResult."""

    def setUp(self):
        self.model_artifacts = ModelArtifacts(
            output_s3_path="s3://bucket/output/",
            output_model_arn="arn:aws:sagemaker:us-east-1:123456789012:model-package/grp/1",
        )

    @patch("amzn_nova_forge.core.result.training_result.boto3.client")
    def test_wait_raises_for_non_mtrl(self, _mock_client):
        result = SMTJTrainingResult(
            job_id="sft-job",
            started_time=datetime(2026, 1, 1),
            method=TrainingMethod.SFT_LORA,
            model_artifacts=self.model_artifacts,
            model_type=Model.NOVA_LITE_2,
        )
        with self.assertRaises(NotImplementedError):
            result.wait()

    @patch("amzn_nova_forge.core.result.training_result.boto3.client")
    def test_get_training_metrics_raises_for_non_mtrl(self, _mock_client):
        result = SMTJTrainingResult(
            job_id="sft-job",
            started_time=datetime(2026, 1, 1),
            method=TrainingMethod.SFT_LORA,
            model_artifacts=self.model_artifacts,
            model_type=Model.NOVA_LITE_2,
        )
        with self.assertRaises(NotImplementedError):
            result.get_training_metrics()

    @patch("amzn_nova_forge.core.result.training_result.boto3.client")
    def test_wait_delegates_to_rft_job(self, _mock_client):
        import sys
        import types

        mock_agent_module = types.ModuleType("sagemaker.train.agent_rft_job")
        mock_rft_job = Mock()
        mock_rft_job.output_model_package_arn = (
            "arn:aws:sagemaker:us-east-1:123456789012:model-package/grp/2"
        )
        mock_agent_cls = Mock()
        mock_agent_cls.get.return_value = mock_rft_job
        mock_agent_module.AgentRFTJob = mock_agent_cls

        result = SMTJTrainingResult(
            job_id="mtrl-job",
            started_time=datetime(2026, 1, 1),
            method=TrainingMethod.RFT_MULTITURN_LORA,
            model_artifacts=ModelArtifacts(output_s3_path="s3://bucket/output/"),
            model_type=Model.NOVA_LITE_2,
            region="us-east-1",
        )

        with patch.dict("sys.modules", {"sagemaker.train.agent_rft_job": mock_agent_module}):
            result.wait(poll=5, timeout=60)

        mock_rft_job.wait.assert_called_once_with(poll=5, timeout=60)
        mock_rft_job.refresh.assert_called_once()
        self.assertEqual(
            result.model_artifacts.output_model_arn,
            "arn:aws:sagemaker:us-east-1:123456789012:model-package/grp/2",
        )

    @patch("amzn_nova_forge.core.result.training_result.boto3.client")
    def test_get_training_metrics_delegates_to_rft_job(self, _mock_client):
        import sys
        import types

        mock_agent_module = types.ModuleType("sagemaker.train.agent_rft_job")
        mock_rft_job = Mock()
        mock_rft_job.output_model_package_arn = None
        mock_rft_job.get_training_metrics.return_value = [{"step": 1, "reward": 0.5}]
        mock_agent_cls = Mock()
        mock_agent_cls.get.return_value = mock_rft_job
        mock_agent_module.AgentRFTJob = mock_agent_cls

        result = SMTJTrainingResult(
            job_id="mtrl-job",
            started_time=datetime(2026, 1, 1),
            method=TrainingMethod.RFT_MULTITURN_LORA,
            model_artifacts=ModelArtifacts(output_s3_path="s3://bucket/output/"),
            model_type=Model.NOVA_LITE_2,
            region="us-east-1",
        )

        with patch.dict("sys.modules", {"sagemaker.train.agent_rft_job": mock_agent_module}):
            metrics = result.get_training_metrics()

        self.assertEqual(metrics, [{"step": 1, "reward": 0.5}])


class TestMTRLStatusManager(unittest.TestCase):
    """Tests for MTRLStatusManager."""

    def test_get_job_status_in_progress(self):
        import sys
        import types

        from amzn_nova_forge.core.result.job_result import MTRLStatusManager

        mock_agent_module = types.ModuleType("sagemaker.train.agent_rft_job")
        mock_rft_job = Mock()
        mock_rft_job.job_status = "InProgress"
        mock_agent_cls = Mock()
        mock_agent_cls.get.return_value = mock_rft_job
        mock_agent_module.AgentRFTJob = mock_agent_cls

        manager = MTRLStatusManager(region="us-east-1")

        with patch.dict("sys.modules", {"sagemaker.train.agent_rft_job": mock_agent_module}):
            status, raw = manager.get_job_status("mtrl-job-123")

        self.assertEqual(status, JobStatus.IN_PROGRESS)
        self.assertEqual(raw, "InProgress")

    def test_get_job_status_completed(self):
        import sys
        import types

        from amzn_nova_forge.core.result.job_result import MTRLStatusManager

        mock_agent_module = types.ModuleType("sagemaker.train.agent_rft_job")
        mock_rft_job = Mock()
        mock_rft_job.job_status = "Completed"
        mock_agent_cls = Mock()
        mock_agent_cls.get.return_value = mock_rft_job
        mock_agent_module.AgentRFTJob = mock_agent_cls

        manager = MTRLStatusManager(region="us-east-1")

        with patch.dict("sys.modules", {"sagemaker.train.agent_rft_job": mock_agent_module}):
            status, raw = manager.get_job_status("mtrl-job-123")

        self.assertEqual(status, JobStatus.COMPLETED)
        self.assertEqual(raw, "Completed")

    def test_get_job_status_caches_terminal_state(self):
        import sys
        import types

        from amzn_nova_forge.core.result.job_result import MTRLStatusManager

        mock_agent_module = types.ModuleType("sagemaker.train.agent_rft_job")
        mock_rft_job = Mock()
        mock_rft_job.job_status = "Completed"
        mock_agent_cls = Mock()
        mock_agent_cls.get.return_value = mock_rft_job
        mock_agent_module.AgentRFTJob = mock_agent_cls

        manager = MTRLStatusManager(region="us-east-1")

        with patch.dict("sys.modules", {"sagemaker.train.agent_rft_job": mock_agent_module}):
            manager.get_job_status("mtrl-job-123")
            status, raw = manager.get_job_status("mtrl-job-123")

        # Should only call API once due to caching
        mock_agent_cls.get.assert_called_once()
        self.assertEqual(status, JobStatus.COMPLETED)

    def test_resolve_start_time(self):
        import sys
        import types

        from amzn_nova_forge.core.result.job_result import MTRLStatusManager

        mock_agent_module = types.ModuleType("sagemaker.train.agent_rft_job")
        mock_rft_job = Mock()
        mock_rft_job.creation_time = datetime(2026, 5, 20, 10, 0, 0)
        mock_agent_cls = Mock()
        mock_agent_cls.get.return_value = mock_rft_job
        mock_agent_module.AgentRFTJob = mock_agent_cls

        manager = MTRLStatusManager(region="us-east-1")

        with patch.dict("sys.modules", {"sagemaker.train.agent_rft_job": mock_agent_module}):
            start_time = manager.resolve_start_time("mtrl-job-123")

        self.assertEqual(start_time, datetime(2026, 5, 20, 10, 0, 0))


class TestMTRLLogMonitor(unittest.TestCase):
    """Tests for MTRLLogMonitor."""

    def test_from_job_id(self):
        from amzn_nova_forge.monitor.log_monitor import MTRLLogMonitor

        monitor = MTRLLogMonitor.from_job_id(job_id="mtrl-job-123", region="us-east-1")
        self.assertEqual(monitor.job_id, "mtrl-job-123")
        self.assertEqual(monitor._region, "us-east-1")

    def test_show_logs_completed_job(self):
        import sys
        import types

        from amzn_nova_forge.monitor.log_monitor import MTRLLogMonitor

        mock_agent_module = types.ModuleType("sagemaker.train.agent_rft_job")
        mock_rft_job = Mock()
        mock_rft_job.job_status = "Completed"
        mock_agent_cls = Mock()
        mock_agent_cls.get.return_value = mock_rft_job
        mock_agent_module.AgentRFTJob = mock_agent_cls

        monitor = MTRLLogMonitor(job_id="mtrl-job-123", region="us-east-1")

        with patch.dict("sys.modules", {"sagemaker.train.agent_rft_job": mock_agent_module}):
            monitor.show_logs()

        mock_rft_job.refresh.assert_called_once()
        mock_rft_job.get_training_metrics.assert_called_once()
        mock_rft_job.wait.assert_not_called()

    def test_show_logs_running_job_calls_wait(self):
        import sys
        import types

        from amzn_nova_forge.monitor.log_monitor import MTRLLogMonitor

        mock_agent_module = types.ModuleType("sagemaker.train.agent_rft_job")
        mock_rft_job = Mock()
        mock_rft_job.job_status = "InProgress"
        mock_agent_cls = Mock()
        mock_agent_cls.get.return_value = mock_rft_job
        mock_agent_module.AgentRFTJob = mock_agent_cls

        monitor = MTRLLogMonitor(job_id="mtrl-job-123", region="us-east-1")

        with patch.dict("sys.modules", {"sagemaker.train.agent_rft_job": mock_agent_module}):
            monitor.show_logs(poll=10, timeout=120)

        mock_rft_job.wait.assert_called_once_with(poll=10, timeout=120)


if __name__ == "__main__":
    unittest.main()
