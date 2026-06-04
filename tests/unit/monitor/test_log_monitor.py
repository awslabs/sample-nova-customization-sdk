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
from datetime import datetime
from unittest.mock import Mock, patch

from amzn_nova_forge.core.enums import Platform, TrainingMethod
from amzn_nova_forge.core.result.job_result import (
    BaseJobResult,
    SMHPStatusManager,
    SMTJStatusManager,
)
from amzn_nova_forge.monitor.log_monitor import (
    BedrockStrategy,
    CloudWatchLogMonitor,
    SMHPStrategy,
)


class TestCloudWatchLogMonitor(unittest.TestCase):
    def setUp(self):
        self._jsm_patcher = patch.object(
            CloudWatchLogMonitor, "_create_job_status_manager", return_value=Mock()
        )
        self._jsm_patcher.start()
        self.mock_client = Mock()
        self.job_id = "test-job-123"
        self.platform = Platform.SMTJ
        self.started_time = 1234567890

    def tearDown(self):
        self._jsm_patcher.stop()

    def test_init_smtj_platform(self):
        self.mock_client.describe_log_streams.return_value = {
            "logStreams": [{"logStreamName": "test-job-123/algo-1-1234567890"}]
        }

        monitor = CloudWatchLogMonitor(
            job_id=self.job_id,
            platform=self.platform,
            started_time=self.started_time,
            cloudwatch_logs_client=self.mock_client,
        )

        self.assertEqual(monitor.job_id, self.job_id)
        self.assertEqual(monitor.platform, self.platform)
        self.assertEqual(monitor.log_group_name, "/aws/sagemaker/TrainingJobs")
        self.assertEqual(monitor.log_stream_name, "test-job-123/algo-1-1234567890")

    def test_find_log_stream_no_streams(self):
        self.mock_client.describe_log_streams.return_value = {"logStreams": []}

        monitor = CloudWatchLogMonitor(
            job_id=self.job_id,
            platform=self.platform,
            started_time=self.started_time,
            cloudwatch_logs_client=self.mock_client,
        )

        self.assertIsNone(monitor.log_stream_name)

    def test_get_logs_no_stream(self):
        self.mock_client.describe_log_streams.return_value = {"logStreams": []}

        monitor = CloudWatchLogMonitor(
            job_id=self.job_id,
            platform=self.platform,
            started_time=self.started_time,
            cloudwatch_logs_client=self.mock_client,
        )
        result = monitor.get_logs()

        self.assertEqual(result, [])

    def test_get_logs_basic(self):
        self.mock_client.describe_log_streams.return_value = {
            "logStreams": [{"logStreamName": "test-job-123/algo-1-1234567890"}]
        }
        self.mock_client.get_log_events.side_effect = [
            {
                "events": [{"message": "log1"}, {"message": "log2"}],
                "nextBackwardToken": "token1",
            },
            {"events": [], "nextBackwardToken": "token1"},
        ]

        monitor = CloudWatchLogMonitor(
            job_id=self.job_id,
            platform=self.platform,
            started_time=self.started_time,
            cloudwatch_logs_client=self.mock_client,
        )
        result = monitor.get_logs()

        self.assertEqual(len(result), 2)
        self.assertEqual(result[0]["message"], "log1")
        self.assertEqual(self.mock_client.get_log_events.call_count, 2)

    def test_get_logs_with_limit(self):
        self.mock_client.describe_log_streams.return_value = {
            "logStreams": [{"logStreamName": "test-job-123/algo-1-1234567890"}]
        }
        self.mock_client.get_log_events.return_value = {
            "events": [{"message": "log1"}, {"message": "log2"}, {"message": "log3"}],
            "nextBackwardToken": "token1",
        }

        monitor = CloudWatchLogMonitor(
            job_id=self.job_id,
            platform=self.platform,
            started_time=self.started_time,
            cloudwatch_logs_client=self.mock_client,
        )
        result = monitor.get_logs(limit=2)

        self.assertEqual(len(result), 2)

    def test_get_logs_pagination(self):
        self.mock_client.describe_log_streams.return_value = {
            "logStreams": [{"logStreamName": "test-job-123/algo-1-1234567890"}]
        }
        self.mock_client.get_log_events.side_effect = [
            {"events": [{"message": "log1"}], "nextBackwardToken": "token2"},
            {"events": [{"message": "log2"}], "nextBackwardToken": "token2"},
        ]

        monitor = CloudWatchLogMonitor(
            job_id=self.job_id,
            platform=self.platform,
            started_time=self.started_time,
            cloudwatch_logs_client=self.mock_client,
        )
        result = monitor.get_logs()

        self.assertEqual(len(result), 2)
        self.assertEqual(self.mock_client.get_log_events.call_count, 2)

    @patch("builtins.print")
    def test_show_logs_with_events(self, mock_print):
        self.mock_client.describe_log_streams.return_value = {
            "logStreams": [{"logStreamName": "test-job-123/algo-1-1234567890"}]
        }
        self.mock_client.get_log_events.side_effect = [
            {
                "events": [{"message": "log1\n"}, {"message": "log2\n"}],
                "nextBackwardToken": "token1",
            },
            {"events": [], "nextBackwardToken": "token1"},
        ]

        monitor = CloudWatchLogMonitor(
            job_id=self.job_id,
            platform=self.platform,
            started_time=self.started_time,
            cloudwatch_logs_client=self.mock_client,
        )
        monitor.show_logs()

        mock_print.assert_any_call("log1")
        mock_print.assert_any_call("log2")
        self.assertEqual(mock_print.call_count, 2)

    @patch("builtins.print")
    def test_show_logs_no_events(self, mock_print):
        self.mock_client.describe_log_streams.return_value = {"logStreams": []}

        monitor = CloudWatchLogMonitor(
            job_id=self.job_id,
            platform=self.platform,
            started_time=self.started_time,
            cloudwatch_logs_client=self.mock_client,
        )
        monitor.show_logs()

        mock_print.assert_called_once_with(f"No logs found for job {self.job_id} yet")

    def test_get_logs_with_start_and_end_time(self):
        self.mock_client.describe_log_streams.return_value = {
            "logStreams": [{"logStreamName": "test-job-123/algo-1-1234567890"}]
        }
        self.mock_client.get_log_events.return_value = {
            "events": [{"message": "log1"}],
            "nextBackwardToken": "token1",
        }

        monitor = CloudWatchLogMonitor(
            job_id=self.job_id,
            platform=self.platform,
            started_time=self.started_time,
            cloudwatch_logs_client=self.mock_client,
        )
        end_time = 1234567999
        monitor.get_logs(end_time=end_time)

        self.mock_client.get_log_events.assert_called_with(
            endTime=end_time,
            logGroupName="/aws/sagemaker/TrainingJobs",
            logStreamName="test-job-123/algo-1-1234567890",
            startFromHead=False,
            startTime=self.started_time,
            nextToken="token1",
        )

    def test_from_job_id_smtj(self):
        job_id = "test-job-456"
        platform = Platform.SMTJ
        started_time = datetime(2023, 1, 1, 12, 0, 0)

        with patch.object(CloudWatchLogMonitor, "__init__", return_value=None) as mock_init:
            monitor = CloudWatchLogMonitor.from_job_id(
                job_id=job_id, platform=platform, started_time=started_time
            )

            mock_init.assert_called_once_with(
                job_id=job_id,
                platform=platform,
                started_time=int(started_time.timestamp() * 1000),
            )

    def test_from_job_id_smhp(self):
        job_id = "test-job-789"
        platform = Platform.SMHP
        cluster_name = "test-cluster"
        namespace = "test-namespace"

        with patch.object(CloudWatchLogMonitor, "__init__", return_value=None) as mock_init:
            monitor = CloudWatchLogMonitor.from_job_id(
                job_id=job_id,
                platform=platform,
                cluster_name=cluster_name,
                namespace=namespace,
            )

            mock_init.assert_called_once_with(
                job_id=job_id,
                platform=platform,
                started_time=None,
                cluster_name=cluster_name,
                namespace=namespace,
            )

    def test_from_job_result_smtj(self):
        mock_job_result = Mock(spec=BaseJobResult)
        mock_job_result.job_id = "smtj-job-123"
        mock_job_result.platform = Platform.SMTJ
        mock_job_result.started_time = datetime(2023, 1, 1, 12, 0, 0)

        mock_client = Mock()

        with patch.object(CloudWatchLogMonitor, "__init__", return_value=None) as mock_init:
            monitor = CloudWatchLogMonitor.from_job_result(
                job_result=mock_job_result, cloudwatch_logs_client=mock_client
            )

            mock_init.assert_called_once_with(
                job_id="smtj-job-123",
                platform=Platform.SMTJ,
                started_time=int(mock_job_result.started_time.timestamp() * 1000),
                cloudwatch_logs_client=mock_client,
                region=None,
            )

    def test_from_job_result_smhp(self):
        mock_job_result = Mock(spec=BaseJobResult)
        mock_job_result.job_id = "smhp-job-456"
        mock_job_result.platform = Platform.SMHP
        mock_job_result.started_time = datetime(2023, 1, 1, 12, 0, 0)

        mock_status_manager = Mock(spec=SMHPStatusManager)
        mock_status_manager.cluster_name = "test-cluster"
        mock_status_manager.namespace = "test-namespace"
        mock_job_result.status_manager = mock_status_manager

        mock_client = Mock()

        with patch.object(CloudWatchLogMonitor, "__init__", return_value=None) as mock_init:
            monitor = CloudWatchLogMonitor.from_job_result(
                job_result=mock_job_result, cloudwatch_logs_client=mock_client
            )

            mock_init.assert_called_once_with(
                job_id="smhp-job-456",
                platform=Platform.SMHP,
                started_time=int(mock_job_result.started_time.timestamp() * 1000),
                cloudwatch_logs_client=mock_client,
                cluster_name="test-cluster",
                namespace="test-namespace",
                region=None,
            )

    @patch("boto3.client")
    def test_bedrock_get_log_group_name_returns_bedrock_path(self, mock_boto_client):
        """Test that get_log_group_name returns Bedrock log group path.

        Even though Bedrock doesn't use CloudWatch logs, this method
        returns a path for interface compatibility.
        """
        from amzn_nova_forge.monitor.log_monitor import BedrockStrategy

        strategy = BedrockStrategy()
        result = strategy.get_log_group_name("test-job-id")

        self.assertEqual(result, "/aws/bedrock/modelcustomizationjobs")

    @patch("boto3.client")
    def test_bedrock_find_log_stream_returns_none(self, mock_boto_client):
        """Test that find_log_stream returns None for Bedrock jobs.

        Bedrock customization jobs don't create CloudWatch log streams.
        """
        from amzn_nova_forge.monitor.log_monitor import BedrockStrategy

        strategy = BedrockStrategy()
        mock_client = Mock()

        result = strategy.find_log_stream(
            "test-job-id", mock_client, "/aws/bedrock/modelcustomizationjobs"
        )

        self.assertIsNone(result)

    @patch("amzn_nova_forge.monitor.log_monitor.get_bedrock_job_details")
    @patch("amzn_nova_forge.monitor.log_monitor.log_bedrock_job_status")
    def test_bedrock_get_logs_returns_empty_list_and_displays_status(
        self, mock_log_status, mock_get_details
    ):
        """Test that get_logs returns empty list and displays job status.

        Since Bedrock doesn't stream logs, this method displays the current
        job status instead and returns an empty list for interface compatibility.
        """
        from amzn_nova_forge.monitor.log_monitor import BedrockStrategy

        # Mock Bedrock job details
        mock_get_details.return_value = {
            "status": "InProgress",
            "jobName": "test-job",
            "jobArn": "arn:aws:bedrock:us-east-1:123456789012:model-customization-job/test-job",
        }

        # Create strategy with mock client
        mock_bedrock_client = Mock()
        strategy = BedrockStrategy(bedrock_client=mock_bedrock_client)

        # Call get_logs
        result = strategy.get_logs(
            job_id="arn:aws:bedrock:us-east-1:123456789012:model-customization-job/test-job",
            cloudwatch_logs_client=Mock(),
            log_group_name="/aws/bedrock/modelcustomizationjobs",
            log_stream_name=None,
            limit=100,
            start_from_head=True,
        )

        # Verify empty list is returned
        self.assertEqual(result, [])

        # Verify job details were fetched
        mock_get_details.assert_called_once_with(
            mock_bedrock_client,
            "arn:aws:bedrock:us-east-1:123456789012:model-customization-job/test-job",
        )

        # Verify job status was logged
        mock_log_status.assert_called_once()

    @patch("amzn_nova_forge.monitor.log_monitor.get_bedrock_job_details")
    def test_bedrock_get_logs_handles_error_gracefully(self, mock_get_details):
        """Test that get_logs handles errors when fetching job status.

        If there's an error retrieving job status, it should be logged
        and an empty list should still be returned.
        """
        from amzn_nova_forge.monitor.log_monitor import BedrockStrategy

        # Mock error when getting job details
        mock_get_details.side_effect = Exception("API error")

        # Create strategy with mock client
        mock_bedrock_client = Mock()
        strategy = BedrockStrategy(bedrock_client=mock_bedrock_client)

        # Call get_logs - should not raise exception
        result = strategy.get_logs(
            job_id="test-job-id",
            cloudwatch_logs_client=Mock(),
            log_group_name="/aws/bedrock/modelcustomizationjobs",
            log_stream_name=None,
            limit=100,
            start_from_head=True,
        )

        # Verify empty list is returned even with error
        self.assertEqual(result, [])

    def test_bedrock_strategy_uses_default_client_if_not_provided(self):
        """Test that BedrockStrategy creates default client if none provided."""
        from amzn_nova_forge.monitor.log_monitor import BedrockStrategy

        with patch("boto3.client") as mock_boto_client:
            mock_client = Mock()
            mock_boto_client.return_value = mock_client

            strategy = BedrockStrategy()

            # Verify boto3.client was called to create bedrock client
            mock_boto_client.assert_called_once_with("bedrock", region_name=None)
            self.assertEqual(strategy.bedrock_client, mock_client)

    def test_bedrock_strategy_uses_provided_client(self):
        """Test that BedrockStrategy uses provided client if given."""
        from amzn_nova_forge.monitor.log_monitor import BedrockStrategy

        mock_client = Mock()
        strategy = BedrockStrategy(bedrock_client=mock_client)

        self.assertEqual(strategy.bedrock_client, mock_client)

    def test_get_metrics_smhp_sft_1(self):
        mock_sagemaker = Mock()
        mock_sagemaker.describe_cluster.return_value = {
            "ClusterArn": "arn:aws:sagemaker:us-west-2:123456789012:cluster/test-cluster"
        }

        monitor = CloudWatchLogMonitor(
            job_id=self.job_id,
            platform=Platform.SMHP,
            cluster_name="test-cluster",
            cloudwatch_logs_client=self.mock_client,
            sagemaker_client=mock_sagemaker,
        )

        training_method = TrainingMethod.SFT_FULL
        logs = [
            {
                "message": "[my-full-rank-sft-1] [2025-12-17 04:54:49,774] [INFO] "
                "Epoch 1: : 1it [00:04,  4.36s/it, v_num=d3c7, reduced_train_loss=0.674, "
                "global_step=1.000, consumed_samples=128.0, train_step_timing in s=3.370, "
                "batch_time=4.400, samples/sec=14.60]     2025-12-17 04:54:49,774 [INFO] "
                "- nlp_overrides.py:552 - Time elapsed dumping optimizer_states with gather_on_root=True "
                "and state_dict_format=None: 0.22 seconds"
            }
        ]
        metrics = ["training_loss"]

        result = monitor.strategy.get_metrics(training_method, logs, metrics)

        self.assertEqual(len(result), 1)
        self.assertEqual(list(result.columns), ["global_step", "training_loss"])
        self.assertEqual(result.iloc[0]["global_step"], 1)
        self.assertAlmostEqual(result.iloc[0]["training_loss"], 0.674)

    def test_get_metrics_smhp_sft_2(self):
        mock_sagemaker = Mock()
        mock_sagemaker.describe_cluster.return_value = {
            "ClusterArn": "arn:aws:sagemaker:us-west-2:123456789012:cluster/test-cluster"
        }

        monitor = CloudWatchLogMonitor(
            job_id=self.job_id,
            platform=Platform.SMHP,
            cluster_name="test-cluster",
            cloudwatch_logs_client=self.mock_client,
            sagemaker_client=mock_sagemaker,
        )

        training_method = TrainingMethod.SFT_FULL
        logs = [
            {
                "message": "[my-full-rank-sft-2] [2025-12-15 22:39:23,107] [INFO] "
                "Training epoch 0, iteration 99/99 | lr: 1.003e-06 | global_batch_size: 32 | "
                "global_step: 99 | reduced_train_loss: 0.7932 | train_step_timing in s: 73.79 | "
                "consumed_samples: 3200"
            }
        ]
        metrics = ["training_loss"]

        result = monitor.strategy.get_metrics(training_method, logs, metrics)

        self.assertEqual(len(result), 1)
        self.assertEqual(list(result.columns), ["global_step", "training_loss"])
        self.assertEqual(result.iloc[0]["global_step"], 99)
        self.assertAlmostEqual(result.iloc[0]["training_loss"], 0.7932)

    def test_get_metrics_smhp_cpt_1(self):
        mock_sagemaker = Mock()
        mock_sagemaker.describe_cluster.return_value = {
            "ClusterArn": "arn:aws:sagemaker:us-west-2:123456789012:cluster/cluster-id-123"
        }

        monitor = CloudWatchLogMonitor(
            job_id=self.job_id,
            platform=Platform.SMHP,
            cluster_name="test-cluster",
            cloudwatch_logs_client=self.mock_client,
            sagemaker_client=mock_sagemaker,
        )

        training_method = TrainingMethod.CPT
        logs = [
            {
                "message": (
                    "[my-cpt-1] [2026-01-07 17:48:18,810] [INFO]  "
                    "Epoch 0:   0%|          | 3/5000000 [01:42<47225:24:50, 34.00s/it, "
                    "v_num=3136, reduced_train_loss=2.230, global_step=1.000, "
                    "consumed_samples=512.0, train_step_timing in s=10.70, "
                    "batch_time=2.17e+5, samples/sec=0.00118]\n"
                    "Epoch 0:   0%|          | 3/5000000 [01:42<47225:38:33, 34.00s/it, "
                    "v_num=3136, reduced_train_loss=2.150, global_step=2.000, "
                    "consumed_samples=768.0, train_step_timing in s=10.70, "
                    "batch_time=11.70, samples/sec=21.90]\n"
                    "Epoch 0:   0%|          | 4/5000000 [01:53<39461:34:31, 28.41s/it, "
                    "v_num=3136, reduced_train_loss=2.150, global_step=2.000, "
                    "consumed_samples=768.0, train_step_timing in s=10.70, "
                    "batch_time=11.70, samples/sec=21.90]\n"
                    "Epoch 0:   0%|          | 4/5000000 [01:53<39461:44:27, 28.41s/it, "
                    "v_num=3136, reduced_train_loss=2.160, global_step=3.000, "
                    "consumed_samples=1024.0, train_step_timing in s=10.90, "
                    "batch_time=11.40, samples/sec=22.40]\n"
                    "Epoch 0:   0%|          | 5/5000000 [02:05<34824:57:09, 25.07s/it, "
                    "v_num=3136, reduced_train_loss=2.160, global_step=3.000, "
                    "consumed_samples=1024.0, train_step_timing in s=10.90, "
                    "batch_time=11.40, samples/sec=22.40]\n"
                    "Epoch 0:   0%|          | 5/5000000 [02:05<34825:06:26, 25.07s/it, "
                    "v_num=3136, reduced_train_loss=2.080, global_step=4.000, "
                    "consumed_samples=1280.0, train_step_timing in s=10.70, "
                    "batch_time=11.90, samples/sec=21.50]\n"
                    "Epoch 0:   0%|          | 6/5000000 [02:16<31702:09:10, 22.83s/it, "
                    "v_num=3136, reduced_train_loss=2.080, global_step=4.000, "
                    "consumed_samples=1280.0, train_step_timing in s=10.70, "
                    "batch_time=11.90, samples/sec=21.50]\n"
                    "Epoch 0:   0%|          | 6/5000000 [02:16<31702:16:08, 22.83s/it, "
                    "v_num=3136, reduced_train_loss=2.060, global_step=5.000, "
                    "consumed_samples=1536.0, train_step_timing in s=10.80, "
                    "batch_time=11.50, samples/sec=22.30]\n"
                    "Epoch 0:   0%|          | 7/5000000 [02:28<29501:46:49, 21.24s/it, "
                    "v_num=3136, reduced_train_loss=2.060, global_step=5.000, "
                    "consumed_samples=1536.0, train_step_timing in s=10.80, "
                    "batch_time=11.50, samples/sec=22.30]\n"
                    "Epoch 0:   0%|          | 7/5000000 [02:28<29501:52:23, 21.24s/it, "
                    "v_num=3136, reduced_train_loss=2.040, global_step=6.000, "
                    "consumed_samples=1792.0, train_step_timing in s=10.70, "
                    "batch_time=11.80, samples/sec=21.60]"
                )
            }
        ]
        metrics = ["training_loss"]

        result = monitor.strategy.get_metrics(training_method, logs, metrics)

        self.assertEqual(len(result), 10)
        self.assertEqual(list(result.columns), ["global_step", "training_loss"])
        self.assertEqual(result.iloc[0]["global_step"], 1)
        self.assertAlmostEqual(result.iloc[0]["training_loss"], 2.230)
        self.assertEqual(result.iloc[-1]["global_step"], 6)
        self.assertAlmostEqual(result.iloc[-1]["training_loss"], 2.040)

    def test_get_metrics_smhp_cpt_2(self):
        mock_sagemaker = Mock()
        mock_sagemaker.describe_cluster.return_value = {
            "ClusterArn": "arn:aws:sagemaker:us-west-2:123456789012:cluster/test-cluster"
        }

        monitor = CloudWatchLogMonitor(
            job_id=self.job_id,
            platform=Platform.SMHP,
            cluster_name="test-cluster",
            cloudwatch_logs_client=self.mock_client,
            sagemaker_client=mock_sagemaker,
        )

        training_method = TrainingMethod.CPT
        logs = [
            {
                "message": "[my-cpt-2] [2025-12-18 02:45:53,585] [INFO] "
                "Training epoch 0, iteration 9/9 | lr: 9.091e-06 | dl_wait_time: 0.0002742 "
                "| padding_percentage: 0.8996 | global_step: 9 | token_count: 20971520 | "
                "reduced_train_loss: 10.54 | training_loss_step: 10.54 | grad_norm: 161.8 | "
                "train_step_timing in s: 59.7 | batch_time: 59.71 | samples/sec: 4.287 | "
                "avg_samples/sec: nan | peak_samples/sec: 4.288"
            }
        ]
        metrics = ["training_loss"]

        result = monitor.strategy.get_metrics(training_method, logs, metrics)

        self.assertEqual(len(result), 1)
        self.assertEqual(list(result.columns), ["global_step", "training_loss"])
        self.assertEqual(result.iloc[0]["global_step"], 9)
        self.assertAlmostEqual(result.iloc[0]["training_loss"], 10.54)

    def test_get_metrics_smhp_rft(self):
        mock_sagemaker = Mock()
        mock_sagemaker.describe_cluster.return_value = {
            "ClusterArn": "arn:aws:sagemaker:us-west-2:123456789012:cluster/test-cluster"
        }

        monitor = CloudWatchLogMonitor(
            job_id=self.job_id,
            platform=Platform.SMHP,
            cluster_name="test-cluster",
            cloudwatch_logs_client=self.mock_client,
            sagemaker_client=mock_sagemaker,
        )

        training_method = TrainingMethod.RFT_LORA
        logs = [
            {
                "message": "[my-lora-rft] [2026-03-08 22:40:07,102] [INFO] "
                "Training epoch 0, iteration 119/999 | lr: 7e-07 | "
                "global_batch_size: 256 | global_step: 119 | "
                "reduced_train_loss: 0.007075 | prompt_length: 7.26e+03 | "
                "completion_length: 635.2 | global_seq_length: 7.895e+03 | "
                "adv: 0.02718 | reasoning_length_size: 0 | "
                "train_rm_score: 0.6797 | kl: 0.0008378 | ent: 0.1944 | "
                "clipfrac: 0.004458 | train_step_timing in s: 1.232e+03 | "
                "consumed_samples: 30720"
            }
        ]
        metrics = ["reward_score"]

        result = monitor.strategy.get_metrics(training_method, logs, metrics)

        self.assertEqual(len(result), 1)
        self.assertEqual(list(result.columns), ["global_step", "reward_score"])
        self.assertEqual(result.iloc[0]["global_step"], 119)
        self.assertAlmostEqual(result.iloc[0]["reward_score"], 0.6797)

    def test_get_metrics_smtj_rft(self):
        self.mock_client.describe_log_streams.return_value = {
            "logStreams": [{"logStreamName": "test-job-123/algo-1-1234567890"}]
        }

        monitor = CloudWatchLogMonitor(
            job_id=self.job_id,
            platform=Platform.SMTJ,
            started_time=self.started_time,
            cloudwatch_logs_client=self.mock_client,
        )

        training_method = TrainingMethod.RFT_LORA
        logs = [
            {
                "message": (
                    "[2026-02-13 15:15:07,022] [INFO] \033[36m(TaskRunner pid=12647)\033[0m "
                    "step:295 - global_seqlen/min:2087 - global_seqlen/max:33220 - "
                    "global_seqlen/minmax_diff:31133 - global_seqlen/balanced_min:3118 - "
                    "global_seqlen/balanced_max:10160 - global_seqlen/mean:4009.21875 - "
                    "actor/entropy:2.542900800704956 - "
                    "actor/reward_kl_penalty:-0.29242539405822754 - "
                    "actor/reward_kl_penalty_coeff:0.02 - actor/pg_loss:0.0 - "
                    "actor/pg_clipfrac:0.0 - actor/ppo_kl:0.0 - "
                    "actor/pg_clipfrac_lower:0.0 - actor/grad_norm:0.3703329265117645 - "
                    "perf/mfu/actor:0.0 - perf/max_memory_allocated_gb:80.45536231994629 - "
                    "perf/max_memory_reserved_gb:84.939453125 - "
                    "perf/cpu_memory_used_gb:354.0128479003906 - actor/lr:1e-05 - "
                    "training/global_step:295 - training/epoch:5 - "
                    "critic/score/mean:0.8046875 - critic/score/max:1.0 - "
                    "critic/score/min:-1.0 - "
                    "critic/rewards/mean:0.8313651084899902 - "
                    "critic/rewards/max:1.097672462463379 - "
                    "critic/rewards/min:-0.9990701675415039 - "
                    "critic/advantages/mean:0.00265043624676764 - "
                    "critic/advantages/max:1.4999932050704956 - "
                    "critic/advantages/min:-1.4999825954437256 - "
                    "critic/returns/mean:0.00265043624676764 - "
                    "critic/returns/max:1.4999932050704956 - "
                    "critic/returns/min:-1.4999825954437256 - "
                    "response_length/mean:4.76171875 - response_length/max:5.0 - "
                    "response_length/min:4.0 - response_length/clip_ratio:0.0 - "
                    "perf/throughput:38.81311253930573"
                )
            }
        ]
        metrics = ["reward_score"]

        result = monitor.strategy.get_metrics(training_method, logs, metrics)

        self.assertEqual(len(result), 1)
        self.assertEqual(list(result.columns), ["global_step", "reward_score"])
        self.assertEqual(result.iloc[0]["global_step"], 295)
        self.assertAlmostEqual(result.iloc[0]["reward_score"], 0.8313651084899902)

    def test_get_metrics_smtj_sft_1(self):
        self.mock_client.describe_log_streams.return_value = {
            "logStreams": [{"logStreamName": "test-job-123/algo-1-1234567890"}]
        }

        monitor = CloudWatchLogMonitor(
            job_id=self.job_id,
            platform=Platform.SMTJ,
            started_time=self.started_time,
            cloudwatch_logs_client=self.mock_client,
        )

        training_method = TrainingMethod.SFT_FULL
        logs = [
            {
                "message": (
                    "[2026-03-01 21:16:40,581] [INFO] #015Epoch 1: : "
                    "3it [00:32, 10.68s/it, v_num=1feb, reduced_train_loss=2.66e-5, global_step=6.000, "
                    "consumed_samples=448.0, train_step_timing in s=10.10, batch_time=10.80, samples/sec=5.920]{\n"
                    '"timestamp": "2026-03-01 21:16:40",\n"epoch": 1,'
                    '"stepId": 2,\n"stepType": "TRAIN",'
                    '"totalProcessedTokenCount": 3402\n}'
                )
            }
        ]
        metrics = ["training_loss"]

        result = monitor.strategy.get_metrics(training_method, logs, metrics)

        self.assertEqual(len(result), 1)
        self.assertEqual(list(result.columns), ["global_step", "training_loss"])
        self.assertEqual(result.iloc[0]["global_step"], 6)
        self.assertAlmostEqual(result.iloc[0]["training_loss"], 0.0000266)

    def test_get_metrics_smtj_sft_2(self):
        self.mock_client.describe_log_streams.return_value = {
            "logStreams": [{"logStreamName": "test-job-123/algo-1-1234567890"}]
        }

        monitor = CloudWatchLogMonitor(
            job_id=self.job_id,
            platform=Platform.SMTJ,
            started_time=self.started_time,
            cloudwatch_logs_client=self.mock_client,
        )

        training_method = TrainingMethod.SFT_FULL
        logs = [
            {
                "message": (
                    "[2026-01-28 17:48:54,839] [INFO] Training epoch 0, iteration 8/9 | lr: 5.625e-06 "
                    "| global_batch_size: 32 | global_step: 8 | reduced_train_loss: 0.0013 | "
                    "train_step_timing in s: 75.02 | consumed_samples: 28"
                )
            }
        ]
        metrics = ["training_loss"]

        result = monitor.strategy.get_metrics(training_method, logs, metrics)

        self.assertEqual(len(result), 1)
        self.assertEqual(list(result.columns), ["global_step", "training_loss"])
        self.assertEqual(result.iloc[0]["global_step"], 8)
        self.assertAlmostEqual(result.iloc[0]["training_loss"], 0.0013)

    def _make_monitor_and_df(self):
        """Helper to create a monitor and a sample metrics DataFrame."""
        self.mock_client.describe_log_streams.return_value = {
            "logStreams": [{"logStreamName": "test-job-123/algo-1-1234567890"}]
        }
        monitor = CloudWatchLogMonitor(
            job_id=self.job_id,
            platform=Platform.SMTJ,
            started_time=self.started_time,
            cloudwatch_logs_client=self.mock_client,
        )
        import pandas

        df = pandas.DataFrame(
            {
                "global_step": [1, 2, 2, 3, 4, 4, 5, 6, 7, 8, 9, 10],
                "training_loss": [
                    -0.9,
                    -0.8,
                    -0.8,
                    -0.7,
                    0.6,
                    0.6,
                    0.5,
                    0.4,
                    0.3,
                    0.2,
                    0.1,
                    0.05,
                ],
            }
        )
        return monitor, df

    def test_get_in_range_dataframe_both_bounds(self):
        monitor, df = self._make_monitor_and_df()
        result = monitor._get_in_range_dataframe(df, starting_step=3, ending_step=7)
        self.assertEqual(list(result["global_step"]), [3, 4, 5, 6, 7])
        self.assertEqual(list(result["training_loss"]), [-0.7, 0.6, 0.5, 0.4, 0.3])

    def test_get_in_range_dataframe_starting_step_only(self):
        monitor, df = self._make_monitor_and_df()
        result = monitor._get_in_range_dataframe(df, starting_step=8)
        self.assertEqual(list(result["global_step"]), [8, 9, 10])

    def test_get_in_range_dataframe_ending_step_only(self):
        monitor, df = self._make_monitor_and_df()
        result = monitor._get_in_range_dataframe(df, ending_step=3)
        self.assertEqual(list(result["global_step"]), [1, 2, 3])

    def test_get_in_range_dataframe_no_bounds(self):
        monitor, df = self._make_monitor_and_df()
        result = monitor._get_in_range_dataframe(df)
        self.assertEqual(list(result["global_step"]), [1, 2, 3, 4, 5, 6, 7, 8, 9, 10])

    def test_get_in_range_dataframe_single_row_match(self):
        monitor, df = self._make_monitor_and_df()
        result = monitor._get_in_range_dataframe(df, starting_step=5, ending_step=5)
        self.assertEqual(list(result["global_step"]), [5])

    def test_get_in_range_dataframe_no_rows_in_range(self):
        monitor, df = self._make_monitor_and_df()
        with self.assertRaises(ValueError):
            monitor._get_in_range_dataframe(df, starting_step=20, ending_step=30)

    def test_get_metrics_smtj_sft_no_metrics_passed(self):
        self.mock_client.describe_log_streams.return_value = {
            "logStreams": [{"logStreamName": "test-job-123/algo-1-1234567890"}]
        }
        monitor = CloudWatchLogMonitor(
            job_id=self.job_id,
            platform=Platform.SMTJ,
            started_time=self.started_time,
            cloudwatch_logs_client=self.mock_client,
        )
        logs = [{"message": "global_step: 1 | reduced_train_loss: 0.5"}]
        result = monitor.strategy.get_metrics(TrainingMethod.SFT_FULL, logs)
        self.assertEqual(list(result.columns), ["global_step", "training_loss"])

    def test_get_metrics_smtj_cpt_no_metrics_passed(self):
        self.mock_client.describe_log_streams.return_value = {
            "logStreams": [{"logStreamName": "test-job-123/algo-1-1234567890"}]
        }
        monitor = CloudWatchLogMonitor(
            job_id=self.job_id,
            platform=Platform.SMTJ,
            started_time=self.started_time,
            cloudwatch_logs_client=self.mock_client,
        )
        logs = [{"message": "global_step: 1 | reduced_train_loss: 0.5"}]
        result = monitor.strategy.get_metrics(TrainingMethod.CPT, logs)
        self.assertEqual(list(result.columns), ["global_step", "training_loss"])

    def test_get_metrics_smtj_rft_no_metrics_passed(self):
        self.mock_client.describe_log_streams.return_value = {
            "logStreams": [{"logStreamName": "test-job-123/algo-1-1234567890"}]
        }
        monitor = CloudWatchLogMonitor(
            job_id=self.job_id,
            platform=Platform.SMTJ,
            started_time=self.started_time,
            cloudwatch_logs_client=self.mock_client,
        )
        logs = [{"message": "global_step: 1 | reward_score: 0.75"}]
        result = monitor.strategy.get_metrics(TrainingMethod.RFT_FULL, logs)
        self.assertEqual(list(result.columns), ["global_step", "reward_score"])

    def test_get_metrics_smhp_no_metrics_passed(self):
        mock_sagemaker = Mock()
        mock_sagemaker.describe_cluster.return_value = {
            "ClusterArn": "arn:aws:sagemaker:us-east-1:123456789012:cluster/test"
        }
        monitor = CloudWatchLogMonitor(
            job_id=self.job_id,
            platform=Platform.SMHP,
            started_time=self.started_time,
            cloudwatch_logs_client=self.mock_client,
            cluster_name="test-cluster",
            namespace="kubeflow",
            sagemaker_client=mock_sagemaker,
        )
        logs = [{"message": "Step: 1 | training_loss: 0.5"}]
        sft_result = monitor.strategy.get_metrics(TrainingMethod.SFT_FULL, logs)
        self.assertEqual(list(sft_result.columns), ["global_step", "training_loss"])
        cpt_result = monitor.strategy.get_metrics(TrainingMethod.CPT, logs)
        self.assertEqual(list(cpt_result.columns), ["global_step", "training_loss"])
        rft_result = monitor.strategy.get_metrics(TrainingMethod.RFT_LORA, logs)
        self.assertEqual(list(rft_result.columns), ["global_step", "reward_score"])

    def test_get_metrics_smtj_unsupported_metric_raises(self):
        self.mock_client.describe_log_streams.return_value = {
            "logStreams": [{"logStreamName": "test-job-123/algo-1-1234567890"}]
        }
        monitor = CloudWatchLogMonitor(
            job_id=self.job_id,
            platform=Platform.SMTJ,
            started_time=self.started_time,
            cloudwatch_logs_client=self.mock_client,
        )
        logs = [{"message": "global_step: 1 | reduced_train_loss: 0.5"}]
        with self.assertRaises(NotImplementedError):
            monitor.strategy.get_metrics(TrainingMethod.SFT_FULL, logs, ["bogus_metric"])

    @patch("amzn_nova_forge.monitor.log_monitor.pyplot")
    def test_plot_metrics_logs_already_cached(self, mock_pyplot):
        """plot_metrics uses cached logs and does not call get_logs again."""
        self.mock_client.describe_log_streams.return_value = {
            "logStreams": [{"logStreamName": "test-job-123/algo-1-1234567890"}]
        }
        monitor = CloudWatchLogMonitor(
            job_id=self.job_id,
            platform=Platform.SMTJ,
            started_time=self.started_time,
            cloudwatch_logs_client=self.mock_client,
        )
        monitor.logs = [
            {"message": "global_step: 1 | reduced_train_loss: 0.5"},
            {"message": "global_step: 2 | reduced_train_loss: 0.4"},
        ]
        monitor.job_status_manager.get_job_status.return_value = ("Completed", None)

        with patch.object(monitor, "get_logs") as mock_get_logs:
            monitor.plot_metrics(TrainingMethod.SFT_FULL, ["training_loss"])
            mock_get_logs.assert_not_called()

        mock_pyplot.show.assert_called_once()

    @patch("amzn_nova_forge.monitor.log_monitor.pyplot")
    def test_plot_metrics_logs_not_cached_calls_get_logs(self, mock_pyplot):
        """plot_metrics calls get_logs when logs are not yet cached."""
        self.mock_client.describe_log_streams.return_value = {
            "logStreams": [{"logStreamName": "test-job-123/algo-1-1234567890"}]
        }
        monitor = CloudWatchLogMonitor(
            job_id=self.job_id,
            platform=Platform.SMTJ,
            started_time=self.started_time,
            cloudwatch_logs_client=self.mock_client,
        )
        self.assertIsNone(monitor.logs)
        monitor.job_status_manager.get_job_status.return_value = ("Completed", None)

        def fake_get_logs(**kwargs):
            monitor.logs = [
                {"message": "global_step: 1 | reduced_train_loss: 0.5"},
            ]
            return monitor.logs

        with patch.object(monitor, "get_logs", side_effect=fake_get_logs) as mock_get_logs:
            monitor.plot_metrics(TrainingMethod.SFT_FULL, ["training_loss"])
            mock_get_logs.assert_called_once()

        mock_pyplot.show.assert_called_once()

    def test_plot_metrics_starting_step_gt_ending_step_raises(self):
        self.mock_client.describe_log_streams.return_value = {
            "logStreams": [{"logStreamName": "test-job-123/algo-1-1234567890"}]
        }
        monitor = CloudWatchLogMonitor(
            job_id=self.job_id,
            platform=Platform.SMTJ,
            started_time=self.started_time,
            cloudwatch_logs_client=self.mock_client,
        )
        with self.assertRaises(ValueError):
            monitor.plot_metrics(
                TrainingMethod.SFT_FULL,
                ["training_loss"],
                starting_step=10,
                ending_step=5,
            )

    def test_plot_metrics_empty_logs_raises(self):
        self.mock_client.describe_log_streams.return_value = {
            "logStreams": [{"logStreamName": "test-job-123/algo-1-1234567890"}]
        }
        monitor = CloudWatchLogMonitor(
            job_id=self.job_id,
            platform=Platform.SMTJ,
            started_time=self.started_time,
            cloudwatch_logs_client=self.mock_client,
        )
        monitor.job_status_manager.get_job_status.return_value = ("Completed", None)

        with patch.object(monitor, "get_logs", return_value=[]):
            with self.assertRaises(ValueError):
                monitor.plot_metrics(TrainingMethod.SFT_FULL, ["training_loss"])

    @patch("amzn_nova_forge.monitor.log_monitor.pyplot")
    def test_plot_metrics_job_in_progress_refreshes_logs(self, mock_pyplot):
        """When job is in-progress, plot_metrics calls get_logs even if logs are cached."""
        from amzn_nova_forge.core.result.job_result import JobStatus

        self.mock_client.describe_log_streams.return_value = {
            "logStreams": [{"logStreamName": "test-job-123/algo-1-1234567890"}]
        }
        monitor = CloudWatchLogMonitor(
            job_id=self.job_id,
            platform=Platform.SMTJ,
            started_time=self.started_time,
            cloudwatch_logs_client=self.mock_client,
        )
        monitor.logs = [
            {"message": "global_step: 1 | reduced_train_loss: 0.9"},
        ]
        monitor.job_status_manager.get_job_status.return_value = (
            JobStatus.IN_PROGRESS,
            None,
        )

        def fake_get_logs(**kwargs):
            monitor.logs = [
                {"message": "global_step: 1 | reduced_train_loss: 0.9"},
                {"message": "global_step: 2 | reduced_train_loss: 0.7"},
            ]
            return monitor.logs

        with patch.object(monitor, "get_logs", side_effect=fake_get_logs) as mock_get_logs:
            monitor.plot_metrics(TrainingMethod.SFT_FULL, ["training_loss"])
            mock_get_logs.assert_called_once()

        mock_pyplot.show.assert_called_once()

    def test_plot_metrics_bedrock_get_metrics_raises_not_implemented(self):
        """BedrockStrategy.get_metrics raises NotImplementedError through plot_metrics."""
        with patch("boto3.client"):
            monitor = CloudWatchLogMonitor(
                job_id=self.job_id,
                platform=Platform.BEDROCK,
                cloudwatch_logs_client=self.mock_client,
            )
        monitor.logs = [{"message": "some log"}]
        monitor.job_status_manager.get_job_status.return_value = ("Completed", None)

        with self.assertRaises(NotImplementedError):
            monitor.plot_metrics(TrainingMethod.SFT_FULL)


class TestMTRLLogMonitor(unittest.TestCase):
    def test_from_job_id_no_category(self):
        from amzn_nova_forge.monitor.log_monitor import MTRLLogMonitor

        monitor = MTRLLogMonitor.from_job_id(job_id="test-job", region="us-east-1")
        self.assertEqual(monitor.job_id, "test-job")
        self.assertIsNone(monitor._job_category)

    def test_from_job_id_with_category(self):
        from amzn_nova_forge.monitor.log_monitor import MTRLLogMonitor

        monitor = MTRLLogMonitor.from_job_id(
            job_id="test-job", region="us-east-1", job_category="AgentRFTEvaluation"
        )
        self.assertEqual(monitor._job_category, "AgentRFTEvaluation")

    @patch("boto3.client")
    def test_detect_job_category_finds_eval(self, mock_boto3):
        from amzn_nova_forge.monitor.log_monitor import MTRLLogMonitor

        mock_logs_client = Mock()
        mock_logs_client.describe_log_streams.side_effect = [
            {"logStreams": []},
            {"logStreams": [{"logStreamName": "test-eval-job/"}]},
        ]
        mock_boto3.return_value = mock_logs_client

        monitor = MTRLLogMonitor(job_id="test-eval-job", region="us-east-1")
        category = monitor._detect_job_category()

        self.assertEqual(category, "AgentRFTEvaluation")

    @patch("boto3.client")
    def test_detect_job_category_finds_training(self, mock_boto3):
        from amzn_nova_forge.monitor.log_monitor import MTRLLogMonitor

        mock_logs_client = Mock()
        mock_logs_client.describe_log_streams.side_effect = [
            {"logStreams": [{"logStreamName": "test-train-job/"}]},
        ]
        mock_boto3.return_value = mock_logs_client

        monitor = MTRLLogMonitor(job_id="test-train-job", region="us-east-1")
        category = monitor._detect_job_category()

        self.assertEqual(category, "AgentRFT")

    @patch("boto3.client")
    def test_detect_job_category_skips_if_already_set(self, mock_boto3):
        from amzn_nova_forge.monitor.log_monitor import MTRLLogMonitor

        monitor = MTRLLogMonitor(
            job_id="test-job", region="us-east-1", job_category="AgentRFTEvaluation"
        )
        category = monitor._detect_job_category()

        self.assertEqual(category, "AgentRFTEvaluation")
        mock_boto3.assert_not_called()

    @patch("builtins.print")
    @patch("boto3.client")
    def test_show_eval_logs(self, mock_boto3, mock_print):
        from amzn_nova_forge.monitor.log_monitor import MTRLLogMonitor

        mock_logs_client = Mock()
        mock_logs_client.describe_log_streams.return_value = {
            "logStreams": [{"logStreamName": "eval-job-123/"}]
        }
        mock_logs_client.get_log_events.return_value = {
            "events": [
                {"message": "Progress: 5/50\n"},
                {"message": "Progress: 10/50\n"},
            ]
        }
        mock_boto3.return_value = mock_logs_client

        monitor = MTRLLogMonitor(
            job_id="eval-job-123", region="us-east-1", job_category="AgentRFTEvaluation"
        )
        monitor.show_logs(limit=5)

        mock_print.assert_any_call("Progress: 5/50")
        mock_print.assert_any_call("Progress: 10/50")

    @patch("builtins.print")
    @patch("boto3.client")
    def test_show_eval_logs_no_streams(self, mock_boto3, mock_print):
        from amzn_nova_forge.monitor.log_monitor import MTRLLogMonitor

        mock_logs_client = Mock()
        mock_logs_client.describe_log_streams.return_value = {"logStreams": []}
        mock_boto3.return_value = mock_logs_client

        monitor = MTRLLogMonitor(
            job_id="no-such-job", region="us-east-1", job_category="AgentRFTEvaluation"
        )
        monitor.show_logs()

        mock_print.assert_called_once_with("No log stream found for job 'no-such-job'")

    @patch("amzn_nova_forge.monitor.log_monitor.MTRLLogMonitor.from_job_id")
    def test_forge_evaluator_get_logs_routes_to_mtrl_monitor(self, mock_from_job_id):
        """ForgeEvaluator.get_logs() routes MTRL eval results to MTRLLogMonitor."""
        from datetime import datetime, timezone
        from unittest.mock import patch as mock_patch

        from amzn_nova_forge.core.enums import EvaluationTask

        mock_monitor = Mock()
        mock_from_job_id.return_value = mock_monitor

        mock_result = Mock()
        mock_result.job_id = "arn:aws:sagemaker:us-east-1:123456789012:pipeline/Eval/execution/abc"
        mock_result.eval_task = EvaluationTask.RFT_MULTITURN_EVAL
        mock_result.started_time = datetime(2026, 6, 3, tzinfo=timezone.utc)

        from amzn_nova_forge.evaluator.forge_evaluator import ForgeEvaluator

        with mock_patch.object(ForgeEvaluator, "__init__", return_value=None):
            evaluator = ForgeEvaluator.__new__(ForgeEvaluator)
            evaluator.region = "us-east-1"
            evaluator.get_logs(job_result=mock_result, limit=10)

        mock_from_job_id.assert_called_once_with(
            job_id=mock_result.job_id,
            region="us-east-1",
            job_category="AgentRFTEvaluation",
        )
        mock_monitor.show_logs.assert_called_once_with(limit=10)


class TestRegionPropagation(unittest.TestCase):
    @patch("boto3.client")
    def test_cloudwatch_log_monitor_passes_region_to_logs_client(self, mock_boto_client):
        mock_client = Mock()
        mock_client.describe_log_streams.return_value = {"logStreams": []}
        mock_boto_client.return_value = mock_client
        with patch.object(CloudWatchLogMonitor, "_create_job_status_manager", return_value=Mock()):
            CloudWatchLogMonitor(
                job_id="test",
                platform=Platform.SMTJ,
                region="eu-west-1",
                cloudwatch_logs_client=None,
            )
        mock_boto_client.assert_any_call("logs", region_name="eu-west-1")

    @patch("boto3.client")
    def test_smhp_strategy_passes_region_to_sagemaker_client(self, mock_boto_client):
        mock_boto_client.return_value = Mock()
        SMHPStrategy(
            cluster_name="test", namespace="default", region="eu-west-1", sagemaker_client=None
        )
        mock_boto_client.assert_called_once_with("sagemaker", region_name="eu-west-1")

    @patch("boto3.client")
    def test_bedrock_strategy_passes_region_to_bedrock_client(self, mock_boto_client):
        mock_boto_client.return_value = Mock()
        BedrockStrategy(region="eu-west-1", bedrock_client=None)
        mock_boto_client.assert_called_once_with("bedrock", region_name="eu-west-1")


if __name__ == "__main__":
    unittest.main()
