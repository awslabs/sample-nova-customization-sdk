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
"""Tests for SFT training metrics CSV generation.

Covers epoch extraction, CSV orchestration, and ForgeTrainer integration.
"""

import logging
from datetime import datetime, timezone
from unittest.mock import MagicMock, PropertyMock, create_autospec, patch

import pandas
import pytest

from amzn_nova_forge.core.enums import Model, Platform, TrainingMethod
from amzn_nova_forge.core.result.job_result import JobStatus
from amzn_nova_forge.manager.runtime_manager import SMHPRuntimeManager
from amzn_nova_forge.trainer.forge_trainer import ForgeTrainer
from amzn_nova_forge.util.metric_util import (
    _assign_epochs_to_steps,
    _extract_epoch_boundaries,
    get_metrics,
)


def _epoch_event(epoch_idx: int, ts: int) -> dict:
    return {
        "timestamp": ts,
        "message": f'[job] [INFO] {{"EpochIdx": {epoch_idx}}}',
    }


def _step_event(step: int, loss: float, ts: int) -> dict:
    return {
        "timestamp": ts,
        "message": f"[job] [INFO] global_step: {step} | reduced_train_loss: {loss}",
    }


SAMPLE_LOG_EVENTS = [
    _epoch_event(0, 1000),
    _step_event(1, 3.711, 2000),
    _step_event(2, 3.589, 3000),
    _epoch_event(1, 4000),
    _step_event(3, 3.456, 5000),
]


def _make_trainer(method=TrainingMethod.SFT_LORA, platform=Platform.SMHP):
    infra = create_autospec(SMHPRuntimeManager)
    infra.instance_type = "ml.p5.48xlarge"
    infra.instance_count = 2
    infra.kms_key_id = None
    infra.platform = platform
    infra.cluster_name = "my-cluster"
    infra.namespace = "kubeflow"
    infra.rft_lambda_arn = None
    with (
        patch(
            "amzn_nova_forge.trainer.forge_trainer.set_output_s3_path",
            return_value="s3://bucket/output",
        ),
        patch("boto3.session.Session") as mock_session,
    ):
        type(mock_session.return_value).region_name = PropertyMock(return_value="us-east-1")
        return ForgeTrainer(
            model=Model.NOVA_MICRO,
            method=method,
            infra=infra,
            training_data_s3_path="s3://bucket/data",
        )


class TestEpochExtraction:
    """Epoch boundary extraction and step assignment."""

    def test_extract_boundaries_and_assign_epochs(self):
        """Epoch boundaries are extracted and steps are assigned correctly."""
        boundaries = _extract_epoch_boundaries(SAMPLE_LOG_EVENTS)
        assert boundaries == [(0, 1000), (1, 4000)]

        metrics_df = get_metrics(
            platform=Platform.SMHP,
            training_method=TrainingMethod.SFT_LORA,
            logs=SAMPLE_LOG_EVENTS,
        )
        result = _assign_epochs_to_steps(metrics_df, boundaries, SAMPLE_LOG_EVENTS)
        assert list(result["epoch_number"]) == [0, 0, 1]

    def test_steps_before_any_epoch_default_to_zero(self):
        """Steps with no preceding epoch boundary get epoch 0."""
        events = [
            _step_event(1, 3.9, 1000),
            _step_event(2, 3.8, 2000),
            _epoch_event(1, 3000),
            _step_event(3, 3.7, 4000),
        ]
        metrics_df = get_metrics(
            platform=Platform.SMHP,
            training_method=TrainingMethod.SFT_LORA,
            logs=events,
        )
        boundaries = _extract_epoch_boundaries(events)
        result = _assign_epochs_to_steps(metrics_df, boundaries, events)
        assert list(result["epoch_number"]) == [0, 0, 1]

    def test_empty_logs_return_empty(self):
        """Empty log list produces empty boundaries and handles empty DataFrame."""
        assert _extract_epoch_boundaries([]) == []
        empty_df = pandas.DataFrame(columns=["global_step", "training_loss"])
        result = _assign_epochs_to_steps(empty_df, [], [])
        assert "epoch_number" in result.columns
        assert len(result) == 0


class TestBuildAndUploadTrainingMetricsCsv:
    """The _build_and_upload_training_metrics_csv() utility function."""

    def test_successful_generation(self):
        """Returns S3 URI and uploads CSV with correct columns."""
        from amzn_nova_forge.util.metric_util import _build_and_upload_training_metrics_csv

        mock_s3 = MagicMock()

        result = _build_and_upload_training_metrics_csv(
            job_id="job-123",
            log_events=SAMPLE_LOG_EVENTS,
            output_s3_path="s3://bucket/prefix",
            training_method=TrainingMethod.SFT_LORA,
            s3_client=mock_s3,
        )

        assert result == "s3://bucket/prefix/job-123/step_wise_training_metrics.csv"
        mock_s3.upload_file.assert_called_once()
        _, bucket, key = mock_s3.upload_file.call_args[0]
        assert bucket == "bucket"
        assert key == "prefix/job-123/step_wise_training_metrics.csv"

    def test_no_logs_returns_none(self):
        """Returns None when no logs are provided."""
        from amzn_nova_forge.util.metric_util import _build_and_upload_training_metrics_csv

        result = _build_and_upload_training_metrics_csv(
            job_id="job-123",
            log_events=[],
            output_s3_path="s3://bucket/prefix",
            training_method=TrainingMethod.SFT_LORA,
        )
        assert result is None

    def test_missing_output_path_raises(self):
        """Raises ValueError when output_s3_path is empty."""
        from amzn_nova_forge.util.metric_util import _build_and_upload_training_metrics_csv

        with pytest.raises(ValueError, match="output_s3_path"):
            _build_and_upload_training_metrics_csv(
                job_id="job-123",
                log_events=SAMPLE_LOG_EVENTS,
                output_s3_path="",
                training_method=TrainingMethod.SFT_LORA,
            )


class TestForgeTrainerIntegration:
    """ForgeTrainer.generate_training_metrics_csv() validation."""

    def test_non_smhp_platform_raises_error(self):
        """Non-SMHP platform raises ValueError."""
        trainer = _make_trainer()
        trainer._platform = Platform.SMTJ
        with pytest.raises(ValueError, match="SMHP"):
            trainer.generate_training_metrics_csv(job_id="j", output_s3_path="s3://b/o")

    def test_non_sft_method_raises_error(self):
        """Non-SFT method raises ValueError."""
        trainer = _make_trainer(method=TrainingMethod.CPT)
        with pytest.raises(ValueError, match="SFT"):
            trainer.generate_training_metrics_csv(job_id="j", output_s3_path="s3://b/o")

    @patch("amzn_nova_forge.trainer.forge_trainer.CloudWatchLogMonitor")
    @patch("amzn_nova_forge.trainer.forge_trainer._build_and_upload_training_metrics_csv")
    def test_delegates_to_metric_util_with_correct_params(self, mock_gen, mock_monitor_cls):
        """Fetches logs and delegates to metric_util function."""
        mock_monitor_cls.return_value.get_logs.return_value = SAMPLE_LOG_EVENTS
        mock_gen.return_value = "s3://bucket/output/job-abc/step_wise_training_metrics.csv"
        trainer = _make_trainer()

        job_result = MagicMock()
        job_result.job_id = "job-abc"
        job_result.started_time = datetime(2025, 6, 1, tzinfo=timezone.utc)
        job_result.cluster_name = "hp-cluster"
        job_result.namespace = "ns"
        job_result.model_artifacts.output_s3_path = "s3://bucket/output"
        job_result.get_job_status.return_value = (JobStatus.COMPLETED, "Succeeded")

        result = trainer.generate_training_metrics_csv(job_result=job_result)

        assert result == "s3://bucket/output/job-abc/step_wise_training_metrics.csv"
        mock_gen.assert_called_once_with(
            job_id="job-abc",
            log_events=SAMPLE_LOG_EVENTS,
            output_s3_path="s3://bucket/output",
            training_method=TrainingMethod.SFT_LORA,
            region="us-east-1",
        )
