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
import logging
import os
import re
import tempfile
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional

import boto3
import pandas

from amzn_nova_forge.core.enums import Platform, TrainingMethod
from amzn_nova_forge.util.s3_utils import parse_s3_uri

logger = logging.getLogger(__name__)

DEFAULT_LOOKBACK_DAYS = 7
EPOCH_IDX_PATTERN = re.compile(r'"EpochIdx"\s*:\s*(\d+)')
GLOBAL_STEP_REGEX = r"global_step[=:]\s*([\d.]+)"
TRAINING_LOSS_REGEX = r"reduced_train_loss[=:]\s*(-?[\d.]+(?:[eE][+-]?\d+)?)"
SMHP_RFT_REWARD_SCORE_REGEX = r"train_rm_score:\s*(-?[\d.]+(?:[eE][+-]?\d+)?)"
SMTJ_RFT_REWARD_SCORE_REGEX = r"critic/rewards/mean[=:]\s*(-?[\d.]+(?:[eE][+-]?\d+)?)"
CPT = "cpt"
SFT = "sft"
RFT = "rft"
AVAILABLE_METRICS = {
    Platform.SMTJ: {
        CPT: {"training_loss": TRAINING_LOSS_REGEX},
        SFT: {"training_loss": TRAINING_LOSS_REGEX},
        RFT: {"reward_score": SMTJ_RFT_REWARD_SCORE_REGEX},
    },
    Platform.SMHP: {
        CPT: {"training_loss": TRAINING_LOSS_REGEX},
        SFT: {"training_loss": TRAINING_LOSS_REGEX},
        RFT: {"reward_score": SMHP_RFT_REWARD_SCORE_REGEX},
    },
}


def get_metrics(
    platform: Platform,
    training_method: TrainingMethod,
    logs: Optional[List[Dict]] = None,
    metrics: Optional[List] = None,
) -> pandas.DataFrame:
    patterns = []
    training_category = training_method.value[:3]
    if not metrics:
        metrics = list(AVAILABLE_METRICS[platform][training_category].keys())
    for metric in metrics:
        if metric not in AVAILABLE_METRICS[platform][training_category]:
            raise NotImplementedError(
                f"Unsupported metric for {training_category} on {platform}: {metric}"
            )
        patterns.append(AVAILABLE_METRICS[platform][training_category][metric])

    all_metrics: List[List[float]] = []
    log_lines = [line for log in (logs or []) for line in log["message"].splitlines()]

    for line in log_lines:
        global_step_match = re.search(GLOBAL_STEP_REGEX, line)
        if not global_step_match:
            continue
        try:
            step_metrics: List[float] = [int(float(global_step_match.group(1)))]
            for pattern in patterns:
                match = re.search(pattern, line)
                if match is None:
                    raise ValueError(f"Pattern {pattern} not found in line")
                step_metrics.append(float(match.group(1)))
            all_metrics.append(step_metrics)
        except Exception:
            pass

    return pandas.DataFrame(all_metrics, columns=["global_step"] + metrics)


def _extract_epoch_boundaries(log_events: list[dict]) -> list[tuple[int, int]]:
    """
    Extract epoch start timestamps from CloudWatch log events.
    Returns a list of (epoch_index, timestamp_ms) tuples sorted by timestamp.
    """
    boundaries: list[tuple[int, int]] = []
    for event in log_events:
        message = event.get("message", "")
        match = EPOCH_IDX_PATTERN.search(message)
        if match:
            epoch_index = int(match.group(1))
            timestamp_ms = event.get("timestamp", 0)
            boundaries.append((epoch_index, timestamp_ms))
    boundaries.sort(key=lambda x: x[1])
    return boundaries


def _assign_epochs_to_steps(
    metrics_df: pandas.DataFrame,
    epoch_boundaries: list[tuple[int, int]],
    log_events: list[dict],
) -> pandas.DataFrame:
    """Adds epoch_number column to the metrics DataFrame.

    Compares the timestamp of the step to the most recent epoch boundary
    that precedes the step. Defaults to epoch 0 if an epoch doesn't precede
    a step value.
    """
    if metrics_df.empty:
        metrics_df = metrics_df.copy()
        metrics_df["epoch_number"] = pandas.Series(dtype=int)
        return metrics_df

    # Build a mapping from global_step to timestamp by scanning log events
    step_timestamps: dict[int, int] = {}
    for event in log_events:
        message = event.get("message", "")
        for line in message.splitlines():
            step_match = re.search(GLOBAL_STEP_REGEX, line)
            if step_match:
                step_value = int(float(step_match.group(1)))
                if step_value not in step_timestamps:
                    step_timestamps[step_value] = event.get("timestamp", 0)

    # Assign epoch numbers based on most recent preceding epoch boundary
    epoch_numbers: list[int] = []
    for _, row in metrics_df.iterrows():
        step = int(row["global_step"])
        step_ts = step_timestamps.get(step, 0)

        # Find the most recent epoch boundary at or before this step's timestamp
        assigned_epoch = 0
        for epoch_index, boundary_ts in epoch_boundaries:
            if boundary_ts <= step_ts:
                assigned_epoch = epoch_index
            else:
                break

        epoch_numbers.append(assigned_epoch)

    result_df = metrics_df.copy()
    result_df["epoch_number"] = epoch_numbers
    return result_df


def _parse_user_time(started_time) -> datetime:
    """Parse start or end time into a datetime.

    Accepts: datetime object (returned as-is), ISO date string (e.g. "2025-05-26")
    If "None" is provided, defaults to 7 days.
    """
    if started_time is None:
        return datetime.now(tz=timezone.utc) - timedelta(days=DEFAULT_LOOKBACK_DAYS)

    if isinstance(started_time, datetime):
        return started_time

    if isinstance(started_time, str):
        try:
            parsed = datetime.fromisoformat(started_time.replace("Z", "+00:00"))
            return parsed
        except ValueError:
            raise ValueError(
                f"Cannot parse started_time '{started_time}'. "
                "Use ISO date format, e.g., '2025-05-26'."
            )

    raise TypeError(
        f"started_time must be a datetime, ISO date string, or None. "
        f"Got {type(started_time).__name__}."
    )


def _build_and_upload_training_metrics_csv(
    job_id: str,
    log_events: List[Dict],
    output_s3_path: str,
    training_method: TrainingMethod,
    region: Optional[str] = None,
    s3_client=None,
) -> Optional[str]:
    """Generate step_wise_training_metrics.csv from SMHP SFT log events.

    Parses the provided log events to extract step-level training metrics,
    writes them to a CSV, and uploads to S3.

    Args:
        job_id: The SMHP training job ID.
        log_events: CloudWatch log events (list of dicts with 'message' and 'timestamp').
        output_s3_path: S3 URI for output (e.g., s3://bucket/prefix).
        training_method: The training method (SFT_LORA or SFT_FULL).
        region: AWS region (used if s3_client is not provided).
        s3_client: Optional boto3 S3 client.

    Returns:
        S3 URI of the uploaded CSV, or None if no metrics could be extracted.

    Raises:
        ValueError: If output_s3_path is not provided.
    """
    if not output_s3_path:
        raise ValueError("output_s3_path is required but was not provided.")

    if not log_events:
        logger.warning(
            "No CloudWatch logs found for job %s. Cannot generate training metrics CSV.",
            job_id,
        )
        return None

    metrics_df = get_metrics(
        platform=Platform.SMHP,
        training_method=training_method,
        logs=log_events,
    )
    if metrics_df.empty:
        logger.warning(
            "No training metrics could be extracted from logs for job %s. "
            "Cannot generate training metrics CSV.",
            job_id,
        )
        return None

    # Extract epoch boundaries and assign epochs to steps
    epoch_boundaries = _extract_epoch_boundaries(log_events)
    metrics_df = _assign_epochs_to_steps(metrics_df, epoch_boundaries, log_events)

    # Construct final DataFrame with expected column names
    final_df = pandas.DataFrame(
        {
            "step_number": metrics_df["global_step"].astype(int),
            "epoch_number": metrics_df["epoch_number"].astype(int),
            "training_loss": metrics_df["training_loss"],
        }
    )

    # Write CSV to a temp file
    csv_filename = "step_wise_training_metrics.csv"
    tmp_file = None
    try:
        tmp_file = tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False)
        final_df.to_csv(tmp_file.name, index=False)
        tmp_file.close()

        # Parse S3 path for upload
        bucket, base_key = parse_s3_uri(output_s3_path)
        base_key = base_key.strip("/")

        # Construct the full S3 key
        s3_key = f"{base_key}/{job_id}/{csv_filename}" if base_key else f"{job_id}/{csv_filename}"

        # Upload to S3
        if s3_client is None:
            s3_client = boto3.client("s3", region_name=region)

        s3_client.upload_file(tmp_file.name, bucket, s3_key)

        # Construct and return the S3 URI
        s3_uri = f"s3://{bucket}/{s3_key}"
        logger.info("Training metrics CSV uploaded to %s", s3_uri)
        return s3_uri

    finally:
        # Clean up temp file
        if tmp_file and os.path.exists(tmp_file.name):
            os.unlink(tmp_file.name)
