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
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any, Dict, List, Optional, cast

import boto3
import pandas
from matplotlib import pyplot

from amzn_nova_forge.core.constants import (
    MTRL_EVAL_LOG_GROUP,
    MTRL_PIPELINE_EXECUTION_RE,
    MTRL_TRAIN_LOG_GROUP,
)
from amzn_nova_forge.core.enums import Platform, TrainingMethod
from amzn_nova_forge.core.result.job_result import (
    BaseJobResult,
    JobStatus,
    JobStatusManager,
    SMHPStatusManager,
    SMTJStatusManager,
)
from amzn_nova_forge.telemetry.constants import UNKNOWN, Feature
from amzn_nova_forge.telemetry.telemetry_logging import _telemetry_emitter
from amzn_nova_forge.util.bedrock import (
    get_bedrock_job_details,
    log_bedrock_job_status,
)
from amzn_nova_forge.util.logging import logger
from amzn_nova_forge.util.metric_util import get_metrics

DEFAULT_SMHP_NAMESPACE = "kubeflow"

_SMTJ_PLATFORMS = {Platform.SMTJ, Platform.SMTJServerless}


class PlatformStrategy(ABC):
    @abstractmethod
    def get_log_group_name(self, job_id: str) -> str:
        pass

    @abstractmethod
    def find_log_stream(
        self, job_id: str, cloudwatch_logs_client, log_group_name: str
    ) -> Optional[str]:
        pass

    @abstractmethod
    def get_logs(
        self,
        job_id: str,
        cloudwatch_logs_client,
        log_group_name: str,
        log_stream_name: str,
        limit: Optional[int],
        start_from_head: bool,
        start_time: Optional[int] = None,
        end_time: Optional[int] = None,
    ) -> List[Dict]:
        pass

    @abstractmethod
    def get_metrics(
        self,
        training_method: TrainingMethod,
        logs: Optional[List[Dict]] = None,
        metrics: Optional[List] = None,
    ) -> pandas.DataFrame:
        pass


class SMTJStrategy(PlatformStrategy):
    def get_log_group_name(self, job_id: str) -> str:
        return "/aws/sagemaker/TrainingJobs"

    def find_log_stream(
        self, job_id: str, cloudwatch_logs_client, log_group_name: str
    ) -> Optional[str]:
        response = cloudwatch_logs_client.describe_log_streams(
            logGroupName=log_group_name, logStreamNamePrefix=job_id
        )
        return response["logStreams"][0]["logStreamName"] if response["logStreams"] else None

    def get_logs(
        self,
        job_id: str,
        cloudwatch_logs_client,
        log_group_name: str,
        log_stream_name: str,
        limit: Optional[int],
        start_from_head: bool,
        start_time: Optional[int] = None,
        end_time: Optional[int] = None,
    ) -> List[Dict]:
        if not log_stream_name:
            return []

        all_events: List[Dict] = []
        next_token = None

        end_time = end_time or int(datetime.now().timestamp() * 1000)
        while True:
            params: Dict[str, Any] = {
                "endTime": end_time,
                "logGroupName": log_group_name,
                "logStreamName": log_stream_name,
                "startFromHead": start_from_head,
            }

            if limit:
                params["limit"] = min(limit - len(all_events), 10000)

            if start_time:
                params["startTime"] = start_time

            if next_token:
                params["nextToken"] = next_token

            response = cloudwatch_logs_client.get_log_events(**params)
            events = response["events"]

            all_events.extend(events)

            if limit and len(all_events) >= limit:
                all_events = all_events[:limit]
                break

            current_token = next_token
            next_token = response.get(
                "nextForwardToken" if start_from_head else "nextBackwardToken"
            )
            if next_token == current_token:
                break

        return all_events

    def get_metrics(
        self,
        training_method: TrainingMethod,
        logs: Optional[List[Dict]] = None,
        metrics: Optional[List] = None,
    ) -> pandas.DataFrame:
        return get_metrics(
            platform=Platform.SMTJ,
            training_method=training_method,
            logs=logs,
            metrics=metrics,
        )


class SMHPStrategy(PlatformStrategy):
    def __init__(
        self, cluster_name: str, namespace: str, sagemaker_client=None, region: Optional[str] = None
    ):
        self.cluster_name = cluster_name
        self.namespace = namespace
        self.sagemaker_client = sagemaker_client or boto3.client("sagemaker", region_name=region)
        self._cluster_id: Optional[str] = None

    def get_log_group_name(self, job_id: str) -> str:
        if not self._cluster_id:
            response = self.sagemaker_client.describe_cluster(ClusterName=self.cluster_name)
            cluster_arn = response["ClusterArn"]
            self._cluster_id = cluster_arn.split("/")[-1]
        return f"/aws/sagemaker/Clusters/{self.cluster_name}/{self._cluster_id}"

    def find_log_stream(
        self, job_id: str, cloudwatch_logs_client, log_group_name: str
    ) -> Optional[str]:
        # TODO: add logic to find log stream if we can find nodeId from job_id
        # Currently the SMHP log stream is separated by nodeID rather than job id
        return None

    def get_logs(
        self,
        job_id: str,
        cloudwatch_logs_client,
        log_group_name: str,
        log_stream_name: str,
        limit: Optional[int],
        start_from_head: bool,
        start_time: Optional[int] = None,
        end_time: Optional[int] = None,
    ) -> List[Dict]:
        all_events: List[Dict] = []
        next_token = None

        end_time = end_time or int(datetime.now().timestamp() * 1000)
        while True:
            # TODO: Add log_stream_name into filter params if it's not None
            params: Dict[str, Any] = {
                "endTime": end_time,
                "logGroupName": log_group_name,
                "logStreamNamePrefix": "SagemakerHyperPodTrainingJob",
                "filterPattern": f"%{job_id}%",
            }

            if limit:
                params["limit"] = min(limit - len(all_events), 10000)

            if start_time:
                params["startTime"] = start_time

            if next_token:
                params["nextToken"] = next_token

            # TODO: change to use get_log_events once SMHP supports separating log stream by job id
            response = cloudwatch_logs_client.filter_log_events(**params)
            events = response["events"]

            all_events.extend(events)

            if limit and len(all_events) >= limit:
                all_events = all_events[:limit]
                break

            next_token = response.get("nextToken")
            if not next_token:
                break

        return all_events

    def get_metrics(
        self,
        training_method: TrainingMethod,
        logs: Optional[List[Dict]] = None,
        metrics: Optional[List] = None,
    ) -> pandas.DataFrame:
        return get_metrics(
            platform=Platform.SMHP,
            training_method=training_method,
            logs=logs,
            metrics=metrics,
        )


class BedrockStrategy(PlatformStrategy):
    def __init__(self, bedrock_client=None, region: Optional[str] = None):
        self.bedrock_client = bedrock_client or boto3.client("bedrock", region_name=region)

    def get_log_group_name(self, job_id: str) -> str:
        # Bedrock customization jobs do not create CloudWatch logs
        # This method is kept for interface compatibility
        return "/aws/bedrock/modelcustomizationjobs"

    def find_log_stream(
        self, job_id: str, cloudwatch_logs_client, log_group_name: str
    ) -> Optional[str]:
        # Bedrock customization jobs do not create CloudWatch logs
        # Return None to indicate logs are not available
        return None

    def get_logs(
        self,
        job_id: str,
        cloudwatch_logs_client,
        log_group_name: str,
        log_stream_name: str,
        limit: Optional[int],
        start_from_head: bool,
        start_time: Optional[int] = None,
        end_time: Optional[int] = None,
    ) -> List[Dict]:
        """
        Bedrock customization jobs do not stream logs to CloudWatch.
        Instead, display job status and provide guidance on monitoring.

        Returns empty list to maintain interface compatibility.
        """

        logger.warning("CloudWatch logs are not available for Bedrock customization jobs.")

        # Get and display current job status using shared utility
        try:
            response = get_bedrock_job_details(self.bedrock_client, job_id)
            log_bedrock_job_status(response)
        except Exception as e:
            logger.error(f"Error retrieving job status: {e}")

        # Return empty list to maintain interface compatibility
        return []

    def get_metrics(
        self,
        training_method: TrainingMethod,
        logs: Optional[List[Dict]] = None,
        metrics: Optional[List] = None,
    ) -> pandas.DataFrame:
        raise NotImplementedError(f"Metrics not available for Bedrock jobs")


class CloudWatchLogMonitor:
    def __init__(
        self,
        job_id: str,
        platform: Platform,
        started_time: Optional[int] = None,
        cloudwatch_logs_client=None,
        region: Optional[str] = None,
        **kwargs,
    ):
        self.job_id = job_id
        self.platform = platform
        self.started_time = started_time
        self.cloudwatch_logs_client = cloudwatch_logs_client or boto3.client(
            "logs", region_name=region
        )
        self.strategy = self._create_strategy(platform, region=region, **kwargs)
        self.log_group_name = self._get_log_group_name()
        self.log_stream_name = self._find_log_stream()
        self.job_status_manager = self._create_job_status_manager()
        self.logs: Optional[List[Dict]] = None

    @staticmethod
    def _create_strategy(platform: Platform, **kwargs):
        region = kwargs.get("region")
        if platform in _SMTJ_PLATFORMS:
            return SMTJStrategy()
        elif platform == Platform.SMHP:
            cluster_name = kwargs.get("cluster_name")
            namespace = kwargs.get("namespace")
            sagemaker_client = kwargs.get("sagemaker_client")
            if not namespace:
                namespace = DEFAULT_SMHP_NAMESPACE
                logger.info(f"No namespace provided, using {namespace}` as default")
            if not cluster_name:
                raise ValueError("SMHP platform requires 'cluster_name' parameters")
            return SMHPStrategy(cluster_name, namespace, sagemaker_client, region=region)
        elif platform == Platform.BEDROCK:
            bedrock_client = kwargs.get("bedrock_client")
            return BedrockStrategy(bedrock_client, region=region)
        else:
            raise NotImplementedError(f"Unsupported platform: {platform}")

    @classmethod
    @_telemetry_emitter(
        Feature.MONITOR,
        "from_job_id",
        extra_info_fn=lambda cls, *args, **kwargs: {
            "platform": kwargs.get("platform", UNKNOWN),
        },
    )
    def from_job_id(
        cls,
        job_id: str,
        platform: Platform,
        started_time: Optional[datetime] = None,
        **kwargs,
    ):
        resolved_ms = None
        if started_time:
            resolved_ms = int(started_time.timestamp() * 1000)
        else:
            resolved_ms = cls._resolve_start_time_ms(job_id, platform, **kwargs)

        return cls(
            job_id=job_id,
            platform=platform,
            started_time=resolved_ms,
            **kwargs,
        )

    @staticmethod
    def _resolve_start_time_ms(job_id: str, platform: Platform, **kwargs) -> Optional[int]:
        """Try to resolve start time from the platform API. Returns epoch ms or None."""
        try:
            manager: JobStatusManager
            if platform in _SMTJ_PLATFORMS:
                sagemaker_client = kwargs.get("sagemaker_client")
                manager = SMTJStatusManager(sagemaker_client)
            elif platform == Platform.SMHP:
                cluster_name = kwargs.get("cluster_name")
                namespace = kwargs.get("namespace", DEFAULT_SMHP_NAMESPACE)
                if not cluster_name:
                    return None
                manager = SMHPStatusManager(cluster_name, namespace)
            else:
                return None

            dt = manager.resolve_start_time(job_id)
            resolved_ms = int(dt.timestamp() * 1000)
            logger.info(f"Resolved start time for job {job_id}: {dt}")
            return resolved_ms
        except Exception as e:
            logger.warning(
                f"Could not resolve start time for job {job_id}: {e}. "
                f"Log retrieval may be slow without a start time filter."
            )
            return None

    @classmethod
    @_telemetry_emitter(Feature.MONITOR, "from_job_result")
    def from_job_result(
        cls, job_result: BaseJobResult, cloudwatch_logs_client=None, region: Optional[str] = None
    ):
        region = region or getattr(job_result, "_region", None)
        if job_result.platform in _SMTJ_PLATFORMS:
            return cls(
                job_id=job_result.job_id,
                platform=job_result.platform,
                started_time=int(job_result.started_time.timestamp() * 1000),
                cloudwatch_logs_client=cloudwatch_logs_client,
                region=region,
            )
        elif job_result.platform == Platform.SMHP:
            job_status_manager = cast(SMHPStatusManager, job_result.status_manager)
            return cls(
                job_id=job_result.job_id,
                platform=job_result.platform,
                started_time=int(job_result.started_time.timestamp() * 1000),
                cloudwatch_logs_client=cloudwatch_logs_client,
                cluster_name=job_status_manager.cluster_name,
                namespace=job_status_manager.namespace,
                region=region,
            )
        elif job_result.platform == Platform.BEDROCK:
            # Bedrock doesn't use CloudWatch logs, but we still create the monitor
            # for interface compatibility. The BedrockStrategy will handle showing
            # job status instead of logs.
            return cls(
                job_id=job_result.job_id,
                platform=job_result.platform,
                started_time=int(job_result.started_time.timestamp() * 1000),
                cloudwatch_logs_client=cloudwatch_logs_client,
                region=region,
            )
        else:
            raise NotImplementedError(f"Unsupported platform: {job_result.platform}")

    def _get_log_group_name(self):
        return self.strategy.get_log_group_name(self.job_id)

    def _find_log_stream(self):
        return self.strategy.find_log_stream(
            self.job_id, self.cloudwatch_logs_client, self.log_group_name
        )

    def _create_job_status_manager(self):
        if self.platform in _SMTJ_PLATFORMS:
            return SMTJStatusManager()
        elif self.platform == Platform.SMHP:
            strategy = cast(SMHPStrategy, self.strategy)
            return SMHPStatusManager(strategy.cluster_name, strategy.namespace)

    def _get_in_range_dataframe(
        self,
        metrics_df: pandas.DataFrame,
        starting_step: Optional[int] = None,
        ending_step: Optional[int] = None,
    ):
        metrics_df = metrics_df.drop_duplicates(subset=["global_step"], keep="last")
        if starting_step and ending_step:
            metrics_df = metrics_df[
                (metrics_df["global_step"] >= starting_step)
                & (metrics_df["global_step"] <= ending_step)
            ]
        elif starting_step:
            metrics_df = metrics_df[metrics_df["global_step"] >= starting_step]
        elif ending_step:
            metrics_df = metrics_df[metrics_df["global_step"] <= ending_step]

        if metrics_df.empty:
            raise ValueError(
                f"No metrics found in the specified step range [{starting_step or 'start'}-{ending_step or 'end'}]"
            )
        return metrics_df

    @_telemetry_emitter(Feature.MONITOR, "get_logs")
    def get_logs(
        self,
        limit: Optional[int] = None,
        start_from_head: bool = False,
        end_time: Optional[int] = None,
    ) -> Optional[List[Dict]]:
        self.log_stream_name = self.log_stream_name or self._find_log_stream()
        # Cache the latest logs
        self.logs = self.strategy.get_logs(
            job_id=self.job_id,
            cloudwatch_logs_client=self.cloudwatch_logs_client,
            log_group_name=self.log_group_name,
            log_stream_name=self.log_stream_name,
            limit=limit,
            start_from_head=start_from_head,
            start_time=self.started_time,
            end_time=end_time,
        )

        return self.logs

    @_telemetry_emitter(Feature.MONITOR, "show_logs")
    def show_logs(
        self,
        limit: Optional[int] = None,
        start_from_head: bool = False,
        end_time: Optional[int] = None,
    ):
        events = self.get_logs(limit=limit, start_from_head=start_from_head, end_time=end_time)
        if events:
            for event in events:
                print(event["message"].strip())
        else:
            print(f"No logs found for job {self.job_id} yet")

    @_telemetry_emitter(Feature.MONITOR, "plot_metrics")
    def plot_metrics(
        self,
        training_method: TrainingMethod,
        metrics: Optional[List] = None,
        starting_step: Optional[int] = None,
        ending_step: Optional[int] = None,
    ):
        if starting_step and ending_step and starting_step > ending_step:
            raise ValueError("Starting iteration must be less than or equal to ending iteration")

        try:
            job_in_progress = (
                self.job_status_manager.get_job_status(self.job_id)[0] == JobStatus.IN_PROGRESS
            )
        except Exception:
            job_in_progress = False

        if (not self.logs) or job_in_progress:
            self.get_logs()
        if not self.logs:
            raise ValueError("No logs found for this job")

        metrics_df = self.strategy.get_metrics(training_method, self.logs, metrics)
        metrics_df = self._get_in_range_dataframe(metrics_df, starting_step, ending_step)
        metrics_df = metrics_df.sort_values("global_step").reset_index(drop=True)

        pyplot.figure(figsize=(8, 5))

        for col in metrics_df.columns.drop("global_step"):
            pyplot.plot(metrics_df["global_step"], metrics_df[col], label=col)
        pyplot.xlabel("global_step")
        pyplot.title("Training Metrics")
        pyplot.legend()
        pyplot.grid(True)
        pyplot.style.use("seaborn-v0_8-white")
        pyplot.show()


class MTRLLogMonitor:
    """Log monitor for MTRL jobs.

    Provides the same ``from_job_id`` / ``show_logs`` interface as
    ``CloudWatchLogMonitor`` so the user experience is consistent.
    Internally delegates to the ``AgentRFTJob.wait()`` for live
    progress and ``get_training_metrics()`` for completed jobs.
    Auto-detects whether the job is a training or evaluation job.
    """

    def __init__(
        self, job_id: str, region: Optional[str] = None, job_category: Optional[str] = None
    ):
        self.job_id = job_id
        self._region = region
        self._job_category = job_category
        self._rft_job = None

    @classmethod
    @_telemetry_emitter(Feature.MONITOR, "mtrl_from_job_id")
    def from_job_id(
        cls, job_id: str, region: Optional[str] = None, job_category: Optional[str] = None, **kwargs
    ) -> "MTRLLogMonitor":
        return cls(job_id=job_id, region=region, job_category=job_category)

    def _detect_job_category(self) -> str:
        """Auto-detect whether this is a training or evaluation job."""
        if self._job_category:
            return self._job_category

        # Pipeline execution ARNs are always evaluation jobs
        if MTRL_PIPELINE_EXECUTION_RE.match(self.job_id):
            self._job_category = "AgentRFTEvaluation"
            return self._job_category

        try:
            logs_client = boto3.client("logs", region_name=self._region)

            for category, log_group in [
                ("AgentRFT", MTRL_TRAIN_LOG_GROUP),
                ("AgentRFTEvaluation", MTRL_EVAL_LOG_GROUP),
            ]:
                try:
                    resp = logs_client.describe_log_streams(
                        logGroupName=log_group,
                        logStreamNamePrefix=self.job_id,
                        limit=1,
                    )
                    if resp.get("logStreams"):
                        self._job_category = category
                        return category
                except logs_client.exceptions.ResourceNotFoundException:
                    continue
        except Exception:
            pass

        self._job_category = "AgentRFT"
        return self._job_category

    def _get_rft_job(self):
        if self._rft_job is None:
            from sagemaker.train.agent_rft_job import AgentRFTJob

            session = boto3.Session(region_name=self._region) if self._region else None
            self._rft_job = AgentRFTJob.get(self.job_id, session=session)
        return self._rft_job

    @_telemetry_emitter(Feature.MONITOR, "mtrl_show_logs")
    def show_logs(
        self, poll: int = 30, timeout: int = 7200, limit: Optional[int] = None, **kwargs
    ) -> None:
        """Show MTRL job progress.

        If the job is still running, blocks and displays a live progress panel.
        If the job is completed, prints training metrics.
        For eval jobs, reads CloudWatch logs directly.
        """
        category = self._detect_job_category()

        if category == "AgentRFTEvaluation":
            self._show_eval_logs(limit=limit)
            return

        rft_job = self._get_rft_job()
        rft_job.refresh()
        if rft_job.job_status in ("Completed", "Failed", "Stopped"):
            print(f"Job '{self.job_id}' status: {rft_job.job_status}")
            if rft_job.job_status == "Completed":
                rft_job.get_training_metrics()
        else:
            rft_job.wait(poll=poll, timeout=timeout)

    def _resolve_eval_job_names(self) -> List[tuple]:
        """Resolve the underlying eval job name(s) from a pipeline execution ARN.

        Returns list of (step_name, job_name) tuples.
        """
        if not MTRL_PIPELINE_EXECUTION_RE.match(self.job_id):
            return [("", self.job_id)]

        region = self.job_id.split(":")[3]
        sm_client = boto3.client("sagemaker", region_name=region)
        steps = sm_client.list_pipeline_execution_steps(PipelineExecutionArn=self.job_id)
        jobs = []
        for step in steps.get("PipelineExecutionSteps", []):
            job_meta = step.get("Metadata", {}).get("Job", {})
            if job_meta:
                job_name = job_meta["Arn"].rsplit("/", 1)[-1]
                step_name = step.get("StepName", "")
                jobs.append((step_name, job_name))
        return jobs or [("", self.job_id)]

    def _show_eval_logs(self, limit: Optional[int] = None) -> None:
        """Read and display CloudWatch logs for an MTRL evaluation job."""
        jobs = self._resolve_eval_job_names()
        region = (
            self.job_id.split(":")[3]
            if MTRL_PIPELINE_EXECUTION_RE.match(self.job_id)
            else self._region
        )
        logs_client = boto3.client("logs", region_name=region)

        for step_name, job_name in jobs:
            if step_name:
                print(f"\n--- {step_name} ---")

            try:
                resp = logs_client.describe_log_streams(
                    logGroupName=MTRL_EVAL_LOG_GROUP,
                    logStreamNamePrefix=job_name,
                    limit=5,
                )
            except logs_client.exceptions.ResourceNotFoundException:
                print(f"No log group found: {MTRL_EVAL_LOG_GROUP}")
                return

            streams = resp.get("logStreams", [])
            if not streams:
                print(f"No log stream found for job '{job_name}'")
                continue

            stream_name = streams[0]["logStreamName"]
            params: Dict[str, Any] = {
                "logGroupName": MTRL_EVAL_LOG_GROUP,
                "logStreamName": stream_name,
                "startFromHead": False,
            }
            if limit:
                params["limit"] = limit

            events_resp = logs_client.get_log_events(**params)
            events = events_resp.get("events", [])

            if not events:
                print(f"No logs available yet for job '{job_name}'")
                continue

            for event in events:
                print(event["message"].strip())
