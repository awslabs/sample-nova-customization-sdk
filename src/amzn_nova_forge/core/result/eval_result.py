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
import logging
import shutil
import tarfile
import tempfile
from abc import ABC
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional
from urllib.parse import urlparse

import boto3

from amzn_nova_forge.core.enums import EvaluationTask
from amzn_nova_forge.core.result.job_result import (
    BaseJobResult,
    BedrockStatusManager,
    JobStatus,
    JobStatusManager,
    MTRLStatusManager,
    SMHPStatusManager,
    SMTJStatusManager,
)

logger = logging.getLogger("nova_forge_sdk")


@dataclass
class EvaluationResult(BaseJobResult, ABC):
    eval_task: EvaluationTask
    eval_output_path: str

    def __init__(
        self,
        job_id: str,
        started_time: datetime,
        eval_task: EvaluationTask,
        eval_output_path: str,
        s3_client=None,
        region: Optional[str] = None,
    ):
        self.eval_task = eval_task
        self.eval_output_path = eval_output_path
        self._cached_results_dir: Optional[str] = None
        self._region = region
        self._s3_client = s3_client or boto3.client("s3", region_name=region)
        super().__init__(job_id, started_time)

    def _download_eval_results(self) -> str:
        # Check if results are already cached
        if self._cached_results_dir and Path(self._cached_results_dir).exists():
            return self._cached_results_dir
        try:
            logger.info(f"Downloading results from {self.eval_output_path}")
            parsed = urlparse(self.eval_output_path)
            bucket = parsed.netloc
            key = parsed.path.lstrip("/")

            # Create temp dir for caching results
            self._cached_results_dir = tempfile.mkdtemp(prefix=f"eval_results_{self.job_id}_")

            # Check if it's a tar.gz file or S3 directory
            if self.eval_output_path.endswith(".tar.gz"):
                # Handle tar.gz file
                local_file = Path(self._cached_results_dir) / "output.tar.gz"
                self._s3_client.download_file(bucket, key, str(local_file))
                with tarfile.open(local_file, "r:gz") as tar:
                    tar.extractall(self._cached_results_dir, filter="data")
            else:
                # Handle S3 directory
                paginator = self._s3_client.get_paginator("list_objects_v2")
                for page in paginator.paginate(Bucket=bucket, Prefix=key):
                    for obj in page.get("Contents", []):
                        obj_key = obj["Key"]
                        relative_path = obj_key[len(key) :].lstrip("/")
                        if relative_path:  # Skip empty paths
                            local_path = Path(self._cached_results_dir) / relative_path
                            local_path.parent.mkdir(parents=True, exist_ok=True)
                            self._s3_client.download_file(bucket, obj_key, str(local_path))

            logger.info(f"Successfully downloaded eval results to {self._cached_results_dir}")
            return self._cached_results_dir
        except Exception as e:
            print(f"Error retrieving evaluation results: {e}")
            print(f"Results available at: {self.eval_output_path}")
            raise

    def get(self) -> Dict:
        job_status, raw_status = self.get_job_status()
        if job_status == JobStatus.COMPLETED:
            print(f"Job '{self.job_id}' completed successfully.")
            results_dir = self._download_eval_results()
            results_files = list(Path(results_dir).rglob("results_*.json"))
            if results_files:
                with open(results_files[0], "r") as f:
                    return json.load(f)
            else:
                logger.info(f"No evaluation results json file found for job {self.job_id}")
                print(f"No evaluation results json file found for job {self.job_id}")
                return {}
        elif job_status == JobStatus.IN_PROGRESS:
            print(
                f"Job '{self.job_id}' still running in progress. Please wait until the job is completed."
            )
            return {}
        else:
            print(f"Cannot show eval result. Job '{self.job_id}' in {raw_status} status.")
            return {}

    def show(self):
        results = self.get()
        if results:
            print(
                f"\nEvaluation Results for job_id={self.job_id}, eval_task={self.eval_task.value}:"
            )
            print(json.dumps(results, indent=2))

    def upload_tensorboard_results(self, tensorboard_s3_path: Optional[str] = None):
        """
        Upload eval tensorboard result to s3
        :param tensorboard_s3_path: Optional, the s3 path you want the tensorboard result upload to. Will use
        eval_output_path if not provide.
        :return: None
        """
        # Ensure results are downloaded first
        results_dir = self._download_eval_results()

        # Search for tensorboard_results directory in subdirectories
        tensorboard_dirs = list(Path(results_dir).rglob("tensorboard_results"))
        if not tensorboard_dirs:
            logger.warning(
                f"No tensorboard_results directory found in {results_dir} or its subdirectories"
            )
            print(f"No tensorboard_results directory found in {results_dir} or its subdirectories")
            return

        tensorboard_dir = tensorboard_dirs[0]  # Use the first found directory

        # Determine target S3 path
        if tensorboard_s3_path is None:
            # Replace output.tar.gz with tensorboard_results/
            eval_s3_path = self.eval_output_path
            tensorboard_s3_path = eval_s3_path.replace("/output.tar.gz", "/tensorboard_results/")

        # Parse S3 path
        parsed = urlparse(tensorboard_s3_path)
        bucket = parsed.netloc
        key_prefix = parsed.path.lstrip("/")

        # Upload all files in tensorboard_results directory
        for file_path in tensorboard_dir.rglob("*"):
            if file_path.is_file():
                relative_path = file_path.relative_to(tensorboard_dir)
                s3_key = f"{key_prefix}{relative_path}"

                try:
                    self._s3_client.upload_file(str(file_path), bucket, s3_key)
                    logger.debug(f"Uploaded {file_path} to s3://{bucket}/{s3_key}")
                except Exception as e:
                    logger.error(f"Failed to upload {file_path}: {e}")
                    raise

        logger.info(f"Successfully uploaded tensorboard results to {tensorboard_s3_path}")
        print(f"Successfully uploaded tensorboard results to {tensorboard_s3_path}")

    def clean(self):
        """Clean up local results cache"""
        if (
            hasattr(self, "_cached_results_dir")
            and self._cached_results_dir
            and Path(self._cached_results_dir).exists()
        ):
            try:
                shutil.rmtree(self._cached_results_dir)
                logger.debug(f"Successfully cleaned temp file in {self._cached_results_dir}")
                self._cached_results_dir = None
            except Exception as e:
                logger.error(
                    f"Failed to remove cached results directory: {self._cached_results_dir} due to {str(e)}"
                )

    def __del__(self):
        # Clean the cache during GC
        self.clean()


@dataclass
class SMTJEvaluationResult(EvaluationResult):
    _job_name: Optional[str] = None

    def __init__(
        self,
        job_id: str,
        started_time: datetime,
        eval_task: EvaluationTask,
        eval_output_path: str,
        sagemaker_client=None,
        s3_client=None,
        region: Optional[str] = None,
    ):
        self._region = region
        self._job_name: Optional[str] = None
        self._sagemaker_client = sagemaker_client or boto3.client("sagemaker", region_name=region)
        super().__init__(
            job_id, started_time, eval_task, eval_output_path, s3_client, region=region
        )

    def _create_status_manager(self) -> JobStatusManager:
        if self.eval_task == EvaluationTask.RFT_MULTITURN_EVAL:
            return MTRLStatusManager(region=self._region)
        return SMTJStatusManager(self._sagemaker_client)

    def _to_dict(self):
        d = {
            "job_id": self.job_id,
            "started_time": self.started_time.isoformat(),
            "eval_task": self.eval_task.value,
            "eval_output_path": self.eval_output_path,
        }
        if self._region:
            d["region"] = self._region
        if getattr(self, "_job_name", None):
            d["job_name"] = self._job_name
        return d

    @classmethod
    def _from_dict(cls, data) -> "SMTJEvaluationResult":
        result = cls(
            job_id=data["job_id"],
            started_time=datetime.fromisoformat(data["started_time"]),
            eval_task=EvaluationTask(data["eval_task"]),
            eval_output_path=data["eval_output_path"],
            region=data.get("region"),
        )
        if data.get("job_name"):
            result._job_name = data["job_name"]
        return result


@dataclass
class SMHPEvaluationResult(EvaluationResult):
    cluster_name: str
    namespace: str

    def __init__(
        self,
        job_id: str,
        started_time: datetime,
        eval_task: EvaluationTask,
        eval_output_path: str,
        cluster_name: str,
        namespace: str = "kubeflow",
        region: Optional[str] = None,
    ):
        self.cluster_name = cluster_name
        self.namespace = namespace
        super().__init__(job_id, started_time, eval_task, eval_output_path, region=region)

    def _create_status_manager(self) -> JobStatusManager:
        return SMHPStatusManager(self.cluster_name, self.namespace)

    def _to_dict(self):
        return {
            "job_id": self.job_id,
            "started_time": self.started_time.isoformat(),
            "eval_task": self.eval_task.value,
            "eval_output_path": self.eval_output_path,
            "cluster_name": self.cluster_name,
            "namespace": self.namespace,
        }

    @classmethod
    def _from_dict(cls, data) -> "SMHPEvaluationResult":
        return cls(
            job_id=data["job_id"],
            started_time=datetime.fromisoformat(data["started_time"]),
            eval_task=EvaluationTask(data["eval_task"]),
            eval_output_path=data["eval_output_path"],
            cluster_name=data["cluster_name"],
            namespace=data["namespace"],
        )


class BedrockEvaluationResult(EvaluationResult):
    """Evaluation result for Bedrock platform."""

    def __init__(
        self,
        job_id: str,
        started_time: datetime,
        eval_task: EvaluationTask,
        eval_output_path: str,
        region: Optional[str] = None,
    ):
        self._region = region
        super().__init__(job_id, started_time, eval_task, eval_output_path, region=region)

    def _create_status_manager(self) -> JobStatusManager:
        return BedrockStatusManager(region=self._region)

    def _to_dict(self):
        return {
            "job_id": self.job_id,
            "started_time": self.started_time.isoformat(),
            "eval_task": self.eval_task.value,
            "eval_output_path": self.eval_output_path,
        }

    @classmethod
    def _from_dict(cls, data) -> "BedrockEvaluationResult":
        return cls(
            job_id=data["job_id"],
            started_time=datetime.fromisoformat(data["started_time"]),
            eval_task=EvaluationTask(data["eval_task"]),
            eval_output_path=data["eval_output_path"],
        )
