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
"""
Shared validation patterns for Nova Forge SDK.

Single source of truth for regex patterns used by both core/ and validation/.
"""

import re

# https://docs.aws.amazon.com/sagemaker/latest/APIReference/API_TrainingJob.html
JOB_NAME_REGEX = re.compile(r"^[a-zA-Z0-9]([a-zA-Z0-9-]{0,61}[a-zA-Z0-9])?$")

# https://kubernetes.io/docs/concepts/overview/working-with-objects/namespaces/
NAMESPACE_REGEX = re.compile(r"^[a-z0-9]([a-z0-9-]{0,61}[a-z0-9])?$")

# https://docs.aws.amazon.com/eks/latest/APIReference/API_CreateCluster.html#API_CreateCluster_RequestParameters
CLUSTER_NAME_REGEX = re.compile(r"^[0-9A-Za-z][A-Za-z0-9\-_]{1,100}$")

MODEL_PACKAGE_ARN_REGEX = re.compile(
    r"^arn:aws[\w-]*:sagemaker:[a-z0-9-]+:\d{12}:model-package/.+$"
)


def validate_job_name(job_name: str) -> None:
    if not JOB_NAME_REGEX.match(job_name):
        raise ValueError(f"Job name must fit pattern {JOB_NAME_REGEX.pattern}")


def validate_namespace(namespace: str) -> None:
    if not NAMESPACE_REGEX.match(namespace):
        raise ValueError(f"Namespace must fit pattern {NAMESPACE_REGEX.pattern}")


def validate_cluster_name(cluster_name: str) -> None:
    if not CLUSTER_NAME_REGEX.match(cluster_name):
        raise ValueError(f"Cluster name must fit pattern {CLUSTER_NAME_REGEX.pattern}")
