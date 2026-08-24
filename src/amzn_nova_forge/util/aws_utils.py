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
"""AWS utility helpers shared across the SDK."""

import logging
from typing import Optional

import boto3

logger = logging.getLogger(__name__)

_account_id_cache: Optional[str] = None


def get_caller_account_id(region: str = "us-east-1") -> str:
    """Return the AWS account ID of the caller, cached to avoid redundant STS calls.

    Only caches successful results — transient STS failures return "*" without poisoning
    the cache, so subsequent calls will retry.
    """
    global _account_id_cache
    if _account_id_cache is None:
        try:
            _account_id_cache = boto3.client("sts", region_name=region).get_caller_identity()[
                "Account"
            ]
        except Exception:
            logger.warning(
                "Failed to retrieve caller account ID via STS in region %s; "
                "falling back to wildcard '*'",
                region,
                exc_info=True,
            )
            return "*"
    return _account_id_cache
