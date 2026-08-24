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
"""Unit tests for amzn_nova_forge.util.aws_utils."""

import unittest
from unittest.mock import MagicMock, patch

from botocore.exceptions import ClientError

from amzn_nova_forge.util import aws_utils
from amzn_nova_forge.util.aws_utils import get_caller_account_id


class TestGetCallerAccountId(unittest.TestCase):
    """Tests for get_caller_account_id caching and fallback behavior."""

    def setUp(self):
        # Reset module-level cache between tests
        aws_utils._account_id_cache = None

    @patch("amzn_nova_forge.util.aws_utils.boto3.client")
    def test_returns_account_id_on_success(self, mock_client):
        mock_sts = MagicMock()
        mock_sts.get_caller_identity.return_value = {"Account": "123456789012"}
        mock_client.return_value = mock_sts

        result = get_caller_account_id("us-east-1")

        self.assertEqual(result, "123456789012")
        mock_client.assert_called_once_with("sts", region_name="us-east-1")

    @patch("amzn_nova_forge.util.aws_utils.boto3.client")
    def test_caches_successful_result(self, mock_client):
        mock_sts = MagicMock()
        mock_sts.get_caller_identity.return_value = {"Account": "123456789012"}
        mock_client.return_value = mock_sts

        first = get_caller_account_id("us-east-1")
        second = get_caller_account_id("us-west-2")

        self.assertEqual(first, "123456789012")
        self.assertEqual(second, "123456789012")
        # Only one STS call — second call uses cache
        mock_sts.get_caller_identity.assert_called_once()

    @patch("amzn_nova_forge.util.aws_utils.boto3.client")
    def test_returns_wildcard_on_exception(self, mock_client):
        mock_sts = MagicMock()
        mock_sts.get_caller_identity.side_effect = ClientError(
            {"Error": {"Code": "ExpiredToken", "Message": "token expired"}},
            "GetCallerIdentity",
        )
        mock_client.return_value = mock_sts

        result = get_caller_account_id("us-east-1")

        self.assertEqual(result, "*")

    @patch("amzn_nova_forge.util.aws_utils.boto3.client")
    def test_does_not_cache_failed_result(self, mock_client):
        mock_sts = MagicMock()
        # First call fails
        mock_sts.get_caller_identity.side_effect = [
            Exception("network error"),
            {"Account": "123456789012"},
        ]
        mock_client.return_value = mock_sts

        first = get_caller_account_id("us-east-1")
        second = get_caller_account_id("us-east-1")

        self.assertEqual(first, "*")
        self.assertEqual(second, "123456789012")
        # Two STS calls — failure was not cached
        self.assertEqual(mock_sts.get_caller_identity.call_count, 2)
