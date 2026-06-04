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
"""Utilities for handling subprocess stderr output from HyperPod CLI commands."""

import logging
import re

logger = logging.getLogger(__name__)

# Matches the Python warning file:line: prefix format
# e.g. "/path/to/urllib3/__init__.py:35: " or "module.py:10: "
_WARNING_FILE_PREFIX = r"(?:(?:\S+/)*\S+(?:\.\w+)?:\d*: )"

# Benign warning class names that should not be treated as errors
_BENIGN_WARNING_NAMES = [
    r"NotOpenSSLWarning:",
    r"DeprecationWarning:",
    r"FutureWarning:",
    r"InsecureRequestWarning:",
    r"RequestsDependencyWarning:",
    r"ResourceWarning:",
    r"UserWarning:",
    r"urllib3\.[a-zA-Z_]+Warning:",
]

# Each pattern matches either:
#   1. file:line: WarningClass: (standard Python warning format)
#   2. WarningClass: at line start (bare warning without file prefix)
_BENIGN_STDERR_PATTERNS = [
    re.compile(rf"^(?:{_WARNING_FILE_PREFIX})?{name}", re.MULTILINE)
    for name in _BENIGN_WARNING_NAMES
]


def _check_hyperpod_stderr(stderr: str) -> None:
    """Check HyperPod CLI stderr output and raise if it contains real errors.

    The HyperPod CLI can exit 0 while writing real errors to stderr
    (e.g. "Cluster with name not found"). This function distinguishes
    benign warnings (urllib3, deprecation, etc.) from actual errors.

    Args:
        stderr: The stderr output from a subprocess call.

    Raises:
        RuntimeError: If stderr contains lines that don't match known benign patterns.
    """
    if not stderr or not stderr.strip():
        return

    non_benign_lines = []
    prev_stripped = None
    last_benign_stripped = None
    for line in stderr.strip().splitlines():
        stripped = line.strip()
        prev_line_benign = (
            last_benign_stripped is not None and last_benign_stripped == prev_stripped
        )
        last_benign_stripped = None  # one-shot: only protects the immediately next line
        prev_stripped = stripped

        if not stripped:
            continue
        if any(pattern.search(stripped) for pattern in _BENIGN_STDERR_PATTERNS):
            last_benign_stripped = stripped
            continue
        # Python warnings module indents the source line following the warning
        if prev_line_benign and line.startswith((" ", "\t")):
            continue
        non_benign_lines.append(stripped)

    if not non_benign_lines:
        logger.debug("HyperPod CLI stderr (benign warnings only): %s", stderr.strip())
        return

    error_output = "\n".join(non_benign_lines)
    logger.error("HyperPod CLI stderr contains errors: %s", error_output)
    raise RuntimeError(stderr.strip())
