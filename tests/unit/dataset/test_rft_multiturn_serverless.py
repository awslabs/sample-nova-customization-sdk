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
Test cases for RFT Multiturn Serverless dataset transformation and validation.

Tests the platform-specific behavior:
- Platform.SMTJServerless: flat {"prompt": "..."} format
- Platform.SMHP: nested {"id": "...", "metadata": {"prompt": "..."}} format
- No platform: raises ValueError for RFT Multiturn
"""

import json
import tempfile
from pathlib import Path

import pytest

from amzn_nova_forge.core.enums import Model, Platform, TrainingMethod
from amzn_nova_forge.dataset import JSONLDatasetLoader
from amzn_nova_forge.dataset.dataset_validator.rft_multiturn_dataset_validator import (
    RFTMultiturnServerlessSample,
    RFTMultiturnServerlessValidator,
)


def _write_jsonl(data):
    """Write data to a temp jsonl file and return the path."""
    f = tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False)
    for record in data:
        f.write(json.dumps(record) + "\n")
    f.close()
    return f.name


def _read_jsonl(path):
    """Read jsonl file and return list of dicts."""
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


class TestServerlessTransform:
    """Test transform with Platform.SMTJServerless produces flat format."""

    def test_flat_input_produces_flat_output(self):
        data = [
            {"id": "q1", "prompt": "What is 2+2?", "answer": "4"},
            {"id": "q2", "prompt": "What is 3+3?", "answer": "6"},
        ]
        path = _write_jsonl(data)
        try:
            loader = JSONLDatasetLoader(id="id", prompt="prompt", answer="answer")
            loader.load(path)
            loader.transform(
                method=TrainingMethod.RFT_MULTITURN_LORA,
                model=Model.NOVA_LITE_2,
                platform=Platform.SMTJServerless,
            )
            output = tempfile.mktemp(suffix=".jsonl")
            loader.save(output)
            results = _read_jsonl(output)

            assert len(results) == 2
            assert results[0] == {"prompt": "What is 2+2?"}
            assert results[1] == {"prompt": "What is 3+3?"}
        finally:
            Path(path).unlink()

    def test_json_string_prompt_preserved(self):
        data = [
            {
                "id": "q1",
                "prompt": '{"instance": "Solve 3+3", "reward_spec": {"ground_truth": "6"}}',
            },
        ]
        path = _write_jsonl(data)
        try:
            loader = JSONLDatasetLoader(id="id", prompt="prompt")
            loader.load(path)
            loader.transform(
                method=TrainingMethod.RFT_MULTITURN_LORA,
                model=Model.NOVA_LITE_2,
                platform=Platform.SMTJServerless,
            )
            output = tempfile.mktemp(suffix=".jsonl")
            loader.save(output)
            results = _read_jsonl(output)

            assert len(results) == 1
            assert (
                results[0]["prompt"]
                == '{"instance": "Solve 3+3", "reward_spec": {"ground_truth": "6"}}'
            )
        finally:
            Path(path).unlink()

    def test_nested_input_extracts_prompt(self):
        """When input has metadata.prompt (nested format), serverless transform extracts it."""
        data = [
            {"metadata": {"prompt": "Hello world"}},
        ]
        path = _write_jsonl(data)
        try:
            loader = JSONLDatasetLoader(prompt="prompt")
            loader.load(path)
            loader.transform(
                method=TrainingMethod.RFT_MULTITURN_LORA,
                model=Model.NOVA_LITE_2,
                platform=Platform.SMTJServerless,
            )
            output = tempfile.mktemp(suffix=".jsonl")
            loader.save(output)
            results = _read_jsonl(output)

            assert results[0] == {"prompt": "Hello world"}
        finally:
            Path(path).unlink()


class TestHyperPodTransform:
    """Test transform with Platform.SMHP produces nested format."""

    def test_produces_nested_format(self):
        data = [
            {"id": "q1", "prompt": "What is 2+2?", "answer": "4"},
        ]
        path = _write_jsonl(data)
        try:
            loader = JSONLDatasetLoader(id="id", prompt="prompt", answer="answer")
            loader.load(path)
            loader.transform(
                method=TrainingMethod.RFT_MULTITURN_LORA,
                model=Model.NOVA_LITE_2,
                platform=Platform.SMHP,
            )
            output = tempfile.mktemp(suffix=".jsonl")
            loader.save(output)
            results = _read_jsonl(output)

            assert results[0] == {"id": "q1", "metadata": {"prompt": "What is 2+2?", "answer": "4"}}
        finally:
            Path(path).unlink()


class TestPlatformRequired:
    """Test that platform is required for RFT Multiturn transforms."""

    def test_no_platform_raises_error(self):
        data = [{"id": "q1", "prompt": "test"}]
        path = _write_jsonl(data)
        try:
            loader = JSONLDatasetLoader(id="id", prompt="prompt")
            loader.load(path)
            loader.transform(
                method=TrainingMethod.RFT_MULTITURN_LORA,
                model=Model.NOVA_LITE_2,
            )
            with pytest.raises(ValueError, match="platform is required"):
                loader.save(tempfile.mktemp(suffix=".jsonl"))
        finally:
            Path(path).unlink()

    def test_sft_does_not_require_platform(self):
        """Non-MTRL methods should not require platform."""
        data = [
            {
                "messages": [
                    {"role": "user", "content": [{"text": "hi"}]},
                    {"role": "assistant", "content": [{"text": "hello"}]},
                ]
            },
        ]
        path = _write_jsonl(data)
        try:
            loader = JSONLDatasetLoader()
            loader.load(path)
            # SFT should not raise even without platform
            loader.transform(
                method=TrainingMethod.SFT_LORA,
                model=Model.NOVA_LITE_2,
            )
            # Just verify no ValueError about platform — don't save (avoids STS calls)
            loader.execute()
        finally:
            Path(path).unlink()


class TestServerlessValidator:
    """Test RFTMultiturnServerlessValidator."""

    def test_valid_sample(self):
        sample = {"prompt": "What is 2+2?"}
        RFTMultiturnServerlessSample.model_validate(sample)

    def test_empty_prompt_fails(self):
        with pytest.raises(Exception, match="cannot be empty"):
            RFTMultiturnServerlessSample.model_validate({"prompt": ""})

    def test_missing_prompt_fails(self):
        with pytest.raises(Exception):
            RFTMultiturnServerlessSample.model_validate({"id": "q1"})

    def test_extra_fields_allowed(self):
        sample = {"prompt": "test", "extra_field": "value"}
        validated = RFTMultiturnServerlessSample.model_validate(sample)
        assert validated.prompt == "test"

    def test_validator_init_rejects_non_lite2(self):
        with pytest.raises(ValueError, match="NOVA_LITE_2"):
            RFTMultiturnServerlessValidator(Model.NOVA_LITE)

    def test_validator_init_accepts_lite2(self):
        validator = RFTMultiturnServerlessValidator(Model.NOVA_LITE_2)
        assert validator is not None


class TestServerlessValidateOperation:
    """Test validate() with platform=Platform.SMTJServerless."""

    def test_validate_serverless_format(self):
        data = [
            {"prompt": "What is 2+2?"},
            {"prompt": "Explain gravity"},
        ]
        path = _write_jsonl(data)
        try:
            loader = JSONLDatasetLoader(prompt="prompt")
            loader.load(path)
            loader.validate(
                method=TrainingMethod.RFT_MULTITURN_LORA,
                model=Model.NOVA_LITE_2,
                platform=Platform.SMTJServerless,
            )
            loader.execute()
        finally:
            Path(path).unlink()

    def test_validate_rejects_empty_prompt(self):
        """Validator rejects samples with empty/whitespace-only prompt."""
        sample = {"prompt": "   "}
        with pytest.raises(Exception, match="cannot be empty"):
            RFTMultiturnServerlessSample.model_validate(sample)
