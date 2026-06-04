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
Test cases for RFT Multiturn OpenAI tool calling format validation.

This file tests the OpenAI tool calling format including:
- Single and multiple tool calls
- Tool responses with tool_call_id
- String prompts vs OpenAI message format
"""

import json
import tempfile
from pathlib import Path

import pytest

from amzn_nova_forge.core.enums import EvaluationTask, Model, Platform, TrainingMethod
from amzn_nova_forge.dataset import JSONLDatasetLoader


class TestRFTMultiturnToolCalling:
    """Test RFT Multiturn with OpenAI tool calling format."""

    def test_single_tool_call_valid(self):
        """Test that OpenAI messages with single tool call are valid."""
        data = [
            {
                "id": "sample-001",
                "prompt": [
                    {"role": "user", "content": "What's the weather in Boston?"},
                    {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            {
                                "id": "call_abc123",
                                "type": "function",
                                "function": {
                                    "name": "get_weather",
                                    "arguments": '{"location": "Boston"}',
                                },
                            }
                        ],
                    },
                    {
                        "role": "tool",
                        "tool_call_id": "call_abc123",
                        "content": "72 degrees and sunny",
                    },
                    {
                        "role": "assistant",
                        "content": "The weather in Boston is 72 degrees and sunny.",
                    },
                ],
            },
        ]

        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            for record in data:
                f.write(json.dumps(record) + "\n")
            temp_path = f.name

        try:
            loader = JSONLDatasetLoader(id="id", prompt="prompt")
            loader.load(temp_path)
            loader.transform(
                method=TrainingMethod.RFT_MULTITURN_LORA,
                model=Model.NOVA_LITE_2,
                platform=Platform.SMHP,
            )
            loader.validate(method=TrainingMethod.RFT_MULTITURN_LORA, model=Model.NOVA_LITE_2)
        finally:
            Path(temp_path).unlink()

    def test_multiple_tool_calls_valid(self):
        """Test that assistant can make multiple tool calls in one message."""
        data = [
            {
                "id": "sample-001",
                "prompt": [
                    {
                        "role": "user",
                        "content": "What's the weather in Boston and New York?",
                    },
                    {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            {
                                "id": "call_abc123",
                                "type": "function",
                                "function": {
                                    "name": "get_weather",
                                    "arguments": '{"location": "Boston"}',
                                },
                            },
                            {
                                "id": "call_def456",
                                "type": "function",
                                "function": {
                                    "name": "get_weather",
                                    "arguments": '{"location": "New York"}',
                                },
                            },
                        ],
                    },
                    {
                        "role": "tool",
                        "tool_call_id": "call_abc123",
                        "content": "Boston: 72 degrees and sunny",
                    },
                    {
                        "role": "tool",
                        "tool_call_id": "call_def456",
                        "content": "New York: 68 degrees and cloudy",
                    },
                    {
                        "role": "assistant",
                        "content": "Boston is 72 degrees and sunny. New York is 68 degrees and cloudy.",
                    },
                ],
            },
        ]

        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            for record in data:
                f.write(json.dumps(record) + "\n")
            temp_path = f.name

        try:
            loader = JSONLDatasetLoader(id="id", prompt="prompt")
            loader.load(temp_path)
            loader.transform(
                method=TrainingMethod.RFT_MULTITURN_LORA,
                model=Model.NOVA_LITE_2,
                platform=Platform.SMHP,
            )
            loader.validate(method=TrainingMethod.RFT_MULTITURN_LORA, model=Model.NOVA_LITE_2)
        finally:
            Path(temp_path).unlink()

    def test_prompt_as_simple_string_valid(self):
        """Test that prompt can be a simple string (not OpenAI messages)."""
        data = [
            {
                "id": "sample-001",
                "prompt": "What is the capital of France?",
                "answer": "Paris",
            },
            {
                "id": "sample-002",
                "prompt": "Explain quantum physics in simple terms",
                "answer": "Quantum physics studies matter at atomic scale",
            },
        ]

        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            for record in data:
                f.write(json.dumps(record) + "\n")
            temp_path = f.name

        try:
            loader = JSONLDatasetLoader(id="id", prompt="prompt", answer="answer")
            loader.load(temp_path)
            loader.transform(
                method=TrainingMethod.RFT_MULTITURN_LORA,
                model=Model.NOVA_LITE_2,
                platform=Platform.SMHP,
            )
            loader.validate(method=TrainingMethod.RFT_MULTITURN_LORA, model=Model.NOVA_LITE_2)
        finally:
            Path(temp_path).unlink()

    def test_assistant_with_both_content_and_tool_calls_valid(self):
        """Test that assistant can have both content and tool_calls."""
        data = [
            {
                "id": "sample-001",
                "prompt": [
                    {"role": "user", "content": "What's the weather in Boston?"},
                    {
                        "role": "assistant",
                        "content": "Let me check that for you.",
                        "tool_calls": [
                            {
                                "id": "call_abc123",
                                "type": "function",
                                "function": {
                                    "name": "get_weather",
                                    "arguments": '{"location": "Boston"}',
                                },
                            }
                        ],
                    },
                    {
                        "role": "tool",
                        "tool_call_id": "call_abc123",
                        "content": "72 degrees",
                    },
                    {
                        "role": "assistant",
                        "content": "It's 72 degrees in Boston.",
                    },
                ],
            },
        ]

        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            for record in data:
                f.write(json.dumps(record) + "\n")
            temp_path = f.name

        try:
            loader = JSONLDatasetLoader(id="id", prompt="prompt")
            loader.load(temp_path)
            loader.transform(
                method=TrainingMethod.RFT_MULTITURN_LORA,
                model=Model.NOVA_LITE_2,
                platform=Platform.SMHP,
            )
            loader.validate(method=TrainingMethod.RFT_MULTITURN_LORA, model=Model.NOVA_LITE_2)
        finally:
            Path(temp_path).unlink()

    # Failing tests

    def test_tool_message_missing_tool_call_id_fails(self):
        """Test that tool message without tool_call_id fails."""
        data = [
            {
                "id": "sample-001",
                "prompt": [
                    {
                        "role": "tool",
                        "content": "result",  # Missing tool_call_id
                    },
                ],
            },
        ]

        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            for record in data:
                f.write(json.dumps(record) + "\n")
            temp_path = f.name

        try:
            with pytest.raises(ValueError):
                loader = JSONLDatasetLoader(id="id", prompt="prompt")
                loader.load(temp_path)
                loader.transform(
                    method=TrainingMethod.RFT_MULTITURN_LORA,
                    model=Model.NOVA_LITE_2,
                    platform=Platform.SMHP,
                )
                loader.validate(method=TrainingMethod.RFT_MULTITURN_LORA, model=Model.NOVA_LITE_2)
        finally:
            Path(temp_path).unlink()

    def test_assistant_without_content_or_tool_calls_fails(self):
        """Test that assistant message without content or tool_calls fails."""
        data = [
            {
                "id": "sample-001",
                "prompt": [
                    {"role": "user", "content": "Hello"},
                    {
                        "role": "assistant"
                        # Missing both content and tool_calls
                    },
                ],
            },
        ]

        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            for record in data:
                f.write(json.dumps(record) + "\n")
            temp_path = f.name

        try:
            with pytest.raises(ValueError):
                loader = JSONLDatasetLoader(id="id", prompt="prompt")
                loader.load(temp_path)
                loader.transform(
                    method=TrainingMethod.RFT_MULTITURN_LORA,
                    model=Model.NOVA_LITE_2,
                    platform=Platform.SMHP,
                )
                loader.validate(method=TrainingMethod.RFT_MULTITURN_LORA, model=Model.NOVA_LITE_2)
        finally:
            Path(temp_path).unlink()

    def test_empty_string_prompt_fails(self):
        """Test that empty string prompt fails validation."""
        data = [
            {
                "id": "sample-001",
                "prompt": "",  # Empty string
            },
        ]

        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            for record in data:
                f.write(json.dumps(record) + "\n")
            temp_path = f.name

        try:
            with pytest.raises(ValueError):
                loader = JSONLDatasetLoader(id="id", prompt="prompt")
                loader.load(temp_path)
                loader.transform(
                    method=TrainingMethod.RFT_MULTITURN_LORA,
                    model=Model.NOVA_LITE_2,
                    platform=Platform.SMHP,
                )
                loader.validate(method=TrainingMethod.RFT_MULTITURN_LORA, model=Model.NOVA_LITE_2)
        finally:
            Path(temp_path).unlink()

    def test_tool_call_invalid_type_fails(self):
        """Test that tool call with invalid type fails."""
        data = [
            {
                "id": "sample-001",
                "prompt": [
                    {"role": "user", "content": "Hello"},
                    {
                        "role": "assistant",
                        "tool_calls": [
                            {
                                "id": "call_123",
                                "type": "invalid_type",  # Must be 'function'
                                "function": {
                                    "name": "test",
                                    "arguments": "{}",
                                },
                            }
                        ],
                    },
                ],
            },
        ]

        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            for record in data:
                f.write(json.dumps(record) + "\n")
            temp_path = f.name

        try:
            with pytest.raises(ValueError):
                loader = JSONLDatasetLoader(id="id", prompt="prompt")
                loader.load(temp_path)
                loader.transform(
                    method=TrainingMethod.RFT_MULTITURN_LORA,
                    model=Model.NOVA_LITE_2,
                    platform=Platform.SMHP,
                )
                loader.validate(method=TrainingMethod.RFT_MULTITURN_LORA, model=Model.NOVA_LITE_2)
        finally:
            Path(temp_path).unlink()

    def test_extra_fields_forbidden_in_metadata(self):
        """Test that extra fields in metadata are rejected (extra='forbid')."""
        data = [
            {
                "id": "sample-001",
                "metadata": {
                    "prompt": "Hello",
                    "extra_field": "should_fail",  # Extra field not in schema
                },
            },
        ]

        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            for record in data:
                f.write(json.dumps(record) + "\n")
            temp_path = f.name

        try:
            with pytest.raises(ValueError, match="Extra inputs are not permitted"):
                loader = JSONLDatasetLoader(id="id", prompt="prompt")
                loader.load(temp_path)
                loader.transform(
                    method=TrainingMethod.RFT_MULTITURN_LORA,
                    model=Model.NOVA_LITE_2,
                    platform=Platform.SMHP,
                )
                loader.validate(method=TrainingMethod.RFT_MULTITURN_LORA, model=Model.NOVA_LITE_2)
        finally:
            Path(temp_path).unlink()

    def test_extra_fields_forbidden_at_top_level(self):
        """Test that extra fields at top level are rejected (extra='forbid')."""
        data = [
            {
                "id": "sample-001",
                "metadata": {
                    "prompt": "Hello",
                },
                "unexpected_field": "should_fail",  # Extra field not in schema
            },
        ]

        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            for record in data:
                f.write(json.dumps(record) + "\n")
            temp_path = f.name

        try:
            with pytest.raises(ValueError, match="Extra inputs are not permitted"):
                loader = JSONLDatasetLoader(id="id", prompt="prompt")
                loader.load(temp_path)
                loader.transform(
                    method=TrainingMethod.RFT_MULTITURN_LORA,
                    model=Model.NOVA_LITE_2,
                    platform=Platform.SMHP,
                )
                loader.validate(method=TrainingMethod.RFT_MULTITURN_LORA, model=Model.NOVA_LITE_2)
        finally:
            Path(temp_path).unlink()

    def test_evaluation_method_with_rft_multiturn_eval_task(self):
        """Test that EVALUATION method with RFT_MULTITURN_EVAL task works correctly."""
        data = [
            {
                "id": "sample-001",
                "metadata": {
                    "prompt": "What is 2+2?",
                    "answer": "4",
                },
            },
        ]

        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            for record in data:
                f.write(json.dumps(record) + "\n")
            temp_path = f.name

        try:
            loader = JSONLDatasetLoader(id="id", prompt="prompt")
            loader.load(temp_path)

            # Transform with EVALUATION method and RFT_MULTITURN_EVAL task
            loader.transform(
                method=TrainingMethod.EVALUATION,
                eval_task=EvaluationTask.RFT_MULTITURN_EVAL,
                model=Model.NOVA_LITE_2,
                platform=Platform.SMHP,
            )

            # Validate with EVALUATION method and RFT_MULTITURN_EVAL task
            loader.validate(
                method=TrainingMethod.EVALUATION,
                eval_task=EvaluationTask.RFT_MULTITURN_EVAL,
                model=Model.NOVA_LITE_2,
            )

            # If we get here, validation passed
            assert True
        finally:
            Path(temp_path).unlink()

    def test_evaluation_method_with_rft_multiturn_eval_openai_format(self):
        """Test EVALUATION method with RFT_MULTITURN_EVAL using OpenAI message format."""
        data = [
            {
                "id": "sample-001",
                "metadata": {
                    "prompt": [
                        {"role": "user", "content": "What's the weather?"},
                        {
                            "role": "assistant",
                            "content": None,
                            "tool_calls": [
                                {
                                    "id": "call_123",
                                    "type": "function",
                                    "function": {
                                        "name": "get_weather",
                                        "arguments": '{"location": "Boston"}',
                                    },
                                }
                            ],
                        },
                        {
                            "role": "tool",
                            "tool_call_id": "call_123",
                            "content": "72 degrees",
                        },
                        {"role": "assistant", "content": "It's 72 degrees in Boston."},
                    ],
                },
            },
        ]

        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            for record in data:
                f.write(json.dumps(record) + "\n")
            temp_path = f.name

        try:
            loader = JSONLDatasetLoader(id="id", prompt="prompt")
            loader.load(temp_path)

            # Transform with EVALUATION method and RFT_MULTITURN_EVAL task
            loader.transform(
                method=TrainingMethod.EVALUATION,
                eval_task=EvaluationTask.RFT_MULTITURN_EVAL,
                model=Model.NOVA_LITE_2,
                platform=Platform.SMHP,
            )

            # Validate with EVALUATION method and RFT_MULTITURN_EVAL task
            loader.validate(
                method=TrainingMethod.EVALUATION,
                eval_task=EvaluationTask.RFT_MULTITURN_EVAL,
                model=Model.NOVA_LITE_2,
            )

            # If we get here, validation passed
            assert True
        finally:
            Path(temp_path).unlink()

    def test_evaluation_method_rft_multiturn_eval_invalid_data_fails(self):
        """Test that EVALUATION method with RFT_MULTITURN_EVAL rejects invalid data."""
        data = [
            {
                "id": "sample-001",
                "metadata": {
                    "prompt": "",  # Empty prompt should fail
                },
            },
        ]

        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            for record in data:
                f.write(json.dumps(record) + "\n")
            temp_path = f.name

        try:
            with pytest.raises(ValueError):
                loader = JSONLDatasetLoader(id="id", prompt="prompt")
                loader.load(temp_path)
                loader.transform(
                    method=TrainingMethod.EVALUATION,
                    eval_task=EvaluationTask.RFT_MULTITURN_EVAL,
                    model=Model.NOVA_LITE_2,
                )
                loader.validate(
                    method=TrainingMethod.EVALUATION,
                    eval_task=EvaluationTask.RFT_MULTITURN_EVAL,
                    model=Model.NOVA_LITE_2,
                )
        finally:
            Path(temp_path).unlink()
