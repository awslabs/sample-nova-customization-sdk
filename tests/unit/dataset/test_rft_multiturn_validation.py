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
Test cases for RFT Multiturn data validation and transformation.

This file contains both passing and failing examples to demonstrate
proper usage of the RFT Multiturn dataset loader and validator.
"""

import json
import tempfile
from pathlib import Path

import pytest

from amzn_nova_forge.core.enums import EvaluationTask, Model, Platform, TrainingMethod
from amzn_nova_forge.dataset import (
    CSVDatasetLoader,
    JSONDatasetLoader,
    JSONLDatasetLoader,
)


class TestRFTMultiturnValidation:
    """Test RFT Multiturn validation with passing and failing examples."""

    # ========== PASSING EXAMPLES ==========

    def test_flat_format_minimal_valid(self):
        """Test flat format with only required fields (id and prompt)."""
        data = [
            {"id": "sample-001", "prompt": "What is 2+2?"},
            {"id": "sample-002", "prompt": "Explain quantum physics"},
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
            loader.execute()
            loader.execute()
            loader.validate(method=TrainingMethod.RFT_MULTITURN_LORA, model=Model.NOVA_LITE_2)
        finally:
            Path(temp_path).unlink()

    def test_flat_format_all_fields_valid(self):
        """Test flat format with all optional fields."""
        data = [
            {
                "id": "sample-001",
                "prompt": "What is 2+2?",
                "answer": "4",
                "task": "math",
                "info": {"difficulty": "easy", "category": "arithmetic"},
            },
            {
                "id": "sample-002",
                "prompt": "What is the capital of France?",
                "answer": "Paris",
                "task": "geography",
                "info": {"difficulty": "easy", "region": "Europe"},
            },
        ]

        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            for record in data:
                f.write(json.dumps(record) + "\n")
            temp_path = f.name

        try:
            loader = JSONLDatasetLoader(
                id="id", prompt="prompt", answer="answer", task="task", info="info"
            )
            loader.load(temp_path)
            loader.transform(
                method=TrainingMethod.RFT_MULTITURN_LORA,
                model=Model.NOVA_LITE_2,
                platform=Platform.SMHP,
            )
            loader.execute()
            loader.execute()
            loader.validate(method=TrainingMethod.RFT_MULTITURN_LORA, model=Model.NOVA_LITE_2)
        finally:
            Path(temp_path).unlink()

    def test_nested_format_valid(self):
        """Test nested format (already in correct structure)."""
        data = [
            {
                "id": "sample-001",
                "metadata": {
                    "prompt": "What is 2+2?",
                    "answer": "4",
                    "task": "math",
                    "info": {"difficulty": "easy"},
                },
            },
            {
                "id": "sample-002",
                "metadata": {
                    "prompt": "Explain photosynthesis",
                    "answer": "Process by which plants convert light to energy",
                    "task": "biology",
                    "info": {"difficulty": "medium"},
                },
            },
        ]

        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            for record in data:
                f.write(json.dumps(record) + "\n")
            temp_path = f.name

        try:
            loader = JSONLDatasetLoader(id="id", metadata="metadata")
            loader.load(temp_path)
            loader.transform(
                method=TrainingMethod.RFT_MULTITURN_LORA,
                model=Model.NOVA_LITE_2,
                platform=Platform.SMHP,
            )
            loader.execute()
            loader.execute()
            loader.validate(method=TrainingMethod.RFT_MULTITURN_LORA, model=Model.NOVA_LITE_2)
        finally:
            Path(temp_path).unlink()

    def test_openai_messages_format_valid(self):
        """Test prompt as OpenAI messages format (multi-turn conversation)."""
        data = [
            {
                "id": "sample-001",
                "prompt": [
                    {"role": "system", "content": "You are a helpful math tutor."},
                    {"role": "user", "content": "What is 2+2?"},
                ],
                "answer": "4",
                "task": "math",
                "info": {},
            },
            {
                "id": "sample-002",
                "prompt": [
                    {"role": "user", "content": "What is the capital of France?"},
                    {"role": "assistant", "content": "The capital is Paris."},
                    {"role": "user", "content": "What about Germany?"},
                ],
                "answer": "Berlin",
                "task": "geography",
                "info": {},
            },
        ]

        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            for record in data:
                f.write(json.dumps(record) + "\n")
            temp_path = f.name

        try:
            loader = JSONLDatasetLoader(
                id="id", prompt="prompt", answer="answer", task="task", info="info"
            )
            loader.load(temp_path)
            loader.transform(
                method=TrainingMethod.RFT_MULTITURN_LORA,
                model=Model.NOVA_LITE_2,
                platform=Platform.SMHP,
            )
            loader.execute()
            loader.execute()
            loader.validate(method=TrainingMethod.RFT_MULTITURN_LORA, model=Model.NOVA_LITE_2)
        finally:
            Path(temp_path).unlink()

    def test_empty_info_dict_valid(self):
        """Test with empty info dict (valid)."""
        data = [
            {"id": "sample-001", "prompt": "Hello world", "info": {}},
        ]

        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            for record in data:
                f.write(json.dumps(record) + "\n")
            temp_path = f.name

        try:
            loader = JSONLDatasetLoader(id="id", prompt="prompt", info="info")
            loader.load(temp_path)
            loader.transform(
                method=TrainingMethod.RFT_MULTITURN_LORA,
                model=Model.NOVA_LITE_2,
                platform=Platform.SMHP,
            )
            loader.execute()
            loader.execute()
            loader.validate(method=TrainingMethod.RFT_MULTITURN_LORA, model=Model.NOVA_LITE_2)
        finally:
            Path(temp_path).unlink()

    def test_json_format_valid(self):
        """Test JSON format (not JSONL)."""
        data = [
            {"id": "sample-001", "prompt": "What is AI?", "info": {}},
            {"id": "sample-002", "prompt": "Explain ML", "info": {}},
        ]

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(data, f)
            temp_path = f.name

        try:
            loader = JSONDatasetLoader(id="id", prompt="prompt", info="info")
            loader.load(temp_path)
            loader.transform(
                method=TrainingMethod.RFT_MULTITURN_LORA,
                model=Model.NOVA_LITE_2,
                platform=Platform.SMHP,
            )
            loader.execute()
            loader.execute()
            loader.validate(method=TrainingMethod.RFT_MULTITURN_LORA, model=Model.NOVA_LITE_2)
        finally:
            Path(temp_path).unlink()

    def test_csv_format_valid(self):
        """Test CSV format with JSON string in info column."""
        csv_content = """id,prompt,answer,task,info
sample-001,"What is 2+2?","4","math","{""difficulty"": ""easy""}"
sample-002,"Capital of France?","Paris","geography","{""difficulty"": ""easy""}"
"""

        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            f.write(csv_content)
            temp_path = f.name

        try:
            loader = CSVDatasetLoader(
                id="id", prompt="prompt", answer="answer", task="task", info="info"
            )
            loader.load(temp_path)
            loader.transform(
                method=TrainingMethod.RFT_MULTITURN_LORA,
                model=Model.NOVA_LITE_2,
                platform=Platform.SMHP,
            )
            loader.execute()
            loader.execute()
            loader.validate(method=TrainingMethod.RFT_MULTITURN_LORA, model=Model.NOVA_LITE_2)
        finally:
            Path(temp_path).unlink()

    # ========== FAILING EXAMPLES ==========

    def test_info_as_invalid_json_string_fails(self):
        """Test that info field as invalid JSON string fails validation."""
        data = [
            {
                "id": "sample-001",
                "prompt": "What is 2+2?",
                "info": "{'text': 'invalid json'}",  # Invalid JSON (single quotes)
            },
        ]

        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            for record in data:
                f.write(json.dumps(record) + "\n")
            temp_path = f.name

        try:
            with pytest.raises(Exception):
                loader = JSONLDatasetLoader(id="id", prompt="prompt", info="info")
                loader.load(temp_path)
                loader.transform(
                    method=TrainingMethod.RFT_MULTITURN_LORA,
                    model=Model.NOVA_LITE_2,
                    platform=Platform.SMHP,
                )
                loader.execute()
                loader.execute()
                loader.validate(method=TrainingMethod.RFT_MULTITURN_LORA, model=Model.NOVA_LITE_2)
        finally:
            Path(temp_path).unlink()

    def test_empty_prompt_fails(self):
        """Test that empty prompt fails validation."""
        data = [
            {"id": "sample-001", "prompt": ""},  # Empty prompt
        ]

        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            for record in data:
                f.write(json.dumps(record) + "\n")
            temp_path = f.name

        try:
            with pytest.raises(Exception):
                loader = JSONLDatasetLoader(id="id", prompt="prompt")
                loader.load(temp_path)
                loader.transform(
                    method=TrainingMethod.RFT_MULTITURN_LORA,
                    model=Model.NOVA_LITE_2,
                    platform=Platform.SMHP,
                )
                loader.execute()
                loader.execute()
                loader.validate(method=TrainingMethod.RFT_MULTITURN_LORA, model=Model.NOVA_LITE_2)
        finally:
            Path(temp_path).unlink()

    def test_missing_prompt_fails(self):
        """Test that missing prompt fails validation."""
        data = [
            {"id": "sample-001", "answer": "4"},  # Missing prompt
        ]

        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            for record in data:
                f.write(json.dumps(record) + "\n")
            temp_path = f.name

        try:
            with pytest.raises(Exception):
                loader = JSONLDatasetLoader(id="id", prompt="prompt")
                loader.load(temp_path)
                loader.transform(
                    method=TrainingMethod.RFT_MULTITURN_LORA,
                    model=Model.NOVA_LITE_2,
                    platform=Platform.SMHP,
                )
                loader.execute()
                loader.execute()
                loader.validate(method=TrainingMethod.RFT_MULTITURN_LORA, model=Model.NOVA_LITE_2)
        finally:
            Path(temp_path).unlink()

    def test_invalid_openai_role_fails(self):
        """Test that invalid role in OpenAI messages fails."""
        data = [
            {
                "id": "sample-001",
                "prompt": [
                    {"role": "invalid_role", "content": "Hello"},  # Invalid role
                ],
            },
        ]

        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            for record in data:
                f.write(json.dumps(record) + "\n")
            temp_path = f.name

        try:
            with pytest.raises(Exception):
                loader = JSONLDatasetLoader(id="id", prompt="prompt")
                loader.load(temp_path)
                loader.transform(
                    method=TrainingMethod.RFT_MULTITURN_LORA,
                    model=Model.NOVA_LITE_2,
                    platform=Platform.SMHP,
                )
                loader.execute()
                loader.execute()
                loader.validate(method=TrainingMethod.RFT_MULTITURN_LORA, model=Model.NOVA_LITE_2)
        finally:
            Path(temp_path).unlink()

    def test_openai_missing_content_fails(self):
        """Test that OpenAI message without content fails."""
        data = [
            {
                "id": "sample-001",
                "prompt": [
                    {"role": "user"},  # Missing content
                ],
            },
        ]

        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            for record in data:
                f.write(json.dumps(record) + "\n")
            temp_path = f.name

        try:
            with pytest.raises(Exception):
                loader = JSONLDatasetLoader(id="id", prompt="prompt")
                loader.load(temp_path)
                loader.transform(
                    method=TrainingMethod.RFT_MULTITURN_LORA,
                    model=Model.NOVA_LITE_2,
                    platform=Platform.SMHP,
                )
                loader.execute()
                loader.execute()
                loader.validate(method=TrainingMethod.RFT_MULTITURN_LORA, model=Model.NOVA_LITE_2)
        finally:
            Path(temp_path).unlink()

    def test_empty_openai_content_fails(self):
        """Test that empty content in OpenAI message fails."""
        data = [
            {
                "id": "sample-001",
                "prompt": [
                    {"role": "user", "content": ""},  # Empty content
                ],
            },
        ]

        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            for record in data:
                f.write(json.dumps(record) + "\n")
            temp_path = f.name

        try:
            with pytest.raises(Exception):
                loader = JSONLDatasetLoader(id="id", prompt="prompt")
                loader.load(temp_path)
                loader.transform(
                    method=TrainingMethod.RFT_MULTITURN_LORA,
                    model=Model.NOVA_LITE_2,
                    platform=Platform.SMHP,
                )
                loader.execute()
                loader.execute()
                loader.validate(method=TrainingMethod.RFT_MULTITURN_LORA, model=Model.NOVA_LITE_2)
        finally:
            Path(temp_path).unlink()

    # ========== ID AUTO-GENERATION TESTS ==========

    def test_id_auto_generation_when_missing(self):
        """Test that IDs are auto-generated when missing from flat format."""
        data = [
            {"prompt": "Question 1"},  # No ID
            {"prompt": "Question 2"},  # No ID
            {"prompt": "Question 3"},  # No ID
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
            loader.execute()
            loader.execute()

            # Check that IDs were auto-generated
            transformed_data = list(loader.dataset())
            assert len(transformed_data) == 3
            assert transformed_data[0]["id"] == "sample_001"
            assert transformed_data[1]["id"] == "sample_002"
            assert transformed_data[2]["id"] == "sample_003"
        finally:
            Path(temp_path).unlink()

    def test_id_auto_generation_when_empty_string(self):
        """Test that IDs are auto-generated when empty string in flat format."""
        data = [
            {"id": "", "prompt": "Question 1"},  # Empty ID
            {"id": "", "prompt": "Question 2"},  # Empty ID
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
            loader.execute()
            loader.execute()

            # Check that IDs were auto-generated
            transformed_data = list(loader.dataset())
            assert len(transformed_data) == 2
            assert transformed_data[0]["id"] == "sample_001"
            assert transformed_data[1]["id"] == "sample_002"
        finally:
            Path(temp_path).unlink()

    def test_id_auto_generation_nested_format(self):
        """Test that IDs are auto-generated for nested format when missing."""
        data = [
            {"metadata": {"prompt": "Question 1"}},  # No ID
            {"metadata": {"prompt": "Question 2"}},  # No ID
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
            loader.execute()
            loader.execute()

            # Check that IDs were auto-generated
            transformed_data = list(loader.dataset())
            assert len(transformed_data) == 2
            assert transformed_data[0]["id"] == "sample_001"
            assert transformed_data[1]["id"] == "sample_002"
        finally:
            Path(temp_path).unlink()

    def test_id_counter_persistence_across_records(self):
        """Test that ID counter persists and increments across multiple records."""
        data = [
            {"prompt": "Q1"},
            {"id": "custom-001", "prompt": "Q2"},  # Has ID, shouldn't affect counter
            {"prompt": "Q3"},  # Should continue counter
            {"prompt": "Q4"},
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
            loader.execute()
            loader.execute()

            # Check sequential counter behavior
            transformed_data = list(loader.dataset())
            assert transformed_data[0]["id"] == "sample_001"
            assert transformed_data[1]["id"] == "custom-001"  # User-provided ID
            assert transformed_data[2]["id"] == "sample_002"  # Counter continues
            assert transformed_data[3]["id"] == "sample_003"
        finally:
            Path(temp_path).unlink()

    # ========== TYPE CONVERSION TESTS ==========

    def test_answer_type_conversion_integer(self):
        """Test that integer answer values are converted to strings."""
        data = [
            {"id": "s1", "prompt": "What is 2+2?", "answer": 4},  # Integer answer
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
            loader.execute()
            loader.execute()

            # Check that answer was converted to string
            transformed_data = list(loader.dataset())
            assert transformed_data[0]["metadata"]["answer"] == "4"
            assert isinstance(transformed_data[0]["metadata"]["answer"], str)
        finally:
            Path(temp_path).unlink()

    def test_answer_type_conversion_float(self):
        """Test that float answer values are converted to strings."""
        data = [
            {"id": "s1", "prompt": "What is pi?", "answer": 3.14159},  # Float answer
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
            loader.execute()
            loader.execute()

            # Check that answer was converted to string
            transformed_data = list(loader.dataset())
            assert transformed_data[0]["metadata"]["answer"] == "3.14159"
            assert isinstance(transformed_data[0]["metadata"]["answer"], str)
        finally:
            Path(temp_path).unlink()

    def test_task_type_conversion_integer(self):
        """Test that integer task values are converted to strings."""
        data = [
            {"id": "s1", "prompt": "Question", "task": 123},  # Integer task
        ]

        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            for record in data:
                f.write(json.dumps(record) + "\n")
            temp_path = f.name

        try:
            loader = JSONLDatasetLoader(id="id", prompt="prompt", task="task")
            loader.load(temp_path)
            loader.transform(
                method=TrainingMethod.RFT_MULTITURN_LORA,
                model=Model.NOVA_LITE_2,
                platform=Platform.SMHP,
            )
            loader.execute()
            loader.execute()

            # Check that task was converted to string
            transformed_data = list(loader.dataset())
            assert transformed_data[0]["metadata"]["task"] == "123"
            assert isinstance(transformed_data[0]["metadata"]["task"], str)
        finally:
            Path(temp_path).unlink()

    def test_answer_and_task_type_conversion_combined(self):
        """Test that both answer and task are converted when non-string."""
        data = [
            {
                "id": "s1",
                "prompt": "Question",
                "answer": 42,  # Integer
                "task": 99,  # Integer
            },
        ]

        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            for record in data:
                f.write(json.dumps(record) + "\n")
            temp_path = f.name

        try:
            loader = JSONLDatasetLoader(id="id", prompt="prompt", answer="answer", task="task")
            loader.load(temp_path)
            loader.transform(
                method=TrainingMethod.RFT_MULTITURN_LORA,
                model=Model.NOVA_LITE_2,
                platform=Platform.SMHP,
            )
            loader.execute()
            loader.execute()

            # Check that both were converted to strings
            transformed_data = list(loader.dataset())
            assert transformed_data[0]["metadata"]["answer"] == "42"
            assert transformed_data[0]["metadata"]["task"] == "99"
            assert isinstance(transformed_data[0]["metadata"]["answer"], str)
            assert isinstance(transformed_data[0]["metadata"]["task"], str)
        finally:
            Path(temp_path).unlink()

    def test_answer_empty_string_preserved(self):
        """Test that empty string answer is preserved (not converted)."""
        data = [
            {"id": "s1", "prompt": "Question", "answer": ""},  # Empty string
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
            loader.execute()
            loader.execute()

            # Empty string should be filtered out (not included in metadata)
            transformed_data = list(loader.dataset())
            assert "answer" not in transformed_data[0]["metadata"]
        finally:
            Path(temp_path).unlink()

    def test_answer_with_special_characters(self):
        """Test that answer with special characters is handled correctly."""
        data = [
            {
                "id": "s1",
                "prompt": "Question",
                "answer": "Answer with\nnewlines\tand\ttabs",
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
            loader.execute()
            loader.execute()
            loader.validate(method=TrainingMethod.RFT_MULTITURN_LORA, model=Model.NOVA_LITE_2)

            # Should pass validation with special characters
            transformed_data = list(loader.dataset())
            assert transformed_data[0]["metadata"]["answer"] == "Answer with\nnewlines\tand\ttabs"
        finally:
            Path(temp_path).unlink()
