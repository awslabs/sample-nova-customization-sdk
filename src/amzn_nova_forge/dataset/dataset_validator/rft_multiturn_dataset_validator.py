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
RFT Multiturn dataset validator for SDK format.

This module implements validation for RFT Multiturn datasets in the SDK format,
ensuring they meet all requirements before being sent to the training environment.
"""

from typing import Any, Dict, Iterator, List, Optional, Union

from pydantic import BaseModel, ConfigDict, field_validator, model_validator

from amzn_nova_forge.core.enums import Model

from .dataset_validator import BaseDatasetValidator

OPTIONAL_FIELDS = ["metadata.answer", "metadata.task", "metadata.info"]


class ToolCall(BaseModel):
    """Represents a tool call in OpenAI format."""

    id: str
    type: str
    function: Dict[str, Any]

    @field_validator("type")
    @classmethod
    def validate_type(cls, v: str) -> str:
        """Validate type is 'function'."""
        if v != "function":
            raise ValueError(f"Tool call type must be 'function', got '{v}'")
        return v

    @field_validator("function")
    @classmethod
    def validate_function(cls, v: Dict[str, Any]) -> Dict[str, Any]:
        """Validate function has required fields."""
        if "name" not in v:
            raise ValueError("Tool call function must have 'name' field")
        if "arguments" not in v:
            raise ValueError("Tool call function must have 'arguments' field")
        return v


class OpenAIMessage(BaseModel):
    """
    OpenAI message format for prompt field.

    Supports standard messages and tool calling:
    - Standard: role + content
    - Assistant with tool calls: role + tool_calls (content optional)
    - Tool response: role + tool_call_id + content
    """

    role: str
    content: Optional[str] = None
    tool_calls: Optional[List[ToolCall]] = None
    tool_call_id: Optional[str] = None

    @field_validator("role")
    @classmethod
    def validate_role(cls, v: str) -> str:
        """Validate role is one of the allowed values."""
        allowed_roles = {"system", "user", "assistant", "tool"}
        if v not in allowed_roles:
            raise ValueError(f"Invalid role '{v}'. Must be one of: {', '.join(allowed_roles)}")
        return v

    @model_validator(mode="after")
    def validate_message_structure(self) -> "OpenAIMessage":
        """Validate message has appropriate fields for its role."""
        role = self.role

        # Assistant messages can have content and/or tool_calls
        if role == "assistant":
            if self.content is None and self.tool_calls is None:
                raise ValueError("Assistant message must have either 'content' or 'tool_calls'")
            if self.tool_calls is not None and len(self.tool_calls) == 0:
                raise ValueError("Assistant 'tool_calls' cannot be empty list")

        # Tool messages must have tool_call_id and content
        if role == "tool":
            if self.tool_call_id is None:
                raise ValueError("Tool message must have 'tool_call_id' field")
            if self.content is None or not self.content.strip():
                raise ValueError("Tool message must have non-empty 'content'")

        # System and user messages must have content
        elif role in ("system", "user"):
            if self.content is None or not self.content.strip():
                raise ValueError(f"{role.capitalize()} message must have non-empty 'content'")

        return self


class RFTMultiturnMetadata(BaseModel):
    """Metadata for RFT Multiturn sample."""

    model_config = ConfigDict(extra="forbid")

    prompt: Union[str, List[Dict[str, Any]]]
    answer: Optional[str] = None
    task: Optional[str] = None
    info: Optional[Union[Dict[str, Any], str]] = None

    @field_validator("prompt")
    @classmethod
    def validate_prompt(cls, v: Any) -> Union[str, List[Dict[str, Any]]]:
        """Validate prompt is either string or list of message dicts."""
        if isinstance(v, str):
            if not v.strip():
                raise ValueError("Prompt string cannot be empty")
            return v

        if isinstance(v, list):
            if not v:
                raise ValueError("Prompt messages list cannot be empty")

            # Validate each message has required fields
            for i, msg in enumerate(v):
                if not isinstance(msg, dict):
                    raise ValueError(
                        f"Message at index {i} must be a dict, got {type(msg).__name__}"
                    )
                if "role" not in msg:
                    raise ValueError(f"Message at index {i} missing 'role' field")

                # Validate using OpenAIMessage model
                OpenAIMessage(**msg)

            return v

        raise ValueError(
            f"Prompt must be a string or list of message dicts, got {type(v).__name__}"
        )

    @field_validator("answer")
    @classmethod
    def validate_answer(cls, v: Optional[str]) -> Optional[str]:
        """Validate answer is string if provided."""
        if v is not None and not isinstance(v, str):
            raise ValueError(f"Answer must be a string, got {type(v).__name__}")
        return v

    @field_validator("task")
    @classmethod
    def validate_task(cls, v: Optional[str]) -> Optional[str]:
        """Validate task is string if provided."""
        if v is not None and not isinstance(v, str):
            raise ValueError(f"Task must be a string, got {type(v).__name__}")
        return v

    @field_validator("info")
    @classmethod
    def validate_info(cls, v: Any) -> Optional[Union[Dict[str, Any], str]]:
        """Validate info is either a dict or a valid JSON string."""
        if v is None:
            return None

        if isinstance(v, dict):
            return v

        if isinstance(v, str):
            # Validate that the string is valid JSON
            import json

            try:
                json.loads(v)
                return v
            except json.JSONDecodeError as e:
                raise ValueError(
                    f"'info' string must be valid JSON. Got: {v!r}. JSON parse error: {str(e)}"
                )

        raise ValueError(
            f"'info' must be either a dictionary or a valid JSON string, got {type(v).__name__}"
        )


class RFTMultiturnDatasetSample(BaseModel):
    """
    Represents an RFT Multiturn dataset sample in SDK format.

    Expected format:
    {
        "id": str,                       # REQUIRED - non-empty string
        "metadata": {                    # REQUIRED
            "prompt": str | List[Dict],  # REQUIRED - str OR OpenAI messages
            "answer": str,               # OPTIONAL
            "task": str,                 # OPTIONAL
            "info": dict | str           # OPTIONAL - dict or valid JSON string
        }
    }
    """

    model_config = ConfigDict(extra="forbid")

    id: str
    metadata: RFTMultiturnMetadata

    @field_validator("id")
    @classmethod
    def validate_id_not_empty(cls, v: str) -> str:
        """Ensure ID is not empty."""
        if not v or not v.strip():
            raise ValueError("'id' cannot be empty")
        return v.strip()


class RFTMultiturnServerlessSample(BaseModel):
    """
    Represents an RFT Multiturn dataset sample for Serverless MTRL.

    Expected format:
    {
        "prompt": str  # REQUIRED - the prompt string passed as-is to the agent
    }
    """

    model_config = ConfigDict(extra="allow")

    prompt: str

    @field_validator("prompt")
    @classmethod
    def validate_prompt_not_empty(cls, v: str) -> str:
        """Ensure prompt is not empty."""
        if not v or not v.strip():
            raise ValueError("'prompt' cannot be empty")
        return v


class RFTMultiturnDatasetValidator(BaseDatasetValidator):
    """
    Validator for RFT Multiturn datasets in SDK format (HyperPod).

    RFT Multiturn is only supported on Nova 2.0 and requires:
    - Unique sample IDs
    - Prompt (string or OpenAI messages format)
    - Optional answer, task, and info fields
    - Info field must be dict or valid JSON string if provided
    """

    def __init__(self, model: Model):
        """
        Initialize the RFT Multiturn dataset validator.

        Args:
            model: The Nova model being used (must be NOVA_LITE_2 for RFT Multiturn)

        Raises:
            ValueError: If the model isn't NOVA_LITE_2.
        """
        super().__init__()
        if model != Model.NOVA_LITE_2:
            raise ValueError(
                f"RFT Multiturn is only supported on Nova 2.0 Lite (NOVA_LITE_2). "
                f"Current model: {model}. Please use Model.NOVA_LITE_2 for validating and using RFT Multiturn datasets."
            )

    def get_sample_model(self):
        """
        Returns:
            RFTMultiturnDatasetSample: The Pydantic model for RFT Multiturn dataset validation
        """
        return RFTMultiturnDatasetSample

    def get_success_message(self) -> str:
        """
        Returns:
            str: Success message with sample count
        """
        return f"Validation succeeded for {self.num_samples} samples on an RFT Multiturn dataset."

    def get_optional_fields(self) -> List[str]:
        """
        Returns:
            OPTIONAL_FIELDS: A list of all the optional fields for RFT Multiturn.
        """
        return OPTIONAL_FIELDS


class RFTMultiturnServerlessValidator(BaseDatasetValidator):
    """
    Validator for RFT Multiturn datasets in Serverless format.

    Validates flat format: {"prompt": str}
    The service reads only the prompt column and passes it as-is to the agent.
    """

    def __init__(self, model: Model):
        super().__init__()
        if model != Model.NOVA_LITE_2:
            raise ValueError(
                f"RFT Multiturn is only supported on Nova 2.0 Lite (NOVA_LITE_2). "
                f"Current model: {model}."
            )

    def get_sample_model(self):
        return RFTMultiturnServerlessSample

    def get_success_message(self) -> str:
        return f"Validation succeeded for {self.num_samples} samples on a Serverless RFT Multiturn dataset."

    def get_optional_fields(self) -> List[str]:
        return []
