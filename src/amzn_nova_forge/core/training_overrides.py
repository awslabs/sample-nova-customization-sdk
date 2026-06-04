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
"""TypedDict for training recipe overrides."""

from typing_extensions import TypedDict


class TrainingOverrides(TypedDict, total=False):
    """Typed hints for training recipe override keys.

    Provides IDE autocomplete and static type checking for common override
    parameters passed to ``ForgeTrainer.train()``, ``ForgeEvaluator.evaluate()``,
    and ``RecipeBuilder.build_and_validate()``. All keys are optional
    (``total=False``).

    Plain dicts remain fully supported — this is structural typing, so
    ``{"lr": 0.001}`` satisfies the annotation without constructing a
    ``TrainingOverrides`` instance. Unknown keys from new recipe templates
    pass through at runtime without restriction.

    Example::

        overrides: TrainingOverrides = {
            "lr": 5e-6,
            "max_steps": 1000,
            "global_batch_size": 128,
        }
    """

    lr: float
    loraplus_lr_ratio: float
    lora_plus_lr_ratio: float

    max_steps: int
    max_epochs: int
    save_steps: int
    warmup_steps: int

    global_batch_size: int
    max_length: int

    alpha: float

    beta: float

    reasoning_enabled: bool
    reasoning_effort: str | None

    val_check_interval: int
    validation_s3_path: str
    validation_data_s3_path: str

    top_logprobs: int

    # MTRL-specific overrides
    advantage_method: str
    lora_rank: int
    loss_fn: str
    max_tokens: int
    group_size: int
    rollout_max_concurrency: int
    rollout_timeout: int
    sampling_temperature: float
    top_p: float
    save_every: int
    eval_every: int


NULLABLE_OVERRIDE_FIELDS: frozenset[str] = frozenset({"reasoning_effort"})
