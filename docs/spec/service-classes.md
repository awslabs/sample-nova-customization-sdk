# Service Classes

The modular service classes are the recommended API for Nova model customization. Each class handles a single concern — training, evaluation, deployment, or inference — and can be used independently.

All service classes accept an optional `ForgeConfig` dataclass for shared configuration (KMS keys, output paths, caching, etc.).

### ForgeConfig

Shared configuration dataclass for all service classes.

**Signature:**
```python
@dataclass
class ForgeConfig:
    kms_key_id: Optional[str] = None
    output_s3_path: Optional[str] = None
    generated_recipe_dir: Optional[str] = None
    validation_config: Optional[ValidationConfig] = None
    image_uri: Optional[str] = None
    mlflow_monitor: Optional[MLflowMonitor] = None
    enable_job_caching: bool = False
    job_cache_dir: str = "~/.nova-forge/cache"
    job_caching_config: Optional[JobCachingConfig] = None
```

**Parameters:**
- `kms_key_id` (Optional[str]): KMS key ID for S3 encryption
- `output_s3_path` (Optional[str]): S3 path for output artifacts. Auto-generated if not provided
- `generated_recipe_dir` (Optional[str]): Local path to save generated recipe files
- `validation_config` (Optional[ValidationConfig]): Controls pre-flight validation. Fields: `iam` (bool), `infra` (bool), `recipe` (bool) — all default to True
- `image_uri` (Optional[str]): Custom container image URI override. For InspectLens, overrides the default orchestrator container image.
- `mlflow_monitor` (Optional[MLflowMonitor]): MLflow monitoring configuration (SageMaker only)
- `enable_job_caching` (bool): Enable caching of completed job results for reuse. Default: False
- `job_cache_dir` (str): Directory for cached job results. Default: `~/.nova-forge/cache`
- `job_caching_config` (Optional[JobCachingConfig]): Advanced caching configuration. Fields: `include_core` (bool), `include_recipe` (bool), `include_infra` (bool), `include_params` (List[str]), `exclude_params` (List[str]), `allowed_statuses` (List[JobStatus])

**Example:**
```python
from amzn_nova_forge.core import ForgeConfig, ValidationConfig
from amzn_nova_forge.monitor import MLflowMonitor

config = ForgeConfig(
    output_s3_path="s3://my-bucket/output",
    kms_key_id="my-kms-key-id",
    validation_config=ValidationConfig(iam=True, infra=True, recipe=True),
    mlflow_monitor=MLflowMonitor(
        tracking_uri="arn:aws:sagemaker:us-east-1:123456789012:mlflow-app/app-xxx",
        experiment_name="nova-customization"
    ),
    enable_job_caching=True
)
```
---

### ForgeTrainer

Handles training job configuration and execution for Nova models.

#### Constructor

**Signature:**
```python
def __init__(
    self,
    model: Model,
    method: TrainingMethod,
    infra: RuntimeManager,
    training_data_s3_path: Optional[str] = None,
    model_s3_path: Optional[str] = None,
    model_arn: Optional[str] = None,
    data_mixing_enabled: bool = False,
    holdout_data_s3_path: Optional[str] = None,
    val_check_interval: Optional[int] = None,
    config: Optional[ForgeConfig] = None,
    region: Optional[str] = None,
    is_multimodal: Optional[bool] = None,
    hub_content_version: Optional[str] = None,
    enable_batch_sample_tracing: bool = False,
)
```

**Parameters:**
- `model` (Model): The Nova model to train (e.g., `Model.NOVA_MICRO`, `Model.NOVA_LITE_2`)
- `method` (TrainingMethod): The fine-tuning method (e.g., `TrainingMethod.SFT_LORA`, `TrainingMethod.RFT`)
- `infra` (RuntimeManager): Runtime infrastructure manager (e.g., `SMTJRuntimeManager`, `SMHPRuntimeManager`, `BedrockRuntimeManager`)
- `training_data_s3_path` (Optional[str]): S3 path to the training dataset
- `model_s3_path` (Optional[str]): S3 path for the base or previously trained model (SMHP/SMTJ). For SMTJServerless, use `model_arn` instead.
- `model_arn` (Optional[str]): Model package ARN for iterative training on SMTJServerless. Pass the `output_model_arn` from a previous job to train on top of it. Cannot be combined with `model_s3_path`.
- `data_mixing_enabled` (bool): Enable data mixing for CPT and SFT training on SMHP, and SFT text-only on Nova Lite 2 on SMTJServerless. Default: False
- `holdout_data_s3_path` (Optional[str]): S3 path to holdout/validation data (optional, used for CPT and SFT on SMTJ/SMTJServerless/SMHP, or any method on Bedrock)
- `val_check_interval` (Optional[int]): How often (in training steps) to run validation. Defaults to 2500 if omitted. Only used when `holdout_data_s3_path` is provided.
- `config` (Optional[ForgeConfig]): Shared configuration. If not provided, defaults are used
- `region` (Optional[str]): AWS region. Auto-detected if not provided
- `is_multimodal` (Optional[bool]): Explicitly set multimodal mode when `data_mixing_enabled=True`. If None, auto-detects from data
- `hub_content_version` (Optional[str]): Version of the hub content to retrieve from SageMaker Hub. If None, uses the latest version
- `enable_batch_sample_tracing` (bool): Activate per-step batch hashing during training, which enables `trace_batch()` post-training. Supported platform/method combinations are validated at construction time. Default: False

**Raises:**
- `ValueError`: If `enable_batch_sample_tracing=True` is used with an unsupported platform or training method

**Example:**
```python
from amzn_nova_forge.trainer import ForgeTrainer
from amzn_nova_forge.core import ForgeConfig
from amzn_nova_forge.manager import SMTJRuntimeManager
from amzn_nova_forge.model.model_enums import Model, TrainingMethod

infra = SMTJRuntimeManager(instance_type="ml.p5.48xlarge", instance_count=2)

trainer = ForgeTrainer(
    model=Model.NOVA_MICRO,
    method=TrainingMethod.SFT_LORA,
    infra=infra,
    training_data_s3_path="s3://my-bucket/training-data/data.jsonl",
    config=ForgeConfig(output_s3_path="s3://my-bucket/output")
)
```
---

#### Methods

##### `train()`
Generates the recipe YAML, configures the runtime, and launches a training job.

**Signature:**
```python
def train(
    self,
    job_name: str,
    recipe_path: Optional[str] = None,
    overrides: Optional[Dict[str, Any]] = None,
    rft_lambda_arn: Optional[str] = None,
    dry_run: bool = False,
    rft_multiturn_infra=None,
) -> Optional[TrainingResult]
```

**Parameters:**
- `job_name` (str): User-defined name for the training job
- `recipe_path` (Optional[str]): Path for a YAML recipe file (S3 or local)
- `overrides` (Optional[Dict[str, Any]]): Dictionary of configuration overrides (e.g., `max_epochs`, `lr`, `warmup_steps`, `global_batch_size`)
- `rft_lambda_arn` (Optional[str]): Rewards Lambda ARN (only for RFT methods). Takes priority over `rft_lambda_arn` on the RuntimeManager
- `dry_run` (bool): If True, performs validation only without starting a job. Default: False
- `rft_multiturn_infra`: Optional RFTMultiturnInfrastructure for RFT multiturn training

**Returns:**
- `TrainingResult`: Metadata object containing `job_id`, `method`, `started_time`, `model_artifacts`, and `model_type`. Returns `None` if `dry_run=True`

**Raises:**
- `Exception`: If job execution fails
- `ValueError`: If training method is not supported

**Example:**
```python
result = trainer.train(
    job_name="my-training-job",
    overrides={
        "max_epochs": 10,
        "lr": 5e-6,
        "warmup_steps": 20,
        "global_batch_size": 128
    }
)
print(f"Training job started: {result.job_id}")
print(f"Checkpoint path: {result.model_artifacts.checkpoint_s3_path}")
```
---

##### `get_config()`
Returns the overridable training parameters for the current model/method/platform without starting a job or running validation. Use this to inspect defaults, valid ranges, and available parameters before calling `train()`.

**Signature:**
```python
def get_config(
    self,
    overrides: Optional[TrainingOverrides] = None,
) -> RecipeConfig
```

**Parameters:**
- `overrides` (Optional[TrainingOverrides]): Optional overrides to merge with defaults. When provided, `ConfigParameter.default` values in the result reflect the merged values. Plain `dict` values are also accepted

**Returns:**
- `RecipeConfig`: Frozen dataclass with:
  - `model` (Model): The model
  - `method` (TrainingMethod): The training method
  - `platform` (Platform): The infrastructure platform
  - `parameters` (tuple[ConfigParameter, ...]): All overridable parameters, each with `name`, `type`, `default`, `description`, `min`, `max`, `enum`, `required`
  - `to_dict()`: Returns `{name: default}` mapping directly passable to `train(overrides=...)`

**Example:**
```python
# Inspect default configuration
config = trainer.get_config()
print(config)
# RecipeConfig(model=NOVA_MICRO, method=SFT_LORA, platform=SMTJ)
#   lr: float = 5e-06 [1e-06..0.0001]
#   max_epochs: integer = 2 [1..5]
#   warmup_steps: integer = 10 [0..100]
#   global_batch_size: integer = 64 [1..512]

# Preview with your overrides applied
config = trainer.get_config(overrides={"lr": 1e-5, "max_epochs": 4})
print(config.to_dict())
# {"lr": 1e-5, "max_epochs": 4, "warmup_steps": 10, "global_batch_size": 64, ...}
```

**Note:** `get_config()` is not supported for `RFT_MULTITURN_*` training methods. For those methods, use `train(dry_run=True)` to inspect the recipe.

---

##### `get_logs()`
Retrieves and displays CloudWatch logs for a training job.

**Signature:**
```python
def get_logs(
    self,
    job_result=None,
    job_id=None,
    started_time=None,
    limit=None,
    start_from_head: bool = False,
    end_time=None,
) -> None
```

**Parameters:**
- `job_result` (Optional[TrainingResult]): Job result to retrieve logs for. If not provided, uses `job_id`
- `job_id` (Optional[str]): Job identifier. Used if `job_result` is not provided
- `started_time` (Optional[datetime]): Job start time to filter logs
- `limit` (Optional[int]): Maximum number of log lines to retrieve
- `start_from_head` (bool): If True, start from the beginning of logs. Default: False
- `end_time` (Optional[int]): End time in epoch milliseconds for searching a log time range

**Returns:**
- None (prints logs to console)

**Example:**
```python
trainer.get_logs(job_result=result, limit=100, start_from_head=True)
```
---

##### `trace_batch()`
Extracts the lines from your input training data that were used in a specific training step's batch. Useful for diagnosing gradient spikes or training anomalies — given a step number, it matches the container's batch hash logs against your source data and writes the matched lines to an output file.

The training job must have been launched with `enable_batch_sample_tracing=True` so that batch hash logs are written during training. Supported platform/method combinations are validated at `ForgeTrainer` construction time. If you create a new `ForgeTrainer` instance to trace a previously-launched job, the flag is not strictly required on the new instance — but a warning will be emitted.

**Signature:**
```python
def trace_batch(
    self,
    training_result: TrainingResult,
    step: int,
    output_path: str | None = None,
    cache_dir: str = "~/.nova-forge/batch_trace_cache",
) -> Path | None
```

**Parameters:**
- `training_result` (TrainingResult): Result from a completed training job. The method extracts `training_result.model_artifacts.output_s3_path` and `training_result.job_id` to locate the batch hash logs at `{output_s3_path}/{job_id}/batch_tracing/`.
- `step` (int): Training step number to investigate (must match the step numbers in the container's batch hash logs).
- `output_path` (Optional[str]): Path for the output file containing matched lines. Default: `step_<N>_samples.jsonl` in the current working directory.
- `cache_dir` (str): Directory for caching downloaded files and fingerprint indices. Default: `~/.nova-forge/batch_trace_cache`. The cache stores downloaded S3 files (training data, hash logs) and a CSV fingerprint index. For large datasets the cache may grow to match the source data size. The cache is keyed by S3 path — if you replace the file at an existing S3 URI, delete the cache directory to avoid stale matches.

**Returns:**
- `Path | None`: Path to the output JSONL file containing the matched lines (verbatim copies from your source data, sorted by line number). Returns `None` if either the step had no logged batch data (step out of range or job still running) or the step's batch contained no samples from your file.

**Raises:**
- `ValueError`: If `training_data_s3_path` or `training_result.model_artifacts.output_s3_path` is not available
- `BatchTraceError`: If batch tracing encounters an unrecoverable error (e.g., missing log files, AWS auth failure)
  Import: `from amzn_nova_forge.trainer.utils import BatchTraceError`

**Example:**
```python
trainer = ForgeTrainer(
    model=Model.NOVA_LITE_2,
    method=TrainingMethod.CPT,
    infra=infra,
    training_data_s3_path="s3://my-bucket/data.jsonl",
    enable_batch_sample_tracing=True,
)

result = trainer.train(job_name="my-cpt-job")

# After job completes, investigate step 42.
# Output writes to ./step_42_samples.jsonl by default.
matched_file = trainer.trace_batch(result, step=42)
if matched_file:
    print(f"Matched lines written to: {matched_file}")

# Explicit output path:
matched_file = trainer.trace_batch(result, step=42, output_path="/tmp/flagged.jsonl")
```
---

##### `generate_training_metrics_csv()`

Generates a `step_wise_training_metrics.csv` file from CloudWatch logs for a completed SMHP SFT training job and uploads it to S3. The CSV contains step-level metrics (step number, epoch number, training loss) in the same format produced by SMTJ/Bedrock SFT jobs.

**Signature:**
```python
def generate_training_metrics_csv(
    self,
    job_result: Optional[SMHPTrainingResult] = None,
    job_id: Optional[str] = None,
    started_time=None,
    end_time=None,
    output_s3_path: Optional[str] = None,
) -> Optional[str]
```

**Parameters:**
- `job_result` (Optional[SMHPTrainingResult]): Result from a completed SMHP training job. If provided, `job_id`, `started_time`, and `output_s3_path` are extracted automatically.
- `job_id` (Optional[str]): The SMHP training job ID. Used if `job_result` is not provided.
- `started_time`: Start time for log filtering. Accepts a `datetime` object or an ISO date string (e.g., `"2025-05-26"`). Defaults to 7 days ago if not provided.
- `end_time`: Optional end time to bound the log search. Accepts a `datetime` object or an ISO date string. If not provided, searches up to the current time. Providing this significantly speeds up log retrieval for older jobs.
- `output_s3_path` (Optional[str]): S3 URI for the output. Defaults to the trainer's configured `output_s3_path`.

**Returns:**
- `str | None`: S3 URI of the uploaded CSV (e.g., `s3://bucket/prefix/job-id/step_wise_training_metrics.csv`), or `None` if no metrics could be extracted.

**Raises:**
- `ValueError`: If platform is not SMHP, method is not SFT_LORA/SFT_FULL, or required parameters are missing.

**Example:**
```python
# From a job result (simplest)
result = trainer.train(job_name="my-sft-job")
csv_uri = trainer.generate_training_metrics_csv(job_result=result)

# Standalone with job ID
csv_uri = trainer.generate_training_metrics_csv(
    job_id="my-job-id",
    started_time="2026-04-17",
    end_time="2026-04-18",
)

# Minimal — uses trainer's output_s3_path and defaults to 7-day lookback
csv_uri = trainer.generate_training_metrics_csv(job_id="my-job-id")
```

**Notes:**
- Only supported for SMHP platform with SFT_LORA or SFT_FULL training methods.
- For older jobs, providing `started_time` and `end_time` significantly reduces log retrieval time.
- If the job is still in progress or has failed, a warning is emitted but metrics are still extracted on a best-effort basis.

---

### ForgeEvaluator

Handles evaluation job configuration and execution for Nova models.

#### Constructor

**Signature:**
```python
def __init__(
    self,
    model: Model,
    infra: RuntimeManager,
    data_s3_path: Optional[str] = None,
    config: Optional[ForgeConfig] = None,
    region: Optional[str] = None,
    hub_content_version: Optional[str] = None,
)
```

**Parameters:**
- `model` (Model): The Nova model to evaluate
- `infra` (RuntimeManager): Runtime infrastructure manager
- `data_s3_path` (Optional[str]): S3 path to evaluation data (required for BYOD evaluation tasks)
- `config` (Optional[ForgeConfig]): Shared configuration
- `region` (Optional[str]): AWS region. Auto-detected if not provided
- `hub_content_version` (Optional[str]): Version of the hub content to retrieve from SageMaker Hub. If None, uses the latest version

**Example:**
```python
from amzn_nova_forge.evaluator import ForgeEvaluator
from amzn_nova_forge.manager import SMTJRuntimeManager
from amzn_nova_forge.model.model_enums import Model

infra = SMTJRuntimeManager(instance_type="ml.p5.48xlarge", instance_count=2)

evaluator = ForgeEvaluator(
    model=Model.NOVA_MICRO,
    infra=infra,
    data_s3_path="s3://my-bucket/eval-data/data.jsonl"
)
```
---

#### Methods

##### `evaluate()`
Generates the recipe YAML, configures the runtime, and launches an evaluation job.

**Signature:**
```python
def evaluate(
    self,
    job_name: str,
    eval_task: EvaluationTask,
    model_path: Optional[str] = None,
    task_config: Optional[EvalTaskConfig] = None,
    recipe_path: Optional[str] = None,
    overrides: Optional[Dict[str, Any]] = None,
    dry_run: bool = False,
    job_result: Optional[TrainingResult] = None,
    rft_multiturn_infra=None,
) -> Optional[EvaluationResult]
```

**Parameters:**
- `job_name` (str): User-defined name for the evaluation job
- `eval_task` (EvaluationTask): The evaluation task (e.g., `EvaluationTask.MMLU`)
- `model_path` (Optional[str]): S3 path to the model to evaluate. If not provided, extracted from `job_result`
- `task_config` (Optional[EvalTaskConfig]): Task-specific configuration. Fields: `subtask`, `processor`, `rl_env`, `override_data_s3_path`, `evaluate_base_model` (MTRL only — when True, evaluates both base and fine-tuned model in one pipeline)
- `recipe_path` (Optional[str]): Path for a YAML recipe file (S3 or local)
- `overrides` (Optional[Dict[str, Any]]): Inference configuration overrides (e.g., `max_new_tokens`, `temperature`, `top_p`)
- `dry_run` (bool): If True, performs validation only. Default: False
- `job_result` (Optional[TrainingResult]): Training result to extract checkpoint path from
- `rft_multiturn_infra`: Optional RFTMultiturnInfrastructure for RFT evaluation

**Returns:**
- `EvaluationResult`: Metadata object containing `job_id`, `started_time`, `eval_output_path`, and `eval_task`. Returns `None` if `dry_run=True`

**Example:**
```python
from amzn_nova_forge.core import EvaluationTask

eval_result = evaluator.evaluate(
    job_name="my-eval-job",
    eval_task=EvaluationTask.MMLU,
    model_path="s3://my-bucket/checkpoints/my-model",
    overrides={
        "max_new_tokens": 2048,
        "temperature": 0,
        "top_p": 1.0
    }
)
print(f"Evaluation job started: {eval_result.job_id}")

# Chain from training result
eval_result = evaluator.evaluate(
    job_name="my-eval-job",
    eval_task=EvaluationTask.MMLU,
    job_result=training_result
)
```
---

##### `get_logs()`
Retrieves and displays CloudWatch logs for an evaluation job.

**Signature:**
```python
def get_logs(
    self,
    job_result=None,
    job_id=None,
    started_time=None,
    limit=None,
    start_from_head: bool = False,
    end_time=None,
) -> None
```

**Parameters:**
- `job_result` (Optional[EvaluationResult]): Job result to retrieve logs for
- `job_id` (Optional[str]): Job identifier
- `started_time` (Optional[datetime]): Job start time to filter logs
- `limit` (Optional[int]): Maximum number of log lines
- `start_from_head` (bool): If True, start from the beginning of logs. Default: False
- `end_time` (Optional[int]): End time in epoch milliseconds for searching a log time range

**Returns:**
- None (prints logs to console)

**Example:**
```python
evaluator.get_logs(job_result=eval_result, limit=50)
```
---

##### InspectLens Evaluation (`EvaluationTask.INSPECT_LENS`)

InspectLens runs [Inspect AI](https://inspect.ai-safety-institute.org.uk/) benchmarks as a SageMaker Training Job. The job acts as a CPU-only orchestrator — inference is delegated to a Bedrock endpoint or a SageMaker endpoint. No GPU is needed for the orchestrator instance.

Pass `eval_task=EvaluationTask.INSPECT_LENS` and provide an `InspectLensConfig` via the `inspect_lens_config` parameter.

**`InspectLensConfig`**

```python
@dataclass
class InspectLensConfig:
    benchmarks_path: Optional[str] = None
    tasks: List[Dict[str, Any]] = field(default_factory=list)
    output_s3_path: Optional[str] = None
    output_format: Optional[str] = None
    bedrock_model_id: Optional[str] = None
    endpoint_name: Optional[str] = None
    model_s3_uri: Optional[str] = None
    inference_image_uri: Optional[str] = None
    endpoint_instance_type: Optional[str] = None
    endpoint_instance_count: int = 1
    endpoint_execution_role_arn: Optional[str] = None
    context_length: Optional[str] = None
    max_concurrency: Optional[str] = None
    enable_rai: bool = True
    cleanup_endpoint: bool = True
    endpoint_prefix: str = "inspectlens"
    endpoint_environment: Optional[Dict[str, str]] = None
    extra_args: Optional[List[str]] = None
    environment: Optional[Dict[str, str]] = None
```

**Parameters:**
- `benchmarks_path` (str): S3 URI to benchmark `.py` files — required at runtime. Use `evaluator.upload_benchmarks(local_dir, s3_path)` to upload a local directory first. Validated: must be a non-empty `s3://` URI; task names are checked against `@task`-decorated functions at job submission time.
- `tasks` (List[Dict]): List of task dicts with a required `"name"` key and optional `"limit"` and `"epochs"`. Leave empty to run all `@task` functions in `benchmarks_path`. Validated: each entry must be a dict with a `"name"` key.
- `output_s3_path` (Optional[str]): S3 prefix for eval result JSON logs. Defaults to the `ForgeEvaluator` output path. Validated: must be an `s3://` URI if provided.
- `output_format` (Optional[str]): Output format for eval results. One of `"eval"`, `"csv"`, `"jsonl"`, `"json"`. Defaults to `"eval"`.
- `bedrock_model_id` (Optional[str]): Bedrock model ID or ARN. Falls back to the `ForgeEvaluator.model` cross-region inference profile.
- `endpoint_name` (Optional[str]): Existing SageMaker endpoint name. Mutually exclusive with `model_s3_uri`.
- `model_s3_uri` (Optional[str]): S3 URI of model artifacts for creating a new endpoint. Requires `inference_image_uri`. Validated: must be an `s3://` URI if provided.
- `inference_image_uri` (Optional[str]): ECR image URI for the new endpoint container. Requires `model_s3_uri`. Validated: must match ECR URI format if provided.
- `endpoint_instance_type` (Optional[str]): Instance type for the new endpoint.
- `endpoint_instance_count` (int): Number of instances for the new endpoint. Default: 1.
- `endpoint_execution_role_arn` (Optional[str]): IAM role ARN for the new endpoint. Validated: must match IAM ARN format if provided.
- `context_length` (Optional[str]): Context length for the new endpoint.
- `max_concurrency` (Optional[str]): Max concurrency for the new endpoint.
- `enable_rai` (bool): Enable RAI guardrails on the endpoint. Default: True.
- `cleanup_endpoint` (bool): Delete the endpoint after evaluation. Default: True.
- `endpoint_prefix` (str): Prefix for auto-created endpoint names. Default: `"inspectlens"`.
- `endpoint_environment` (Optional[Dict[str, str]]): Extra environment variables for the inference endpoint container. Example: `{"HF_TOKEN": "hf_xxx"}`.
- `extra_args` (Optional[List[str]]): Additional CLI args forwarded to `inspect eval`.
- `environment` (Optional[Dict[str, str]]): Arbitrary environment variables passed to the SageMaker Training Job container. Example: `{"HF_TOKEN": "hf_xxx"}`.

**Inference provider modes:**

| Mode                | When used                                                                                                 |
|---------------------|-----------------------------------------------------------------------------------------------------------|
| Bedrock             | No `endpoint_name` or `model_s3_uri` set. Uses `bedrock_model_id` or the evaluator's model enum.         |
| Existing endpoint   | `endpoint_name` is set.                                                                                   |
| Create new endpoint | `model_s3_uri` + `inference_image_uri` are set. Endpoint is created, evaluated, then optionally deleted. |

**Decoding overrides:** (`temperature`, `top_p`, `max_tokens`, `max_connections`, `max_retries`, `timeout`) are passed via the `overrides` parameter on `evaluate()`.

**MLflow tracking:** Pass an `MLflowMonitor` via `ForgeConfig.mlflow_monitor`. The SDK automatically injects the tracking section into the InspectLens YAML config.

**Example:**

```python
from amzn_nova_forge.evaluator import ForgeEvaluator, InspectLensConfig
from amzn_nova_forge.core.enums import EvaluationTask, Model
from amzn_nova_forge.core.types import ForgeConfig
from amzn_nova_forge.manager import SMTJRuntimeManager

infra = SMTJRuntimeManager(
    instance_type="ml.m5.large",   # CPU only — no GPU needed
    instance_count=1,
    execution_role="arn:aws:iam::123456789012:role/InspectLensEvalRole",
)
evaluator = ForgeEvaluator(
    model=Model.NOVA_LITE_2,
    infra=infra,
    config=ForgeConfig(output_s3_path="s3://my-bucket/inspectlens/"),
)

# Upload local benchmarks first
benchmarks_s3_uri = evaluator.upload_benchmarks(
    "./my_benchmarks/",
    "s3://my-bucket/inspectlens/benchmarks/my_benchmarks/",
)

# Run evaluation
eval_result = evaluator.evaluate(
    job_name="inspectlens-eval",
    eval_task=EvaluationTask.INSPECT_LENS,
    inspect_lens_config=InspectLensConfig(
        benchmarks_path=benchmarks_s3_uri,
        tasks=[{"name": "boolq_pt", "limit": 100}],
    ),
    overrides={"temperature": 0.0, "max_tokens": 4096},
)
```

See `samples/inspectlens_quickstart.ipynb` for a full walkthrough including all three inference modes (Bedrock, existing endpoint, new endpoint).

For a conceptual guide covering prerequisites, IAM setup, benchmark file format, inference modes, MLflow tracking, and job caching, see [docs/user-guides/inspect_eval.md](../user-guides/inspect_eval.md).

---

##### `upload_benchmarks()`
Uploads a local benchmarks directory to S3 before starting an InspectLens job.

**Signature:**
```python
def upload_benchmarks(self, local_dir: str, s3_path: str) -> str
```

**Parameters:**
- `local_dir` (str): Path to the local directory containing benchmark `.py` files with `@task` decorators.
- `s3_path` (str): S3 URI (`s3://bucket/prefix/`) where the benchmark files will be uploaded.

**Returns:**
- The S3 URI where benchmarks were uploaded (normalized with a trailing slash).

---

### ForgeDeployer

Handles model deployment to Amazon Bedrock and SageMaker endpoints.

#### Constructor

**Signature:**
```python
def __init__(
    self,
    region: str,
    model: Model,
    deployment_mode: DeploymentMode = DeploymentMode.FAIL_IF_EXISTS,
    config: Optional[ForgeConfig] = None,
    method: Optional[TrainingMethod] = None,
)
```

**Parameters:**
- `region` (str): AWS region for deployment
- `model` (Model): The Nova model being deployed
- `deployment_mode` (DeploymentMode): Behavior when endpoint already exists. Default: `FAIL_IF_EXISTS`
- `config` (Optional[ForgeConfig]): Shared configuration
- `method` (Optional[TrainingMethod]): Training method used (needed for SageMaker deployment image selection)

**Example:**
```python
from amzn_nova_forge.deployer import ForgeDeployer
from amzn_nova_forge.model.model_enums import Model, DeploymentMode

deployer = ForgeDeployer(
    region="us-east-1",
    model=Model.NOVA_MICRO,
    deployment_mode=DeploymentMode.FAIL_IF_EXISTS
)
```
---

#### Methods

##### `deploy()`
Creates a custom model and deploys it to Amazon Bedrock or SageMaker in a single step.

**Signature:**
```python
def deploy(
    self,
    model_artifact_path: str,
    deploy_platform: DeployPlatform = DeployPlatform.BEDROCK_OD,
    endpoint_name: Optional[str] = None,
    unit_count: int = 1,
    execution_role_name: Optional[str] = None,
    sagemaker_instance_type: str = "ml.p5.48xlarge",
    sagemaker_environment: Optional[SageMakerEndpointEnvironment] = None,
    skip_model_reuse: bool = False,
    inference_component_configs: List[InferenceComponentConfig] = [],
) -> DeploymentResult
```

**Parameters:**
- `model_artifact_path` (str): S3 path to the trained model checkpoint, or a SageMaker Model Package name/ARN. Model package ARNs are auto-detected for SageMaker deployments and passed via `ModelPackageName`
- `deploy_platform` (DeployPlatform): Platform to deploy to (`BEDROCK_OD`, `BEDROCK_PT`, or `SAGEMAKER`). Default: `BEDROCK_OD`
- `endpoint_name` (Optional[str]): Name of the endpoint (auto-generated if not provided)
- `unit_count` (int): Number of PT units (Bedrock PT) or instances (SageMaker). Default: 1
- `execution_role_name` (Optional[str]): IAM role name. If omitted, the SDK creates a default role
- `sagemaker_instance_type` (str): Instance type for SageMaker deployment. Default: `"ml.p5.48xlarge"`
- `sagemaker_environment` (Optional[SageMakerEndpointEnvironment]): SageMaker endpoint environment config. Fields:
  - `CONTEXT_LENGTH` (int, default: 4000), `MAX_CONCURRENCY` (int, default: 1)
  - Optional generation defaults: `DEFAULT_TEMPERATURE` (0–2), `DEFAULT_TOP_P` (1e-10–1), `DEFAULT_TOP_K` (-1 to disable, or ≥1), `DEFAULT_MAX_NEW_TOKENS` (≥1), `DEFAULT_LOGPROBS` (1–20)
  - Optional speculative decoding: `SPECULATIVE_DECODING_METHOD` (`"eagle3"` or `"suffix"`), `DISABLE_SPECULATIVE_DECODING` (`"true"` or `"false"`), `NUM_SPECULATIVE_TOKENS` (1–10), `SUFFIX_DECODING_MAX_TREE_DEPTH`, `SUFFIX_DECODING_MAX_CACHED_REQUESTS`, `SUFFIX_DECODING_MAX_SPEC_FACTOR`, `SUFFIX_DECODING_MIN_TOKEN_PROB`
  - Optional memory/quantization: `KV_CACHE_DTYPE` (`"fp8"`), `QUANTIZATION_DTYPE` (`"fp8"`)
- `skip_model_reuse` (bool): If True, always create a new model. Default: False
- `inference_component_configs` (List[InferenceComponentConfig]): List of inference component configs. When provided with `SAGEMAKER` platform, creates an IC-compatible endpoint and deploys the inference component(s) in one step.

**Returns:**
- `DeploymentResult`: Contains `endpoint` (EndpointInfo), `platform`, `endpoint_name`, `uri`, `model_artifact_path`, and `created_at`

**Raises:**
- `Exception`: When unable to deploy the model
- `ValueError`: If platform is not supported

**Example:**
```python
# Deploy from S3 artifacts to Bedrock
deployment = deployer.deploy(
    model_artifact_path="s3://escrow-bucket/my-model-artifacts/",
    deploy_platform=DeployPlatform.BEDROCK_OD,
    endpoint_name="my-custom-nova-model"
)
print(f"Model deployed: {deployment.endpoint.uri}")

# Deploy to SageMaker using a model package ARN (auto-detected)
deployment = deployer.deploy(
    model_artifact_path="arn:aws:sagemaker:us-east-1:123456789012:model-package/my-group/1",
    deploy_platform=DeployPlatform.SAGEMAKER,
    unit_count=1,
    sagemaker_instance_type="ml.p5.48xlarge",
)
print(f"Model deployed: {deployment.endpoint.uri}")
```
---

##### `create_custom_model()`
Creates a Bedrock custom model from S3 artifacts or a model package ARN without deploying to an endpoint.

Either `model_artifact_path` (maps to `modelSourceConfig`) or `custom_model_data_source` (maps to `customModelDataSource`) must be provided, but not both.

**Signature:**
```python
def create_custom_model(
    self,
    model_artifact_path: Optional[str] = None,
    endpoint_name: Optional[str] = None,
    execution_role_name: Optional[str] = None,
    tags: Optional[List[Dict[str, str]]] = None,
    skip_model_reuse: bool = False,
    custom_model_data_source: Optional[Dict[str, Any]] = None,
) -> ModelDeployResult
```

**Parameters:**
- `model_artifact_path` (Optional[str]): S3 path to trained model checkpoint. Used to populate `modelSourceConfig.s3DataSource.s3Uri`
- `endpoint_name` (Optional[str]): Optional name prefix for the model name
- `execution_role_name` (Optional[str]): IAM role name for Bedrock
- `tags` (Optional[List[Dict[str, str]]]): Optional list of `{"key": str, "value": str}` dicts for tracking
- `skip_model_reuse` (bool): If True, always create a new model. Default: False
- `custom_model_data_source` (Optional[Dict[str, Any]]): Alternative data source configuration for the custom model. When provided, `modelSourceConfig` is omitted from the API call

**Returns:**
- `ModelDeployResult`: Contains `model_arn`, `model_name`, `escrow_uri`, and `created_at`

**Raises:**
- `ValueError`: If neither or both of `model_artifact_path` and `custom_model_data_source` are provided

**Example:**
```python
# Create from S3 artifacts
publish_result = deployer.create_custom_model(
    model_artifact_path="s3://escrow-bucket/my-model-artifacts/"
)
print(f"Model ARN: {publish_result.model_arn}")
publish_result.dump(file_path="./results/")

# Create from a model package ARN
publish_result = deployer.create_custom_model(
    custom_model_data_source={
        "modelPackageArnDataSource": {
            "modelPackageArn": "arn:aws:sagemaker:us-east-1:123456789012:model-package/my-group/1"
        }
    }
)
print(f"Model ARN: {publish_result.model_arn}")
```
---

##### `deploy_to_bedrock()`
Deploys a published Bedrock custom model to an endpoint.

**Signature:**
```python
def deploy_to_bedrock(
    self,
    model_deploy_result: Optional[ModelDeployResult] = None,
    model_arn: Optional[str] = None,
    deploy_platform: DeployPlatform = DeployPlatform.BEDROCK_OD,
    pt_units: Optional[int] = None,
    endpoint_name: Optional[str] = None,
) -> DeploymentResult
```

**Parameters:**
- `model_deploy_result` (Optional[ModelDeployResult]): Result from `create_custom_model()`. Cannot be combined with `model_arn`
- `model_arn` (Optional[str]): Direct model ARN. Cannot be combined with `model_deploy_result`
- `deploy_platform` (DeployPlatform): `BEDROCK_OD` (default) or `BEDROCK_PT`
- `pt_units` (Optional[int]): Number of PT units (required for `BEDROCK_PT`)
- `endpoint_name` (Optional[str]): Endpoint name (auto-generated if not provided)

**Returns:**
- `DeploymentResult`: Contains `endpoint`, `created_at`, and `model_publish`

**Raises:**
- `ValueError`: When both `model_deploy_result` and `model_arn` are provided, or when no model ARN is available
- `RuntimeError`: When deployment creation fails

**Example:**
```python
# Two-step deploy: create model, then deploy
publish_result = deployer.create_custom_model(
    model_artifact_path="s3://escrow-bucket/my-model-artifacts/"
)
deployment = deployer.deploy_to_bedrock(
    model_deploy_result=publish_result,
    endpoint_name="my-endpoint"
)

# Or deploy from an existing model ARN
deployment = deployer.deploy_to_bedrock(
    model_arn="arn:aws:bedrock:us-east-1:123456789012:custom-model/my-model"
)
```
---

##### `find_published_model()`
Finds an existing published model by platform and escrow path to enable model reuse.

**Signature:**
```python
def find_published_model(
    self,
    platform: str,
    escrow_path: str,
    skip_model_reuse: bool = False,
) -> Optional[str]
```

**Parameters:**
- `platform` (str): Target platform (`"bedrock"` or `"sagemaker"`)
- `escrow_path` (str): S3 path of the model artifacts
- `skip_model_reuse` (bool): If True, always returns None (skips lookup). Default: False

**Returns:**
- `Optional[str]`: Existing model ARN if found, otherwise None

**Example:**
```python
existing_arn = deployer.find_published_model(
    platform="bedrock",
    escrow_path="s3://escrow-bucket/my-model-artifacts/"
)
if existing_arn:
    print(f"Reusing existing model: {existing_arn}")
```
---

##### `get_status()`
Gets the deployment status for a DeploymentResult.

**Signature:**
```python
def get_status(self, result: DeploymentResult) -> JobStatus
```

**Parameters:**
- `result` (DeploymentResult): The deployment result to check

**Returns:**
- `JobStatus`: Current status (`IN_PROGRESS`, `COMPLETED`, or `FAILED`)

---

##### `get_status_by_arn()`
Gets the deployment status by endpoint ARN and platform.

**Signature:**
```python
def get_status_by_arn(
    self,
    endpoint_arn: str,
    platform: DeployPlatform,
) -> Optional[JobStatus]
```

**Parameters:**
- `endpoint_arn` (str): The endpoint ARN to check
- `platform` (DeployPlatform): The deployment platform

**Returns:**
- `Optional[JobStatus]`: Current status, or None if status cannot be determined

---

##### `get_logs()`
Retrieves and displays logs for a deployment.

**Signature:**
```python
def get_logs(
    self,
    job_result=None,
    endpoint_arn=None,
    platform=None,
) -> None
```

**Parameters:**
- `job_result` (Optional[DeploymentResult]): Deployment result to retrieve logs for
- `endpoint_arn` (Optional[str]): Endpoint ARN (used if `job_result` is not provided)
- `platform` (Optional[DeployPlatform]): Deployment platform (used with `endpoint_arn`)

**Returns:**
- None (prints logs to console)

---

##### `create_inference_component()`
Creates an inference component on an existing SageMaker endpoint. Returns immediately without waiting for the component to become active.

**Signature:**
```python
def create_inference_component(
    self,
    inference_component_name: str,
    model_name: str,
    num_cpus: int,
    num_accelerators: int,
    min_memory_in_mb: int,
    endpoint_name: str,
    variant_name: str = "primary",
    copy_count: int = 1,
) -> DeploymentResult
```

**Parameters:**
- `inference_component_name` (str): Unique name for the inference component
- `model_name` (str): Name of the existing SageMaker model to use
- `num_cpus` (int): Number of vCPUs to allocate
- `num_accelerators` (int): Number of accelerators (GPUs) to allocate
- `min_memory_in_mb` (int): Minimum memory in MB to allocate
- `endpoint_name` (str): Name of the existing SageMaker endpoint (must be InService)
- `variant_name` (str): Production variant name on the endpoint. Default: `"primary"`
- `copy_count` (int): Number of model copies to deploy. Default: 1

**Returns:**
- `DeploymentResult`: Contains endpoint info with the inference component ARN as the URI and the deployer's `region` set on `EndpointInfo`

**Raises:**
- `Exception`: If the endpoint does not exist, is not InService, or the API call fails

**Example:**
```python
result = deployer.create_inference_component(
    inference_component_name="my-model-ic",
    model_name="my-sagemaker-model",
    num_cpus=15,
    num_accelerators=4,
    min_memory_in_mb=25000,
    endpoint_name="my-endpoint",
)
print(f"Inference component ARN: {result.endpoint.uri}")
```
---

##### `monitor_inference_component()`
Polls an inference component until it reaches a terminal state (InService or Failed).

**Signature:**
```python
def monitor_inference_component(self, inference_component_name: str) -> str
```

**Parameters:**
- `inference_component_name` (str): Name of the inference component to monitor

**Returns:**
- `str`: Final status (`"InService"`)

**Raises:**
- `Exception`: If the component reaches Failed status or the API call errors

**Example:**
```python
status = deployer.monitor_inference_component(inference_component_name="my-model-ic")
print(f"Inference component is now: {status}")
```
---

### ForgeInference

Handles single and batch inference on trained Nova models.

#### Constructor

**Signature:**
```python
def __init__(
    self,
    region: Optional[str] = None,
    model: Optional[Model] = None,
    infra: Optional[RuntimeManager] = None,
    config: Optional[ForgeConfig] = None,
    method: Optional[TrainingMethod] = None,
    hub_content_version: Optional[str] = None,
)
```

**Parameters:**
- `region` (Optional[str]): AWS region. Auto-detected if not provided
- `model` (Optional[Model]): The Nova model (required for batch inference)
- `infra` (Optional[RuntimeManager]): Runtime infrastructure manager (required for batch inference)
- `config` (Optional[ForgeConfig]): Shared configuration
- `method` (Optional[TrainingMethod]): Training method (used for batch inference recipe generation)
- `hub_content_version` (Optional[str]): Version of the hub content to retrieve from SageMaker Hub. If None, uses the latest version

**Example:**
```python
from amzn_nova_forge.inference import ForgeInference

# For single inference (minimal setup)
inference = ForgeInference(region="us-east-1")

# For batch inference
inference = ForgeInference(
    region="us-east-1",
    model=Model.NOVA_MICRO,
    infra=SMTJRuntimeManager(instance_type="ml.p5.48xlarge", instance_count=1),
    method=TrainingMethod.SFT_LORA
)
```
---

#### Methods

##### `invoke()`
Invokes a single inference on a deployed model endpoint.

**Signature:**
```python
def invoke(
    self,
    endpoint_arn: str,
    request_body: Dict[str, Any],
) -> Any
```

**Parameters:**
- `endpoint_arn` (str): Endpoint ARN to invoke
- `request_body` (Dict[str, Any]): Inference request body

**Returns:**
- `Any`: Inference response

**Example:**
```python
response = inference.invoke(
    endpoint_arn="arn:aws:bedrock:us-east-1:123456789012:endpoint/my-endpoint",
    request_body={
        "messages": [{"role": "user", "content": "Hello! How are you?"}],
        "max_tokens": 100,
        "stream": False
    }
)
```
---

##### `invoke_batch()`
Launches a batch inference job on a trained model.

**Signature:**
```python
def invoke_batch(
    self,
    job_name: str,
    input_path: str,
    output_s3_path: str,
    model_path: Optional[str] = None,
    recipe_path: Optional[str] = None,
    overrides: Optional[Dict[str, Any]] = None,
    dry_run: bool = False,
    job_result: Optional[TrainingResult] = None,
) -> Optional[InferenceResult]
```

**Parameters:**
- `job_name` (str): Name for the batch inference job
- `input_path` (str): S3 path to input data
- `output_s3_path` (str): S3 path for inference outputs
- `model_path` (Optional[str]): S3 path to the model checkpoint
- `recipe_path` (Optional[str]): Path for a YAML recipe file
- `overrides` (Optional[Dict[str, Any]]): Inference configuration overrides (e.g., `max_new_tokens`, `temperature`, `top_p`)
- `dry_run` (bool): If True, performs validation only. Default: False
- `job_result` (Optional[TrainingResult]): Training result to extract checkpoint path from

**Returns:**
- `InferenceResult`: Metadata object containing `job_id`, `started_time`, and `inference_output_path`. Returns `None` if `dry_run=True`

**Example:**
```python
inference_result = inference.invoke_batch(
    job_name="batch-inference-job",
    input_path="s3://my-bucket/inference-input",
    output_s3_path="s3://my-bucket/inference-output",
    model_path="s3://my-bucket/trained-model"
)
print(f"Batch inference started: {inference_result.job_id}")
```
---

##### `get_logs()`
Retrieves and displays CloudWatch logs for an inference job.

**Signature:**
```python
def get_logs(
    self,
    job_result=None,
    job_id=None,
    started_time=None,
    limit=None,
    start_from_head: bool = False,
    end_time=None,
) -> None
```

**Parameters:**
- `job_result` (Optional[InferenceResult]): Job result to retrieve logs for
- `job_id` (Optional[str]): Job identifier
- `started_time` (Optional[datetime]): Job start time to filter logs
- `limit` (Optional[int]): Maximum number of log lines
- `start_from_head` (bool): If True, start from the beginning of logs. Default: False
- `end_time` (Optional[int]): End time in epoch milliseconds for searching a log time range

**Returns:**
- None (prints logs to console)

**Example:**
```python
inference.get_logs(job_result=inference_result, limit=100)
```
---

### End-to-End Example (Service Classes)

```python
from amzn_nova_forge.trainer import ForgeTrainer
from amzn_nova_forge.evaluator import ForgeEvaluator
from amzn_nova_forge.deployer import ForgeDeployer
from amzn_nova_forge.inference import ForgeInference
from amzn_nova_forge.core import ForgeConfig
from amzn_nova_forge.manager import SMTJRuntimeManager
from amzn_nova_forge.model.model_enums import Model, TrainingMethod, DeployPlatform
from amzn_nova_forge.core import EvaluationTask

# Shared configuration
config = ForgeConfig(
    output_s3_path="s3://my-bucket/output",
    enable_job_caching=True
)
infra = SMTJRuntimeManager(instance_type="ml.p5.48xlarge", instance_count=2)

# 1. Train
trainer = ForgeTrainer(
    model=Model.NOVA_MICRO,
    method=TrainingMethod.SFT_LORA,
    infra=infra,
    training_data_s3_path="s3://my-bucket/data.jsonl",
    config=config
)
train_result = trainer.train(job_name="my-training-job")

# 2. Evaluate
evaluator = ForgeEvaluator(model=Model.NOVA_MICRO, infra=infra, config=config)
eval_result = evaluator.evaluate(
    job_name="my-eval-job",
    eval_task=EvaluationTask.MMLU,
    job_result=train_result
)

# 3. Deploy
deployer = ForgeDeployer(region="us-east-1", model=Model.NOVA_MICRO)
deployment = deployer.deploy(
    model_artifact_path=train_result.model_artifacts.checkpoint_s3_path,
    deploy_platform=DeployPlatform.BEDROCK_OD,
    endpoint_name="my-nova-endpoint"
)

# 4. Inference
inference_client = ForgeInference(region="us-east-1")
response = inference_client.invoke(
    endpoint_arn=deployment.endpoint.uri,
    request_body={
        "messages": [{"role": "user", "content": "Hello!"}],
        "max_tokens": 100
    }
)
```

---
