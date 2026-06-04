# Utility Functions

### verify_reward_function()

Verifies a reward function with sample data before using it in RFT training or evaluation. This utility helps you test your reward function implementation to ensure it works correctly and returns the expected format.

**Signature:**
```python
def verify_reward_function(
    reward_function: str,
    sample_data: List[Dict[str, Any]],
    region: str = "us-east-1",
    validate_format: bool = True,
    platform: Optional[Platform] = None,
) -> Dict[str, Any]
```

**Parameters:**
- `reward_function` (str): Either a Lambda ARN (string starting with `'arn:aws:lambda:'`) or a path to a local Python file containing the reward function.
- `sample_data` (List[Dict[str, Any]]): List of conversation samples to test. Each sample should be a dict with `'id'`, `'messages'`, and optionally `'reference_answer'` keys.
- `region` (str): AWS region for Lambda invocation (default: "us-east-1").
- `validate_format` (bool): If True, validates that sample_data matches RFT format and output matches expected format (default: True).
- `platform` (Platform): Platform enum (Platform.SMHP or Platform.SMTJ). **Required when using Lambda ARN**. When set to Platform.SMHP, validates that Lambda ARN contains 'SageMaker' in the function name as required by SageMaker HyperPod. Optional for local files.

**Returns:**
- `Dict[str, Any]`: Dictionary containing:
  - `success` (bool): Always True if no exception raised
  - `results` (list): List of individual test results
  - `total_samples` (int): Total number of samples tested
  - `successful_samples` (int): Number of successful tests
  - `warnings` (list): List of warning messages (e.g., missing reference_answer)

**Raises:**
- `ValueError`: If any validation errors are encountered, with a detailed error message listing all issues found.

**Example**
```python
from amzn_nova_forge import verify_reward_function
from amzn_nova_forge.model import Platform

# Test with Lambda ARN (platform required for Lambda ARNs)
result = verify_reward_function(
    reward_function="arn:aws:lambda:us-east-1:123456789012:function:MySageMakerReward",
    sample_data=[
        {
            "id": "sample_1",
            "reference_answer": "correct answer",
            "messages": [
                {"role": "user", "content": "question"},
                {"role": "assistant", "content": "response"}
            ]
        }
    ],
    platform=Platform.SMHP  # Required for Lambda ARNs
)

print(f"Verification: {'PASSED' if result['success'] else 'FAILED'}")
print(f"Tested {result['total_samples']} samples, {result['successful_samples']} successful")

if result.get('warnings'):
    print(f"\nWarnings:")
    for warning in result['warnings']:
        print(f"  - {warning}")

# Test with local Python file (platform optional)
result = verify_reward_function(
    reward_function="./my_reward_function.py",
    sample_data=[
        {
            "id": "sample_1",
            "reference_answer": "correct answer",
            "messages": [
                {"role": "user", "content": "question"},
                {"role": "assistant", "content": "response"}
            ]
        }
    ]
)
```
**Output Format Requirements from Lambda:**

```python
{
    "id": "sample_1",                   # Required: string
    "aggregate_reward_score": 0.75,     # Required: float or int
    "metrics_list": [                   # Optional: validated if present
        {
            "name": "accuracy",         # Required: string
            "value": 0.85,              # Required: float or int
            "type": "Metric"            # Required: "Metric" or "Reward"
        }
    ]
}
```

**Common Validation Errors:**
- Missing required fields in input (`messages` field is required)
- Missing required fields in output (`id` and `aggregate_reward_score` are required)
- Invalid data types (e.g., `aggregate_reward_score` must be a number)
- Missing `platform` parameter when using Lambda ARN
- SMHP Lambda ARN doesn't contain 'SageMaker' in function name
- Invalid `metrics_list` structure (must be list of dicts with `name`, `value`, `type`)
- Invalid metric `type` (must be "Metric" or "Reward")

**Warnings:**
- Missing `reference_answer`: While optional in RFT datasets, reference answers are recommended for meaningful reward calculations. Without ground truth, your reward function cannot compare model outputs against expected answers.



**Note:** The `metrics_list` field is optional. If provided, it will be validated for proper structure and logged during training/evaluation.

---
## Monitoring

### CloudWatchLogMonitor

Monitors CloudWatch logs and plots training metrics for Nova model training jobs. Supports both SageMaker Training Jobs (SMTJ) and SageMaker HyperPod (SMHP) platforms.

#### Factory Methods

##### `from_job_id()`

Creates a CloudWatchLogMonitor from a job ID.

**Signature:**
```python
@classmethod
def from_job_id(
    cls,
    job_id: str,
    platform: Platform,
    started_time: Optional[datetime] = None,
    **kwargs,
) -> "CloudWatchLogMonitor"
```

**Parameters:**
- `job_id` (str): The training job identifier
- `platform` (Platform): Execution platform (`Platform.SMTJ` or `Platform.SMHP`)
- `started_time` (Optional[datetime]): Job start time (used to filter logs)
- `**kwargs`: Platform-specific parameters:
  - SMHP requires: `cluster_name` (str), optional `namespace` (str, defaults to "kubeflow")

**Returns:**
- `CloudWatchLogMonitor`: Monitor instance

**Example:**
```python
from amzn_nova_forge.monitor import CloudWatchLogMonitor
from amzn_nova_forge.model import Platform

# SMTJ
monitor = CloudWatchLogMonitor.from_job_id(
    job_id="my-training-job",
    platform=Platform.SMTJ,
    started_time=datetime(2026, 1, 15, 12, 0, 0)
)

# SMHP
monitor = CloudWatchLogMonitor.from_job_id(
    job_id="my-hyperpod-job",
    platform=Platform.SMHP,
    cluster_name="my-cluster",
    namespace="kubeflow"
)
```

---

##### `from_job_result()`

Creates a CloudWatchLogMonitor from a training job result object.

**Signature:**
```python
@classmethod
def from_job_result(
    cls,
    job_result: BaseJobResult,
    cloudwatch_logs_client=None
) -> "CloudWatchLogMonitor"
```

**Parameters:**
- `job_result` (BaseJobResult): A training or evaluation result object (e.g., `TrainingResult`)
- `cloudwatch_logs_client` (Optional): Boto3 CloudWatch Logs client (auto-created if not provided)

**Returns:**
- `CloudWatchLogMonitor`: Monitor instance

**Example:**
```python
result = customizer.train(job_name="my-job")
monitor = CloudWatchLogMonitor.from_job_result(job_result=result)
```

---

#### Methods

##### `get_logs()`

Retrieves CloudWatch log events for the job.

**Signature:**
```python
def get_logs(
    self,
    limit: Optional[int] = None,
    start_from_head: bool = False,
    end_time: Optional[int] = None,
) -> List[Dict]
```

**Parameters:**
- `limit` (Optional[int]): Maximum number of log events to retrieve
- `start_from_head` (bool): If True, start from the beginning of logs; if False, start from the end
- `end_time` (Optional[int]): End time in epoch milliseconds

**Returns:**
- `List[Dict]`: List of log event dictionaries, each containing a `"message"` key

**Example:**
```python
logs = monitor.get_logs(limit=100)
```

---

##### `show_logs()`

Prints CloudWatch log messages to the console.

**Signature:**
```python
def show_logs(
    self,
    limit: Optional[int] = None,
    start_from_head: bool = False,
    end_time: Optional[int] = None,
) -> None
```

**Parameters:**
- `limit` (Optional[int]): Maximum number of log events to display
- `start_from_head` (bool): If True, start from the beginning of logs; if False, start from the end
- `end_time` (Optional[int]): End time in epoch milliseconds

**Example:**
```python
monitor.show_logs(limit=50, start_from_head=True)
```

---

##### `plot_metrics()`

Parses training metrics from CloudWatch logs and displays them as matplotlib plots. Automatically fetches the latest logs if the job is still in progress or logs have not been retrieved yet.

**Signature:**
```python
def plot_metrics(
    self,
    training_method: TrainingMethod,
    metrics: Optional[List[str]] = None,
    starting_step: Optional[int] = None,
    ending_step: Optional[int] = None,
) -> None
```

**Parameters:**
- `training_method` (TrainingMethod): The training method used for the job (e.g., `TrainingMethod.SFT_LORA`, `TrainingMethod.CPT`, `TrainingMethod.RFT_LORA`)
- `metrics` (Optional[List[str]]): List of metric names to plot. Available metrics depend on training method:
  - CPT / SFT: `"training_loss"`
  - RFT: `"reward_score"`
- `starting_step` (Optional[int]): Filter to only show metrics from this global step onward
- `ending_step` (Optional[int]): Filter to only show metrics up to this global step

**Raises:**
- `ValueError`: If `starting_step` > `ending_step`, or if no logs are found for the job
- `NotImplementedError`: If an unsupported metric is requested for the given training method/platform

**Example:**
```python
from amzn_nova_forge.monitor import CloudWatchLogMonitor
from amzn_nova_forge.model import Platform, TrainingMethod

# Create monitor from a training result
monitor = CloudWatchLogMonitor.from_job_result(job_result=training_result)

# Plot training loss for an SFT job
monitor.plot_metrics(
    training_method=TrainingMethod.SFT_LORA,
    metrics=["training_loss"]
)

# Plot reward score for an RFT job, filtered to steps 50-200
monitor.plot_metrics(
    training_method=TrainingMethod.RFT_LORA,
    metrics=["reward_score"],
    starting_step=50,
    ending_step=200
)
```

---

### MTRLLogMonitor

Log monitor for MTRL (Multi-Turn RL) training and evaluation jobs. Auto-detects whether a job is a training job or an evaluation job by checking CloudWatch log groups.

#### Constructor

**Signature:**
```python
def __init__(
    self,
    job_id: str,
    region: Optional[str] = None,
    job_category: Optional[str] = None,
)
```

**Parameters:**
- `job_id` (str): The MTRL job name (training or evaluation)
- `region` (Optional[str]): AWS region
- `job_category` (Optional[str]): Job category override (`"AgentRFT"` for training, `"AgentRFTEvaluation"` for eval). Auto-detected if not provided.

#### Class Methods

##### `from_job_id()`

**Signature:**
```python
@classmethod
def from_job_id(
    cls,
    job_id: str,
    region: Optional[str] = None,
    job_category: Optional[str] = None,
) -> "MTRLLogMonitor"
```

**Example:**
```python
from amzn_nova_forge.monitor import MTRLLogMonitor

# Auto-detects training vs eval
monitor = MTRLLogMonitor.from_job_id(
    job_id="my-mtrl-job",
    region="us-east-1",
)
monitor.show_logs(limit=20)
```

#### Methods

##### `show_logs()`

Displays job logs. For training jobs, uses `AgentRFTJob.wait()` for live progress. For evaluation jobs, reads CloudWatch logs from `/aws/sagemaker/Job/AgentRFTEvaluation`.

**Signature:**
```python
def show_logs(
    self,
    poll: int = 30,
    timeout: int = 7200,
    limit: Optional[int] = None,
) -> None
```

**Parameters:**
- `poll` (int): Polling interval in seconds (training jobs only)
- `timeout` (int): Maximum wait time in seconds (training jobs only)
- `limit` (Optional[int]): Maximum number of log events to display (eval jobs)

**Example:**
```python
# Monitor an eval job
monitor = MTRLLogMonitor.from_job_id(job_id="my-eval-job", region="us-east-1")
monitor.show_logs(limit=50)

# Also accessible via ForgeEvaluator.get_logs()
evaluator.get_logs(job_result=eval_result, limit=20)
```

---

### MLflowMonitor

MLflow monitoring configuration for Nova model training. This class provides experiment tracking capabilities through MLflow integration.

**Note:** MLflow monitoring is only supported for SageMaker platforms (SMTJ, SMHP). It is not available for Bedrock platform.

**MLflow Integration Features:**
- Automatic logging of training metrics
- Model artifact and checkpoint tracking
- Hyperparameter recording
- Support for SageMaker MLflow tracking servers
- Custom MLflow tracking server support (with proper network configuration)


#### Constructor

**Signature:**
```python
def __init__(
 self,
 tracking_uri: Optional[str] = None,
 experiment_name: Optional[str] = None,
 run_name: Optional[str] = None,
)
```

**Parameters:**
- `tracking_uri` (Optional[str]): MLflow tracking server URI or SageMaker MLflow app ARN. If not provided, attempts to use a default SageMaker MLflow tracking server if one exists
- `experiment_name` (Optional[str]): Name of the MLflow experiment. If not provided, will use the job name
- `run_name` (Optional[str]): Name of the MLflow run. If not provided, will be auto-generated

**Raises:**
- `ValueError`: If MLflow configuration validation fails

**Example:**
```python
from amzn_nova_forge.monitor import *

# With explicit tracking URI
monitor = MLflowMonitor(
    tracking_uri="arn:aws:sagemaker:us-east-1:123456789012:mlflow-app/app-xxx",
    experiment_name="nova-customization",
    run_name="sft-run-1"
)

# With default tracking URI (if available)
monitor = MLflowMonitor(
    experiment_name="nova-customization",
    run_name="sft-run-1"
)

# Use with NovaModelCustomizer
customizer = NovaModelCustomizer(
    model=Model.NOVA_LITE_2,
    method=TrainingMethod.SFT_LORA,
    infra=runtime_manager,
    data_s3_path="s3://bucket/data",
    mlflow_monitor=monitor
)
```

#### Methods

##### `to_dict()`

Converts MLflow configuration to dictionary format for use in recipe overrides.

**Signature:**
```python
def to_dict(
 self
) -> dict
```

**Returns:**
- `dict`: Dictionary with mlflow_* keys for recipe configuration. Returns empty dict if no tracking URI is available

**Example:**
```python
monitor = MLflowMonitor(
 tracking_uri="arn:aws:sagemaker:us-east-1:123456789012:mlflow-app/app-xxx",
 experiment_name="nova-customization"
)

config_dict = monitor.to_dict()
# Returns: {
#   "mlflow_tracking_uri": "arn:aws:sagemaker:us-east-1:123456789012:mlflow-app/app-xxx",
#   "mlflow_experiment_name": "nova-customization"
# }
```

##### `get_presigned_url()`

Generates a presigned URL for accessing the MLflow tracking server UI directly without navigating through the AWS Console.

**Signature:**
```python
def get_presigned_url(
 self,
 session_expiration_duration_in_seconds: int = 43200,
 expires_in_seconds: int = 300
) -> str
```

**Parameters:**
- `session_expiration_duration_in_seconds` (int, optional): Duration in seconds for which the MLflow UI session is valid after accessing the presigned URL. Default is 43200 seconds (12 hours). Valid range: 1800-43200 seconds
- `expires_in_seconds` (int, optional): Duration in seconds for which the presigned URL itself is valid. The URL must be accessed within this time. Default is 300 seconds (5 minutes). Valid range: 5-300 seconds

**Returns:**
- `str`: Presigned URL for accessing the MLflow tracking server UI. This URL must be used within `expires_in_seconds`

**Raises:**
- `ValueError`: If tracking_uri is not set
- `RuntimeError`: If unable to generate presigned URL

**Example:**
```python
monitor = MLflowMonitor(
    tracking_uri="arn:aws:sagemaker:us-east-1:123456789012:mlflow-app/app-xxx",
    experiment_name="nova-customization"
)

# Generate presigned URL with defaults
# URL expires in 5 minutes, but session lasts 12 hours once accessed
url = monitor.get_presigned_url()
print(f"Access MLflow UI at: {url}")

# Generate URL with custom expiration times
url = monitor.get_presigned_url(
    session_expiration_duration_in_seconds=3600,  # 1 hour session
    expires_in_seconds=60  # URL expires in 1 minute
)
```

#### MLflow Integration Notes

When MLflow monitoring is enabled:
1. Training metrics will be automatically logged to the specified MLflow tracking server
2. Model artifacts and checkpoints will be tracked in MLflow
3. Hyperparameters and configuration will be recorded as MLflow parameters
4. You can view experiment results in the MLflow UI

The MLflow integration supports:
- SageMaker MLflow tracking servers
- Custom MLflow tracking servers (with appropriate network configuration)
- Automatic experiment and run creation
- Metric logging during training
- Artifact tracking

---
