# NovaModelCustomizer (Deprecated)

> **Deprecated**: `NovaModelCustomizer` is a legacy facade. Use `ForgeTrainer`, `ForgeEvaluator`, `ForgeDeployer`, and `ForgeInference` instead.

The main entrypoint class for customizing and training Nova models.

### Constructor

#### `__init__()`
Initializes a NovaModelCustomizer instance.

**Signature:**
```python
def __init__(
 self,
 model: Model,
 method: TrainingMethod,
 infra: RuntimeManager,
 data_s3_path: Optional[str] = None,
 output_s3_path: Optional[str] = None,
 model_path: Optional[str] = None,
 validation_config: Optional[Dict[str, bool]] = None,
 generated_recipe_dir: Optional[str] = None,
 mlflow_monitor: Optional[MLflowMonitor] = None,
 deployment_mode: DeploymentMode = DeploymentMode.FAIL_IF_EXISTS,
 data_mixing_enabled: bool = False,
 enable_job_caching: bool = False,
 is_multimodal: Optional[bool] = None,
 hub_content_version: Optional[str] = None,
)
```
**Parameters:**
- `model` (Model): The Nova model to be trained (e.g., `Model.NOVA_MICRO`, `Model.NOVA_LITE`, `Model.NOVA_LITE_2`, `Model.NOVA_PRO`)
- `method` (TrainingMethod): The fine-tuning method (e.g., `TrainingMethod.SFT_LORA`, `TrainingMethod.RFT`)
- `infra` (RuntimeManager): Runtime infrastructure manager (e.g., `SMTJRuntimeManager`, `SMHPRuntimeManager`, or `BedrockRuntimeManager`)
- `data_s3_path` (Optional[str]): S3 path to the training dataset
- `output_s3_path` (Optional[str]): S3 path for output artifacts. If not provided, will be auto-generated
- `model_path` (Optional[str]): S3 path for model path
- `validation_config` (Optional[Dict[str, bool]]): Optional dict to control validation. Defaults to `{'iam': True, 'infra': True, 'recipe': True}`.
  - `iam` (bool): Enable IAM permission validation (default: True)
  - `infra` (bool): Enable infrastructure validation (default: True)
  - `recipe` (bool): Enable recipe constraint validation (default: True)
- `generated_recipe_dir` (Optional[str]): Optional local path to save the generated recipe
- `mlflow_monitor` (Optional[MLflowMonitor]): Optional MLflow monitoring configuration for experiment tracking (SageMaker only, not supported on Bedrock)
- `deployment_mode` (DeploymentMode): Behavior when deploying to existing endpoint name. Options: FAIL_IF_EXISTS (default), UPDATE_IF_EXISTS
- `data_mixing_enabled` (bool): Enable data mixing feature for CPT and SFT training on SageMaker HyperPod, and SFT text-only on Nova Lite 2 on SMTJServerless. Default is False
  - **Note:** The `data_mixing_enabled` parameter must be set to `True` during initialization to use data mixing features.
  - **Note:** On SMHP: supported for CPT, SFT_LORA, and SFT_FULL across Nova 1 and Nova 2 models. On SMTJServerless: supported for SFT_LORA and SFT_FULL with text-only data on Nova Lite 2 only.
- `is_multimodal` (Optional[bool]): Only applicable when `data_mixing_enabled=True`. Explicitly set multimodal mode. If None (default), auto-detects from data. Set to False to skip detection for performance on large text-only datasets. Ignored when `data_mixing_enabled=False`
- `enable_job_caching` (bool): Whether to enable job result caching. When enabled, completed job results are cached to `job_cache_dir` (default: `.cached-nova-jobs/`) and reused for identical job configurations. Default: False
- `hub_content_version` (Optional[str]): Version of the hub content to retrieve from SageMaker Hub. If None, uses the latest version

**Raises:**
- `ValueError`: If region is unsupported or model is invalid

**Example:**
```python
from amzn_nova_forge import *

# SageMaker Training Jobs (SMTJ)
infra = SMTJRuntimeManager(instance_type="ml.p5.48xlarge", instance_count=2)

customizer = NovaModelCustomizer(
 model=Model.NOVA_MICRO,
 method=TrainingMethod.SFT_LORA,
 infra=infra,
 data_s3_path="s3://my-bucket/training-data/data.jsonl",
 output_s3_path="s3://my-bucket/output"
)

# Amazon Bedrock (fully managed)
bedrock_infra = BedrockRuntimeManager(
 execution_role="arn:aws:iam::123456789012:role/BedrockRole",
 base_model_identifier="arn:aws:bedrock:us-east-1::custom-model/amazon.nova-2-lite-v1:0:256k:abcdefghijk"
)

bedrock_customizer = NovaModelCustomizer(
 model=Model.NOVA_MICRO,
 method=TrainingMethod.SFT_LORA,
 infra=bedrock_infra,
 data_s3_path="s3://my-bucket/training-data/data.jsonl",
 output_s3_path="s3://my-bucket/output"
)

# With MLflow monitoring (SageMaker only)
mlflow_monitor = MLflowMonitor(
 tracking_uri="arn:aws:sagemaker:us-east-1:123456789012:mlflow-app/app-xxx",
 experiment_name="nova-customization",
 run_name="sft-run-1"
)

customizer_with_mlflow = NovaModelCustomizer(
 model=Model.NOVA_MICRO,
 method=TrainingMethod.SFT_LORA,
 infra=infra,
 data_s3_path="s3://my-bucket/training-data/data.jsonl",
 output_s3_path="s3://my-bucket/output",
 mlflow_monitor=mlflow_monitor
)
```
---
### Methods

#### `get_data_mixing_config()`
Get the current data mixing configuration.

**Signature:**
```python
def get_data_mixing_config(
 self
) -> Dict[str, Any]
```

**Returns:**
- `Dict[str, Any]`: Dictionary containing the data mixing configuration

**Example:**
```python
config = customizer.get_data_mixing_config()
print(config)
# Output: {'customer_data_percent': 50, 'nova_code_percent': 30, 'nova_general_percent': 70}
```

---

#### `set_data_mixing_config()`
Set the data mixing configuration.

**Signature:**
```python
def set_data_mixing_config(
 self,
 config: Dict[str, Any]
) -> None
```

**Parameters:**
- `config` (Dict[str, Any]): Dictionary containing the data mixing configuration
  - `customer_data_percent` (int/float): Percentage of customer data (0-100)
  - `nova_code_percent` (int/float): Percentage of Nova code data (0-100)
  - `nova_general_percent` (int/float): Percentage of Nova general data (0-100)
  - Nova percentages must sum to 100%

**Raises:**
- `ValueError`: If data mixing is not enabled or configuration is invalid

**Note:**
- The SDK automatically detects whether your dataset is multimodal (contains images, videos, or documents) by scanning the data
- The appropriate Nova dataset catalog (text-only or multimodal) and nova data mixing fields are selected automatically based on this detection

**Example:**
```python
# Text datamixing (auto-detection)
customizer = NovaModelCustomizer(
    model=Model.NOVA_LITE_2,
    method=TrainingMethod.SFT_LORA,
    infra=SMHPRuntimeManager(...),
    data_s3_path="s3://bucket/data.jsonl",
    data_mixing_enabled=True,
    # is_multimodal=False,  # Optional: skip auto-detection for performance
)

customizer.set_data_mixing_config({
    "customer_data_percent": 50,
    "nova_code_percent": 30,
    "nova_general_percent": 70
})

# Multimodal datamixing (explicit)
customizer_mm = NovaModelCustomizer(
    model=Model.NOVA_LITE_2,
    method=TrainingMethod.SFT_LORA,
    infra=SMHPRuntimeManager(...),
    data_s3_path="s3://bucket/multimodal_data.jsonl",
    data_mixing_enabled=True,
    is_multimodal=True,
)

customizer_mm.set_data_mixing_config({
    "customer_data_percent": 50,
    "nova_general_percent": 70,
    "nova_code_percent": 30,
})
```

---

#### `train()`
Generates the recipe YAML, configures runtime, and launches a training job.

**Signature:**
```python
def train(
 self,
 job_name: str,
 recipe_path: Optional[str] = None,
 overrides: Optional[Dict[str, Any]] = None,
 rft_lambda_arn: Optional[str] = None,
 validation_data_s3_path: Optional[str] = None,
 val_check_interval: Optional[int] = None,
 dry_run: bool = False
) -> TrainingResult
```

**Parameters:**
- `job_name` (str): User-defined name for the training job
- `recipe_path` (Optional[str]): Path for a YAML recipe file (both S3 and local paths are accepted)
- `overrides` (Optional[Dict[str, Any]]): Dictionary of configuration overrides. Example overrides below:
  - `max_epochs` (int): Maximum number of training epochs
  - `lr` (float): Learning rate
  - `warmup_steps` (int): Number of warmup steps
  - `loraplus_lr_ratio` (float): LoRA+ learning rate ratio
  - `global_batch_size` (int): Global batch size
  - `max_length` (int): Maximum sequence length
  - `model_name_or_path` (str): S3 checkpoint path or a SageMaker Model Package ARN for iterative training
  - A full list of available overrides can be found via the [Nova Customization public documentation](https://docs.aws.amazon.com/nova/latest/userguide/customize-fine-tune-sagemaker.html) or by referencing the training recipes [here](https://docs.aws.amazon.com/sagemaker/latest/dg/nova-model-recipes.html).
- `rft_lambda_arn` (Optional[str]): Rewards Lambda ARN (only used for RFT training methods). If passed, takes priority over `rft_lambda_arn` set on the `RuntimeManager`.
- `validation_data_s3_path` (Optional[str]): Validation S3 path, applicable for CPT and SFT on SMTJ/SMTJServerless/SMHP, or any method on Bedrock (optional)
- `val_check_interval` (Optional[int]): How often (in training steps) to run validation. Defaults to 2500 if omitted. Only used when `validation_data_s3_path` is provided.
- `dry_run` (bool): Actually starts a job if False, otherwise just performs validation.

**Returns:**
- `TrainingResult`: Metadata object (either `SMTJTrainingResult`, `SMHPTrainingResult`, or `BedrockTrainingResult`) containing:
 - `job_id` (str): The training job identifier
 - `method` (TrainingMethod): The training method used
 - `started_time` (datetime): Job start timestamp
 - `model_artifacts` (ModelArtifacts): Paths to model checkpoints and outputs
   - `checkpoint_s3_path` (str, Optional): Path to the model checkpoint/trained model. For `SMTJServerless`, populated after job completion via `get_model_artifacts()`.
   - `output_s3_path` (str, Optional): Path to the metrics and output tar file. For MTRL jobs, fetched from the AgentRFT API.
   - `output_model_arn` (str, Optional): Model package ARN for `SMTJServerless` jobs. Use as `model_arn` in `ForgeTrainer` for iterative training.
 - `model_type` (Model): Model type of the model being trained

**Raises:**
- `Exception`: If job execution fails
- `ValueError`: If training method is not supported

**Example:**
```python
result = customizer.train(
 job_name="my-training-job",
 overrides={
 'max_epochs': 10,
 'lr': 5e-6,
 'warmup_steps': 20,
 'global_batch_size': 128
 }
)
print(f"Training job started: {result.job_id}")
print(f"Checkpoint path: {result.model_artifacts.checkpoint_s3_path}")
```
---

#### `get_config()`
Returns the overridable training parameters. Delegates to `ForgeTrainer.get_config()`.

> **Deprecated**: Use `ForgeTrainer.get_config()` instead.

**Signature:**
```python
def get_config(
 self,
 overrides: Optional[Dict[str, Any]] = None,
) -> RecipeConfig
```

**Parameters:**
- `overrides` (Optional[Dict[str, Any]]): Optional overrides to merge with defaults

**Returns:**
- `RecipeConfig`: Frozen dataclass containing all overridable parameters with defaults and constraints. See `ForgeTrainer.get_config()` for details.

**Example:**
```python
config = customizer.get_config()
print(config)

# With overrides
config = customizer.get_config(overrides={"lr": 1e-5})
print(config.to_dict())
```
---
#### `evaluate()`
Generates the recipe YAML, configures runtime, and launches an evaluation job.

**Signature:**
```python
def evaluate(
 self,
 job_name: str,
 eval_task: EvaluationTask,
 model_path: Optional[str] = None,
 subtask: Optional[str] = None,
 data_s3_path: Optional[str] = None,
 recipe_path: Optional[str] = None,
 overrides: Optional[Dict[str, Any]] = None,
 processor: Optional[Dict[str, Any]] = None,
 rl_env: Optional[Dict[str, Any]] = None,
 dry_run: Optional[bool] = False,
 job_result: Optional[TrainingResult] = None
) -> EvaluationResult | None
```

**Parameters:**
- `job_name` (str): User-defined name for the evaluation job
- `eval_task` (EvaluationTask): The evaluation task to be performed (e.g., `EvaluationTask.MMLU`)
  - The list of available tasks can be found here: [AWS Documentation](https://docs.aws.amazon.com/sagemaker/latest/dg/nova-model-evaluation.html#nova-model-evaluation-benchmark)
- `model_path` (Optional[str]): S3 path for model to evaluate. If not provided, will attempt to extract from `job_result` or the customizer's most recent training job.
- `data_s3_path` (Optional[str]): S3 URI for the dataset. Only required for BYOD (Bring Your Own Data) evaluation tasks.
- `subtask` (Optional[str]): Subtask for evaluation (task-specific)
  - The list of available subtasks per task can be found here: [Subtasks](https://docs.aws.amazon.com/sagemaker/latest/dg/nova-model-evaluation.html#nova-model-evaluation-subtasks)
- `recipe_path` (Optional[str]): Path for a YAML recipe file (both S3 and local paths are accepted)
- `overrides` (Optional[Dict[str, Any]]): Dictionary of inference configuration overrides
  - `max_new_tokens` (int): Maximum tokens to generate
  - `top_k` (int): Top-k sampling parameter
  - `top_p` (float): Top-p (nucleus) sampling parameter
  - `temperature` (float): Temperature for sampling
- `processor` (Optional[Dict[str, Any]]): Optional, Bring Your Own Metrics/RFT lambda Configuration
- `rl_env` (Optional[Dict[str, Any]]): Optional, Bring your own reinforcement learning environment config. For `RFT_EVAL`, if either `processor` or `rl_env` is explicitly passed, it takes priority over `rft_lambda_arn` set on the `RuntimeManager`.
- `dry_run` (Optional[bool]): Actually starts a job if False, otherwise just performs validation.
- `job_result` (Optional[TrainingResult]): Optional TrainingResult object to extract checkpoint path from. If provided and `model_path` is None, will automatically extract the checkpoint path from the training job's output and validate platform compatibility.

**Returns:**
- `EvaluationResult(BaseJobResult)`: Metadata object (either `SMTJEvaluationResult`, `SMHPEvaluationResult`, or `BedrockEvaluationResult`) containing:
  - `job_id` (str): The evaluation job identifier
  - `started_time` (datetime): Job start timestamp
  - `eval_output_path` (str): S3 path to evaluation results
  - `eval_task` (EvaluationTask): The Evaluation task
- Returns `None` if `dry_run=True`

**Example:**

```python
from amzn_nova_forge.recipe import *

# General eval task (with overrides)
eval_result = customizer.evaluate(
    job_name="my-eval-job",
    eval_task=EvaluationTask.MMLU,
    model_path="s3://my-bucket/checkpoints/my-model",
    overrides={
        'max_new_tokens': 2048,
        'temperature': 0,
        'top_p': 1.0
    }
)
print(f"Evaluation job started: {eval_result.job_id}")

# BYOM eval task (by providing processor config)
byom_eval_result = customizer.evaluate(
    job_name='my-eval-test-byom',
    eval_task=EvaluationTask.GEN_QA,
    data_s3_path="s3://bucket/data",
    processor={
        "lambda_arn": "arn:aws:lambda:<region>:123456789012:function:byom-lambda"
    }
)
```
---
#### `deploy()`
Creates a custom model and deploys it to Amazon Bedrock or SageMaker.

Deployment behavior when endpoint already exists is controlled by the `deployment_mode`
parameter set during NovaModelCustomizer initialization:
- FAIL_IF_EXISTS: Raise error (default, safest)
- UPDATE_IF_EXISTS: Try in-place update, fail if not supported (PT only)

**Signature:**
```python
def deploy(
  self,
  model_artifact_path: Optional[str] = None,
  deploy_platform: DeployPlatform = DeployPlatform.BEDROCK_OD,
  unit_count: Optional[int] = None,
  endpoint_name: Optional[str] = None,
  job_result: Optional[TrainingResult] = None,
  execution_role_name: Optional[str] = None,
  sagemaker_instance_type: Optional[str] = "ml.p5.48xlarge",
  sagemaker_environment: Optional[SageMakerEndpointEnvironment] = None,
) -> DeploymentResult
```

* **Note:** If DeployPlatform.BEDROCK_PT or DeployPlatform.SAGEMAKER is selected, you must include a value for unit_count.
* **Note:** If `model_artifact_path` is provided, we will NOT attempt to resolve `model_artifact_path` from `job_result` or the enclosing `NovaModelCustomizer` object.

**Parameters:**
- `model_artifact_path` (Optional[str]): S3 path to the trained model checkpoint, or a SageMaker Model Package name/ARN. Model package ARNs are auto-detected for SageMaker deployments and passed via `ModelPackageName`. If not provided, will attempt to extract from job_result or the `job_id` field of the Customizer.
- `deploy_platform` (DeployPlatform): Platform to deploy the model to
 - `DeployPlatform.BEDROCK_OD`: Bedrock On-Demand
 - `DeployPlatform.BEDROCK_PT`: Bedrock Provisioned Throughput
 - `DeployPlatform.SAGEMAKER`: SageMaker 
- `unit_count` (Optional[int]): Used in Bedrock Provisioned Throughput number of PT to purchase or SageMaker number of initial instances
- `endpoint_name` (Optional[str]): Name of the deployed model's endpoint (auto-generated if not provided)
- `job_result` (Optional[TrainingResult]): Training job result object to use for extracting checkpoint path and validating job completion. Also used to retrieve job_id if it's not provided.
- `execution_role_name` (Optional[str]): IAM role for Bedrock or SageMaker. If provided, used as-is — no policies created or attached. If omitted, the SDK creates and manages a default role with required policies.
- `sagemaker_instance_type`: Optional EC2 instance type for SageMaker deployment, defaults to ml.p5.48xlarge
- `sagemaker_environment` (Optional[SageMakerEndpointEnvironment]): SageMaker endpoint environment config. See `SageMakerEndpointEnvironment` for available fields and defaults.
**Returns:**
- `DeploymentResult`: Contains:
 - `endpoint` (EndpointInfo): Endpoint information
 - `platform` (DeployPlatform): Deployment platform
 - `endpoint_name` (str): Endpoint name
 - `uri` (str): Model ARN
 - `model_artifact_path` (str): S3 path to artifacts
 - `created_at` (datetime): Deployment creation timestamp

**Raises:**
- `Exception`: When unable to successfully deploy the model
- `ValueError`: If platform is not supported

**Example:**
```python
from amzn_nova_forge.model import *

bedrock_deployment = customizer.deploy(
 model_artifact_path="s3://escrow-bucket/my-model-artifacts/",
 deploy_platform=DeployPlatform.BEDROCK_OD,
 endpoint_name="my-custom-nova-model-bedrock"
)
print(f"Model deployed: {bedrock_deployment.endpoint.uri}")
print(f"Endpoint: {bedrock_deployment.endpoint.endpoint_name}")
print(f"Status: {bedrock_deployment.status}")

sagemaker_deployment = customizer.deploy(
 model_artifact_path="s3://escrow-bucket/my-model-artifacts/",
 deploy_platform=DeployPlatform.SAGEMAKER,
 unit_count=1,
 endpoint_name="my-custom-nova-model-sagemaker",
 sagemaker_environment=SageMakerEndpointEnvironment(
   context_length=8000,
   max_concurrency=4,
 )
)
print(f"Model deployed: {sagemaker_deployment.endpoint.uri}")
print(f"Endpoint: {sagemaker_deployment.endpoint.endpoint_name}")
print(f"Status: {sagemaker_deployment.status}")

# Deploy to SageMaker using a model package ARN (auto-detected)
sagemaker_mp_deployment = customizer.deploy(
 model_artifact_path="arn:aws:sagemaker:us-east-1:123456789012:model-package/my-group/1",
 deploy_platform=DeployPlatform.SAGEMAKER,
 unit_count=1,
 endpoint_name="my-model-package-endpoint",
)
print(f"Model deployed: {sagemaker_mp_deployment.endpoint.uri}")
```

Optionally, you can provide a Bedrock execution role name to be used in deployment.
Otherwise, a default Bedrock execution role will be created on your behalf.
You can also use the following method to create a Bedrock execution role with scoped down IAM permissions.
 
 
```python
from amzn_nova_forge.iam import create_bedrock_execution_role
 
iam_client = boto3.client("iam")
 
create_bedrock_execution_role(
    iam_client=iam_client, 
    role_name="BedrockDeployModelExecutionRole",
    bedrock_resource="your-model-name", # Optional: Name of the bedrock resources that IAM role should have restricted create and get access to
    s3_resource="s3-bucket" # Optional: S3 resource that IAM role should have restricted read access to such as the training output bucket
)
 
```
---
#### `create_custom_model()`
Creates a Bedrock custom model from S3 artifacts or a model package ARN, decoupled from endpoint deployment.

This method extracts the model-creation step from the deploy flow so users can
create a model independently of endpoint deployment, enabling retry of deployment
if it fails after model creation.

Either `model_artifact_path` (maps to `modelSourceConfig`) or `custom_model_data_source`
(maps to `customModelDataSource`) must be provided, but not both. If neither is provided,
the method attempts to resolve from `job_result`.

**Signature:**
```python
def create_custom_model(
  self,
  model_artifact_path: Optional[str] = None,
  job_result: Optional[TrainingResult] = None,
  endpoint_name: Optional[str] = None,
  execution_role_name: Optional[str] = None,
  tags: Optional[List[Dict[str, str]]] = None,
  skip_model_reuse: bool = False,
  custom_model_data_source: Optional[Dict[str, Any]] = None,
) -> ModelDeployResult
```

**Parameters:**
- `model_artifact_path` (Optional[str]): S3 path to trained model checkpoint. Used to populate `modelSourceConfig.s3DataSource.s3Uri`. Takes precedence over `job_result` if both are provided.
- `job_result` (Optional[TrainingResult]): Training job result to extract checkpoint path from.
- `endpoint_name` (Optional[str]): Optional name prefix for the model name (auto-generated if not provided).
- `execution_role_name` (Optional[str]): IAM role name for Bedrock. Defaults to `BedrockDeployModelExecutionRole`.
- `tags` (Optional[List[Dict[str, str]]]): Optional list of `{"key": str, "value": str}` dicts for source tracking.
- `skip_model_reuse` (bool): If True, always create a new model even if one with the same escrow URI already exists. Default: False.
- `custom_model_data_source` (Optional[Dict[str, Any]]): Alternative data source configuration for the custom model. When provided, `modelSourceConfig` is omitted from the API call.

**Returns:**
- `ModelDeployResult`: Contains:
  - `model_arn` (str): The Bedrock custom model ARN
  - `model_name` (str): The model name passed to CreateCustomModel
  - `escrow_uri` (str): S3 artifacts path or model package ARN used to create the model
  - `created_at` (datetime): UTC timestamp when the model was created

**Raises:**
- `ValueError`: When neither `model_artifact_path`, `job_result`, nor `custom_model_data_source` is provided, or when both `model_artifact_path` and `custom_model_data_source` are provided.
- `RuntimeError`: When IAM role creation or custom model creation fails.

**Example:**
```python
# Create a custom model from training artifacts
publish_result = customizer.create_custom_model(
  model_artifact_path="s3://escrow-bucket/my-model-artifacts/"
)
print(f"Model ARN: {publish_result.model_arn}")

# Save for later use
publish_result.dump(file_path="./results/")

# Or create from a training job result
publish_result = customizer.create_custom_model(job_result=training_result)

# Or create from a model package ARN
publish_result = customizer.create_custom_model(
  custom_model_data_source={
      "modelPackageArnDataSource": {
          "modelPackageArn": "arn:aws:sagemaker:us-east-1:123456789012:model-package/my-group/1"
      }
  }
)
print(f"Model ARN: {publish_result.model_arn}")
```
---
#### `deploy_to_bedrock()`
Deploys a published Bedrock custom model to an endpoint.

Use after `create_custom_model()` to deploy the model, or provide a model ARN directly.
This decoupled approach allows retrying deployment without re-creating the model.

When a `model_deploy_result` is provided, the model's status is validated before deployment:
- **Active**: Proceeds immediately.
- **Creating**: Waits for the model to become Active (30s poll interval, 15min timeout).
- **Failed**: Raises `ValueError`.

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
- `model_deploy_result` (Optional[ModelDeployResult]): Result from `create_custom_model()`. Cannot be combined with `model_arn`.
- `model_arn` (Optional[str]): Direct model ARN. Cannot be combined with `model_deploy_result`.
- `deploy_platform` (DeployPlatform): `BEDROCK_OD` (default) or `BEDROCK_PT`.
- `pt_units` (Optional[int]): Number of PT units (required for `BEDROCK_PT`).
- `endpoint_name` (Optional[str]): Endpoint name (auto-generated if not provided).

**Returns:**
- `DeploymentResult`: Contains:
  - `endpoint` (EndpointInfo): Endpoint information
  - `created_at` (datetime): Deployment creation timestamp
  - `model_publish` (Optional[ModelDeployResult]): The model deploy result, if available
  - `escrow_uri` (Optional[str]): Convenience property delegating to `model_publish.escrow_uri`

**Raises:**
- `ValueError`: When both `model_deploy_result` and `model_arn` are provided, or when no model ARN is available.
- `RuntimeError`: When deployment creation fails.

**Example:**
```python
# Two-step deploy: create model, then deploy
publish_result = customizer.create_custom_model(
  model_artifact_path="s3://escrow-bucket/my-model-artifacts/"
)
deployment = customizer.deploy_to_bedrock(
  model_deploy_result=publish_result,
  endpoint_name="my-endpoint"
)

# Or deploy directly from a model ARN
deployment = customizer.deploy_to_bedrock(
  model_arn="arn:aws:bedrock:us-east-1:123456789012:custom-model/my-model"
)

# Retry deployment if it fails (model already created)
deployment = customizer.deploy_to_bedrock(
  model_arn=publish_result.model_arn
)
```
---
#### `deploy_to_sagemaker()`
Deploys a model to a SageMaker Inference endpoint.

Can reuse an existing SageMaker model via `model_deploy_result` (e.g., from a
previous deploy that created the model but failed on endpoint creation),
or create a new model from `model_artifact_path`.

**Signature:**
```python
def deploy_to_sagemaker(
  self,
  instance_type: str,
  model_deploy_result: Optional[ModelDeployResult] = None,
  model_artifact_path: Optional[str] = None,
  unit_count: int = 1,
  endpoint_name: Optional[str] = None,
  sagemaker_environment: Optional[SageMakerEndpointEnvironment] = None,
  execution_role_name: Optional[str] = None,
  skip_model_reuse: bool = False,
) -> DeploymentResult
```

**Parameters:**
- `instance_type` (str): SageMaker instance type (required).
- `model_deploy_result` (Optional[ModelDeployResult]): Result from a previous deploy containing the SM model ARN. Skips model creation. Cannot be combined with `model_artifact_path`.
- `model_artifact_path` (Optional[str]): S3 path to model artifacts. Creates a new SM model.
- `unit_count` (int): Number of instances. Default: 1.
- `endpoint_name` (Optional[str]): Endpoint name (auto-generated if not provided).
- `sagemaker_environment` (Optional[SageMakerEndpointEnvironment]): SageMaker endpoint environment config. See `SageMakerEndpointEnvironment` for available fields and defaults.
- `execution_role_name` (str): IAM execution role name.
- `skip_model_reuse` (bool): If True, always create a new model (skip tag-based discovery). Default: False.

**Returns:**
- `DeploymentResult`: Contains endpoint info and `model_publish` (ModelDeployResult).

**Raises:**
- `ValueError`: When both `model_deploy_result` and `model_artifact_path` are provided, when neither is provided, or when `instance_type` is missing.
- `RuntimeError`: When endpoint creation fails. The error message includes the model ARN and a retry command using `customizer.last_model_publish`.

**Example:**
```python
# Deploy from S3 artifacts
result = customizer.deploy_to_sagemaker(
  model_artifact_path="s3://escrow-bucket/checkpoint/",
  instance_type="ml.g5.12xlarge",
)

# Retry after endpoint failure (model already created)
result = customizer.deploy_to_sagemaker(
  model_deploy_result=customizer.last_model_publish,
  instance_type="ml.g5.12xlarge",
  endpoint_name="my-endpoint",
)
```
---

#### `invoke_inference()`
Invokes a single inference on a trained model.

**Signature:**
```python
def invoke_inference(
 self,
 request_body: Dict[str, Any], 
 endpoint_arn: Optional[str]
) -> InferenceResult
```
**Parameters:**
- `request_body` (Dict[str, Any]): Inference request body
- `endpoint_arn` (Optional[str]):Endpoint ARN to invoke inference. Optional if user wants to send request to an already deployed endpoint on customizer

**Returns:**
- `InferenceResult`: Metadata object (`SingleInferenceResult`) containing:
 - `job_id` (str): Batch inference job identifier
 - `started_time` (datetime): Job start timestamp
 - `inference_output_path` (str): Empty string

**Example:**
```python
inference_result = customizer.invoke_inference(
    request_body={
      "messages": [{"role": "user", "content": "Hello! How are you?"}],
      "max_tokens": 100,
      "stream": False,
    },
    endpoint_arn="arn:aws:sagemaker:us-east-1:123456789012:endpoint/endpoint",
)
inference_result.show()
```
---
#### `batch_inference()`
Launches a batch inference job on a trained model.

**Signature:**
```python
def batch_inference(
 self,
 job_name: str,
 input_path: str,
 output_s3_path: str,
 model_path: Optional[str] = None,
 recipe_path: Optional[str] = None,
 overrides: Optional[Dict[str, Any]] = None,
 dry_run: Optional[bool] = False
) -> InferenceResult
```
**Parameters:**
- `job_name` (str): Name for the batch inference job
- `input_path` (str): S3 path to input data for inference
- `output_s3_path` (str): S3 path for inference outputs
- `model_path` (Optional[str]): S3 path to the model
- `recipe_path` (Optional[str]): Path for a YAML recipe file
- `overrides` (Optional[Dict[str, Any]]): Configuration overrides for inference
 - `max_new_tokens` (int): Maximum tokens to generate
 - `top_k` (int): Top-k sampling parameter
 - `top_p` (float): Top-p (nucleus) sampling parameter
 - `temperature` (float): Temperature for sampling
 - `top_logprobs` (int): Number of top log probabilities to return
- `dry_run` (Optional[bool]): Actually starts a job if False, otherwise just performs validation.

**Returns:**
- `InferenceResult`: Metadata object (`SMTJBatchInferenceResult`) containing:
 - `job_id` (str): Batch inference job identifier
 - `started_time` (datetime): Job start timestamp
 - `inference_output_path` (str): S3 path to inference results
 - Note: Batch inference is only supported on SageMaker platforms (SMTJ, SMHP)

**Example:**
```python
inference_result = customizer.batch_inference(
 job_name="batch-inference-job",
 input_path="s3://my-bucket/inference-input",
 output_s3_path="s3://my-bucket/inference-output",
 model_path="s3://my-bucket/trained-model"
)
print(f"Batch inference started: {inference_result.job_id}")
```
In a separate notebook cell, you can run the following commands to get the job status and download a formatted result file when the jobs completes.
```python
inference_result.get_job_status() # Gets the job status.
inference_result.get("s3://my-bucket/save-location/file-name.jsonl") # Uploads a formatted inference_results.jsonl file to the given s3 location.
```
---
#### `get_logs()`
Retrieves and displays CloudWatch logs for the current job.

**Signature:**
```python
def get_logs(
 self,
 limit: Optional[int] = None,
 start_from_head: bool = False,
 end_time: Optional[int] = None
)
```

**Parameters:**
- `limit` (Optional[int]): Maximum number of log lines to retrieve
- `start_from_head` (bool): If True, start from the beginning of logs; if False, start from the end
- `end_time` (Optional[int]): End time in epoch milliseconds for searching a log time range

**Returns:**
- None (prints logs to console)

**Example:**
```python
# After starting a training job
customizer.train(job_name="my-job")
customizer.get_logs(limit=100, start_from_head=True)
```
---
