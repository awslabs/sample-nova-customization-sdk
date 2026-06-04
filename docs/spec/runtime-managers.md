# Runtime Managers
Runtime managers handle the infrastructure for executing training and evaluation jobs,
leveraging the `JobConfig` dataclass to do so:
```python
@dataclass
class JobConfig:
    job_name: str
    image_uri: str
    recipe_path: str
    output_s3_path: Optional[str] = None
    data_s3_path: Optional[str] = None
    input_s3_data_type: Optional[str] = None
    mlflow_tracking_uri: Optional[str] = None
    mlflow_experiment_name: Optional[str] = None
    mlflow_run_name: Optional[str] = None
```
* The specific instance types that can be used with the runtime managers (SMTJ, SMHP) can be found in `docs/user-guides/instance_type_spec.md`.
* This file also defines which instance types can be used with a specific model and method.
* Bedrock is fully managed and does not require instance type configuration.

### Shared RuntimeManager Methods

The following methods are available on all `RuntimeManager` subclasses.

#### Properties (shared)
- `rft_lambda` (Optional[str]): Lambda ARN, SageMaker hub-content ARN (SMTJServerless only), or local `.py` file path. Assigning a new value automatically updates `rft_lambda_arn` — if the value is a Lambda ARN or hub-content ARN it is resolved immediately; if it is a file path, `rft_lambda_arn` is cleared until `deploy_lambda()` is called.
- `rft_lambda_arn` (Optional[str]): Resolved Lambda ARN or hub-content ARN. Set immediately when `rft_lambda` is assigned an ARN (Lambda or hub-content), or populated by `deploy_lambda()` when `rft_lambda` is a file path.

**Example:**
```python
# Set an ARN directly — rft_lambda_arn is updated immediately
runtime.rft_lambda = 'arn:aws:lambda:us-east-1:123456789012:function:my-reward-fn'
print(runtime.rft_lambda_arn)  # 'arn:aws:lambda:us-east-1:123456789012:function:my-reward-fn'

# Set a file path — rft_lambda_arn is cleared until deploy_lambda() is called
runtime.rft_lambda = 'reward.py'
print(runtime.rft_lambda_arn)  # None
runtime.deploy_lambda(lambda_name='my-reward-fn')
print(runtime.rft_lambda_arn)
#'arn:aws:lambda:us-east-1:123456789012:function:my-reward-fn'
```

---

#### `deploy_lambda()`

Packages a local Python file into a zip and creates or updates a Lambda function. The source file is read from `self.rft_lambda`, which must be set to a local `.py` file path before calling this method.

**Signature:**
```python
def deploy_lambda(
    self,
    lambda_name: Optional[str] = None,
    execution_role_arn: Optional[str] = None,
) -> str
```

**Parameters:**
- `lambda_name` (Optional[str]): Name for the Lambda function. Defaults to the source filename stem (underscores replaced with hyphens).
- `execution_role_arn` (Optional[str]): IAM role ARN for the Lambda. Falls back to the runtime manager's `execution_role` attribute if not provided.

**Returns:**
- `str`: The deployed Lambda function ARN. Also sets `self.rft_lambda_arn` on the manager.

**Raises:**
- `ValueError`: If `rft_lambda` is not set, is already an ARN (nothing to deploy), the source file is not found, or no execution role can be resolved.

**Example:**
```python
runtime.rft_lambda = 'rft_training_reward.py'
lambda_arn = runtime.deploy_lambda(lambda_name='my-reward-fn')
# runtime.rft_lambda_arn is now set automatically
```

---

#### `validate_lambda()`

Validates the RFT reward lambda with sample data from S3. Reads the lambda to validate from `self.rft_lambda` / `self.rft_lambda_arn`:
- If `rft_lambda` is an ARN (or `rft_lambda_arn` is set), invokes the deployed Lambda with samples from `data_s3_path`.
- If `rft_lambda` is a local `.py` path, validates by executing `lambda_handler` directly without deploying.

**Signature:**
```python
def validate_lambda(
    self,
    data_s3_path: str,
    validation_samples: int = 10,
) -> None
```

**Parameters:**
- `data_s3_path` (str): S3 path to the training dataset for pulling sample data.
- `validation_samples` (int): Number of samples to load from `data_s3_path` (default: 10).

**Raises:**
- `ValueError`: If `rft_lambda` is not set, or if validation fails.

**Example:**
```python
# Validate a local file without deploying
runtime.rft_lambda = 'rft_training_reward.py'
runtime.validate_lambda(data_s3_path='s3://bucket/data.jsonl')

# Validate a deployed lambda
runtime.rft_lambda = 'arn:aws:lambda:us-east-1:123456789012:function:my-reward-fn'
runtime.validate_lambda(data_s3_path='s3://bucket/data.jsonl', validation_samples=20)
```

---

### SMTJRuntimeManager
Manages SageMaker Training Jobs.

#### Constructor

**Signature:**
```python
def __init__(
    self,
    instance_type: str,
    instance_count: int,
    execution_role: Optional[str] = None,
    kms_key_id: Optional[str] = None,
    encrypt_inter_container_traffic: bool = False,
    subnets: Optional[list[str]] = None,
    security_group_ids: Optional[list[str]] = None,
    max_job_runtime: Optional[int] = 86400,
    job_submit_poll_timeout: int = 30,
    rft_lambda: Optional[str] = None,
)
```

**Parameters:**
- `instance_type` (str): EC2 instance type (e.g., "ml.p5.48xlarge", "ml.p4d.24xlarge")
- `instance_count` (int): Number of instances to use
- `execution_role` (Optional[str]): The execution role for the training job
- `kms_key_id` (Optional[str]): Optional KMS Key Id to use in S3 Bucket encryption, training jobs and deployments.
- `encrypt_inter_container_traffic` (bool): Boolean that determines whether to encrypt inter-container traffic. Default value is False.
- `subnets` (Optional[list[str]]): Optional list of strings representing subnets. Default value is None.
- `security_group_ids` (Optional[list[str]]): Optional list of strings representing security group IDs. Default value is None.
- `max_job_runtime` (Optional[int]): Max Job Runtime in seconds (default: 1 day)
- `job_submit_poll_timeout` (int): Maximum seconds to poll for the training job after submission (default: 30). Uses exponential backoff.
- `rft_lambda` (Optional[str]): Lambda ARN or local `.py` file path for RFT reward function. Can also be set or updated after construction.

**Example:**
```python
from amzn_nova_forge.manager import *
infra = SMTJRuntimeManager(
 instance_type="ml.p5.48xlarge",
 instance_count=2
)
```
#### Properties
- `instance_type` (str): Returns the instance type
- `instance_count` (int): Returns the number of instances

#### Methods

##### `execute()`
Starts a SageMaker training job.

**Signature:**
```python
def execute(
 self,
 job_config: JobConfig
) -> str
```

**Returns:**
- `str`: Training job name/ID

##### `cleanup()`
Stops and cleans up a training job.

**Signature:**
```python
def cleanup(
 self,
 job_name: str
) -> None
```
---
### SMHPRuntimeManager
Manages SageMaker HyperPod jobs.

#### Constructor

**Signature:**
```python
def __init__(
 self,
 instance_type: str,
 instance_count: int,
 cluster_name: str,
 namespace: str,
 kms_key_id: Optional[str] = None,
 rft_lambda: Optional[str] = None,
)
```

**Parameters:**
- `instance_type` (str): EC2 instance type
- `instance_count` (int): Number of instances
- `cluster_name` (str): HyperPod cluster name
- `namespace` (str): Kubernetes namespace
- `kms_key_id` (Optional[str]): Optional KMS Key Id to use in S3 Bucket encryption
- `rft_lambda` (Optional[str]): Lambda ARN or local `.py` file path for RFT reward function. Can also be set or updated after construction.

**Example:**
```python
from amzn_nova_forge.manager import *
infra = SMHPRuntimeManager(
 instance_type="ml.p5.48xlarge",
 instance_count=4,
 cluster_name="my-hyperpod-cluster",
 namespace="default"
)
```
#### Properties
- `instance_type` (str): Returns the instance type
- `instance_count` (int): Returns the number of instances

#### Methods

##### `execute()`

Starts a SageMaker HyperPod job.
**Signature:**
```python
def execute(
 self,
 job_config=JobConfig
) -> str
```
**Returns:**
- `str`: HyperPod job ID

##### `cleanup()`
Cancels and cleans up a HyperPod job.

**Signature:**
```python
def cleanup(
 self,
 job_name: str
) -> None
```

##### `scale_cluster()`
Scale a HyperPod cluster instance group up or down. 
The scaling operation is asynchronous - the cluster status will change to 'Updating' while scaling, and 'InService' when ready.

**Signature:**
```python
def scale_cluster(
 self,
 instance_group_name: str,
 target_instance_count: int,
) -> Dict[str, Any]
```

**Parameters:**
- `instance_group_name` (str): Name of the instance group to scale (e.g., 'worker-group')
- `target_instance_count` (int): Desired number of instances for the group (must be non-negative)

**Returns:**
- `Dict[str, Any]`: Response containing:
  - `ClusterArn` (str): ARN of the updated cluster
  - `InstanceGroupName` (str): Name of the scaled instance group
  - `InstanceType` (str): Instance type being scaled
  - `PreviousCount` (int): Current instance count before scaling
  - `TargetCount` (int): Target instance count after scaling

**Raises:**
- `ValueError`: If target_instance_count is negative or instance group name is invalid
- `ClientError`: If scaling fails due to insufficient quota, capacity or other cluster issues.

**Example:**
```python
from amzn_nova_forge.manager import *

# Create a runtime manager for your cluster
manager = SMHPRuntimeManager(
    instance_type="ml.p4d.24xlarge",
    instance_count=4,
    cluster_name="my-hyperpod-cluster",
    namespace="default"
)

# Scale up the worker group from 4 to 8 instances
result = manager.scale_cluster(
    instance_group_name="worker-group",
    target_instance_count=8
)

# Scale down to 2 instances
result = manager.scale_cluster(
    instance_group_name="worker-group",
    target_instance_count=2
)
```
**Notes:**
- This method only works with Restricted Instance Groups (RIGs) in HyperPod clusters. The cluster must be in 'InService' state before scaling can be initiated.
- This method can only scale up a SMHP cluster when there is sufficient Service Quota available.
You will need to request a quota increase **before** scaling up a RIG in your HyperPod cluster.
You can learn more [here](https://docs.aws.amazon.com/servicequotas/latest/userguide/request-quota-increase.html).
  - Specifically, you will need to request a service quota increase for "INSTANCE_TYPE for cluster usage".

##### `get_instance_groups()`
Gets the RIGs associated with the current cluster defined in the SMHPRuntimeManager.
Prints the values to the terminal and returns it as a list of dictionary entries.

**Signature:**
```python
def get_instance_groups(
 self
) -> List[Dict[str, Any]]
```

**Returns:**
- `List[Dict[str, Any]]`: Response containing:
  - InstanceGroupName: Name of the instance group
  - InstanceType: EC2 instance type (e.g., 'ml.p5.48xlarge')
  - CurrentCount: Current number of instances in the group

**Raises:**
- `ClientError`: If unable to describe the cluster

**Example:**
```python
from amzn_nova_forge.manager import *

# Create a runtime manager for your cluster
manager = SMHPRuntimeManager(
    instance_type="ml.p4d.24xlarge",
    instance_count=4,
    cluster_name="my-hyperpod-cluster",
    namespace="default"
)

# Get the instance groups available on the current cluster.
instance_groups = manager.get_instance_groups()
```

---
### BedrockRuntimeManager
Manages Amazon Bedrock model customization jobs.

#### Constructor

**Signature:**
```python
def __init__(
 self,
 execution_role: str,
 base_model_identifier: Optional[str] = None,
 kms_key_id: Optional[str] = None,
 rft_lambda: Optional[str] = None,
)
```

**Parameters:**
- `execution_role` (str): IAM role ARN for Bedrock job execution
- `base_model_identifier` (Optional[str]): Base model ARN (e.g., "arn:aws:bedrock:us-east-1::foundation-model/amazon.nova-2-lite-v1:0:256k")
- `kms_key_id` (Optional[str]): Optional KMS Key Id for encryption
- `rft_lambda` (Optional[str]): Lambda ARN or local `.py` file path for RFT reward function. Can also be set or updated after construction.

**Example:**
```python
from amzn_nova_forge.manager import *
infra = BedrockRuntimeManager(
 execution_role="arn:aws:iam::123456789012:role/BedrockRole",
 base_model_identifier="arn:aws:bedrock:us-east-1::custom-model/amazon.nova-2-lite-v1:0:256k:abcdefghijk" # optional: your custom model ARN for iterative training
)
```

#### Methods

##### `execute()`

Starts a Bedrock model customization job.
**Signature:**
```python
def execute(
 self,
 job_config: JobConfig
) -> str
```
**Returns:**
- `str`: Bedrock job ARN

##### `cleanup()`
Stops a Bedrock customization job.

**Signature:**
```python
def cleanup(
 self,
 job_name: str
) -> None
```
---
### SMTJServerlessRuntimeManager
Manages SageMaker Training Jobs.

> **Note:** `AWS_DEFAULT_REGION` must be set when using SageMaker Serverless training.
> The SageMaker SDK's `DataSet` API requires a region to connect to the SageMaker backend.
> Set it before running your script:
> ```bash
> export AWS_DEFAULT_REGION=<your-region>
> ```

#### Constructor

**Signature:**
```python
def __init__(
    self,
    model_package_group_name: str,
    execution_role: Optional[str] = None,
    kms_key_id: Optional[str] = None,
    encrypt_inter_container_traffic: bool = False,
    subnets: Optional[list[str]] = None,
    security_group_ids: Optional[list[str]] = None,
    max_job_runtime: Optional[int] = 86400,
    rft_lambda: Optional[str] = None,
    agent_core_arn: Optional[str] = None,
    evaluator_name: Optional[str] = None,
    intermediate_model_package_group_name: Optional[str] = None,
)
```

**Parameters:**
- `model_package_group_name` (str): Model package group name to use with SageMaker Model registry (required for SMTJ Serverless)
- `execution_role` (Optional[str]): The execution role for the training job
- `kms_key_id` (Optional[str]): Optional KMS Key Id to use in S3 Bucket encryption, training jobs and deployments.
- `encrypt_inter_container_traffic` (bool): Boolean that determines whether to encrypt inter-container traffic. Default value is False.
- `subnets` (Optional[list[str]]): Optional list of strings representing subnets. Default value is None.
- `security_group_ids` (Optional[list[str]]): Optional list of strings representing security group IDs. Default value is None.
- `max_job_runtime` (Optional[int]): Max Job Runtime in seconds (default: 1 day)
- `rft_lambda` (Optional[str]): Lambda ARN, SageMaker hub-content ARN, or local `.py` file path for the RFT reward function. Used for single-turn RFT or as an alternative agent environment for MTRL on serverless.
  - **Lambda ARN**: Automatically registered as a hub-content `JsonDoc` evaluator during `train()`. The hub-content ARN is passed as `EvaluatorArn` in `ServerlessJobConfig`.
  - **Hub-content ARN**: Passed directly as `EvaluatorArn` — no registration needed.
  - **Local `.py` file**: Call `deploy_lambda()` first to deploy and get a Lambda ARN.
- `agent_core_arn` (Optional[str]): Bedrock AgentCore runtime ARN for MTRL training. If both `agent_core_arn` and `rft_lambda` are set, `agent_core_arn` takes priority as the agent environment.
- `evaluator_name` (Optional[str]): Optional human-readable name for the hub-content evaluator entry when auto-registering a Lambda ARN. Defaults to the Lambda function name.
- `intermediate_model_package_group_name` (Optional[str]): Model package group name for intermediate checkpoints during MTRL training. If not set, defaults to `{model_package_group_name}-checkpoints`.

**Example:**
```python
from amzn_nova_forge.manager import *

# With Bedrock AgentCore (for MTRL training)
infra = SMTJServerlessRuntimeManager(
    model_package_group_name="nova-rft-serverless",
    execution_role="arn:aws:iam::123456789012:role/my-role",
    agent_core_arn="arn:aws:bedrock-agentcore:us-east-1:123456789012:runtime/my-agent",
)

# With a Lambda ARN (auto-registered as hub-content during train())
infra = SMTJServerlessRuntimeManager(
    model_package_group_name="nova-rft-serverless",
    rft_lambda="arn:aws:lambda:us-east-1:123456789012:function:my-reward-fn",
)

# With a hub-content ARN (passed directly as EvaluatorArn)
infra = SMTJServerlessRuntimeManager(
    model_package_group_name="nova-rft-serverless",
    rft_lambda="arn:aws:sagemaker:us-east-1:123456789012:hub-content/my-hub/JsonDoc/my-evaluator/0.0.1",
)

# With a local .py file (deploy_lambda() required before train())
infra = SMTJServerlessRuntimeManager(
    model_package_group_name="nova-rft-serverless",
    rft_lambda="my_reward.py",
)
infra.deploy_lambda(lambda_name="my-reward-fn")
```
#### Properties
- `model_package_group_name` (str): Model Package Group name

#### Methods

##### `execute()`
Starts a SageMaker training job.

**Signature:**
```python
def execute(
 self,
 job_config: JobConfig
) -> str
```

**Returns:**
- `str`: Training job name/ID

##### `cleanup()`
Stops and cleans up a training job.

**Signature:**
```python
def cleanup(
 self,
 job_name: str
) -> None
```
---

