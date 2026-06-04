# RFT Multiturn Infrastructure

For RFT multiturn training and evaluation, you need to set up infrastructure to run reward functions.

### Helper Functions

#### create_rft_execution_role

Creates an IAM role with required permissions for RFT multiturn infrastructure.

**Function:**
```python
def create_rft_execution_role(
    region: str = "us-east-1",
    role_name: Optional[str] = None,
    custom_policy_path: Optional[str] = None
) -> str
```

**Parameters:**
- `region` (str): AWS region. Default: "us-east-1"
- `role_name` (Optional[str]): Custom role name. Default: "RFTExecutionRoleNovaSDK"
- `custom_policy_path` (Optional[str]): Path to custom policy JSON file. If not provided, uses SDK default.

**Returns:**
- `str`: ARN of the created/existing role

**Example:**
```python
from amzn_nova_forge import create_rft_execution_role

# Create role with default name
role_arn = create_rft_execution_role(region="us-east-1")

# Create role with custom name
role_arn = create_rft_execution_role(region="us-east-1", role_name="my-custom-rft-role")
```

#### list_rft_stacks

Lists CloudFormation stacks in the region, optionally filtering for Nova SDK stacks.

**Function:**
```python
def list_rft_stacks(
    region: str = "us-east-1",
    all_stacks: bool = False
) -> List[str]
```

**Parameters:**
- `region` (str): AWS region. Default: "us-east-1"
- `all_stacks` (bool): If True, list all stacks. If False, only list Nova SDK stacks (ending with "NovaForgeSDK"). Default: False

**Returns:**
- `List[str]`: List of stack names

**Example:**
```python
from amzn_nova_forge import list_rft_stacks

# List only Nova SDK stacks
nova_stacks = list_rft_stacks(region="us-east-1")

# List all CloudFormation stacks
all_stacks = list_rft_stacks(region="us-east-1", all_stacks=True)
```

### RFTMultiturnInfrastructure

Manages infrastructure for RFT multiturn training (reward function workers).

**Constructor:**
```python
def __init__(
    self,
    stack_name: str,
    region: str = "us-east-1",
    vf_env_id: Optional[VFEnvId] = None,
    custom_env: Optional[CustomEnvironment] = None,
    infrastructure_arn: Optional[str] = None,
    python_venv_name: Optional[str] = None,
    vpc_config: Optional[Dict[str, Any]] = None,
    cpu: Optional[str] = None,
    memory: Optional[str] = None,
    rft_role_name: Optional[str] = None,
)
```

**Parameters:**
- `stack_name` (str): CloudFormation stack name
- `region` (str): AWS region. Default: "us-east-1"
- `vf_env_id` (Optional[VFEnvId]): Built-in environment ID (VFEnvId.WORDLE or VFEnvId.TERMINAL_BENCH)
- `custom_env` (Optional[CustomEnvironment]): Custom environment (mutually exclusive with vf_env_id)
- `infrastructure_arn` (Optional[str]): Platform ARN (EC2 instance ID, ECS cluster ARN, or None for LOCAL)
- `python_venv_name` (Optional[str]): Python virtual environment name (required for LOCAL/EC2, optional for ECS)
- `vpc_config` (Optional[Dict]): VPC configuration for ECS only. Dict with keys:
  - `subnets`: List[str] - Subnet IDs
  - `security_groups`: List[str] - Security group IDs
- `cpu` (Optional[str]): CPU units for ECS tasks (e.g., "2048"). Ignored for LOCAL/EC2.
- `memory` (Optional[str]): Memory in MB for ECS tasks (e.g., "4096"). Ignored for LOCAL/EC2.
- `rft_role_name` (Optional[str]): IAM role name for RFT infrastructure. If not provided, uses default role or creates one.

**Example:**
```python
from amzn_nova_forge import RFTMultiturnInfrastructure, CustomEnvironment, VFEnvId

# Option 1: LOCAL with built-in environment
rft_infra = RFTMultiturnInfrastructure(
    stack_name="my-rft-stack",
    region="us-east-1",
    python_venv_name="my_rft_venv",
    vf_env_id=VFEnvId.WORDLE
)

# Option 2: ECS with custom environment and VPC config
custom_env = CustomEnvironment(
    env_id="my-custom-env", 
    output_dir="~/custom_envs/", 
    env_type="single_turn"
).create(overwrite=True)

rft_infra = RFTMultiturnInfrastructure(
    stack_name="my-rft-stack",
    custom_env=custom_env,
    infrastructure_arn="arn:aws:ecs:us-east-1:123456789012:cluster/my-cluster",
    vpc_config={
        "subnets": ["subnet-12345", "subnet-67890"],
        "security_groups": ["sg-12345"]
    },
    cpu="4096",
    memory="8192"
)

# Deploy infrastructure
rft_infra.setup()

# Start training environment
rft_infra.start_training_environment()

# Use with NovaModelCustomizer
customizer = NovaModelCustomizer(
    model=Model.NOVA_LITE_2,
    method=TrainingMethod.RFT_MULTITURN_LORA,
    infra=runtime,
    data_s3_path="s3://bucket/data.jsonl"
)

training_result = customizer.train(
    job_name="rft-training",
    rft_multiturn_infra=rft_infra
)
```

### CustomEnvironment

Create custom reward functions for RFT multiturn training.

**Constructor:**
```python
def __init__(
    self,
    env_id: str,
    local_path: str = None,
    output_dir: str =  "~/custom_envs",
    env_type: str = "single_turn"
)
```

**Methods:**
- `create(overwrite: bool = False)`: Create environment structure
- `package_and_upload(bucket: Optional[str] = None)`: Upload to S3

**Example:**
```python
custom_env = CustomEnvironment(
    env_id="my-custom-env",
    output_dir="~/custom_envs/",
    env_type="single_turn"
).create(overwrite=True)

custom_env.validate()
custom_env.package_and_upload()
print(f"Uploaded to: {custom_env.s3_uri}")
```

### RFT Multiturn Methods

**Infrastructure Management:**
- `setup()`: Deploy CloudFormation stack (Lambda, SQS, DynamoDB)
- `start_training_environment(vf_env_args: Dict = None)`: Start training workers
- `start_evaluation_environment(vf_env_args: Dict = None)`: Start evaluation workers
- `kill_task(env_type: EnvType)`: Stop workers
- `cleanup(delete_stack: bool = False, cleanup_environment: bool = False)`: Clean up resources
  - `delete_stack`: If True, delete CloudFormation stack
  - `cleanup_environment`: If True, clean up environment resources:
    - LOCAL/EC2: Delete virtual environment and starter kit directories
    - ECS: Deregister task definitions

**Monitoring:**
- `get_logs(env_type: EnvType, limit: int = 100, start_from_head: bool = False, log_stream_name: Optional[str] = None, tail: bool = False)`: View worker logs
  - `tail`: If True, continuously stream logs in real-time (blocks until Ctrl+C)
- `check_all_queues()`: Check SQS queue status
- `flush_all_queues()`: Clear all queues

**Configuration:**
- `get_configuration()`: Get infrastructure config
- `get_recipe_overrides()`: Get recipe overrides for training

**Note:** RFT multiturn supports SageMaker HyperPod (SMHP) and SMTJServerless platforms with Nova 2.0 models (NOVA_LITE_2).

---

### Platform Requirements

#### MLFlow Required for MTRL on SMTJServerless

MTRL training and evaluation on `SMTJServerless` require `ForgeConfig.mlflow_monitor` with a valid `tracking_uri` — raises `ValueError` if missing.

MLFlow is optional on SMHP.

---
