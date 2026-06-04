# Amazon Nova Forge SDK

A comprehensive Python SDK for fine-tuning and customizing Amazon Nova models. This SDK provides a unified interface for training, evaluation, deployment, and monitoring of Nova models across both SageMaker Training Jobs and SageMaker HyperPod.

## Table of Contents

- [Installation](#installation)
- [Setup](#setup)
- [Supported Models and Training Methods](#supported-models-and-training-methods)
- [Data Preparation](#data-preparation)
- [Core Modules Overview](#core-modules-overview)
- [Additional Features](#additional-features)
- [Telemetry](#telemetry)
- [Getting Started](#getting-started)
- [Security Best Practices for SDK Users](#security-best-practices-for-sdk-users)

## Installation

```bash
pip install amzn-nova-forge
```
* The SDK requires [sagemaker](https://pypi.org/project/sagemaker/), which is automatically set by pip.


## Setup

In most cases, the SDK will inform you if the environment lacks the required setup to run a Nova customization job.

Below are some common requirements which you can set up in advance before trying to run a job.

### Supported Python Versions
Nova Forge SDK is tested on:
* Python 3.12

### IAM Roles/Policies
* You will need an IAM role with sufficient permissions in order to use the Nova Forge SDK. You can find a list of these permissions in the `docs/user-guides/iam_setup.md` file.

### Instances

Nova customization jobs also require access to enough of the right instance type to run:
- The requested instance type and count should be compatible with the requested job. The SDK will validate your instance configuration for you.
- The [SageMaker account quotas](https://docs.aws.amazon.com/general/latest/gr/sagemaker.html) for using the requested instance type in training jobs (for SMTJ) or HyperPod clusters (for SMHP) should allow the requested number of instances.
- (For SMHP) The selected HyperPod cluster should have a [Restricted Instance Group](https://docs.aws.amazon.com/sagemaker/latest/dg/nova-hp-cluster.html) with enough instances of the right type to run the requested job. The SDK will validate that your cluster contains a valid instance group.
- You can look in the `docs/user-guides/instance_type_spec.md` file for the different instance types and combinations for specific jobs and methods.

### HyperPod CLI

For HyperPod-based customization jobs, the SDK uses the [SageMaker HyperPod CLI](https://github.com/aws/sagemaker-hyperpod-cli/) to connect to HyperPod Clusters and start jobs.

#### Prerequisites (required for both Forge and Non-Forge customers)

1. **Install Helm 3.** Verify with `helm version`. If not installed:
    ```bash
    curl -fsSL -o get_helm.sh https://raw.githubusercontent.com/helm/helm/main/scripts/get-helm-3
    chmod 700 get_helm.sh
    ./get_helm.sh
    rm -f ./get_helm.sh
    ```

2. If you are using a Python virtual environment, activate it before installing the CLI:
    ```bash
    source <path to venv>/bin/activate
    ```

#### For Non-Forge Customers

1. Clone the `release_v2` branch of the HyperPod CLI:
    ```bash
    git clone -b release_v2 https://github.com/aws/sagemaker-hyperpod-cli.git
    ```
2. Install the CLI:
    ```bash
    cd sagemaker-hyperpod-cli
    pip install .
    ```
3. Verify the installation:
    ```bash
    hyperpod --help
    ```

#### For Forge Customers

1. Download the latest HyperPod CLI repo with Forge feature support from S3:
    ```bash
    aws s3 cp s3://nova-forge-c7363-206080352451-us-east-1/v1/ ./ --recursive
    mkdir -p src/hyperpod_cli/sagemaker_hyperpod_recipes/launcher/nemo
    git clone https://github.com/NVIDIA/NeMo-Framework-Launcher.git src/hyperpod_cli/sagemaker_hyperpod_recipes/launcher/nemo/nemo_framework_launcher --recursive
    pip install -e .
    ```
2. Verify the installation:
    ```bash
    hyperpod --help
    ```
---
## Supported Models and Training Methods

### Models

| Model         | Version | Model Type                     | Context Length |
| ------------- | ------- | ------------------------------ | -------------- |
| `NOVA_MICRO`  | 1.0     | `amazon.nova-micro-v1:0:128k`  | 128k tokens    |
| `NOVA_LITE`   | 1.0     | `amazon.nova-lite-v1:0:300k`   | 300k tokens    |
| `NOVA_LITE_2` | 2.0     | `amazon.nova-2-lite-v1:0:256k` | 256k tokens    |
| `NOVA_PRO`    | 1.0     | `amazon.nova-pro-v1:0:300k`    | 300k tokens    |

### Training Methods

| Method       | Description                              | Supported Models       |
|--------------|------------------------------------------|------------------------|
| `CPT`                | Continued Pre-Training                   | All models (SMHP only) |
| `DPO_LORA`           | Direct Preference Optimization with LoRA | Nova 1.0 models        |
| `DPO_FULL`           | Full-rank Direct Preference Optimization | Nova 1.0 models        |
| `SFT_LORA`           | Supervised Fine-tuning with LoRA         | All models             |
| `SFT_FULL`           | Full-rank Supervised Fine-tuning         | All models             |
| `RFT_LORA`           | Reinforcement Fine-tuning with LoRA      | Nova 2.0 models        |
| `RFT_FULL`           | Full Reinforcement Fine-tuning           | Nova 2.0 models        |
| `RFT_MULTITURN_LORA` | RFT Multiturn with LoRA                  | Nova 2.0 models        |
| `RFT_MULTITURN_FULL` | Full RFT Multiturn                       | Nova 2.0 models        |
| `EVALUATION`         | Model evaluation                         | All models             |

### Platform Support

| Platform          | Description                                | Models Supported | Supported Methods                        |
| ----------------- | ------------------------------------------ | ---------------- | ---------------------------------------- |
| `SMTJ`            | SageMaker Training Jobs                    | All models       | All methods                              |
| `SMTJServerless`  | SageMaker Serverless Training (no infra)   | All models       | SFT, DPO, RFT_LORA, EVALUATION           |
| `SMHP`            | SageMaker HyperPod                         | All models       | All methods                              |
| `BEDROCK`         | Amazon Bedrock (Managed Service)           | All models       | SFT, DPO, RFT                            |

## Data Preparation

Before launching a training job, your data needs to be in the right format. The SDK's dataset module handles loading, transforming, validating, filtering, and saving training data — supporting JSONL, JSON, CSV, Parquet, and Arrow formats from local files or S3.

A typical data preparation workflow:

```python
from amzn_nova_forge.dataset import JSONLDatasetLoader
from amzn_nova_forge.model import TrainingMethod, Model, TransformMethod, ValidateMethod

loader = JSONLDatasetLoader()
loader.load("s3://my-bucket/raw-data.jsonl")
loader.transform(method=TransformMethod.SCHEMA, training_method=TrainingMethod.SFT_LORA, model=Model.NOVA_LITE_2)
loader.validate(method=ValidateMethod.INVALID_RECORDS, training_method=TrainingMethod.SFT_LORA, model=Model.NOVA_LITE_2)
loader.save("s3://my-bucket/prepared-data.jsonl")
```

For the complete guide — including column mappings, dataset splitting, filtering, chaining operations, and end-to-end examples — see **[Data Preparation Guide](docs/user-guides/data_prep.md)**.

For a hands-on notebook walkthrough, see [`samples/dataprep_quickstart.ipynb`](samples/dataprep_quickstart.ipynb).

---
## Core Modules Overview

The Nova Forge SDK is organized into the following modules:

| Module             | Purpose                                       | Key Components                                                   |
| ------------------ | --------------------------------------------- | ---------------------------------------------------------------- |
| **Dataset**        | Data loading, transformation, filtering, and preparation | `JSONLDatasetLoader`, `JSONDatasetLoader`, `CSVDatasetLoader`    |
| **Manager**        | Runtime infrastructure management             | `SMTJRuntimeManager`, `SMTJServerlessRuntimeManager`, `SMHPRuntimeManager`, `BedrockRuntimeManager` |
| **Model**          | Main SDK entrypoint and orchestration         | `NovaModelCustomizer`                                             |
| **Monitor**        | Job monitoring and logging                    | `CloudWatchLogMonitor`, `MLflowMonitor`                          |
| **RFT Multiturn**  | Reinforcement fine-tuning infrastructure      | `RFTMultiturnInfrastructure`                                      |

* For detailed API documentation: See [`docs/spec/`](docs/spec/index.md)
* For usage examples: See [`samples/nova_quickstart.ipynb`](samples/nova_quickstart.ipynb)
* For RFT Singleturn examples: See [`samples/rft_singleturn_quickstart.ipynb`](samples/rft_singleturn_quickstart.ipynb)
* For RFT Multiturn documentation: See [`docs/user-guides/rft_multiturn.md`](docs/user-guides/rft_multiturn.md)
* For RFT Multiturn examples: See [`samples/rft_multiturn_quickstart.ipynb`](samples/rft_multiturn_quickstart.ipynb)

### Service Classes (Recommended)

As of v1.4.0, the SDK provides modular service classes that replace `NovaModelCustomizer`. We recommend using these for all new work:

| Class | Purpose |
|-------|---------|
| `ForgeTrainer` | Training jobs (SFT, CPT, DPO, RFT) |
| `ForgeEvaluator` | Evaluation jobs (MMLU, GEN_QA, LLM_JUDGE, etc.) |
| `ForgeDeployer` | Deploy to SageMaker or Bedrock |
| `ForgeInference` | Real-time and batch inference |

Each class takes its own configuration in the constructor. Shared settings like `output_s3_path` and `mlflow_monitor` are passed via a `ForgeConfig` object.

**Example — Train and Deploy with Service Classes:**

```python
from amzn_nova_forge import (
    ForgeTrainer, ForgeEvaluator, ForgeDeployer, ForgeInference,
    ForgeConfig, Model, TrainingMethod, DeployPlatform,
    SMTJRuntimeManager,
)

# Configure infrastructure
runtime = SMTJRuntimeManager(
    instance_type="ml.p5.48xlarge",
    instance_count=4,
)

# Train
trainer = ForgeTrainer(
    model=Model.NOVA_LITE_2,
    method=TrainingMethod.SFT_LORA,
    infra=runtime,
    training_data_s3_path="s3://bucket/train.jsonl",
    config=ForgeConfig(output_s3_path="s3://bucket/output"),
)
training_result = trainer.train(job_name="my-sft-job")

# Deploy
deployer = ForgeDeployer(model=Model.NOVA_LITE_2)
deployment_result = deployer.deploy(
    model_artifact_path=training_result.model_artifacts.checkpoint_s3_path,
    deploy_platform=DeployPlatform.SAGEMAKER,
    unit_count=1,
    endpoint_name="my-nova-endpoint",
)

# Invoke
inference = ForgeInference()
result = inference.invoke(
    endpoint_arn=deployment_result.endpoint.endpoint_arn,
    request_body={
        "messages": [{"role": "user", "content": "Hello!"}],
        "max_tokens": 100,
    },
)
result.show()
```

For the equivalent workflow using the legacy `NovaModelCustomizer`, see the [Model Module](#model-module-deprecated) section below.

For detailed API documentation on all service classes, see [`docs/spec/`](docs/spec/service-classes.md).

### Dataset Module
Handles data loading, transformation, validation, filtering, and persistence for training datasets. Supports JSONL, JSON, CSV, Parquet, and Arrow formats from local files or S3.

See the [Data Preparation](#data-preparation) section above for usage overview, or the full **[Data Preparation Guide](docs/user-guides/data_prep.md)** for detailed documentation.

### Manager Module
Manages runtime infrastructure for executing training and evaluation jobs.
For the allowed instance types for each model/method combination, see `docs/user-guides/instance_type_spec.md`.

**Main Methods:**
- `execute()` - Start a training or evaluation job
- `cleanup()` - Stop and clean up a running job
- `scale_cluster()` - (SMHP only) Scale HyperPod cluster instance groups up or down
- `get_instance_groups()` - (SMHP only) View instance groups and current instance counts

**Key Classes:**
- `SMTJRuntimeManager` - For SageMaker Training Jobs
- `SMHPRuntimeManager` - For SageMaker HyperPod clusters
- `BedrockRuntimeManager` - For Amazon Bedrock managed service

**Cluster Scaling (SMHP):**

The `SMHPRuntimeManager` provides a `scale_cluster()` method to dynamically adjust the number of instances in a HyperPod cluster instance group:

```python
from amzn_nova_forge.manager import SMHPRuntimeManager

# Create a runtime manager for your cluster
manager = SMHPRuntimeManager(
    instance_type="ml.p4d.24xlarge",
    instance_count=4,
    cluster_name="my-hyperpod-cluster",
    namespace="default"
)

# View the available instance groups to update
available_instance_groups = manager.get_instance_groups()

# Scale up the worker group from 4 to 8 instances
result = manager.scale_cluster(
    instance_group_name="worker-group",
    target_instance_count=8
)
```
For more cluster scaling documentation, see [`docs/spec/runtime-managers.md`](docs/spec/runtime-managers.md).

### Model Module (Deprecated)

> ⚠️ `NovaModelCustomizer` is deprecated as of v1.4.0 and will be removed in a future version. It remains fully functional, but we recommend using the modular [Service Classes](#service-classes-recommended) (`ForgeTrainer`, `ForgeEvaluator`, `ForgeDeployer`, `ForgeInference`) for all new work.

Provides the main SDK entrypoint for orchestrating model customization workflows.

**Main Methods:**
- `train()` - Launch a training job
- `evaluate()` - Launch an evaluation job
- `deploy()` - Deploy trained model to Amazon SageMaker or Bedrock
- `batch_inference()` - Run batch inference on trained model
- `get_logs()` - Retrieve CloudWatch logs for current job
- `get_data_mixing_config()` - Get data mixing configuration
- `set_data_mixing_config()` - Set data mixing configuration

**Key Class:**
- `NovaModelCustomizer` - Main orchestration class

### Monitor Module
Provides job monitoring and experiment tracking capabilities.

**Main Methods:**
- `show_logs()` - Display CloudWatch logs
- `get_logs()` - Retrieve logs as list
- `from_job_result()` - Create monitor from job result
- `from_job_id()` - Create monitor from job ID

**Key Classes:**
- `CloudWatchLogMonitor` - For viewing job logs
- `MLflowMonitor` - For experiment tracking with presigned URL generation

---

### RFT Multiturn Module
Manages infrastructure for reinforcement fine-tuning with multi-turn conversational tasks.

**Main Methods:**
- `setup()` - Deploy SAM stack and validate platform
- `start_training_environment()` - Start training environment
- `start_evaluation_environment()` - Start evaluation environment
- `get_logs()` - Retrieve environment logs
- `kill_task()` - Stop running task
- `cleanup()` - Clean up infrastructure resources
- `check_all_queues()` - Check message counts in all queues
- `flush_all_queues()` - Purge all messages from queues

**Key Classes:**
- `RFTMultiturnInfrastructure` - Main infrastructure management class
- `CustomEnvironment` - For creating custom reward environments

**Supported Platforms:**
- `LOCAL` - Local development environment
- `EC2` - Amazon EC2 instances
- `ECS` - Amazon ECS clusters

**Built-in Environments:**
- `VFEnvId.WORDLE` - Wordle game environment
- `VFEnvId.TERMINAL_BENCH` - Terminal benchmark environment

---
### Iterative Training

The Nova Forge SDK supports iterative fine-tuning of Nova models.

This is done by progressively running fine-tuning jobs on the output checkpoint from the previous job:

``` python
# Stage 1: Initial training on base model
stage1_trainer = ForgeTrainer(
    model=Model.NOVA_LITE,
    method=TrainingMethod.SFT_LORA,
    infra=infra,
    training_data_s3_path="s3://bucket/stage1-data.jsonl",
    config=ForgeConfig(output_s3_path="s3://bucket/stage1-output"),
)

stage1_result = stage1_trainer.train(job_name="stage1-training")
# Wait for completion...
stage1_checkpoint = stage1_result.model_artifacts.checkpoint_s3_path

# Stage 2: Continue training from Stage 1 checkpoint
stage2_trainer = ForgeTrainer(
    model=Model.NOVA_LITE,
    method=TrainingMethod.SFT_LORA,
    infra=infra,
    training_data_s3_path="s3://bucket/stage2-data.jsonl",
    model_s3_path=stage1_checkpoint,  # Use previous checkpoint
    config=ForgeConfig(output_s3_path="s3://bucket/stage2-output"),
)

stage2_result = stage2_trainer.train(job_name="stage2-training")
```

**Note:** Iterative fine-tuning requires using the same model and training method (LoRA vs Full-Rank) across all stages.

### Dry Run

The Nova Forge SDK supports `dry_run` mode for the following functions: `train()`, `evaluate()`, and `batch_inference()`.

When calling any of the above functions, you can set the `dry_run` parameter to `True`.
The SDK will still generate your recipe and validate your input, but it won't begin a job.
This feature is useful whenever you want to test or validate inputs and still have a recipe generated, without starting a job.

``` python
# Training dry run
trainer.train(
    job_name="train_dry_run",
    dry_run=True,
    ...
)

# Evaluation dry run
evaluator.evaluate(
    job_name="evaluate_dry_run",
    dry_run=True,
    ...
)
```

### Data Mixing
Data mixing allows you to blend your custom training data with Nova's high-quality curated datasets, helping maintain the model's broad capabilities while adding your domain-specific knowledge.

**Key Features:**
- Available for CPT and SFT training for Nova 1 and Nova 2 (both LoRA and Full-Rank) on SageMaker HyperPod
- Available for SFT (text-only, LoRA and Full-Rank) on Nova 2 Lite on SMTJServerless
- Mix customer data (0-100%) with Nova's curated data
- Nova data categories include general knowledge and code
- Nova data percentages must sum to 100%

**Example Usage:**

```python
# Initialize with data mixing enabled (HyperPod)
trainer = ForgeTrainer(
    model=Model.NOVA_LITE_2,
    method=TrainingMethod.SFT_LORA,
    infra=SMHPRuntimeManager(...),
    training_data_s3_path="s3://bucket/data.jsonl",
    data_mixing_enabled=True,
    config=ForgeConfig(output_s3_path="s3://bucket/output"),
)

# Or use SMTJServerless
trainer = ForgeTrainer(
    model=Model.NOVA_LITE_2,
    method=TrainingMethod.SFT_LORA,
    infra=SMTJServerlessRuntimeManager(...),
    training_data_s3_path="s3://bucket/data.jsonl",
    data_mixing_enabled=True,
    config=ForgeConfig(output_s3_path="s3://bucket/output"),
)

# Configure data mixing percentages
trainer.data_mixing.set_config({
    "customer_data_percent": 50,  # 50% your data
    "nova_code_percent": 30,      # 30% Nova code data (30% of Nova's 50%)
    "nova_general_percent": 70    # 70% Nova general data (70% of Nova's 50%)
})

# Or use 100% customer data (no Nova mixing)
trainer.data_mixing.set_config({
    "customer_data_percent": 100,
    "nova_code_percent": 0,
    "nova_general_percent": 0
})
```
**Important Notes:**
- The `dataset_catalog` field is system-managed and cannot be set by users
- Data mixing is available on SageMaker HyperPod and SMTJServerless for Forge customers.
- Refer to the [Get Forge Subscription]('https://docs.aws.amazon.com/sagemaker/latest/dg/nova-forge.html#nova-forge-prereq-access') page to enable Nova subscription in your account to use this feature.

### Job Notifications

Get email notifications when your training jobs complete, fail, or are stopped. The SDK automatically sets up the required AWS infrastructure (CloudFormation, DynamoDB, SNS, Lambda, EventBridge) to monitor job status and send notifications.

**Features:**
- Automatic AWS infrastructure setup and management
- Email notifications for terminal job states (Completed, Failed, Stopped)
- Email notifications for SMHP master pods that are stuck in a crash loop
- Output artifact validation for successful jobs (manifest.json)
- Optional customer key KMS encryption for SNS topics

**Platform Support:**
- **SMTJ** (SageMaker Training Jobs): Minimal configuration required
- **SMTJServerless** (SageMaker Serverless): No instance type needed — SageMaker manages compute automatically
- **SMHP** (SageMaker HyperPod): Requires kubectl Lambda layer + additional parameters (see [`docs/spec/runtime-managers.md`](docs/spec/runtime-managers.md) for more details)
- **Bedrock** (Amazon Bedrock): Fully managed, no infrastructure configuration required

**Quick Example:**
```python
# Start a training job
result = customizer.train(job_name="my-job")

# Enable notifications (SMTJ)
result.enable_job_notifications(
    emails=["user@example.com"]
)

# Enable notifications (SMHP)
result.enable_job_notifications(
    emails=["user@example.com"],
    namespace="kubeflow",  # Required for SMHP
    kubectl_layer_arn="arn:aws:lambda:<region>:123456789012:layer:kubectl:1"  # Required for SMHP
)
```

**Important Notes:**
- Users must confirm their email subscription by clicking the link in the AWS SNS confirmation email
- SMHP job notifications requires a kubectl Lambda layer (see [Job Notifications Guide](docs/user-guides/job_notifications.md))
- Notification infrastructure is created once per region (SMTJ) or once per cluster (SMHP) and shared across jobs.
- See [`docs/user-guides/job_notifications.md`](docs/user-guides/job_notifications.md) for detailed setup instructions, troubleshooting, and advanced usage
- See [`docs/spec/notifications.md`](docs/spec/notifications.md) for complete API documentation on job notifications.

### Batch Sample Tracing

Diagnose gradient spikes by identifying which training data lines were used in a specific training step. Enable with `enable_batch_sample_tracing=True` on `ForgeTrainer`, then call `trainer.trace_batch(result, step=N)` after the job completes. See [`docs/spec/service-classes.md`](docs/spec/service-classes.md) for full API details.


---
## Telemetry

The Nova Forge SDK has telemetry enabled to help us better understand user needs, diagnose issues, and deliver new features. This telemetry tracks the usage of various SDK functions. If you prefer to opt out of telemetry, you can do so by setting the `TELEMETRY_OPT_OUT` environment variable to `true`:

```bash
export TELEMETRY_OPT_OUT=true
```

---
## Getting Started
This comprehensive SDK enables end-to-end customization of Amazon Nova models with support for multiple training methods, deployment platforms, and monitoring capabilities. Each module is designed to work together seamlessly while providing flexibility for advanced use cases.

To get started customizing Nova models, please see the following files:
* Notebook with "quick start" examples to start customizing at `samples/nova_quickstart.ipynb`
* Specification document with detailed information about each module at [`docs/spec/`](docs/spec/index.md)

---
## Security Best Practices for SDK Users

### 1. IAM and Access Management

**Execution Roles**

- Use dedicated execution roles for SageMaker training jobs with minimal required permissions
- Avoid using admin roles - follow the principle of least privilege
- Regularly audit role permissions and remove unused policies

```python
# Good: Explicit execution role
runtime = SMTJRuntimeManager(
    instance_type="ml.p5.48xlarge",
    instance_count=2,
    execution_role="arn:aws:iam::123456789012:role/SageMakerNovaTrainingRole"
)

# Avoid: Using default role without validation
```

**Required Permissions**

The SDK requires specific IAM permissions. Review the [IAM](#iam-rolespolicies) section and:

* Grant only the minimum permissions needed for your use case
* Use condition statements to restrict resource access
* Regularly review and rotate access keys

**Verifying Your IAM Setup**

The SDK automatically validates your IAM permissions before every `train()`, `evaluate()`, and `batch_inference()` call. If your role is missing required permissions, the SDK will report the specific missing permissions before attempting to start a job.

If you want to verify your setup without starting a job, use `dry_run=True`. This runs the full validation (IAM permissions, recipe, infrastructure) and reports any issues, but does not submit a job:

```python
trainer = ForgeTrainer(
    model=Model.NOVA_LITE_2,
    method=TrainingMethod.SFT_LORA,
    infra=SMTJRuntimeManager(instance_type="ml.p5.48xlarge", instance_count=4),
    training_data_s3_path="s3://bucket/train.jsonl",
    config=ForgeConfig(output_s3_path="s3://bucket/output"),
)

# Validate everything without starting a job
trainer.train(job_name="verify-setup", dry_run=True)
```

The validation checks include:
- **Calling role permissions** — verifies your current role has the IAM actions needed for the chosen platform (e.g., `sagemaker:CreateTrainingJob` for SMTJ, `bedrock:CreateModelCustomizationJob` for Bedrock) using `iam:SimulatePrincipalPolicy` and policy inspection.
- **Execution role** (SMTJ/SMTJServerless) — confirms the execution role exists, trusts `sagemaker.amazonaws.com`, and has the required S3 permissions (`s3:GetObject`, `s3:PutObject`, `s3:ListBucket`).
- **Cluster access** (SMHP) — verifies the HyperPod cluster exists and your role can describe it.

The IAM validation requires `iam:SimulatePrincipalPolicy` and policy-read permissions (e.g., `iam:GetRole`, `iam:ListRolePolicies`, `iam:ListAttachedRolePolicies`) on your calling role. If your role lacks these actions, the validation will fail with an error rather than silently skipping the checks. 

You can disable the IAM validation check and rely on the AWS service to report permission errors at job submission time:

```python
from amzn_nova_forge.core import ValidationConfig

trainer = ForgeTrainer(
    ...
    config=ForgeConfig(
        validation_config=ValidationConfig(iam=False),
    ),
)
```

### 2. Credential Management

**AWS Credentials**

* Never hardcode credentials in code or configuration files
* Use IAM roles instead of access keys when possible
* Rotate credentials regularly
* Use AWS Secrets Manager for application secrets
* Enable credential monitoring through AWS Config

**MLflow Integration**

* Secure MLflow tracking URIs with proper authentication
* Use encrypted connections to MLflow servers
* Implement access controls on experiment data
* Regularly audit MLflow access logs

### 3. Data Security and Privacy

**Training Data Protection**

- Encrypt data at rest in S3 using KMS keys
- Use S3 bucket policies to restrict access
- Validate data sources before processing

```python
# Ensure your S3 buckets have proper encryption and access controls
customizer = NovaModelCustomizer(
    model=Model.NOVA_LITE_2,
    method=TrainingMethod.SFT_LORA,
    infra=runtime,
    data_s3_path="s3://secure-training-bucket/encrypted-data/data.jsonl",
    output_s3_path="s3://secure-output-bucket/results"
)
```

### 4. Network Security

**VPC Configuration**

* Deploy in private subnets when possible
* Use VPC endpoints for AWS service access
* Implement security groups with minimal required ports
* Enable VPC Flow Logs for network monitoring

### 5. Secure Communication

- Always use HTTPS endpoints
- Never disable SSL certificate verification
- Keep TLS libraries updated

### 6. Input Validation

- Always validate user inputs before passing to SDK
- Sanitize data that will be stored or processed
- Check resource quotas before job submission
- Sanitize job names and resource identifiers


```python
# The SDK includes built-in validation
loader = JSONLDatasetLoader(question="input", answer="output")
loader.load("s3://your-bucket/training-data.jsonl")
# Always validate your data format
loader.validate(method=TrainingMethod.SFT_LORA, model=Model.NOVA_LITE_2)
```

### 7. Monitoring & Logging

- Enable CloudTrail for API audit logs
- Use CloudWatch for operational monitoring
- Never log sensitive data (tokens, credentials, PII)
- Monitor job logs through CloudWatch
- Set up alerts for suspicious activities

**Security Monitoring**
- Monitor failed authentication attempts
- Track unusual resource access patterns
- Log all model deployment activities

### 8. Deployment Security

**Bedrock Deployment**

- Use least privilege policies for Bedrock access
- Implement endpoint access controls
- Monitor model inference patterns
- Enable request/response logging when appropriate

### 9. Validation

The SDK includes built-in validation:

- IAM permission validation before job execution
- Input sanitization for user-provided parameters

Validation is enabled by default.