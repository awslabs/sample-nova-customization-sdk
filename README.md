> **DEPRECATED** — This package (`amzn-nova-forge`) is deprecated and will no longer receive feature updates.
> Please use the **SageMaker Python SDK V3** (`pip install "sagemaker>=3.19.0"`) for Amazon Nova model customization.
> 📓 SageMaker SDK sample notebook: [Nova Serverless End-to-End Example on GitHub](https://github.com/aws/sagemaker-python-sdk/blob/master/v3-examples/model-customization-examples/serverless/serverless_e2e_example.ipynb)

# Amazon Nova Forge SDK

A comprehensive Python SDK for fine-tuning and customizing Amazon Nova models. This SDK provides a unified interface for training, evaluation, deployment, and monitoring of Nova models across both SageMaker Training Jobs and SageMaker HyperPod.

---

# Migrating from Nova Forge SDK to SageMaker Python SDK V3

## Why Migrate

The `amzn-nova-forge` package is deprecated. Amazon Nova model customization functionality is available in the SageMaker Python SDK V3.

## What's Different (Summary)

- Compute is a config object (`HyperPodCompute`, `TrainingJobCompute`), not a runtime manager
- Model is a string identifier (e.g. `"nova-textgeneration-lite-v2"`), not an enum; also accepts S3 checkpoint paths for iterative training
- Deployment uses `ModelBuilder`/`BedrockModelBuilder` pattern instead of `ForgeDeployer`
- Overrides use full recipe paths (e.g. `"recipes.training_config.trainer.lr"`); use `trainer.get_resolved_recipe()` to inspect the final merged recipe
- No `ForgeConfig` object — shared settings are passed directly to trainer constructors
- Job notifications currently support SMTJ only — pass a `notifications` dict with SNS topic and EventBridge event bus ARNs

## Installation

```bash
pip install "sagemaker>=3.19.0"
```

Requires Python 3.10 or later.

## Concept Mapping

| Forge SDK Concept | SageMaker SDK V3 Equivalent |
|---|---|
| `ForgeTrainer` (SFT) | `sagemaker.train.sft_trainer.SFTTrainer` |
| `ForgeTrainer` (CPT) | `sagemaker.train.cpt_trainer.CPTTrainer` |
| `ForgeTrainer` (DPO) | `sagemaker.train.dpo_trainer.DPOTrainer` |
| `ForgeTrainer` (RFT) | `sagemaker.train.rlvr_trainer.RLVRTrainer` |
| `ForgeTrainer` (MTRL) | `sagemaker.train.multi_turn_rl_trainer.MultiTurnRLTrainer` |
| `ForgeEvaluator` | `BenchMarkEvaluator`, `LLMAsJudgeEvaluator`, `InspectAIEvaluator`, `CustomScorerEvaluator`, `MultiTurnRLEvaluator` |
| `SMHPRuntimeManager` | `sagemaker.core.training.configs.HyperPodCompute` |
| `SMTJRuntimeManager` | `TrainingJobCompute` for serverful or omit for serverless |
| `data_mixing_enabled` | `sagemaker.train.data_mixing_config.DataMixingConfig` |
| `NovaModelCustomizer` | Individual trainer classes above |
| `ForgeDeployer` | `BedrockModelBuilder` or `ModelBuilder` |
| `ForgeInference` | SageMaker SDK `Predictor` / Bedrock `InvokeModel` |

## Full Quickstart Migration (Step-by-Step)

### Step 1: Import Modules

**Before (Forge SDK):**

```python
from amzn_nova_forge import (
    ForgeTrainer,
    ForgeEvaluator,
    ForgeDeployer,
    ForgeInference,
    ForgeConfig,
    Model,
    TrainingMethod,
    DeployPlatform,
    SMTJRuntimeManager,
    SMHPRuntimeManager,
    SMTJServerlessRuntimeManager,
    BedrockRuntimeManager,
    CloudWatchLogMonitor,
    MLflowMonitor,
    JSONLDatasetLoader,
    TransformMethod,
    ValidateMethod,
    DataMixingConfig,
    EvalTaskConfig,
    EvaluationTask,
)
```

**After (SageMaker SDK V3):**

```python
from sagemaker.train import SFTTrainer, CPTTrainer, DPOTrainer
from sagemaker.train.evaluate import BenchMarkEvaluator, get_benchmarks
from sagemaker.train.data_mixing_config import DataMixingConfig
from sagemaker.core.training.configs import HyperPodCompute, TrainingJobCompute
```

### Step 2: Configure Compute

**Before (Forge SDK) — SMTJ:**

```python
runtime = SMTJRuntimeManager(instance_type="ml.p5.48xlarge", instance_count=4)
```

**After (SageMaker SDK V3) — SMTJ:**

```python
compute = TrainingJobCompute(instance_type="ml.p5.48xlarge", instance_count=4)
```

**Before (Forge SDK) — SMHP:**

```python
runtime = SMHPRuntimeManager(
    instance_type="ml.p5.48xlarge",
    instance_count=4,
    cluster_name="my-cluster",
    namespace="default",
)
```

**After (SageMaker SDK V3) — SMHP:**

```python
compute = HyperPodCompute(
    cluster_name="my-cluster",
    instance_type="ml.p5.48xlarge",
    node_count=4,
)
```

**Before (Forge SDK) — Serverless:**

```python
runtime = SMTJServerlessRuntimeManager(model_package_group_name="test-package")
```

**After (SageMaker SDK V3) — Serverless:**

Omit the `compute` parameter entirely. The trainer runs serverless by default.

### Step 3: Training (SFT)

**Before (Forge SDK):**

```python
trainer = ForgeTrainer(
    model=Model.NOVA_LITE_2,
    method=TrainingMethod.SFT_LORA,
    infra=runtime,
    training_data_s3_path="s3://bucket/train.jsonl",
    config=ForgeConfig(output_s3_path="s3://bucket/output"),
)
result = trainer.train(job_name="my-sft-job", overrides={"lr": 5e-6, "warmup_steps": 100})
```

**After (SageMaker SDK V3):**

```python
trainer = SFTTrainer(
    model="nova-textgeneration-lite-v2",
    compute=compute,
    training_dataset="s3://bucket/train.jsonl",
    s3_output_path="s3://bucket/output/",
    overrides={
        "recipes.training_config.trainer.lr": 5e-6,
        "recipes.training_config.trainer.warmup_steps": 100,
    },
)
job_name = trainer.train(wait=False)
```

### Step 4: Data Mixing (Optional)

**Before (Forge SDK):**

```python
trainer = ForgeTrainer(..., data_mixing_enabled=True)
trainer.data_mixing.set_config(
    {
        "customer_data_percent": 50,
        "nova_code_percent": 30,
        "nova_general_percent": 70,
    }
)
```

**After (SageMaker SDK V3):**

```python
from sagemaker.train.data_mixing_config import DataMixingConfig

data_mixing = DataMixingConfig(
    customer_data_percent=50.0,
    nova_data_percentages={"code": 30.0, "reasoning": 70.0},
)
trainer = SFTTrainer(..., data_mixing_config=data_mixing)
```

### Step 5: Monitor, Notifications & Dry Run

#### Log Streaming

**Before (Forge SDK):**

```python
trainer.get_logs(job_result=result, limit=50)
monitor = CloudWatchLogMonitor.from_job_id(job_id=result.job_id, platform=platform)
monitor.show_logs(limit=100)
```

**After (SageMaker SDK V3):**

```python
# Stream logs (works on both trainer and evaluator)
trainer.stream_logs()
trainer.stream_logs(tail_logs=50)  # last 50 log entries
```

#### Metrics Visualization

**Before (Forge SDK):**

```python
monitor.plot_metrics(training_method=TrainingMethod.SFT_LORA)
```

**After (SageMaker SDK V3):**

```python
trainer.show_metrics()
```

#### Job Notifications (SMTJ only)

**Before (Forge SDK):**

```python
result = trainer.train(job_name="my-job")
result.enable_job_notifications(emails=["user@example.com"])
```

**After (SageMaker SDK V3):**

```python
trainer = SFTTrainer(
    model="amazon.nova-lite-v2",
    training_dataset="s3://bucket/train.jsonl",
    notifications={
        "sns_topic_arn": "arn:aws:sns:us-east-1:123456789012:my-topic",
        "event_bus_arn": "arn:aws:events:us-east-1:123456789012:event-bus/my-bus",
        "events": ["Completed", "Failed"],
        "job_name_prefix": "my-team-",
    },
)
trainer.train()
```

Requires a pre-created SNS topic. Notifications fire on job state changes (Completed, Failed, Stopped).

#### Dry Run Mode

**Before (Forge SDK):**

```python
trainer.train(job_name="my-job", dry_run=True)
```

**After (SageMaker SDK V3):**

```python
trainer.train(dry_run=True)
```

Runs all validations (IAM, compute, dataset) without submitting a job.

### Step 6: Evaluate

**Before (Forge SDK):**

```python
evaluator = ForgeEvaluator(
    model=Model.NOVA_LITE_2,
    infra=eval_infra,
    data_s3_path="s3://bucket/eval-data.jsonl",
    config=ForgeConfig(output_s3_path="s3://bucket/eval-output"),
)
mmlu_result = evaluator.evaluate(job_name="eval-mmlu", eval_task=EvaluationTask.MMLU)
byod_result = evaluator.evaluate(
    job_name="eval-byod",
    eval_task=EvaluationTask.GEN_QA,
    task_config=EvalTaskConfig(override_data_s3_path="s3://bucket/custom-eval.jsonl"),
)
```

**After (SageMaker SDK V3) — Benchmark (MMLU):**

```python
from sagemaker.train.evaluate import BenchMarkEvaluator, get_benchmarks

Benchmark = get_benchmarks()
evaluator = BenchMarkEvaluator(
    benchmark=Benchmark.MMLU,
    model="nova-textgeneration-lite-v2",
    s3_output_path="s3://bucket/eval-output/",
)
execution = evaluator.evaluate(checkpoint_path="s3://bucket/output/checkpoint/")
```

**After (SageMaker SDK V3) — Custom Evaluator:**

```python
from sagemaker.train.evaluate import CustomScorerEvaluator

evaluator = CustomScorerEvaluator(
    model="nova-textgeneration-lite-v2",
    eval_dataset="s3://bucket/custom-eval.jsonl",
    s3_output_path="s3://bucket/eval-output/",
)
execution = evaluator.evaluate(checkpoint_path="s3://bucket/output/checkpoint/")
```

**After (SageMaker SDK V3) — InspectAI Evaluator:**

```python
from sagemaker.train.evaluate import InspectAIEvaluator

evaluator = InspectAIEvaluator(
    model="nova-textgeneration-lite",
    bedrock_model_id="us.amazon.nova-lite-v1:0",
    benchmarks_path="s3://bucket/benchmarks/boolq/",
    tasks=[{"name": "boolq_pt", "limit": 10}],
    s3_output_path="s3://bucket/inspectai-eval-output/",
    instance_type="ml.m5.large",
)
execution = evaluator.evaluate()
execution.wait(target_status="Succeeded")
execution.show_results()
```

### Step 7: Deploy & Inference

**Before (Forge SDK):**

```python
# Deploy
deployer = ForgeDeployer(model=Model.NOVA_LITE_2)
result = deployer.deploy(
    model_artifact_path=training_result.model_artifacts.checkpoint_s3_path,
    deploy_platform=DeployPlatform.SAGEMAKER,
    unit_count=1,
    endpoint_name="my-endpoint",
)

# Inference
inference = ForgeInference()
result = inference.invoke(
    endpoint_arn=deployment_result.endpoint.endpoint_arn,
    request_body={"messages": [{"role": "user", "content": "Hello!"}], "max_tokens": 100},
)
result.show()
```

**After (SageMaker SDK V3) — SageMaker Endpoint:**

```python
import json
from sagemaker.serve import ModelBuilder

# Deploy
builder = ModelBuilder(
    model=trainer,
    role_arn="arn:aws:iam::123456789012:role/SageMakerRole",
    instance_type="ml.p4d.24xlarge",
)
builder.accept_eula = True
builder.build(region="us-east-1")
endpoint = builder.deploy(
    endpoint_name="my-endpoint",
    instance_type="ml.p4d.24xlarge",
)

# Inference
response = endpoint.invoke(
    body=json.dumps(
        {"messages": [{"role": "user", "content": [{"type": "text", "text": "Hello!"}]}]}
    ),
    content_type="application/json",
    accept="application/json",
)
body = json.loads(response.body.read())
```

**After (SageMaker SDK V3) — Bedrock:**

```python
import json
import boto3
from sagemaker.serve.bedrock_model_builder import BedrockModelBuilder

# Deploy
builder = BedrockModelBuilder(model="s3://bucket/output/checkpoint/")
result = builder.deploy(
    custom_model_name="my-custom-model",
    role_arn="arn:aws:iam::123456789012:role/SageMakerRole",
)
model_arn = result["modelArn"]

# Inference
bedrock_runtime = boto3.client("bedrock-runtime", region_name="us-east-1")
response = bedrock_runtime.invoke_model(
    modelId=model_arn,
    contentType="application/json",
    accept="application/json",
    body=json.dumps(
        {"messages": [{"role": "user", "content": [{"type": "text", "text": "Hello!"}]}]}
    ),
)
body = json.loads(response["body"].read())
```

### Additional Training Methods

#### CPT (Continued Pre-Training)

**Before:**

```python
ForgeTrainer(model=Model.NOVA_LITE_2, method=TrainingMethod.CPT, infra=smhp_runtime, ...)
```

**After:**

```python
CPTTrainer(model="nova-textgeneration-lite-v2", compute=HyperPodCompute(...), ...)
```

#### DPO (Direct Preference Optimization)

**Before:**

```python
ForgeTrainer(model=Model.NOVA_MICRO, method=TrainingMethod.DPO_LORA, infra=runtime, ...)
```

**After:**

```python
DPOTrainer(model="nova-textgeneration-micro", compute=compute, ...)
```

#### RLVR (Reinforcement Learning with Verifiable Rewards)

**Before:**

```python
ForgeTrainer(model=Model.NOVA_LITE_2, method=TrainingMethod.RFT_LORA, infra=runtime, ...)
```

**After:**

```python
from sagemaker.train import RLVRTrainer

trainer = RLVRTrainer(
    model="nova-textgeneration-lite-v2",
    compute=compute,
    training_dataset="s3://bucket/rlvr-data.jsonl",
    custom_reward_function="arn:aws:lambda:us-east-1:123456789012:function:my-reward",
    s3_output_path="s3://bucket/output/",
)
trainer.train()
```

#### Iterative Training (Resume from Checkpoint)

**Before:**

```python
trainer = ForgeTrainer(
    model=Model.NOVA_LITE_2,
    method=TrainingMethod.SFT_LORA,
    infra=runtime,
    training_data_s3_path="s3://bucket/stage2-data.jsonl",
    model_s3_path="s3://bucket/stage1-output/checkpoint/",
    config=ForgeConfig(output_s3_path="s3://bucket/stage2-output"),
)
```

**After:**

```python
trainer = SFTTrainer(
    model="s3://bucket/stage1-output/checkpoint/",
    compute=compute,
    training_dataset="s3://bucket/stage2-data.jsonl",
    s3_output_path="s3://bucket/stage2-output/",
)
trainer.train()
```

## Support

- SageMaker Python SDK docs: https://sagemaker.readthedocs.io/en/stable/
- SageMaker Python SDK GitHub: https://github.com/aws/sagemaker-python-sdk
