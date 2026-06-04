# Dataset Loaders
Dataset loaders handle loading, transforming, and saving datasets in various formats.

### Base Class: DatasetLoader
Base class for all dataset loaders. Handles path resolution (relative, tilde, `..` paths normalized to absolute), directory detection (scans for matching files, loads in lexicographic order), and delegates single-file loading to subclasses.

#### Constructor

**Signature:**
```python
def __init__(
 self,
 **column_mappings
)
```

**Parameters:**
- `**column_mappings`: Keyword arguments mapping standard column names to dataset column names
  - Example: `question="input"` where "question" is the standard name and "input" is your column name
#### Column Mappings
If you are transforming a plain JSON, JSONL, or CSV file from a generic format (e.g. 'input/output') to another format (e.g. Converse for SFT), you need to provide "column mappings" to connect your generic column/field name to the expected ones in the transformation function.

For example, if your plain dataset has "input" and "output" columns, and you want to transform it for SFT (which requrires "question" and "answer"), you would provide the following:
```python
loader = JSONDatasetLoader(
    question="input",
    answer="output"
)
```
Below is a list of accepted column mapping parameters for transformations.
* SFT: `question`, `answer`
  * Optional: `system`, [image/video required options]: `image_format`/`video_format`, `s3_uri`, `bucket_owner`
  * 2.0: `reasoning_text`, `tools`/`toolsConfig`*
* RFT: `question`, `reference_answer`
  * Optional: `system`, `id`, `tools`*
* Eval: `query`, `response`
  * Optional: `images`, `metadata`
* CPT: `text`

Additional Notes:
* If you're providing multimodal data in a generic format, you need to provide ALL three of the following fields:
  * `image_format` OR `video_format` + `s3_uri`, `bucket_owner`
* *`tools/toolsConfig` (SFT 2.0) and `tools` (RFT) parameters can *only* be provided when transforming from OpenAI Messages format to Converse or OpenAI. A generic format *cannot* be provided for this transformation to work.

---
### JSONLDatasetLoader
Loads datasets from JSONL (JSON Lines) files.

#### Methods

##### `load()`
Loads dataset from a JSONL file (local or S3).

**Signature:**
```python
def load(
 self,
 path: str
) -> "DatasetLoader"
```

**Parameters:**
- `path` (str): Path to JSONL file (local path or S3 URI)

**Returns:**
- `DatasetLoader`: Self (for method chaining)

**Example:**
```python
from amzn_nova_forge.dataset import *
loader = JSONLDatasetLoader()
loader.load("s3://my-bucket/data/training.jsonl")
```
---
### JSONDatasetLoader
Loads datasets from JSON files.

#### Methods

##### `load()`
Loads dataset from a JSON file (local or S3).

**Signature:**
```python
def load(
 self,
 path: str
) -> "DatasetLoader"
```

**Parameters:**
- `path` (str): Path to JSON file (local path or S3 URI)

**Returns:**
- `DatasetLoader`: Self (for method chaining)

**Example:**
```python
from amzn_nova_forge.dataset import *
loader = JSONDatasetLoader()
loader.load("data/training.json")
```
---
### CSVDatasetLoader
Loads datasets from CSV files.

#### Methods

##### `load()`
Loads dataset from a CSV file.

**Signature:**
```python
def load(
 self,
 path: str
) -> "DatasetLoader"
```

**Parameters:**
- `path` (str): Path to CSV file (local path or S3 URI)

**Returns:**
- `DatasetLoader`: Self (for method chaining)

**Example:**
```python
from amzn_nova_forge.dataset import *
loader = CSVDatasetLoader(question="user_query", answer="bot_response")
loader.load("data/conversations.csv")
```
---
### ParquetDatasetLoader
Loads datasets from Apache Parquet files. Uses lazy batch-based iteration via PyArrow — only one row group is in memory at a time.

#### Methods

##### `load()`
Loads dataset from a Parquet file or directory (local or S3).

**Signature:**
```python
def load(
 self,
 path: str
) -> "DatasetLoader"
```

**Parameters:**
- `path` (str): Path to Parquet file or directory (local path or S3 URI). Accepts `.parquet` and `.pq` extensions.

**Returns:**
- `DatasetLoader`: Self (for method chaining)

**Example:**
```python
from amzn_nova_forge.dataset import *
loader = ParquetDatasetLoader()
loader.load("s3://my-bucket/data/training.parquet")
```
---
### ArrowDatasetLoader
Loads datasets from Apache Arrow IPC and Feather files. Supports both IPC Stream and IPC File formats with automatic fallback.

#### Methods

##### `load()`
Loads dataset from an Arrow IPC or Feather file or directory (local or S3).

**Signature:**
```python
def load(
 self,
 path: str
) -> "DatasetLoader"
```

**Parameters:**
- `path` (str): Path to Arrow file or directory (local path or S3 URI). Accepts `.arrow`, `.feather`, and `.ipc` extensions.

**Returns:**
- `DatasetLoader`: Self (for method chaining)

**Example:**
```python
from amzn_nova_forge.dataset import *
loader = ArrowDatasetLoader()
loader.load("s3://my-bucket/data/training.arrow")
```
---
### HuggingFaceDatasetLoader
Loads datasets from [HuggingFace Hub](https://huggingface.co/datasets) and streams them through the Nova Forge data prep pipeline. Records are streamed lazily — no network calls happen at `load()` time. The dataset is fetched when a terminal operation (`save()`, `show()`, etc.) executes the pipeline.

Requires the optional `datasets` dependency:
```bash
pip install amzn-nova-forge[huggingface]
```

Authentication for private or gated datasets is handled by the `datasets` library's built-in credential resolution. Set the `HF_TOKEN` environment variable or run `huggingface-cli login` before using the loader.

#### Methods

##### `load()`
Loads a dataset from HuggingFace Hub.

**Signature:**
```python
def load(
 self,
 path: str,
 split: Optional[str] = None,
 name: Optional[str] = None,
 revision: Optional[str] = None,
 data_files: Optional[Union[str, Sequence[str], Mapping[str, Union[str, Sequence[str]]]]] = None,
 data_dir: Optional[str] = None,
) -> "DatasetLoader"
```

**Parameters:**
- `path` (str): HuggingFace dataset identifier.
- `split` (Optional[str]): Dataset split, forwarded to `datasets.load_dataset(split=...)`. Accepts any value supported by the HF library, including a single split name (`"train"`, `"test"`), a slice (`"train[:100]"`, `"train[:10%]"`), or concatenated splits (`"train+test"`). When `None`, the loader probes available splits: if the dataset has exactly one split it is auto-selected, and if the dataset has multiple splits a `DataPrepError` is raised listing the available splits.
- `name` (Optional[str]): Configuration name for multi-config datasets (forwarded to `datasets.load_dataset(name=...)`).
- `revision` (Optional[str]): Git tag or commit hash to pin the dataset version (forwarded to `datasets.load_dataset(revision=...)`).
- `data_files` (Optional[Union[str, Sequence[str], Mapping[str, Union[str, Sequence[str]]]]]): Specific data file(s) to load, forwarded to `datasets.load_dataset(data_files=...)`. When `None` (default), the kwarg is omitted so the library's own default applies.
- `data_dir` (Optional[str]): Subdirectory of the dataset repository to load from, forwarded to `datasets.load_dataset(data_dir=...)`. When `None` (default), the kwarg is omitted.

**Returns:**
- `DatasetLoader`: Self (for method chaining)

**Raises:**
- `ImportError`: If the `datasets` package is not installed. The error message includes the install command.
- `DataPrepError`: If the dataset cannot be loaded. Authentication failures (401/403) surface with credential guidance; all other failures include the dataset path and the original error. Raised lazily when the pipeline is executed, not at `load()` time.

**Example — public dataset:**
```python
from amzn_nova_forge.dataset import *

# Load a public dataset and save to S3
loader = HuggingFaceDatasetLoader()
loader.load("foo/bar", split="train") \
    .save("s3://my-bucket/data/foo_train.jsonl")
```


---
### CloudWatchDatasetLoader
Loads datasets from Amazon CloudWatch Logs via Insights queries and streams them through the Nova Forge data prep pipeline. The loader is lazy — `load()` stores configuration only, and the query executes when a terminal operation (`save()`, `show()`, etc.) triggers iteration.

Uses the default AWS credential chain. Required IAM permissions: `logs:StartQuery` and `logs:GetQueryResults`.

#### Methods

##### `load()`
Loads data from a CloudWatch Logs Insights query.

**Signature:**
```python
def load(
 self,
 log_group: str,
 query: str,
 start_time: datetime,
 end_time: datetime,
) -> "CloudWatchDatasetLoader"
```

**Parameters:**
- `log_group` (str): CloudWatch log group name (e.g. `"/my-app/api-logs"`).
- `query` (str): A [CloudWatch Logs Insights query](https://docs.aws.amazon.com/AmazonCloudWatch/latest/logs/CWL_QuerySyntax.html). The query determines which fields appear as keys in the yielded dictionaries. Use `fields` to select specific fields from JSON logs, or `parse` to extract values from semi-structured text. See examples below.
- `start_time` (datetime): Query start time (inclusive).
- `end_time` (datetime): Query end time (exclusive).

**Returns:**
- `CloudWatchDatasetLoader`: Self (for method chaining)

**Raises:**
- `DataPrepError`: If the query fails during iteration (e.g. log group not found, invalid query syntax, insufficient permissions, query timeout).

**Example — JSON logs:**

If your log group contains JSON events like `{"endpoint": "/chat", "request": "What is AI?", "response": "AI is artificial intelligence."}`, use `fields` to select them:

```python
from datetime import datetime, timezone
from amzn_nova_forge import CloudWatchDatasetLoader

loader = CloudWatchDatasetLoader()
loader.load(
    log_group="/my-app/api-logs",
    query="fields request, response | filter endpoint = '/chat'",
    start_time=datetime(2025, 1, 1, tzinfo=timezone.utc),
    end_time=datetime(2025, 1, 8, tzinfo=timezone.utc),
)
# Yields: {"request": "What is AI?", "response": "AI is artificial intelligence."}
loader.transform(
    column_mappings={"question": "request", "answer": "response"},
)
loader.save("s3://my-bucket/data/chat_logs.jsonl")
```

For nested JSON objects, use dot-notation to reach into deeper fields. Array elements are accessed by index. Use `as` to alias the extracted path. For example, given logs like:

```json
{"input": {"messages": [{"role": "user", "content": "What is AI?"}]}, "output": {"content": "AI is artificial intelligence."}}
```

You can extract the nested values with:

```
fields input.messages.0.content as question, output.content as answer
```

**Example — semi-structured text logs:**

If your logs are text with a known pattern like `[INFO] endpoint=/api/chat input="What is AI?" output="AI is artificial intelligence."`, use `parse` with a regex to extract named fields:

```python
loader = CloudWatchDatasetLoader()
loader.load(
    log_group="/my-app/inference-logs",
    query='''
        fields @message
        | parse @message /input="(?<input>[^"]+)" output="(?<output>[^"]+)"/
        | filter ispresent(input)
    ''',
    start_time=datetime(2025, 1, 1, tzinfo=timezone.utc),
    end_time=datetime(2025, 1, 8, tzinfo=timezone.utc),
)
# Yields: {"@message": "[INFO] endpoint=/api/chat ...", "input": "What is AI?", "output": "AI is artificial intelligence."}
loader.transform(
    column_mappings={"question": "input", "answer": "output"},
)
loader.save("s3://my-bucket/data/inference_logs.jsonl")
```


---
### Common DatasetLoader Methods
These methods are available on all DatasetLoader subclasses.

#### `show()`
Displays the first n rows of the dataset.
**Signature:**
```python
def show(
 self,
 n: int = 10
) -> None
```

**Parameters:**
- `n` (int): Number of rows to display (default: 10)

**Example:**
```python
loader.show(5) # Show first 5 rows
```
---
#### `split_data()`
Splits dataset into train, validation, and test sets.

**Signature:**
```python
def split_data(
 self,
 train_ratio: float = 0.8,
 val_ratio: float = 0.1,
 test_ratio: float = 0.1,
 seed: int = 42,
) -> Tuple["DatasetLoader", "DatasetLoader", "DatasetLoader"]
```

**Parameters:**
- `train_ratio` (float): Proportion of data for training (default: 0.8)
- `val_ratio` (float): Proportion of data for validation (default: 0.1)
- `test_ratio` (float): Proportion of data for testing (default: 0.1)
- `seed` (int): Random seed for reproducibility (default: 42)

**Returns:**
- `Tuple[DatasetLoader, DatasetLoader, DatasetLoader]`: Three DatasetLoader objects (train, val, test)

**Raises:**
- `DataPrepError`: If ratios don't sum to 1.0 or dataset is empty

**Example:**
```python
train_loader, val_loader, test_loader = loader.split_data(
 train_ratio=0.7,
 val_ratio=0.2,
 test_ratio=0.1
)
```
---
#### `transform()`
Transforms dataset to the required format for a specific training method and model. Currently the following transformations are supported:
* Q/A-formatted CSV/JSON/JSONL to SFT 1.0, SFT 2.0 (without reasoningContent, Tools), RFT, Eval, CPT
* OpenAI Messages format to SFT 1.0 and SFT 2.0 (with Tools)

**Signature:**
```python
def transform(
 self,
 method: TransformMethod = TransformMethod.SCHEMA,
 model: Optional[Model] = None,
 eval_task: Optional[EvaluationTask] = None,
 **kwargs,
) -> "DatasetLoader"
```
**Parameters:**
- `method` (TransformMethod): The transform method (default: `TransformMethod.SCHEMA`). Also accepts a `TrainingMethod` enum for backward compatibility (deprecated).
- `model` (Optional[Model]): The Nova model version (e.g., `Model.NOVA_LITE_2`). Can be passed positionally for backward compatibility.
- `eval_task` (Optional[EvaluationTask]): Evaluation task. Can be passed positionally for backward compatibility.
- `**kwargs`: Method-specific arguments passed to the operation. For `TransformMethod.SCHEMA`:
  - `training_method` (TrainingMethod): Required. The training method (e.g., `TrainingMethod.SFT_LORA`).
  - `model` (Model): Required. The target model.
  - `eval_task` (EvaluationTask): Optional. Required when `training_method` is `EVALUATION`.
  - `platform` (Platform): Required for RFT Multiturn methods (`RFT_MULTITURN_LORA`, `RFT_MULTITURN_FULL`). Determines the output format: `Platform.SMTJServerless` produces flat `{"prompt": "..."}` format, `Platform.SMHP` produces nested `{"id": "...", "metadata": {"prompt": "..."}}` format.
  - `column_mappings` (dict): Optional. Maps standard column names to your dataset's column names.
  - `multimodal_data_s3_path` (Optional[str]): S3 prefix where images will be uploaded during conversion (e.g., `s3://my-bucket/images/`). Required when the dataset contains `image_url` content blocks in OpenAI format. When saving the output to S3, the save path must use the same bucket.
  - `multimodal_data_bucket_owner` (Optional[str]): AWS account ID that owns the S3 bucket. If not provided, auto-resolved via STS.

**Returns:**
- `DatasetLoader`: Self (for method chaining)

**Raises:**
- `ValueError`: If method/model combination is not supported
- `DataPrepError`: If transformation fails

**Example:**
```python
# Text-only transform
loader.transform(
 method=TransformMethod.SCHEMA,
 training_method=TrainingMethod.SFT_LORA,
 model=Model.NOVA_MICRO,
)

# Multimodal transform with automatic S3 image upload
loader.transform(
 method=TransformMethod.SCHEMA,
 training_method=TrainingMethod.SFT_LORA,
 model=Model.NOVA_LITE_2,
 multimodal_data_s3_path="s3://my-bucket/images/",
)
```
---
#### `validate()`
Validates dataset when given the user's intended training method and model.

**Signature:**
```python
def validate(
 self,
 method: ValidateMethod = ValidateMethod.INVALID_RECORDS,
 model: Model (Optional),
 eval_task: EvaluationTask (Optional),
 platform: Platform (Optional)
) -> None
```
**Parameters:**
- `method` (ValidateMethod): The validation method (default: `ValidateMethod.INVALID_RECORDS`). `ValidateMethod.SCHEMA` is deprecated but still supported for backward compatibility.
- `model` (Model): The Nova model version (e.g., `Model.NOVA_LITE`)
- `eval_task` (EvaluationTask): Optional. The evaluation task (e.g., `EvaluationTask.GEN_QA`)
- `platform` (Platform): Optional. The target platform (`Platform.SMTJServerless`, `Platform.SMHP`, `Platform.SMTJ`, or `Platform.BEDROCK`). Required for RFT Multiturn methods — routes to `RFTMultiturnServerlessValidator` when `Platform.SMTJServerless`, validating the flat `{"prompt": "..."}` format.

**Returns:**
- None

**Raises:**
- `ValueError`: If method/model combination is not supported or validation is unsuccessful.

**Row-Count Checks:**

The following row-count checks are enforced during `train()`. They do not currently run on the `loader.validate()` path.

| Check | Applies To | Limit |
|---|---|---|
| Maximum dataset rows | `BEDROCK` / `SFT_LORA`, `SFT_FULL`, `DPO_LORA`, `DPO_FULL`, `CPT` / `NOVA_MICRO`, `NOVA_LITE`, `NOVA_PRO` | 20,000 rows |
| Minimum dataset rows | `BEDROCK` / `SFT_LORA`, `SFT_FULL`, `DPO_LORA`, `DPO_FULL`, `CPT` / `NOVA_MICRO`, `NOVA_LITE`, `NOVA_PRO` | 8 rows |
| Maximum dataset rows | `BEDROCK` / `SFT_LORA`, `SFT_FULL` / `NOVA_LITE_2` | 20,000 rows |
| Minimum dataset rows | `BEDROCK` / `SFT_LORA`, `SFT_FULL` / `NOVA_LITE_2` | 200 rows |
| Maximum dataset rows | `BEDROCK` / `RFT_LORA`, `RFT_FULL` / `NOVA_LITE_2` | 20,000 rows |
| Minimum dataset rows | `BEDROCK` / `RFT_LORA`, `RFT_FULL` / `NOVA_LITE_2` | 100 rows |

> **Note:** Recipe-dependent checks (e.g. dataset size ≥ `global_batch_size`) are also automatically enforced during `train()` where the fully-built recipe is available.

**Example:**
```python
loader.validate(
 method=ValidateMethod.INVALID_RECORDS,
 training_method=TrainingMethod.SFT_LORA,
 model=Model.NOVA_MICRO
)
```

If you're validating a BYOD Evaluation dataset, you need to provide another parameter, `eval_task` to the `validate` function. For example:
```
loader.validate(
    method=ValidateMethod.INVALID_RECORDS,
    training_method=TrainingMethod.EVALUATION,
    model=Model.NOVA_LITE_2,
    eval_task=EvaluationTask.GEN_QA
)

>> Validation succeeded for 22 samples on an Evaluation BYOD dataset
```
---
#### `filter()`
Queues a data filtering operation on the loader. All filters are lazy — `filter()` records the intent and execution happens when `execute()` is called (or implicitly via `transform()`, `show()`, `save()`). Multiple `filter()` calls can be chained.

**Note:** Different filters work on different data formats. Text filters (`DEFAULT_TEXT_FILTER`, `EXACT_DEDUP`, `FUZZY_DEDUP`) operate on raw data with a flat text field. `INVALID_RECORDS` works on schema-formatted data. Using a filter on an incompatible data format may produce unexpected results.

**Signature:**
```python
def filter(
 self,
 method: FilterMethod = FilterMethod.DEFAULT_TEXT_FILTER,
 **kwargs,
) -> "DatasetLoader"
```

**Text Filters** (below text filters work on raw data with a flat text field):

| Value | Description |
|---|---|
| `FilterMethod.DEFAULT_TEXT_FILTER` | Removes records with excessive URL content |
| `FilterMethod.EXACT_DEDUP` | Removes exact duplicate records by content hash |
| `FilterMethod.FUZZY_DEDUP` | Removes near-duplicate records using MinHash LSH similarity |

Text filters run on AWS Glue (Ray) and use S3 path chaining. See the [Filtering Data](../user-guides/data_prep.md#filtering-data) section in the data prep guide for kwargs (`output_path`, `runtime_manager`, etc.).

**INVALID_RECORDS Filter** (works on data in the expected format for the target training method):

| Value | Description |
|---|---|
| `FilterMethod.INVALID_RECORDS` | Removes records that don't conform to expected input for the target model and training technique (schema validation + reserved keyword detection) |

INVALID_RECORDS runs locally and modifies the loader's dataset in place.

**Parameters for `INVALID_RECORDS` (passed as kwargs):**
- `training_method` (TrainingMethod): Required. Determines which checks apply.
- `model` (Model): Required. The target model.
- `platform` (Platform): Required. The target platform (`Platform.SMTJ`, `Platform.SMHP`, or `Platform.BEDROCK`).
- `eval_task` (EvaluationTask): Optional. Required when `training_method` is `EVALUATION`.

**Returns:**
- `DatasetLoader`: Self (for method chaining)

> **Note:** After `INVALID_RECORDS` runs, use `loader.show()` to inspect the filtered dataset or `loader.save()` to persist the results.

**Raises:**
- `ValueError`: If required kwargs (`training_method`, `model`, `platform`) are missing for `INVALID_RECORDS`.

> **Note:** If your data is already in the correct schema format (e.g. previously transformed and saved), you can call `filter(INVALID_RECORDS)` directly after `load()` without `transform()`.

**Example (full chain):**
```python
loader.load("data.jsonl")
loader.filter(
    method=FilterMethod.DEFAULT_TEXT_FILTER,
).filter(
    method=FilterMethod.EXACT_DEDUP,
)
loader.transform(
    method=TransformMethod.SCHEMA,
    training_method=TrainingMethod.SFT_LORA,
    model=Model.NOVA_LITE_2,
)
loader.filter(
    method=FilterMethod.INVALID_RECORDS,
    training_method=TrainingMethod.SFT_LORA,
    model=Model.NOVA_LITE_2,
    platform=Platform.SMTJ,
)
loader.execute()          # flush the pending filter
loader.validate(
    method=ValidateMethod.SCHEMA,
    training_method=TrainingMethod.SFT_LORA,
    model=Model.NOVA_LITE_2,
)
```

**Example (already-formatted data — skip `transform()`):**
```python
loader = JSONLDatasetLoader()
loader.load("pre_transformed_data.jsonl")
loader.filter(
    method=FilterMethod.INVALID_RECORDS,
    training_method=TrainingMethod.SFT_LORA,
    model=Model.NOVA_LITE_2,
    platform=Platform.SMTJ,
)
loader.execute()
```

---
#### `save_data()`
Saves the dataset to a local or S3 location.
**Signature:**
```python
def save_data(
 self,
 save_path: str
) -> str
```
**Parameters:**
- `save_path` (str): Path where to save the file (local or S3, must end in .json or .jsonl)

**Returns:**
- `str`: Path where the file was saved

**Raises:**
- `DataPrepError`: If save fails or format is unsupported

**Example:**
```python
# Save locally
loader.save_data("output/training_data.jsonl")
# Save to S3
loader.save_data("s3://my-bucket/data/training_data.jsonl")
```
---
