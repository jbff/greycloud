# GreyCloud Usage Documentation

## Overview

GreyCloud is a reusable, configurable Python module for interacting with Google's Vertex AI and GenAI services. It provides a unified interface for authentication, content generation, batch processing, and file management on top of the `google-genai` client.

## Installation

GreyCloud itself declares its core dependencies in `pyproject.toml`. To use all features, you typically need:

```bash
pip install greycloud[storage]  # includes google-genai, google-auth, google-cloud-storage
```

If you prefer installing dependencies manually:

```bash
pip install google-genai google-auth google-cloud-storage
```

## Quick Start

```python
from greycloud import GreyCloudConfig, GreyCloudClient
from google.genai import types

# Create configuration
config = GreyCloudConfig(
    project_id="your-project-id",
    location="us-central1",
    # Default model is a Gemini 3 flash model; you can override if desired.
    model="gemini-3-flash-preview",
)

# Create client
client = GreyCloudClient(config)

# Generate content
contents = [
    types.Content(
        role="user",
        parts=[types.Part.from_text(text="Hello, how are you?")]
    )
]

response = client.generate_content(contents)
print(response.text)
```

## Configuration

### GreyCloudConfig

The `GreyCloudConfig` class provides comprehensive configuration options:

#### Project and Location

```python
config = GreyCloudConfig(
    project_id="your-project-id",  # Required (or resolved from gcloud)
    location="us-central1",        # Default: "us-east4" if LOCATION/GCP_LOCATION unset
)
```

#### Authentication

```python
# Option 1: OAuth with optional service account impersonation (recommended)
config = GreyCloudConfig(
    sa_email="service-account@project.iam.gserviceaccount.com",  # Optional
    use_api_key=False,
)

# Option 2: API key authentication
config = GreyCloudConfig(
    use_api_key=True,
    api_key_file="GOOGLE_CLOUD_API_KEY",  # Path to file containing API key
)
```

#### Model Configuration

```python
config = GreyCloudConfig(
    model="gemini-3-flash-preview",  # Default text model
    endpoint="https://aiplatform.googleapis.com",  # API endpoint
    api_version="v1",  # API version
)
```

#### Generation Parameters

```python
config = GreyCloudConfig(
    temperature=1.0,           # Default: 1.0
    top_p=0.95,                # Default: 0.95
    seed=None,                 # Default: None (no fixed seed)
    max_output_tokens=65535,   # Default: 65535
)
```

#### Safety Settings

```python
# When safety_settings is None (the default), Vertex AI uses its own defaults.
config = GreyCloudConfig(
    safety_settings=None,
)

# To explicitly control safety:
config = GreyCloudConfig(
    safety_settings=[
        {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
        {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
    ]
)

# Passing an empty list [] will send an explicit empty safety_settings list.
config = GreyCloudConfig(
    safety_settings=[],
)
```

#### System Instruction

```python
config = GreyCloudConfig(
    system_instruction="You are a helpful assistant."
)
```

#### Vertex AI Search

```python
config = GreyCloudConfig(
    use_vertex_ai_search=True,
    vertex_ai_search_datastore="projects/PROJECT_ID/locations/LOCATION/collections/COLLECTION/dataStores/DATASTORE_ID"
)
```

#### Thinking Config

```python
config = GreyCloudConfig(
    thinking_level=None  # Default: None; or "LOW", "MEDIUM", "HIGH" for supported models
)
```

#### Batch Processing

```python
config = GreyCloudConfig(
    batch_gcs_bucket="your-project-batch-jobs",  # GCS bucket for batch jobs (must exist)
    batch_location="global",                     # Batch jobs require global location
    batch_poll_interval=30,                      # Polling interval in seconds
)
```

### Environment Variables

GreyCloud can also be configured via environment variables:

```bash
export PROJECT_ID="your-project-id"
export LOCATION="us-central1"
export SA_EMAIL="service-account@project.iam.gserviceaccount.com"
export USE_API_KEY="0"  # "1" to use API key, "0" for OAuth
export API_KEY_FILE="GOOGLE_CLOUD_API_KEY"
export VERTEX_AI_SEARCH_DATASTORE="projects/.../dataStores/..."
export BATCH_GCS_BUCKET="your-project-batch-jobs"
```

## Content Generation

### Non-Streaming Generation

```python
from greycloud import GreyCloudClient
from google.genai import types

client = GreyCloudClient()

contents = [
    types.Content(
        role="user",
        parts=[types.Part.from_text(text="What is the capital of France?")]
    )
]

response = client.generate_content(contents)
print(response.text)
```

### Streaming Generation

```python
contents = [
    types.Content(
        role="user",
        parts=[types.Part.from_text(text="Write a short story about a robot.")]
    )
]

for chunk in client.generate_content_stream(contents):
    print(chunk, end="", flush=True)
```

### Conversation History

```python
# Build conversation history
history = [
    types.Content(
        role="user",
        parts=[types.Part.from_text(text="Hello")]
    ),
    types.Content(
        role="model",
        parts=[types.Part.from_text(text="Hi! How can I help you?")]
    ),
    types.Content(
        role="user",
        parts=[types.Part.from_text(text="What's the weather like?")]
    )
]

response = client.generate_content(history)
```

### Override Parameters

You can override config parameters per request:

```python
response = client.generate_content(
    contents,
    temperature=0.7,  # Override config temperature
    max_output_tokens=1000,  # Override config max_output_tokens
    system_instruction="You are a helpful assistant."  # Override config system_instruction
)
```

### Generation with Retry

```python
# Automatically retries on failure with exponential backoff
response = client.generate_with_retry(
    contents,
    max_retries=5,
    streaming=False
)
```

## Token Counting

```python
contents = [
    types.Content(
        role="user",
        parts=[types.Part.from_text(text="Count the tokens in this message.")]
    )
]

token_count = client.count_tokens(contents)
print(f"Total tokens: {token_count}")

# With system instruction
token_count = client.count_tokens(
    contents,
    system_instruction="You are a helpful assistant."
)
```

## Batch Processing

### Upload Files to GCS

```python
from greycloud import GreyCloudBatch

batch = GreyCloudBatch()

# Upload a single file
gcs_uri = batch.upload_file_to_gcs(
    file_path="data.txt",
    blob_name="data.txt",
    bucket_name="my-bucket"
)

# Upload string content
gcs_uri = batch.upload_string_to_gcs(
    content="File content here",
    blob_name="file.txt",
    bucket_name="my-bucket"
)

# Upload multiple files
files = [
    {"name": "file1.txt", "content": "Content 1"},
    {"name": "file2.txt", "content": "Content 2"}
]
file_uris = batch.upload_files_to_gcs(files, bucket_name="my-bucket")
```

### Create Batch Job

```python
from greycloud import GreyCloudBatch
from google.genai import types

batch = GreyCloudBatch()

# Create batch requests
batch_requests = [
    types.InlinedRequest(
        model="gemini-3-pro-preview",
        contents=[
            {
                "role": "user",
                "parts": [
                    {"text": "Analyze this data: "},
                    {"file_data": {"file_uri": "gs://bucket/file1.txt", "mime_type": "text/plain"}}
                ]
            }
        ],
        config=types.GenerateContentConfig(
            temperature=0.2,
            max_output_tokens=65535
        )
    )
]

# Create batch job
batch_job = batch.create_batch_job(
    batch_requests,
    model="gemini-3-pro-preview"
)
print(f"Batch job created: {batch_job.name}")
```

### Monitor Batch Job

```python
# Monitor with default polling interval
batch_job = batch.monitor_batch_job(batch_job)

# Monitor with custom polling interval and callback
def status_callback(job, state):
    print(f"Batch job state: {state}")

batch_job = batch.monitor_batch_job(
    batch_job,
    poll_interval=60,  # Poll every 60 seconds
    callback=status_callback
)
```

### Download Batch Results

```python
# Download results to file
output_file = batch.download_batch_results(
    batch_job,
    output_file="results.jsonl",
    bucket_name="my-bucket"
)
```

### Complete Batch Workflow

```python
from greycloud import GreyCloudConfig, GreyCloudBatch
from google.genai import types

# Configure
config = GreyCloudConfig(
    project_id="your-project-id",
    batch_gcs_bucket="your-project-batch-jobs"
)

batch = GreyCloudBatch(config)

# 1. Upload files to GCS
files = [
    {"name": "data1.json", "content": json.dumps({"key": "value"})},
    {"name": "data2.json", "content": json.dumps({"key2": "value2"})}
]
file_uris = batch.upload_files_to_gcs(files)

# 2. Create batch requests with file references
batch_requests = []
for filename, gcs_uri in file_uris.items():
    batch_requests.append(
        types.InlinedRequest(
            model="gemini-3-pro-preview",
            contents=[
                {
                    "role": "user",
                    "parts": [
                        {"text": f"Analyze {filename}: "},
                        {"file_data": {"file_uri": gcs_uri, "mime_type": "application/json"}}
                    ]
                }
            ],
            config=types.GenerateContentConfig(
                temperature=0.2,
                max_output_tokens=65535
            )
        )
    )

# 3. Create batch job
batch_job = batch.create_batch_job(batch_requests)

# 4. Monitor batch job
batch_job = batch.monitor_batch_job(batch_job)

# 5. Download results
output_file = batch.download_batch_results(batch_job, "results.jsonl")
```

## Advanced Usage

### Custom Authentication

```python
from greycloud.auth import create_client

# Create client with custom authentication
client = create_client(
    project_id="your-project-id",
    location="us-east4",
    sa_email="service-account@project.iam.gserviceaccount.com",
    use_api_key=False,
    endpoint="https://aiplatform.googleapis.com",
    api_version="v1",
    auto_reauth=True
)
```

### Error Handling

```python
from greycloud import GreyCloudClient

client = GreyCloudClient()

try:
    response = client.generate_content(contents)
except RuntimeError as e:
    if "authentication" in str(e).lower():
        print("Authentication error. Please run: gcloud auth application-default login")
    else:
        print(f"Error: {e}")
```

### Retry Logic

The `generate_with_retry` method automatically handles:
- Authentication errors (re-authenticates if `auto_reauth=True`)
- Network errors (exponential backoff with jitter)
- Rate limiting (automatic retries)

```python
response = client.generate_with_retry(
    contents,
    max_retries=5,
    streaming=False
)
```

## Best Practices

1. **Reuse Clients**: Create a single `GreyCloudClient` instance and reuse it for multiple requests.

2. **Configuration Management**: Use environment variables for sensitive configuration (project IDs, service account emails).

3. **Error Handling**: Always wrap API calls in try-except blocks and handle authentication errors appropriately.

4. **Token Management**: Use `count_tokens` to monitor context window usage, especially for long conversations.

5. **Batch Processing**: Use batch jobs for long-running or high-volume tasks to avoid timeout issues.

6. **File References**: For large files, upload to GCS and reference them in batch requests rather than including content inline.

## Troubleshooting

### Authentication Errors

If you encounter authentication errors:

1. Ensure you're logged into gcloud:
   ```bash
   gcloud auth application-default login
   ```

2. Check service account permissions:
   ```bash
   gcloud projects get-iam-policy PROJECT_ID
   ```

3. Verify service account exists:
   ```bash
   gcloud iam service-accounts list
   ```

### Batch Job Issues

- Batch jobs require `global` location (automatically handled by `GreyCloudBatch`)
- Ensure GCS bucket exists and is accessible
- Check batch job status in Google Cloud Console

### Import Errors

If you get import errors:

```bash
pip install google-genai google-auth google-cloud-storage
```

## API Reference

### GreyCloudConfig

Main configuration class. See configuration section above for all parameters.

### GreyCloudClient

- `generate_content(contents, **kwargs)` - Generate content (non-streaming)
- `generate_content_stream(contents, **kwargs)` - Generate content (streaming)
- `count_tokens(contents, system_instruction=None, model=None)` - Count tokens
- `generate_with_retry(contents, max_retries=5, streaming=False, **kwargs)` - Generate with automatic retry

### GreyCloudBatch

- `upload_file_to_gcs(file_path, blob_name=None, bucket_name=None, content_type=None)` - Upload file to GCS
- `upload_string_to_gcs(content, blob_name, bucket_name=None, content_type="text/plain")` - Upload string to GCS
- `upload_files_to_gcs(files, bucket_name=None, prefix="batch_files")` - Upload multiple files
- `upload_batch_requests_to_gcs(batch_requests, bucket_name=None)` - Upload batch requests
- `create_batch_job(batch_requests, model=None, bucket_name=None, results_prefix=None)` - Create batch job
- `monitor_batch_job(batch_job, poll_interval=None, callback=None)` - Monitor batch job
- `download_batch_results(batch_job, output_file, bucket_name=None)` - Download results
