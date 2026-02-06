## GreyCloud

A comprehensive, configurable Python package for interacting with Google's Vertex AI and GenAI services (Gemini), including authentication, content generation, batch processing, token counting, and file management.

GreyCloud wraps the lower-level `google-genai` client with:

- **Unified authentication** (API key or OAuth + optional service account impersonation)
- **Resilient content generation** with automatic retry and re-authentication
- **Config-driven client setup** via a single `GreyCloudConfig` dataclass
- **Context caching** for 75-90% cost savings on repeated queries
- **Optional Vertex AI Search tools** for retrieval-augmented generation
- **Batch helpers** for large offline jobs and GCS integration

---

## 1. What GreyCloud Does

GreyCloud provides four main building blocks:

- `GreyCloudConfig` – configuration object populated from environment variables or code
- `GreyCloudClient` – high-level client for content generation, streaming, token counting, and retries
- `GreyCloudCache` – context caching for cost-efficient repeated queries on the same content
- `GreyCloudBatch` – helper for batch jobs and GCS-backed workflows

High-level capabilities:

- **Content generation** (streaming and non-streaming) with per-request overrides
- **Automatic retry** with exponential backoff and authentication-aware recovery
- **Context caching** with 75-90% cost savings on cached input tokens
- **Token counting** with graceful approximation fallback
- **Vertex AI Search integration** via a simple flag and datastore string
- **Batch processing** to upload files, create jobs, monitor, and download results

---

## 2. Why Use GreyCloud Instead of `google-genai` Directly?

Using `google-genai` directly is flexible but verbose. GreyCloud focuses on **developer ergonomics** and **resilience**:

- **Unified auth helper**
  - One function (`create_client` / `GreyCloudClient`) that:
    - Uses Application Default Credentials when available
    - Optionally impersonates a service account when `sa_email` is set
    - Falls back to `gcloud auth print-access-token` when needed
    - Supports API key authentication via a simple config flag
  - Clear error messages that point to:
    - `gcloud auth application-default login`
    - IAM role requirements for impersonation

- **Config normalization**
  - A single dataclass (`GreyCloudConfig`) encapsulates:
    - Project, location, endpoint, model
    - Auth choices (API key vs OAuth + SA impersonation)
    - Generation parameters (temperature, top_p, max_output_tokens, seed)
    - Safety settings
    - Thinking configuration
    - Vertex AI Search datastore
    - Batch/GCS bucket settings

- **Resilient generation**
  - `GreyCloudClient.generate_with_retry(...)`:
    - Detects auth-related vs transient errors
    - Performs exponential backoff with jitter
    - Attempts re-authentication when appropriate (for OAuth-based flows)
    - Re-creates the underlying `genai.Client` as needed

- **Tools & Search wiring**
  - Vertex AI Search is turned on with:
    - `use_vertex_ai_search=True`
    - `vertex_ai_search_datastore="projects/.../dataStores/..."`.
  - GreyCloud constructs the appropriate `types.Tool` and wires it into calls.

- **Batch utilities**
  - `GreyCloudBatch` wraps the more verbose raw batch APIs:
    - Handles JSONL creation
    - Manages GCS paths and result locations
    - Tries multiple model naming formats (`publishers/google/models/...` vs short name)

- **Sync vs async**
  - Same config (`GreyCloudConfig`) and same method names for sync and async.
  - Use `GreyCloudClient` for synchronous code; use `GreyCloudAsyncClient` for async/rate-limited usage.
  - The async client applies RPM, TPM, and concurrency limits via `VertexRateLimiter`; use it when you need to stay within quotas (e.g. in web backends).
  - API mapping:

    | Sync (`GreyCloudClient`) | Async (`GreyCloudAsyncClient`) |
    |--------------------------|---------------------------------|
    | `generate_content(...)` | `await generate_content(...)` |
    | `generate_content_stream(...)` | `async for x in generate_content_stream(...)` |
    | `generate_with_retry(..., streaming=False)` | `await generate_with_retry(...)` |
    | `generate_with_retry(..., streaming=True)` | `async for x in (await generate_with_retry(..., streaming=True))` |
    | `count_tokens(...)` | `await count_tokens(...)` |

  - For advanced use the underlying `genai.Client` is available as `.client` on both clients; rate-limited generation should go through the client’s methods, not raw `client.aio.models.*`.

---

## 3. Installation

### Basic Installation

```bash
pip install greycloud
```

### Development Installation

```bash
git clone https://github.com/jbff/greycloud.git
cd greycloud
pip install -e ".[dev]"
```

## 4. Quick Start: Basic Client and Single Call

```python
from greycloud import GreyCloudConfig, GreyCloudClient
from google.genai import types

# Create configuration (override defaults as needed)
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

---

## 5. Detailed Examples

### 5.1 Creating a Client from Environment Only

Environment:

```bash
export PROJECT_ID="your-project-id"
export LOCATION="us-central1"
```

Code:

```python
from greycloud import GreyCloudClient
from google.genai import types

client = GreyCloudClient()  # GreyCloudConfig is created from env

contents = [
    types.Content(
        role="user",
        parts=[types.Part.from_text(text="Summarize the benefits of Vertex AI.")]
    )
]

response = client.generate_content(contents)
print(response.text)
```

### 5.2 Per-Request Overrides

```python
response = client.generate_content(
    contents,
    temperature=0.7,
    max_output_tokens=1024,
    system_instruction="You are a concise technical assistant.",
)
```

### 5.3 Streaming Generation

```python
for chunk in client.generate_content_stream(contents):
    print(chunk, end="", flush=True)
```

### 5.4 Automatic Retry & Auth Recovery

```python
from google.genai import types

contents = [
    types.Content(
        role="user",
        parts=[types.Part.from_text(text="Give me a short creative story about a robot therapist.")]
    )
]

response = client.generate_with_retry(
    contents,
    max_retries=5,
    streaming=False,
)

print(response.text)
```

For streaming with retry:

```python
for chunk in client.generate_with_retry(
    contents,
    max_retries=5,
    streaming=True,
):
    print(chunk, end="", flush=True)
```

### 5.5 Token Counting with Fallback

```python
from google.genai import types

contents = [
    types.Content(
        role="user",
        parts=[types.Part.from_text(text="Count the tokens in this example message.")]
    )
]

token_count = client.count_tokens(
    contents,
    system_instruction="You are a helpful assistant.",
)

print(f"Total tokens: {token_count}")
```

If the underlying API is unavailable, GreyCloud falls back to an approximate character-based count.

### 5.6 Context Caching for Cost Savings

Context caching allows you to cache large content (documents, code, media) and reuse it across multiple requests without re-sending tokens each time. This provides significant cost savings:

- **Cached token discount**: 75-90% off input token costs (depending on model)
- **Storage cost**: $1.00 per million tokens per hour (prorated by minute)

```python
from greycloud import GreyCloudConfig, GreyCloudCache

config = GreyCloudConfig(project_id="your-project-id")
cache_client = GreyCloudCache(config)

# Cache a large document (must meet minimum token threshold: 1,024-4,096 tokens)
large_document = "..." # Your large content here

cache = cache_client.create_cache_from_text(
    text=large_document,
    display_name="my-document-cache",
    system_instruction="You are a helpful document analyst.",
    ttl_seconds=3600,  # 1 hour
)

print(f"Cache created: {cache.name}")
print(f"Cached tokens: {cache.usage_metadata.total_token_count}")

# Query the cache multiple times (each query uses cached tokens at discounted rate)
questions = [
    "Summarize the main points",
    "What are the key findings?",
    "List any recommendations",
]

for question in questions:
    response = cache_client.generate_with_cache(
        cache_name=cache.name,
        prompt=question,
    )
    print(f"Q: {question}")
    print(f"A: {response.text}\n")

# IMPORTANT: Delete cache when done to stop storage charges
cache_client.delete_cache(cache.name)
```

You can also cache GCS files:

```python
cache = cache_client.create_cache_from_files(
    file_uris=[
        "gs://your-bucket/document1.pdf",
        "gs://your-bucket/document2.txt",
    ],
    display_name="multi-file-cache",
    ttl_seconds=7200,  # 2 hours
)
```

Cache management:

```python
# List all caches
for cached_content in cache_client.list_caches():
    info = cache_client.get_cache_info(cached_content)
    print(f"{info['name']}: {info.get('total_token_count', 'N/A')} tokens")

# Extend cache TTL before it expires
cache_client.update_cache_ttl(cache.name, ttl_seconds=7200)

# Delete all caches with a specific display name
cache_client.delete_all_caches(display_name_filter="my-document-cache")
```

**Note**: Context caching is a paid feature and not available in the free tier.

### 5.7 Using Cached Content with GreyCloudClient

You can also use cached content directly with `GreyCloudClient` by passing the `cached_content` parameter:

```python
from greycloud import GreyCloudConfig, GreyCloudClient, GreyCloudCache
from google.genai import types

config = GreyCloudConfig(project_id="your-project-id")

# Create cache
cache_client = GreyCloudCache(config)
cache = cache_client.create_cache_from_text(
    text=large_document,
    display_name="my-cache",
    ttl_seconds=3600,
)

# Use with GreyCloudClient
client = GreyCloudClient(config)

contents = [
    types.Content(
        role="user",
        parts=[types.Part.from_text(text="Summarize the document")]
    )
]

response = client.generate_content(
    contents,
    cached_content=cache.name,  # Use the cache
)

# Streaming also works with cached content
for chunk in client.generate_content_stream(
    contents,
    cached_content=cache.name,
):
    print(chunk, end="", flush=True)

# Clean up
cache_client.delete_cache(cache.name)
```

### 5.8 Vertex AI Search as a Tool

```python
from greycloud import GreyCloudConfig, GreyCloudClient
from google.genai import types

config = GreyCloudConfig(
    project_id="your-project-id",
    location="us-central1",
    use_vertex_ai_search=True,
    vertex_ai_search_datastore=(
        "projects/PROJECT_ID/locations/LOCATION/"
        "collections/default_collection/dataStores/DATASTORE_ID"
    ),
)

client = GreyCloudClient(config)

contents = [
    types.Content(
        role="user",
        parts=[types.Part.from_text(text="Using the knowledge base, explain the diagnostic steps for adult ASD.")]
    )
]

response = client.generate_content(contents)
print(response.text)
```

### 5.9 Batch Processing with GCS

Batch jobs use a GCS bucket for request input and result output. Set `batch_gcs_bucket` (and optionally `gcs_bucket` for general uploads). The batch API expects JSONL input: one line per request, each line a JSON object with a `request` key containing `model`, `contents`, and optional `config`/`metadata`. Results are written by Vertex to `predictions.jsonl` under the job’s destination prefix; `download_batch_results` finds and downloads that file.

```python
from greycloud import GreyCloudConfig, GreyCloudBatch
from google.genai import types
import json

config = GreyCloudConfig(
    project_id="your-project-id",
    batch_gcs_bucket="your-project-batch-jobs",  # Must exist; used for batch I/O
)

batch = GreyCloudBatch(config)

# Upload a couple of JSON docs (use same bucket via bucket_name)
files = [
    {"name": "data1.json", "content": json.dumps({"key": "value"})},
    {"name": "data2.json", "content": json.dumps({"key2": "value2"})},
]

file_uris = batch.upload_files_to_gcs(files, bucket_name=config.batch_gcs_bucket)

batch_requests = []
for filename, gcs_uri in file_uris.items():
    batch_requests.append(
        types.InlinedRequest(
            model=config.model,
            contents=[
                {
                    "role": "user",
                    "parts": [
                        {"text": f"Analyze {filename}: "},
                        {"file_data": {"file_uri": gcs_uri, "mime_type": "application/json"}},
                    ],
                }
            ],
            config=types.GenerateContentConfig(
                temperature=0.2,
                max_output_tokens=65535,
            ),
        )
    )

batch_job = batch.create_batch_job(batch_requests)
batch_job = batch.monitor_batch_job(batch_job)

output_file = batch.download_batch_results(batch_job, "results.jsonl")
print(f"Batch results saved to: {output_file}")
```

### 5.10 Custom Auth (Advanced)

```python
from greycloud.auth import create_client

client = create_client(
    project_id="your-project-id",
    location="us-central1",
    sa_email="service-account@project.iam.gserviceaccount.com",  # Optional
    use_api_key=False,
)
```

---

## Documentation

All usage and configuration details are documented in this `README.md`. For additional examples, see:

- `examples/simple.py` – minimal content-generation script.
- `examples/caching.py` – context caching for cost-efficient repeated queries.

## Requirements

- Python 3.10+
- Google Cloud Project with Vertex AI enabled
- `google-genai` package (installed with `greycloud`)
- `google-auth` package (installed with `greycloud`, for OAuth)
- `google-cloud-storage` package (installed with `greycloud`; only needed if you use batch/GCS helpers)

---

## Testing

Run the test suite:

```bash
pytest
```

Run with coverage:

```bash
pytest --cov=greycloud --cov-report=html
```

## License

MIT License (see `LICENSE` file).

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
