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
    - Vertex AI Search datastore + grounding mode (`"inject"` | `"tool"`)
    - Grounding skip threshold (`min_grounding_query_chars`)
    - Batch/GCS bucket settings

- **Resilient generation**
  - `GreyCloudClient.generate_with_retry(...)`:
    - Detects auth-related vs transient errors
    - Performs exponential backoff with jitter
    - Attempts re-authentication when appropriate (for OAuth-based flows)
    - Re-creates the underlying `genai.Client` as needed

- **Vertex AI Search grounding**
  - Vertex AI Search is turned on with:
    - `use_vertex_ai_search=True`
    - `vertex_ai_search_datastore="projects/.../dataStores/..."`.
  - Grounding mode (`grounding_mode`, default `"inject"`):
    - `"inject"` (default): GreyCloud runs the Discovery Engine search itself with your existing credentials and injects the top results as an attributed `<grounding_sources>` block into the prompt. This works with **every** model version — the server-side `tools.retrieval` grounding silently returns zero chunks on Gemini 3.x (see §5.8).
    - `"tool"`: legacy behavior — GreyCloud constructs the `types.Tool(retrieval=...)` and wires it into calls (for model versions where it still works, e.g. gemini-2.5).
  - Per-call overrides on `generate_content` / `generate_content_stream` (both clients, and via `generate_with_retry`):
    - `grounding_query="..."` — search this string instead of the verbatim last user message (the last turn is often a workflow instruction, not content); the block is still injected into the last user message.
    - `grounding=False` — skip grounding entirely for this call (no search, no injection).
    - `on_grounding=callback` — invoked once per generate with the exact list of `GroundingSource` being injected (`[]` when the search ran but found nothing); not invoked when grounding is skipped entirely. Coroutine callbacks are awaited on the async client; callback exceptions are logged at WARNING and never propagate.
  - `min_grounding_query_chars` (default `0`): when > 0, inject-mode grounding skips the search when the effective query is shorter than this many characters.
  - `extractive_content_spec` (default `False`): when `True`, the `:search` request additionally asks for paragraph-scale extractive answers (`extractiveContentSpec: {maxExtractiveAnswerCount: 2}`); results prefer an extractive answer per source, falling back to the keyword snippet. **Opt-in** — datastores created with chunking config reject the field with HTTP 400, so the snippets-only default (accepted by every datastore type) is what 0.3.15+ sends unless you know your datastore supports extractive answers. If a chunking-config datastore rejects the request, the failure is logged at ERROR with a hint to disable the flag.

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

By default, streaming yields plain text strings representing the generated content chunks:

```python
for chunk in client.generate_content_stream(contents):
    print(chunk, end="", flush=True)
```

If you need access to candidate metadata, usage metrics, or safety ratings, you can pass `return_chunks=True` to yield the raw `GenerateContentResponse` chunk objects instead of strings:

```python
for chunk in client.generate_content_stream(contents, return_chunks=True):
    # chunk is a google.genai.types.GenerateContentResponse object
    print(chunk.text, end="", flush=True)
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

To stream raw response chunk objects with retry logic, pass `return_chunks=True`:

```python
for chunk in client.generate_with_retry(
    contents,
    max_retries=5,
    streaming=True,
    return_chunks=True,
):
    print(chunk.text, end="", flush=True)
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
        parts=[types.Part.from_text(text="Using the knowledge base, explain the benefits of renewable energy.")]
    )
]

response = client.generate_content(contents)
print(response.text)
```

With `use_vertex_ai_search=True`, the default `grounding_mode="inject"` runs a GreyCloud-side Discovery Engine search and injects the top results as a `<grounding_sources>` block into the prompt (the legacy `tools.retrieval` tool is dropped). Set `grounding_mode="tool"` to keep the legacy `tools.retrieval` behavior instead. The active path is logged at INFO level per request — enable INFO logging to see whether `grounding_mode=inject` or `tool` is in use.

By default the search query is the verbatim text of the last user message. Pass `grounding_query` to search a distilled query instead — useful when the last turn is a workflow instruction ("Great, now let's do the summary.") rather than content — and `grounding=False` to skip grounding for a single call:

```python
# Ground a template/step turn with a distilled clinical (or domain) query:
response = client.generate_content(
    contents, grounding_query="ABAS functional impairment adult collateral reports"
)

# A conversational turn: no search at all.
response = client.generate_content(contents, grounding=False)
```

For automatic suppression of short conversational turns, set `min_grounding_query_chars` in `GreyCloudConfig` (default `0`, off): when the effective query — `grounding_query` if given, else the last user message — is shorter than the threshold, the search is skipped and generation proceeds ungrounded.

**Changelog note (0.3.15):** 0.3.12–0.3.14 requested paragraph-scale extractive answers unconditionally, which broke search for any datastore built with *chunking config* (HTTP 400: `max_extractive_answer_count must be not specified when the datastore is using 'chunking config'`). From 0.3.15 the extractive spec is opt-in via `extractive_content_spec=True` and the snippets-only payload (the pre-0.3.12 wire behavior, accepted by every datastore type) is the default. Callers on chunked datastores need no action; callers wanting extractive answers set `extractive_content_spec=True` in `GreyCloudConfig`.

To see exactly which sources informed a response (e.g. to show the clinician the reference material behind a citation), pass `on_grounding`, a callback receiving the list of `GroundingSource` objects that are being injected — fired once per generate, after the search decision and before the model call. It fires with an empty list when the search ran but found nothing, and does not fire when grounding was skipped entirely (`grounding=False`, threshold skip, tools override, or inject mode disabled):

```python
def show_sources(sources):
    for s in sources:
        print(f"[{s.index}] {s.title} — {s.link}")

response = client.generate_content(contents, on_grounding=show_sources)
```

### 5.9 Batch Processing with GCS

Batch jobs use a GCS bucket for request input and result output. Set `batch_gcs_bucket` (and optionally `gcs_bucket` for general uploads). The batch API expects JSONL input following the Vertex AI REST `GenerateContentRequest` schema: one line per request, each line a JSON object with a `request` key containing `model`, `contents`, and optional `generationConfig`, `systemInstruction`, and `safetySettings` (all camelCase). Note: `InlinedRequest.metadata` is **not** forwarded to batch JSONL because Vertex rejects numeric string values in proto label fields — use prompt-embedded ID tags (e.g. `[SLICE_ID:...]`) for request matching. Results are written by Vertex to `predictions.jsonl` under the job's destination prefix; `download_batch_results` finds and downloads that file.

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
