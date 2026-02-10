# GreyCloud updates plan (for greycloud repo)

This document describes changes to make in the **greycloud** package (e.g. in `../greycloud` or the canonical greycloud repo) so that the async rate-limited client is a true mirror of the sync client. Consumers (e.g. BlackSheep) can then switch between sync and async interfaces with analogous classes and calls.

---

## Goal

- **Parity**: `GreyCloudAsyncClient` should offer the same capabilities as `GreyCloudClient`, with analogous method names and signatures, so users can choose sync vs async without losing features.
- **Rate limiting**: The async client continues to use `VertexRateLimiter` (RPM, TPM, concurrency) for all generation and token-count calls.
- **Consistency**: Config building (tools, safety_settings, thinking_config, cached_content, etc.) should match between sync and async clients.

---

## 1. Config building parity in GreyCloudAsyncClient

The sync client builds `GenerateContentConfig` with:

- `temperature`, `top_p`, `max_output_tokens`
- `safety_settings` (list of SafetySetting or dicts)
- `seed`
- `system_instruction`
- `thinking_config` (from `config.thinking_level`)
- `tools` (from `_build_tools()` when `use_vertex_ai_search` and `vertex_ai_search_datastore` are set)
- `cached_content` (when provided)

**Action**: Update `GreyCloudAsyncClient._build_generate_config()` to include:

- **Tools**: When `self.config.use_vertex_ai_search` and `self.config.vertex_ai_search_datastore` are set, add the same `types.Tool(retrieval=types.Retrieval(vertex_ai_search=...))` as in sync `GreyCloudClient._build_tools()`.
- **Safety settings**: Same handling as sync client (None vs empty list vs list of dicts/SafetySetting), and add to config dict when present.

This keeps behavior identical for Vertex AI Search and safety when switching from sync to async.

---

## 2. Streaming generation in GreyCloudAsyncClient

The sync client has:

- `generate_content(contents, ...)` → single response
- `generate_content_stream(contents, ...)` → generator of text chunks
- `generate_with_retry(contents, ..., streaming=True|False)` → either full response or generator of chunks

**Action**: Add analogous async APIs:

- **`async def generate_content_stream(self, contents, model=None, system_instruction=None, ...)**  
  - Async generator that yields `str` chunks.
  - Estimate prompt tokens; call `await self.rate_limiter.call_with_limits(token_est, ...)` with a coroutine that starts the stream. Implementation option: acquire rate limits (RPM, TPM, semaphore), then `async for chunk in self._client.aio.models.generate_content_stream(...)`, yield `chunk.text` when present (same logic as sync client for extracting text).
  - Use `_build_generate_config()` for config (including tools and safety after step 1).
  - Pass through `model`, `system_instruction`, `temperature`, `top_p`, `max_output_tokens`, `thinking_level`, `cached_content`, and any kwargs used by sync.

- **`async def generate_with_retry(self, ..., streaming: bool = False)`**  
  - Today the async client has `generate_with_retry` only for non-streaming.
  - Extend it so that when `streaming=True`, it returns an async generator that uses `generate_content_stream` (with retry logic if desired). Retry for streaming can be “on exception, re-acquire and restart stream” (same as sync’s collect-then-yield approach). Signature should mirror sync: `generate_with_retry(contents, max_retries=5, streaming=False, **generate_kwargs)` returning either `GenerateContentResponse` or an async generator of `str`.

After this, the mapping is:

| Sync (`GreyCloudClient`)     | Async (`GreyCloudAsyncClient`)        |
|-----------------------------|----------------------------------------|
| `generate_content(...)`     | `await generate_content(...)`          |
| `generate_content_stream(...)` | `async for ... in generate_content_stream(...)` |
| `generate_with_retry(..., streaming=False)` | `await generate_with_retry(...)`  |
| `generate_with_retry(..., streaming=True)`  | `async for ... in generate_with_retry(..., streaming=True)` |
| `count_tokens(...)`         | `await count_tokens(...)`             |

---

## 3. count_tokens and other non-generation APIs

- **count_tokens**: Already async in `GreyCloudAsyncClient`. Ensure it accepts the same arguments as sync (`contents`, `system_instruction`, `model`) and returns `int`. No change if already aligned.
- **client property**: Sync client exposes `.client` (genai.Client). Async client could expose `.client` for advanced use (same underlying client); document that rate-limited generation should go through the client’s methods, not raw `.client.aio.models.*`, so limits are applied.

---

## 4. __init__ and rate limiter parameters

- Keep `GreyCloudAsyncClient.__init__(config=None, rpm=60, tpm=250_000, max_concurrency=10)` so callers can tune rate limits.
- Optionally support reading `rpm`/`tpm`/`max_concurrency` from `GreyCloudConfig` (e.g. optional attributes) so a single config object can drive both client types. Not required for parity; env vars or explicit args are enough.

---

## 5. Retry and auth behavior (optional alignment)

- Sync `generate_with_retry` handles auth errors (e.g. re-auth via gcloud) and exponential backoff. Async `generate_with_retry` currently does backoff only. For full parity, consider detecting auth errors in the async path and applying the same re-auth logic (or document that async client assumes valid credentials and callers handle 401/403 separately). Lower priority if BlackSheep only needs rate limiting and backoff.

---

## 6. Tests in greycloud

- Add tests for `generate_content_stream`: mock `_client.aio.models.generate_content_stream` to return an async generator of chunk objects; assert rate limiter is used (e.g. `call_with_limits` called with token estimate) and that yielded text matches.
- Add tests for `generate_with_retry(..., streaming=True)` (if implemented): same as above, plus retry on exception.
- Add tests for config building: when `use_vertex_ai_search` and `vertex_ai_search_datastore` are set, assert generated config includes tools; when `safety_settings` is set, assert they appear in config.
- Keep existing tests for async `generate_content` and `generate_with_retry(streaming=False)` and `count_tokens`.

---

## 7. Exports and docs

- `__all__` in `greycloud/__init__.py` already exports `GreyCloudAsyncClient` and `VertexRateLimiter`.
- In greycloud README or docstring, add a short “Sync vs async” section: same config, same method names; use `GreyCloudClient` for sync and `GreyCloudAsyncClient` for async/rate-limited usage, with the table above.

---

## Summary checklist (greycloud repo)

- [ ] Add tools (Vertex AI Search) and safety_settings to `GreyCloudAsyncClient._build_generate_config()`.
- [ ] Implement `generate_content_stream` (async generator, rate-limited, same kwargs as sync).
- [ ] Extend `generate_with_retry` to support `streaming=True` (async generator) with same signature shape as sync.
- [ ] Align `count_tokens` and any other non-generation APIs with sync signatures.
- [ ] Add tests for async streaming and for config (tools, safety).
- [ ] Document sync/async parity (README or docstrings).

After these updates, BlackSheep (or any consumer) can depend on greycloud and use `GreyCloudAsyncClient` for all generation with the same options as the sync client, plus rate limiting.
