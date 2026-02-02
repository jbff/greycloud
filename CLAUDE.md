# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

GreyCloud is a Python package providing resilient, config-driven wrappers around Google's `google-genai` client for Vertex AI and GenAI services (Gemini). Key features: unified authentication (API key or OAuth with optional service account impersonation), automatic retry with exponential backoff, context caching for cost savings, and batch processing with GCS integration.

## Development Commands

```bash
# Install in editable mode with dev dependencies
pip install -e ".[dev]"

# Run all tests
pytest

# Run tests with coverage (HTML report)
pytest --cov=greycloud --cov-report=html

# Run specific test file
pytest tests/test_config.py

# Run specific test class or method
pytest tests/test_client.py::TestGreyCloudClient::test_generate_content

# Run tests by marker
pytest -m auth      # Authentication tests
pytest -m batch     # Batch processing tests

# Code formatting
black greycloud/ tests/

# Linting
flake8 greycloud/ tests/

# Type checking
mypy greycloud/
```

## Architecture

Five main modules in `greycloud/`:

- **`config.py`** - `GreyCloudConfig` dataclass: centralizes all configuration (auth, generation params, batch settings). Reads from environment variables with sensible defaults.

- **`auth.py`** - `create_client()` factory: creates authenticated `genai.Client` instances. Supports API key auth or OAuth with optional service account impersonation. Has fallback chain: default credentials → SA impersonation → gcloud CLI → auto-login prompt.

- **`client.py`** - `GreyCloudClient`: main interface for content generation. Key methods: `generate_content()`, `generate_content_stream()`, `generate_with_retry()` (exponential backoff with re-auth), `count_tokens()` (with character-based fallback). Supports `cached_content` parameter for using context caches.

- **`cache.py`** - `GreyCloudCache`: context caching for cost-efficient repeated queries. Key methods: `create_cache()`, `create_cache_from_text()`, `create_cache_from_files()`, `list_caches()`, `get_cache()`, `update_cache_ttl()`, `delete_cache()`, `generate_with_cache()`. Provides 75-90% discount on cached input tokens; storage costs $1/million tokens/hour.

- **`batch.py`** - `GreyCloudBatch`: batch job utilities. Handles GCS upload/download, batch job creation with JSONL input, job monitoring with polling. Batch API always uses `global` location.

## Key Environment Variables

| Variable | Purpose |
|----------|---------|
| `PROJECT_ID` or `GCP_PROJECT` | GCP project ID (required) |
| `LOCATION` or `GCP_LOCATION` | Default region (default: `us-east4`) |
| `USE_API_KEY` | Use API key auth instead of OAuth |
| `SA_EMAIL` | Service account email for impersonation |
| `BATCH_GCS_BUCKET` | GCS bucket for batch operations |

## Testing Notes

- All tests use mocks - no real Google Cloud credentials required
- Test fixtures in `conftest.py` automatically set `PROJECT_ID` and `LOCATION`
- Pytest markers available: `unit`, `integration`, `auth`, `batch`, `cache`
- The `filterwarnings` in `pyproject.toml` suppresses known google-genai deprecation warnings

## Design Patterns

- Single `GreyCloudConfig` dataclass holds all configuration
- All generation parameters can be overridden per-request
- Retry logic includes automatic re-authentication on auth failures
- Token counting gracefully falls back to character-based approximation when API unavailable
- Batch JSONL format expects `{"request": {"model": ..., "contents": ..., "config": ...}}` per line
