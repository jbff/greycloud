# AGENTS.md

This file provides core instructions and context for any AI agent (Claude Code, Cursor, Antigravity/Gemini-CLI) working within the GreyCloud repository.

## Critical Instructions: Memory & Task Tracking
This project uses **Beads** (`bd`) for persistent long-term memory and cross-session planning.
- **Initialization:** Before starting any work, run `bd list` to synchronize with the current project state and active tasks.
- **Updates:** After completing a sub-task, resolving a bug, or making a significant architectural decision, use `bd note "..."` or `bd add "..."` to record the progress.
- **Handover:** If you hit a rate limit or session timeout, ensure the final state is documented in `bd` so the next agent can resume seamlessly.

---

## Project Overview
GreyCloud is a Python package providing resilient, config-driven wrappers around Google's `google-genai` client for Vertex AI and GenAI services (Gemini). 

**Key features:**
- Unified authentication (API key or OAuth with optional service account impersonation).
- Automatic retry with exponential backoff.
- Context caching for cost savings.
- Batch processing with GCS integration.

---

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

---

### Versioning & Git Workflow
This project uses a "floating tag" strategy for the current development version and `uv` for publishing.

#### Commit & Tag Strategy
- **Tag Matching:** After every commit, ensure a git tag exists that matches the current version string in `greycloud/__init__.py`.
- **Floating Tags:** If the tag already exists, move it to the latest commit:
  `git tag -f $(python3 -c "import greycloud; print(greycloud.__version__)")`
- **Automatic Push:** Immediately after committing and tagging, push both to the remote:
  `git push origin main --tags`

#### Publishing to PyPI

> **Version history note:** Versions 0.3.0 through 0.3.3 were published with
> mismatched version numbers, broken tags, or other release hygiene issues
> caused by automated tooling errors. **0.3.4 is the first coherent release
> in the 0.3.x series.** We apologize for the mess — please use 0.3.4+.

- **Trigger:** Only publish to PyPI when explicitly requested.
- **Token:** PyPI API token lives in `~/.pypirc`. Pass it via `UV_PUBLISH_TOKEN` env var or let `uv` read `~/.pypirc` directly.
- **Critical order — commit and tag BEFORE building:**
  1. Ensure version matches in all three files: `greycloud/__init__.py`, `pyproject.toml`, `tests/test_init.py`
  2. Run `pytest` — all tests must pass
  3. Commit the release version
  4. Tag that commit: `git tag v<version>`
  5. Build and publish from the tagged commit:
     ```bash
     rm -rf dist/
     uv build
     UV_PUBLISH_TOKEN="$(python3 -c "import configparser; c=configparser.ConfigParser(); c.read('$HOME/.pypirc'); print(c['pypi']['password'])")" uv publish
     ```
  6. **Then** bump patch version (e.g. 0.3.4 → 0.3.5) in all three files
  7. Commit and tag the bump
  8. Push: `git push origin <branch> --tags`

  **If you build before committing, the PyPI artifact won't match the git tag source.**

- **Post-Publish Bump:** Increment patch version in `greycloud/__init__.py`, `pyproject.toml`, **and** `tests/test_init.py`. All three must always match.

#### Execution Summary for Agents
1. Make code changes.
2. Update version in `__init__.py`, `pyproject.toml`, and `tests/test_init.py` (all must match).
3. Run `pytest` — must pass.
4. Commit changes.
5. Tag: `git tag v<version>`.
6. If publishing: `rm -rf dist/ && uv build && uv publish`, then bump version in all three files, commit, tag.
7. Push: `git push origin <branch> --tags`.

---

### Testing Standards & Pre-Commit Requirements
Quality and reliability are maintained through a strict test-driven development (TDD) approach.

- **New Feature Testing:** Whenever new code or logic is added, corresponding tests must be created in the `tests/` directory to ensure coverage.
- **Mocking Requirement:** All new tests must use mocks. Do not write tests that require real Google Cloud credentials or live API calls.
- **Pre-Commit Verification:** Before executing any `git commit`, you must run the test suite and ensure all tests pass. 
  - Run `pytest` to verify the entire project.
  - Run `pytest -m <module>` to verify specific changes.
- **Strict Rule:** Never commit code that causes a test failure or decreases existing test coverage.
```

---

### Markdown pollution avoidance
- **Do not** add/commit every markdown file you create to git repo.
- **DO** add/commit critical documentation that an end user would benefit from, and any markdown files explicitly instructed to track
- You should thus avoid the git repo exploding with temporary/out of date markdown files

---

## Architecture
The logic is contained within five main modules in `greycloud/`:

- **`config.py`** - `GreyCloudConfig` dataclass: Centralizes all configuration (auth, generation params, batch settings). Reads from environment variables with sensible defaults.
- **`auth.py`** - `create_client()` factory: Creates authenticated `genai.Client` instances. Supports API key auth or OAuth. Fallback chain: default credentials → SA impersonation → gcloud CLI → auto-login prompt.
- **`client.py`** - `GreyCloudClient`: Main interface for content generation. Implements `generate_with_retry()` (exponential backoff with re-auth) and `count_tokens()` (with character-based fallback).
- **`cache.py`** - `GreyCloudCache`: Context caching utilities. Provides methods for creating/managing caches from text or files. Note: Provides 75-90% discount on cached input tokens.
- **`batch.py`** - `GreyCloudBatch`: Batch job utilities handling GCS upload/download and JSONL job monitoring.

---

## Key Environment Variables

| Variable | Purpose |
|----------|---------|
| `PROJECT_ID` or `GCP_PROJECT` | GCP project ID (required) |
| `LOCATION` or `GCP_LOCATION` | Default region (default: `us-east4`) |
| `USE_API_KEY` | Use API key auth instead of OAuth |
| `SA_EMAIL` | Service account email for impersonation |
| `BATCH_GCS_BUCKET` | GCS bucket for batch operations |

---

## Testing & Design Standards
- **Mocking:** All tests use mocks; no real GCP credentials are required for the test suite.
- **Markers:** Use `-m unit`, `integration`, `auth`, `batch`, or `cache` to filter tests.
- **Token Management:** Token counting must gracefully fall back to character-based approximation if the API is unreachable.
- **Retry Logic:** Always include automatic re-authentication on auth-related failures within the retry loop.
- **Batch Format:** JSONL input follows the Vertex AI REST schema: `{"request": {"model": ..., "contents": ..., "generationConfig": ..., "systemInstruction": ..., "safetySettings": ...}}`. All field names are camelCase. Metadata/labels are **not** forwarded (Vertex rejects numeric string label values).

## Landing the Plane (Session Completion)

**When ending a work session**, you MUST complete ALL steps below. Work is NOT complete until `git push` succeeds.

**MANDATORY WORKFLOW:**

1. **File issues for remaining work** - Create issues for anything that needs follow-up
2. **Run quality gates** (if code changed) - Tests, linters, builds
3. **Update issue status** - Close finished work, update in-progress items
4. **PUSH TO REMOTE** - This is MANDATORY:
   ```bash
   git pull --rebase
   bd sync
   git push
   git status  # MUST show "up to date with origin"
   ```
5. **Clean up** - Clear stashes, prune remote branches
6. **Verify** - All changes committed AND pushed
7. **Hand off** - Provide context for next session

**CRITICAL RULES:**
- Work is NOT complete until `git push` succeeds
- NEVER stop before pushing - that leaves work stranded locally
- NEVER say "ready to push when you are" - YOU must push
- If push fails, resolve and retry until it succeeds
