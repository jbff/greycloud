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
- **Trigger:** Only publish to PyPI when explicitly requested.
- **Build & Release:** Use `uv` for the publishing flow:
  `uv build ; uv publish`
- **Post-Publish Bump:** Immediately after a successful publish, increment the patch version (0.0.1) in `greycloud/__init__.py`.
- **New Version Cycle:** The very next commit must create the new tag corresponding to this bumped version and move that tag to the latest commit until the next release.

#### Execution Summary for Agents
1. Make code changes.
2. Read `greycloud/__init__.py` to get `current_version`.
3. Commit changes.
4. Update/Create tag: `git tag -f <current_version>`.
5. Push: `git push origin <branch> --tags`.
6. If publishing: Run `uv build ; uv publish`, then bump version in `__init__.py`.

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
- **Batch Format:** JSONL input must follow the structure: `{"request": {"model": ..., "contents": ..., "config": ...}}`.

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
