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
The single invariant this workflow protects: **a `v<version>` tag exists on exactly the commit that was published to PyPI as that version.** Every published version has one tag, on a commit whose tree says that version; no tag ever points at a commit that was never published.

#### Single-writer rule (hard requirement)
**Never execute — or mutate the state of — a release while any other session, fork, or thread may be working on the same one.** Signs you are racing: new commits appearing under you between checks, a `git status` that is clean but different from what you last saw, tags or version fields you did not change. On any of these: **stop immediately**, make no further commits, pushes, tag deletions, or publishes, and surface the conflict to the maintainer to pick one owner. Reconciling two agents' contradictory release decisions by acting first is prohibited — a release is not a race to win. Releasing requires explicit user authorization in the thread that does it.

#### Tag lifecycle rule
- **A tag is immutable only after its version is published to PyPI.** Until the version exists on PyPI, its tag may be deleted or moved freely (it protects nothing — no consumer exists). Once the version is published, the tag is permanent.
- **Never tag a post-release bump commit.** Bumping the version reserves nothing; the next release's tag is created when that release actually ships. Tagging a bump burns the version and creates permanent version-number holes (this happened to 0.3.13, which was tagged but never published).
- If a `v<version>` tag already exists but that version was never published (e.g. left over from the old tag-the-bump workflow), delete it locally and remotely (`git tag -d v<version> && git push origin :refs/tags/v<version>`) and tag the release commit instead. This is a workflow correction, not a floating tag.

#### Where the version number lives between releases
- After each publish, the tree version is bumped to the next patch and committed **untagged**. Master between releases therefore says the *next* intended version while PyPI's latest is one behind — that is correct and expected.
- All three version locations must always match: `greycloud/__init__.py`, `pyproject.toml`, `tests/test_init.py`.

#### Publishing to PyPI (when explicitly requested)
1. Ensure version matches in all three files: `greycloud/__init__.py`, `pyproject.toml`, `tests/test_init.py`
2. Run `pytest` — all tests must pass
3. Commit any pending changes (code, docs, tests)
4. If `v<version>` already exists and the version is **not** on PyPI: delete the tag locally and remotely, then re-tag the release commit
5. Tag the release commit: `git tag v<version>`
6. Build and publish from the tagged commit:
   ```bash
   rm -rf dist/
   uv build
   UV_PUBLISH_TOKEN="$(python3 -c "import configparser; c=configparser.ConfigParser(); c.read('$HOME/.pypirc'); print(c['pypi']['password'])")" uv publish
   ```
7. **Verify on PyPI:** `curl -s https://pypi.org/pypi/greycloud/json | python3 -c "import json,sys; print(json.load(sys.stdin)['info']['version'])"` must print the released version
8. Bump patch version in all three files, commit, push — **no tag**
9. Push everything: `git push origin <branch> --tags`

**If you build before committing and tagging, the PyPI artifact won't match the git tag source.**

> **Version history note:** Versions 0.3.0 through 0.3.3 were published with
> mismatched version numbers, broken tags, or other release hygiene issues
> caused by automated tooling errors. **0.3.4 was the first coherent release
> in the 0.3.x series.** 0.3.6 additionally failed under an old "floating tag"
> strategy, since removed.
>
> **0.3.13 / 0.3.15 note:** the old workflow tagged post-release bump commits,
> consuming version numbers that were never published (0.3.13 was skipped
> entirely; 0.3.15's tag was deleted and re-created on its actual release
> commit under the current workflow). The tag-lifecycle rule above exists to
> prevent this class of hole.

#### Execution Summary for Agents
1. Make code changes (tests first).
2. Run `pytest` — must pass.
3. Commit changes.
4. When publishing: verify version consistency in all three files, tag `v<version>`, `rm -rf dist/ && uv build && uv publish`, verify on PyPI, bump to next patch in all three files, commit **untagged**, push branch and tags.

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
