# Testing Guide

## Running Tests

### Basic Test Execution

```bash
# Run all tests
pytest

# Run with verbose output
pytest -v

# Run specific test file
pytest tests/test_config.py

# Run specific test class
pytest tests/test_config.py::TestGreyCloudConfig

# Run specific test method
pytest tests/test_config.py::TestGreyCloudConfig::test_config_from_environment
```

### Test Coverage

```bash
# Run tests with coverage report
pytest --cov=greycloud --cov-report=term

# Generate HTML coverage report
pytest --cov=greycloud --cov-report=html

# View coverage report
open htmlcov/index.html  # macOS
xdg-open htmlcov/index.html  # Linux
start htmlcov/index.html  # Windows
```

### Test Markers

Tests are organized with markers for selective execution:

```bash
# Run only authentication tests
pytest -m auth

# Run only batch processing tests
pytest -m batch

# Run unit tests (exclude integration)
pytest -m unit

# Run integration tests
pytest -m integration
```

### Test Output Options

```bash
# Show print statements
pytest -s

# Show local variables on failure
pytest -l

# Stop on first failure
pytest -x

# Show slowest tests
pytest --durations=10
```

## Test Structure

### Test Files

- `test_config.py` - Tests for `GreyCloudConfig` class
- `test_auth.py` - Tests for authentication module
- `test_client.py` - Tests for `GreyCloudClient` class
- `test_batch.py` - Tests for `GreyCloudBatch` class
- `test_init.py` - Tests for package initialization

### What Passing Tests Mean

When all tests pass, you can be confident that:

- **Configuration (`GreyCloudConfig`)**
  - Reads configuration correctly from environment variables and explicit arguments
  - Applies sane defaults for model, temperature, top_p, max_output_tokens
  - Leaves safety settings as `None` by default (Vertex uses its own defaults)
  - Does **not** auto-generate service account emails or GCS buckets
  - Validates that a project ID is available (env or `gcloud config`)

- **Authentication (`greycloud.auth`)**
  - Correctly chooses between API key and OAuth-based authentication
  - Uses service account impersonation when `sa_email` is provided
  - Falls back to `gcloud auth application-default login` when appropriate (mocked)
  - Surfaces clear, actionable errors when authentication fails

- **Client (`GreyCloudClient`)**
  - Builds `GenerateContentConfig` correctly from config and per-call overrides
  - Handles safety settings and seed behavior as designed
  - Implements streaming and non-streaming generation flows against a mocked client
  - Implements token counting and approximate fallback when the API fails
  - Implements `generate_with_retry` with exponential backoff and auth-aware retries

- **Batch (`GreyCloudBatch`)**
  - Uploads files/content to GCS using a mocked storage client
  - Builds and uploads JSONL batch request files correctly
  - Creates batch jobs using the correct model and GCS locations (mocked)
  - Monitors jobs and downloads results with sane error handling

- **Package Initialization**
  - Exposes the expected symbols (`GreyCloudConfig`, `GreyCloudClient`, `GreyCloudBatch`)
  - Keeps the version string in sync with `pyproject.toml`

### What Passing Tests Do *Not* Guarantee

Even with 100% pass rate, note the following limitations:

- **No real network calls**
  - All Google Cloud and Vertex AI interactions are mocked.
  - Tests do **not** verify your credentials, IAM permissions, or actual model responses.

- **No real GCS or Batch jobs**
  - `google-cloud-storage` and batch APIs are mocked.
  - Tests do not create real buckets, files, or batch jobs in your project.

- **No end-to-end latency or quota behavior**
  - Retries and backoff are tested for control flow, not for real-world timing or quota limits.

- **No guarantees about specific model outputs**
  - Tests ensure requests are constructed correctly and methods behave as expected.
  - They do **not** assert on the semantic quality or stability of Gemini responses.

- **No coverage of future API changes**
  - If Google changes the behavior or required fields of `google-genai` or Vertex AI APIs,
    code may need updates even if the current tests pass.

### Fixtures

Common fixtures are defined in `conftest.py`:

- `mock_genai_client` - Mock genai.Client
- `mock_storage_client` - Mock GCS client
- `mock_credentials` - Mock Google credentials
- `sample_config` - Sample GreyCloudConfig
- `sample_contents` - Sample Content objects
- `mock_batch_job` - Mock BatchJob
- `mock_generate_response` - Mock GenerateContentResponse

## Writing New Tests

### Example Test

```python
import pytest
from unittest.mock import patch, MagicMock
from greycloud import GreyCloudConfig

class TestNewFeature:
    """Test new feature"""
    
    def test_feature_basic(self, sample_config):
        """Test basic feature functionality"""
        with patch('greycloud.some_module.some_function') as mock_func:
            mock_func.return_value = "expected_result"
            
            result = some_function_under_test()
            
            assert result == "expected_result"
            mock_func.assert_called_once()
```

### Best Practices

1. **Use fixtures** - Reuse common test objects via fixtures
2. **Mock external dependencies** - Mock Google API calls, file I/O, etc.
3. **Test edge cases** - Include tests for error conditions
4. **Use descriptive names** - Test names should describe what they test
5. **One assertion per test** - When possible, focus each test on one behavior
6. **Use markers** - Mark tests appropriately (unit, integration, auth, batch)

## Continuous Integration

Tests should be run in CI/CD pipelines. Example GitHub Actions workflow:

```yaml
name: Tests
on: [push, pull_request]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - uses: actions/setup-python@v2
        with:
          python-version: '3.8'
      - run: pip install -e ".[dev]"
      - run: pytest --cov=greycloud --cov-report=xml
      - uses: codecov/codecov-action@v2
```

## Troubleshooting

### Import Errors

If you get import errors, ensure you've installed the package:

```bash
pip install -e .
```

### Mock Issues

If mocks aren't working as expected, check:
1. Patch path matches the actual import path
2. Mock is applied before the code under test runs
3. Mock return values are set correctly

### Test Failures

Common issues:
- **Authentication errors**: Tests mock authentication, but real credentials might interfere
- **Network timeouts**: All network calls should be mocked
- **File system**: Use `tmp_path` fixture for file operations
