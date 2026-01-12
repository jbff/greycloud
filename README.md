# GreyCloud

A comprehensive, configurable Python package for interacting with Google's Vertex AI and GenAI services, including authentication, content generation, batch processing, and file management.

## Installation

### Basic Installation

```bash
pip install greycloud
```

### With GCS Support (for batch processing)

```bash
pip install greycloud[storage]
```

### Development Installation

```bash
git clone https://github.com/yourusername/greycloud.git
cd greycloud
pip install -e ".[storage]"
pip install -e ".[dev]"
```

## Quick Start

```python
from greycloud import GreyCloudConfig, GreyCloudClient
from google.genai import types

# Create configuration
config = GreyCloudConfig(
    project_id="your-project-id",
    location="us-east4",
    model="gemini-3-pro-preview"
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

## Features

- **Unified Authentication**: Supports both API key and OAuth authentication with automatic re-authentication
- **Content Generation**: Non-streaming and streaming content generation with configurable parameters
- **Batch Processing**: Complete batch job workflow with GCS integration
- **Token Counting**: Accurate token counting with fallback estimation
- **Error Handling**: Automatic retry with exponential backoff
- **Configuration Management**: Flexible configuration via code or environment variables

## Documentation

For detailed documentation, see [GREYCLOUD_USAGE.md](GREYCLOUD_USAGE.md).

## Requirements

- Python 3.8+
- Google Cloud Project with Vertex AI enabled
- `google-genai` package
- `google-auth` package (for OAuth)
- `google-cloud-storage` package (optional, for batch processing)

## Configuration

GreyCloud can be configured via code or environment variables:

```python
# Via code
config = GreyCloudConfig(
    project_id="your-project-id",
    location="us-east4",
    sa_email="service-account@project.iam.gserviceaccount.com",
    use_api_key=False,
    model="gemini-3-pro-preview"
)
```

```bash
# Via environment variables
export PROJECT_ID="your-project-id"
export LOCATION="us-east4"
export SA_EMAIL="service-account@project.iam.gserviceaccount.com"
export USE_API_KEY="0"
```

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

MIT License

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
