"""Pytest configuration and fixtures"""

import os
import pytest
from unittest.mock import Mock, MagicMock, patch
from typing import Optional

# Set test environment variables
os.environ.setdefault("PROJECT_ID", "test-project-id")
os.environ.setdefault("LOCATION", "us-east4")
os.environ.setdefault("GCP_PROJECT", "test-project-id")
os.environ.setdefault("GCP_LOCATION", "us-east4")


@pytest.fixture
def mock_genai_client():
    """Mock genai.Client"""
    client = MagicMock()
    client.models = MagicMock()
    return client


@pytest.fixture
def mock_storage_client():
    """Mock google.cloud.storage.Client"""
    client = MagicMock()
    bucket = MagicMock()
    blob = MagicMock()
    client.bucket.return_value = bucket
    bucket.blob.return_value = blob
    return client


@pytest.fixture
def mock_credentials():
    """Mock Google credentials"""
    creds = MagicMock()
    creds.token = "test-token"
    creds.expired = False
    creds.valid = True
    return creds


@pytest.fixture
def sample_config():
    """Sample GreyCloudConfig for testing"""
    from greycloud.config import GreyCloudConfig
    import subprocess
    
    with patch.object(subprocess, 'run') as mock_run:
        mock_run.return_value.returncode = 0
        mock_run.return_value.stdout = "test-project-id"
        config = GreyCloudConfig(
            project_id="test-project-id",
            location="us-east4",
            use_api_key=False,
            sa_email=None
        )
    return config


@pytest.fixture
def sample_config_with_api_key():
    """Sample GreyCloudConfig with API key for testing"""
    from greycloud.config import GreyCloudConfig
    
    with patch('greycloud.config.subprocess') as mock_subprocess:
        mock_subprocess.run.return_value.returncode = 0
        mock_subprocess.run.return_value.stdout = "test-project-id"
        config = GreyCloudConfig(
            project_id="test-project-id",
            location="us-east4",
            use_api_key=True,
            api_key_file="test_api_key.txt"
        )
    return config


@pytest.fixture
def sample_contents():
    """Sample Content objects for testing"""
    from google.genai import types
    
    return [
        types.Content(
            role="user",
            parts=[types.Part.from_text(text="Hello, how are you?")]
        )
    ]


@pytest.fixture
def mock_batch_job():
    """Mock BatchJob object"""
    from google.genai import types
    
    job = MagicMock(spec=types.BatchJob)
    job.name = "projects/test-project/locations/global/batches/test-batch"
    job.state = "JOB_STATE_PENDING"
    job.done = False
    job.error = None
    
    dest = MagicMock()
    dest.gcs_uri = "gs://test-bucket/batch_results/results.jsonl"
    job.dest = dest
    
    return job


@pytest.fixture
def mock_generate_response():
    """Mock GenerateContentResponse"""
    from google.genai import types
    
    response = MagicMock(spec=types.GenerateContentResponse)
    response.text = "This is a test response"
    response.candidates = [MagicMock()]
    response.candidates[0].content = MagicMock()
    response.candidates[0].content.parts = [MagicMock()]
    response.candidates[0].content.parts[0].text = "This is a test response"
    return response
