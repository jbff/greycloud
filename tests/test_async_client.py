"""Tests for GreyCloudAsyncClient"""

import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from google.genai import types

from greycloud.async_client import GreyCloudAsyncClient
from greycloud.config import GreyCloudConfig


@pytest.fixture
def async_sample_config():
    """Sample config for async client tests"""
    import subprocess
    with patch.object(subprocess, "run") as mock_run:
        mock_run.return_value.returncode = 0
        mock_run.return_value.stdout = "test-project-id"
        config = GreyCloudConfig(
            project_id="test-project-id",
            location="us-east4",
            use_api_key=False,
            sa_email=None,
        )
    return config


@pytest.fixture
def mock_async_genai_client():
    """Mock genai.Client with aio surface"""
    client = MagicMock()
    client.aio = MagicMock()
    client.aio.models = MagicMock()
    client.aio.models.generate_content = AsyncMock()
    client.aio.models.generate_content_stream = AsyncMock()
    client.aio.models.count_tokens = AsyncMock()
    return client


class TestGreyCloudAsyncClientInit:
    """Tests for initialization"""

    def test_init_with_config(self, async_sample_config):
        """Initialize with provided config"""
        with patch("greycloud.async_client.create_client"):
            client = GreyCloudAsyncClient(async_sample_config)
            assert client.config == async_sample_config

    def test_init_creates_rate_limiter(self, async_sample_config):
        """Rate limiter is created with defaults"""
        with patch("greycloud.async_client.create_client"):
            client = GreyCloudAsyncClient(async_sample_config)
            assert client.rate_limiter is not None
            assert client.rate_limiter.rpm == 60

    def test_init_custom_rate_limits(self, async_sample_config):
        """Custom rate limits are passed to limiter"""
        with patch("greycloud.async_client.create_client"):
            client = GreyCloudAsyncClient(
                async_sample_config, rpm=30, tpm=100_000, max_concurrency=5
            )
            assert client.rate_limiter.rpm == 30
            assert client.rate_limiter.tpm == 100_000
            assert client.rate_limiter.max_concurrency == 5


class TestGreyCloudAsyncClientGenerate:
    """Tests for async generate_content"""

    @pytest.mark.asyncio
    async def test_generate_content(self, async_sample_config, mock_async_genai_client):
        """generate_content returns response"""
        mock_response = MagicMock()
        mock_response.text = "Hello world"
        mock_async_genai_client.aio.models.generate_content.return_value = mock_response

        with patch("greycloud.async_client.create_client", return_value=mock_async_genai_client):
            client = GreyCloudAsyncClient(async_sample_config)
            contents = [
                types.Content(role="user", parts=[types.Part.from_text(text="Hi")])
            ]
            result = await client.generate_content(contents)
            assert result.text == "Hello world"
            mock_async_genai_client.aio.models.generate_content.assert_called_once()

    @pytest.mark.asyncio
    async def test_generate_content_with_rate_limiting(self, async_sample_config, mock_async_genai_client):
        """generate_content goes through rate limiter"""
        mock_response = MagicMock()
        mock_async_genai_client.aio.models.generate_content.return_value = mock_response

        with patch("greycloud.async_client.create_client", return_value=mock_async_genai_client):
            client = GreyCloudAsyncClient(async_sample_config)
            with patch.object(client.rate_limiter, "call_with_limits", new_callable=AsyncMock) as mock_limiter:
                mock_limiter.return_value = mock_response
                contents = [
                    types.Content(role="user", parts=[types.Part.from_text(text="Hi")])
                ]
                result = await client.generate_content(contents)
                mock_limiter.assert_called_once()
                assert result == mock_response


class TestGreyCloudAsyncClientCountTokens:
    """Tests for async count_tokens"""

    @pytest.mark.asyncio
    async def test_count_tokens(self, async_sample_config, mock_async_genai_client):
        """count_tokens returns token count"""
        mock_token_response = MagicMock()
        mock_token_response.total_tokens = 42
        mock_async_genai_client.aio.models.count_tokens.return_value = mock_token_response

        with patch("greycloud.async_client.create_client", return_value=mock_async_genai_client):
            client = GreyCloudAsyncClient(async_sample_config)
            contents = [
                types.Content(role="user", parts=[types.Part.from_text(text="Hello")])
            ]
            count = await client.count_tokens(contents)
            assert count == 42

    @pytest.mark.asyncio
    async def test_count_tokens_fallback(self, async_sample_config, mock_async_genai_client):
        """count_tokens falls back to character estimate on failure"""
        mock_async_genai_client.aio.models.count_tokens.side_effect = Exception("API down")

        with patch("greycloud.async_client.create_client", return_value=mock_async_genai_client):
            client = GreyCloudAsyncClient(async_sample_config)
            contents = [
                types.Content(role="user", parts=[types.Part.from_text(text="a" * 400)])
            ]
            count = await client.count_tokens(contents)
            assert count == 100  # 400 chars // 4


class TestGreyCloudAsyncClientRetry:
    """Tests for async generate_with_retry"""

    @pytest.mark.asyncio
    async def test_retry_on_failure(self, async_sample_config, mock_async_genai_client):
        """Retries on transient error then succeeds"""
        mock_response = MagicMock()
        mock_response.text = "Success"
        mock_async_genai_client.aio.models.generate_content.side_effect = [
            RuntimeError("429 Too Many Requests"),
            mock_response,
        ]

        with patch("greycloud.async_client.create_client", return_value=mock_async_genai_client):
            client = GreyCloudAsyncClient(async_sample_config)
            contents = [
                types.Content(role="user", parts=[types.Part.from_text(text="Hi")])
            ]
            result = await client.generate_with_retry(contents, max_retries=3)
            assert result.text == "Success"
            assert mock_async_genai_client.aio.models.generate_content.call_count == 2

    @pytest.mark.asyncio
    async def test_retry_exhausted_raises(self, async_sample_config, mock_async_genai_client):
        """Raises after max retries exhausted"""
        mock_async_genai_client.aio.models.generate_content.side_effect = RuntimeError("500 Server Error")

        with patch("greycloud.async_client.create_client", return_value=mock_async_genai_client):
            client = GreyCloudAsyncClient(async_sample_config)
            contents = [
                types.Content(role="user", parts=[types.Part.from_text(text="Hi")])
            ]
            with pytest.raises(RuntimeError, match="500 Server Error"):
                await client.generate_with_retry(contents, max_retries=2)
            assert mock_async_genai_client.aio.models.generate_content.call_count == 3
