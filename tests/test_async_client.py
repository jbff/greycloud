"""Tests for GreyCloudAsyncClient"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from google.genai import types

from greycloud.async_client import GreyCloudAsyncClient
from greycloud.config import GreyCloudConfig


async def _mock_stream_chunks(*chunks):
    """Helper: async generator yielding chunk mocks with .text."""
    for c in chunks:
        yield c


async def _mock_stream_awaitable(*chunks):
    """Coroutine that resolves to an async generator (for API where generate_content_stream is awaited)."""
    return _mock_stream_chunks(*chunks)


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

    def test_client_property(self, async_sample_config):
        """Async client exposes .client for advanced use"""
        with patch("greycloud.async_client.create_client") as mock_create:
            mock_client = MagicMock()
            mock_create.return_value = mock_client
            client = GreyCloudAsyncClient(async_sample_config)
            assert client.client is mock_client


class TestGreyCloudAsyncClientConfigBuilding:
    """Tests for _build_tools and _build_generate_config parity with sync"""

    def test_build_tools_with_vertex_search(self, async_sample_config):
        """When use_vertex_ai_search and datastore set, config includes tools"""
        config = GreyCloudConfig(
            project_id="test-project-id",
            location="us-east4",
            use_vertex_ai_search=True,
            vertex_ai_search_datastore="projects/test/locations/us/datastores/test-ds",
        )
        with patch("greycloud.async_client.create_client"):
            client = GreyCloudAsyncClient(config)
            tools = client._build_tools()
            assert len(tools) == 1
            assert tools[0].retrieval is not None

    def test_build_tools_without_vertex_search(self, async_sample_config):
        """Without vertex search, _build_tools returns empty list"""
        with patch("greycloud.async_client.create_client"):
            client = GreyCloudAsyncClient(async_sample_config)
            tools = client._build_tools()
            assert len(tools) == 0

    def test_build_generate_config_with_safety_settings(self, async_sample_config):
        """Generated config includes safety_settings when provided"""
        safety_settings = [
            {
                "category": "HARM_CATEGORY_HATE_SPEECH",
                "threshold": "BLOCK_MEDIUM_AND_ABOVE",
            }
        ]
        with patch("greycloud.async_client.create_client"):
            client = GreyCloudAsyncClient(async_sample_config)
            config = client._build_generate_config(safety_settings=safety_settings)
            assert len(config.safety_settings) == 1
            assert config.safety_settings[0].category == "HARM_CATEGORY_HATE_SPEECH"


class TestGreyCloudAsyncClientGenerate:
    """Tests for async generate_content"""

    @pytest.mark.asyncio
    async def test_generate_content(self, async_sample_config, mock_async_genai_client):
        """generate_content returns response"""
        mock_response = MagicMock()
        mock_response.text = "Hello world"
        mock_async_genai_client.aio.models.generate_content.return_value = mock_response

        with patch(
            "greycloud.async_client.create_client", return_value=mock_async_genai_client
        ):
            client = GreyCloudAsyncClient(async_sample_config)
            contents = [
                types.Content(role="user", parts=[types.Part.from_text(text="Hi")])
            ]
            result = await client.generate_content(contents)
            assert result.text == "Hello world"
            mock_async_genai_client.aio.models.generate_content.assert_called_once()

    @pytest.mark.asyncio
    async def test_generate_content_with_rate_limiting(
        self, async_sample_config, mock_async_genai_client
    ):
        """generate_content goes through rate limiter"""
        mock_response = MagicMock()
        mock_async_genai_client.aio.models.generate_content.return_value = mock_response

        with patch(
            "greycloud.async_client.create_client", return_value=mock_async_genai_client
        ):
            client = GreyCloudAsyncClient(async_sample_config)
            with patch.object(
                client.rate_limiter, "call_with_limits", new_callable=AsyncMock
            ) as mock_limiter:
                mock_limiter.return_value = mock_response
                contents = [
                    types.Content(role="user", parts=[types.Part.from_text(text="Hi")])
                ]
                result = await client.generate_content(contents)
                mock_limiter.assert_called_once()
                assert result == mock_response


class TestGreyCloudAsyncClientGenerateContentStream:
    """Tests for async generate_content_stream"""

    @pytest.mark.asyncio
    async def test_generate_content_stream_yields_chunks(
        self, async_sample_config, mock_async_genai_client
    ):
        """generate_content_stream yields text chunks from stream"""
        mock_chunk1 = MagicMock()
        mock_chunk1.candidates = [MagicMock()]
        mock_chunk1.candidates[0].content = MagicMock()
        mock_chunk1.candidates[0].content.parts = [MagicMock()]
        mock_chunk1.text = "Hello "
        mock_chunk2 = MagicMock()
        mock_chunk2.candidates = [MagicMock()]
        mock_chunk2.candidates[0].content = MagicMock()
        mock_chunk2.candidates[0].content.parts = [MagicMock()]
        mock_chunk2.text = "World"
        # API is awaited and returns stream; mock returns coroutine that resolves to generator
        mock_async_genai_client.aio.models.generate_content_stream = MagicMock(
            side_effect=lambda *a, **kw: _mock_stream_awaitable(
                mock_chunk1, mock_chunk2
            )
        )

        with patch(
            "greycloud.async_client.create_client", return_value=mock_async_genai_client
        ):
            client = GreyCloudAsyncClient(async_sample_config)
            contents = [
                types.Content(role="user", parts=[types.Part.from_text(text="Hi")])
            ]
            chunks = []
            async for chunk in client.generate_content_stream(contents):
                chunks.append(chunk)
            assert chunks == ["Hello ", "World"]

    @pytest.mark.asyncio
    async def test_generate_content_stream_uses_rate_limiter(
        self, async_sample_config, mock_async_genai_client
    ):
        """generate_content_stream goes through rate limiter"""
        mock_chunk = MagicMock()
        mock_chunk.candidates = [MagicMock()]
        mock_chunk.candidates[0].content = MagicMock()
        mock_chunk.candidates[0].content.parts = [MagicMock()]
        mock_chunk.text = "x"
        mock_async_genai_client.aio.models.generate_content_stream = MagicMock(
            side_effect=lambda *a, **kw: _mock_stream_awaitable(mock_chunk)
        )

        async def fake_call_with_limits(token_est, coro):
            return await coro

        with patch(
            "greycloud.async_client.create_client", return_value=mock_async_genai_client
        ):
            client = GreyCloudAsyncClient(async_sample_config)
            with patch.object(
                client.rate_limiter,
                "call_with_limits",
                new_callable=AsyncMock,
                side_effect=fake_call_with_limits,
            ) as mock_limiter:
                contents = [
                    types.Content(role="user", parts=[types.Part.from_text(text="Hi")])
                ]
                chunks = []
                async for chunk in client.generate_content_stream(contents):
                    chunks.append(chunk)
                assert chunks == ["x"]
                mock_limiter.assert_called_once()
                call_args = mock_limiter.call_args
                assert call_args[0][0] >= 1  # token_est


class TestGreyCloudAsyncClientCountTokens:
    """Tests for async count_tokens"""

    @pytest.mark.asyncio
    async def test_count_tokens(self, async_sample_config, mock_async_genai_client):
        """count_tokens returns token count"""
        mock_token_response = MagicMock()
        mock_token_response.total_tokens = 42
        mock_async_genai_client.aio.models.count_tokens.return_value = (
            mock_token_response
        )

        with patch(
            "greycloud.async_client.create_client", return_value=mock_async_genai_client
        ):
            client = GreyCloudAsyncClient(async_sample_config)
            contents = [
                types.Content(role="user", parts=[types.Part.from_text(text="Hello")])
            ]
            count = await client.count_tokens(contents)
            assert count == 42

    @pytest.mark.asyncio
    async def test_count_tokens_fallback(
        self, async_sample_config, mock_async_genai_client
    ):
        """count_tokens falls back to character estimate on failure"""
        mock_async_genai_client.aio.models.count_tokens.side_effect = Exception(
            "API down"
        )

        with patch(
            "greycloud.async_client.create_client", return_value=mock_async_genai_client
        ):
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

        with patch(
            "greycloud.async_client.create_client", return_value=mock_async_genai_client
        ):
            client = GreyCloudAsyncClient(async_sample_config)
            contents = [
                types.Content(role="user", parts=[types.Part.from_text(text="Hi")])
            ]
            result = await client.generate_with_retry(contents, max_retries=3)
            assert result.text == "Success"
            assert mock_async_genai_client.aio.models.generate_content.call_count == 2

    @pytest.mark.asyncio
    async def test_retry_exhausted_raises(
        self, async_sample_config, mock_async_genai_client
    ):
        """Raises after max retries exhausted"""
        mock_async_genai_client.aio.models.generate_content.side_effect = RuntimeError(
            "500 Server Error"
        )

        with patch(
            "greycloud.async_client.create_client", return_value=mock_async_genai_client
        ):
            client = GreyCloudAsyncClient(async_sample_config)
            contents = [
                types.Content(role="user", parts=[types.Part.from_text(text="Hi")])
            ]
            with pytest.raises(RuntimeError, match="500 Server Error"):
                await client.generate_with_retry(contents, max_retries=2)
            assert mock_async_genai_client.aio.models.generate_content.call_count == 3

    @pytest.mark.asyncio
    async def test_generate_with_retry_streaming_yields_chunks(
        self, async_sample_config, mock_async_genai_client
    ):
        """generate_with_retry(streaming=True) returns async generator of text chunks"""
        mock_chunk1 = MagicMock()
        mock_chunk1.candidates = [MagicMock()]
        mock_chunk1.candidates[0].content = MagicMock()
        mock_chunk1.candidates[0].content.parts = [MagicMock()]
        mock_chunk1.text = "Hello "
        mock_chunk2 = MagicMock()
        mock_chunk2.candidates = [MagicMock()]
        mock_chunk2.candidates[0].content = MagicMock()
        mock_chunk2.candidates[0].content.parts = [MagicMock()]
        mock_chunk2.text = "World"
        mock_async_genai_client.aio.models.generate_content_stream = MagicMock(
            side_effect=lambda *a, **kw: _mock_stream_awaitable(
                mock_chunk1, mock_chunk2
            )
        )

        with patch(
            "greycloud.async_client.create_client", return_value=mock_async_genai_client
        ):
            client = GreyCloudAsyncClient(async_sample_config)
            contents = [
                types.Content(role="user", parts=[types.Part.from_text(text="Hi")])
            ]
            gen = await client.generate_with_retry(contents, streaming=True)
            chunks = []
            async for chunk in gen:
                chunks.append(chunk)
            assert chunks == ["Hello ", "World"]

    @pytest.mark.asyncio
    async def test_generate_with_retry_streaming_retries_on_exception(
        self, async_sample_config, mock_async_genai_client
    ):
        """generate_with_retry(streaming=True) retries on stream exception"""
        mock_chunk = MagicMock()
        mock_chunk.candidates = [MagicMock()]
        mock_chunk.candidates[0].content = MagicMock()
        mock_chunk.candidates[0].content.parts = [MagicMock()]
        mock_chunk.text = "OK"

        async def stream_raise_then_ok():
            yield mock_chunk  # first chunk
            raise RuntimeError("stream broken")

        async def stream_ok():
            yield mock_chunk

        call_count = 0

        async def make_stream_awaitable(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return stream_raise_then_ok()
            return stream_ok()

        mock_async_genai_client.aio.models.generate_content_stream = MagicMock(
            side_effect=make_stream_awaitable
        )

        with patch(
            "greycloud.async_client.create_client", return_value=mock_async_genai_client
        ):
            client = GreyCloudAsyncClient(async_sample_config)
            contents = [
                types.Content(role="user", parts=[types.Part.from_text(text="Hi")])
            ]
            gen = await client.generate_with_retry(
                contents, streaming=True, max_retries=2
            )
            chunks = []
            async for chunk in gen:
                chunks.append(chunk)
            assert chunks == ["OK", "OK"]
            assert call_count == 2
