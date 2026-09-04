"""Tests for GreyCloudAsyncClient"""

import os
import sys

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from google.genai import types

from greycloud.async_client import GreyCloudAsyncClient
from greycloud.config import GreyCloudConfig
from greycloud.grounding import GroundingSource


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
        """When use_vertex_ai_search and datastore set, config includes tools

        Under the default ``grounding_mode="inject"`` the retrieval tool is
        dropped (grounding is injected into the prompt instead); it is only
        included in the legacy ``grounding_mode="tool"`` mode.
        """
        datastore = "projects/test/locations/us/datastores/test-ds"

        with patch("greycloud.async_client.create_client"):
            # Default inject mode: retrieval tool dropped even with datastore.
            client = GreyCloudAsyncClient(
                GreyCloudConfig(
                    project_id="test-project-id",
                    location="us-east4",
                    use_vertex_ai_search=True,
                    vertex_ai_search_datastore=datastore,
                )
            )
            assert client._build_tools() == []

            # Legacy "tool" mode: retrieval tool present.
            client = GreyCloudAsyncClient(
                GreyCloudConfig(
                    project_id="test-project-id",
                    location="us-east4",
                    use_vertex_ai_search=True,
                    vertex_ai_search_datastore=datastore,
                    grounding_mode="tool",
                )
            )
            tools = client._build_tools()
            assert len(tools) == 1
            assert tools[0].retrieval is not None
            assert tools[0].retrieval.vertex_ai_search.datastore == datastore

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
    async def test_generate_content_stream_return_chunks(
        self, async_sample_config, mock_async_genai_client
    ):
        """generate_content_stream yields raw chunks from stream if return_chunks=True"""
        mock_chunk = MagicMock()
        mock_chunk.candidates = [MagicMock()]
        mock_chunk.candidates[0].content = MagicMock()
        mock_chunk.candidates[0].content.parts = [MagicMock()]
        mock_chunk.candidates[0].content.parts[0].text = "Hello"
        mock_chunk.text = "Hello"
        mock_async_genai_client.aio.models.generate_content_stream = MagicMock(
            side_effect=lambda *a, **kw: _mock_stream_awaitable(mock_chunk)
        )

        with patch(
            "greycloud.async_client.create_client", return_value=mock_async_genai_client
        ):
            client = GreyCloudAsyncClient(async_sample_config)
            contents = [
                types.Content(role="user", parts=[types.Part.from_text(text="Hi")])
            ]
            chunks = []
            async for chunk in client.generate_content_stream(
                contents, return_chunks=True
            ):
                chunks.append(chunk)
            assert len(chunks) == 1
            assert chunks[0] == mock_chunk
            assert chunks[0].candidates is not None
            assert chunks[0].candidates[0].content.parts[0].text == "Hello"

    @pytest.mark.asyncio
    async def test_generate_with_retry_streaming_return_chunks(
        self, async_sample_config, mock_async_genai_client
    ):
        """generate_with_retry yields raw chunks if return_chunks=True"""
        mock_chunk = MagicMock()
        mock_chunk.candidates = [MagicMock()]
        mock_chunk.candidates[0].content = MagicMock()
        mock_chunk.candidates[0].content.parts = [MagicMock()]
        mock_chunk.candidates[0].content.parts[0].text = "Hello"
        mock_chunk.text = "Hello"
        mock_async_genai_client.aio.models.generate_content_stream = MagicMock(
            side_effect=lambda *a, **kw: _mock_stream_awaitable(mock_chunk)
        )

        with patch(
            "greycloud.async_client.create_client", return_value=mock_async_genai_client
        ):
            client = GreyCloudAsyncClient(async_sample_config)
            contents = [
                types.Content(role="user", parts=[types.Part.from_text(text="Hi")])
            ]
            stream = await client.generate_with_retry(
                contents, streaming=True, return_chunks=True
            )
            chunks = []
            async for chunk in stream:
                chunks.append(chunk)
            assert len(chunks) == 1
            assert chunks[0] == mock_chunk
            assert chunks[0].candidates is not None
            assert chunks[0].candidates[0].content.parts[0].text == "Hello"

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


class TestGreyCloudAsyncClientAuthError:
    """Tests for authentication error detection and re-auth"""

    def test_is_authentication_error(self, async_sample_config):
        """Test authentication error detection"""
        with patch("greycloud.async_client.create_client"):
            client = GreyCloudAsyncClient(async_sample_config)

            auth_errors = [
                Exception("401 Unauthorized"),
                Exception("403 Forbidden"),
                Exception("authentication failed"),
                Exception("token expired"),
                Exception("permission denied"),
            ]

            for error in auth_errors:
                assert client._is_authentication_error(error) is True

            non_auth_errors = [
                Exception("Network error"),
                Exception("Timeout"),
                ValueError("Invalid input"),
            ]

            for error in non_auth_errors:
                assert client._is_authentication_error(error) is False

    def test_is_authentication_error_detects_expired_keyword(self, async_sample_config):
        """Test that 'expired' by itself triggers auth error detection.

        Google Auth errors like 'Reauthentication is needed' often contain
        'expired' without the 'token' prefix. This test verifies we catch those.
        """
        with patch("greycloud.async_client.create_client"):
            client = GreyCloudAsyncClient(async_sample_config)

            expired_errors = [
                Exception("credentials expired"),
                Exception("session expired"),
                Exception("token expired"),
                Exception("Your credentials have expired"),
                Exception("Reauthentication is needed - credentials expired"),
            ]

            for error in expired_errors:
                assert (
                    client._is_authentication_error(error) is True
                ), f"Should detect 'expired' in: {error}"

    def test_force_reauth_with_api_key(self, async_sample_config):
        """Force re-auth returns False when using API key"""
        config = GreyCloudConfig(
            project_id="test-project-id",
            location="us-east4",
            use_api_key=True,
        )
        with patch("greycloud.async_client.create_client"):
            client = GreyCloudAsyncClient(config)
            assert client._force_reauth() is False

    def test_force_reauth_disabled(self, async_sample_config):
        """Force re-auth returns False when auto_reauth is False"""
        config = GreyCloudConfig(
            project_id="test-project-id",
            location="us-east4",
            use_api_key=False,
            auto_reauth=False,
        )
        with patch("greycloud.async_client.create_client"):
            client = GreyCloudAsyncClient(config)
            assert client._force_reauth() is False

    def test_force_reauth_allows_user_interaction(self, async_sample_config):
        """Force re-auth allows user to see gcloud prompts.

        When re-authentication is needed, gcloud prints a URL for the user to
        visit in their browser. We must NOT suppress this output with --quiet
        or capture_output=True, otherwise the user cannot complete the flow.
        """
        import subprocess

        config = GreyCloudConfig(
            project_id="test-project-id",
            location="us-east4",
            use_api_key=False,
            auto_reauth=True,
        )

        with patch("greycloud.async_client.create_client"):
            client = GreyCloudAsyncClient(config)

            with patch("greycloud.async_client.subprocess.run") as mock_run:
                mock_run.return_value = MagicMock(returncode=0)

                # Simulate interactive environment (has DISPLAY or TTY)
                with patch.dict(os.environ, {"DISPLAY": ":0"}):
                    result = client._force_reauth()

                    assert result is True
                    mock_run.assert_called_once()
                    call_args = mock_run.call_args

                    # Check the command does NOT include --quiet
                    cmd = call_args[0][0]
                    assert (
                        "--quiet" not in cmd
                    ), "gcloud command should NOT use --quiet (suppresses user interaction)"

                    # Check capture_output is False to allow user to see prompts
                    assert (
                        call_args[1].get("capture_output") is False
                    ), "capture_output should be False to show gcloud URL/prompts to user"

    def test_force_reauth_non_interactive_uses_no_browser(self, async_sample_config):
        """Force re-auth uses --no-browser (not --quiet) in non-interactive mode.

        In non-interactive environments, we should use --no-browser to get a URL
        the user can visit manually, but still NOT use --quiet so they can see it.
        """
        import subprocess

        config = GreyCloudConfig(
            project_id="test-project-id",
            location="us-east4",
            use_api_key=False,
            auto_reauth=True,
        )

        with patch("greycloud.async_client.create_client"):
            client = GreyCloudAsyncClient(config)

            with patch("greycloud.async_client.subprocess.run") as mock_run:
                mock_run.return_value = MagicMock(returncode=0)

                # Simulate non-interactive environment (no DISPLAY, no TTY)
                with patch.dict(os.environ, {}, clear=True):
                    with patch("sys.stdin") as mock_stdin:
                        mock_stdin.isatty.return_value = False
                        result = client._force_reauth()

                        assert result is True
                        mock_run.assert_called_once()
                        call_args = mock_run.call_args
                        cmd = call_args[0][0]

                        # Should use --no-browser in non-interactive mode
                        assert (
                            "--no-browser" in cmd
                        ), "Should use --no-browser in non-interactive mode"
                        # Should NOT use --quiet
                        assert (
                            "--quiet" not in cmd
                        ), "Should NOT use --quiet even in non-interactive mode"
                        # capture_output should be False
                        assert (
                            call_args[1].get("capture_output") is False
                        ), "capture_output should be False"


DATASTORE = "projects/test/locations/us/datastores/test-ds"


def _async_grounding_config(**kwargs):
    """GreyCloudConfig with vertex-ai-search enabled (inject mode by default)."""
    defaults = dict(
        project_id="test-project-id",
        location="us-east4",
        use_vertex_ai_search=True,
        vertex_ai_search_datastore=DATASTORE,
    )
    defaults.update(kwargs)
    return GreyCloudConfig(**defaults)


def _async_two_fake_sources():
    return [
        GroundingSource(
            title="Doc1", link="gs://bucket/doc1.pdf", snippet="Passage one.", index=1
        ),
        GroundingSource(
            title="Doc2", link="gs://bucket/doc2.pdf", snippet="Passage two.", index=2
        ),
    ]


class TestGreyCloudAsyncClientGroundingInjection:
    """Integration tests for Discovery Engine grounding injection (inject mode)."""

    @pytest.mark.asyncio
    async def test_generate_content_inject_mode_injects_grounding_block(
        self, async_sample_config, mock_async_genai_client
    ):
        """Inject mode: search runs on the last user query, sources are prepended
        to a copy of the last user message, and no retrieval tool is sent."""
        mock_response = MagicMock()
        mock_response.text = "Hello world"
        mock_async_genai_client.aio.models.generate_content.return_value = mock_response

        with patch(
            "greycloud.async_client.create_client", return_value=mock_async_genai_client
        ):
            with patch(
                "greycloud.async_client.asearch_sources",
                new_callable=AsyncMock,
                return_value=_async_two_fake_sources(),
            ) as mock_search:
                client = GreyCloudAsyncClient(_async_grounding_config())
                contents = [
                    types.Content(
                        role="user", parts=[types.Part.from_text(text="Hello")]
                    )
                ]
                await client.generate_content(contents)

        # Search is performed with (config, last user query).
        mock_search.assert_called_once()
        assert mock_search.call_args[0][0] is client.config
        assert mock_search.call_args[0][1] == "Hello"
        # retry_unquoted defaults to True and is threaded through.
        assert mock_search.call_args[1]["retry_unquoted"] is True

        call_args = mock_async_genai_client.aio.models.generate_content.call_args
        sent_contents = call_args[1]["contents"]
        sent_config = call_args[1]["config"]

        # Retrieval tool absent in inject mode.
        assert sent_config.tools == []

        # Grounding block prepended to the (copied) last user message.
        last_user = sent_contents[-1]
        assert last_user.role == "user"
        assert last_user.parts[0].text.startswith("<grounding_sources>")
        assert "[1] (Doc1" in last_user.parts[0].text
        assert "[2] (Doc2" in last_user.parts[0].text
        # Original user text still present after the injected block.
        assert last_user.parts[-1].text == "Hello"

    @pytest.mark.asyncio
    async def test_generate_content_passes_retry_unquoted_flag(
        self, async_sample_config, mock_async_genai_client
    ):
        """retry_unquoted is threaded from generate_content to asearch_sources."""
        mock_response = MagicMock()
        mock_response.text = "Hello world"
        mock_async_genai_client.aio.models.generate_content.return_value = mock_response

        with patch(
            "greycloud.async_client.create_client", return_value=mock_async_genai_client
        ):
            with patch(
                "greycloud.async_client.asearch_sources",
                new_callable=AsyncMock,
                return_value=_async_two_fake_sources(),
            ) as mock_search:
                client = GreyCloudAsyncClient(_async_grounding_config())
                contents = [
                    types.Content(
                        role="user", parts=[types.Part.from_text(text="Hello")]
                    )
                ]
                await client.generate_content(contents, retry_unquoted=False)

        mock_search.assert_called_once()
        assert mock_search.call_args[1]["retry_unquoted"] is False

    @pytest.mark.asyncio
    async def test_generate_content_stream_inject_mode_injects_grounding_block(
        self, async_sample_config, mock_async_genai_client
    ):
        """Streaming path applies the same injection as the non-streaming path."""
        mock_chunk = MagicMock()
        mock_chunk.candidates = [MagicMock()]
        mock_chunk.candidates[0].content = MagicMock()
        mock_chunk.candidates[0].content.parts = [MagicMock()]
        mock_chunk.text = "Hello"
        mock_async_genai_client.aio.models.generate_content_stream = MagicMock(
            side_effect=lambda *a, **kw: _mock_stream_awaitable(mock_chunk)
        )

        with patch(
            "greycloud.async_client.create_client", return_value=mock_async_genai_client
        ):
            with patch(
                "greycloud.async_client.asearch_sources",
                new_callable=AsyncMock,
                return_value=_async_two_fake_sources(),
            ) as mock_search:
                client = GreyCloudAsyncClient(_async_grounding_config())
                contents = [
                    types.Content(
                        role="user", parts=[types.Part.from_text(text="Hello")]
                    )
                ]
                chunks = []
                async for chunk in client.generate_content_stream(contents):
                    chunks.append(chunk)

        assert chunks == ["Hello"]
        mock_search.assert_called_once()
        assert mock_search.call_args[0][1] == "Hello"

        call_args = mock_async_genai_client.aio.models.generate_content_stream.call_args
        sent_contents = call_args[1]["contents"]
        sent_config = call_args[1]["config"]
        assert sent_config.tools == []
        last_user = sent_contents[-1]
        assert last_user.parts[0].text.startswith("<grounding_sources>")
        assert last_user.parts[-1].text == "Hello"

    @pytest.mark.asyncio
    async def test_generate_content_inject_mode_empty_sources_degrades_ungrounded(
        self, async_sample_config, mock_async_genai_client
    ):
        """Empty search results degrade to ungrounded generation: still called,
        contents unmodified, no grounding part, no exception."""
        mock_response = MagicMock()
        mock_async_genai_client.aio.models.generate_content.return_value = mock_response

        with patch(
            "greycloud.async_client.create_client", return_value=mock_async_genai_client
        ):
            with patch(
                "greycloud.async_client.asearch_sources",
                new_callable=AsyncMock,
                return_value=[],
            ) as mock_search:
                client = GreyCloudAsyncClient(_async_grounding_config())
                contents = [
                    types.Content(
                        role="user", parts=[types.Part.from_text(text="Hello")]
                    )
                ]
                result = await client.generate_content(contents)

        assert result == mock_response
        mock_search.assert_called_once()

        call_args = mock_async_genai_client.aio.models.generate_content.call_args
        sent_contents = call_args[1]["contents"]
        # Same list object passed through untouched.
        assert sent_contents is contents
        assert len(contents[0].parts) == 1
        assert contents[0].parts[0].text == "Hello"
        assert "<grounding_sources>" not in contents[0].parts[0].text

    @pytest.mark.asyncio
    async def test_generate_content_tool_mode_no_search_uses_retrieval_tool(
        self, async_sample_config, mock_async_genai_client
    ):
        """Tool mode is unchanged: retrieval tool sent, no search performed,
        contents unmodified."""
        mock_response = MagicMock()
        mock_async_genai_client.aio.models.generate_content.return_value = mock_response

        with patch(
            "greycloud.async_client.create_client", return_value=mock_async_genai_client
        ):
            with patch("greycloud.async_client.asearch_sources") as mock_search:
                client = GreyCloudAsyncClient(
                    _async_grounding_config(grounding_mode="tool")
                )
                contents = [
                    types.Content(
                        role="user", parts=[types.Part.from_text(text="Hello")]
                    )
                ]
                await client.generate_content(contents)

        mock_search.assert_not_called()

        call_args = mock_async_genai_client.aio.models.generate_content.call_args
        sent_config = call_args[1]["config"]
        assert len(sent_config.tools) == 1
        assert sent_config.tools[0].retrieval is not None
        assert (
            sent_config.tools[0].retrieval.vertex_ai_search.datastore
            == client.config.vertex_ai_search_datastore
        )
        # Contents passed through unmodified.
        assert call_args[1]["contents"] is contents

    @pytest.mark.asyncio
    async def test_generate_content_inject_mode_does_not_mutate_caller_contents(
        self, async_sample_config, mock_async_genai_client
    ):
        """Caller's contents list and Content objects are never mutated."""
        mock_response = MagicMock()
        mock_async_genai_client.aio.models.generate_content.return_value = mock_response

        with patch(
            "greycloud.async_client.create_client", return_value=mock_async_genai_client
        ):
            with patch(
                "greycloud.async_client.asearch_sources",
                new_callable=AsyncMock,
                return_value=_async_two_fake_sources(),
            ):
                client = GreyCloudAsyncClient(_async_grounding_config())
                original_user = types.Content(
                    role="user", parts=[types.Part.from_text(text="Hello")]
                )
                contents = [original_user]
                await client.generate_content(contents)

        # Original list, Content object, and parts are untouched.
        assert len(contents) == 1
        assert contents[0] is original_user
        assert len(original_user.parts) == 1
        assert original_user.parts[0].text == "Hello"
        assert "<grounding_sources>" not in original_user.parts[0].text

    @pytest.mark.asyncio
    async def test_generate_content_explicit_tools_override_skips_injection(
        self, async_sample_config, mock_async_genai_client
    ):
        """An explicit tools= override is honored as-is and skips injection."""
        explicit_tool = types.Tool(
            retrieval=types.Retrieval(
                vertex_ai_search=types.VertexAISearch(datastore=DATASTORE)
            )
        )
        mock_response = MagicMock()
        mock_async_genai_client.aio.models.generate_content.return_value = mock_response

        with patch(
            "greycloud.async_client.create_client", return_value=mock_async_genai_client
        ):
            with patch("greycloud.async_client.asearch_sources") as mock_search:
                client = GreyCloudAsyncClient(_async_grounding_config())
                contents = [
                    types.Content(
                        role="user", parts=[types.Part.from_text(text="Hello")]
                    )
                ]
                await client.generate_content(contents, tools=[explicit_tool])

        mock_search.assert_not_called()

        call_args = mock_async_genai_client.aio.models.generate_content.call_args
        sent_config = call_args[1]["config"]
        assert sent_config.tools == [explicit_tool]
        # Contents passed through unmodified.
        assert call_args[1]["contents"] is contents


class TestGroundingQueryAndSkip:
    """Per-call grounding_query override, grounding skip flag, and the
    config-level min_grounding_query_chars threshold (RAG proposal items 1-2).
    All mock asearch_sources; no live calls."""

    @staticmethod
    def _contents(text="Hello"):
        return [types.Content(role="user", parts=[types.Part.from_text(text=text)])]

    @pytest.mark.asyncio
    async def test_generate_content_grounding_query_used_as_search_query(
        self, async_sample_config, mock_async_genai_client
    ):
        """grounding_query replaces the verbatim last user message as the
        Discovery Engine query; injection still targets the last user message."""
        mock_response = MagicMock()
        mock_response.text = "Hello world"
        mock_async_genai_client.aio.models.generate_content.return_value = mock_response

        with patch(
            "greycloud.async_client.create_client", return_value=mock_async_genai_client
        ):
            with patch(
                "greycloud.async_client.asearch_sources",
                new_callable=AsyncMock,
                return_value=_async_two_fake_sources(),
            ) as mock_search:
                client = GreyCloudAsyncClient(_async_grounding_config())
                await client.generate_content(
                    self._contents("Great, now let's do the summary."),
                    grounding_query="ABAS functional impairment adult collateral reports",
                )

        mock_search.assert_called_once()
        assert (
            mock_search.call_args[0][1]
            == "ABAS functional impairment adult collateral reports"
        )

    @pytest.mark.asyncio
    async def test_generate_content_blank_grounding_query_falls_back_to_user_message(
        self, async_sample_config, mock_async_genai_client
    ):
        """A whitespace-only grounding_query is treated as absent."""
        mock_response = MagicMock()
        mock_response.text = "Hello world"
        mock_async_genai_client.aio.models.generate_content.return_value = mock_response

        with patch(
            "greycloud.async_client.create_client", return_value=mock_async_genai_client
        ):
            with patch(
                "greycloud.async_client.asearch_sources",
                new_callable=AsyncMock,
                return_value=_async_two_fake_sources(),
            ) as mock_search:
                client = GreyCloudAsyncClient(_async_grounding_config())
                await client.generate_content(self._contents(), grounding_query="   ")

        assert mock_search.call_args[0][1] == "Hello"

    @pytest.mark.asyncio
    async def test_generate_content_grounding_false_skips_search(
        self, async_sample_config, mock_async_genai_client
    ):
        """grounding=False suppresses the search and injection for one call."""
        mock_response = MagicMock()
        mock_response.text = "Hello world"
        mock_async_genai_client.aio.models.generate_content.return_value = mock_response

        with patch(
            "greycloud.async_client.create_client", return_value=mock_async_genai_client
        ):
            with patch(
                "greycloud.async_client.asearch_sources",
                new_callable=AsyncMock,
                return_value=_async_two_fake_sources(),
            ) as mock_search:
                client = GreyCloudAsyncClient(_async_grounding_config())
                contents = self._contents()
                await client.generate_content(contents, grounding=False)

        mock_search.assert_not_called()
        call_args = mock_async_genai_client.aio.models.generate_content.call_args
        assert call_args[1]["contents"] is contents

    @pytest.mark.asyncio
    async def test_generate_content_stream_grounding_query_used_as_search_query(
        self, async_sample_config, mock_async_genai_client
    ):
        """The streaming path honors grounding_query the same way."""
        mock_chunk = MagicMock()
        mock_chunk.candidates = [MagicMock()]
        mock_chunk.candidates[0].content = MagicMock()
        mock_chunk.candidates[0].content.parts = [MagicMock()]
        mock_chunk.text = "Hello"
        mock_async_genai_client.aio.models.generate_content_stream = MagicMock(
            side_effect=lambda *a, **kw: _mock_stream_awaitable(mock_chunk)
        )

        with patch(
            "greycloud.async_client.create_client", return_value=mock_async_genai_client
        ):
            with patch(
                "greycloud.async_client.asearch_sources",
                new_callable=AsyncMock,
                return_value=_async_two_fake_sources(),
            ) as mock_search:
                client = GreyCloudAsyncClient(_async_grounding_config())
                async for _ in client.generate_content_stream(
                    self._contents("Ok, standby."),
                    grounding_query="distressed child custody intake",
                ):
                    pass

        assert mock_search.call_args[0][1] == "distressed child custody intake"

    @pytest.mark.asyncio
    async def test_generate_content_stream_grounding_false_skips_search(
        self, async_sample_config, mock_async_genai_client
    ):
        mock_chunk = MagicMock()
        mock_chunk.candidates = [MagicMock()]
        mock_chunk.candidates[0].content = MagicMock()
        mock_chunk.candidates[0].content.parts = [MagicMock()]
        mock_chunk.text = "Hello"
        mock_async_genai_client.aio.models.generate_content_stream = MagicMock(
            side_effect=lambda *a, **kw: _mock_stream_awaitable(mock_chunk)
        )

        with patch(
            "greycloud.async_client.create_client", return_value=mock_async_genai_client
        ):
            with patch(
                "greycloud.async_client.asearch_sources",
                new_callable=AsyncMock,
                return_value=_async_two_fake_sources(),
            ) as mock_search:
                client = GreyCloudAsyncClient(_async_grounding_config())
                contents = self._contents()
                async for _ in client.generate_content_stream(
                    contents, grounding=False
                ):
                    pass

        mock_search.assert_not_called()

    @pytest.mark.asyncio
    async def test_min_grounding_query_chars_skips_short_query(
        self, async_sample_config, mock_async_genai_client
    ):
        """A last-user message shorter than the threshold skips the search."""
        mock_response = MagicMock()
        mock_response.text = "Hello world"
        mock_async_genai_client.aio.models.generate_content.return_value = mock_response

        with patch(
            "greycloud.async_client.create_client", return_value=mock_async_genai_client
        ):
            with patch(
                "greycloud.async_client.asearch_sources",
                new_callable=AsyncMock,
                return_value=_async_two_fake_sources(),
            ) as mock_search:
                client = GreyCloudAsyncClient(
                    _async_grounding_config(min_grounding_query_chars=10)
                )
                contents = self._contents("thanks")
                await client.generate_content(contents)

        mock_search.assert_not_called()
        call_args = mock_async_genai_client.aio.models.generate_content.call_args
        assert call_args[1]["contents"] is contents

    @pytest.mark.asyncio
    async def test_min_grounding_query_chars_long_query_still_searches(
        self, async_sample_config, mock_async_genai_client
    ):
        mock_response = MagicMock()
        mock_response.text = "Hello world"
        mock_async_genai_client.aio.models.generate_content.return_value = mock_response

        with patch(
            "greycloud.async_client.create_client", return_value=mock_async_genai_client
        ):
            with patch(
                "greycloud.async_client.asearch_sources",
                new_callable=AsyncMock,
                return_value=_async_two_fake_sources(),
            ) as mock_search:
                client = GreyCloudAsyncClient(
                    _async_grounding_config(min_grounding_query_chars=5)
                )
                await client.generate_content(self._contents("ABAS impairment domains"))

        assert mock_search.call_args[0][1] == "ABAS impairment domains"

    @pytest.mark.asyncio
    async def test_min_grounding_query_chars_applies_to_grounding_query(
        self, async_sample_config, mock_async_genai_client
    ):
        """The threshold gates the effective query: a short user message with
        a long grounding_query still searches (the override is the query)."""
        mock_response = MagicMock()
        mock_response.text = "Hello world"
        mock_async_genai_client.aio.models.generate_content.return_value = mock_response

        with patch(
            "greycloud.async_client.create_client", return_value=mock_async_genai_client
        ):
            with patch(
                "greycloud.async_client.asearch_sources",
                new_callable=AsyncMock,
                return_value=_async_two_fake_sources(),
            ) as mock_search:
                client = GreyCloudAsyncClient(
                    _async_grounding_config(min_grounding_query_chars=10)
                )
                await client.generate_content(
                    self._contents("ok"),
                    grounding_query="ABAS functional impairment adult collateral reports",
                )

        assert (
            mock_search.call_args[0][1]
            == "ABAS functional impairment adult collateral reports"
        )

    @pytest.mark.asyncio
    async def test_generate_with_retry_threads_grounding_query(
        self, async_sample_config, mock_async_genai_client
    ):
        """grounding_query flows through generate_with_retry's **kwargs."""
        mock_response = MagicMock()
        mock_response.text = "Hello world"
        mock_async_genai_client.aio.models.generate_content.return_value = mock_response

        with patch(
            "greycloud.async_client.create_client", return_value=mock_async_genai_client
        ):
            with patch(
                "greycloud.async_client.asearch_sources",
                new_callable=AsyncMock,
                return_value=_async_two_fake_sources(),
            ) as mock_search:
                client = GreyCloudAsyncClient(_async_grounding_config())
                await client.generate_with_retry(
                    self._contents(),
                    grounding_query="ABAS functional impairment",
                )

        assert mock_search.call_args[0][1] == "ABAS functional impairment"


class TestOnGroundingCallback:
    """on_grounding contract (RAG proposal §5): invoked once per generate with
    the exact source list being injected, with [] when the search ran but
    returned nothing, and not at all when grounding was skipped entirely.
    Callback exceptions never propagate."""

    @staticmethod
    def _contents(text="Hello"):
        return [types.Content(role="user", parts=[types.Part.from_text(text=text)])]

    @pytest.mark.asyncio
    async def test_called_with_injected_sources(
        self, async_sample_config, mock_async_genai_client
    ):
        """Sources found: on_grounding fires with the exact list injected."""
        mock_response = MagicMock()
        mock_response.text = "Hello world"
        mock_async_genai_client.aio.models.generate_content.return_value = mock_response

        sources = _async_two_fake_sources()
        with patch(
            "greycloud.async_client.create_client",
            return_value=mock_async_genai_client,
        ):
            with patch(
                "greycloud.async_client.asearch_sources",
                new_callable=AsyncMock,
                return_value=sources,
            ):
                client = GreyCloudAsyncClient(_async_grounding_config())
                seen = []
                await client.generate_content(
                    self._contents(), on_grounding=seen.extend
                )

        assert seen == sources

    @pytest.mark.asyncio
    async def test_coroutine_callback_awaited(
        self, async_sample_config, mock_async_genai_client
    ):
        """A coroutine-function callback is awaited."""
        mock_response = MagicMock()
        mock_response.text = "Hello world"
        mock_async_genai_client.aio.models.generate_content.return_value = mock_response

        ran = []

        async def async_cb(sources):
            ran.append([s.title for s in sources])

        with patch(
            "greycloud.async_client.create_client",
            return_value=mock_async_genai_client,
        ):
            with patch(
                "greycloud.async_client.asearch_sources",
                new_callable=AsyncMock,
                return_value=_async_two_fake_sources(),
            ):
                client = GreyCloudAsyncClient(_async_grounding_config())
                await client.generate_content(self._contents(), on_grounding=async_cb)

        assert ran == [["Doc1", "Doc2"]]

    @pytest.mark.asyncio
    async def test_callback_exception_does_not_break_generation(
        self, async_sample_config, mock_async_genai_client
    ):
        """A raising callback is logged and generation proceeds (grounded)."""
        mock_response = MagicMock()
        mock_response.text = "Hello world"
        mock_async_genai_client.aio.models.generate_content.return_value = mock_response

        def boom(sources):
            raise RuntimeError("callback exploded")

        with patch(
            "greycloud.async_client.create_client",
            return_value=mock_async_genai_client,
        ):
            with patch(
                "greycloud.async_client.asearch_sources",
                new_callable=AsyncMock,
                return_value=_async_two_fake_sources(),
            ):
                client = GreyCloudAsyncClient(_async_grounding_config())
                response = await client.generate_content(
                    self._contents(), on_grounding=boom
                )

        assert response is mock_response
        call_args = mock_async_genai_client.aio.models.generate_content.call_args
        assert (
            call_args[1]["contents"][-1].parts[0].text.startswith("<grounding_sources>")
        )

    @pytest.mark.asyncio
    async def test_called_with_empty_list_when_search_returns_nothing(
        self, async_sample_config, mock_async_genai_client
    ):
        """Search ran, nothing found: distinguishable from 'grounding skipped'."""
        mock_response = MagicMock()
        mock_response.text = "Hello world"
        mock_async_genai_client.aio.models.generate_content.return_value = mock_response

        with patch(
            "greycloud.async_client.create_client",
            return_value=mock_async_genai_client,
        ):
            with patch(
                "greycloud.async_client.asearch_sources",
                new_callable=AsyncMock,
                return_value=[],
            ):
                client = GreyCloudAsyncClient(_async_grounding_config())
                seen = []
                await client.generate_content(
                    self._contents(), on_grounding=seen.extend
                )

        assert seen == []

    @pytest.mark.asyncio
    async def test_not_called_when_grounding_false(
        self, async_sample_config, mock_async_genai_client
    ):
        mock_response = MagicMock()
        mock_response.text = "Hello world"
        mock_async_genai_client.aio.models.generate_content.return_value = mock_response

        with patch(
            "greycloud.async_client.create_client",
            return_value=mock_async_genai_client,
        ):
            with patch(
                "greycloud.async_client.asearch_sources",
                new_callable=AsyncMock,
                return_value=_async_two_fake_sources(),
            ) as mock_search:
                client = GreyCloudAsyncClient(_async_grounding_config())
                seen = []
                await client.generate_content(
                    self._contents(), grounding=False, on_grounding=seen.extend
                )

        mock_search.assert_not_called()
        assert seen == []

    @pytest.mark.asyncio
    async def test_not_called_when_threshold_skips(
        self, async_sample_config, mock_async_genai_client
    ):
        mock_response = MagicMock()
        mock_response.text = "Hello world"
        mock_async_genai_client.aio.models.generate_content.return_value = mock_response

        with patch(
            "greycloud.async_client.create_client",
            return_value=mock_async_genai_client,
        ):
            with patch(
                "greycloud.async_client.asearch_sources",
                new_callable=AsyncMock,
                return_value=_async_two_fake_sources(),
            ) as mock_search:
                client = GreyCloudAsyncClient(
                    _async_grounding_config(min_grounding_query_chars=50)
                )
                seen = []
                await client.generate_content(
                    self._contents("thanks"), on_grounding=seen.extend
                )

        mock_search.assert_not_called()
        assert seen == []

    @pytest.mark.asyncio
    async def test_not_called_when_tools_override(
        self, async_sample_config, mock_async_genai_client
    ):
        mock_response = MagicMock()
        mock_response.text = "Hello world"
        mock_async_genai_client.aio.models.generate_content.return_value = mock_response

        with patch(
            "greycloud.async_client.create_client",
            return_value=mock_async_genai_client,
        ):
            with patch(
                "greycloud.async_client.asearch_sources",
                new_callable=AsyncMock,
                return_value=_async_two_fake_sources(),
            ) as mock_search:
                client = GreyCloudAsyncClient(_async_grounding_config())
                seen = []
                await client.generate_content(
                    self._contents(), tools=[MagicMock()], on_grounding=seen.extend
                )

        mock_search.assert_not_called()
        assert seen == []

    @pytest.mark.asyncio
    async def test_not_called_when_inject_mode_disabled(
        self, async_sample_config, mock_async_genai_client
    ):
        mock_response = MagicMock()
        mock_response.text = "Hello world"
        mock_async_genai_client.aio.models.generate_content.return_value = mock_response

        with patch(
            "greycloud.async_client.create_client",
            return_value=mock_async_genai_client,
        ):
            with patch(
                "greycloud.async_client.asearch_sources",
                new_callable=AsyncMock,
                return_value=_async_two_fake_sources(),
            ) as mock_search:
                client = GreyCloudAsyncClient(
                    _async_grounding_config(use_vertex_ai_search=False)
                )
                seen = []
                await client.generate_content(
                    self._contents(), on_grounding=seen.extend
                )

        mock_search.assert_not_called()
        assert seen == []

    @pytest.mark.asyncio
    async def test_stream_called_with_injected_sources(
        self, async_sample_config, mock_async_genai_client
    ):
        mock_chunk = MagicMock()
        mock_chunk.candidates = [MagicMock()]
        mock_chunk.candidates[0].content = MagicMock()
        mock_chunk.candidates[0].content.parts = [MagicMock()]
        mock_chunk.text = "Hello"
        mock_async_genai_client.aio.models.generate_content_stream = MagicMock(
            side_effect=lambda *a, **kw: _mock_stream_awaitable(mock_chunk)
        )

        with patch(
            "greycloud.async_client.create_client",
            return_value=mock_async_genai_client,
        ):
            with patch(
                "greycloud.async_client.asearch_sources",
                new_callable=AsyncMock,
                return_value=_async_two_fake_sources(),
            ):
                client = GreyCloudAsyncClient(_async_grounding_config())
                seen = []
                async for _ in client.generate_content_stream(
                    self._contents(), on_grounding=seen.extend
                ):
                    pass

        assert [s.title for s in seen] == ["Doc1", "Doc2"]

    @pytest.mark.asyncio
    async def test_generate_with_retry_threads_on_grounding(
        self, async_sample_config, mock_async_genai_client
    ):
        mock_response = MagicMock()
        mock_response.text = "Hello world"
        mock_async_genai_client.aio.models.generate_content.return_value = mock_response

        with patch(
            "greycloud.async_client.create_client",
            return_value=mock_async_genai_client,
        ):
            with patch(
                "greycloud.async_client.asearch_sources",
                new_callable=AsyncMock,
                return_value=_async_two_fake_sources(),
            ):
                client = GreyCloudAsyncClient(_async_grounding_config())
                seen = []
                await client.generate_with_retry(
                    self._contents(), on_grounding=seen.extend
                )

        assert len(seen) == 2
