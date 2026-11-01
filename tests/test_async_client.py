"""Tests for GreyCloudAsyncClient"""

import os

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

    def test_build_generate_config_non_gemini_4_includes_thoughts(
        self, async_sample_config
    ):
        """Non-G4 thinking config omits include_thoughts by default (0.4.x
        behavior); explicit config opt-in/opt-out is honored."""
        from greycloud.config import GreyCloudConfig

        with patch("greycloud.async_client.create_client"):
            client = GreyCloudAsyncClient(async_sample_config)
            config = client._build_generate_config(
                model="gemini-3.8-flash", thinking_level="HIGH"
            )
            assert config.thinking_config.include_thoughts is None

            optin_config = GreyCloudConfig(
                project_id="test-proj", include_thoughts=True
            )
            client_optin = GreyCloudAsyncClient(optin_config)
            config_optin = client_optin._build_generate_config(
                model="gemini-3.8-flash", thinking_level="HIGH"
            )
            assert config_optin.thinking_config.include_thoughts is True

            quiet_config = GreyCloudConfig(
                project_id="test-proj", include_thoughts=False
            )
            client_quiet = GreyCloudAsyncClient(quiet_config)
            config_quiet = client_quiet._build_generate_config(
                model="gemini-3.8-flash", thinking_level="HIGH"
            )
            assert config_quiet.thinking_config.include_thoughts is False

    def test_build_generate_config_gemini_4_strips_penalties_and_sampling(
        self, async_sample_config
    ):
        """Gemini 4 models strip sampling and penalty parameters and default thinking."""
        with patch("greycloud.async_client.create_client"):
            client = GreyCloudAsyncClient(async_sample_config)
            cfg = client._build_generate_config(
                model="gemini-4-argon",
                frequency_penalty=0.5,
                presence_penalty=0.5,
                temperature=0.7,
                top_p=0.9,
                top_k=40,
            )
            assert cfg.temperature is None
            assert cfg.top_p is None
            assert cfg.top_k is None
            assert cfg.frequency_penalty is None
            assert cfg.presence_penalty is None
            assert cfg.thinking_config is not None
            assert cfg.thinking_config.thinking_level == "MEDIUM"
            assert cfg.thinking_config.include_thoughts is True

    def test_build_generate_config_gemini_4_strips_camel_case_penalties_and_sampling(
        self, async_sample_config
    ):
        """Gemini 4 models strip camelCase sampling and penalty parameters from kwargs."""
        with patch("greycloud.async_client.create_client"):
            client = GreyCloudAsyncClient(async_sample_config)
            cfg = client._build_generate_config(
                model="gemini-4-argon",
                frequencyPenalty=0.5,
                presencePenalty=0.5,
                topP=0.9,
                topK=40,
            )
            assert cfg.top_p is None
            assert cfg.top_k is None
            assert cfg.frequency_penalty is None
            assert cfg.presence_penalty is None

    def test_build_generate_config_gemini_4_rejects_cached_content(
        self, async_sample_config
    ):
        """Gemini 4 models reject explicit cached_content."""
        with patch("greycloud.async_client.create_client"):
            client = GreyCloudAsyncClient(async_sample_config)
            with pytest.raises(ValueError, match="implicit durable caching"):
                client._build_generate_config(
                    model="gemini-4-argon", cached_content="caches/123"
                )

    def test_build_generate_config_gemini_4_rejects_camelcase_cached_content(
        self, async_sample_config
    ):
        """The REST-style cachedContent alias must be rejected too."""
        with patch("greycloud.async_client.create_client"):
            client = GreyCloudAsyncClient(async_sample_config)
            with pytest.raises(ValueError, match="implicit durable caching"):
                client._build_generate_config(
                    model="gemini-4-argon", cachedContent="caches/123"
                )

            cfg = client._build_generate_config(
                model="gemini-4-argon", cachedContent=None, cached_content=None
            )
            assert cfg.thinking_config.thinking_level == "MEDIUM"

    def test_build_generate_config_gemini_3_preserves_sampling(
        self, async_sample_config
    ):
        """Non-Gemini 4 models preserve temperature and top_p."""
        with patch("greycloud.async_client.create_client"):
            client = GreyCloudAsyncClient(async_sample_config)
            cfg = client._build_generate_config(
                model="gemini-3.8-flash", temperature=0.7, top_p=0.9
            )
            assert cfg.temperature == 0.7
            assert cfg.top_p == 0.9

    def test_build_generate_config_gemini_4_rejects_thinking_budget(
        self, async_sample_config
    ):
        """Gemini 4 models reject thinking_budget."""
        with patch("greycloud.async_client.create_client"):
            client = GreyCloudAsyncClient(async_sample_config)
            with pytest.raises(ValueError, match="thinking_budget"):
                client._build_generate_config(
                    model="gemini-4-argon", thinking_budget=1024
                )

    def test_build_generate_config_gemini_4_accepts_none_thinking_budget(
        self, async_sample_config
    ):
        """thinking_budget=None means unset and must not be rejected."""
        with patch("greycloud.async_client.create_client"):
            client = GreyCloudAsyncClient(async_sample_config)

            cfg = client._build_generate_config(
                model="gemini-4-argon",
                thinking_budget=None,
                thinkingBudget=None,
            )
            assert cfg.thinking_config.thinking_level == "MEDIUM"

            cfg2 = client._build_generate_config(
                model="gemini-4-argon",
                thinking_config={"thinkingBudget": None, "thinkingLevel": "LOW"},
            )
            assert cfg2.thinking_config.thinking_level == "LOW"

    def test_build_generate_config_gemini_4_rejects_minimal_thinking_level(
        self, async_sample_config
    ):
        """Gemini 4 models reject MINIMAL thinking level."""
        with patch("greycloud.async_client.create_client"):
            client = GreyCloudAsyncClient(async_sample_config)
            with pytest.raises(ValueError, match="MINIMAL"):
                client._build_generate_config(
                    model="gemini-4-argon", thinking_level="MINIMAL"
                )

    def test_build_generate_config_gemini_4_custom_thinking_level(
        self, async_sample_config
    ):
        """Gemini 4 accepts valid explicit thinking_level (LOW, MEDIUM, HIGH)."""
        with patch("greycloud.async_client.create_client"):
            client = GreyCloudAsyncClient(async_sample_config)
            cfg = client._build_generate_config(
                model="gemini-4-argon", thinking_level="HIGH"
            )
            assert cfg.thinking_config.thinking_level == "HIGH"
            assert cfg.thinking_config.include_thoughts is True

    def test_build_generate_config_gemini_4_from_client_config_model(self):
        """When model is not passed, client.config.model is used for sanitization."""
        from greycloud.config import GreyCloudConfig

        cfg_inst = GreyCloudConfig(
            project_id="test-proj",
            model="gemini-4-argon",
            temperature=0.8,
            top_p=0.95,
        )
        with patch("greycloud.async_client.create_client"):
            client = GreyCloudAsyncClient(cfg_inst)
            cfg = client._build_generate_config()
            assert cfg.temperature is None
            assert cfg.top_p is None
            assert cfg.thinking_config.thinking_level == "MEDIUM"

    def test_build_generate_config_gemini_4_kwargs_thinking_config_cannot_overwrite(
        self, async_sample_config
    ):
        """Passing thinking_config in kwargs is sanitized and does not overwrite config."""
        with patch("greycloud.async_client.create_client"):
            client = GreyCloudAsyncClient(async_sample_config)
            raw_tc = types.ThinkingConfig(thinking_level="HIGH")
            cfg = client._build_generate_config(
                model="gemini-4-argon", thinking_config=raw_tc
            )
            assert cfg.thinking_config.thinking_level == "HIGH"
            assert cfg.thinking_config.include_thoughts is True

            cfg2 = client._build_generate_config(
                model="gemini-4-argon",
                thinking_level="LOW",
                thinking_config=raw_tc,
            )
            assert cfg2.thinking_config.thinking_level == "LOW"
            assert cfg2.thinking_config.include_thoughts is True

    def test_build_generate_config_gemini_4_honors_camelcase_thinking_level(
        self, async_sample_config
    ):
        """A REST-style thinkingLevel kwarg is honored, not silently dropped."""
        with patch("greycloud.async_client.create_client"):
            client = GreyCloudAsyncClient(async_sample_config)

            cfg = client._build_generate_config(
                model="gemini-4-argon", thinkingLevel="HIGH"
            )
            assert cfg.thinking_config.thinking_level == "HIGH"

            cfg2 = client._build_generate_config(
                model="gemini-4-argon", thinking_level="LOW", thinkingLevel="HIGH"
            )
            assert cfg2.thinking_config.thinking_level == "LOW"

            with pytest.raises(ValueError, match="MINIMAL"):
                client._build_generate_config(
                    model="gemini-4-argon", thinkingLevel="MINIMAL"
                )
            with pytest.raises(ValueError, match="Invalid thinking_level"):
                client._build_generate_config(
                    model="gemini-4-argon", thinkingLevel="BOGUS"
                )

    def test_build_generate_config_gemini_4_kwargs_thinking_config_validates(
        self, async_sample_config
    ):
        """thinking_config in kwargs validates budget and minimal level."""
        with patch("greycloud.async_client.create_client"):
            client = GreyCloudAsyncClient(async_sample_config)

            with pytest.raises(ValueError, match="thinking_budget"):
                client._build_generate_config(
                    model="gemini-4-argon",
                    thinking_config=types.ThinkingConfig(thinking_budget=1024),
                )

            with pytest.raises(ValueError, match="thinking_budget"):
                client._build_generate_config(
                    model="gemini-4-argon",
                    thinking_config={"thinkingBudget": 1024},
                )

            with pytest.raises(ValueError, match="MINIMAL"):
                client._build_generate_config(
                    model="gemini-4-argon",
                    thinking_config=types.ThinkingConfig(thinking_level="MINIMAL"),
                )

            with pytest.raises(ValueError, match="MINIMAL"):
                client._build_generate_config(
                    model="gemini-4-argon",
                    thinking_config={"thinkingLevel": "MINIMAL"},
                )


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
    async def test_generate_content_gemini_4_sanitizes_config(
        self, async_sample_config, mock_async_genai_client
    ):
        """generate_content with Gemini 4 model sanitizes config before passing to underlying client"""
        mock_response = MagicMock()
        mock_response.text = "Gemini 4 response"
        mock_async_genai_client.aio.models.generate_content.return_value = mock_response

        with patch(
            "greycloud.async_client.create_client", return_value=mock_async_genai_client
        ):
            client = GreyCloudAsyncClient(async_sample_config)
            contents = [
                types.Content(role="user", parts=[types.Part.from_text(text="Hi")])
            ]
            result = await client.generate_content(
                contents,
                model="gemini-4-argon",
                temperature=0.7,
                top_p=0.9,
                frequency_penalty=0.4,
            )
            assert result.text == "Gemini 4 response"
            call_args = mock_async_genai_client.aio.models.generate_content.call_args
            assert call_args[1]["model"] == "gemini-4-argon"
            cfg = call_args[1]["config"]
            assert cfg.temperature is None
            assert cfg.top_p is None
            assert cfg.frequency_penalty is None
            assert cfg.thinking_config.thinking_level == "MEDIUM"
            assert cfg.thinking_config.include_thoughts is True

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

    @pytest.mark.asyncio
    async def test_generate_content_request_timeout(
        self, async_sample_config, mock_async_genai_client
    ):
        """generate_content propagates request_timeout to config.http_options"""
        mock_response = MagicMock()
        mock_async_genai_client.aio.models.generate_content.return_value = mock_response

        with patch(
            "greycloud.async_client.create_client", return_value=mock_async_genai_client
        ):
            client = GreyCloudAsyncClient(async_sample_config)
            contents = [
                types.Content(role="user", parts=[types.Part.from_text(text="Hi")])
            ]
            await client.generate_content(contents, request_timeout=55)

            call_args = mock_async_genai_client.aio.models.generate_content.call_args
            cfg = call_args[1]["config"]
            assert cfg.http_options is not None
            assert cfg.http_options.timeout == 55000

    @pytest.mark.asyncio
    async def test_generate_content_request_timeout_merges_http_options(
        self, async_sample_config, mock_async_genai_client
    ):
        """A per-call timeout must not discard the caller's other options."""
        mock_async_genai_client.aio.models.generate_content.return_value = MagicMock()

        with patch(
            "greycloud.async_client.create_client", return_value=mock_async_genai_client
        ):
            client = GreyCloudAsyncClient(async_sample_config)
            contents = [
                types.Content(role="user", parts=[types.Part.from_text(text="Hi")])
            ]
            await client.generate_content(
                contents,
                request_timeout=30,
                http_options=types.HttpOptions(
                    headers={"x-goog-user-project": "billing-project"},
                    base_url="https://example.test",
                ),
            )

            cfg = mock_async_genai_client.aio.models.generate_content.call_args[1][
                "config"
            ]
            assert cfg.http_options.timeout == 30000
            assert cfg.http_options.headers == {
                "x-goog-user-project": "billing-project"
            }
            assert cfg.http_options.base_url == "https://example.test"

    @pytest.mark.asyncio
    async def test_generate_content_reconciles_usage(
        self, async_sample_config, mock_async_genai_client
    ):
        """generate_content calls rate_limiter.reconcile_usage when usage_metadata has total_token_count"""
        mock_response = MagicMock()
        mock_response.text = "Hello world"
        mock_meta = MagicMock()
        mock_meta.total_token_count = 1500
        mock_response.usage_metadata = mock_meta
        mock_async_genai_client.aio.models.generate_content.return_value = mock_response

        with patch(
            "greycloud.async_client.create_client", return_value=mock_async_genai_client
        ):
            client = GreyCloudAsyncClient(async_sample_config)
            client.rate_limiter.reconcile_usage = AsyncMock()
            contents = [
                types.Content(role="user", parts=[types.Part.from_text(text="Hi")])
            ]
            result = await client.generate_content(contents)
            assert result == mock_response
            client.rate_limiter.reconcile_usage.assert_awaited_once()
            args, _ = client.rate_limiter.reconcile_usage.await_args
            assert args[1] == 1500

    @pytest.mark.asyncio
    async def test_generate_content_skips_reconcile_without_usage_metadata(
        self, async_sample_config, mock_async_genai_client
    ):
        """generate_content skips reconcile_usage when usage_metadata is None"""
        mock_response = MagicMock()
        mock_response.text = "Hello world"
        mock_response.usage_metadata = None
        mock_async_genai_client.aio.models.generate_content.return_value = mock_response

        with patch(
            "greycloud.async_client.create_client", return_value=mock_async_genai_client
        ):
            client = GreyCloudAsyncClient(async_sample_config)
            client.rate_limiter.reconcile_usage = AsyncMock()
            contents = [
                types.Content(role="user", parts=[types.Part.from_text(text="Hi")])
            ]
            await client.generate_content(contents)
            client.rate_limiter.reconcile_usage.assert_not_called()


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

    def test_generate_content_stream_positional_args_keep_pre_050_order(
        self, async_sample_config
    ):
        """New options must come after the pre-0.5.0 parameters."""
        import inspect

        with patch("greycloud.async_client.create_client"):
            client = GreyCloudAsyncClient(async_sample_config)
            sig = inspect.signature(client.generate_content_stream)

        cb = MagicMock()
        bound = sig.bind(
            [],
            None,  # model
            None,  # system_instruction
            None,  # temperature
            None,  # top_p
            None,  # max_output_tokens
            None,  # safety_settings
            None,  # tools
            None,  # thinking_level
            None,  # cached_content
            False,  # return_chunks
            False,  # retry_unquoted
            "q",  # grounding_query
            False,  # grounding
            cb,  # on_grounding
        )
        assert bound.arguments.get("retry_unquoted") is False
        assert bound.arguments.get("grounding_query") == "q"
        assert bound.arguments.get("grounding") is False
        assert bound.arguments.get("on_grounding") is cb
        assert "on_thought" not in bound.arguments
        assert "return_events" not in bound.arguments

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

    @pytest.mark.asyncio
    async def test_stream_thinking_chunks_default_yields_only_text(
        self, async_sample_config, mock_async_genai_client
    ):
        """Default async stream skips thinking chunks and yields only finalized text"""
        part1 = MagicMock()
        part1.text = "I am thinking"
        part1.thought = True
        chunk1 = MagicMock()
        chunk1.candidates = [MagicMock()]
        chunk1.candidates[0].content = MagicMock()
        chunk1.candidates[0].content.parts = [part1]

        part2 = MagicMock()
        part2.text = "Here is the answer"
        part2.thought = False
        chunk2 = MagicMock()
        chunk2.candidates = [MagicMock()]
        chunk2.candidates[0].content = MagicMock()
        chunk2.candidates[0].content.parts = [part2]

        mock_async_genai_client.aio.models.generate_content_stream = MagicMock(
            side_effect=lambda *a, **kw: _mock_stream_awaitable(chunk1, chunk2)
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

            assert chunks == ["Here is the answer"]

    @pytest.mark.asyncio
    async def test_stream_return_events_yields_thought_and_text_dicts(
        self, async_sample_config, mock_async_genai_client
    ):
        """return_events=True yields thought and text dict events in async stream"""
        part1 = MagicMock()
        part1.text = "I am thinking"
        part1.thought = True
        chunk1 = MagicMock()
        chunk1.candidates = [MagicMock()]
        chunk1.candidates[0].content = MagicMock()
        chunk1.candidates[0].content.parts = [part1]

        part2 = MagicMock()
        part2.text = "Here is the answer"
        part2.thought = False
        chunk2 = MagicMock()
        chunk2.candidates = [MagicMock()]
        chunk2.candidates[0].content = MagicMock()
        chunk2.candidates[0].content.parts = [part2]

        mock_async_genai_client.aio.models.generate_content_stream = MagicMock(
            side_effect=lambda *a, **kw: _mock_stream_awaitable(chunk1, chunk2)
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
                contents, return_events=True
            ):
                chunks.append(chunk)

            assert chunks == [
                {"thought": "I am thinking"},
                {"text": "Here is the answer"},
            ]

    @pytest.mark.asyncio
    async def test_stream_on_thought_callback(
        self, async_sample_config, mock_async_genai_client
    ):
        """on_thought callback receives thought text (both sync and async callbacks)"""
        part1 = MagicMock()
        part1.text = "I am thinking"
        part1.thought = True
        chunk1 = MagicMock()
        chunk1.candidates = [MagicMock()]
        chunk1.candidates[0].content = MagicMock()
        chunk1.candidates[0].content.parts = [part1]

        part2 = MagicMock()
        part2.text = "Here is the answer"
        part2.thought = False
        chunk2 = MagicMock()
        chunk2.candidates = [MagicMock()]
        chunk2.candidates[0].content = MagicMock()
        chunk2.candidates[0].content.parts = [part2]

        mock_async_genai_client.aio.models.generate_content_stream = MagicMock(
            side_effect=lambda *a, **kw: _mock_stream_awaitable(chunk1, chunk2)
        )

        with patch(
            "greycloud.async_client.create_client", return_value=mock_async_genai_client
        ):
            client = GreyCloudAsyncClient(async_sample_config)
            contents = [
                types.Content(role="user", parts=[types.Part.from_text(text="Hi")])
            ]

            # 1. Sync callback
            captured_sync = []
            chunks = []
            async for chunk in client.generate_content_stream(
                contents, on_thought=lambda t: captured_sync.append(t)
            ):
                chunks.append(chunk)
            assert captured_sync == ["I am thinking"]
            assert chunks == ["Here is the answer"]

            # 2. Async callback
            captured_async = []

            async def async_cb(t):
                captured_async.append(t)

            chunks = []
            async for chunk in client.generate_content_stream(
                contents, on_thought=async_cb
            ):
                chunks.append(chunk)
            assert captured_async == ["I am thinking"]
            assert chunks == ["Here is the answer"]

            # 3. Failing callback does not crash stream
            async def failing_async_cb(t):
                raise RuntimeError("async callback crash")

            chunks = []
            async for chunk in client.generate_content_stream(
                contents, on_thought=failing_async_cb
            ):
                chunks.append(chunk)
            assert chunks == ["Here is the answer"]

    @pytest.mark.asyncio
    async def test_generate_content_stream_request_timeout(
        self, async_sample_config, mock_async_genai_client
    ):
        """generate_content_stream propagates request_timeout to config.http_options"""
        mock_chunk = MagicMock()
        mock_chunk.candidates = None
        mock_chunk.text = ""

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
            async for _ in client.generate_content_stream(contents, request_timeout=35):
                pass

            call_args = (
                mock_async_genai_client.aio.models.generate_content_stream.call_args
            )
            cfg = call_args[1]["config"]
            assert cfg.http_options is not None
            assert cfg.http_options.timeout == 35000

    @pytest.mark.asyncio
    async def test_generate_content_stream_reconciles_usage(
        self, async_sample_config, mock_async_genai_client
    ):
        """generate_content_stream calls rate_limiter.reconcile_usage upon completion with total_token_count"""
        chunk1 = MagicMock()
        chunk1.candidates = [MagicMock()]
        chunk1.candidates[0].content = MagicMock()
        chunk1.candidates[0].content.parts = [MagicMock(text="Part 1", thought=False)]
        chunk1.text = "Part 1"
        chunk1.usage_metadata = None

        chunk2 = MagicMock()
        chunk2.candidates = [MagicMock()]
        chunk2.candidates[0].content = MagicMock()
        chunk2.candidates[0].content.parts = [MagicMock(text="Part 2", thought=False)]
        chunk2.text = "Part 2"
        mock_meta = MagicMock()
        mock_meta.total_token_count = 2500
        chunk2.usage_metadata = mock_meta

        mock_async_genai_client.aio.models.generate_content_stream = MagicMock(
            side_effect=lambda *a, **kw: _mock_stream_awaitable(chunk1, chunk2)
        )

        with patch(
            "greycloud.async_client.create_client", return_value=mock_async_genai_client
        ):
            client = GreyCloudAsyncClient(async_sample_config)
            client.rate_limiter.reconcile_usage = AsyncMock()
            contents = [
                types.Content(role="user", parts=[types.Part.from_text(text="Hi")])
            ]
            chunks = []
            async for chunk in client.generate_content_stream(contents):
                chunks.append(chunk)

            assert chunks == ["Part 1", "Part 2"]
            client.rate_limiter.reconcile_usage.assert_awaited_once()
            args, _ = client.rate_limiter.reconcile_usage.await_args
            assert args[1] == 2500

    @pytest.mark.asyncio
    async def test_generate_content_stream_skips_reconcile_without_usage_metadata(
        self, async_sample_config, mock_async_genai_client
    ):
        """generate_content_stream skips reconcile_usage when no chunk has usage_metadata"""
        chunk1 = MagicMock()
        chunk1.candidates = [MagicMock()]
        chunk1.candidates[0].content = MagicMock()
        chunk1.candidates[0].content.parts = [MagicMock(text="Part 1", thought=False)]
        chunk1.text = "Part 1"
        chunk1.usage_metadata = None

        mock_async_genai_client.aio.models.generate_content_stream = MagicMock(
            side_effect=lambda *a, **kw: _mock_stream_awaitable(chunk1)
        )

        with patch(
            "greycloud.async_client.create_client", return_value=mock_async_genai_client
        ):
            client = GreyCloudAsyncClient(async_sample_config)
            client.rate_limiter.reconcile_usage = AsyncMock()
            contents = [
                types.Content(role="user", parts=[types.Part.from_text(text="Hi")])
            ]
            async for _ in client.generate_content_stream(contents):
                pass

            client.rate_limiter.reconcile_usage.assert_not_called()


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
    async def test_generate_with_retry_fails_fast_on_invalid_config(
        self, async_sample_config
    ):
        """Deterministic config errors must not be retried or run a search."""
        async_sample_config.use_vertex_ai_search = True
        async_sample_config.vertex_ai_search_datastore = "test-datastore"

        with patch("greycloud.async_client.create_client", return_value=MagicMock()):
            client = GreyCloudAsyncClient(async_sample_config)
            contents = [
                types.Content(role="user", parts=[types.Part.from_text(text="Hi")])
            ]
            with (
                patch(
                    "greycloud.async_client.asearch_sources", new_callable=AsyncMock
                ) as mock_search,
                patch(
                    "greycloud.async_client.asyncio.sleep", new_callable=AsyncMock
                ) as mock_sleep,
            ):
                with pytest.raises(ValueError, match="MINIMAL"):
                    await client.generate_with_retry(
                        contents,
                        model="gemini-4-argon",
                        thinking_level="MINIMAL",
                    )
                mock_search.assert_not_awaited()
                mock_sleep.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_generate_with_retry_streaming_does_not_replay_thoughts(
        self, async_sample_config
    ):
        """Thoughts are not replayed when a stream fails before any text."""
        part = MagicMock()
        part.text = "thinking hard"
        part.thought = True
        chunk = MagicMock()
        chunk.candidates = [MagicMock()]
        chunk.candidates[0].content = MagicMock()
        chunk.candidates[0].content.parts = [part]

        async def _failing_awaitable():
            async def stream():
                yield chunk
                raise RuntimeError("connection dropped")

            return stream()

        with patch("greycloud.async_client.create_client") as mock_create:
            mock_genai_client = MagicMock()
            mock_genai_client.aio.models.generate_content_stream = MagicMock(
                side_effect=lambda *a, **kw: _failing_awaitable()
            )
            mock_create.return_value = mock_genai_client

            client = GreyCloudAsyncClient(async_sample_config)
            contents = [
                types.Content(role="user", parts=[types.Part.from_text(text="Hi")])
            ]
            captured = []

            async def on_thought(text):
                captured.append(text)

            stream = await client.generate_with_retry(
                contents, streaming=True, on_thought=on_thought
            )
            with patch(
                "greycloud.async_client.asyncio.sleep", new_callable=AsyncMock
            ) as mock_sleep:
                with pytest.raises(RuntimeError, match="connection dropped"):
                    async for _ in stream:
                        pass

            assert captured == ["thinking hard"]
            mock_sleep.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_generate_with_retry_streaming_fails_fast_on_invalid_config(
        self, async_sample_config
    ):
        """The streaming retry wrapper validates before any search or sleep."""
        async_sample_config.use_vertex_ai_search = True
        async_sample_config.vertex_ai_search_datastore = "test-datastore"

        with patch("greycloud.async_client.create_client", return_value=MagicMock()):
            client = GreyCloudAsyncClient(async_sample_config)
            contents = [
                types.Content(role="user", parts=[types.Part.from_text(text="Hi")])
            ]
            with (
                patch(
                    "greycloud.async_client.asearch_sources", new_callable=AsyncMock
                ) as mock_search,
                patch(
                    "greycloud.async_client.asyncio.sleep", new_callable=AsyncMock
                ) as mock_sleep,
            ):
                # F7: the ValueError must fire on the await itself, before any
                # async-generator body runs on the caller's first __anext__.
                with pytest.raises(ValueError, match="MINIMAL"):
                    await client.generate_with_retry(
                        contents,
                        streaming=True,
                        model="gemini-4-argon",
                        thinking_level="MINIMAL",
                    )
                mock_search.assert_not_awaited()
                mock_sleep.assert_not_awaited()

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
    async def test_generate_with_retry_streaming_retries_before_first_yield(
        self, async_sample_config, mock_async_genai_client
    ):
        """generate_with_retry(streaming=True) retries when exception occurs before first yield."""
        mock_chunk = MagicMock()
        mock_chunk.candidates = [MagicMock()]
        mock_chunk.candidates[0].content = MagicMock()
        mock_chunk.candidates[0].content.parts = [MagicMock()]
        mock_chunk.text = "OK"

        async def stream_raise_immediately():
            if False:
                yield
            raise RuntimeError("stream connect failed")

        async def stream_ok():
            yield mock_chunk

        call_count = 0

        async def make_stream_awaitable(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return stream_raise_immediately()
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
            assert chunks == ["OK"]
            assert call_count == 2

    @pytest.mark.asyncio
    async def test_generate_with_retry_streaming_fails_after_first_yield(
        self, async_sample_config, mock_async_genai_client
    ):
        """generate_with_retry(streaming=True) immediately raises without retry if failure occurs after first yield."""
        mock_chunk = MagicMock()
        mock_chunk.candidates = [MagicMock()]
        mock_chunk.candidates[0].content = MagicMock()
        mock_chunk.candidates[0].content.parts = [MagicMock()]
        mock_chunk.text = "OK"

        async def stream_raise_midway():
            yield mock_chunk
            raise RuntimeError("stream broken midway")

        call_count = 0

        async def make_stream_awaitable(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            return stream_raise_midway()

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
            with pytest.raises(RuntimeError, match="stream broken midway"):
                async for chunk in gen:
                    chunks.append(chunk)
            assert chunks == ["OK"]
            assert call_count == 1  # No retry attempted after yielding!

    @pytest.mark.asyncio
    async def test_generate_with_retry_forwards_streaming_options(
        self, async_sample_config, mock_async_genai_client, mocker
    ):
        """generate_with_retry(streaming=True) forwards return_events, on_thought, and request_timeout."""
        mock_chunk = MagicMock()
        mock_chunk.candidates = [MagicMock()]
        mock_chunk.candidates[0].content = MagicMock()
        mock_chunk.candidates[0].content.parts = [MagicMock()]
        mock_chunk.text = "OK"

        async def stream_ok():
            yield mock_chunk

        async def make_stream_awaitable(*args, **kwargs):
            return stream_ok()

        mock_async_genai_client.aio.models.generate_content_stream = MagicMock(
            side_effect=make_stream_awaitable
        )

        with patch(
            "greycloud.async_client.create_client", return_value=mock_async_genai_client
        ):
            client = GreyCloudAsyncClient(async_sample_config)
            spy = mocker.spy(client, "generate_content_stream")
            on_th = MagicMock()
            contents = [
                types.Content(role="user", parts=[types.Part.from_text(text="Hi")])
            ]
            gen = await client.generate_with_retry(
                contents,
                streaming=True,
                return_events=True,
                on_thought=on_th,
                request_timeout=45,
            )
            async for _ in gen:
                pass
            spy.assert_called_once()
            _, kwargs = spy.call_args
            assert kwargs["return_events"] is True
            # The retry wrapper wraps the callback to detect replay; what
            # matters is that it still delegates to the caller's callback.
            await kwargs["on_thought"]("thinking")
            on_th.assert_called_once_with("thinking")
            assert kwargs["request_timeout"] == 45

    @pytest.mark.asyncio
    async def test_generate_with_retry_forwards_request_timeout(
        self, async_sample_config, mock_async_genai_client, mocker
    ):
        """generate_with_retry(streaming=False) forwards request_timeout to generate_content."""
        with patch(
            "greycloud.async_client.create_client", return_value=mock_async_genai_client
        ):
            client = GreyCloudAsyncClient(async_sample_config)
            spy = mocker.spy(client, "generate_content")
            contents = [
                types.Content(role="user", parts=[types.Part.from_text(text="Hi")])
            ]
            await client.generate_with_retry(
                contents,
                streaming=False,
                request_timeout=60,
            )
            spy.assert_called_once()
            _, kwargs = spy.call_args
            assert kwargs["request_timeout"] == 60


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

    def test_force_reauth_blocked_under_pytest(self, async_sample_config):
        """Interactive browser re-auth is never spawned under pytest.

        Even with auto_reauth=True explicitly opted in, running under pytest
        (PYTEST_CURRENT_TEST set) must never spawn gcloud browser login.
        """
        config = GreyCloudConfig(
            project_id="test-project-id",
            location="us-east4",
            use_api_key=False,
            auto_reauth=True,
        )
        with patch("greycloud.async_client.create_client"):
            client = GreyCloudAsyncClient(config)
            with patch("greycloud.async_client.subprocess.run") as mock_run:
                assert client._force_reauth() is False
                mock_run.assert_not_called()

    def test_force_reauth_allows_user_interaction(self, async_sample_config):
        """Force re-auth allows user to see gcloud prompts.

        When re-authentication is needed, gcloud prints a URL for the user to
        visit in their browser. We must NOT suppress this output with --quiet
        or capture_output=True, otherwise the user cannot complete the flow.
        """
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

                # Patch the shared gate (instead of nuking the whole
                # environment) so the mocked interactive login path runs.
                with (
                    patch(
                        "greycloud.async_client._auto_reauth_allowed", return_value=True
                    ),
                    patch.dict(os.environ, {"DISPLAY": ":0"}),
                ):
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

                # Patch the shared gate instead of wiping the environment;
                # drop DISPLAY only (targeted, not a full env nuke)
                env = {k: v for k, v in os.environ.items() if k != "DISPLAY"}
                with (
                    patch(
                        "greycloud.async_client._auto_reauth_allowed", return_value=True
                    ),
                    patch.dict(os.environ, env, clear=True),
                    patch("sys.stdin") as mock_stdin,
                ):
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
    config-level min_grounding_query_chars threshold.
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
                    grounding_query="distributed consensus raft election safety invariants",
                )

        mock_search.assert_called_once()
        assert (
            mock_search.call_args[0][1]
            == "distributed consensus raft election safety invariants"
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
                await client.generate_content(
                    self._contents("distributed consensus raft")
                )

        assert mock_search.call_args[0][1] == "distributed consensus raft"

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
                    grounding_query="distributed consensus raft election safety invariants",
                )

        assert (
            mock_search.call_args[0][1]
            == "distributed consensus raft election safety invariants"
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
                    grounding_query="distributed consensus",
                )

        assert mock_search.call_args[0][1] == "distributed consensus"


class TestOnGroundingCallback:
    """on_grounding contract: invoked once per generate with
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


class TestReauthBehavior:
    """Opt-in interactive re-login, always-on non-interactive self-heal."""

    @pytest.mark.asyncio
    async def test_auth_error_recovers_non_interactively_with_reauth_off(
        self, async_sample_config, mock_async_genai_client
    ):
        """Async twin of the sync recovery test: default (auto_reauth off)
        still re-mints credentials non-interactively on an auth error."""
        mock_async_genai_client.aio.models.generate_content.side_effect = [
            RuntimeError("401 Unauthorized"),
            MagicMock(text="Success"),
        ]
        with patch(
            "greycloud.async_client.create_client", return_value=mock_async_genai_client
        ) as mock_create:
            client = GreyCloudAsyncClient(async_sample_config)
            contents = [
                types.Content(role="user", parts=[types.Part.from_text(text="Hi")])
            ]
            with patch("asyncio.sleep"):
                with patch("greycloud.async_client.subprocess.run") as mock_run:
                    result = await client.generate_with_retry(contents, max_retries=3)

            assert result.text == "Success"
            # construction + non-interactive recovery, nothing interactive
            assert mock_create.call_count == 2
            mock_run.assert_not_called()

    @pytest.mark.asyncio
    async def test_auth_error_exhausted_reports_disabled_interactive_reauth(
        self, async_sample_config, mock_async_genai_client
    ):
        """Async twin of the sync message test."""
        mock_async_genai_client.aio.models.generate_content.side_effect = RuntimeError(
            "401 Unauthorized"
        )
        with patch(
            "greycloud.async_client.create_client",
            side_effect=[mock_async_genai_client, RuntimeError("no creds")],
        ):
            client = GreyCloudAsyncClient(async_sample_config)
            contents = [
                types.Content(role="user", parts=[types.Part.from_text(text="Hi")])
            ]
            with patch("asyncio.sleep"):
                with pytest.raises(
                    RuntimeError, match="Interactive re-login is disabled"
                ):
                    await client.generate_with_retry(contents, max_retries=1)


class TestValidateBeforeGroundingAsync:
    """F2: config errors must be raised before any grounding search runs."""

    def _grounding_client(self, async_sample_config):
        with patch("greycloud.async_client.create_client"):
            client = GreyCloudAsyncClient(async_sample_config)
        client.config.grounding_mode = "inject"
        client.config.use_vertex_ai_search = True
        client.config.vertex_ai_search_datastore = "test-datastore"
        return client

    @staticmethod
    def _contents():
        return [
            types.Content(role="user", parts=[types.Part.from_text(text="Hi there")])
        ]

    @pytest.mark.parametrize(
        "invalid_kwargs",
        [
            {"cached_content": "caches/1"},
            {"thinking_level": "MINIMAL"},
        ],
    )
    @pytest.mark.asyncio
    async def test_async_generate_content_rejects_invalid_g4_before_search(
        self, async_sample_config, invalid_kwargs
    ):
        client = self._grounding_client(async_sample_config)
        with patch("greycloud.async_client.asearch_sources") as mock_search:
            mock_search.return_value = []
            with pytest.raises(ValueError):
                await client.generate_content(
                    self._contents(),
                    model="gemini-4-argon",
                    grounding=True,
                    **invalid_kwargs,
                )
        assert mock_search.await_count == 0

    @pytest.mark.parametrize(
        "invalid_kwargs",
        [
            {"cached_content": "caches/1"},
            {"thinking_level": "MINIMAL"},
        ],
    )
    @pytest.mark.asyncio
    async def test_async_stream_rejects_invalid_g4_before_search(
        self, async_sample_config, invalid_kwargs
    ):
        client = self._grounding_client(async_sample_config)
        with patch("greycloud.async_client.asearch_sources") as mock_search:
            mock_search.return_value = []
            with pytest.raises(ValueError):
                async for chunk in client.generate_content_stream(
                    self._contents(),
                    model="gemini-4-argon",
                    grounding=True,
                    **invalid_kwargs,
                ):
                    pass
        assert mock_search.await_count == 0


class TestCallerIncludeThoughtsOnGemini4Async:
    """F5: a caller-supplied include_thoughts must survive the G4 rebuild.

    The G4 branch rebuilds the ThinkingConfig from scratch; it hard-coded
    self.config.include_thoughts, silently flipping a caller's explicit
    include_thoughts=False. The config default must apply only when the
    caller is silent.
    """

    def _client(self, cfg):
        with patch("greycloud.async_client.create_client"):
            return GreyCloudAsyncClient(cfg)

    def test_false_honored_on_g4(self, sample_config):
        cfg = types.ThinkingConfig(thinking_level="HIGH", include_thoughts=False)
        built = self._client(sample_config)._build_generate_config(
            model="gemini-4-argon", thinking_config=cfg
        )
        assert built.thinking_config.include_thoughts is False

    def test_dict_camel_honored_on_g4(self, sample_config):
        built = self._client(sample_config)._build_generate_config(
            model="gemini-4-argon",
            thinking_config={"thinkingLevel": "HIGH", "includeThoughts": False},
        )
        assert built.thinking_config.include_thoughts is False

    def test_dict_snake_honored_on_g4(self, sample_config):
        built = self._client(sample_config)._build_generate_config(
            model="gemini-4-argon",
            thinking_config={"thinking_level": "HIGH", "include_thoughts": False},
        )
        assert built.thinking_config.include_thoughts is False

    def test_config_default_applies_when_caller_silent(self, sample_config):
        built = self._client(sample_config)._build_generate_config(
            model="gemini-4-argon", thinking_level="HIGH"
        )
        assert built.thinking_config.include_thoughts is True

    def test_config_default_applies_when_tc_has_no_flag(self, sample_config):
        cfg = types.ThinkingConfig(thinking_level="HIGH")
        built = self._client(sample_config)._build_generate_config(
            model="gemini-4-argon", thinking_config=cfg
        )
        assert built.thinking_config.include_thoughts is True

    def test_config_false_default_applies_when_caller_silent(self, sample_config):
        sample_config.include_thoughts = False
        built = self._client(sample_config)._build_generate_config(
            model="gemini-4-argon", thinking_level="HIGH"
        )
        assert built.thinking_config.include_thoughts is False

    def test_matches_non_g4_result(self, sample_config):
        """Caller's False must survive identically on the non-G4 path."""
        cfg = types.ThinkingConfig(thinking_level="HIGH", include_thoughts=False)
        built = self._client(sample_config)._build_generate_config(
            model="gemini-3.8-flash", thinking_config=cfg
        )
        assert built.thinking_config.include_thoughts is False


class TestNonG4IncludeThoughtsOmittedAsync:
    """F6: 0.4.x apps using thinking_level on gemini-3.8-flash must not
    suddenly receive thought parts. The non-G4 ThinkingConfig omits
    include_thoughts unless the caller or config explicitly opts in;
    the G4 new default (True) is asserted for contrast."""

    def test_non_g4_omits_include_thoughts_when_silent(self, async_sample_config):
        with patch("greycloud.async_client.create_client"):
            client = GreyCloudAsyncClient(async_sample_config)
        built = client._build_generate_config(
            model="gemini-3.8-flash", thinking_level="HIGH"
        )
        assert built.thinking_config.thinking_level.value == "HIGH"
        assert built.thinking_config.include_thoughts is None

    def test_non_g4_explicit_config_false_honored(self, async_sample_config):
        async_sample_config.include_thoughts = False
        with patch("greycloud.async_client.create_client"):
            client = GreyCloudAsyncClient(async_sample_config)
        built = client._build_generate_config(
            model="gemini-3.8-flash", thinking_level="HIGH"
        )
        assert built.thinking_config.include_thoughts is False

    def test_non_g4_explicit_config_true_honored(self, async_sample_config):
        async_sample_config.include_thoughts = True
        with patch("greycloud.async_client.create_client"):
            client = GreyCloudAsyncClient(async_sample_config)
        built = client._build_generate_config(
            model="gemini-3.8-flash", thinking_level="HIGH"
        )
        assert built.thinking_config.include_thoughts is True

    def test_g4_none_config_default_sends_true(self, async_sample_config):
        async_sample_config.include_thoughts = None
        with patch("greycloud.async_client.create_client"):
            client = GreyCloudAsyncClient(async_sample_config)
        built = client._build_generate_config(
            model="gemini-4-argon", thinking_level="HIGH"
        )
        assert built.thinking_config.include_thoughts is True

    def test_config_bool_validation_still_enforced(self, async_sample_config):
        from greycloud.config import GreyCloudConfig

        with pytest.raises(TypeError):
            GreyCloudConfig(include_thoughts="yes")  # type: ignore[arg-type]


class TestValidateBeforeStreamGeneratorReturnedAsync:
    """F7: generate_with_retry(streaming=True) must raise config errors on
    THE call, not on first __anext__."""

    @pytest.mark.asyncio
    async def test_async_stream_call_raises_value_error(
        self, async_sample_config, sample_contents
    ):
        with patch("greycloud.async_client.create_client"):
            client = GreyCloudAsyncClient(async_sample_config)
        with pytest.raises(ValueError):
            await client.generate_with_retry(
                sample_contents,
                streaming=True,
                model="gemini-4-argon",
                thinking_level="MINIMAL",
                max_retries=3,
            )
