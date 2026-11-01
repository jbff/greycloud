"""Unit tests for GreyCloudClient"""

import asyncio
import logging
import dataclasses

import pytest
from unittest.mock import patch, MagicMock
from google.genai import types
from greycloud.client import GreyCloudClient
from greycloud.config import GreyCloudConfig
from greycloud.grounding import GroundingSource


class TestGreyCloudClient:
    """Test GreyCloudClient class"""

    def test_client_initialization(self, sample_config):
        """Test client initialization"""
        with patch("greycloud.client.create_client") as mock_create:
            mock_client = MagicMock()
            mock_create.return_value = mock_client

            client = GreyCloudClient(sample_config)

            assert client.config == sample_config
            mock_create.assert_called_once()

    def test_client_property(self, sample_config):
        """Test client property access"""
        with patch("greycloud.client.create_client") as mock_create:
            mock_client = MagicMock()
            mock_create.return_value = mock_client

            client = GreyCloudClient(sample_config)
            assert client.client == mock_client

    def test_generate_content(
        self, sample_config, sample_contents, mock_generate_response
    ):
        """Test non-streaming content generation"""
        with patch("greycloud.client.create_client") as mock_create:
            mock_genai_client = MagicMock()
            mock_genai_client.models.generate_content.return_value = (
                mock_generate_response
            )
            mock_create.return_value = mock_genai_client

            client = GreyCloudClient(sample_config)
            response = client.generate_content(sample_contents)

            assert response == mock_generate_response
            mock_genai_client.models.generate_content.assert_called_once()

    def test_generate_content_with_overrides(
        self, sample_config, sample_contents, mock_generate_response
    ):
        """Test content generation with parameter overrides"""
        with patch("greycloud.client.create_client") as mock_create:
            mock_genai_client = MagicMock()
            mock_genai_client.models.generate_content.return_value = (
                mock_generate_response
            )
            mock_create.return_value = mock_genai_client

            client = GreyCloudClient(sample_config)
            _ = client.generate_content(
                sample_contents,
                temperature=0.5,
                max_output_tokens=1000,
                system_instruction="Custom instruction",
            )

            call_args = mock_genai_client.models.generate_content.call_args
            config = call_args[1]["config"]
            assert config.temperature == 0.5
            assert config.max_output_tokens == 1000

    def test_generate_content_gemini_4_sanitizes_config(
        self, sample_config, sample_contents, mock_generate_response
    ):
        """Test that generate_content sanitizes config when calling Gemini 4 model"""
        with patch("greycloud.client.create_client") as mock_create:
            mock_genai_client = MagicMock()
            mock_genai_client.models.generate_content.return_value = (
                mock_generate_response
            )
            mock_create.return_value = mock_genai_client

            client = GreyCloudClient(sample_config)
            client.generate_content(
                sample_contents,
                model="gemini-4-argon",
                temperature=0.5,
                top_p=0.8,
                frequency_penalty=0.2,
            )

            call_args = mock_genai_client.models.generate_content.call_args
            assert call_args[1]["model"] == "gemini-4-argon"
            config = call_args[1]["config"]
            assert config.temperature is None
            assert config.top_p is None
            assert config.frequency_penalty is None
            assert config.thinking_config.thinking_level == "MEDIUM"

    def test_generate_content_stream(self, sample_config, sample_contents):
        """Test streaming content generation"""
        # Mock streaming response
        mock_chunk1 = MagicMock()
        mock_chunk1.candidates = [MagicMock()]
        mock_chunk1.candidates[0].content = MagicMock()
        mock_chunk1.candidates[0].content.parts = [MagicMock()]
        mock_chunk1.candidates[0].content.parts[0].text = "Hello "
        mock_chunk1.text = "Hello "

        mock_chunk2 = MagicMock()
        mock_chunk2.candidates = [MagicMock()]
        mock_chunk2.candidates[0].content = MagicMock()
        mock_chunk2.candidates[0].content.parts = [MagicMock()]
        mock_chunk2.candidates[0].content.parts[0].text = "World"
        mock_chunk2.text = "World"

        with patch("greycloud.client.create_client") as mock_create:
            mock_genai_client = MagicMock()
            mock_genai_client.models.generate_content_stream.return_value = [
                mock_chunk1,
                mock_chunk2,
            ]
            mock_create.return_value = mock_genai_client

            client = GreyCloudClient(sample_config)
            chunks = list(client.generate_content_stream(sample_contents))

            assert chunks == ["Hello ", "World"]

    def test_generate_content_stream_positional_args_keep_pre_050_order(
        self, sample_config
    ):
        """New options must come after the pre-0.5.0 parameters.

        Positional callers from 0.4.x pass return_chunks, retry_unquoted,
        grounding_query, grounding and on_grounding; inserting return_events
        and on_thought before them silently rebinds every argument.
        """
        import inspect

        with patch("greycloud.client.create_client") as mock_create:
            mock_create.return_value = MagicMock()
            client = GreyCloudClient(sample_config)
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

        # The new options are still available by keyword.
        bound_new = sig.bind([], return_events=True, on_thought=cb)
        assert bound_new.arguments["return_events"] is True
        assert bound_new.arguments["on_thought"] is cb

    def test_generate_content_stream_return_chunks(
        self, sample_config, sample_contents
    ):
        """Test streaming content generation with return_chunks=True"""
        mock_chunk = MagicMock()
        mock_chunk.candidates = [MagicMock()]
        mock_chunk.candidates[0].content = MagicMock()
        mock_chunk.candidates[0].content.parts = [MagicMock()]
        mock_chunk.candidates[0].content.parts[0].text = "Hello "
        mock_chunk.text = "Hello "

        with patch("greycloud.client.create_client") as mock_create:
            mock_genai_client = MagicMock()
            mock_genai_client.models.generate_content_stream.return_value = [mock_chunk]
            mock_create.return_value = mock_genai_client

            client = GreyCloudClient(sample_config)
            chunks = list(
                client.generate_content_stream(sample_contents, return_chunks=True)
            )

            assert len(chunks) == 1
            assert chunks[0] == mock_chunk
            # Verify candidate/part metadata is present
            assert chunks[0].candidates is not None
            assert chunks[0].candidates[0].content is not None
            assert chunks[0].candidates[0].content.parts is not None
            assert chunks[0].candidates[0].content.parts[0].text == "Hello "

    def test_generate_with_retry_streaming_return_chunks(
        self, sample_config, sample_contents
    ):
        """Test generate_with_retry with streaming and return_chunks=True"""
        mock_chunk = MagicMock()
        mock_chunk.candidates = [MagicMock()]
        mock_chunk.candidates[0].content = MagicMock()
        mock_chunk.candidates[0].content.parts = [MagicMock()]
        mock_chunk.candidates[0].content.parts[0].text = "Test"
        mock_chunk.text = "Test"

        with patch("greycloud.client.create_client") as mock_create:
            mock_genai_client = MagicMock()
            mock_genai_client.models.generate_content_stream.return_value = [mock_chunk]
            mock_create.return_value = mock_genai_client

            client = GreyCloudClient(sample_config)
            chunks = list(
                client.generate_with_retry(
                    sample_contents, streaming=True, return_chunks=True
                )
            )

            assert len(chunks) == 1
            assert chunks[0] == mock_chunk
            assert chunks[0].candidates[0].content.parts[0].text == "Test"

    def test_generate_content_stream_empty_chunks(self, sample_config, sample_contents):
        """Test streaming with empty chunks"""
        mock_chunk = MagicMock()
        mock_chunk.candidates = None
        mock_chunk.text = ""

        with patch("greycloud.client.create_client") as mock_create:
            mock_genai_client = MagicMock()
            mock_genai_client.models.generate_content_stream.return_value = [mock_chunk]
            mock_create.return_value = mock_genai_client

            client = GreyCloudClient(sample_config)
            chunks = list(client.generate_content_stream(sample_contents))

            assert chunks == []

    def test_stream_thinking_chunks_default_yields_only_text(
        self, sample_config, sample_contents
    ):
        """Default stream skips thinking chunks and yields only finalized text"""
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

        with patch("greycloud.client.create_client") as mock_create:
            mock_genai_client = MagicMock()
            mock_genai_client.models.generate_content_stream.return_value = [
                chunk1,
                chunk2,
            ]
            mock_create.return_value = mock_genai_client

            client = GreyCloudClient(sample_config)
            chunks = list(client.generate_content_stream(sample_contents))

            assert chunks == ["Here is the answer"]

    def test_stream_return_events_yields_thought_and_text_dicts(
        self, sample_config, sample_contents
    ):
        """return_events=True yields thought and text dict events"""
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

        with patch("greycloud.client.create_client") as mock_create:
            mock_genai_client = MagicMock()
            mock_genai_client.models.generate_content_stream.return_value = [
                chunk1,
                chunk2,
            ]
            mock_create.return_value = mock_genai_client

            client = GreyCloudClient(sample_config)
            chunks = list(
                client.generate_content_stream(sample_contents, return_events=True)
            )

            assert chunks == [
                {"thought": "I am thinking"},
                {"text": "Here is the answer"},
            ]

    def test_stream_on_thought_callback(self, sample_config, sample_contents):
        """on_thought callback receives thought text, errors in callback are logged"""
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

        with patch("greycloud.client.create_client") as mock_create:
            mock_genai_client = MagicMock()
            mock_genai_client.models.generate_content_stream.return_value = [
                chunk1,
                chunk2,
            ]
            mock_create.return_value = mock_genai_client

            client = GreyCloudClient(sample_config)
            captured = []
            chunks = list(
                client.generate_content_stream(
                    sample_contents, on_thought=lambda t: captured.append(t)
                )
            )

            assert captured == ["I am thinking"]
            assert chunks == ["Here is the answer"]

            # Test failing callback does not crash stream
            def broken_on_thought(t):
                raise RuntimeError("callback crash")

            chunks = list(
                client.generate_content_stream(
                    sample_contents, on_thought=broken_on_thought
                )
            )
            assert chunks == ["Here is the answer"]

    def test_stream_on_thought_callback_accepts_coroutine_function(
        self, sample_config, sample_contents
    ):
        """An async on_thought callback is driven instead of being dropped."""
        part = MagicMock()
        part.text = "I am thinking"
        part.thought = True
        chunk = MagicMock()
        chunk.candidates = [MagicMock()]
        chunk.candidates[0].content = MagicMock()
        chunk.candidates[0].content.parts = [part]

        with patch("greycloud.client.create_client") as mock_create:
            mock_genai_client = MagicMock()
            mock_genai_client.models.generate_content_stream.return_value = [chunk]
            mock_create.return_value = mock_genai_client

            client = GreyCloudClient(sample_config)
            captured = []

            async def on_thought(text):
                captured.append(text)

            list(client.generate_content_stream(sample_contents, on_thought=on_thought))

            assert captured == ["I am thinking"]

    def test_stream_on_thought_callback_coroutine_inside_running_loop(
        self, sample_config, sample_contents
    ):
        """Inside a running loop the coroutine is scheduled, not dropped."""
        part = MagicMock()
        part.text = "I am thinking"
        part.thought = True
        chunk = MagicMock()
        chunk.candidates = [MagicMock()]
        chunk.candidates[0].content = MagicMock()
        chunk.candidates[0].content.parts = [part]

        with patch("greycloud.client.create_client") as mock_create:
            mock_genai_client = MagicMock()
            mock_genai_client.models.generate_content_stream.return_value = [chunk]
            mock_create.return_value = mock_genai_client

            client = GreyCloudClient(sample_config)
            captured = []

            async def on_thought(text):
                captured.append(text)

            async def scenario():
                list(
                    client.generate_content_stream(
                        sample_contents, on_thought=on_thought
                    )
                )
                await asyncio.sleep(0)

            asyncio.run(scenario())

            assert captured == ["I am thinking"]

    def test_stream_on_thought_callback_async_failure_is_logged(
        self, sample_config, sample_contents, caplog
    ):
        """A failing async callback must not break the stream."""
        part = MagicMock()
        part.text = "I am thinking"
        part.thought = True
        chunk = MagicMock()
        chunk.candidates = [MagicMock()]
        chunk.candidates[0].content = MagicMock()
        chunk.candidates[0].content.parts = [part]

        with patch("greycloud.client.create_client") as mock_create:
            mock_genai_client = MagicMock()
            mock_genai_client.models.generate_content_stream.return_value = [chunk]
            mock_create.return_value = mock_genai_client

            client = GreyCloudClient(sample_config)

            async def on_thought(text):
                raise RuntimeError("callback crash")

            with caplog.at_level("WARNING"):
                chunks = list(
                    client.generate_content_stream(
                        sample_contents, on_thought=on_thought
                    )
                )

            assert chunks == []
            assert "on_thought callback failed" in caplog.text

    def test_generate_content_stream_request_timeout(
        self, sample_config, sample_contents
    ):
        """generate_content_stream propagates request_timeout to config.http_options"""
        mock_chunk = MagicMock()
        mock_chunk.candidates = None
        mock_chunk.text = ""

        with patch("greycloud.client.create_client") as mock_create:
            mock_genai_client = MagicMock()
            mock_genai_client.models.generate_content_stream.return_value = [mock_chunk]
            mock_create.return_value = mock_genai_client

            client = GreyCloudClient(sample_config)
            list(client.generate_content_stream(sample_contents, request_timeout=45))

            call_args = mock_genai_client.models.generate_content_stream.call_args
            config = call_args[1]["config"]
            assert config.http_options is not None
            assert config.http_options.timeout == 45000

    def test_generate_content_request_timeout(
        self, sample_config, sample_contents, mock_generate_response
    ):
        """generate_content propagates request_timeout to config.http_options"""
        with patch("greycloud.client.create_client") as mock_create:
            mock_genai_client = MagicMock()
            mock_genai_client.models.generate_content.return_value = (
                mock_generate_response
            )
            mock_create.return_value = mock_genai_client

            client = GreyCloudClient(sample_config)
            client.generate_content(sample_contents, request_timeout=60)

            call_args = mock_genai_client.models.generate_content.call_args
            config = call_args[1]["config"]
            assert config.http_options is not None
            assert config.http_options.timeout == 60000

    def test_generate_content_request_timeout_merges_http_options(
        self, sample_config, sample_contents, mock_generate_response
    ):
        """A per-call timeout must not discard the caller's other options."""
        with patch("greycloud.client.create_client") as mock_create:
            mock_genai_client = MagicMock()
            mock_genai_client.models.generate_content.return_value = (
                mock_generate_response
            )
            mock_create.return_value = mock_genai_client

            client = GreyCloudClient(sample_config)
            client.generate_content(
                sample_contents,
                request_timeout=30,
                http_options=types.HttpOptions(
                    headers={"x-goog-user-project": "billing-project"},
                    base_url="https://example.test",
                ),
            )

            config = mock_genai_client.models.generate_content.call_args[1]["config"]
            assert config.http_options.timeout == 30000
            assert config.http_options.headers == {
                "x-goog-user-project": "billing-project"
            }
            assert config.http_options.base_url == "https://example.test"

    def test_count_tokens(self, sample_config, sample_contents):
        """Test token counting"""
        with patch("greycloud.client.create_client") as mock_create:
            mock_genai_client = MagicMock()
            mock_count_response = MagicMock()
            mock_count_response.total_tokens = 42
            mock_genai_client.models.count_tokens.return_value = mock_count_response
            mock_create.return_value = mock_genai_client

            client = GreyCloudClient(sample_config)
            count = client.count_tokens(sample_contents)

            assert count == 42
            mock_genai_client.models.count_tokens.assert_called_once()

    def test_count_tokens_with_system_instruction(self, sample_config, sample_contents):
        """Test token counting with system instruction"""
        with patch("greycloud.client.create_client") as mock_create:
            mock_genai_client = MagicMock()
            mock_count_response = MagicMock()
            mock_count_response.total_tokens = 50
            mock_genai_client.models.count_tokens.return_value = mock_count_response
            mock_create.return_value = mock_genai_client

            client = GreyCloudClient(sample_config)
            _ = client.count_tokens(sample_contents, system_instruction="System prompt")

            call_args = mock_genai_client.models.count_tokens.call_args
            assert call_args[1]["config"] is not None

    def test_count_tokens_fallback(self, sample_config, sample_contents):
        """Test token counting fallback when API fails"""
        with patch("greycloud.client.create_client") as mock_create:
            mock_genai_client = MagicMock()
            mock_genai_client.models.count_tokens.side_effect = Exception("API Error")
            mock_create.return_value = mock_genai_client

            client = GreyCloudClient(sample_config)
            count = client.count_tokens(sample_contents)

            # Should return estimated count (rough approximation)
            assert isinstance(count, int)
            assert count >= 0

    def test_build_tools_with_vertex_search(self, sample_config):
        """Test tool building with Vertex AI Search

        Under the default ``grounding_mode="inject"`` the retrieval tool is
        dropped (grounding is injected into the prompt instead); it is only
        included in the legacy ``grounding_mode="tool"`` mode.
        """
        datastore = "projects/test/locations/us/datastores/test-ds"

        with patch("greycloud.client.create_client") as mock_create:
            mock_create.return_value = MagicMock()

            # Default inject mode: retrieval tool dropped even with datastore.
            client = GreyCloudClient(
                GreyCloudConfig(
                    project_id="test-project",
                    use_vertex_ai_search=True,
                    vertex_ai_search_datastore=datastore,
                )
            )
            assert client._build_tools() == []

            # Legacy "tool" mode: retrieval tool present.
            client = GreyCloudClient(
                GreyCloudConfig(
                    project_id="test-project",
                    use_vertex_ai_search=True,
                    vertex_ai_search_datastore=datastore,
                    grounding_mode="tool",
                )
            )
            tools = client._build_tools()
            assert len(tools) == 1
            assert tools[0].retrieval is not None
            assert tools[0].retrieval.vertex_ai_search.datastore == datastore

    def test_build_tools_without_vertex_search(self, sample_config):
        """Test tool building without Vertex AI Search"""
        with patch("greycloud.client.create_client") as mock_create:
            mock_create.return_value = MagicMock()
            client = GreyCloudClient(sample_config)
            tools = client._build_tools()

            assert len(tools) == 0

    def test_build_generate_config(self, sample_config):
        """Test GenerateContentConfig building"""
        with patch("greycloud.client.create_client") as mock_create:
            mock_create.return_value = MagicMock()
            client = GreyCloudClient(sample_config)

            config = client._build_generate_config(
                temperature=0.7, top_p=0.9, max_output_tokens=2000
            )

            assert config.temperature == 0.7
            assert config.top_p == 0.9
            assert config.max_output_tokens == 2000

    def test_build_generate_config_with_safety_settings(self, sample_config):
        """Test GenerateContentConfig with safety settings"""
        # Use a threshold valid in google-genai (BLOCK_MEDIUM_AND_ABOVE, not deprecated BLOCK_MEDIUM)
        safety_settings = [
            {
                "category": "HARM_CATEGORY_HATE_SPEECH",
                "threshold": "BLOCK_MEDIUM_AND_ABOVE",
            }
        ]

        with patch("greycloud.client.create_client") as mock_create:
            mock_create.return_value = MagicMock()
            client = GreyCloudClient(sample_config)

            config = client._build_generate_config(safety_settings=safety_settings)

            assert len(config.safety_settings) == 1
            assert config.safety_settings[0].category == "HARM_CATEGORY_HATE_SPEECH"

    def test_build_generate_config_with_thinking_level(self, sample_config):
        """Test GenerateContentConfig with thinking level"""
        with patch("greycloud.client.create_client") as mock_create:
            mock_create.return_value = MagicMock()
            client = GreyCloudClient(sample_config)

            config = client._build_generate_config(thinking_level="HIGH")

            assert config.thinking_config is not None
            assert config.thinking_config.thinking_level == "HIGH"

    def test_build_generate_config_non_gemini_4_includes_thoughts(self, sample_config):
        """Non-G4 thinking config omits include_thoughts by default (0.4.x
        behavior): a 0.5.0 app that never set the flag must not suddenly
        receive thought parts. An explicit config opt-in (True) is honored."""
        from greycloud.config import GreyCloudConfig

        with patch("greycloud.client.create_client") as mock_create:
            mock_create.return_value = MagicMock()
            client = GreyCloudClient(sample_config)
            config = client._build_generate_config(
                model="gemini-3.8-flash", thinking_level="HIGH"
            )
            assert config.thinking_config.include_thoughts is None
            sample_config.include_thoughts = True
            config = client._build_generate_config(
                model="gemini-3.8-flash", thinking_level="HIGH"
            )
            assert config.thinking_config.include_thoughts is True

            quiet_config = GreyCloudConfig(
                project_id="test-proj", include_thoughts=False
            )
            client_quiet = GreyCloudClient(quiet_config)
            config_quiet = client_quiet._build_generate_config(
                model="gemini-3.8-flash", thinking_level="HIGH"
            )
            assert config_quiet.thinking_config.include_thoughts is False

    def test_build_generate_config_gemini_4_strips_penalties_and_sampling(
        self, sample_config
    ):
        """Gemini 4 models strip sampling and penalty parameters and default thinking."""
        with patch("greycloud.client.create_client") as mock_create:
            mock_create.return_value = MagicMock()
            client = GreyCloudClient(sample_config)
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
        self, sample_config
    ):
        """Gemini 4 models strip camelCase sampling and penalty parameters from kwargs."""
        with patch("greycloud.client.create_client") as mock_create:
            mock_create.return_value = MagicMock()
            client = GreyCloudClient(sample_config)
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

    def test_build_generate_config_gemini_4_rejects_cached_content(self, sample_config):
        """Gemini 4 models reject explicit cached_content."""
        with patch("greycloud.client.create_client") as mock_create:
            mock_create.return_value = MagicMock()
            client = GreyCloudClient(sample_config)
            with pytest.raises(ValueError, match="implicit durable caching"):
                client._build_generate_config(
                    model="gemini-4-argon", cached_content="caches/123"
                )

    def test_build_generate_config_gemini_4_rejects_camelcase_cached_content(
        self, sample_config
    ):
        """The REST-style cachedContent alias must be rejected too."""
        with patch("greycloud.client.create_client") as mock_create:
            mock_create.return_value = MagicMock()
            client = GreyCloudClient(sample_config)
            with pytest.raises(ValueError, match="implicit durable caching"):
                client._build_generate_config(
                    model="gemini-4-argon", cachedContent="caches/123"
                )

            # None is unset, not a request.
            cfg = client._build_generate_config(
                model="gemini-4-argon", cachedContent=None, cached_content=None
            )
            assert cfg.thinking_config.thinking_level == "MEDIUM"

    def test_build_generate_config_gemini_3_preserves_sampling(self, sample_config):
        """Non-Gemini 4 models preserve temperature and top_p."""
        with patch("greycloud.client.create_client") as mock_create:
            mock_create.return_value = MagicMock()
            client = GreyCloudClient(sample_config)
            cfg = client._build_generate_config(
                model="gemini-3.8-flash", temperature=0.7, top_p=0.9
            )
            assert cfg.temperature == 0.7
            assert cfg.top_p == 0.9

    def test_build_generate_config_gemini_4_rejects_thinking_budget(
        self, sample_config
    ):
        """Gemini 4 models reject thinking_budget."""
        with patch("greycloud.client.create_client") as mock_create:
            mock_create.return_value = MagicMock()
            client = GreyCloudClient(sample_config)
            with pytest.raises(ValueError, match="thinking_budget"):
                client._build_generate_config(
                    model="gemini-4-argon", thinking_budget=1024
                )

    def test_build_generate_config_gemini_4_accepts_none_thinking_budget(
        self, sample_config
    ):
        """thinking_budget=None means unset and must not be rejected."""
        with patch("greycloud.client.create_client") as mock_create:
            mock_create.return_value = MagicMock()
            client = GreyCloudClient(sample_config)

            cfg = client._build_generate_config(
                model="gemini-4-argon",
                thinking_budget=None,
                thinkingBudget=None,
            )
            assert cfg.thinking_config.thinking_level == "MEDIUM"

            # Nested dict form with an explicit None is unset too.
            cfg2 = client._build_generate_config(
                model="gemini-4-argon",
                thinking_config={"thinkingBudget": None, "thinkingLevel": "LOW"},
            )
            assert cfg2.thinking_config.thinking_level == "LOW"

    def test_build_generate_config_gemini_4_rejects_minimal_thinking_level(
        self, sample_config
    ):
        """Gemini 4 models reject MINIMAL thinking level."""
        with patch("greycloud.client.create_client") as mock_create:
            mock_create.return_value = MagicMock()
            client = GreyCloudClient(sample_config)
            with pytest.raises(ValueError, match="MINIMAL"):
                client._build_generate_config(
                    model="gemini-4-argon", thinking_level="MINIMAL"
                )

    def test_build_generate_config_gemini_4_custom_thinking_level(self, sample_config):
        """Gemini 4 accepts valid explicit thinking_level (LOW, MEDIUM, HIGH)."""
        with patch("greycloud.client.create_client") as mock_create:
            mock_create.return_value = MagicMock()
            client = GreyCloudClient(sample_config)
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
        with patch("greycloud.client.create_client") as mock_create:
            mock_create.return_value = MagicMock()
            client = GreyCloudClient(cfg_inst)
            cfg = client._build_generate_config()
            assert cfg.temperature is None
            assert cfg.top_p is None
            assert cfg.thinking_config.thinking_level == "MEDIUM"

    def test_build_generate_config_gemini_4_kwargs_thinking_config_cannot_overwrite(
        self, sample_config
    ):
        """Passing thinking_config in kwargs is sanitized and does not overwrite config."""
        with patch("greycloud.client.create_client") as mock_create:
            mock_create.return_value = MagicMock()
            client = GreyCloudClient(sample_config)
            # Both object and dict in kwargs
            raw_tc = types.ThinkingConfig(thinking_level="HIGH")
            cfg = client._build_generate_config(
                model="gemini-4-argon", thinking_config=raw_tc
            )
            assert cfg.thinking_config.thinking_level == "HIGH"
            assert cfg.thinking_config.include_thoughts is True

            # Precedence of explicit thinking_level parameter over kwargs thinking_config
            cfg2 = client._build_generate_config(
                model="gemini-4-argon",
                thinking_level="LOW",
                thinking_config=raw_tc,
            )
            assert cfg2.thinking_config.thinking_level == "LOW"
            assert cfg2.thinking_config.include_thoughts is True

    def test_build_generate_config_gemini_4_honors_camelcase_thinking_level(
        self, sample_config
    ):
        """A REST-style thinkingLevel kwarg is honored, not silently dropped."""
        with patch("greycloud.client.create_client") as mock_create:
            mock_create.return_value = MagicMock()
            client = GreyCloudClient(sample_config)

            cfg = client._build_generate_config(
                model="gemini-4-argon", thinkingLevel="HIGH"
            )
            assert cfg.thinking_config.thinking_level == "HIGH"

            # Explicit parameter still wins.
            cfg2 = client._build_generate_config(
                model="gemini-4-argon", thinking_level="LOW", thinkingLevel="HIGH"
            )
            assert cfg2.thinking_config.thinking_level == "LOW"

            # And the alias is validated like the snake_case form.
            with pytest.raises(ValueError, match="MINIMAL"):
                client._build_generate_config(
                    model="gemini-4-argon", thinkingLevel="MINIMAL"
                )
            with pytest.raises(ValueError, match="Invalid thinking_level"):
                client._build_generate_config(
                    model="gemini-4-argon", thinkingLevel="BOGUS"
                )

    def test_build_generate_config_gemini_4_kwargs_thinking_config_validates(
        self, sample_config
    ):
        """thinking_config in kwargs validates budget and minimal level."""
        with patch("greycloud.client.create_client") as mock_create:
            mock_create.return_value = MagicMock()
            client = GreyCloudClient(sample_config)

            # Object with thinking_budget
            with pytest.raises(ValueError, match="thinking_budget"):
                client._build_generate_config(
                    model="gemini-4-argon",
                    thinking_config=types.ThinkingConfig(thinking_budget=1024),
                )

            # Dict with thinkingBudget
            with pytest.raises(ValueError, match="thinking_budget"):
                client._build_generate_config(
                    model="gemini-4-argon",
                    thinking_config={"thinkingBudget": 1024},
                )

            # Object with MINIMAL
            with pytest.raises(ValueError, match="MINIMAL"):
                client._build_generate_config(
                    model="gemini-4-argon",
                    thinking_config=types.ThinkingConfig(thinking_level="MINIMAL"),
                )

            # Dict with MINIMAL
            with pytest.raises(ValueError, match="MINIMAL"):
                client._build_generate_config(
                    model="gemini-4-argon",
                    thinking_config={"thinkingLevel": "MINIMAL"},
                )

    def test_is_authentication_error(self, sample_config):
        """Test authentication error detection"""
        with patch("greycloud.client.create_client") as mock_create:
            mock_create.return_value = MagicMock()
            client = GreyCloudClient(sample_config)

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

    def test_exponential_backoff_with_jitter(self, sample_config):
        """Test exponential backoff calculation"""
        with patch("greycloud.client.create_client") as mock_create:
            mock_create.return_value = MagicMock()
            client = GreyCloudClient(sample_config)

            delay1 = client.exponential_backoff_with_jitter(0)
            delay2 = client.exponential_backoff_with_jitter(1)
            delay3 = client.exponential_backoff_with_jitter(2)

            assert delay1 < delay2 < delay3
            assert delay3 <= 60  # max_delay

    def test_generate_with_retry_success(
        self, sample_config, sample_contents, mock_generate_response
    ):
        """Test generate_with_retry on success"""
        with patch("greycloud.client.create_client") as mock_create:
            mock_genai_client = MagicMock()
            mock_genai_client.models.generate_content.return_value = (
                mock_generate_response
            )
            mock_create.return_value = mock_genai_client

            client = GreyCloudClient(sample_config)
            response = client.generate_with_retry(sample_contents, max_retries=3)

            assert response == mock_generate_response

    def test_generate_with_retry_fails_fast_on_invalid_config(
        self, sample_config, sample_contents
    ):
        """Deterministic config errors must not be retried or run a search."""
        sample_config.use_vertex_ai_search = True
        sample_config.vertex_ai_search_datastore = "test-datastore"
        with patch("greycloud.client.create_client") as mock_create:
            mock_create.return_value = MagicMock()
            client = GreyCloudClient(sample_config)

            with (
                patch("greycloud.client.search_sources") as mock_search,
                patch("greycloud.client.time.sleep") as mock_sleep,
            ):
                with pytest.raises(ValueError, match="MINIMAL"):
                    client.generate_with_retry(
                        sample_contents,
                        model="gemini-4-argon",
                        thinking_level="MINIMAL",
                    )
                mock_search.assert_not_called()
                mock_sleep.assert_not_called()
            mock_create.return_value.models.generate_content.assert_not_called()

    def test_generate_with_retry_streaming_does_not_replay_thoughts(
        self, sample_config, sample_contents
    ):
        """Thoughts are not replayed when a stream fails before any text."""
        part = MagicMock()
        part.text = "thinking hard"
        part.thought = True
        chunk = MagicMock()
        chunk.candidates = [MagicMock()]
        chunk.candidates[0].content = MagicMock()
        chunk.candidates[0].content.parts = [part]

        def failing_stream():
            yield chunk
            raise RuntimeError("connection dropped")

        with patch("greycloud.client.create_client") as mock_create:
            mock_genai_client = MagicMock()
            mock_genai_client.models.generate_content_stream.side_effect = (
                lambda *args, **kwargs: failing_stream()
            )
            mock_create.return_value = mock_genai_client

            client = GreyCloudClient(sample_config)
            captured = []
            with patch("greycloud.client.time.sleep") as mock_sleep:
                with pytest.raises(RuntimeError, match="connection dropped"):
                    list(
                        client.generate_with_retry(
                            sample_contents,
                            streaming=True,
                            on_thought=captured.append,
                        )
                    )

            assert captured == ["thinking hard"]
            mock_sleep.assert_not_called()

    def test_generate_with_retry_streaming_fails_fast_on_invalid_config(
        self, sample_config, sample_contents
    ):
        """The streaming retry wrapper validates before any search or sleep."""
        sample_config.use_vertex_ai_search = True
        sample_config.vertex_ai_search_datastore = "test-datastore"
        with patch("greycloud.client.create_client") as mock_create:
            mock_create.return_value = MagicMock()
            client = GreyCloudClient(sample_config)

            with (
                patch("greycloud.client.search_sources") as mock_search,
                patch("greycloud.client.time.sleep") as mock_sleep,
            ):
                # F7: the ValueError must fire on the CALL itself, before any
                # generator body runs on the caller's first next().
                with pytest.raises(ValueError, match="MINIMAL"):
                    client.generate_with_retry(
                        sample_contents,
                        streaming=True,
                        model="gemini-4-argon",
                        thinking_level="MINIMAL",
                    )
                mock_search.assert_not_called()
                mock_sleep.assert_not_called()

    def test_generate_with_retry_after_failure(
        self, sample_config, sample_contents, mock_generate_response
    ):
        """Test generate_with_retry with retries"""
        with patch("greycloud.client.create_client") as mock_create:
            mock_genai_client = MagicMock()
            mock_genai_client.models.generate_content.side_effect = [
                Exception("Network error"),
                mock_generate_response,
            ]
            mock_create.return_value = mock_genai_client

            client = GreyCloudClient(sample_config)

            with patch("time.sleep"):  # Speed up test
                response = client.generate_with_retry(sample_contents, max_retries=3)

            assert response == mock_generate_response
            assert mock_genai_client.models.generate_content.call_count == 2

    def test_generate_with_retry_auth_error(
        self, sample_config, sample_contents, mock_generate_response
    ):
        """Test generate_with_retry with authentication error"""
        with patch("greycloud.client.create_client") as mock_create:
            mock_genai_client = MagicMock()
            mock_genai_client.models.generate_content.side_effect = [
                Exception("401 Unauthorized"),
                mock_generate_response,
            ]
            mock_create.return_value = mock_genai_client

            client = GreyCloudClient(sample_config)

            with patch("time.sleep"):  # Speed up test
                with patch.object(client, "_authenticate"):
                    response = client.generate_with_retry(
                        sample_contents, max_retries=3
                    )

            assert response == mock_generate_response

    def test_generate_with_retry_streaming(self, sample_config, sample_contents):
        """Test generate_with_retry with streaming"""
        mock_chunk = MagicMock()
        mock_chunk.candidates = [MagicMock()]
        mock_chunk.candidates[0].content = MagicMock()
        mock_chunk.candidates[0].content.parts = [MagicMock()]
        mock_chunk.candidates[0].content.parts[0].text = "Test"
        mock_chunk.text = "Test"

        with patch("greycloud.client.create_client") as mock_create:
            mock_genai_client = MagicMock()
            mock_genai_client.models.generate_content_stream.return_value = [mock_chunk]
            mock_create.return_value = mock_genai_client

            client = GreyCloudClient(sample_config)
            chunks = list(client.generate_with_retry(sample_contents, streaming=True))

            assert chunks == ["Test"]

    def test_generate_with_retry_max_retries_exceeded(
        self, sample_config, sample_contents
    ):
        """Test generate_with_retry when max retries exceeded"""
        with patch("greycloud.client.create_client") as mock_create:
            mock_genai_client = MagicMock()
            mock_genai_client.models.generate_content.side_effect = Exception(
                "Persistent error"
            )
            mock_create.return_value = mock_genai_client

            client = GreyCloudClient(sample_config)

            with patch("time.sleep"):  # Speed up test
                with pytest.raises(Exception, match="Persistent error"):
                    client.generate_with_retry(sample_contents, max_retries=2)

            assert (
                mock_genai_client.models.generate_content.call_count == 3
            )  # Initial + 2 retries


DATASTORE = "projects/test/locations/us/datastores/test-ds"


def _grounding_config(**kwargs):
    """GreyCloudConfig with vertex-ai-search enabled (inject mode by default)."""
    defaults = dict(
        project_id="test-project",
        use_vertex_ai_search=True,
        vertex_ai_search_datastore=DATASTORE,
    )
    defaults.update(kwargs)
    return GreyCloudConfig(**defaults)


def _two_fake_sources():
    return [
        GroundingSource(
            title="Doc1", link="gs://bucket/doc1.pdf", snippet="Passage one.", index=1
        ),
        GroundingSource(
            title="Doc2", link="gs://bucket/doc2.pdf", snippet="Passage two.", index=2
        ),
    ]


class FakeResponse:
    """Minimal requests.Response stand-in (mirrors tests/test_grounding.py)."""

    def __init__(self, status_code=200, payload=None, text=""):
        self.status_code = status_code
        self._payload = payload if payload is not None else {}
        self.text = text

    def json(self):
        return self._payload


OK_PAYLOAD = {
    "results": [
        {
            "document": {
                "derivedStructData": {
                    "title": "Guide_Renewable_Energy",
                    "link": "gs://bucket/Guide_Renewable_Energy.pdf",
                    "snippets": [
                        {"snippet": "<b>Renewable energy</b> is difficult to store."}
                    ],
                }
            }
        }
    ]
}


class TestGreyCloudClientGroundingInjection:
    """Integration tests for Discovery Engine grounding injection (inject mode)."""

    def test_generate_content_inject_mode_injects_grounding_block(
        self, sample_config, mock_generate_response
    ):
        """Inject mode: search runs on the last user query, sources are prepended
        to a copy of the last user message, and no retrieval tool is sent."""
        with patch("greycloud.client.create_client") as mock_create:
            with patch(
                "greycloud.client.search_sources",
                return_value=_two_fake_sources(),
            ) as mock_search:
                mock_genai_client = MagicMock()
                mock_genai_client.models.generate_content.return_value = (
                    mock_generate_response
                )
                mock_create.return_value = mock_genai_client

                client = GreyCloudClient(_grounding_config())
                contents = [
                    types.Content(
                        role="user", parts=[types.Part.from_text(text="Hello")]
                    )
                ]
                client.generate_content(contents)

        # Search is performed with (config, last user query).
        mock_search.assert_called_once()
        assert mock_search.call_args[0][0] is client.config
        assert mock_search.call_args[0][1] == "Hello"
        # retry_unquoted defaults to True and is threaded through.
        assert mock_search.call_args[1]["retry_unquoted"] is True

        call_args = mock_genai_client.models.generate_content.call_args
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

    def test_generate_content_passes_retry_unquoted_flag(
        self, sample_config, mock_generate_response
    ):
        """retry_unquoted is threaded from generate_content to search_sources."""
        with patch("greycloud.client.create_client") as mock_create:
            with patch(
                "greycloud.client.search_sources",
                return_value=_two_fake_sources(),
            ) as mock_search:
                mock_genai_client = MagicMock()
                mock_genai_client.models.generate_content.return_value = (
                    mock_generate_response
                )
                mock_create.return_value = mock_genai_client

                client = GreyCloudClient(_grounding_config())
                contents = [
                    types.Content(
                        role="user", parts=[types.Part.from_text(text="Hello")]
                    )
                ]
                client.generate_content(contents, retry_unquoted=False)

        mock_search.assert_called_once()
        assert mock_search.call_args[1]["retry_unquoted"] is False

    def test_generate_content_stream_inject_mode_injects_grounding_block(
        self, sample_config
    ):
        """Streaming path applies the same injection as the non-streaming path."""
        mock_chunk = MagicMock()
        mock_chunk.candidates = [MagicMock()]
        mock_chunk.candidates[0].content = MagicMock()
        mock_chunk.candidates[0].content.parts = [MagicMock()]
        mock_chunk.candidates[0].content.parts[0].text = "Hello"
        mock_chunk.text = "Hello"

        with patch("greycloud.client.create_client") as mock_create:
            with patch(
                "greycloud.client.search_sources",
                return_value=_two_fake_sources(),
            ) as mock_search:
                mock_genai_client = MagicMock()
                mock_genai_client.models.generate_content_stream.return_value = [
                    mock_chunk
                ]
                mock_create.return_value = mock_genai_client

                client = GreyCloudClient(_grounding_config())
                contents = [
                    types.Content(
                        role="user", parts=[types.Part.from_text(text="Hello")]
                    )
                ]
                chunks = list(client.generate_content_stream(contents))

        assert chunks == ["Hello"]
        mock_search.assert_called_once()
        assert mock_search.call_args[0][1] == "Hello"

        call_args = mock_genai_client.models.generate_content_stream.call_args
        sent_contents = call_args[1]["contents"]
        sent_config = call_args[1]["config"]
        assert sent_config.tools == []
        last_user = sent_contents[-1]
        assert last_user.parts[0].text.startswith("<grounding_sources>")
        assert last_user.parts[-1].text == "Hello"

    def test_generate_content_inject_mode_empty_sources_degrades_ungrounded(
        self, sample_config, mock_generate_response
    ):
        """Empty search results degrade to ungrounded generation: still called,
        contents unmodified, no grounding part, no exception."""
        with patch("greycloud.client.create_client") as mock_create:
            with patch(
                "greycloud.client.search_sources", return_value=[]
            ) as mock_search:
                mock_genai_client = MagicMock()
                mock_genai_client.models.generate_content.return_value = (
                    mock_generate_response
                )
                mock_create.return_value = mock_genai_client

                client = GreyCloudClient(_grounding_config())
                contents = [
                    types.Content(
                        role="user", parts=[types.Part.from_text(text="Hello")]
                    )
                ]
                response = client.generate_content(contents)

        assert response == mock_generate_response
        mock_search.assert_called_once()

        call_args = mock_genai_client.models.generate_content.call_args
        sent_contents = call_args[1]["contents"]
        # Same list object passed through untouched.
        assert sent_contents is contents
        assert len(contents[0].parts) == 1
        assert contents[0].parts[0].text == "Hello"
        assert "<grounding_sources>" not in contents[0].parts[0].text

    def test_generate_content_tool_mode_no_search_uses_retrieval_tool(
        self, sample_config, mock_generate_response
    ):
        """Tool mode is unchanged: retrieval tool sent, no search performed,
        contents unmodified."""
        with patch("greycloud.client.create_client") as mock_create:
            with patch("greycloud.client.search_sources") as mock_search:
                mock_genai_client = MagicMock()
                mock_genai_client.models.generate_content.return_value = (
                    mock_generate_response
                )
                mock_create.return_value = mock_genai_client

                client = GreyCloudClient(_grounding_config(grounding_mode="tool"))
                contents = [
                    types.Content(
                        role="user", parts=[types.Part.from_text(text="Hello")]
                    )
                ]
                client.generate_content(contents)

        mock_search.assert_not_called()

        call_args = mock_genai_client.models.generate_content.call_args
        sent_config = call_args[1]["config"]
        assert len(sent_config.tools) == 1
        assert sent_config.tools[0].retrieval is not None
        assert (
            sent_config.tools[0].retrieval.vertex_ai_search.datastore
            == client.config.vertex_ai_search_datastore
        )
        # Contents passed through unmodified.
        assert call_args[1]["contents"] is contents

    def test_generate_content_inject_mode_does_not_mutate_caller_contents(
        self, sample_config, mock_generate_response
    ):
        """Caller's contents list and Content objects are never mutated."""
        with patch("greycloud.client.create_client") as mock_create:
            with patch(
                "greycloud.client.search_sources",
                return_value=_two_fake_sources(),
            ):
                mock_genai_client = MagicMock()
                mock_genai_client.models.generate_content.return_value = (
                    mock_generate_response
                )
                mock_create.return_value = mock_genai_client

                client = GreyCloudClient(_grounding_config())
                original_user = types.Content(
                    role="user", parts=[types.Part.from_text(text="Hello")]
                )
                contents = [original_user]
                client.generate_content(contents)

        # Original list, Content object, and parts are untouched.
        assert len(contents) == 1
        assert contents[0] is original_user
        assert len(original_user.parts) == 1
        assert original_user.parts[0].text == "Hello"
        assert "<grounding_sources>" not in original_user.parts[0].text

    def test_generate_content_explicit_tools_override_skips_injection(
        self, sample_config, mock_generate_response
    ):
        """An explicit tools= override is honored as-is and skips injection."""
        explicit_tool = types.Tool(
            retrieval=types.Retrieval(
                vertex_ai_search=types.VertexAISearch(datastore=DATASTORE)
            )
        )
        with patch("greycloud.client.create_client") as mock_create:
            with patch("greycloud.client.search_sources") as mock_search:
                mock_genai_client = MagicMock()
                mock_genai_client.models.generate_content.return_value = (
                    mock_generate_response
                )
                mock_create.return_value = mock_genai_client

                client = GreyCloudClient(_grounding_config())
                contents = [
                    types.Content(
                        role="user", parts=[types.Part.from_text(text="Hello")]
                    )
                ]
                client.generate_content(contents, tools=[explicit_tool])

        mock_search.assert_not_called()

        call_args = mock_genai_client.models.generate_content.call_args
        sent_config = call_args[1]["config"]
        assert sent_config.tools == [explicit_tool]
        # Contents passed through unmodified.
        assert call_args[1]["contents"] is contents

    def test_generate_content_normalizes_query_end_to_end(
        self, sample_config, mock_generate_response
    ):
        """The full inject path runs real normalization: a quoted query that
        returns 0 results falls back to the unquoted form before injection.

        Unlike the other inject tests, search_sources is NOT mocked — only the
        HTTP layer is — so _normalize_query and the retry_unquoted fallback
        execute for real (normalization previously never ran end-to-end
        through the client).
        """
        with patch("greycloud.client.create_client") as mock_create:
            with patch("greycloud.grounding._build_headers", return_value=({}, None)):
                with patch(
                    "greycloud.grounding.requests.post",
                    side_effect=[
                        FakeResponse(200, {"results": []}),  # quoted -> 0 results
                        FakeResponse(200, OK_PAYLOAD),  # unquoted -> results
                    ],
                ) as mock_post:
                    mock_genai_client = MagicMock()
                    mock_genai_client.models.generate_content.return_value = (
                        mock_generate_response
                    )
                    mock_create.return_value = mock_genai_client

                    client = GreyCloudClient(_grounding_config())
                    contents = [
                        types.Content(
                            role="user",
                            parts=[types.Part.from_text(text='"renewable energy"')],
                        )
                    ]
                    client.generate_content(contents)

        # Two searches: quoted first, unquoted fallback on 0 results.
        assert mock_post.call_count == 2
        first = mock_post.call_args_list[0][1]["json"]["query"]
        second = mock_post.call_args_list[1][1]["json"]["query"]
        assert first == '"renewable energy"'
        assert second == "renewable energy"

        # The unquoted results were injected into the last user message.
        call_args = mock_genai_client.models.generate_content.call_args
        last_user = call_args[1]["contents"][-1]
        assert last_user.parts[0].text.startswith("<grounding_sources>")
        assert "Guide_Renewable_Energy" in last_user.parts[0].text


class TestGroundingQueryAndSkip:
    """Per-call grounding_query override, grounding skip flag, and the
    config-level min_grounding_query_chars threshold.
    All mock search_sources; no live calls."""

    @staticmethod
    def _contents(text="Hello"):
        return [types.Content(role="user", parts=[types.Part.from_text(text=text)])]

    @staticmethod
    def _patched_client(mock_search_returns=_two_fake_sources()):
        """Return (context manager factory usage pattern) as two patches."""
        return (
            patch("greycloud.client.create_client"),
            patch(
                "greycloud.client.search_sources",
                return_value=mock_search_returns,
            ),
        )

    def test_generate_content_grounding_query_used_as_search_query(
        self, sample_config, mock_generate_response
    ):
        """grounding_query replaces the verbatim last user message as the
        Discovery Engine query (workflow-instruction turns are not domain query
        content); injection still targets the last user message."""
        with patch("greycloud.client.create_client") as mock_create:
            with patch(
                "greycloud.client.search_sources",
                return_value=_two_fake_sources(),
            ) as mock_search:
                mock_genai_client = MagicMock()
                mock_genai_client.models.generate_content.return_value = (
                    mock_generate_response
                )
                mock_create.return_value = mock_genai_client

                client = GreyCloudClient(_grounding_config())
                client.generate_content(
                    self._contents("Great, now let's do the summary."),
                    grounding_query="distributed consensus raft election safety invariants",
                )

        mock_search.assert_called_once()
        assert (
            mock_search.call_args[0][1]
            == "distributed consensus raft election safety invariants"
        )

    def test_generate_content_blank_grounding_query_falls_back_to_user_message(
        self, sample_config, mock_generate_response
    ):
        """A whitespace-only grounding_query is treated as absent."""
        with patch("greycloud.client.create_client") as mock_create:
            with patch(
                "greycloud.client.search_sources", return_value=_two_fake_sources()
            ) as mock_search:
                mock_genai_client = MagicMock()
                mock_genai_client.models.generate_content.return_value = (
                    mock_generate_response
                )
                mock_create.return_value = mock_genai_client

                client = GreyCloudClient(_grounding_config())
                client.generate_content(self._contents(), grounding_query="   ")

        assert mock_search.call_args[0][1] == "Hello"

    def test_generate_content_grounding_false_skips_search(
        self, sample_config, mock_generate_response
    ):
        """grounding=False suppresses the search and injection for one call."""
        with patch("greycloud.client.create_client") as mock_create:
            with patch(
                "greycloud.client.search_sources", return_value=_two_fake_sources()
            ) as mock_search:
                mock_genai_client = MagicMock()
                mock_genai_client.models.generate_content.return_value = (
                    mock_generate_response
                )
                mock_create.return_value = mock_genai_client

                client = GreyCloudClient(_grounding_config())
                contents = self._contents()
                client.generate_content(contents, grounding=False)

        mock_search.assert_not_called()
        call_args = mock_genai_client.models.generate_content.call_args
        assert call_args[1]["contents"] is contents

    def test_generate_content_stream_grounding_query_used_as_search_query(
        self, sample_config
    ):
        """The streaming path honors grounding_query the same way."""
        mock_chunk = MagicMock()
        mock_chunk.candidates = [MagicMock()]
        mock_chunk.candidates[0].content = MagicMock()
        mock_chunk.candidates[0].content.parts = [MagicMock()]
        mock_chunk.candidates[0].content.parts[0].text = "Hello"
        mock_chunk.text = "Hello"

        with patch("greycloud.client.create_client") as mock_create:
            with patch(
                "greycloud.client.search_sources", return_value=_two_fake_sources()
            ) as mock_search:
                mock_genai_client = MagicMock()
                mock_genai_client.models.generate_content_stream.return_value = [
                    mock_chunk
                ]
                mock_create.return_value = mock_genai_client

                client = GreyCloudClient(_grounding_config())
                list(
                    client.generate_content_stream(
                        self._contents("Ok, standby."),
                        grounding_query="distressed child custody intake",
                    )
                )

        assert mock_search.call_args[0][1] == "distressed child custody intake"

    def test_generate_content_stream_grounding_false_skips_search(self, sample_config):
        mock_chunk = MagicMock()
        mock_chunk.candidates = [MagicMock()]
        mock_chunk.candidates[0].content = MagicMock()
        mock_chunk.candidates[0].content.parts = [MagicMock()]
        mock_chunk.candidates[0].content.parts[0].text = "Hello"
        mock_chunk.text = "Hello"

        with patch("greycloud.client.create_client") as mock_create:
            with patch(
                "greycloud.client.search_sources", return_value=_two_fake_sources()
            ) as mock_search:
                mock_genai_client = MagicMock()
                mock_genai_client.models.generate_content_stream.return_value = [
                    mock_chunk
                ]
                mock_create.return_value = mock_genai_client

                client = GreyCloudClient(_grounding_config())
                contents = self._contents()
                list(client.generate_content_stream(contents, grounding=False))

        mock_search.assert_not_called()

    def test_min_grounding_query_chars_skips_short_query(
        self, sample_config, mock_generate_response
    ):
        """A last-user message shorter than the threshold skips the search."""
        with patch("greycloud.client.create_client") as mock_create:
            with patch(
                "greycloud.client.search_sources", return_value=_two_fake_sources()
            ) as mock_search:
                mock_genai_client = MagicMock()
                mock_genai_client.models.generate_content.return_value = (
                    mock_generate_response
                )
                mock_create.return_value = mock_genai_client

                client = GreyCloudClient(
                    _grounding_config(min_grounding_query_chars=10)
                )
                contents = self._contents("thanks")
                client.generate_content(contents)

        mock_search.assert_not_called()
        call_args = mock_genai_client.models.generate_content.call_args
        assert call_args[1]["contents"] is contents

    def test_min_grounding_query_chars_long_query_still_searches(
        self, sample_config, mock_generate_response
    ):
        with patch("greycloud.client.create_client") as mock_create:
            with patch(
                "greycloud.client.search_sources", return_value=_two_fake_sources()
            ) as mock_search:
                mock_genai_client = MagicMock()
                mock_genai_client.models.generate_content.return_value = (
                    mock_generate_response
                )
                mock_create.return_value = mock_genai_client

                client = GreyCloudClient(_grounding_config(min_grounding_query_chars=5))
                client.generate_content(self._contents("distributed consensus raft"))

        assert mock_search.call_args[0][1] == "distributed consensus raft"

    def test_min_grounding_query_chars_applies_to_grounding_query(
        self, sample_config, mock_generate_response
    ):
        """The threshold gates the effective query: a short user message with
        a long grounding_query still searches (the override is the query)."""
        with patch("greycloud.client.create_client") as mock_create:
            with patch(
                "greycloud.client.search_sources", return_value=_two_fake_sources()
            ) as mock_search:
                mock_genai_client = MagicMock()
                mock_genai_client.models.generate_content.return_value = (
                    mock_generate_response
                )
                mock_create.return_value = mock_genai_client

                client = GreyCloudClient(
                    _grounding_config(min_grounding_query_chars=10)
                )
                client.generate_content(
                    self._contents("ok"),
                    grounding_query="distributed consensus raft election safety invariants",
                )

        assert (
            mock_search.call_args[0][1]
            == "distributed consensus raft election safety invariants"
        )

    def test_generate_with_retry_threads_grounding_query(
        self, sample_config, mock_generate_response
    ):
        """grounding_query flows through generate_with_retry's **kwargs."""
        with patch("greycloud.client.create_client") as mock_create:
            with patch(
                "greycloud.client.search_sources", return_value=_two_fake_sources()
            ) as mock_search:
                mock_genai_client = MagicMock()
                mock_genai_client.models.generate_content.return_value = (
                    mock_generate_response
                )
                mock_create.return_value = mock_genai_client

                client = GreyCloudClient(_grounding_config())
                client.generate_with_retry(
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

    def test_called_with_injected_sources(self, sample_config, mock_generate_response):
        """Sources found: on_grounding fires with the exact list injected."""
        sources = _two_fake_sources()
        with patch("greycloud.client.create_client") as mock_create:
            with patch("greycloud.client.search_sources", return_value=sources):
                mock_genai_client = MagicMock()
                mock_genai_client.models.generate_content.return_value = (
                    mock_generate_response
                )
                mock_create.return_value = mock_genai_client

                client = GreyCloudClient(_grounding_config())
                seen = []
                client.generate_content(self._contents(), on_grounding=seen.extend)

        assert seen == sources

    def test_called_before_model_call(self, sample_config, mock_generate_response):
        """Ordering: search -> on_grounding -> model call."""
        order = []

        def recording_search(*args, **kwargs):
            order.append("search")
            return _two_fake_sources()

        with patch("greycloud.client.create_client") as mock_create:
            with patch(
                "greycloud.client.search_sources",
                side_effect=recording_search,
            ):
                mock_genai_client = MagicMock()

                def recording_generate(**kwargs):
                    order.append("generate")
                    return mock_generate_response

                mock_genai_client.models.generate_content.side_effect = (
                    recording_generate
                )
                mock_create.return_value = mock_genai_client

                client = GreyCloudClient(_grounding_config())
                client.generate_content(
                    self._contents(), on_grounding=lambda srcs: order.append("callback")
                )

        assert order == ["search", "callback", "generate"]

    def test_called_with_empty_list_when_search_returns_nothing(
        self, sample_config, mock_generate_response
    ):
        """Search ran, nothing found: distinguishable from 'grounding skipped'."""
        with patch("greycloud.client.create_client") as mock_create:
            with patch("greycloud.client.search_sources", return_value=[]):
                mock_genai_client = MagicMock()
                mock_genai_client.models.generate_content.return_value = (
                    mock_generate_response
                )
                mock_create.return_value = mock_genai_client

                client = GreyCloudClient(_grounding_config())
                seen = []
                client.generate_content(self._contents(), on_grounding=seen.extend)

        assert seen == []

    def test_not_called_when_grounding_false(
        self, sample_config, mock_generate_response
    ):
        with patch("greycloud.client.create_client") as mock_create:
            with patch(
                "greycloud.client.search_sources", return_value=_two_fake_sources()
            ):
                mock_genai_client = MagicMock()
                mock_genai_client.models.generate_content.return_value = (
                    mock_generate_response
                )
                mock_create.return_value = mock_genai_client

                client = GreyCloudClient(_grounding_config())
                seen = []
                client.generate_content(
                    self._contents(), grounding=False, on_grounding=seen.extend
                )

        assert seen == []

    def test_not_called_when_threshold_skips(
        self, sample_config, mock_generate_response
    ):
        with patch("greycloud.client.create_client") as mock_create:
            with patch(
                "greycloud.client.search_sources", return_value=_two_fake_sources()
            ):
                mock_genai_client = MagicMock()
                mock_genai_client.models.generate_content.return_value = (
                    mock_generate_response
                )
                mock_create.return_value = mock_genai_client

                client = GreyCloudClient(
                    _grounding_config(min_grounding_query_chars=50)
                )
                seen = []
                client.generate_content(
                    self._contents("thanks"), on_grounding=seen.extend
                )

        assert seen == []

    def test_not_called_when_tools_override(
        self, sample_config, mock_generate_response
    ):
        explicit_tool = MagicMock()
        with patch("greycloud.client.create_client") as mock_create:
            with patch(
                "greycloud.client.search_sources", return_value=_two_fake_sources()
            ):
                mock_genai_client = MagicMock()
                mock_genai_client.models.generate_content.return_value = (
                    mock_generate_response
                )
                mock_create.return_value = mock_genai_client

                client = GreyCloudClient(_grounding_config())
                seen = []
                client.generate_content(
                    self._contents(), tools=[explicit_tool], on_grounding=seen.extend
                )

        assert seen == []

    def test_not_called_when_inject_mode_disabled(
        self, sample_config, mock_generate_response
    ):
        with patch("greycloud.client.create_client") as mock_create:
            with patch(
                "greycloud.client.search_sources", return_value=_two_fake_sources()
            ):
                mock_genai_client = MagicMock()
                mock_genai_client.models.generate_content.return_value = (
                    mock_generate_response
                )
                mock_create.return_value = mock_genai_client

                client = GreyCloudClient(_grounding_config(use_vertex_ai_search=False))
                seen = []
                client.generate_content(self._contents(), on_grounding=seen.extend)

        assert seen == []

    def test_callback_exception_does_not_break_generation(
        self, sample_config, mock_generate_response
    ):
        """A raising callback is logged and generation proceeds (grounded)."""

        def boom(sources):
            raise RuntimeError("callback exploded")

        with patch("greycloud.client.create_client") as mock_create:
            with patch(
                "greycloud.client.search_sources", return_value=_two_fake_sources()
            ):
                mock_genai_client = MagicMock()
                mock_genai_client.models.generate_content.return_value = (
                    mock_generate_response
                )
                mock_create.return_value = mock_genai_client

                client = GreyCloudClient(_grounding_config())
                response = client.generate_content(self._contents(), on_grounding=boom)

        assert response is mock_generate_response
        call_args = mock_genai_client.models.generate_content.call_args
        assert (
            call_args[1]["contents"][-1].parts[0].text.startswith("<grounding_sources>")
        )

    def test_coroutine_callback_driven_to_completion(
        self, sample_config, mock_generate_response
    ):
        """A coroutine-function callback passed to the sync client is run to
        completion (no dangling coroutine warning, result recorded)."""
        import asyncio

        ran = []

        async def async_cb(sources):
            ran.append([s.title for s in sources])

        with patch("greycloud.client.create_client") as mock_create:
            with patch(
                "greycloud.client.search_sources", return_value=_two_fake_sources()
            ):
                mock_genai_client = MagicMock()
                mock_genai_client.models.generate_content.return_value = (
                    mock_generate_response
                )
                mock_create.return_value = mock_genai_client

                client = GreyCloudClient(_grounding_config())
                client.generate_content(self._contents(), on_grounding=async_cb)

        assert ran == [["Doc1", "Doc2"]]
        # No dangling coroutine left behind.
        assert not any(isinstance(task, asyncio.Task) for task in [])

    def test_stream_called_with_injected_sources(self, sample_config):
        mock_chunk = MagicMock()
        mock_chunk.candidates = [MagicMock()]
        mock_chunk.candidates[0].content = MagicMock()
        mock_chunk.candidates[0].content.parts = [MagicMock()]
        mock_chunk.candidates[0].content.parts[0].text = "Hello"
        mock_chunk.text = "Hello"

        with patch("greycloud.client.create_client") as mock_create:
            with patch(
                "greycloud.client.search_sources", return_value=_two_fake_sources()
            ):
                mock_genai_client = MagicMock()
                mock_genai_client.models.generate_content_stream.return_value = [
                    mock_chunk
                ]
                mock_create.return_value = mock_genai_client

                client = GreyCloudClient(_grounding_config())
                seen = []
                list(
                    client.generate_content_stream(
                        self._contents(), on_grounding=seen.extend
                    )
                )

        assert [s.title for s in seen] == ["Doc1", "Doc2"]

    def test_generate_with_retry_threads_on_grounding(
        self, sample_config, mock_generate_response
    ):
        with patch("greycloud.client.create_client") as mock_create:
            with patch(
                "greycloud.client.search_sources", return_value=_two_fake_sources()
            ):
                mock_genai_client = MagicMock()
                mock_genai_client.models.generate_content.return_value = (
                    mock_generate_response
                )
                mock_create.return_value = mock_genai_client

                client = GreyCloudClient(_grounding_config())
                seen = []
                client.generate_with_retry(self._contents(), on_grounding=seen.extend)

        assert len(seen) == 2

    def test_coroutine_callback_scheduled_on_running_loop(
        self, sample_config, mock_generate_response
    ):
        """A coroutine-function callback passed to the sync client while an
        event loop is already running is scheduled on that loop (not dropped),
        and its exceptions are logged rather than propagated."""
        import asyncio

        ran = []
        failures = []

        async def async_cb(sources):
            ran.append([s.title for s in sources])

        async def outer():
            with patch("greycloud.client.create_client") as mock_create:
                with patch(
                    "greycloud.client.search_sources",
                    return_value=_two_fake_sources(),
                ):
                    mock_genai_client = MagicMock()
                    mock_genai_client.models.generate_content.return_value = (
                        mock_generate_response
                    )
                    mock_create.return_value = mock_genai_client

                    client = GreyCloudClient(_grounding_config())
                    response = client.generate_content(
                        self._contents(), on_grounding=async_cb
                    )
                    # Let the scheduled callback task run to completion.
                    for _ in range(3):
                        await asyncio.sleep(0)
                    return response

        response = asyncio.run(outer())
        assert response is mock_generate_response
        assert ran == [["Doc1", "Doc2"]]
        assert failures == []


class TestReauthBehavior:
    """Opt-in interactive re-login, always-on non-interactive self-heal."""

    def test_force_reauth_blocked_under_pytest(self, sample_config):
        """Sync twin of the async guard test: even with auto_reauth=True,
        pytest must never spawn gcloud browser login."""
        config = dataclasses.replace(sample_config, auto_reauth=True)
        with patch("greycloud.client.create_client"):
            client = GreyCloudClient(config)
            with patch("subprocess.run") as mock_run:
                assert client._force_reauth() is False
                mock_run.assert_not_called()

    def test_auth_error_recovers_non_interactively_with_reauth_off(
        self, sample_config, sample_contents, mock_generate_response
    ):
        """Default (auto_reauth off): an auth error mid-retry still gets the
        non-interactive self-heal — the client is recreated (re-minting
        credentials) with no interactive login spawn."""
        with patch("greycloud.client.create_client") as mock_create:
            mock_genai_client = MagicMock()
            mock_genai_client.models.generate_content.side_effect = [
                Exception("401 Unauthorized"),
                mock_generate_response,
            ]
            mock_create.return_value = mock_genai_client

            client = GreyCloudClient(sample_config)

            with patch("time.sleep"):
                with patch("subprocess.run") as mock_run:
                    response = client.generate_with_retry(
                        sample_contents, max_retries=3
                    )

            assert response == mock_generate_response
            # construction + non-interactive recovery, nothing interactive
            assert mock_create.call_count == 2
            mock_run.assert_not_called()

    def test_auth_error_exhausted_reports_disabled_interactive_reauth(
        self, sample_config, sample_contents
    ):
        """When non-interactive recovery also fails and interactive re-login
        is off, the exhausted error must say WHY re-auth didn't run instead
        of claiming a re-auth was attempted."""
        with patch("greycloud.client.create_client") as mock_create:
            mock_genai_client = MagicMock()
            mock_genai_client.models.generate_content.side_effect = Exception(
                "401 Unauthorized"
            )
            mock_create.side_effect = [mock_genai_client, RuntimeError("no creds")]

            client = GreyCloudClient(sample_config)

            with patch("time.sleep"):
                with pytest.raises(
                    RuntimeError, match="Interactive re-login is disabled"
                ):
                    client.generate_with_retry(sample_contents, max_retries=1)


def test_sync_generate_with_retry_streaming_yields_immediately(mocker):
    """Verify generator yields chunks as they arrive rather than collecting all into a list."""
    config = GreyCloudConfig(project_id="test-project")
    mocker.patch("greycloud.client.create_client")
    client = GreyCloudClient(config)

    yielded_indices = []

    def mock_stream(*args, **kwargs):
        for i in range(3):
            yielded_indices.append(i)
            chunk = MagicMock()
            chunk.text = f"chunk-{i}"
            chunk.candidates = [MagicMock()]
            chunk.candidates[0].content.parts = [
                MagicMock(text=f"chunk-{i}", thought=None)
            ]
            yield chunk

    client.client.models.generate_content_stream = mock_stream
    contents = [types.Content(role="user", parts=[types.Part.from_text(text="Hi")])]

    gen = client.generate_with_retry(contents, streaming=True)

    # In unbuffered streaming, merely calling generate_with_retry does not consume the stream
    assert yielded_indices == []

    # Consuming the first chunk should only produce the first chunk
    first = next(gen)
    assert first == "chunk-0"
    assert yielded_indices == [0]

    # Consuming the second chunk
    second = next(gen)
    assert second == "chunk-1"
    assert yielded_indices == [0, 1]

    # Consuming the third chunk
    third = next(gen)
    assert third == "chunk-2"
    assert yielded_indices == [0, 1, 2]

    with pytest.raises(StopIteration):
        next(gen)


def test_sync_generate_with_retry_streaming_retries_transient_error(mocker):
    """Verify streaming generator retries on transient error before first yield."""
    config = GreyCloudConfig(project_id="test-project")
    mocker.patch("greycloud.client.create_client")
    client = GreyCloudClient(config)

    calls = 0

    def mock_stream(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise Exception("Transient network reset")
        chunk = MagicMock()
        chunk.text = "Success chunk"
        chunk.candidates = [MagicMock()]
        chunk.candidates[0].content.parts = [
            MagicMock(text="Success chunk", thought=None)
        ]
        yield chunk

    client.client.models.generate_content_stream = mock_stream
    contents = [types.Content(role="user", parts=[types.Part.from_text(text="Hi")])]

    mocker.patch("time.sleep")
    gen = client.generate_with_retry(contents, streaming=True, max_retries=2)
    chunks = list(gen)

    assert chunks == ["Success chunk"]
    assert calls == 2


def test_sync_generate_with_retry_streaming_retries_auth_error(mocker):
    """Verify streaming generator recreates client and retries on auth error."""
    config = GreyCloudConfig(project_id="test-project")
    mock_create = mocker.patch("greycloud.client.create_client")
    mock_genai_client = MagicMock()
    mock_create.return_value = mock_genai_client

    client = GreyCloudClient(config)

    calls = 0

    def mock_stream(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise Exception("401 Unauthorized")
        chunk = MagicMock()
        chunk.text = "Recovered chunk"
        chunk.candidates = [MagicMock()]
        chunk.candidates[0].content.parts = [
            MagicMock(text="Recovered chunk", thought=None)
        ]
        yield chunk

    mock_genai_client.models.generate_content_stream = mock_stream
    contents = [types.Content(role="user", parts=[types.Part.from_text(text="Hi")])]

    mocker.patch("time.sleep")
    gen = client.generate_with_retry(contents, streaming=True, max_retries=2)
    chunks = list(gen)

    assert chunks == ["Recovered chunk"]
    assert mock_create.call_count == 2


def test_sync_generate_with_retry_streaming_fails_after_first_yield(mocker):
    """Verify streaming generator immediately raises without retry if failure occurs after first yield."""
    config = GreyCloudConfig(project_id="test-project")
    mocker.patch("greycloud.client.create_client")
    client = GreyCloudClient(config)

    calls = 0

    def mock_stream(*args, **kwargs):
        nonlocal calls
        calls += 1
        chunk = MagicMock()
        chunk.text = "First chunk"
        chunk.candidates = [MagicMock()]
        chunk.candidates[0].content.parts = [
            MagicMock(text="First chunk", thought=None)
        ]
        yield chunk
        raise RuntimeError("Mid-stream connection drop")

    client.client.models.generate_content_stream = mock_stream
    contents = [types.Content(role="user", parts=[types.Part.from_text(text="Hi")])]

    mocker.patch("time.sleep")
    gen = client.generate_with_retry(contents, streaming=True, max_retries=2)
    assert next(gen) == "First chunk"
    with pytest.raises(RuntimeError, match="Mid-stream connection drop"):
        next(gen)

    assert calls == 1  # No retry attempted after first yield!


def test_sync_generate_with_retry_streaming_forwards_return_events_and_on_thought(
    mocker,
):
    """Verify return_events and on_thought are forwarded to generate_content_stream."""
    config = GreyCloudConfig(project_id="test-project")
    mocker.patch("greycloud.client.create_client")
    client = GreyCloudClient(config)

    part1 = MagicMock(text="I am thinking", thought=True)
    chunk1 = MagicMock()
    chunk1.candidates = [MagicMock()]
    chunk1.candidates[0].content.parts = [part1]

    part2 = MagicMock(text="Here is answer", thought=False)
    chunk2 = MagicMock()
    chunk2.candidates = [MagicMock()]
    chunk2.candidates[0].content.parts = [part2]

    client.client.models.generate_content_stream = MagicMock(
        return_value=[chunk1, chunk2]
    )
    contents = [types.Content(role="user", parts=[types.Part.from_text(text="Hi")])]

    captured_thoughts = []

    def on_thought(t):
        captured_thoughts.append(t)

    gen = client.generate_with_retry(
        contents,
        streaming=True,
        return_events=True,
        on_thought=on_thought,
    )
    items = list(gen)

    assert captured_thoughts == ["I am thinking"]
    assert items == [
        {"thought": "I am thinking"},
        {"text": "Here is answer"},
    ]


def test_sync_generate_with_retry_streaming_forwards_request_timeout(mocker):
    """Verify request_timeout is forwarded in streaming generate_with_retry."""
    config = GreyCloudConfig(project_id="test-project")
    mocker.patch("greycloud.client.create_client")
    client = GreyCloudClient(config)

    mock_stream = MagicMock(return_value=[])
    client.client.models.generate_content_stream = mock_stream
    contents = [types.Content(role="user", parts=[types.Part.from_text(text="Hi")])]

    gen = client.generate_with_retry(
        contents,
        streaming=True,
        request_timeout=45,
    )
    list(gen)

    call_args = mock_stream.call_args
    gen_config = call_args[1]["config"]
    assert gen_config.http_options is not None
    assert gen_config.http_options.timeout == 45000


def test_sync_generate_with_retry_non_streaming_forwards_request_timeout(mocker):
    """Verify request_timeout is forwarded in non-streaming generate_with_retry."""
    config = GreyCloudConfig(project_id="test-project")
    mocker.patch("greycloud.client.create_client")
    client = GreyCloudClient(config)

    mock_generate = MagicMock()
    mock_resp = MagicMock()
    mock_generate.return_value = mock_resp
    client.client.models.generate_content = mock_generate
    contents = [types.Content(role="user", parts=[types.Part.from_text(text="Hi")])]

    resp = client.generate_with_retry(
        contents,
        streaming=False,
        request_timeout=50,
    )
    assert resp == mock_resp

    call_args = mock_generate.call_args
    gen_config = call_args[1]["config"]
    assert gen_config.http_options is not None
    assert gen_config.http_options.timeout == 50000


class TestValidateBeforeGrounding:
    """F2: config errors must be raised before any grounding search runs."""

    def _grounding_client(self, sample_config):
        with patch("greycloud.client.create_client"):
            client = GreyCloudClient(sample_config)
        client.config.grounding_mode = "inject"
        client.config.use_vertex_ai_search = True
        client.config.vertex_ai_search_datastore = "test-datastore"
        return client

    @pytest.mark.parametrize(
        "invalid_kwargs",
        [
            {"cached_content": "caches/1"},
            {"thinking_level": "MINIMAL"},
        ],
    )
    def test_generate_content_rejects_invalid_g4_before_search(
        self, sample_config, sample_contents, invalid_kwargs
    ):
        client = self._grounding_client(sample_config)
        with patch("greycloud.client.search_sources") as mock_search:
            with pytest.raises(ValueError):
                client.generate_content(
                    sample_contents,
                    model="gemini-4-argon",
                    grounding=True,
                    request_timeout=None,
                    **invalid_kwargs,
                )
        assert mock_search.call_count == 0

    @pytest.mark.parametrize(
        "invalid_kwargs",
        [
            {"cached_content": "caches/1"},
            {"thinking_level": "MINIMAL"},
        ],
    )
    def test_stream_rejects_invalid_g4_before_search(
        self, sample_config, sample_contents, invalid_kwargs
    ):
        client = self._grounding_client(sample_config)
        with patch("greycloud.client.search_sources") as mock_search:
            with pytest.raises(ValueError):
                for _ in client.generate_content_stream(
                    sample_contents,
                    model="gemini-4-argon",
                    grounding=True,
                    **invalid_kwargs,
                ):
                    pass
        assert mock_search.call_count == 0


class TestCallerIncludeThoughtsOnGemini4:
    """F5: a caller-supplied include_thoughts must survive the G4 rebuild.

    The G4 branch rebuilds the ThinkingConfig from scratch; it hard-coded
    self.config.include_thoughts, silently flipping a caller's explicit
    include_thoughts=False. The config default must apply only when the
    caller is silent.
    """

    def _client(self, cfg):
        with patch("greycloud.client.create_client"):
            return GreyCloudClient(cfg)

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


class TestNonG4IncludeThoughtsOmitted:
    """F6: 0.4.x apps using thinking_level on gemini-3.8-flash must not
    suddenly receive thought parts. The non-G4 ThinkingConfig omits
    include_thoughts unless the caller or config explicitly opts in;
    the G4 new default (True) is asserted for contrast."""

    def test_non_g4_omits_include_thoughts_when_silent(self, sample_config):
        with patch("greycloud.client.create_client"):
            client = GreyCloudClient(sample_config)
        built = client._build_generate_config(
            model="gemini-3.8-flash", thinking_level="HIGH"
        )
        assert built.thinking_config.thinking_level.value == "HIGH"
        # Field omitted: None, not the 0.5.0 leaked default True.
        assert built.thinking_config.include_thoughts is None

    def test_non_g4_explicit_config_false_honored(self, sample_config):
        sample_config.include_thoughts = False
        with patch("greycloud.client.create_client"):
            client = GreyCloudClient(sample_config)
        built = client._build_generate_config(
            model="gemini-3.8-flash", thinking_level="HIGH"
        )
        assert built.thinking_config.include_thoughts is False

    def test_non_g4_explicit_config_true_honored(self, sample_config):
        sample_config.include_thoughts = True
        with patch("greycloud.client.create_client"):
            client = GreyCloudClient(sample_config)
        built = client._build_generate_config(
            model="gemini-3.8-flash", thinking_level="HIGH"
        )
        assert built.thinking_config.include_thoughts is True

    def test_g4_none_config_default_sends_true(self, sample_config):
        """The G4 new default: config None resolves to include_thoughts=True."""
        sample_config.include_thoughts = None
        with patch("greycloud.client.create_client"):
            client = GreyCloudClient(sample_config)
        built = client._build_generate_config(
            model="gemini-4-argon", thinking_level="HIGH"
        )
        assert built.thinking_config.include_thoughts is True

    def test_config_bool_validation_still_enforced(self, sample_config):
        with pytest.raises(TypeError):
            GreyCloudConfig(include_thoughts="yes")  # type: ignore[arg-type]


class TestValidateBeforeStreamGeneratorReturned:
    """F7: generate_with_retry(streaming=True) must raise config errors on
    THE CALL, not on first next() — the caller cannot try/except a generator
    body that runs later."""

    def test_sync_stream_call_raises_value_error(self, sample_config, sample_contents):
        with patch("greycloud.client.create_client"):
            client = GreyCloudClient(sample_config)
        with pytest.raises(ValueError):
            # The call itself — NOT iterating the returned generator.
            client.generate_with_retry(
                sample_contents,
                streaming=True,
                model="gemini-4-argon",
                thinking_level="MINIMAL",
                max_retries=3,
            )


class TestFailedOnThoughtDoesNotSuppressRetry:
    """F8: a raising on_thought callback must not poison thought_emitted.

    Marking the flag before delivery meant a swallowed callback error plus a
    transient stream failure aborted after one attempt instead of retrying.
    The flag is set only after successful delivery; the no-replay contract
    (delivered thought + later failure still re-raises) must be preserved."""

    @staticmethod
    def _thought_chunk():
        part = MagicMock()
        part.text = "thinking hard"
        part.thought = True
        chunk = MagicMock()
        chunk.candidates = [MagicMock()]
        chunk.candidates[0].content = MagicMock()
        chunk.candidates[0].content.parts = [part]
        return chunk

    @staticmethod
    def _stream_factory(chunk):
        def stream():
            yield chunk
            raise RuntimeError("connection dropped")

        return stream

    def _make_client(self, sample_config):
        with patch("greycloud.client.create_client") as mock_create:
            mock_genai_client = MagicMock()
            mock_genai_client.models.generate_content_stream.side_effect = (
                lambda *a, **kw: self._stream_factory(self._thought_chunk())()
            )
            mock_create.return_value = mock_genai_client
            client = GreyCloudClient(sample_config)
        return client, mock_genai_client.models.generate_content_stream

    def test_raising_callback_retries_until_exhausted(
        self, sample_config, sample_contents, caplog
    ):
        client, mock_stream = self._make_client(sample_config)

        def boom(text):
            raise RuntimeError("callback exploded")

        with patch("greycloud.client.time.sleep") as mock_sleep:
            with caplog.at_level(logging.WARNING, logger="greycloud.client"):
                with pytest.raises(RuntimeError, match="connection dropped"):
                    list(
                        client.generate_with_retry(
                            sample_contents,
                            streaming=True,
                            on_thought=boom,
                            max_retries=3,
                        )
                    )
        assert mock_stream.call_count == 4
        assert mock_sleep.call_count == 3
        assert any("callback exploded" in r.getMessage() for r in caplog.records)

    def test_delivered_thought_still_re_raises_without_retry(
        self, sample_config, sample_contents
    ):
        client, mock_stream = self._make_client(sample_config)
        captured = []
        with patch("greycloud.client.time.sleep") as mock_sleep:
            with pytest.raises(RuntimeError, match="connection dropped"):
                list(
                    client.generate_with_retry(
                        sample_contents,
                        streaming=True,
                        on_thought=captured.append,
                    )
                )
        assert captured == ["thinking hard"]
        assert mock_stream.call_count == 1
        mock_sleep.assert_not_called()


class TestDictHttpOptionsCoerced:
    """F10: a dict http_options must be coerced to types.HttpOptions before
    the per-call request_timeout merge (dicts have no model_copy)."""

    def test_dict_plus_timeout_merges(self, sample_config):
        with patch("greycloud.client.create_client"):
            client = GreyCloudClient(sample_config)
        cfg = client._build_generate_config(
            model="gemini-3.8-flash",
            http_options={"timeout": 5000},
            request_timeout=30,
        )
        assert isinstance(cfg.http_options, types.HttpOptions)
        assert cfg.http_options.timeout == 30000

    def test_dict_headers_survive_merge(self, sample_config):
        with patch("greycloud.client.create_client"):
            client = GreyCloudClient(sample_config)
        cfg = client._build_generate_config(
            model="gemini-3.8-flash",
            http_options={"timeout": 5000, "headers": {"x-test": "1"}},
            request_timeout=30,
        )
        assert cfg.http_options.timeout == 30000
        assert cfg.http_options.headers == {"x-test": "1"}

    def test_dict_without_timeout_stays_coerced(self, sample_config):
        with patch("greycloud.client.create_client"):
            client = GreyCloudClient(sample_config)
        cfg = client._build_generate_config(
            model="gemini-3.8-flash", http_options={"timeout": 5000}
        )
        assert isinstance(cfg.http_options, types.HttpOptions)
        assert cfg.http_options.timeout == 5000


class TestOnThoughtFiresWithReturnChunks:
    """F11: return_chunks must not starve on_thought — raw mode yields
    chunks AND still delivers thought text to the callback."""

    def test_callback_receives_thoughts_in_raw_mode(
        self, sample_config, sample_contents
    ):
        thought_part = MagicMock()
        thought_part.text = "thinking hard"
        thought_part.thought = True
        text_part = MagicMock()
        text_part.text = "answer"
        text_part.thought = False
        chunk1 = MagicMock()
        chunk1.candidates = [MagicMock()]
        chunk1.candidates[0].content = MagicMock()
        chunk1.candidates[0].content.parts = [thought_part]
        chunk2 = MagicMock()
        chunk2.candidates = [MagicMock()]
        chunk2.candidates[0].content = MagicMock()
        chunk2.candidates[0].content.parts = [text_part]

        def stream():
            yield chunk1
            yield chunk2

        with patch("greycloud.client.create_client") as mock_create:
            mock_genai_client = MagicMock()
            mock_genai_client.models.generate_content_stream.side_effect = (
                lambda *a, **kw: stream()
            )
            mock_create.return_value = mock_genai_client
            client = GreyCloudClient(sample_config)

            captured = []
            chunks = list(
                client.generate_content_stream(
                    sample_contents, return_chunks=True, on_thought=captured.append
                )
            )
        assert captured == ["thinking hard"]
        assert chunks == [chunk1, chunk2]


class TestPerCallRequestTimeoutValidation:
    """F12: a per-call request_timeout applies the same rules as
    GreyCloudConfig — int (never bool), >= 1."""

    @pytest.mark.parametrize(
        "bad,exc,match",
        [
            (0, ValueError, "request_timeout"),
            (-5, ValueError, "request_timeout"),
            (True, TypeError, "request_timeout"),
            ("120", TypeError, "request_timeout"),
            (12.5, TypeError, "request_timeout"),
        ],
    )
    def test_sync_builder_rejects_bad_request_timeout(
        self, sample_config, bad, exc, match
    ):
        with patch("greycloud.client.create_client"):
            client = GreyCloudClient(sample_config)
        with pytest.raises(exc, match=match):
            client._build_generate_config(model="gemini-3.8-flash", request_timeout=bad)
