"""Unit tests for GreyCloudClient"""

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
            response = client.generate_content(
                sample_contents,
                temperature=0.5,
                max_output_tokens=1000,
                system_instruction="Custom instruction",
            )

            call_args = mock_genai_client.models.generate_content.call_args
            config = call_args[1]["config"]
            assert config.temperature == 0.5
            assert config.max_output_tokens == 1000

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
            count = client.count_tokens(
                sample_contents, system_instruction="System prompt"
            )

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
        execute for real (review finding #14: normalization never ran
        end-to-end through the client).
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
    config-level min_grounding_query_chars threshold (RAG proposal items 1-2).
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
        Discovery Engine query (workflow-instruction turns are not clinical
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
                    grounding_query="ABAS functional impairment adult collateral reports",
                )

        mock_search.assert_called_once()
        assert (
            mock_search.call_args[0][1]
            == "ABAS functional impairment adult collateral reports"
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
                client.generate_content(self._contents("ABAS impairment domains"))

        assert mock_search.call_args[0][1] == "ABAS impairment domains"

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
                    grounding_query="ABAS functional impairment adult collateral reports",
                )

        assert (
            mock_search.call_args[0][1]
            == "ABAS functional impairment adult collateral reports"
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
                    grounding_query="ABAS functional impairment",
                )

        assert mock_search.call_args[0][1] == "ABAS functional impairment"
