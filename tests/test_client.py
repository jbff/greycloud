"""Unit tests for GreyCloudClient"""

import pytest
from unittest.mock import patch, MagicMock
from google.genai import types
from greycloud.client import GreyCloudClient
from greycloud.config import GreyCloudConfig


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
        """Test tool building with Vertex AI Search"""
        config = GreyCloudConfig(
            project_id="test-project",
            use_vertex_ai_search=True,
            vertex_ai_search_datastore="projects/test/locations/us/datastores/test-ds",
        )

        with patch("greycloud.client.create_client") as mock_create:
            mock_create.return_value = MagicMock()
            client = GreyCloudClient(config)
            tools = client._build_tools()

            assert len(tools) == 1
            assert tools[0].retrieval is not None

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
