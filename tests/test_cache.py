"""Tests for GreyCloudCache"""

import datetime
import pytest
from unittest.mock import Mock, MagicMock, patch
from google.genai import types

from greycloud.cache import (
    GreyCloudCache,
    get_min_tokens_for_model,
    MIN_TOKENS_BY_MODEL,
    DEFAULT_MIN_TOKENS,
)
from greycloud.config import GreyCloudConfig


class TestGetMinTokensForModel:
    """Tests for get_min_tokens_for_model helper"""

    def test_gemini_2_5_flash(self):
        """Gemini 2.5 Flash requires 1024 tokens"""
        assert get_min_tokens_for_model("gemini-2.5-flash") == 1024
        assert get_min_tokens_for_model("gemini-2.5-flash-preview") == 1024

    def test_gemini_2_5_pro(self):
        """Gemini 2.5 Pro requires 4096 tokens"""
        assert get_min_tokens_for_model("gemini-2.5-pro") == 4096
        assert get_min_tokens_for_model("gemini-2.5-pro-preview") == 4096

    def test_gemini_3_flash(self):
        """Gemini 3 Flash requires 1024 tokens"""
        assert get_min_tokens_for_model("gemini-3-flash") == 1024
        assert get_min_tokens_for_model("gemini-3-flash-preview") == 1024

    def test_gemini_3_pro(self):
        """Gemini 3 Pro requires 4096 tokens"""
        assert get_min_tokens_for_model("gemini-3-pro") == 4096
        assert get_min_tokens_for_model("gemini-3-pro-preview") == 4096

    def test_gemini_2_0(self):
        """Gemini 2.0 requires 2048 tokens"""
        assert get_min_tokens_for_model("gemini-2.0-flash") == 2048

    def test_unknown_model(self):
        """Unknown model returns default"""
        assert get_min_tokens_for_model("unknown-model") == DEFAULT_MIN_TOKENS

    def test_case_insensitive(self):
        """Model name matching is case insensitive"""
        assert get_min_tokens_for_model("GEMINI-2.5-FLASH") == 1024
        assert get_min_tokens_for_model("Gemini-3-Pro") == 4096


class TestGreyCloudCacheInit:
    """Tests for GreyCloudCache initialization"""

    def test_init_with_config(self, sample_config):
        """Initialize with provided config"""
        cache_client = GreyCloudCache(sample_config)
        assert cache_client.config == sample_config
        assert cache_client._client is None

    def test_init_without_config(self):
        """Initialize without config creates default"""
        import subprocess
        with patch.object(subprocess, "run") as mock_run:
            mock_run.return_value.returncode = 0
            mock_run.return_value.stdout = "test-project-id"
            cache_client = GreyCloudCache()
            assert cache_client.config is not None
            assert cache_client.config.project_id == "test-project-id"


class TestGreyCloudCacheClient:
    """Tests for client property"""

    def test_client_creation(self, sample_config, mock_genai_client, mock_credentials):
        """Client is created on first access"""
        cache_client = GreyCloudCache(sample_config)

        with patch("greycloud.cache.create_client") as mock_create:
            mock_create.return_value = mock_genai_client
            client = cache_client.client
            assert client == mock_genai_client
            mock_create.assert_called_once()

    def test_client_cached(self, sample_config, mock_genai_client):
        """Client is cached after first access"""
        cache_client = GreyCloudCache(sample_config)

        with patch("greycloud.cache.create_client") as mock_create:
            mock_create.return_value = mock_genai_client
            client1 = cache_client.client
            client2 = cache_client.client
            assert client1 is client2
            mock_create.assert_called_once()


class TestGreyCloudCacheContentsToTypes:
    """Tests for _contents_to_types helper"""

    def test_content_objects_passthrough(self, sample_config):
        """Content objects pass through unchanged"""
        cache_client = GreyCloudCache(sample_config)
        content = types.Content(
            role="user",
            parts=[types.Part.from_text(text="Hello")]
        )
        result = cache_client._contents_to_types([content])
        assert len(result) == 1
        assert result[0] == content

    def test_dict_with_text(self, sample_config):
        """Dict with text is converted"""
        cache_client = GreyCloudCache(sample_config)
        content_dict = {
            "role": "user",
            "parts": [{"text": "Hello"}]
        }
        result = cache_client._contents_to_types([content_dict])
        assert len(result) == 1
        assert result[0].role == "user"
        assert len(result[0].parts) == 1

    def test_dict_with_string_part(self, sample_config):
        """Dict with string part is converted"""
        cache_client = GreyCloudCache(sample_config)
        content_dict = {
            "role": "user",
            "parts": ["Hello world"]
        }
        result = cache_client._contents_to_types([content_dict])
        assert len(result) == 1
        assert result[0].role == "user"

    def test_dict_with_file_data(self, sample_config):
        """Dict with file_data is converted"""
        cache_client = GreyCloudCache(sample_config)
        content_dict = {
            "role": "user",
            "parts": [{
                "file_data": {
                    "file_uri": "gs://bucket/file.pdf",
                    "mime_type": "application/pdf"
                }
            }]
        }
        result = cache_client._contents_to_types([content_dict])
        assert len(result) == 1


class TestGreyCloudCacheCreateCache:
    """Tests for create_cache method"""

    def test_create_cache_basic(self, sample_config, mock_genai_client):
        """Create cache with basic parameters"""
        cache_client = GreyCloudCache(sample_config)

        mock_cache = MagicMock(spec=types.CachedContent)
        mock_cache.name = "cachedContents/test-cache"
        mock_genai_client.caches.create.return_value = mock_cache

        with patch("greycloud.cache.create_client") as mock_create:
            mock_create.return_value = mock_genai_client

            contents = [{"role": "user", "parts": [{"text": "Large document..."}]}]
            cache = cache_client.create_cache(
                contents=contents,
                display_name="test-cache",
                ttl_seconds=3600,
            )

            assert cache.name == "cachedContents/test-cache"
            mock_genai_client.caches.create.assert_called_once()

    def test_create_cache_with_system_instruction(self, sample_config, mock_genai_client):
        """Create cache with system instruction"""
        cache_client = GreyCloudCache(sample_config)

        mock_cache = MagicMock(spec=types.CachedContent)
        mock_cache.name = "cachedContents/test-cache"
        mock_genai_client.caches.create.return_value = mock_cache

        with patch("greycloud.cache.create_client") as mock_create:
            mock_create.return_value = mock_genai_client

            contents = [{"role": "user", "parts": [{"text": "Large document..."}]}]
            cache = cache_client.create_cache(
                contents=contents,
                system_instruction="You are a helpful assistant.",
                ttl_seconds=7200,
            )

            assert cache.name == "cachedContents/test-cache"
            call_kwargs = mock_genai_client.caches.create.call_args
            assert call_kwargs is not None

    def test_create_cache_with_model_override(self, sample_config, mock_genai_client):
        """Create cache with model override"""
        cache_client = GreyCloudCache(sample_config)

        mock_cache = MagicMock(spec=types.CachedContent)
        mock_genai_client.caches.create.return_value = mock_cache

        with patch("greycloud.cache.create_client") as mock_create:
            mock_create.return_value = mock_genai_client

            contents = [{"role": "user", "parts": [{"text": "Large document..."}]}]
            cache_client.create_cache(
                contents=contents,
                model="gemini-2.5-pro",
            )

            call_kwargs = mock_genai_client.caches.create.call_args
            assert call_kwargs[1]["model"] == "gemini-2.5-pro"


class TestGreyCloudCacheCreateCacheFromText:
    """Tests for create_cache_from_text convenience method"""

    def test_create_cache_from_text(self, sample_config, mock_genai_client):
        """Create cache from plain text"""
        cache_client = GreyCloudCache(sample_config)

        mock_cache = MagicMock(spec=types.CachedContent)
        mock_cache.name = "cachedContents/text-cache"
        mock_genai_client.caches.create.return_value = mock_cache

        with patch("greycloud.cache.create_client") as mock_create:
            mock_create.return_value = mock_genai_client

            cache = cache_client.create_cache_from_text(
                text="This is a large document that needs to be cached.",
                display_name="text-cache",
                ttl_seconds=1800,
            )

            assert cache.name == "cachedContents/text-cache"
            mock_genai_client.caches.create.assert_called_once()


class TestGreyCloudCacheCreateCacheFromFiles:
    """Tests for create_cache_from_files method"""

    def test_create_cache_from_files(self, sample_config, mock_genai_client):
        """Create cache from GCS file URIs"""
        cache_client = GreyCloudCache(sample_config)

        mock_cache = MagicMock(spec=types.CachedContent)
        mock_cache.name = "cachedContents/files-cache"
        mock_genai_client.caches.create.return_value = mock_cache

        with patch("greycloud.cache.create_client") as mock_create:
            mock_create.return_value = mock_genai_client

            cache = cache_client.create_cache_from_files(
                file_uris=[
                    "gs://bucket/doc1.pdf",
                    "gs://bucket/doc2.txt",
                ],
                display_name="files-cache",
            )

            assert cache.name == "cachedContents/files-cache"
            mock_genai_client.caches.create.assert_called_once()

    def test_create_cache_from_files_with_mime_types(self, sample_config, mock_genai_client):
        """Create cache from files with explicit MIME types"""
        cache_client = GreyCloudCache(sample_config)

        mock_cache = MagicMock(spec=types.CachedContent)
        mock_genai_client.caches.create.return_value = mock_cache

        with patch("greycloud.cache.create_client") as mock_create:
            mock_create.return_value = mock_genai_client

            cache_client.create_cache_from_files(
                file_uris=["gs://bucket/doc.bin"],
                mime_types=["application/octet-stream"],
            )

            mock_genai_client.caches.create.assert_called_once()


class TestGreyCloudCacheListCaches:
    """Tests for list_caches method"""

    def test_list_caches(self, sample_config, mock_genai_client):
        """List all caches"""
        cache_client = GreyCloudCache(sample_config)

        mock_caches = [
            MagicMock(spec=types.CachedContent, name="cache1"),
            MagicMock(spec=types.CachedContent, name="cache2"),
        ]
        mock_genai_client.caches.list.return_value = iter(mock_caches)

        with patch("greycloud.cache.create_client") as mock_create:
            mock_create.return_value = mock_genai_client

            caches = list(cache_client.list_caches())

            assert len(caches) == 2
            mock_genai_client.caches.list.assert_called_once()


class TestGreyCloudCacheGetCache:
    """Tests for get_cache method"""

    def test_get_cache(self, sample_config, mock_genai_client):
        """Get a specific cache by name"""
        cache_client = GreyCloudCache(sample_config)

        mock_cache = MagicMock(spec=types.CachedContent)
        mock_cache.name = "cachedContents/test-cache"
        mock_cache.display_name = "Test Cache"
        mock_genai_client.caches.get.return_value = mock_cache

        with patch("greycloud.cache.create_client") as mock_create:
            mock_create.return_value = mock_genai_client

            cache = cache_client.get_cache("cachedContents/test-cache")

            assert cache.name == "cachedContents/test-cache"
            mock_genai_client.caches.get.assert_called_once_with(
                name="cachedContents/test-cache"
            )


class TestGreyCloudCacheUpdateCacheTTL:
    """Tests for update_cache_ttl method"""

    def test_update_ttl_seconds(self, sample_config, mock_genai_client):
        """Update cache TTL with seconds"""
        cache_client = GreyCloudCache(sample_config)

        mock_cache = MagicMock(spec=types.CachedContent)
        mock_genai_client.caches.update.return_value = mock_cache

        with patch("greycloud.cache.create_client") as mock_create:
            mock_create.return_value = mock_genai_client

            cache_client.update_cache_ttl(
                name="cachedContents/test-cache",
                ttl_seconds=7200,
            )

            mock_genai_client.caches.update.assert_called_once()
            call_kwargs = mock_genai_client.caches.update.call_args
            assert call_kwargs[1]["name"] == "cachedContents/test-cache"

    def test_update_expire_time(self, sample_config, mock_genai_client):
        """Update cache with specific expire time"""
        cache_client = GreyCloudCache(sample_config)

        mock_cache = MagicMock(spec=types.CachedContent)
        mock_genai_client.caches.update.return_value = mock_cache

        with patch("greycloud.cache.create_client") as mock_create:
            mock_create.return_value = mock_genai_client

            expire_at = datetime.datetime.now(datetime.timezone.utc) + datetime.timedelta(hours=4)
            cache_client.update_cache_ttl(
                name="cachedContents/test-cache",
                expire_time=expire_at,
            )

            mock_genai_client.caches.update.assert_called_once()

    def test_update_both_raises_error(self, sample_config):
        """Providing both ttl_seconds and expire_time raises ValueError"""
        cache_client = GreyCloudCache(sample_config)

        with pytest.raises(ValueError, match="Provide either ttl_seconds or expire_time"):
            cache_client.update_cache_ttl(
                name="cachedContents/test-cache",
                ttl_seconds=3600,
                expire_time=datetime.datetime.now(datetime.timezone.utc),
            )

    def test_update_neither_raises_error(self, sample_config):
        """Providing neither ttl_seconds nor expire_time raises ValueError"""
        cache_client = GreyCloudCache(sample_config)

        with pytest.raises(ValueError, match="Must provide either ttl_seconds or expire_time"):
            cache_client.update_cache_ttl(name="cachedContents/test-cache")


class TestGreyCloudCacheDeleteCache:
    """Tests for delete_cache method"""

    def test_delete_cache(self, sample_config, mock_genai_client):
        """Delete a cache"""
        cache_client = GreyCloudCache(sample_config)

        with patch("greycloud.cache.create_client") as mock_create:
            mock_create.return_value = mock_genai_client

            cache_client.delete_cache("cachedContents/test-cache")

            mock_genai_client.caches.delete.assert_called_once_with(
                name="cachedContents/test-cache"
            )


class TestGreyCloudCacheDeleteAllCaches:
    """Tests for delete_all_caches method"""

    def test_delete_all_caches(self, sample_config, mock_genai_client):
        """Delete all caches"""
        cache_client = GreyCloudCache(sample_config)

        mock_cache1 = MagicMock(spec=types.CachedContent)
        mock_cache1.name = "cachedContents/cache1"
        mock_cache1.display_name = "Cache 1"

        mock_cache2 = MagicMock(spec=types.CachedContent)
        mock_cache2.name = "cachedContents/cache2"
        mock_cache2.display_name = "Cache 2"

        mock_genai_client.caches.list.return_value = iter([mock_cache1, mock_cache2])

        with patch("greycloud.cache.create_client") as mock_create:
            mock_create.return_value = mock_genai_client

            count = cache_client.delete_all_caches()

            assert count == 2
            assert mock_genai_client.caches.delete.call_count == 2

    def test_delete_all_caches_with_filter(self, sample_config, mock_genai_client):
        """Delete caches with display name filter"""
        cache_client = GreyCloudCache(sample_config)

        mock_cache1 = MagicMock(spec=types.CachedContent)
        mock_cache1.name = "cachedContents/cache1"
        mock_cache1.display_name = "test-cache"

        mock_cache2 = MagicMock(spec=types.CachedContent)
        mock_cache2.name = "cachedContents/cache2"
        mock_cache2.display_name = "other-cache"

        mock_genai_client.caches.list.return_value = iter([mock_cache1, mock_cache2])

        with patch("greycloud.cache.create_client") as mock_create:
            mock_create.return_value = mock_genai_client

            count = cache_client.delete_all_caches(display_name_filter="test-cache")

            assert count == 1
            mock_genai_client.caches.delete.assert_called_once_with(
                name="cachedContents/cache1"
            )


class TestGreyCloudCacheGenerateWithCache:
    """Tests for generate_with_cache method"""

    def test_generate_with_cache_string_prompt(self, sample_config, mock_genai_client, mock_generate_response):
        """Generate with cache using string prompt"""
        cache_client = GreyCloudCache(sample_config)

        mock_genai_client.models.generate_content.return_value = mock_generate_response

        with patch("greycloud.cache.create_client") as mock_create:
            mock_create.return_value = mock_genai_client

            response = cache_client.generate_with_cache(
                cache_name="cachedContents/test-cache",
                prompt="Summarize the document",
            )

            assert response.text == "This is a test response"
            mock_genai_client.models.generate_content.assert_called_once()

    def test_generate_with_cache_contents_prompt(self, sample_config, mock_genai_client, mock_generate_response):
        """Generate with cache using Content list prompt"""
        cache_client = GreyCloudCache(sample_config)

        mock_genai_client.models.generate_content.return_value = mock_generate_response

        with patch("greycloud.cache.create_client") as mock_create:
            mock_create.return_value = mock_genai_client

            contents = [{"role": "user", "parts": [{"text": "Summarize"}]}]
            response = cache_client.generate_with_cache(
                cache_name="cachedContents/test-cache",
                prompt=contents,
            )

            assert response.text == "This is a test response"

    def test_generate_with_cache_parameters(self, sample_config, mock_genai_client, mock_generate_response):
        """Generate with cache using custom parameters"""
        cache_client = GreyCloudCache(sample_config)

        mock_genai_client.models.generate_content.return_value = mock_generate_response

        with patch("greycloud.cache.create_client") as mock_create:
            mock_create.return_value = mock_genai_client

            cache_client.generate_with_cache(
                cache_name="cachedContents/test-cache",
                prompt="Summarize",
                temperature=0.5,
                top_p=0.9,
                max_output_tokens=1024,
            )

            mock_genai_client.models.generate_content.assert_called_once()


class TestGreyCloudCacheGenerateWithCacheStream:
    """Tests for generate_with_cache_stream method"""

    def test_generate_with_cache_stream(self, sample_config, mock_genai_client):
        """Generate with cache (streaming)"""
        cache_client = GreyCloudCache(sample_config)

        # Create mock chunks
        mock_chunk1 = MagicMock()
        mock_chunk1.candidates = [MagicMock()]
        mock_chunk1.candidates[0].content = MagicMock()
        mock_chunk1.candidates[0].content.parts = [MagicMock()]
        mock_chunk1.text = "Hello"

        mock_chunk2 = MagicMock()
        mock_chunk2.candidates = [MagicMock()]
        mock_chunk2.candidates[0].content = MagicMock()
        mock_chunk2.candidates[0].content.parts = [MagicMock()]
        mock_chunk2.text = " World"

        mock_genai_client.models.generate_content_stream.return_value = iter([mock_chunk1, mock_chunk2])

        with patch("greycloud.cache.create_client") as mock_create:
            mock_create.return_value = mock_genai_client

            chunks = list(cache_client.generate_with_cache_stream(
                cache_name="cachedContents/test-cache",
                prompt="Summarize",
            ))

            assert chunks == ["Hello", " World"]


class TestGreyCloudCacheGetCacheInfo:
    """Tests for get_cache_info method"""

    def test_get_cache_info(self, sample_config):
        """Get human-readable cache info"""
        cache_client = GreyCloudCache(sample_config)

        mock_cache = MagicMock(spec=types.CachedContent)
        mock_cache.name = "cachedContents/test-cache"
        mock_cache.display_name = "Test Cache"
        mock_cache.model = "gemini-3-flash-preview"
        mock_cache.create_time = datetime.datetime(2024, 1, 1, 12, 0, 0)
        mock_cache.expire_time = datetime.datetime(2024, 1, 1, 13, 0, 0)

        mock_usage = MagicMock()
        mock_usage.total_token_count = 50000
        mock_cache.usage_metadata = mock_usage

        info = cache_client.get_cache_info(mock_cache)

        assert info["name"] == "cachedContents/test-cache"
        assert info["display_name"] == "Test Cache"
        assert info["model"] == "gemini-3-flash-preview"
        assert info["total_token_count"] == 50000


class TestGreyCloudCacheIntegration:
    """Integration-style tests for GreyCloudCache"""

    def test_full_cache_lifecycle(self, sample_config, mock_genai_client, mock_generate_response):
        """Test full cache lifecycle: create, use, delete"""
        cache_client = GreyCloudCache(sample_config)

        # Setup mocks
        mock_cache = MagicMock(spec=types.CachedContent)
        mock_cache.name = "cachedContents/lifecycle-test"
        mock_cache.display_name = "Lifecycle Test"

        mock_usage = MagicMock()
        mock_usage.total_token_count = 10000
        mock_cache.usage_metadata = mock_usage

        mock_genai_client.caches.create.return_value = mock_cache
        mock_genai_client.models.generate_content.return_value = mock_generate_response

        with patch("greycloud.cache.create_client") as mock_create:
            mock_create.return_value = mock_genai_client

            # Create cache
            cache = cache_client.create_cache_from_text(
                text="A" * 10000,  # Large text
                display_name="Lifecycle Test",
                ttl_seconds=3600,
            )
            assert cache.name == "cachedContents/lifecycle-test"

            # Use cache multiple times
            for _ in range(3):
                response = cache_client.generate_with_cache(
                    cache_name=cache.name,
                    prompt="Summarize this content",
                )
                assert response.text == "This is a test response"

            # Delete cache
            cache_client.delete_cache(cache.name)
            mock_genai_client.caches.delete.assert_called_once_with(
                name="cachedContents/lifecycle-test"
            )
