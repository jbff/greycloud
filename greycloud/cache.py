"""
Context caching utilities for GreyCloud

Context caching allows you to cache large content (documents, code, etc.) and reuse
it across multiple requests without re-sending the tokens each time. This provides
significant cost savings (75-90% off input token costs) for repeated queries on
the same content.

Key concepts:
- Caches have a TTL (time-to-live) and incur storage costs while active
- Minimum token requirement: 1,024-4,096 tokens depending on model
- Cached content is immutable; only TTL can be updated
- Delete caches when done to stop storage charges
"""

import datetime
from pathlib import Path
from typing import List, Optional, Union, Iterator, Dict, Any
from google import genai
from google.genai import types

from .config import GreyCloudConfig
from .auth import create_client


# Minimum token requirements by model family
MIN_TOKENS_BY_MODEL = {
    "gemini-2.0": 2048,
    "gemini-2.5-flash": 1024,
    "gemini-2.5-pro": 4096,
    "gemini-3-flash": 1024,
    "gemini-3-pro": 4096,
}

# Default minimum if model not recognized
DEFAULT_MIN_TOKENS = 1024


def get_min_tokens_for_model(model: str) -> int:
    """Get minimum token requirement for a model"""
    model_lower = model.lower()
    for prefix, min_tokens in MIN_TOKENS_BY_MODEL.items():
        if prefix in model_lower:
            return min_tokens
    return DEFAULT_MIN_TOKENS


class GreyCloudCache:
    """
    Context caching client for Vertex AI / GenAI

    Provides methods to create, manage, and use cached content for cost-efficient
    repeated queries on the same large content (documents, code, media, etc.).

    Pricing:
    - Cached tokens: 75-90% cheaper than standard input tokens
    - Storage: $1.00 per million tokens per hour (prorated by minute)

    Example:
        from greycloud import GreyCloudConfig, GreyCloudCache

        config = GreyCloudConfig(project_id="my-project")
        cache_client = GreyCloudCache(config)

        # Create a cache from text content
        cache = cache_client.create_cache(
            contents=[{"role": "user", "parts": [{"text": large_document}]}],
            display_name="my-document-cache",
            ttl_seconds=3600,  # 1 hour
        )

        # Use the cache for multiple queries
        response = cache_client.generate_with_cache(cache.name, "Summarize the document")

        # Delete when done to stop storage charges
        cache_client.delete_cache(cache.name)
    """

    def __init__(self, config: Optional[GreyCloudConfig] = None):
        """
        Initialize cache client

        Args:
            config: GreyCloudConfig instance. If None, creates a new one with defaults.
        """
        self.config = config or GreyCloudConfig()
        self._client: Optional[genai.Client] = None

    @property
    def client(self) -> genai.Client:
        """Get the authenticated client"""
        if self._client is None:
            self._client = create_client(
                project_id=self.config.project_id,
                location="global",
                sa_email=self.config.sa_email,
                use_api_key=self.config.use_api_key,
                api_key_file=self.config.api_key_file,
                endpoint=self.config.endpoint,
                api_version=self.config.api_version,
                auto_reauth=self.config.auto_reauth,
            )
        return self._client

    def _contents_to_types(
        self, contents: List[Union[types.Content, Dict[str, Any]]]
    ) -> List[types.Content]:
        """Convert contents to types.Content objects if needed"""
        result = []
        for content in contents:
            if isinstance(content, types.Content):
                result.append(content)
            elif isinstance(content, dict):
                # Convert dict to Content
                role = content.get("role", "user")
                parts = content.get("parts", [])
                typed_parts = []
                for part in parts:
                    if isinstance(part, types.Part):
                        typed_parts.append(part)
                    elif isinstance(part, dict):
                        if "text" in part:
                            typed_parts.append(types.Part.from_text(text=part["text"]))
                        elif "file_data" in part:
                            file_data = part["file_data"]
                            typed_parts.append(
                                types.Part.from_uri(
                                    file_uri=file_data.get(
                                        "file_uri", file_data.get("fileUri")
                                    ),
                                    mime_type=file_data.get(
                                        "mime_type", file_data.get("mimeType")
                                    ),
                                )
                            )
                    elif isinstance(part, str):
                        typed_parts.append(types.Part.from_text(text=part))
                result.append(types.Content(role=role, parts=typed_parts))
            else:
                # Assume it's already a Content-like object
                result.append(content)
        return result

    def create_cache(
        self,
        contents: List[Union[types.Content, Dict[str, Any]]],
        model: Optional[str] = None,
        display_name: Optional[str] = None,
        system_instruction: Optional[str] = None,
        ttl_seconds: int = 3600,
        tools: Optional[List[types.Tool]] = None,
    ) -> types.CachedContent:
        """
        Create cached content

        Args:
            contents: Content to cache (list of Content objects or dicts)
            model: Model name (defaults to config.model)
            display_name: Human-readable name for the cache (max 128 chars)
            system_instruction: System instruction to include in cache
            ttl_seconds: Time-to-live in seconds (default: 3600 = 1 hour)
            tools: Tool definitions to include in cache

        Returns:
            CachedContent object with name, expire_time, and usage_metadata

        Raises:
            ValueError: If content is below minimum token threshold

        Note:
            Minimum token requirements vary by model:
            - Gemini 2.5 Flash / 3 Flash: 1,024 tokens
            - Gemini 2.5 Pro / 3 Pro: 4,096 tokens
            - Gemini 2.0 Flash: 2,048 tokens
        """
        model_name = model or self.config.model
        typed_contents = self._contents_to_types(contents)

        # Build config
        config_dict: Dict[str, Any] = {
            "ttl": f"{ttl_seconds}s",
            "contents": typed_contents,
        }

        if display_name:
            config_dict["display_name"] = display_name

        if system_instruction:
            config_dict["system_instruction"] = system_instruction

        if tools:
            config_dict["tools"] = tools

        cache_config = types.CreateCachedContentConfig(**config_dict)

        cache = self.client.caches.create(
            model=model_name,
            config=cache_config,
        )

        return cache

    def create_cache_from_text(
        self,
        text: str,
        model: Optional[str] = None,
        display_name: Optional[str] = None,
        system_instruction: Optional[str] = None,
        ttl_seconds: int = 3600,
    ) -> types.CachedContent:
        """
        Create cached content from plain text

        Convenience method for caching a single text document.

        Args:
            text: Text content to cache
            model: Model name (defaults to config.model)
            display_name: Human-readable name for the cache
            system_instruction: System instruction to include in cache
            ttl_seconds: Time-to-live in seconds (default: 3600 = 1 hour)

        Returns:
            CachedContent object
        """
        contents = [types.Content(role="user", parts=[types.Part.from_text(text=text)])]
        return self.create_cache(
            contents=contents,
            model=model,
            display_name=display_name,
            system_instruction=system_instruction,
            ttl_seconds=ttl_seconds,
        )

    def create_cache_from_files(
        self,
        file_uris: List[str],
        mime_types: Optional[List[str]] = None,
        model: Optional[str] = None,
        display_name: Optional[str] = None,
        system_instruction: Optional[str] = None,
        ttl_seconds: int = 3600,
    ) -> types.CachedContent:
        """
        Create cached content from GCS file URIs

        Args:
            file_uris: List of GCS URIs (gs://bucket/path)
            mime_types: List of MIME types (auto-detected if not provided)
            model: Model name (defaults to config.model)
            display_name: Human-readable name for the cache
            system_instruction: System instruction to include in cache
            ttl_seconds: Time-to-live in seconds (default: 3600 = 1 hour)

        Returns:
            CachedContent object
        """
        import mimetypes as mt

        parts = []
        for i, uri in enumerate(file_uris):
            # Determine MIME type
            if mime_types and i < len(mime_types):
                mime_type = mime_types[i]
            else:
                # Auto-detect from extension
                guessed_type, _ = mt.guess_type(uri)
                mime_type = guessed_type or "application/octet-stream"

            parts.append(types.Part.from_uri(file_uri=uri, mime_type=mime_type))

        contents = [types.Content(role="user", parts=parts)]

        return self.create_cache(
            contents=contents,
            model=model,
            display_name=display_name,
            system_instruction=system_instruction,
            ttl_seconds=ttl_seconds,
        )

    def list_caches(self) -> Iterator[types.CachedContent]:
        """
        List all cached content

        Yields:
            CachedContent objects
        """
        return self.client.caches.list()

    def get_cache(self, name: str) -> types.CachedContent:
        """
        Get a specific cache by name

        Args:
            name: Cache name (e.g., "cachedContents/abc123")

        Returns:
            CachedContent object
        """
        return self.client.caches.get(name=name)

    def update_cache_ttl(
        self,
        name: str,
        ttl_seconds: Optional[int] = None,
        expire_time: Optional[datetime.datetime] = None,
    ) -> types.CachedContent:
        """
        Update cache TTL (extend expiration)

        Args:
            name: Cache name
            ttl_seconds: New TTL in seconds (from now)
            expire_time: Specific expiration time (timezone-aware datetime)

        Returns:
            Updated CachedContent object

        Note:
            Provide either ttl_seconds or expire_time, not both.
        """
        if ttl_seconds is not None and expire_time is not None:
            raise ValueError("Provide either ttl_seconds or expire_time, not both")

        if ttl_seconds is None and expire_time is None:
            raise ValueError("Must provide either ttl_seconds or expire_time")

        config_dict: Dict[str, Any] = {}
        if ttl_seconds is not None:
            config_dict["ttl"] = f"{ttl_seconds}s"
        if expire_time is not None:
            config_dict["expire_time"] = expire_time

        update_config = types.UpdateCachedContentConfig(**config_dict)

        return self.client.caches.update(
            name=name,
            config=update_config,
        )

    def delete_cache(self, name: str) -> None:
        """
        Delete a cache to stop storage charges

        Args:
            name: Cache name

        Important:
            Always delete caches when done using them to avoid ongoing storage costs.
        """
        self.client.caches.delete(name=name)

    def delete_all_caches(self, display_name_filter: Optional[str] = None) -> int:
        """
        Delete all caches, optionally filtered by display name

        Args:
            display_name_filter: If provided, only delete caches with matching display_name

        Returns:
            Number of caches deleted
        """
        deleted_count = 0
        for cache in self.list_caches():
            if display_name_filter is None or cache.display_name == display_name_filter:
                self.delete_cache(cache.name)
                deleted_count += 1
        return deleted_count

    def generate_with_cache(
        self,
        cache_name: str,
        prompt: Union[str, List[Union[types.Content, Dict[str, Any]]]],
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        max_output_tokens: Optional[int] = None,
        safety_settings: Optional[List[Dict[str, Any]]] = None,
        **kwargs,
    ) -> types.GenerateContentResponse:
        """
        Generate content using a cache

        Convenience method that generates content using cached context.

        Args:
            cache_name: Name of the cache to use
            prompt: Query/prompt (string or list of Content objects)
            model: Model name (must match cache's model)
            temperature: Temperature parameter
            top_p: Top-p parameter
            max_output_tokens: Maximum output tokens
            safety_settings: Safety settings
            **kwargs: Additional GenerateContentConfig parameters

        Returns:
            GenerateContentResponse
        """
        model_name = model or self.config.model

        # Convert prompt to contents
        if isinstance(prompt, str):
            contents = [
                types.Content(role="user", parts=[types.Part.from_text(text=prompt)])
            ]
        else:
            contents = self._contents_to_types(prompt)

        # Build config
        config_dict: Dict[str, Any] = {
            "cached_content": cache_name,
        }

        if temperature is not None:
            config_dict["temperature"] = temperature
        elif self.config.temperature is not None:
            config_dict["temperature"] = self.config.temperature

        if top_p is not None:
            config_dict["top_p"] = top_p
        elif self.config.top_p is not None:
            config_dict["top_p"] = self.config.top_p

        if max_output_tokens is not None:
            config_dict["max_output_tokens"] = max_output_tokens
        elif self.config.max_output_tokens is not None:
            config_dict["max_output_tokens"] = self.config.max_output_tokens

        if safety_settings is not None:
            safety_settings_list = []
            for setting in safety_settings:
                if isinstance(setting, dict):
                    safety_settings_list.append(
                        types.SafetySetting(
                            category=setting["category"],
                            threshold=setting["threshold"],
                        )
                    )
                else:
                    safety_settings_list.append(setting)
            config_dict["safety_settings"] = safety_settings_list

        config_dict.update(kwargs)

        generate_config = types.GenerateContentConfig(**config_dict)

        return self.client.models.generate_content(
            model=model_name,
            contents=contents,
            config=generate_config,
        )

    def generate_with_cache_stream(
        self,
        cache_name: str,
        prompt: Union[str, List[Union[types.Content, Dict[str, Any]]]],
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        max_output_tokens: Optional[int] = None,
        safety_settings: Optional[List[Dict[str, Any]]] = None,
        **kwargs,
    ) -> Iterator[str]:
        """
        Generate content using a cache (streaming)

        Args:
            cache_name: Name of the cache to use
            prompt: Query/prompt (string or list of Content objects)
            model: Model name (must match cache's model)
            temperature: Temperature parameter
            top_p: Top-p parameter
            max_output_tokens: Maximum output tokens
            safety_settings: Safety settings
            **kwargs: Additional GenerateContentConfig parameters

        Yields:
            str: Chunks of response text
        """
        model_name = model or self.config.model

        # Convert prompt to contents
        if isinstance(prompt, str):
            contents = [
                types.Content(role="user", parts=[types.Part.from_text(text=prompt)])
            ]
        else:
            contents = self._contents_to_types(prompt)

        # Build config
        config_dict: Dict[str, Any] = {
            "cached_content": cache_name,
        }

        if temperature is not None:
            config_dict["temperature"] = temperature
        elif self.config.temperature is not None:
            config_dict["temperature"] = self.config.temperature

        if top_p is not None:
            config_dict["top_p"] = top_p
        elif self.config.top_p is not None:
            config_dict["top_p"] = self.config.top_p

        if max_output_tokens is not None:
            config_dict["max_output_tokens"] = max_output_tokens
        elif self.config.max_output_tokens is not None:
            config_dict["max_output_tokens"] = self.config.max_output_tokens

        if safety_settings is not None:
            safety_settings_list = []
            for setting in safety_settings:
                if isinstance(setting, dict):
                    safety_settings_list.append(
                        types.SafetySetting(
                            category=setting["category"],
                            threshold=setting["threshold"],
                        )
                    )
                else:
                    safety_settings_list.append(setting)
            config_dict["safety_settings"] = safety_settings_list

        config_dict.update(kwargs)

        generate_config = types.GenerateContentConfig(**config_dict)

        for chunk in self.client.models.generate_content_stream(
            model=model_name,
            contents=contents,
            config=generate_config,
        ):
            if (
                chunk.candidates
                and chunk.candidates[0].content
                and chunk.candidates[0].content.parts
            ):
                chunk_text = chunk.text
                if chunk_text:
                    yield chunk_text

    def get_cache_info(self, cache: types.CachedContent) -> Dict[str, Any]:
        """
        Get human-readable information about a cache

        Args:
            cache: CachedContent object

        Returns:
            Dict with cache information
        """
        info = {
            "name": cache.name,
            "display_name": getattr(cache, "display_name", None),
            "model": getattr(cache, "model", None),
            "create_time": getattr(cache, "create_time", None),
            "expire_time": getattr(cache, "expire_time", None),
        }

        # Add usage metadata if available
        usage = getattr(cache, "usage_metadata", None)
        if usage:
            info["total_token_count"] = getattr(usage, "total_token_count", None)

        return info
