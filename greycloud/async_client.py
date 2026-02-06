"""
Async GreyCloud client for rate-limited content generation.

Uses the google-genai async surface (client.aio) with VertexRateLimiter
to enforce RPM, TPM, and concurrency limits.
"""

import asyncio
import random
from typing import List, Optional, Dict, Any

from google import genai
from google.genai import types

from .config import GreyCloudConfig
from .auth import create_client
from .rate_limiter import VertexRateLimiter


class GreyCloudAsyncClient:
    """Async client for rate-limited Vertex AI / GenAI content generation."""

    def __init__(
        self,
        config: Optional[GreyCloudConfig] = None,
        rpm: int = 60,
        tpm: int = 250_000,
        max_concurrency: int = 10,
    ):
        self.config = config or GreyCloudConfig()
        self.rate_limiter = VertexRateLimiter(
            rpm=rpm, tpm=tpm, max_concurrency=max_concurrency
        )
        self._client: genai.Client = create_client(
            project_id=self.config.project_id,
            location=self.config.location,
            sa_email=self.config.sa_email,
            use_api_key=self.config.use_api_key,
            api_key_file=self.config.api_key_file,
            endpoint=self.config.endpoint,
            api_version=self.config.api_version,
            auto_reauth=self.config.auto_reauth,
        )

    def _estimate_prompt_tokens(self, contents: List[types.Content]) -> int:
        """Estimate token count from contents for rate limiter."""
        total_chars = 0
        for content in contents:
            if hasattr(content, "parts") and content.parts:
                for part in content.parts:
                    if hasattr(part, "text") and part.text:
                        total_chars += len(part.text)
        return max(total_chars // 4, 1)

    def _build_generate_config(
        self,
        system_instruction: Optional[str] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        max_output_tokens: Optional[int] = None,
        cached_content: Optional[str] = None,
        **kwargs,
    ) -> types.GenerateContentConfig:
        """Build GenerateContentConfig from parameters and config defaults."""
        config_dict: Dict[str, Any] = {
            "temperature": (
                temperature if temperature is not None else self.config.temperature
            ),
            "top_p": top_p if top_p is not None else self.config.top_p,
            "max_output_tokens": (
                max_output_tokens
                if max_output_tokens is not None
                else self.config.max_output_tokens
            ),
        }
        if system_instruction or self.config.system_instruction:
            si = system_instruction or self.config.system_instruction
            config_dict["system_instruction"] = [types.Part.from_text(text=si)]
        if self.config.seed is not None:
            config_dict["seed"] = self.config.seed
        if self.config.thinking_level:
            config_dict["thinking_config"] = types.ThinkingConfig(
                thinking_level=self.config.thinking_level
            )
        if cached_content:
            config_dict["cached_content"] = cached_content
        config_dict.update(kwargs)
        return types.GenerateContentConfig(**config_dict)

    async def generate_content(
        self,
        contents: List[types.Content],
        model: Optional[str] = None,
        system_instruction: Optional[str] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        max_output_tokens: Optional[int] = None,
        cached_content: Optional[str] = None,
        **kwargs,
    ) -> types.GenerateContentResponse:
        """Generate content with rate limiting."""
        model_name = model or self.config.model
        config = self._build_generate_config(
            system_instruction=system_instruction,
            temperature=temperature,
            top_p=top_p,
            max_output_tokens=max_output_tokens,
            cached_content=cached_content,
            **kwargs,
        )
        token_est = self._estimate_prompt_tokens(contents)
        return await self.rate_limiter.call_with_limits(
            token_est,
            self._client.aio.models.generate_content(
                model=model_name, contents=contents, config=config
            ),
        )

    async def count_tokens(
        self,
        contents: List[types.Content],
        system_instruction: Optional[str] = None,
        model: Optional[str] = None,
    ) -> int:
        """Count tokens, falling back to character estimate on failure."""
        model_name = model or self.config.model
        try:
            config = None
            if system_instruction:
                config = types.CountTokensConfig(
                    system_instruction=[types.Part.from_text(text=system_instruction)]
                )
            response = await self._client.aio.models.count_tokens(
                model=model_name, contents=contents, config=config
            )
            return response.total_tokens
        except Exception:
            return self._estimate_prompt_tokens(contents)

    async def generate_with_retry(
        self,
        contents: List[types.Content],
        max_retries: int = 5,
        base_delay: float = 2.0,
        max_delay: float = 60.0,
        **generate_kwargs,
    ) -> types.GenerateContentResponse:
        """Generate content with exponential backoff retry."""
        model_name = generate_kwargs.pop("model", None) or self.config.model
        config = self._build_generate_config(**generate_kwargs)
        token_est = self._estimate_prompt_tokens(contents)

        for attempt in range(max_retries + 1):
            try:
                return await self.rate_limiter.call_with_limits(
                    token_est,
                    self._client.aio.models.generate_content(
                        model=model_name, contents=contents, config=config
                    ),
                )
            except Exception:
                if attempt >= max_retries:
                    raise
                delay = min(base_delay * (2**attempt), max_delay)
                jitter = random.uniform(0, delay * 0.1)
                await asyncio.sleep(delay + jitter)
