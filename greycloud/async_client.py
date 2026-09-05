"""
Async GreyCloud client for rate-limited content generation.

Uses the google-genai async surface (client.aio) with VertexRateLimiter
to enforce RPM, TPM, and concurrency limits.
"""

import asyncio
import logging
import random
import time
import subprocess
import sys
import os
from typing import Callable, List, Optional, Dict, Any, AsyncGenerator, Union

from google import genai
from google.genai import types

from .config import GreyCloudConfig
from .auth import create_client
from .rate_limiter import VertexRateLimiter
from .grounding import (
    GroundingSource,
    asearch_sources,
    build_grounding_context,
    _ainvoke_grounding_callback,
)

logger = logging.getLogger(__name__)


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

    @property
    def client(self) -> genai.Client:
        """Underlying genai.Client for advanced use. Prefer this client's methods for generation so rate limits are applied."""
        return self._client

    def _force_reauth(self) -> bool:
        """
        Force re-authentication by calling gcloud auth application-default login

        Returns:
            bool: True if re-authentication succeeded, False otherwise
        """
        if self.config.use_api_key:
            return False

        if not self.config.auto_reauth:
            return False

        is_interactive = (
            sys.stdin.isatty() if hasattr(sys.stdin, "isatty") else False
        ) or os.environ.get("DISPLAY") is not None

        try:
            cmd = ["gcloud", "auth", "application-default", "login"]
            if not is_interactive:
                cmd.append("--no-browser")
            # Note: We do NOT use --quiet and always use capture_output=False
            # so the user can see the gcloud URL/prompts and complete authentication

            result = subprocess.run(
                cmd,
                check=True,
                capture_output=False,
                text=True,
                timeout=300,
            )
            return result.returncode == 0
        except subprocess.TimeoutExpired:
            return is_interactive
        except subprocess.CalledProcessError as e:
            error_output = ""
            if hasattr(e, "stderr") and e.stderr:
                error_output = e.stderr.lower()
            if hasattr(e, "stdout") and e.stdout:
                error_output += " " + e.stdout.lower()

            if "already has" in error_output or "already authenticated" in error_output:
                return True
            return False
        except FileNotFoundError:
            return False
        except Exception:
            return False

    def _is_authentication_error(self, error: Exception) -> bool:
        """
        Check if an error is related to authentication/authorization

        Checks for:
        - HTTP status codes 401 (Unauthorized) and 403 (Forbidden)
        - Google Auth exceptions (RefreshError, etc.)
        - Google API Core exceptions (Unauthenticated, PermissionDenied)
        - Error messages containing authentication-related keywords
        """
        try:
            from google.auth.exceptions import RefreshError, DefaultCredentialsError

            if isinstance(error, (RefreshError, DefaultCredentialsError)):
                return True
        except ImportError:
            error_type_name = type(error).__name__
            if error_type_name in ("RefreshError", "DefaultCredentialsError"):
                error_module = type(error).__module__
                if "google.auth" in error_module or "google.oauth2" in error_module:
                    return True

        try:
            from google.api_core.exceptions import Unauthenticated, PermissionDenied

            if isinstance(error, (Unauthenticated, PermissionDenied)):
                return True
        except ImportError:
            pass

        current_error = error
        error_chain = [current_error]
        while hasattr(current_error, "__cause__") and current_error.__cause__:
            current_error = current_error.__cause__
            error_chain.append(current_error)
        while hasattr(current_error, "__context__") and current_error.__context__:
            current_error = current_error.__context__
            error_chain.append(current_error)

        for err in error_chain:
            error_type_name = type(err).__name__
            if (
                "RefreshError" in error_type_name
                or "DefaultCredentialsError" in error_type_name
            ):
                return True
            if (
                "Unauthenticated" in error_type_name
                or "PermissionDenied" in error_type_name
            ):
                return True

        for err in error_chain:
            error_str = str(err).lower()
            error_repr = repr(err).lower()
            error_type = type(err).__name__.lower()
            combined_error = f"{error_str} {error_repr} {error_type}"

            auth_keywords = [
                "401",
                "unauthorized",
                "403",
                "forbidden",
                "authentication",
                "credential",
                "token expired",
                "token invalid",
                "invalid token",
                "expired token",
                "expired",
                "unauthenticated",
                "permission denied",
                "reauth",
                "reauthentication",
                "application-default",
                "gcloud auth application-default login",
                "invalid_grant",
                "invalid_credentials",
                "access_denied",
                "insufficient_permission",
            ]
            if any(keyword in combined_error for keyword in auth_keywords):
                return True

        return False

    def _estimate_prompt_tokens(self, contents: List[types.Content]) -> int:
        """Estimate token count from contents for rate limiter."""
        total_chars = 0
        for content in contents:
            if hasattr(content, "parts") and content.parts:
                for part in content.parts:
                    if hasattr(part, "text") and part.text:
                        total_chars += len(part.text)
        return max(total_chars // 4, 1)

    def _build_tools(self) -> List[types.Tool]:
        """Build tools list based on configuration.

        The ``tools.retrieval.vertexAiSearch`` tool is only included in
        ``grounding_mode == "tool"`` (the legacy escape hatch). In the default
        ``"inject"`` mode the retrieval tool is dropped: grounding is performed
        by an explicit Discovery Engine search whose results are injected into
        the prompt (see ``_apply_grounding_async``).
        """
        tools = []
        if (
            self.config.use_vertex_ai_search
            and self.config.vertex_ai_search_datastore
            and self.config.grounding_mode == "tool"
        ):
            tools.append(
                types.Tool(
                    retrieval=types.Retrieval(
                        vertex_ai_search=types.VertexAISearch(
                            datastore=self.config.vertex_ai_search_datastore
                        )
                    )
                )
            )
        return tools

    @staticmethod
    def _last_user_query(
        contents: List[types.Content],
    ):
        """Return (index, query) of the last user message, or (None, "").

        The query is the concatenated text of the last ``role == "user"``
        Content in the list. Returns ``(None, "")`` when there is no user
        content, in which case grounding must be skipped.
        """
        for i in range(len(contents) - 1, -1, -1):
            content = contents[i]
            if str(getattr(content, "role", None) or "").lower() == "user":
                parts = getattr(content, "parts", None) or []
                text = "".join(
                    part.text for part in parts if getattr(part, "text", None)
                )
                return i, text.strip()
        return None, ""

    async def _apply_grounding_async(
        self,
        contents: List[types.Content],
        tools: Optional[List[types.Tool]] = None,
        retry_unquoted: bool = True,
        grounding_query: Optional[str] = None,
        grounding: bool = True,
        on_grounding: Optional[Callable[[List[GroundingSource]], None]] = None,
    ) -> List[types.Content]:
        """Return the contents to send, injecting Discovery Engine grounding.

        Only active when the config enables vertex-ai-search grounding in
        ``"inject"`` mode and the caller did not pass an explicit ``tools``
        override. Runs the search BEFORE ``GenerateContentConfig`` is built, so
        the injected text is part of ``contents`` for both the API call and
        ``_estimate_prompt_tokens``.

        The caller's ``contents`` list and Content objects are never mutated:
        on injection a shallow-copied list with a copied last user message
        (grounding block prepended as a text part) is returned. A failed or
        empty search degrades to ungrounded generation (``asearch_sources``
        logs the failure and never raises: WARNING for ordinary failures,
        ERROR for the chunking-config 400 when extractive_content_spec is on).

        Args:
            retry_unquoted: Passed through to :func:`asearch_sources`; see its
                docstring for the quoted-query fallback behavior.
            grounding_query: When a non-blank string, used verbatim as the
                Discovery Engine query in place of the last user message's
                text (the last user turn is often a workflow instruction,
                not searchable content). The injection still prepends the
                block to the last user message.
            grounding: When False, suppress grounding entirely for this call:
                no search, no injection.
            on_grounding: Invoked once per generate, after the search decision
                and before the model call, with the exact source list being
                injected — or ``[]`` when the search ran but returned nothing.
                Not invoked when grounding was skipped
                entirely (``grounding=False``, threshold skip, tools override,
                inject mode disabled, no user content). Coroutine-function
                callbacks are awaited; plain functions are called inline.
                Callback exceptions are logged at WARNING and never propagate.
        """
        if tools is not None:
            # Explicit tools override: honor it as today, no injection.
            return contents
        if not grounding:
            # Per-call skip: caller asked for no grounding on this request.
            return contents
        if not (
            self.config.use_vertex_ai_search
            and self.config.grounding_mode == "inject"
            and self.config.vertex_ai_search_datastore
        ):
            return contents

        override = grounding_query.strip() if isinstance(grounding_query, str) else ""
        last_user_index, query = self._last_user_query(contents)
        if last_user_index is None:
            # No user content: nothing to search on, nowhere to inject.
            return contents
        if override:
            query = override
        if not query:
            # Empty user text (and no override): nothing to search on.
            return contents

        # Config-level skip guard: conversational turns ("thanks", "ok,
        # standby") below the threshold should not fire a search at all.
        # Applies to the effective query, so a caller-supplied grounding_query
        # is gated by it too.
        min_chars = getattr(self.config, "min_grounding_query_chars", 0)
        if isinstance(min_chars, int) and min_chars > 0 and len(query) < min_chars:
            logger.info(
                "grounding_mode=inject, query shorter than "
                "min_grounding_query_chars=%d; skipping search",
                min_chars,
            )
            return contents

        sources = await asearch_sources(
            self.config, query, retry_unquoted=retry_unquoted
        )
        if on_grounding is not None:
            # Fire the callback with the decided list — [] included — before
            # the model call (proposal §5). Callback errors never propagate.
            await _ainvoke_grounding_callback(on_grounding, sources)
        if not sources:
            # Search failures are logged inside asearch_sources (WARNING; ERROR for
            # the chunking-config 400 when extractive_content_spec is on); an
            # empty result set is the genuine "no matches" case.
            logger.info(
                "grounding_mode=inject, search returned no sources; "
                "proceeding ungrounded"
            )
            return contents

        block = build_grounding_context(sources)
        logger.info("grounding_mode=inject, injected %d source(s)", len(sources))
        new_contents = list(contents)
        original_parts = list(new_contents[last_user_index].parts or [])
        new_contents[last_user_index] = types.Content(
            role="user",
            parts=[types.Part.from_text(text=block)] + original_parts,
        )
        return new_contents

    def _build_generate_config(
        self,
        system_instruction: Optional[str] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        max_output_tokens: Optional[int] = None,
        safety_settings: Optional[List[Dict[str, Any]]] = None,
        tools: Optional[List[types.Tool]] = None,
        thinking_level: Optional[str] = None,
        cached_content: Optional[str] = None,
        **kwargs,
    ) -> types.GenerateContentConfig:
        """Build GenerateContentConfig from parameters and config defaults."""
        final_temperature = (
            temperature if temperature is not None else self.config.temperature
        )
        final_top_p = top_p if top_p is not None else self.config.top_p
        final_max_output_tokens = (
            max_output_tokens
            if max_output_tokens is not None
            else self.config.max_output_tokens
        )
        final_safety_settings = (
            safety_settings
            if safety_settings is not None
            else self.config.safety_settings
        )
        final_system_instruction = (
            system_instruction
            if system_instruction is not None
            else self.config.system_instruction
        )
        final_tools = tools if tools is not None else self._build_tools()
        final_thinking_level = (
            thinking_level if thinking_level is not None else self.config.thinking_level
        )

        safety_settings_list = None
        if final_safety_settings is not None:
            safety_settings_list = []
            for setting in final_safety_settings:
                if isinstance(setting, dict):
                    safety_settings_list.append(
                        types.SafetySetting(
                            category=setting["category"],
                            threshold=setting["threshold"],
                        )
                    )
                else:
                    safety_settings_list.append(setting)

        config_dict: Dict[str, Any] = {
            "temperature": final_temperature,
            "top_p": final_top_p,
            "max_output_tokens": final_max_output_tokens,
            "tools": final_tools,
        }
        if safety_settings_list is not None:
            config_dict["safety_settings"] = safety_settings_list
        if self.config.seed is not None:
            config_dict["seed"] = self.config.seed
        if final_system_instruction:
            config_dict["system_instruction"] = [
                types.Part.from_text(text=final_system_instruction)
            ]
        if final_thinking_level:
            config_dict["thinking_config"] = types.ThinkingConfig(
                thinking_level=final_thinking_level
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
        safety_settings: Optional[List[Dict[str, Any]]] = None,
        tools: Optional[List[types.Tool]] = None,
        thinking_level: Optional[str] = None,
        cached_content: Optional[str] = None,
        retry_unquoted: bool = True,
        grounding_query: Optional[str] = None,
        grounding: bool = True,
        on_grounding: Optional[Callable[[List[GroundingSource]], None]] = None,
        **kwargs,
    ) -> types.GenerateContentResponse:
        """Generate content with rate limiting.

        Args:
            retry_unquoted: Passed through to the grounding search; see
                :func:`greycloud.grounding.asearch_sources`.
            grounding_query: Optional explicit Discovery Engine query used in
                place of the last user message's verbatim text; the grounding
                block is still injected into the last user message. Ignored
                (and current behavior preserved) when omitted.
            grounding: When False, skip grounding entirely for this call (no
                search, no injection). Defaults to True.
            on_grounding: Optional callback invoked once per generate with the
                exact list of :class:`greycloud.grounding.GroundingSource`
                being injected (``[]`` when the search ran but found nothing);
                not invoked when grounding is skipped entirely. Coroutine
                callbacks are awaited; callback exceptions are logged at
                WARNING and never propagate.
        """
        model_name = model or self.config.model
        contents = await self._apply_grounding_async(
            contents,
            tools,
            retry_unquoted=retry_unquoted,
            grounding_query=grounding_query,
            grounding=grounding,
            on_grounding=on_grounding,
        )
        config = self._build_generate_config(
            system_instruction=system_instruction,
            temperature=temperature,
            top_p=top_p,
            max_output_tokens=max_output_tokens,
            safety_settings=safety_settings,
            tools=tools,
            thinking_level=thinking_level,
            cached_content=cached_content,
            **kwargs,
        )
        # Token estimate accounts for any injected grounding block.
        token_est = self._estimate_prompt_tokens(contents)
        return await self.rate_limiter.call_with_limits(
            token_est,
            self._client.aio.models.generate_content(
                model=model_name, contents=contents, config=config
            ),
        )

    async def generate_content_stream(
        self,
        contents: List[types.Content],
        model: Optional[str] = None,
        system_instruction: Optional[str] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        max_output_tokens: Optional[int] = None,
        safety_settings: Optional[List[Dict[str, Any]]] = None,
        tools: Optional[List[types.Tool]] = None,
        thinking_level: Optional[str] = None,
        cached_content: Optional[str] = None,
        return_chunks: bool = False,
        retry_unquoted: bool = True,
        grounding_query: Optional[str] = None,
        grounding: bool = True,
        on_grounding: Optional[Callable[[List[GroundingSource]], None]] = None,
        **kwargs,
    ) -> AsyncGenerator[Union[str, types.GenerateContentResponse], None]:
        """Generate content (streaming). Yields text chunks or raw response objects. Rate-limited.

        Args:
            retry_unquoted: Passed through to the grounding search; see
                :func:`greycloud.grounding.asearch_sources`.
            grounding_query: Optional explicit Discovery Engine query used in
                place of the last user message's verbatim text; the grounding
                block is still injected into the last user message. Ignored
                (and current behavior preserved) when omitted.
            grounding: When False, skip grounding entirely for this call (no
                search, no injection). Defaults to True.
            on_grounding: Optional callback invoked once per generate with the
                exact list of :class:`greycloud.grounding.GroundingSource`
                being injected (``[]`` when the search ran but found nothing);
                not invoked when grounding is skipped entirely. Coroutine
                callbacks are awaited; callback exceptions are logged at
                WARNING and never propagate. Fires when the async generator is
                first advanced.
        """
        model_name = model or self.config.model
        # Async generator: the grounding search + injection run when the
        # generator is first advanced, before the first yield.
        contents = await self._apply_grounding_async(
            contents,
            tools,
            retry_unquoted=retry_unquoted,
            grounding_query=grounding_query,
            grounding=grounding,
            on_grounding=on_grounding,
        )
        config = self._build_generate_config(
            system_instruction=system_instruction,
            temperature=temperature,
            top_p=top_p,
            max_output_tokens=max_output_tokens,
            safety_settings=safety_settings,
            tools=tools,
            thinking_level=thinking_level,
            cached_content=cached_content,
            **kwargs,
        )
        # Token estimate accounts for any injected grounding block.
        token_est = self._estimate_prompt_tokens(contents)

        async def _start_stream():
            return await self._client.aio.models.generate_content_stream(
                model=model_name, contents=contents, config=config
            )

        stream = await self.rate_limiter.call_with_limits(token_est, _start_stream())
        async for chunk in stream:
            if return_chunks:
                yield chunk
            else:
                if (
                    chunk.candidates
                    and chunk.candidates[0].content
                    and chunk.candidates[0].content.parts
                ):
                    chunk_text = chunk.text
                    if chunk_text:
                        yield chunk_text

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
        streaming: bool = False,
        base_delay: float = 2.0,
        max_delay: float = 60.0,
        return_chunks: bool = False,
        **generate_kwargs,
    ) -> Union[
        types.GenerateContentResponse,
        AsyncGenerator[Union[str, types.GenerateContentResponse], None],
    ]:
        """Generate content with exponential backoff retry. If streaming=True, returns an async generator of text chunks or raw response objects."""
        if streaming:
            return self._generate_with_retry_stream(
                contents,
                max_retries,
                base_delay,
                max_delay,
                return_chunks=return_chunks,
                **generate_kwargs,
            )
        # Route through generate_content (mirroring the sync client) so every
        # entry point inherits grounding injection consistently. The Discovery
        # Engine search simply re-runs on a retry attempt, which is acceptable
        # (retries are rare and a search is cheap).
        for attempt in range(max_retries + 1):
            try:
                return await self.generate_content(contents, **generate_kwargs)
            except Exception as e:
                is_auth_error = self._is_authentication_error(e)

                if is_auth_error:
                    if self.config.auto_reauth and not self.config.use_api_key:
                        reauth_success = False
                        try:
                            reauth_success = self._force_reauth()
                            if reauth_success:
                                self._client = create_client(
                                    project_id=self.config.project_id,
                                    location=self.config.location,
                                    sa_email=self.config.sa_email,
                                    use_api_key=self.config.use_api_key,
                                    api_key_file=self.config.api_key_file,
                                    endpoint=self.config.endpoint,
                                    api_version=self.config.api_version,
                                    auto_reauth=False,
                                )
                            else:
                                try:
                                    self._client = create_client(
                                        project_id=self.config.project_id,
                                        location=self.config.location,
                                        sa_email=self.config.sa_email,
                                        use_api_key=self.config.use_api_key,
                                        api_key_file=self.config.api_key_file,
                                        endpoint=self.config.endpoint,
                                        api_version=self.config.api_version,
                                        auto_reauth=False,
                                    )
                                    reauth_success = True
                                except Exception:
                                    pass

                            if reauth_success and attempt < max_retries:
                                time.sleep(1)
                                continue
                            elif not reauth_success and attempt >= max_retries:
                                raise RuntimeError(
                                    f"Authentication error detected and automatic re-authentication failed after {max_retries + 1} attempts. "
                                    "Please run 'gcloud auth application-default login' manually to refresh your credentials. "
                                    f"Original error: {str(e)}"
                                ) from e
                            elif not reauth_success:
                                pass
                        except Exception as auth_error:
                            if attempt >= max_retries:
                                raise RuntimeError(
                                    f"Re-authentication failed after {max_retries + 1} attempts: {str(auth_error)}. "
                                    "Please run 'gcloud auth application-default login' manually. "
                                    f"Original error: {str(e)}"
                                ) from e

                if attempt >= max_retries:
                    raise
                delay = min(base_delay * (2**attempt), max_delay)
                jitter = random.uniform(0, delay * 0.1)
                await asyncio.sleep(delay + jitter)
        raise RuntimeError("Failed to generate content after retries")

    async def _generate_with_retry_stream(
        self,
        contents: List[types.Content],
        max_retries: int,
        base_delay: float,
        max_delay: float,
        return_chunks: bool = False,
        **generate_kwargs,
    ) -> AsyncGenerator[Union[str, types.GenerateContentResponse], None]:
        """Async generator that yields stream chunks with retry on failure."""
        for attempt in range(max_retries + 1):
            try:
                async for chunk in self.generate_content_stream(
                    contents, return_chunks=return_chunks, **generate_kwargs
                ):
                    yield chunk
                return
            except Exception as e:
                is_auth_error = self._is_authentication_error(e)

                if is_auth_error:
                    if self.config.auto_reauth and not self.config.use_api_key:
                        reauth_success = False
                        try:
                            reauth_success = self._force_reauth()
                            if reauth_success:
                                self._client = create_client(
                                    project_id=self.config.project_id,
                                    location=self.config.location,
                                    sa_email=self.config.sa_email,
                                    use_api_key=self.config.use_api_key,
                                    api_key_file=self.config.api_key_file,
                                    endpoint=self.config.endpoint,
                                    api_version=self.config.api_version,
                                    auto_reauth=False,
                                )
                            else:
                                try:
                                    self._client = create_client(
                                        project_id=self.config.project_id,
                                        location=self.config.location,
                                        sa_email=self.config.sa_email,
                                        use_api_key=self.config.use_api_key,
                                        api_key_file=self.config.api_key_file,
                                        endpoint=self.config.endpoint,
                                        api_version=self.config.api_version,
                                        auto_reauth=False,
                                    )
                                    reauth_success = True
                                except Exception:
                                    pass

                            if reauth_success and attempt < max_retries:
                                time.sleep(1)
                                continue
                            elif not reauth_success and attempt >= max_retries:
                                raise RuntimeError(
                                    f"Authentication error detected and automatic re-authentication failed after {max_retries + 1} attempts. "
                                    "Please run 'gcloud auth application-default login' manually to refresh your credentials. "
                                    f"Original error: {str(e)}"
                                ) from e
                            elif not reauth_success:
                                pass
                        except Exception as auth_error:
                            if attempt >= max_retries:
                                raise RuntimeError(
                                    f"Re-authentication failed after {max_retries + 1} attempts: {str(auth_error)}. "
                                    "Please run 'gcloud auth application-default login' manually. "
                                    f"Original error: {str(e)}"
                                ) from e

                if attempt >= max_retries:
                    raise
                delay = min(base_delay * (2**attempt), max_delay)
                jitter = random.uniform(0, delay * 0.1)
                await asyncio.sleep(delay + jitter)
        raise RuntimeError("Failed to generate content after retries")
