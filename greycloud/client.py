"""
Main GreyCloud client for content generation and interaction
"""

import time
import random
from typing import List, Optional, Generator, Dict, Any
from google import genai
from google.genai import types

from .config import GreyCloudConfig
from .auth import create_client


class GreyCloudClient:
    """Main client for interacting with Vertex AI/GenAI"""
    
    def __init__(self, config: Optional[GreyCloudConfig] = None):
        """
        Initialize GreyCloud client
        
        Args:
            config: GreyCloudConfig instance. If None, creates a new one with defaults.
        """
        self.config = config or GreyCloudConfig()
        self._client: Optional[genai.Client] = None
        self._authenticate()
    
    def _force_reauth(self) -> bool:
        """
        Force re-authentication by calling gcloud auth application-default login
        
        This will attempt to refresh Google Cloud credentials by running
        `gcloud auth application-default login`. This may require user interaction
        (opening a browser) if credentials are not already cached.
        
        In non-interactive environments (like web servers), this will attempt
        to use the --no-browser flag, but may still require manual intervention.
        
        Returns:
            bool: True if re-authentication succeeded, False otherwise
        """
        if self.config.use_api_key:
            # API key auth doesn't need re-authentication
            return False
        
        if not self.config.auto_reauth:
            return False
        
        import subprocess
        import sys
        import os
        
        # Check if we're in an interactive environment (TTY available)
        # Also check for DISPLAY variable (X11) or if we're in a terminal
        is_interactive = (
            (sys.stdin.isatty() if hasattr(sys.stdin, 'isatty') else False) or
            os.environ.get('DISPLAY') is not None
        )
        
        try:
            # First, try to refresh existing credentials without user interaction
            # This works if the user has already authenticated and just needs a refresh
            try:
                # Try to get a new token - this might refresh automatically
                result = subprocess.run(
                    ["gcloud", "auth", "application-default", "print-access-token"],
                    check=True,
                    capture_output=True,
                    text=True,
                    timeout=5
                )
                # If we can get a token, credentials might be valid
                # But we still want to do a full login to ensure they're refreshed
                if result.returncode == 0:
                    # Credentials exist, but let's still do a login to refresh
                    pass
            except (subprocess.CalledProcessError, subprocess.TimeoutExpired, FileNotFoundError):
                # Can't get token, need to do full login
                pass
            
            # Run gcloud auth application-default login to refresh credentials
            # Use --no-browser flag if not interactive to avoid hanging
            cmd = ["gcloud", "auth", "application-default", "login"]
            if not is_interactive:
                # In non-interactive mode, try to use existing credentials
                # or fail gracefully
                cmd.append("--no-browser")
                # Also add --quiet to avoid prompts
                cmd.append("--quiet")
            
            # For interactive environments, allow user interaction
            # For non-interactive, capture output to avoid hanging
            result = subprocess.run(
                cmd,
                check=True,
                capture_output=not is_interactive,  # Don't capture in interactive mode
                text=True,
                timeout=300 if is_interactive else 30  # Shorter timeout for non-interactive
            )
            return result.returncode == 0
        except subprocess.TimeoutExpired:
            # Timeout occurred - in interactive mode, user might have completed it
            # In non-interactive mode, this is a failure
            return is_interactive
        except subprocess.CalledProcessError as e:
            # Re-authentication failed - check the error message
            error_output = ""
            if hasattr(e, 'stderr') and e.stderr:
                error_output = e.stderr.lower()
            if hasattr(e, 'stdout') and e.stdout:
                error_output += " " + e.stdout.lower()
            
            # Check for specific error conditions
            if "no browser" in error_output or "non-interactive" in error_output:
                # Can't do interactive login in this environment
                # But we might still have valid credentials, so return False
                # and let the caller handle it
                return False
            if "already has" in error_output or "already authenticated" in error_output:
                # Already authenticated - this is actually success
                return True
            # Other errors - return False to indicate failure
            return False
        except FileNotFoundError:
            # gcloud command not found
            return False
        except Exception:
            # Any other error
            return False
    
    def _authenticate(self, force_reauth: bool = False):
        """
        Create authenticated client
        
        Args:
            force_reauth: If True, force re-authentication before creating client
        """
        # If force_reauth is requested, try to refresh credentials first
        if force_reauth:
            self._force_reauth()
        
        try:
            self._client = create_client(
                project_id=self.config.project_id,
                location=self.config.location,
                sa_email=self.config.sa_email,
                use_api_key=self.config.use_api_key,
                api_key_file=self.config.api_key_file,
                endpoint=self.config.endpoint,
                api_version=self.config.api_version,
                auto_reauth=self.config.auto_reauth
            )
        except RuntimeError as e:
            # If authentication fails and auto_reauth is enabled, try automatic re-authentication
            if self.config.auto_reauth and not self.config.use_api_key:
                error_str = str(e).lower()
                if "application-default" in error_str or "reauth" in error_str or "login" in error_str:
                    # Try to automatically run gcloud auth application-default login
                    if self._force_reauth():
                        # Retry creating client after re-authentication
                        self._client = create_client(
                            project_id=self.config.project_id,
                            location=self.config.location,
                            sa_email=self.config.sa_email,
                            use_api_key=self.config.use_api_key,
                            api_key_file=self.config.api_key_file,
                            endpoint=self.config.endpoint,
                            api_version=self.config.api_version,
                            auto_reauth=False  # Don't loop
                        )
                        return
                    else:
                        raise RuntimeError(
                            "Automatic re-authentication failed. Please run 'gcloud auth application-default login' manually."
                        ) from e
            # Re-raise the original error if auto_reauth didn't work or wasn't enabled
            raise
    
    @property
    def client(self) -> genai.Client:
        """Get the authenticated client, re-authenticating if needed"""
        if self._client is None:
            self._authenticate()
        return self._client
    
    def _build_tools(self) -> List[types.Tool]:
        """Build tools list based on configuration"""
        tools = []
        
        if self.config.use_vertex_ai_search and self.config.vertex_ai_search_datastore:
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
    
    def _build_generate_config(
        self,
        system_instruction: Optional[str] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        max_output_tokens: Optional[int] = None,
        safety_settings: Optional[List[Dict[str, Any]]] = None,
        tools: Optional[List[types.Tool]] = None,
        thinking_level: Optional[str] = None,
        **kwargs
    ) -> types.GenerateContentConfig:
        """Build GenerateContentConfig from parameters and config"""
        # Use provided values or fall back to config
        final_temperature = temperature if temperature is not None else self.config.temperature
        final_top_p = top_p if top_p is not None else self.config.top_p
        final_max_output_tokens = max_output_tokens if max_output_tokens is not None else self.config.max_output_tokens
        final_safety_settings = safety_settings if safety_settings is not None else self.config.safety_settings
        final_system_instruction = system_instruction if system_instruction is not None else self.config.system_instruction
        final_tools = tools if tools is not None else self._build_tools()
        final_thinking_level = thinking_level if thinking_level is not None else self.config.thinking_level
        
        # Build safety settings
        # None  -> omit from config, letting Vertex defaults apply.
        # []    -> send an explicit empty list.
        # list  -> convert dicts to SafetySetting as needed.
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
        
        # Build config dict
        config_dict: Dict[str, Any] = {
            "temperature": final_temperature,
            "top_p": final_top_p,
            "max_output_tokens": final_max_output_tokens,
            "tools": final_tools,
        }
        
        if safety_settings_list is not None:
            config_dict["safety_settings"] = safety_settings_list
        
        # Add seed if configured
        if self.config.seed is not None:
            config_dict["seed"] = self.config.seed
        
        # Add system instruction if provided
        if final_system_instruction:
            config_dict["system_instruction"] = [types.Part.from_text(text=final_system_instruction)]
        
        # Add thinking config if configured
        if final_thinking_level:
            config_dict["thinking_config"] = types.ThinkingConfig(thinking_level=final_thinking_level)
        
        # Add any additional kwargs
        config_dict.update(kwargs)
        
        return types.GenerateContentConfig(**config_dict)
    
    def generate_content(
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
        **kwargs
    ) -> types.GenerateContentResponse:
        """
        Generate content (non-streaming)
        
        Args:
            contents: List of Content objects representing conversation history
            model: Model name (defaults to config.model)
            system_instruction: System instruction text
            temperature: Temperature parameter
            top_p: Top-p parameter
            max_output_tokens: Maximum output tokens
            safety_settings: Safety settings override
            tools: Tools override
            thinking_level: Thinking level override
            **kwargs: Additional parameters for GenerateContentConfig
        
        Returns:
            GenerateContentResponse
        """
        model_name = model or self.config.model
        config = self._build_generate_config(
            system_instruction=system_instruction,
            temperature=temperature,
            top_p=top_p,
            max_output_tokens=max_output_tokens,
            safety_settings=safety_settings,
            tools=tools,
            thinking_level=thinking_level,
            **kwargs
        )
        
        return self.client.models.generate_content(
            model=model_name,
            contents=contents,
            config=config
        )
    
    def generate_content_stream(
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
        **kwargs
    ) -> Generator[str, None, None]:
        """
        Generate content (streaming)
        
        Args:
            contents: List of Content objects representing conversation history
            model: Model name (defaults to config.model)
            system_instruction: System instruction text
            temperature: Temperature parameter
            top_p: Top-p parameter
            max_output_tokens: Maximum output tokens
            safety_settings: Safety settings override
            tools: Tools override
            thinking_level: Thinking level override
            **kwargs: Additional parameters for GenerateContentConfig
        
        Yields:
            str: Chunks of response text
        """
        model_name = model or self.config.model
        config = self._build_generate_config(
            system_instruction=system_instruction,
            temperature=temperature,
            top_p=top_p,
            max_output_tokens=max_output_tokens,
            safety_settings=safety_settings,
            tools=tools,
            thinking_level=thinking_level,
            **kwargs
        )
        
        for chunk in self.client.models.generate_content_stream(
            model=model_name,
            contents=contents,
            config=config,
        ):
            if chunk.candidates and chunk.candidates[0].content and chunk.candidates[0].content.parts:
                chunk_text = chunk.text
                if chunk_text:
                    yield chunk_text
    
    def count_tokens(
        self,
        contents: List[types.Content],
        system_instruction: Optional[str] = None,
        model: Optional[str] = None
    ) -> int:
        """
        Count tokens in contents
        
        Args:
            contents: List of Content objects
            system_instruction: System instruction text (optional)
            model: Model name (defaults to config.model)
        
        Returns:
            int: Total token count
        """
        model_name = model or self.config.model
        
        try:
            config = None
            if system_instruction:
                config = types.CountTokensConfig(
                    system_instruction=[types.Part.from_text(text=system_instruction)]
                )
            
            response = self.client.models.count_tokens(
                model=model_name,
                contents=contents,
                config=config
            )
            
            return response.total_tokens
        except Exception:
            # Fallback: estimate tokens (rough approximation: ~4 characters per token)
            total_chars = 0
            for content in contents:
                for part in content.parts:
                    if hasattr(part, 'text') and part.text:
                        total_chars += len(part.text)
            
            if system_instruction:
                total_chars += len(system_instruction)
            
            # Rough estimate: 1 token ≈ 4 characters
            estimated_tokens = total_chars // 4
            return estimated_tokens
    
    def _is_authentication_error(self, error: Exception) -> bool:
        """
        Check if an error is related to authentication/authorization
        
        This method checks for:
        - HTTP status codes 401 (Unauthorized) and 403 (Forbidden)
        - Google Auth exceptions (RefreshError, etc.)
        - Google API Core exceptions (Unauthenticated, PermissionDenied)
        - Error messages containing authentication-related keywords
        """
        # Check for specific Google Auth exceptions
        # Try both direct import and string-based checking for robustness
        try:
            from google.auth.exceptions import RefreshError, DefaultCredentialsError
            if isinstance(error, (RefreshError, DefaultCredentialsError)):
                return True
        except ImportError:
            # If import fails, check by type name
            error_type_name = type(error).__name__
            if error_type_name in ('RefreshError', 'DefaultCredentialsError'):
                # Check if it's from google.auth.exceptions by checking the module
                error_module = type(error).__module__
                if 'google.auth' in error_module or 'google.oauth2' in error_module:
                    return True
        
        # Check for Google API Core exceptions
        try:
            from google.api_core.exceptions import Unauthenticated, PermissionDenied
            if isinstance(error, (Unauthenticated, PermissionDenied)):
                return True
        except ImportError:
            pass
        
        # Check the exception chain (in case it's wrapped)
        current_error = error
        error_chain = [current_error]
        while hasattr(current_error, '__cause__') and current_error.__cause__:
            current_error = current_error.__cause__
            error_chain.append(current_error)
        while hasattr(current_error, '__context__') and current_error.__context__:
            current_error = current_error.__context__
            error_chain.append(current_error)
        
        # Check all errors in the chain
        for err in error_chain:
            # Check exception type name (works even if import fails)
            error_type_name = type(err).__name__
            if 'RefreshError' in error_type_name or 'DefaultCredentialsError' in error_type_name:
                return True
            if 'Unauthenticated' in error_type_name or 'PermissionDenied' in error_type_name:
                return True
        
        # Check error message and string representation for all errors in chain
        for err in error_chain:
            error_str = str(err).lower()
            error_repr = repr(err).lower()
            error_type = type(err).__name__.lower()
            combined_error = f"{error_str} {error_repr} {error_type}"
            
            auth_keywords = [
                "401", "unauthorized",
                "403", "forbidden",
                "authentication",
                "credential",
                "token expired",
                "token invalid",
                "invalid token",
                "expired token",
                "unauthenticated",
                "permission denied",
                "reauth",
                "reauthentication",
                "application-default",
                "gcloud auth application-default login",
                "invalid_grant",  # OAuth error
                "invalid_credentials",
                "access_denied",
                "insufficient_permission",
            ]
            if any(keyword in combined_error for keyword in auth_keywords):
                return True
        
        return False
    
    def exponential_backoff_with_jitter(
        self,
        attempt: int,
        base_delay: int = 2,
        max_delay: int = 60,
        jitter_factor: float = 0.1
    ) -> float:
        """Calculate delay for exponential backoff with jitter"""
        delay = min(base_delay * (2 ** attempt), max_delay)
        jitter = random.uniform(0, delay * jitter_factor)
        return delay + jitter
    
    def generate_with_retry(
        self,
        contents: List[types.Content],
        max_retries: int = 5,
        streaming: bool = False,
        **generate_kwargs
    ) -> Any:
        """
        Generate content with automatic retry and re-authentication
        
        Args:
            contents: List of Content objects
            max_retries: Maximum number of retry attempts
            streaming: If True, use streaming generation
            **generate_kwargs: Additional arguments for generate_content or generate_content_stream
        
        Returns:
            GenerateContentResponse (if streaming=False) or Generator[str] (if streaming=True)
        """
        for attempt in range(max_retries + 1):
            try:
                if streaming:
                    # For streaming, we need to collect all chunks and yield them
                    # But we can't easily retry a generator, so we'll collect first
                    chunks = []
                    for chunk in self.generate_content_stream(contents, **generate_kwargs):
                        chunks.append(chunk)
                    # Return a generator that yields the collected chunks
                    def chunk_generator():
                        for chunk in chunks:
                            yield chunk
                    return chunk_generator()
                else:
                    return self.generate_content(contents, **generate_kwargs)
            except Exception as e:
                # Check if this is an authentication error and re-authenticate if needed
                is_auth_error = self._is_authentication_error(e)
                
                if is_auth_error:
                    if self.config.auto_reauth and not self.config.use_api_key:
                        # Force re-authentication by calling gcloud auth application-default login
                        # This refreshes the credentials before recreating the client
                        reauth_success = False
                        try:
                            # Try to force re-authentication
                            reauth_success = self._force_reauth()
                            if reauth_success:
                                # Recreate the client with fresh credentials
                                self._authenticate(force_reauth=False)
                            else:
                                # Re-auth command failed, but try recreating client anyway
                                # (credentials might have been refreshed externally)
                                try:
                                    self._authenticate(force_reauth=False)
                                    reauth_success = True  # Client creation succeeded
                                except Exception:
                                    pass  # Will be handled below
                            
                            if reauth_success and attempt < max_retries:
                                # Small delay before retry to ensure credentials are refreshed
                                time.sleep(1)
                                continue
                            elif not reauth_success and attempt >= max_retries:
                                # Re-auth failed and we're out of retries
                                raise RuntimeError(
                                    f"Authentication error detected and automatic re-authentication failed after {max_retries + 1} attempts. "
                                    "Please run 'gcloud auth application-default login' manually to refresh your credentials. "
                                    f"Original error: {str(e)}"
                                ) from e
                            elif not reauth_success:
                                # Re-auth failed but we have retries left - continue to backoff
                                # Credentials might be refreshed externally between retries
                                pass
                        except Exception as auth_error:
                            if attempt >= max_retries:
                                raise RuntimeError(
                                    f"Re-authentication failed after {max_retries + 1} attempts: {str(auth_error)}. "
                                    "Please run 'gcloud auth application-default login' manually. "
                                    f"Original error: {str(e)}"
                                ) from e
                            # If re-auth failed but we have retries left, continue to exponential backoff
                            # This allows the retry mechanism to potentially work if credentials refresh in the background
                    else:
                        # Auth error detected but auto_reauth is disabled or using API key
                        if attempt >= max_retries:
                            raise RuntimeError(
                                f"Authentication error detected but auto_reauth is disabled. "
                                "Please run 'gcloud auth application-default login' manually. "
                                f"Original error: {str(e)}"
                            ) from e
                
                if attempt >= max_retries:
                    raise
                
                # Exponential backoff
                delay = self.exponential_backoff_with_jitter(attempt)
                time.sleep(delay)
        
        # Should never reach here, but just in case
        raise RuntimeError("Failed to generate content after retries")
