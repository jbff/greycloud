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
    
    def _authenticate(self):
        """Create authenticated client"""
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
                    import subprocess
                    try:
                        subprocess.run(
                            ["gcloud", "auth", "application-default", "login"],
                            check=True,
                            capture_output=False,  # Allow user interaction
                            text=True
                        )
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
                    except subprocess.CalledProcessError:
                        raise RuntimeError(
                            "Automatic re-authentication failed. Please run 'gcloud auth application-default login' manually."
                        ) from e
                    except Exception as login_error:
                        raise RuntimeError(
                            "Re-authentication error. Please run 'gcloud auth application-default login' manually."
                        ) from login_error
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
        safety_settings_list = []
        if final_safety_settings:
            for setting in final_safety_settings:
                if isinstance(setting, dict):
                    safety_settings_list.append(
                        types.SafetySetting(
                            category=setting["category"],
                            threshold=setting["threshold"]
                        )
                    )
                else:
                    safety_settings_list.append(setting)
        
        # Build config dict
        config_dict = {
            "temperature": final_temperature,
            "top_p": final_top_p,
            "max_output_tokens": final_max_output_tokens,
            "safety_settings": safety_settings_list,
            "tools": final_tools,
        }
        
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
        """Check if an error is related to authentication/authorization"""
        error_str = str(error).lower()
        error_repr = repr(error).lower()
        combined_error = error_str + " " + error_repr
        
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
        ]
        return any(keyword in combined_error for keyword in auth_keywords)
    
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
                if self._is_authentication_error(e):
                    if self.config.auto_reauth and not self.config.use_api_key:
                        try:
                            self._authenticate()
                            if attempt < max_retries:
                                continue
                        except Exception as auth_error:
                            if attempt >= max_retries:
                                raise RuntimeError(f"Re-authentication failed: {str(auth_error)}") from e
                
                if attempt >= max_retries:
                    raise
                
                # Exponential backoff
                delay = self.exponential_backoff_with_jitter(attempt)
                time.sleep(delay)
        
        # Should never reach here, but just in case
        raise RuntimeError("Failed to generate content after retries")
