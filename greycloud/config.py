"""
Configuration class for GreyCloud module
"""

import os
from typing import Optional, List, Dict, Any
from dataclasses import dataclass, field


def _env_bool(name: str) -> bool:
    """Read an opt-in boolean environment variable.

    Shared by the env-driven bool flags so their acceptance semantics
    (``1``/``true``/``yes``, case-insensitive) stay in sync.
    """
    return os.environ.get(name, "").lower() in ("1", "true", "yes")


@dataclass
class GreyCloudConfig:
    """Configuration for GreyCloud client"""

    # Project and location
    project_id: str = field(
        default_factory=lambda: os.environ.get(
            "PROJECT_ID", os.environ.get("GCP_PROJECT", "")
        )
    )
    location: str = field(
        default_factory=lambda: os.environ.get(
            "LOCATION", os.environ.get("GCP_LOCATION", "us-east4")
        )
    )

    # Authentication
    use_api_key: bool = field(default_factory=lambda: _env_bool("USE_API_KEY"))
    api_key_file: str = field(
        default_factory=lambda: os.environ.get("API_KEY_FILE", "GOOGLE_CLOUD_API_KEY")
    )
    sa_email: Optional[str] = field(
        default_factory=lambda: os.environ.get("SA_EMAIL", None)
    )
    # Opt-in only: never spawn interactive `gcloud auth application-default
    # login` (browser popup) unless explicitly requested via AUTO_REAUTH env
    # var or auto_reauth=True. Always off under pytest.
    auto_reauth: bool = field(default_factory=lambda: _env_bool("AUTO_REAUTH"))

    # Model configuration
    # Default to a generally available Gemini 3 flash model.
    # You can override this per-request when calling GreyCloudClient.
    model: str = "gemini-3-flash-preview"
    endpoint: str = "https://aiplatform.googleapis.com"
    api_version: str = "v1"

    # Generation parameters
    temperature: float = 1.0
    top_p: float = 0.95
    # When None, no seed is sent and the model behaves stochastically.
    # Set an explicit integer here to make outputs more repeatable.
    seed: Optional[int] = None
    max_output_tokens: int = 65535

    # Safety settings
    # When None, Vertex AI's defaults are used.
    # To explicitly control safety, pass a list of dicts or SafetySetting objects.
    safety_settings: Optional[List[Dict[str, Any]]] = None

    # System instruction
    system_instruction: Optional[str] = None

    # Tools configuration
    vertex_ai_search_datastore: Optional[str] = field(
        default_factory=lambda: os.environ.get("VERTEX_AI_SEARCH_DATASTORE", None)
    )
    use_vertex_ai_search: bool = False
    # Grounding: "inject" (default) runs the Discovery Engine search in GreyCloud and
    # injects results into the prompt. "tool" keeps the legacy tools.retrieval.
    grounding_mode: str = "inject"  # "inject" | "tool"
    # Skip-guard threshold: when > 0, inject-mode
    # grounding skips the Discovery Engine search when the effective query
    # (per-call grounding_query if given, else the last user message) is
    # shorter than this many characters. 0 (default) preserves current
    # behavior — conversational turns like "thanks" then still trigger a search.
    min_grounding_query_chars: int = 0
    # Opt-in extractive answers: when True, the
    # :search request includes extractiveContentSpec to get paragraph-scale
    # passages. False (default) sends the snippets-only payload, which every
    # datastore type accepts — datastores built with chunking config reject
    # extractiveContentSpec with HTTP 400, so this is the caller's opt-in,
    # not a library default (0.3.12 first sent it unconditionally and broke
    # every chunked datastore).
    extractive_content_spec: bool = False

    # Thinking config
    # When None, no thinking config is sent. Set to "LOW", "MEDIUM", or "HIGH"
    # for models that support thinking.
    thinking_level: Optional[str] = None  # None, "LOW", "MEDIUM", "HIGH"

    # Batch processing
    # Buckets must now be configured explicitly when batch/GCS helpers are used.
    batch_gcs_bucket: Optional[str] = field(
        default_factory=lambda: os.environ.get("BATCH_GCS_BUCKET", None)
    )
    batch_location: str = "global"  # Batch jobs require global location
    batch_poll_interval: int = 30

    # File upload
    gcs_bucket: Optional[str] = None

    def __post_init__(self):
        """Validate and set defaults after initialization"""
        if not self.project_id:
            # Try to get from gcloud config as a convenience for local development.
            import subprocess

            try:
                result = subprocess.run(
                    ["gcloud", "config", "get-value", "project"],
                    capture_output=True,
                    text=True,
                    check=False,
                )
                if (
                    result.returncode == 0
                    and result.stdout.strip()
                    and result.stdout.strip() != "(unset)"
                ):
                    self.project_id = result.stdout.strip()
            except (subprocess.CalledProcessError, FileNotFoundError):
                pass

        if not self.project_id:
            raise ValueError(
                "PROJECT_ID or GCP_PROJECT environment variable must be set, or gcloud must be configured.\n"
                "Example: export PROJECT_ID=your-project-id\n"
                "Or run: gcloud config set project your-project-id"
            )

        if self.grounding_mode not in ("inject", "tool"):
            raise ValueError("grounding_mode must be 'inject' or 'tool'")

        if self.min_grounding_query_chars < 0:
            raise ValueError("min_grounding_query_chars must be >= 0")

        # Strict bool checks: a truthy non-bool (e.g. the string "false") is
        # dangerous in both directions — it would silently opt every search
        # into extractiveContentSpec (400 on chunking-config datastores), or
        # silently enable interactive browser re-login while the caller
        # believes the flag is off.
        for bool_field in ("extractive_content_spec", "use_api_key", "auto_reauth"):
            value = getattr(self, bool_field)
            if not isinstance(value, bool):
                raise TypeError(
                    f"{bool_field} must be a bool; got {type(value).__name__}"
                )
