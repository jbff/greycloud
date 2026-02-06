"""
Configuration class for GreyCloud module
"""

import os
from typing import Optional, List, Dict, Any
from dataclasses import dataclass, field


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
    use_api_key: bool = field(
        default_factory=lambda: os.environ.get("USE_API_KEY", "").lower()
        in ("1", "true", "yes")
    )
    api_key_file: str = field(
        default_factory=lambda: os.environ.get("API_KEY_FILE", "GOOGLE_CLOUD_API_KEY")
    )
    sa_email: Optional[str] = field(
        default_factory=lambda: os.environ.get("SA_EMAIL", None)
    )
    auto_reauth: bool = True

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
