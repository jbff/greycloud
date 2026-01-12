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
    project_id: str = field(default_factory=lambda: os.environ.get("PROJECT_ID", os.environ.get("GCP_PROJECT", "")))
    location: str = field(default_factory=lambda: os.environ.get("LOCATION", os.environ.get("GCP_LOCATION", "us-east4")))
    
    # Authentication
    use_api_key: bool = field(default_factory=lambda: os.environ.get("USE_API_KEY", "").lower() in ("1", "true", "yes"))
    api_key_file: str = field(default_factory=lambda: os.environ.get("API_KEY_FILE", "GOOGLE_CLOUD_API_KEY"))
    sa_email: Optional[str] = field(default_factory=lambda: os.environ.get("SA_EMAIL", None))
    auto_reauth: bool = True
    
    # Model configuration
    model: str = "gemini-3-pro-preview"
    endpoint: str = "https://aiplatform.googleapis.com"
    api_version: str = "v1"
    
    # Generation parameters
    temperature: float = 1.0
    top_p: float = 0.95
    seed: Optional[int] = 0
    max_output_tokens: int = 65535
    
    # Safety settings
    safety_settings: Optional[List[Dict[str, Any]]] = None
    
    # System instruction
    system_instruction: Optional[str] = None
    
    # Tools configuration
    vertex_ai_search_datastore: Optional[str] = field(default_factory=lambda: os.environ.get("VERTEX_AI_SEARCH_DATASTORE", None))
    use_vertex_ai_search: bool = False
    
    # Thinking config
    thinking_level: Optional[str] = "HIGH"  # None, "LOW", "MEDIUM", "HIGH"
    
    # Batch processing
    batch_gcs_bucket: Optional[str] = field(default_factory=lambda: os.environ.get("BATCH_GCS_BUCKET", None))
    batch_location: str = "global"  # Batch jobs require global location
    batch_poll_interval: int = 30
    
    # File upload
    gcs_bucket: Optional[str] = None
    
    def __post_init__(self):
        """Validate and set defaults after initialization"""
        if not self.project_id:
            # Try to get from gcloud config
            import subprocess
            try:
                result = subprocess.run(
                    ["gcloud", "config", "get-value", "project"],
                    capture_output=True,
                    text=True,
                    check=False
                )
                if result.returncode == 0 and result.stdout.strip() and result.stdout.strip() != "(unset)":
                    self.project_id = result.stdout.strip()
            except (subprocess.CalledProcessError, FileNotFoundError):
                pass
        
        if not self.project_id:
            raise ValueError(
                "PROJECT_ID or GCP_PROJECT environment variable must be set, or gcloud must be configured. "
                "Example: export PROJECT_ID=your-project-id\n"
                "Or run: gcloud config set project your-project-id"
            )
        
        # Set default batch bucket if not provided
        if not self.batch_gcs_bucket:
            self.batch_gcs_bucket = f"{self.project_id}-batch-jobs"
        
        # Set default GCS bucket if not provided
        if not self.gcs_bucket:
            self.gcs_bucket = self.batch_gcs_bucket
        
        # Set default safety settings if not provided
        if self.safety_settings is None:
            self.safety_settings = [
                {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "OFF"},
                {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "OFF"},
                {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "OFF"},
                {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "OFF"}
            ]
        
        # Set default SA email if not provided but project_id is available
        if not self.sa_email and self.project_id:
            self.sa_email = f"vertex-search-client@{self.project_id}.iam.gserviceaccount.com"
