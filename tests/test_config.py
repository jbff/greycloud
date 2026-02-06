"""Unit tests for GreyCloudConfig"""

import os
import pytest
from unittest.mock import patch, MagicMock
from greycloud.config import GreyCloudConfig


class TestGreyCloudConfig:
    """Test GreyCloudConfig class"""

    def test_config_from_environment(self):
        """Test config creation from environment variables"""
        import subprocess

        with patch.dict(
            os.environ,
            {
                "PROJECT_ID": "env-project-id",
                "LOCATION": "us-west1",
                "SA_EMAIL": "env-sa@project.iam.gserviceaccount.com",
                "USE_API_KEY": "1",
                "API_KEY_FILE": "env_api_key.txt",
            },
        ):
            with patch.object(subprocess, "run") as mock_run:
                mock_run.return_value.returncode = 0
                config = GreyCloudConfig()

                assert config.project_id == "env-project-id"
                assert config.location == "us-west1"
                assert config.sa_email == "env-sa@project.iam.gserviceaccount.com"
                assert config.use_api_key is True
                assert config.api_key_file == "env_api_key.txt"

    def test_config_explicit_values(self):
        """Test config with explicit values"""
        import subprocess

        with patch.object(subprocess, "run") as mock_run:
            mock_run.return_value.returncode = 0
            config = GreyCloudConfig(
                project_id="explicit-project",
                location="us-central1",
                model="gemini-2.0",
                temperature=0.5,
                top_p=0.8,
                max_output_tokens=1000,
            )

            assert config.project_id == "explicit-project"
            assert config.location == "us-central1"
            assert config.model == "gemini-2.0"
            assert config.temperature == 0.5
            assert config.top_p == 0.8
            assert config.max_output_tokens == 1000

    def test_config_defaults(self):
        """Test config default values"""
        import subprocess

        with patch.object(subprocess, "run") as mock_run:
            mock_run.return_value.returncode = 0
            mock_run.return_value.stdout = "test-project"
            config = GreyCloudConfig(project_id="test-project")

            # Defaults
            assert config.model == "gemini-3-flash-preview"
            assert config.temperature == 1.0
            assert config.top_p == 0.95
            assert config.max_output_tokens == 65535
            # Seed defaults to None (no fixed seed)
            assert config.seed is None
            assert config.auto_reauth is True
            assert config.batch_location == "global"
            assert config.batch_poll_interval == 30

    def test_config_project_id_from_gcloud(self):
        """Test getting project_id from gcloud config"""
        import subprocess

        with patch.object(subprocess, "run") as mock_run:
            mock_run.return_value.returncode = 0
            mock_run.return_value.stdout = "gcloud-project-id"

            with patch.dict(os.environ, {}, clear=True):
                config = GreyCloudConfig()
                assert config.project_id == "gcloud-project-id"

    def test_config_missing_project_id(self):
        """Test error when project_id is missing"""
        import subprocess

        with patch.object(subprocess, "run") as mock_run:
            mock_run.return_value.returncode = 1
            mock_run.return_value.stdout = "(unset)"

            with patch.dict(os.environ, {}, clear=True):
                with pytest.raises(ValueError, match="PROJECT_ID"):
                    GreyCloudConfig()

    def test_config_safety_settings_default(self):
        """Default safety settings should defer to Vertex defaults (None)"""
        import subprocess

        with patch.object(subprocess, "run") as mock_run:
            mock_run.return_value.returncode = 0
            config = GreyCloudConfig(project_id="test-project")

            # When not provided, we leave safety_settings as None so Vertex uses its defaults.
            assert config.safety_settings is None

    def test_config_custom_safety_settings(self):
        """Test custom safety settings"""
        import subprocess

        custom_settings = [
            {
                "category": "HARM_CATEGORY_HATE_SPEECH",
                "threshold": "BLOCK_MEDIUM_AND_ABOVE",
            }
        ]

        with patch.object(subprocess, "run") as mock_run:
            mock_run.return_value.returncode = 0
            config = GreyCloudConfig(
                project_id="test-project", safety_settings=custom_settings
            )

            assert config.safety_settings == custom_settings

    def test_config_batch_bucket_from_env(self):
        """Test batch bucket configured from environment"""
        import subprocess

        with patch.dict(os.environ, {"BATCH_GCS_BUCKET": "env-batch-bucket"}):
            with patch.object(subprocess, "run") as mock_run:
                mock_run.return_value.returncode = 0
                config = GreyCloudConfig(project_id="test-project")

            assert config.batch_gcs_bucket == "env-batch-bucket"
            # gcs_bucket is independent and defaults to None unless explicitly set
            assert config.gcs_bucket is None

    def test_config_custom_batch_bucket(self):
        """Test custom batch bucket name"""
        import subprocess

        with patch.object(subprocess, "run") as mock_run:
            mock_run.return_value.returncode = 0
            config = GreyCloudConfig(
                project_id="test-project", batch_gcs_bucket="custom-batch-bucket"
            )

            assert config.batch_gcs_bucket == "custom-batch-bucket"

    def test_config_default_sa_email(self):
        """By default, sa_email should be None unless explicitly provided"""
        import subprocess

        with patch.object(subprocess, "run") as mock_run:
            mock_run.return_value.returncode = 0
            config = GreyCloudConfig(project_id="test-project")

            assert config.sa_email is None

    def test_config_sa_email_from_env(self):
        """Test service account email coming from environment"""
        import subprocess

        with patch.dict(
            os.environ, {"SA_EMAIL": "env-sa@project.iam.gserviceaccount.com"}
        ):
            with patch.object(subprocess, "run") as mock_run:
                mock_run.return_value.returncode = 0
                config = GreyCloudConfig(project_id="test-project")

            assert config.sa_email == "env-sa@project.iam.gserviceaccount.com"

    def test_config_use_api_key_from_env(self):
        """Test use_api_key from environment variable"""
        import subprocess

        test_cases = [
            ("1", True),
            ("true", True),
            ("True", True),
            ("yes", True),
            ("0", False),
            ("false", False),
            ("", False),
        ]

        for env_value, expected in test_cases:
            with patch.dict(os.environ, {"USE_API_KEY": env_value}):
                with patch.object(subprocess, "run") as mock_run:
                    mock_run.return_value.returncode = 0
                    config = GreyCloudConfig(project_id="test-project")
                    assert config.use_api_key == expected

    def test_config_vertex_ai_search(self):
        """Test Vertex AI Search configuration"""
        import subprocess

        with patch.object(subprocess, "run") as mock_run:
            mock_run.return_value.returncode = 0
            config = GreyCloudConfig(
                project_id="test-project",
                use_vertex_ai_search=True,
                vertex_ai_search_datastore="projects/test/locations/us/datastores/test-ds",
            )

            assert config.use_vertex_ai_search is True
            assert (
                config.vertex_ai_search_datastore
                == "projects/test/locations/us/datastores/test-ds"
            )

    def test_config_thinking_level(self):
        """Test thinking level configuration"""
        import subprocess

        with patch.object(subprocess, "run") as mock_run:
            mock_run.return_value.returncode = 0
            config = GreyCloudConfig(project_id="test-project", thinking_level="HIGH")

            assert config.thinking_level == "HIGH"

            config_none = GreyCloudConfig(
                project_id="test-project", thinking_level=None
            )
            assert config_none.thinking_level is None
