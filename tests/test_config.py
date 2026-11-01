"""Unit tests for GreyCloudConfig"""

import os
import pytest
from unittest.mock import patch
from greycloud.config import GreyCloudConfig


class TestGreyCloudConfig:
    """Test GreyCloudConfig class"""

    def test_config_from_environment(self):
        """Test config creation from environment variables"""

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
            config = GreyCloudConfig()

            assert config.project_id == "env-project-id"
            assert config.location == "us-west1"
            assert config.sa_email == "env-sa@project.iam.gserviceaccount.com"
            assert config.use_api_key is True
            assert config.api_key_file == "env_api_key.txt"

    def test_config_explicit_values(self):
        """Test config with explicit values"""

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

        config = GreyCloudConfig(project_id="test-project")

        # Defaults
        assert config.model == "gemini-3.8-flash"
        assert config.temperature == 1.0
        assert config.top_p == 0.95
        assert config.max_output_tokens == 65535
        # Seed defaults to None (no fixed seed)
        assert config.seed is None
        # auto-reauth is opt-in (browser popup); default must be off
        assert config.auto_reauth is False
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

        config = GreyCloudConfig(project_id="test-project")

        # When not provided, we leave safety_settings as None so Vertex uses its defaults.
        assert config.safety_settings is None

    def test_config_custom_safety_settings(self):
        """Test custom safety settings"""

        custom_settings = [
            {
                "category": "HARM_CATEGORY_HATE_SPEECH",
                "threshold": "BLOCK_MEDIUM_AND_ABOVE",
            }
        ]

        config = GreyCloudConfig(
            project_id="test-project", safety_settings=custom_settings
        )

        assert config.safety_settings == custom_settings

    def test_config_batch_bucket_from_env(self):
        """Test batch bucket configured from environment"""

        with patch.dict(os.environ, {"BATCH_GCS_BUCKET": "env-batch-bucket"}):
            config = GreyCloudConfig(project_id="test-project")
            assert config.batch_gcs_bucket == "env-batch-bucket"
            # gcs_bucket is independent and defaults to None unless explicitly set
            assert config.gcs_bucket is None

    def test_config_custom_batch_bucket(self):
        """Test custom batch bucket name"""

        config = GreyCloudConfig(
            project_id="test-project", batch_gcs_bucket="custom-batch-bucket"
        )

        assert config.batch_gcs_bucket == "custom-batch-bucket"

    def test_config_default_sa_email(self):
        """By default, sa_email should be None unless explicitly provided"""

        config = GreyCloudConfig(project_id="test-project")

        assert config.sa_email is None

    def test_config_sa_email_from_env(self):
        """Test service account email coming from environment"""

        with patch.dict(
            os.environ, {"SA_EMAIL": "env-sa@project.iam.gserviceaccount.com"}
        ):
            config = GreyCloudConfig(project_id="test-project")
            assert config.sa_email == "env-sa@project.iam.gserviceaccount.com"

    @pytest.mark.parametrize("value", ["1", "true", "yes", "TRUE"])
    def test_config_auto_reauth_from_env(self, value):
        """AUTO_REAUTH env opt-in parses like USE_API_KEY (shared _env_bool)"""
        with patch.dict(os.environ, {"AUTO_REAUTH": value}):
            config = GreyCloudConfig(project_id="test-project")

            assert config.auto_reauth is True

    def test_config_auto_reauth_default_off(self):
        config = GreyCloudConfig(project_id="test-project")

        assert config.auto_reauth is False

    @pytest.mark.parametrize("bad", ["false", "0", 1])
    def test_config_auto_reauth_rejects_non_bool(self, bad):
        """A truthy non-bool would enable browser re-login while the caller
        believes the flag is off (same hazard class as extractive_content_spec)."""
        with pytest.raises(TypeError, match="auto_reauth"):
            GreyCloudConfig(project_id="test-project", auto_reauth=bad)

    def test_config_use_api_key_rejects_non_bool(self):
        with pytest.raises(TypeError, match="use_api_key"):
            GreyCloudConfig(project_id="test-project", use_api_key="true")

    def test_config_use_api_key_from_env(self):
        """Test use_api_key from environment variable"""

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
                config = GreyCloudConfig(project_id="test-project")
                assert config.use_api_key == expected

    def test_config_grounding_mode_default(self):
        """Default grounding_mode is 'inject' (new behavior, no config change needed)"""

        config = GreyCloudConfig(project_id="test-project")

        assert config.grounding_mode == "inject"

    def test_config_grounding_mode_tool(self):
        """grounding_mode='tool' is accepted (legacy escape hatch)"""

        config = GreyCloudConfig(project_id="test-project", grounding_mode="tool")

        assert config.grounding_mode == "tool"

    def test_config_grounding_mode_invalid_raises(self):
        """An invalid grounding_mode value raises ValueError"""

        with pytest.raises(ValueError, match="grounding_mode"):
            GreyCloudConfig(project_id="test-project", grounding_mode="bogus")

    def test_config_vertex_ai_search(self):
        """Test Vertex AI Search configuration"""

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

        config = GreyCloudConfig(project_id="test-project", thinking_level="HIGH")

        assert config.thinking_level == "HIGH"

        config_none = GreyCloudConfig(project_id="test-project", thinking_level=None)
        assert config_none.thinking_level is None

    def test_config_min_grounding_query_chars_default_off(self):
        """min_grounding_query_chars defaults to 0 (skip guard off)"""

        config = GreyCloudConfig(project_id="test-project")

        assert config.min_grounding_query_chars == 0

    def test_config_min_grounding_query_chars_positive(self):

        config = GreyCloudConfig(
            project_id="test-project", min_grounding_query_chars=25
        )

        assert config.min_grounding_query_chars == 25

    def test_config_min_grounding_query_chars_negative_raises(self):
        """A negative threshold is invalid: it would silently disable nothing"""

        with pytest.raises(ValueError, match="min_grounding_query_chars"):
            GreyCloudConfig(project_id="test-project", min_grounding_query_chars=-1)

    def test_config_extractive_content_spec_default_off(self):
        """extractive_content_spec defaults to False (snippets-only, the wire
        behavior accepted by every datastore type). project_id is passed, so
        the gcloud fallback in __post_init__ never runs and needs no mocking."""
        config = GreyCloudConfig(project_id="test-project")

        assert config.extractive_content_spec is False

    def test_config_extractive_content_spec_opt_in(self):
        config = GreyCloudConfig(
            project_id="test-project", extractive_content_spec=True
        )

        assert config.extractive_content_spec is True

    @pytest.mark.parametrize("bad", ["false", "0", 1])
    def test_config_extractive_content_spec_rejects_non_bool(self, bad):
        """A truthy non-bool (e.g. the string "false" — falsy-looking but
        truthy) must be rejected, not silently coerced: it would opt every
        search into extractiveContentSpec while the caller believes the flag
        is off."""
        with pytest.raises(TypeError, match="extractive_content_spec"):
            GreyCloudConfig(project_id="test-project", extractive_content_spec=bad)

    def test_config_gemini_4_defaults(self):
        """Test Gemini 4 and modernized config defaults"""
        cfg = GreyCloudConfig(project_id="test-proj")
        assert cfg.api_version == "v1beta1"
        assert cfg.request_timeout == 1200
        # Tri-state: None means "unset" — non-G4 omits the field (0.4.x
        # behavior); G4 builders resolve None to include_thoughts=True.
        assert cfg.include_thoughts is None

    @pytest.mark.parametrize("bad_timeout", [0, -1, -100])
    def test_config_request_timeout_invalid_raises(self, bad_timeout):
        with pytest.raises(ValueError, match="request_timeout"):
            GreyCloudConfig(project_id="test-proj", request_timeout=bad_timeout)

    @pytest.mark.parametrize("bad_type", ["1200", None, 12.5, True])
    def test_config_request_timeout_rejects_non_int(self, bad_type):
        with pytest.raises((TypeError, ValueError)):
            GreyCloudConfig(project_id="test-proj", request_timeout=bad_type)

    @pytest.mark.parametrize("bad", ["true", "false", 0, 1])
    def test_config_include_thoughts_rejects_non_bool(self, bad):
        with pytest.raises(TypeError, match="include_thoughts"):
            GreyCloudConfig(project_id="test-proj", include_thoughts=bad)


def test_is_gemini_4():
    """Test is_gemini_4 model detection helper"""
    from greycloud.config import is_gemini_4

    assert is_gemini_4("gemini-4-argon") is True
    assert is_gemini_4("gemini-4-carbon") is True
    assert is_gemini_4("gemini-4-flash") is True
    assert is_gemini_4("gemini-4.0-pro") is True
    assert is_gemini_4("models/gemini-4-argon-preview") is True
    assert is_gemini_4("GEMINI-4-FLASH") is True
    assert is_gemini_4("gemini-3.8-flash") is False
    assert is_gemini_4("gemini-2.5-pro") is False
    assert is_gemini_4("") is False
    assert is_gemini_4(None) is False
    assert is_gemini_4(123) is False


def test_is_gemini_4_ignores_codename_substrings_in_resource_paths():
    """Project/endpoint IDs containing 'argon' or 'carbon' are not G4 models.

    Only a path segment that *starts* with the family or codename identifies a
    Gemini 4 model; a substring inside a project or endpoint id must not.
    """
    from greycloud.config import is_gemini_4

    assert (
        is_gemini_4("projects/acme-carbon/locations/us-central1/endpoints/123") is False
    )
    assert is_gemini_4("projects/argon-lab/locations/us-east4/endpoints/456") is False
    assert is_gemini_4("projects/p/locations/l/endpoints/789") is False

    # Real model segments still match, including tuned endpoints that embed
    # the model name.
    assert (
        is_gemini_4("projects/p/locations/l/publishers/google/models/gemini-4-argon")
        is True
    )
    assert is_gemini_4("argon") is True
    assert is_gemini_4("models/carbon-preview") is True
