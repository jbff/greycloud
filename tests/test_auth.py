"""Unit tests for authentication module"""

import subprocess
import pytest
from unittest.mock import patch, MagicMock, mock_open
from greycloud.auth import create_client, HAS_GOOGLE_AUTH


class TestCreateClient:
    """Test create_client function"""

    @pytest.mark.auth
    def test_create_client_with_api_key(self, tmp_path):
        """Test client creation with API key"""
        api_key_file = tmp_path / "api_key.txt"
        api_key_file.write_text("test-api-key-12345")

        with patch("greycloud.auth.genai") as mock_genai:
            mock_client = MagicMock()
            mock_genai.Client.return_value = mock_client

            client = create_client(
                project_id="test-project",
                location="us-east4",
                use_api_key=True,
                api_key_file=str(api_key_file),
            )

            mock_genai.Client.assert_called_once()
            call_kwargs = mock_genai.Client.call_args[1]
            assert call_kwargs["vertexai"] is True
            assert call_kwargs["api_key"] == "test-api-key-12345"
            assert call_kwargs["project"] == "test-project"
            assert call_kwargs["location"] == "us-east4"

    @pytest.mark.auth
    def test_create_client_api_key_file_not_found(self):
        """Test error when API key file not found"""
        with pytest.raises(FileNotFoundError, match="API key file"):
            create_client(
                project_id="test-project",
                location="us-east4",
                use_api_key=True,
                api_key_file="nonexistent_file.txt",
            )

    @pytest.mark.auth
    @patch("greycloud.auth.HAS_GOOGLE_AUTH", True)
    @patch("greycloud.auth.google.auth")
    def test_create_client_oauth_with_impersonation(self, mock_auth):
        """Test OAuth client creation with service account impersonation"""
        mock_source_creds = MagicMock()
        mock_impersonated_creds = MagicMock()
        mock_auth.default.return_value = (mock_source_creds, None)

        with patch("greycloud.auth.impersonated_credentials") as mock_impersonated:
            mock_impersonated.Credentials.return_value = mock_impersonated_creds

            with patch("greycloud.auth.genai") as mock_genai:
                mock_client = MagicMock()
                mock_genai.Client.return_value = mock_client

                client = create_client(
                    project_id="test-project",
                    location="us-east4",
                    sa_email="sa@project.iam.gserviceaccount.com",
                    use_api_key=False,
                )

                mock_impersonated.Credentials.assert_called_once()
                mock_genai.Client.assert_called_once()
                call_kwargs = mock_genai.Client.call_args[1]
                assert call_kwargs["credentials"] == mock_impersonated_creds

    @pytest.mark.auth
    @patch("greycloud.auth.HAS_GOOGLE_AUTH", True)
    @patch("greycloud.auth.google.auth")
    def test_create_client_oauth_without_impersonation(self, mock_auth):
        """Test OAuth client creation without service account"""
        mock_source_creds = MagicMock()
        mock_auth.default.return_value = (mock_source_creds, None)

        with patch("greycloud.auth.genai") as mock_genai:
            mock_client = MagicMock()
            mock_genai.Client.return_value = mock_client

            client = create_client(
                project_id="test-project", location="us-east4", use_api_key=False
            )

            mock_genai.Client.assert_called_once()
            call_kwargs = mock_genai.Client.call_args[1]
            assert call_kwargs["credentials"] == mock_source_creds

    @pytest.mark.auth
    @patch("greycloud.auth.HAS_GOOGLE_AUTH", False)
    def test_create_client_oauth_without_google_auth(self):
        """Test error when google-auth not available for OAuth"""
        with pytest.raises(ImportError, match="google-auth is required"):
            create_client(
                project_id="test-project", location="us-east4", use_api_key=False
            )

    @pytest.mark.auth
    @patch("greycloud.auth.HAS_GOOGLE_AUTH", True)
    @patch("greycloud.auth.google.auth")
    @patch("greycloud.auth.subprocess")
    def test_create_client_oauth_fallback_to_gcloud(self, mock_subprocess, mock_auth):
        """Test OAuth fallback to gcloud command"""
        # Simulate google.auth.default failing
        mock_auth.default.side_effect = Exception("No credentials")

        # Mock gcloud command
        mock_subprocess.check_output.return_value = "gcloud-access-token"

        with patch("greycloud.auth.genai") as mock_genai:
            mock_client = MagicMock()
            mock_genai.Client.return_value = mock_client

            client = create_client(
                project_id="test-project", location="us-east4", use_api_key=False
            )

            mock_subprocess.check_output.assert_called()
            mock_genai.Client.assert_called_once()
            # Verify credentials were created
            call_kwargs = mock_genai.Client.call_args[1]
            assert call_kwargs["credentials"] is not None

    @pytest.mark.auth
    @patch("greycloud.auth.HAS_GOOGLE_AUTH", True)
    @patch("greycloud.auth.google.auth")
    def test_create_client_oauth_auto_reauth(self, mock_auth):
        """Test automatic re-authentication"""
        import subprocess

        # Simulate authentication error
        mock_auth.default.side_effect = Exception("No credentials")

        # First call fails with auth error, then login succeeds, then token retrieval succeeds
        call_count = [0]

        def check_output_side_effect(*args, **kwargs):
            call_count[0] += 1
            if call_count[0] == 1:
                error = subprocess.CalledProcessError(
                    1, "gcloud", stderr=b"application-default"
                )
                raise error
            return "new-token"

        with patch(
            "greycloud.auth.subprocess.check_output",
            side_effect=check_output_side_effect,
        ):
            with patch(
                "greycloud.auth.subprocess.run", return_value=MagicMock(returncode=0)
            ):
                with patch("greycloud.auth.genai") as mock_genai:
                    mock_client = MagicMock()
                    mock_genai.Client.return_value = mock_client

                    client = create_client(
                        project_id="test-project",
                        location="us-east4",
                        use_api_key=False,
                        auto_reauth=True,
                    )

                    # Should have attempted re-authentication
                    assert call_count[0] >= 2

    @pytest.mark.auth
    @patch("greycloud.auth.HAS_GOOGLE_AUTH", True)
    @patch("greycloud.auth.google.auth")
    @patch("greycloud.auth.impersonated_credentials")
    def test_create_client_impersonation_fallback(self, mock_impersonated, mock_auth):
        """Test fallback when impersonation fails due to missing SA"""
        mock_source_creds = MagicMock()
        mock_auth.default.return_value = (mock_source_creds, None)

        # Simulate impersonation failure
        mock_impersonated.Credentials.side_effect = Exception("gaia id not found")

        with patch("greycloud.auth.genai") as mock_genai:
            mock_client = MagicMock()
            mock_genai.Client.return_value = mock_client

            client = create_client(
                project_id="test-project",
                location="us-east4",
                sa_email="nonexistent@project.iam.gserviceaccount.com",
                use_api_key=False,
            )

            # Should fall back to source credentials
            mock_genai.Client.assert_called_once()
            call_kwargs = mock_genai.Client.call_args[1]
            assert call_kwargs["credentials"] == mock_source_creds

    @pytest.mark.auth
    def test_create_client_custom_endpoint(self, tmp_path):
        """Test client creation with custom endpoint"""
        api_key_file = tmp_path / "api_key.txt"
        api_key_file.write_text("test-api-key")

        with patch("greycloud.auth.genai") as mock_genai:
            mock_client = MagicMock()
            mock_genai.Client.return_value = mock_client

            client = create_client(
                project_id="test-project",
                location="us-east4",
                use_api_key=True,
                api_key_file=str(api_key_file),
                endpoint="https://custom-endpoint.com",
                api_version="v2",
            )

            call_kwargs = mock_genai.Client.call_args[1]
            http_options = call_kwargs["http_options"]
            assert http_options.base_url == "https://custom-endpoint.com"
            assert http_options.api_version == "v2"
