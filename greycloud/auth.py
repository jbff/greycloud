"""
Authentication utilities for GreyCloud
"""

import subprocess
import sys
from typing import Optional

try:
    import google.auth
    from google.auth import impersonated_credentials

    HAS_GOOGLE_AUTH = True
except ImportError:
    HAS_GOOGLE_AUTH = False
    google = None
    impersonated_credentials = None

from google import genai
from google.genai import types


def create_client(
    project_id: str,
    location: str,
    sa_email: Optional[str] = None,
    use_api_key: bool = False,
    api_key_file: str = "GOOGLE_CLOUD_API_KEY",
    endpoint: str = "https://aiplatform.googleapis.com",
    api_version: str = "v1",
    auto_reauth: bool = True,
) -> genai.Client:
    """
    Create and return a genai.Client with appropriate authentication

    Args:
        project_id: GCP project ID
        location: GCP location/region
        sa_email: Service account email for impersonation (optional)
        use_api_key: If True, use API key authentication instead of OAuth
        api_key_file: Path to API key file if using API key auth
        endpoint: API endpoint base URL
        api_version: API version
        auto_reauth: If True, automatically attempt re-authentication on failure

    Returns:
        Authenticated genai.Client instance

    Raises:
        FileNotFoundError: If API key file not found when use_api_key=True
        ImportError: If google-auth not available and not using API key
        RuntimeError: If authentication fails
    """
    if use_api_key:
        try:
            with open(api_key_file, "r", encoding="utf-8") as f:
                api_key = f.read().strip()
        except FileNotFoundError:
            raise FileNotFoundError(
                f"API key file '{api_key_file}' not found. Either create it, or run without USE_API_KEY to use OAuth impersonation."
            )
        except Exception as e:
            raise Exception(f"Error reading API key file '{api_key_file}': {str(e)}")

        return genai.Client(
            vertexai=True,
            api_key=api_key,
            project=project_id,
            location=location,
            http_options=types.HttpOptions(
                base_url=endpoint,
                api_version=api_version,
            ),
        )
    else:
        # Preferred path: use Google Auth + service account impersonation (if sa_email provided)
        # This requires your user to have roles/iam.serviceAccountTokenCreator on the service account.
        if not HAS_GOOGLE_AUTH:
            raise ImportError(
                "google-auth is required for OAuth. Install it in your venv (pip install google-auth)."
            )

        scopes = ["https://www.googleapis.com/auth/cloud-platform"]
        credentials = None

        # Try to get default credentials first
        try:
            source_creds, _ = google.auth.default(scopes=scopes)

            # If sa_email is provided, try impersonation
            if sa_email:
                try:
                    credentials = impersonated_credentials.Credentials(
                        source_credentials=source_creds,
                        target_principal=sa_email,
                        target_scopes=scopes,
                        lifetime=3600,
                    )
                except Exception as imp_error:
                    # If impersonation fails, fall back to regular credentials
                    error_str = str(imp_error).lower()
                    if "not found" in error_str or "gaia id not found" in error_str:
                        # Service account doesn't exist, use regular credentials
                        credentials = source_creds
                    else:
                        raise
            else:
                # No SA email provided, use regular credentials
                credentials = source_creds
        except Exception:
            credentials = None

        # Fallback: ask gcloud to mint an access token
        if credentials is None:
            try:
                if sa_email:
                    # Try impersonation via gcloud
                    token = subprocess.check_output(
                        [
                            "gcloud",
                            "auth",
                            "print-access-token",
                            "--impersonate-service-account",
                            sa_email,
                        ],
                        text=True,
                        stderr=subprocess.PIPE,
                    ).strip()
                else:
                    # Use regular gcloud token
                    token = subprocess.check_output(
                        [
                            "gcloud",
                            "auth",
                            "print-access-token",
                        ],
                        text=True,
                        stderr=subprocess.PIPE,
                    ).strip()
            except subprocess.CalledProcessError as e:
                # Check if this is an authentication error that requires re-login
                error_output = e.stderr.decode("utf-8") if e.stderr else str(e)
                error_str = error_output.lower()

                if auto_reauth and (
                    "application-default" in error_str
                    or "reauth" in error_str
                    or "login" in error_str
                ):
                    # Try to automatically run gcloud auth application-default login
                    try:
                        # Note: This will require user interaction (browser) if not already authenticated
                        subprocess.run(
                            ["gcloud", "auth", "application-default", "login"],
                            check=True,
                            capture_output=False,  # Allow user interaction
                            text=True,
                        )
                        # Retry getting the token after re-authentication
                        if sa_email:
                            token = subprocess.check_output(
                                [
                                    "gcloud",
                                    "auth",
                                    "print-access-token",
                                    "--impersonate-service-account",
                                    sa_email,
                                ],
                                text=True,
                                stderr=subprocess.PIPE,
                            ).strip()
                        else:
                            token = subprocess.check_output(
                                [
                                    "gcloud",
                                    "auth",
                                    "print-access-token",
                                ],
                                text=True,
                                stderr=subprocess.PIPE,
                            ).strip()
                    except subprocess.CalledProcessError as login_error:
                        raise RuntimeError(
                            "Reauthentication is needed. Please run `gcloud auth application-default login` to reauthenticate."
                        ) from login_error
                    except Exception as login_error:
                        raise RuntimeError(
                            "Reauthentication is needed. Please run `gcloud auth application-default login` to reauthenticate."
                        ) from login_error
                elif sa_email and (
                    "not found" in error_str or "gaia id not found" in error_str
                ):
                    # Service account doesn't exist, try without impersonation
                    try:
                        token = subprocess.check_output(
                            [
                                "gcloud",
                                "auth",
                                "print-access-token",
                            ],
                            text=True,
                            stderr=subprocess.PIPE,
                        ).strip()
                    except subprocess.CalledProcessError:
                        raise RuntimeError(
                            f"Service account '{sa_email}' not found. Either create it or set SA_EMAIL to an existing service account, "
                            "or unset SA_EMAIL to use your user credentials."
                        ) from e
                else:
                    raise RuntimeError(
                        "Failed to acquire OAuth credentials. Ensure you're logged into gcloud."
                    ) from e
            except Exception as e:
                raise RuntimeError(
                    "Failed to acquire OAuth credentials. Ensure you're logged into gcloud."
                ) from e

            # Wrap the raw token in a minimal Credentials object.
            from google.auth.credentials import Credentials as _Credentials

            class _StaticTokenCredentials(_Credentials):
                def __init__(self, access_token: str):
                    # Don't call super().__init__() as it tries to set token property
                    # Instead, set the internal _token attribute directly
                    self._token = access_token

                @property
                def expired(self):
                    return False

                @property
                def valid(self):
                    return True

                @property
                def token(self):
                    return self._token

                def refresh(self, request):
                    pass

            credentials = _StaticTokenCredentials(token)

        return genai.Client(
            vertexai=True,
            credentials=credentials,
            project=project_id,
            location=location,
            http_options=types.HttpOptions(
                base_url=endpoint,
                api_version=api_version,
            ),
        )
