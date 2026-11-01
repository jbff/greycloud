"""Unit tests for GreyCloudBatch"""

import json
import pytest
from unittest.mock import patch, MagicMock
from google.genai import types
from greycloud.batch import GreyCloudBatch
from greycloud.config import GreyCloudConfig


class TestGreyCloudBatch:
    """Test GreyCloudBatch class"""

    def test_batch_initialization(self, sample_config):
        """Test batch client initialization"""
        batch = GreyCloudBatch(sample_config)
        assert batch.config == sample_config
        assert batch._client is None
        assert batch._batch_client is None
        assert batch._storage_client is None

    def test_batch_initialization_default_config(self):
        """Test batch client with default config"""
        import subprocess

        with patch.object(subprocess, "run") as mock_run:
            mock_run.return_value.returncode = 0
            mock_run.return_value.stdout = "test-project"
            batch = GreyCloudBatch()
            assert batch.config is not None

    def test_client_property(self, sample_config):
        """Test client property"""
        with patch("greycloud.batch.create_client") as mock_create:
            mock_client = MagicMock()
            mock_create.return_value = mock_client

            batch = GreyCloudBatch(sample_config)
            client = batch.client

            assert client == mock_client
            mock_create.assert_called_once()

    def test_batch_client_property(self, sample_config):
        """Test batch_client property"""
        with patch("greycloud.batch.create_client") as mock_create:
            mock_client = MagicMock()
            mock_create.return_value = mock_client

            batch = GreyCloudBatch(sample_config)
            batch_client = batch.batch_client

            assert batch_client == mock_client
            # Should use batch_location (global)
            call_kwargs = mock_create.call_args[1]
            assert call_kwargs["location"] == "global"

    @pytest.mark.batch
    @patch("greycloud.batch.GCS_AVAILABLE", True)
    def test_storage_client_property(self, sample_config):
        """Test storage_client property"""
        with patch("greycloud.batch.create_client") as mock_create:
            mock_genai_client = MagicMock()
            mock_create.return_value = mock_genai_client

            with patch("greycloud.batch.storage") as mock_storage:
                mock_storage_client = MagicMock()
                mock_storage.Client.return_value = mock_storage_client

                batch = GreyCloudBatch(sample_config)
                storage_client = batch.storage_client

                assert storage_client == mock_storage_client

    @pytest.mark.batch
    @patch("greycloud.batch.GCS_AVAILABLE", False)
    def test_storage_client_not_available(self, sample_config):
        """Test error when GCS not available"""
        batch = GreyCloudBatch(sample_config)

        with pytest.raises(ImportError, match="google-cloud-storage"):
            _ = batch.storage_client

    @pytest.mark.batch
    @patch("greycloud.batch.GCS_AVAILABLE", True)
    def test_upload_file_to_gcs(self, sample_config, tmp_path):
        """Test file upload to GCS"""
        test_file = tmp_path / "test.txt"
        test_file.write_text("Test content")

        with patch("greycloud.batch.create_client") as mock_create:
            mock_create.return_value = MagicMock()

            with patch("greycloud.batch.storage") as mock_storage:
                mock_storage_client = MagicMock()
                mock_bucket = MagicMock()
                mock_blob = MagicMock()
                mock_storage.Client.return_value = mock_storage_client
                mock_storage_client.bucket.return_value = mock_bucket
                mock_bucket.blob.return_value = mock_blob

                batch = GreyCloudBatch(sample_config)
                gcs_uri = batch.upload_file_to_gcs(
                    file_path=test_file, blob_name="test.txt", bucket_name="test-bucket"
                )

                assert gcs_uri == "gs://test-bucket/test.txt"
                mock_blob.upload_from_file.assert_called_once()

    @pytest.mark.batch
    @patch("greycloud.batch.GCS_AVAILABLE", True)
    def test_upload_file_to_gcs_file_not_found(self, sample_config):
        """Test error when file not found"""
        with patch("greycloud.batch.create_client") as mock_create:
            mock_create.return_value = MagicMock()

            with patch("greycloud.batch.storage"):
                batch = GreyCloudBatch(sample_config)

                with pytest.raises(FileNotFoundError):
                    batch.upload_file_to_gcs(
                        file_path="nonexistent.txt", bucket_name="test-bucket"
                    )

    @pytest.mark.batch
    @patch("greycloud.batch.GCS_AVAILABLE", True)
    def test_upload_file_to_gcs_default_bucket(self, sample_config, tmp_path):
        """Test file upload with default bucket from config"""
        test_file = tmp_path / "test.txt"
        test_file.write_text("Test content")

        config = GreyCloudConfig(project_id="test-project", gcs_bucket="default-bucket")

        with patch("greycloud.batch.create_client") as mock_create:
            mock_create.return_value = MagicMock()

            with patch("greycloud.batch.storage") as mock_storage:
                mock_storage_client = MagicMock()
                mock_bucket = MagicMock()
                mock_blob = MagicMock()
                mock_storage.Client.return_value = mock_storage_client
                mock_storage_client.bucket.return_value = mock_bucket
                mock_bucket.blob.return_value = mock_blob

                batch = GreyCloudBatch(config)
                gcs_uri = batch.upload_file_to_gcs(file_path=test_file)

                assert "default-bucket" in gcs_uri

    @pytest.mark.batch
    @patch("greycloud.batch.GCS_AVAILABLE", True)
    def test_upload_string_to_gcs(self, sample_config):
        """Test string upload to GCS"""
        with patch("greycloud.batch.create_client") as mock_create:
            mock_create.return_value = MagicMock()

            with patch("greycloud.batch.storage") as mock_storage:
                mock_storage_client = MagicMock()
                mock_bucket = MagicMock()
                mock_blob = MagicMock()
                mock_storage.Client.return_value = mock_storage_client
                mock_storage_client.bucket.return_value = mock_bucket
                mock_bucket.blob.return_value = mock_blob

                batch = GreyCloudBatch(sample_config)
                gcs_uri = batch.upload_string_to_gcs(
                    content="Test content",
                    blob_name="test.txt",
                    bucket_name="test-bucket",
                )

                assert gcs_uri == "gs://test-bucket/test.txt"
                mock_blob.upload_from_string.assert_called_once_with(
                    "Test content", content_type="text/plain"
                )

    @pytest.mark.batch
    @patch("greycloud.batch.GCS_AVAILABLE", True)
    def test_upload_files_to_gcs_missing_bucket(self):
        """Test error when bucket not provided and config.gcs_bucket is None"""
        config = GreyCloudConfig(project_id="test-project")  # no gcs_bucket
        with patch("greycloud.batch.create_client") as mock_create:
            mock_create.return_value = MagicMock()
            with patch("greycloud.batch.storage"):
                batch = GreyCloudBatch(config)
                with pytest.raises(
                    ValueError,
                    match="bucket_name must be provided or set in config.gcs_bucket",
                ):
                    batch.upload_files_to_gcs(files=[{"name": "x", "content": "y"}])

    @pytest.mark.batch
    @patch("greycloud.batch.GCS_AVAILABLE", True)
    def test_upload_files_to_gcs(self, sample_config):
        """Test multiple file upload"""
        files = [
            {"name": "file1.txt", "content": "Content 1"},
            {"name": "file2.txt", "content": "Content 2"},
        ]

        with patch("greycloud.batch.create_client") as mock_create:
            mock_create.return_value = MagicMock()

            with patch("greycloud.batch.storage") as mock_storage:
                mock_storage_client = MagicMock()
                mock_bucket = MagicMock()
                mock_blob = MagicMock()
                mock_storage.Client.return_value = mock_storage_client
                mock_storage_client.bucket.return_value = mock_bucket
                mock_bucket.blob.return_value = mock_blob

                batch = GreyCloudBatch(sample_config)
                file_uris = batch.upload_files_to_gcs(
                    files=files, bucket_name="test-bucket"
                )

                assert len(file_uris) == 2
                assert "file1.txt" in file_uris
                assert "file2.txt" in file_uris
                assert mock_blob.upload_from_string.call_count == 2

    @pytest.mark.batch
    @patch("greycloud.batch.GCS_AVAILABLE", True)
    def test_upload_batch_requests_to_gcs(self, sample_config):
        """Test batch requests upload"""
        batch_requests = [
            types.InlinedRequest(
                model="gemini-3.8-flash",
                contents=[{"role": "user", "parts": [{"text": "Test"}]}],
                config=types.GenerateContentConfig(temperature=0.2),
            )
        ]

        with patch("greycloud.batch.create_client") as mock_create:
            mock_create.return_value = MagicMock()

            with patch("greycloud.batch.storage") as mock_storage:
                mock_storage_client = MagicMock()
                mock_bucket = MagicMock()
                mock_blob = MagicMock()
                mock_storage.Client.return_value = mock_storage_client
                mock_storage_client.bucket.return_value = mock_bucket
                mock_bucket.blob.return_value = mock_blob

                batch = GreyCloudBatch(sample_config)
                gcs_uri = batch.upload_batch_requests_to_gcs(
                    batch_requests=batch_requests, bucket_name="test-bucket"
                )

                assert gcs_uri.startswith("gs://test-bucket/batch_requests/")
                mock_blob.upload_from_string.assert_called_once()

                # Verify JSONL format: Vertex AI REST schema with camelCase fields
                call_args = mock_blob.upload_from_string.call_args
                uploaded_content = call_args[0][0]
                assert uploaded_content.startswith('{"request":')
                lines = uploaded_content.strip().split("\n")
                assert len(lines) == 1
                obj = json.loads(lines[0])
                assert "request" in obj
                req = obj["request"]
                assert req["model"] == "gemini-3.8-flash"
                assert "contents" in req
                # Config should be serialized as generationConfig (not "config")
                assert "generationConfig" in req
                assert "config" not in req
                assert req["generationConfig"]["temperature"] == 0.2

    @pytest.mark.batch
    @patch("greycloud.batch.GCS_AVAILABLE", True)
    def test_batch_jsonl_camelcase_fields(self, sample_config):
        """Test that JSONL serialization uses camelCase field names"""
        batch_requests = [
            types.InlinedRequest(
                model="gemini-3.8-flash",
                contents=[{"role": "user", "parts": [{"text": "Test"}]}],
                config=types.GenerateContentConfig(
                    temperature=0.5,
                    top_p=0.9,
                    top_k=40,
                    max_output_tokens=1024,
                    candidate_count=1,
                    stop_sequences=["END"],
                    response_mime_type="application/json",
                    seed=42,
                ),
            )
        ]

        with patch("greycloud.batch.create_client") as mock_create:
            mock_create.return_value = MagicMock()
            with patch("greycloud.batch.storage") as mock_storage:
                mock_storage_client = MagicMock()
                mock_bucket = MagicMock()
                mock_blob = MagicMock()
                mock_storage.Client.return_value = mock_storage_client
                mock_storage_client.bucket.return_value = mock_bucket
                mock_bucket.blob.return_value = mock_blob

                batch = GreyCloudBatch(sample_config)
                batch.upload_batch_requests_to_gcs(
                    batch_requests=batch_requests, bucket_name="test-bucket"
                )

                uploaded = mock_blob.upload_from_string.call_args[0][0]
                req = json.loads(uploaded)["request"]
                gen_cfg = req["generationConfig"]
                assert gen_cfg["temperature"] == 0.5
                assert gen_cfg["topP"] == 0.9
                assert gen_cfg["topK"] == 40
                assert gen_cfg["maxOutputTokens"] == 1024
                assert gen_cfg["candidateCount"] == 1
                assert gen_cfg["stopSequences"] == ["END"]
                assert gen_cfg["responseMimeType"] == "application/json"
                assert gen_cfg["seed"] == 42
                # Must not contain snake_case versions
                assert "top_p" not in gen_cfg
                assert "max_output_tokens" not in gen_cfg

    @pytest.mark.batch
    @patch("greycloud.batch.GCS_AVAILABLE", True)
    def test_batch_jsonl_system_instruction(self, sample_config):
        """Test that systemInstruction is serialized as top-level request field"""
        batch_requests = [
            types.InlinedRequest(
                model="gemini-4-argon",
                contents=[{"role": "user", "parts": [{"text": "Hello"}]}],
                config=types.GenerateContentConfig(
                    system_instruction="You are a helpful assistant.",
                    temperature=0.5,
                ),
            )
        ]

        with patch("greycloud.batch.create_client") as mock_create:
            mock_create.return_value = MagicMock()
            with patch("greycloud.batch.storage") as mock_storage:
                mock_storage_client = MagicMock()
                mock_bucket = MagicMock()
                mock_blob = MagicMock()
                mock_storage.Client.return_value = mock_storage_client
                mock_storage_client.bucket.return_value = mock_bucket
                mock_bucket.blob.return_value = mock_blob

                batch = GreyCloudBatch(sample_config)
                batch.upload_batch_requests_to_gcs(
                    batch_requests=batch_requests, bucket_name="test-bucket"
                )

                uploaded = mock_blob.upload_from_string.call_args[0][0]
                req = json.loads(uploaded)["request"]
                # systemInstruction is top-level, not inside generationConfig
                assert "systemInstruction" in req
                assert req["systemInstruction"]["role"] == "user"
                assert req["systemInstruction"]["parts"] == [
                    {"text": "You are a helpful assistant."}
                ]
                # Must not be inside generationConfig
                assert "systemInstruction" not in req.get("generationConfig", {})
                assert "system_instruction" not in req.get("generationConfig", {})

    @pytest.mark.batch
    @patch("greycloud.batch.GCS_AVAILABLE", True)
    def test_batch_jsonl_thinking_config(self, sample_config):
        """Test that thinkingConfig is nested inside generationConfig"""
        thinking = types.ThinkingConfig(thinking_budget=2048)
        batch_requests = [
            types.InlinedRequest(
                model="gemini-3.8-flash",
                contents=[{"role": "user", "parts": [{"text": "Think"}]}],
                config=types.GenerateContentConfig(
                    temperature=0.5,
                    thinking_config=thinking,
                ),
            )
        ]

        with patch("greycloud.batch.create_client") as mock_create:
            mock_create.return_value = MagicMock()
            with patch("greycloud.batch.storage") as mock_storage:
                mock_storage_client = MagicMock()
                mock_bucket = MagicMock()
                mock_blob = MagicMock()
                mock_storage.Client.return_value = mock_storage_client
                mock_storage_client.bucket.return_value = mock_bucket
                mock_bucket.blob.return_value = mock_blob

                batch = GreyCloudBatch(sample_config)
                batch.upload_batch_requests_to_gcs(
                    batch_requests=batch_requests, bucket_name="test-bucket"
                )

                uploaded = mock_blob.upload_from_string.call_args[0][0]
                req = json.loads(uploaded)["request"]
                gen_cfg = req["generationConfig"]
                assert "thinkingConfig" in gen_cfg
                assert gen_cfg["thinkingConfig"]["thinkingBudget"] == 2048

    @pytest.mark.batch
    @patch("greycloud.batch.GCS_AVAILABLE", True)
    def test_batch_jsonl_omits_none_values(self, sample_config):
        """Test that None config values are omitted from JSONL"""
        batch_requests = [
            types.InlinedRequest(
                model="gemini-3.8-flash",
                contents=[{"role": "user", "parts": [{"text": "Test"}]}],
                config=types.GenerateContentConfig(temperature=0.5),
            )
        ]

        with patch("greycloud.batch.create_client") as mock_create:
            mock_create.return_value = MagicMock()
            with patch("greycloud.batch.storage") as mock_storage:
                mock_storage_client = MagicMock()
                mock_bucket = MagicMock()
                mock_blob = MagicMock()
                mock_storage.Client.return_value = mock_storage_client
                mock_storage_client.bucket.return_value = mock_bucket
                mock_bucket.blob.return_value = mock_blob

                batch = GreyCloudBatch(sample_config)
                batch.upload_batch_requests_to_gcs(
                    batch_requests=batch_requests, bucket_name="test-bucket"
                )

                uploaded = mock_blob.upload_from_string.call_args[0][0]
                req = json.loads(uploaded)["request"]
                gen_cfg = req["generationConfig"]
                # Only temperature should be present, not nulls for other fields
                assert gen_cfg == {"temperature": 0.5}

    @pytest.mark.batch
    @patch("greycloud.batch.GCS_AVAILABLE", True)
    def test_batch_jsonl_gemini_4_omits_sampling_parameters(self, sample_config):
        """Test that Gemini 4 batch JSONL omits temperature, topP, topK."""
        thinking = types.ThinkingConfig(
            thinking_level="HIGH",
            include_thoughts=True,
        )
        batch_requests = [
            types.InlinedRequest(
                model="gemini-4-argon",
                contents=[{"role": "user", "parts": [{"text": "Hello"}]}],
                config=types.GenerateContentConfig(
                    temperature=0.7,
                    top_p=0.9,
                    top_k=20,
                    max_output_tokens=1024,
                    thinking_config=thinking,
                ),
            )
        ]

        with patch("greycloud.batch.create_client") as mock_create:
            mock_create.return_value = MagicMock()
            with patch("greycloud.batch.storage") as mock_storage:
                mock_storage_client = MagicMock()
                mock_bucket = MagicMock()
                mock_blob = MagicMock()
                mock_storage.Client.return_value = mock_storage_client
                mock_storage_client.bucket.return_value = mock_bucket
                mock_bucket.blob.return_value = mock_blob

                batch = GreyCloudBatch(sample_config)
                batch.upload_batch_requests_to_gcs(
                    batch_requests=batch_requests, bucket_name="test-bucket"
                )

                uploaded = mock_blob.upload_from_string.call_args[0][0]
                req = json.loads(uploaded)["request"]
                gen_cfg = req["generationConfig"]
                assert "temperature" not in gen_cfg
                assert "topP" not in gen_cfg
                assert "topK" not in gen_cfg
                assert gen_cfg["maxOutputTokens"] == 1024
                assert "thinkingConfig" in gen_cfg
                assert "thinkingBudget" not in gen_cfg["thinkingConfig"]
                assert gen_cfg["thinkingConfig"]["thinkingLevel"] == "HIGH"
                assert gen_cfg["thinkingConfig"]["includeThoughts"] is True

    @pytest.mark.batch
    @patch("greycloud.batch.GCS_AVAILABLE", True)
    def test_batch_jsonl_gemini_4_dict_config_omits_sampling(self, sample_config):
        """Test that actual dict configs for Gemini 4 drop sampling and penalties."""
        mock_req = MagicMock()
        mock_req.model = "gemini-4-argon"
        mock_req.contents = [{"role": "user", "parts": [{"text": "Hello"}]}]
        mock_req.config = {
            "temperature": 0.7,
            "topP": 0.9,
            "topK": 20,
            "frequencyPenalty": 0.5,
            "presencePenalty": 0.5,
            "maxOutputTokens": 500,
            "thinkingConfig": {
                "thinkingLevel": "LOW",
            },
        }

        mock_req_snake = MagicMock()
        mock_req_snake.model = "gemini-4-argon"
        mock_req_snake.contents = [{"role": "user", "parts": [{"text": "Hello 2"}]}]
        mock_req_snake.config = {
            "temperature": 0.7,
            "top_p": 0.9,
            "top_k": 20,
            "frequency_penalty": 0.5,
            "presence_penalty": 0.5,
            "maxOutputTokens": 600,
            "thinking_config": {
                "thinking_level": "HIGH",
            },
        }
        batch_requests = [mock_req, mock_req_snake]

        with patch("greycloud.batch.create_client") as mock_create:
            mock_create.return_value = MagicMock()
            with patch("greycloud.batch.storage") as mock_storage:
                mock_storage_client = MagicMock()
                mock_bucket = MagicMock()
                mock_blob = MagicMock()
                mock_storage.Client.return_value = mock_storage_client
                mock_storage_client.bucket.return_value = mock_bucket
                mock_bucket.blob.return_value = mock_blob

                batch = GreyCloudBatch(sample_config)
                batch.upload_batch_requests_to_gcs(
                    batch_requests=batch_requests, bucket_name="test-bucket"
                )

                uploaded = mock_blob.upload_from_string.call_args[0][0]
                lines = uploaded.strip().split("\n")
                assert len(lines) == 2

                # Line 1: camelCase dict
                req1 = json.loads(lines[0])["request"]
                gen_cfg1 = req1["generationConfig"]
                assert "temperature" not in gen_cfg1
                assert "topP" not in gen_cfg1
                assert "topK" not in gen_cfg1
                assert "frequencyPenalty" not in gen_cfg1
                assert "presencePenalty" not in gen_cfg1
                assert gen_cfg1["maxOutputTokens"] == 500
                assert "thinkingConfig" in gen_cfg1
                assert "thinkingBudget" not in gen_cfg1["thinkingConfig"]
                assert gen_cfg1["thinkingConfig"]["thinkingLevel"] == "LOW"

                # Line 2: snake_case dict
                req2 = json.loads(lines[1])["request"]
                gen_cfg2 = req2["generationConfig"]
                assert "temperature" not in gen_cfg2
                assert "top_p" not in gen_cfg2
                assert "top_k" not in gen_cfg2
                assert "frequency_penalty" not in gen_cfg2
                assert "presence_penalty" not in gen_cfg2
                assert gen_cfg2["maxOutputTokens"] == 600
                assert "thinking_config" in gen_cfg2
                assert "thinking_budget" not in gen_cfg2["thinking_config"]
                assert gen_cfg2["thinking_config"]["thinking_level"] == "HIGH"

    @pytest.mark.batch
    @patch("greycloud.batch.GCS_AVAILABLE", True)
    def test_batch_jsonl_gemini_4_uses_job_level_model(self, sample_config):
        """A job-level model must drive G4 sanitization when request.model is None."""
        batch_requests = [
            types.InlinedRequest(
                contents=[{"role": "user", "parts": [{"text": "Hello"}]}],
                config=types.GenerateContentConfig(
                    temperature=0.7,
                    top_p=0.9,
                    thinking_config=types.ThinkingConfig(thinking_level="HIGH"),
                ),
            )
        ]

        with patch("greycloud.batch.create_client") as mock_create:
            mock_create.return_value = MagicMock()
            with patch("greycloud.batch.storage") as mock_storage:
                mock_storage_client = MagicMock()
                mock_bucket = MagicMock()
                mock_blob = MagicMock()
                mock_storage.Client.return_value = mock_storage_client
                mock_storage_client.bucket.return_value = mock_bucket
                mock_bucket.blob.return_value = mock_blob

                batch = GreyCloudBatch(sample_config)
                batch.upload_batch_requests_to_gcs(
                    batch_requests=batch_requests,
                    bucket_name="test-bucket",
                    model="gemini-4-argon",
                )

                uploaded = mock_blob.upload_from_string.call_args[0][0]
                gen_cfg = json.loads(uploaded)["request"]["generationConfig"]
                assert "temperature" not in gen_cfg
                assert "topP" not in gen_cfg
                assert gen_cfg["thinkingConfig"]["thinkingLevel"] == "HIGH"

    @pytest.mark.batch
    @patch("greycloud.batch.GCS_AVAILABLE", True)
    def test_batch_jsonl_gemini_4_rejects_cached_content(self, sample_config):
        """Gemini 4 batch requests reject explicit caching like the client does."""
        object_request = types.InlinedRequest(
            model="gemini-4-argon",
            contents=[{"role": "user", "parts": [{"text": "Hello"}]}],
            config=types.GenerateContentConfig(cached_content="caches/123"),
        )
        camel_dict_request = types.InlinedRequest(
            model="gemini-4-argon",
            contents=[{"role": "user", "parts": [{"text": "Hello"}]}],
            config={"cachedContent": "caches/123"},
        )
        snake_dict_request = types.InlinedRequest(
            model="gemini-4-argon",
            contents=[{"role": "user", "parts": [{"text": "Hello"}]}],
            config={"cached_content": "caches/123"},
        )

        with patch("greycloud.batch.create_client") as mock_create:
            mock_create.return_value = MagicMock()
            with patch("greycloud.batch.storage"):
                mock_storage_client = MagicMock()
                mock_storage_client.bucket.return_value = MagicMock()
                batch = GreyCloudBatch(sample_config)

                for request in (
                    object_request,
                    camel_dict_request,
                    snake_dict_request,
                ):
                    with pytest.raises(ValueError, match="implicit durable caching"):
                        batch.upload_batch_requests_to_gcs(
                            batch_requests=[request], bucket_name="test-bucket"
                        )

    @pytest.mark.batch
    @patch("greycloud.batch.GCS_AVAILABLE", True)
    def test_batch_jsonl_gemini_4_rejects_thinking_budget(self, sample_config):
        """Gemini 4 batch requests reject thinking_budget like the client does."""
        object_request = types.InlinedRequest(
            model="gemini-4-argon",
            contents=[{"role": "user", "parts": [{"text": "Hello"}]}],
            config=types.GenerateContentConfig(
                thinking_config=types.ThinkingConfig(thinking_budget=2048)
            ),
        )
        camel_dict_request = types.InlinedRequest(
            model="gemini-4-argon",
            contents=[{"role": "user", "parts": [{"text": "Hello"}]}],
            config={"thinkingConfig": {"thinkingBudget": 2048}},
        )
        snake_dict_request = types.InlinedRequest(
            model="gemini-4-argon",
            contents=[{"role": "user", "parts": [{"text": "Hello"}]}],
            config={"thinking_config": {"thinking_budget": 2048}},
        )

        with patch("greycloud.batch.create_client") as mock_create:
            mock_create.return_value = MagicMock()
            with patch("greycloud.batch.storage"):
                mock_storage_client = MagicMock()
                mock_storage_client.bucket.return_value = MagicMock()
                batch = GreyCloudBatch(sample_config)

                for request in (
                    object_request,
                    camel_dict_request,
                    snake_dict_request,
                ):
                    with pytest.raises(ValueError, match="thinking_budget"):
                        batch.upload_batch_requests_to_gcs(
                            batch_requests=[request], bucket_name="test-bucket"
                        )

    @pytest.mark.batch
    @patch("greycloud.batch.GCS_AVAILABLE", True)
    def test_batch_jsonl_gemini_4_rejects_minimal_thinking_level(self, sample_config):
        """Gemini 4 batch requests reject MINIMAL thinking like the client does."""
        object_request = types.InlinedRequest(
            model="gemini-4-argon",
            contents=[{"role": "user", "parts": [{"text": "Hello"}]}],
            config=types.GenerateContentConfig(
                thinking_config=types.ThinkingConfig(thinking_level="MINIMAL")
            ),
        )
        camel_dict_request = types.InlinedRequest(
            model="gemini-4-argon",
            contents=[{"role": "user", "parts": [{"text": "Hello"}]}],
            config={"thinkingConfig": {"thinkingLevel": "MINIMAL"}},
        )
        snake_dict_request = types.InlinedRequest(
            model="gemini-4-argon",
            contents=[{"role": "user", "parts": [{"text": "Hello"}]}],
            config={"thinking_config": {"thinking_level": "MINIMAL"}},
        )

        with patch("greycloud.batch.create_client") as mock_create:
            mock_create.return_value = MagicMock()
            with patch("greycloud.batch.storage"):
                mock_storage_client = MagicMock()
                mock_storage_client.bucket.return_value = MagicMock()
                batch = GreyCloudBatch(sample_config)

                for request in (
                    object_request,
                    camel_dict_request,
                    snake_dict_request,
                ):
                    with pytest.raises(ValueError, match="MINIMAL"):
                        batch.upload_batch_requests_to_gcs(
                            batch_requests=[request], bucket_name="test-bucket"
                        )

    @pytest.mark.batch
    @patch("greycloud.batch.GCS_AVAILABLE", True)
    def test_batch_jsonl_no_metadata(self, sample_config):
        """Test that InlinedRequest.metadata is NOT forwarded to JSONL"""
        batch_requests = [
            types.InlinedRequest(
                model="gemini-4-argon",
                contents=[{"role": "user", "parts": [{"text": "Test"}]}],
                metadata={"request_id": "123"},
            )
        ]

        with patch("greycloud.batch.create_client") as mock_create:
            mock_create.return_value = MagicMock()
            with patch("greycloud.batch.storage") as mock_storage:
                mock_storage_client = MagicMock()
                mock_bucket = MagicMock()
                mock_blob = MagicMock()
                mock_storage.Client.return_value = mock_storage_client
                mock_storage_client.bucket.return_value = mock_bucket
                mock_bucket.blob.return_value = mock_blob

                batch = GreyCloudBatch(sample_config)
                batch.upload_batch_requests_to_gcs(
                    batch_requests=batch_requests, bucket_name="test-bucket"
                )

                uploaded = mock_blob.upload_from_string.call_args[0][0]
                req = json.loads(uploaded)["request"]
                assert "metadata" not in req
                assert "labels" not in req

    @pytest.mark.batch
    @patch("greycloud.batch.GCS_AVAILABLE", True)
    def test_clients_forward_request_timeout(self, sample_config):
        """Batch clients must honor the configured request deadline."""
        with patch("greycloud.batch.create_client") as mock_create:
            mock_create.return_value = MagicMock()
            batch = GreyCloudBatch(sample_config)

            _ = batch.client
            assert mock_create.call_args[1]["request_timeout"] == (
                sample_config.request_timeout
            )

            _ = batch.batch_client
            assert mock_create.call_args[1]["request_timeout"] == (
                sample_config.request_timeout
            )

    @pytest.mark.batch
    @patch("greycloud.batch.GCS_AVAILABLE", True)
    def test_monitor_batch_job_forwards_request_timeout(
        self, sample_config, mock_batch_job
    ):
        """Monitor polls must honor the configured request deadline."""
        mock_batch_job.done = True
        mock_batch_job.error = None

        with patch("greycloud.batch.create_client") as mock_create:
            mock_client = MagicMock()
            mock_client.batches.get.return_value = mock_batch_job
            mock_create.return_value = mock_client

            with patch("greycloud.batch.storage"):
                batch = GreyCloudBatch(sample_config)
                with patch("time.sleep"):
                    batch.monitor_batch_job(batch_job=mock_batch_job)

            assert mock_create.call_args[1]["request_timeout"] == (
                sample_config.request_timeout
            )

    @pytest.mark.batch
    @patch("greycloud.batch.GCS_AVAILABLE", True)
    def test_create_batch_job(self, sample_config):
        """Test batch job creation"""
        batch_requests = [
            types.InlinedRequest(
                model="gemini-4-argon",
                contents=[{"role": "user", "parts": [{"text": "Test"}]}],
                config=types.GenerateContentConfig(),
            )
        ]

        mock_batch_job = MagicMock(spec=types.BatchJob)
        mock_batch_job.name = "test-batch-job"

        with patch("greycloud.batch.create_client") as mock_create:
            mock_batch_client = MagicMock()
            mock_batch_client.batches.create.return_value = mock_batch_job
            mock_create.return_value = mock_batch_client

            with patch("greycloud.batch.storage"):
                batch = GreyCloudBatch(sample_config)

                with patch.object(
                    batch,
                    "upload_batch_requests_to_gcs",
                    return_value="gs://bucket/requests.jsonl",
                ):
                    job = batch.create_batch_job(
                        batch_requests=batch_requests, bucket_name="test-bucket"
                    )

                    assert job == mock_batch_job
                    mock_batch_client.batches.create.assert_called_once()

    @pytest.mark.batch
    @patch("greycloud.batch.GCS_AVAILABLE", True)
    def test_monitor_batch_job(self, sample_config, mock_batch_job):
        """Test batch job monitoring"""
        # Simulate job progression
        mock_batch_job.done = False
        mock_batch_job.state = "JOB_STATE_PENDING"

        mock_batch_job_complete = MagicMock(spec=types.BatchJob)
        mock_batch_job_complete.done = True
        mock_batch_job_complete.state = "JOB_STATE_SUCCEEDED"
        mock_batch_job_complete.error = None
        mock_batch_job_complete.name = mock_batch_job.name

        with patch("greycloud.batch.create_client") as mock_create:
            mock_client = MagicMock()
            mock_client.batches.get.side_effect = [
                mock_batch_job,
                mock_batch_job_complete,
            ]
            mock_create.return_value = mock_client

            with patch("greycloud.batch.storage"):
                batch = GreyCloudBatch(sample_config)

                callback_calls = []

                def callback(job, state):
                    callback_calls.append((job, state))

                with patch("time.sleep"):  # Speed up test
                    result = batch.monitor_batch_job(
                        batch_job=mock_batch_job, poll_interval=1, callback=callback
                    )

                assert result.done is True
                assert len(callback_calls) >= 1

    @pytest.mark.batch
    @patch("greycloud.batch.GCS_AVAILABLE", True)
    def test_monitor_batch_job_failure(self, sample_config, mock_batch_job):
        """Test batch job monitoring with failure"""
        mock_batch_job.done = True
        mock_batch_job.error = MagicMock()
        mock_batch_job.error.message = "Job failed"

        with patch("greycloud.batch.create_client") as mock_create:
            mock_client = MagicMock()
            mock_client.batches.get.return_value = mock_batch_job
            mock_create.return_value = mock_client

            with patch("greycloud.batch.storage"):
                batch = GreyCloudBatch(sample_config)

                with patch("time.sleep"):
                    with pytest.raises(RuntimeError, match="Batch job failed"):
                        batch.monitor_batch_job(batch_job=mock_batch_job)

    @pytest.mark.batch
    @patch("greycloud.batch.GCS_AVAILABLE", True)
    def test_download_batch_results(self, sample_config, mock_batch_job, tmp_path):
        """Test batch results download"""
        output_file = tmp_path / "results.jsonl"

        mock_predictions_blob = MagicMock()
        mock_predictions_blob.name = (
            "batch_results/prediction-model-123/predictions.jsonl"
        )

        mock_other_blob = MagicMock()
        mock_other_blob.name = "batch_results/other-file.txt"

        with patch("greycloud.batch.create_client") as mock_create:
            mock_create.return_value = MagicMock()

            with patch("greycloud.batch.storage") as mock_storage:
                mock_storage_client = MagicMock()
                mock_bucket = MagicMock()
                mock_storage.Client.return_value = mock_storage_client
                mock_storage_client.bucket.return_value = mock_bucket
                mock_bucket.list_blobs.return_value = [
                    mock_other_blob,
                    mock_predictions_blob,
                ]

                batch = GreyCloudBatch(sample_config)
                result_path = batch.download_batch_results(
                    batch_job=mock_batch_job, output_file=output_file
                )

                assert result_path == output_file
                mock_predictions_blob.download_to_filename.assert_called_once_with(
                    str(output_file)
                )

    @pytest.mark.batch
    @patch("greycloud.batch.GCS_AVAILABLE", True)
    def test_download_batch_results_not_found(self, sample_config, mock_batch_job):
        """Test error when predictions file not found"""
        mock_blob = MagicMock()
        mock_blob.name = "batch_results/other-file.txt"

        with patch("greycloud.batch.create_client") as mock_create:
            mock_create.return_value = MagicMock()

            with patch("greycloud.batch.storage") as mock_storage:
                mock_storage_client = MagicMock()
                mock_bucket = MagicMock()
                mock_storage.Client.return_value = mock_storage_client
                mock_storage_client.bucket.return_value = mock_bucket
                mock_bucket.list_blobs.return_value = [mock_blob]

                batch = GreyCloudBatch(sample_config)

                with pytest.raises(FileNotFoundError, match="predictions.jsonl"):
                    batch.download_batch_results(
                        batch_job=mock_batch_job, output_file="results.jsonl"
                    )

    @pytest.mark.batch
    @patch("greycloud.batch.GCS_AVAILABLE", True)
    def test_upload_file_content_type_detection(self, sample_config, tmp_path):
        """Test automatic content type detection"""
        test_file = tmp_path / "test.json"
        test_file.write_text('{"key": "value"}')

        with patch("greycloud.batch.create_client") as mock_create:
            mock_create.return_value = MagicMock()

            with patch("greycloud.batch.storage") as mock_storage:
                mock_storage_client = MagicMock()
                mock_bucket = MagicMock()
                mock_blob = MagicMock()
                mock_storage.Client.return_value = mock_storage_client
                mock_storage_client.bucket.return_value = mock_bucket
                mock_bucket.blob.return_value = mock_blob

                batch = GreyCloudBatch(sample_config)
                batch.upload_file_to_gcs(file_path=test_file, bucket_name="test-bucket")

                call_args = mock_blob.upload_from_file.call_args
                # Should detect JSON content type
                assert call_args is not None


# ---------------------------------------------------------------------------
# Gemini 4 parity: a single logical config must produce the same verdict
# through every entry path — the sync builder, the async builder, the batch
# dict branch and the batch object branch. Verdicts: ValueError (reject) or
# the same sanitized thinkingLevel / seed in the outgoing payload.
# ---------------------------------------------------------------------------

_G4_MODEL = "gemini-4-argon"

G4_PARITY_CASES = [
    pytest.param(
        {
            "dimension": "thinking_level",
            "dict_cfg": {"thinkingConfig": {"thinkingLevel": "EXTREME"}},
            "obj_cfg": dict(
                thinking_config=types.ThinkingConfig(thinking_level="EXTREME")
            ),
            "builder_kwargs": dict(thinking_level="EXTREME"),
            "expect": ("error", None),
        },
        id="extreme-level",
    ),
    pytest.param(
        {
            "dimension": "thinking_level",
            "dict_cfg": {"thinkingConfig": {"thinkingLevel": "MINIMAL"}},
            "obj_cfg": dict(
                thinking_config=types.ThinkingConfig(thinking_level="MINIMAL")
            ),
            "builder_kwargs": dict(thinking_level="MINIMAL"),
            "expect": ("error", None),
        },
        id="minimal-str",
    ),
    pytest.param(
        {
            "dimension": "thinking_level",
            "dict_cfg": {
                "thinkingConfig": {"thinkingLevel": types.ThinkingLevel.MINIMAL}
            },
            "obj_cfg": dict(
                thinking_config=types.ThinkingConfig(
                    thinking_level=types.ThinkingLevel.MINIMAL
                )
            ),
            "builder_kwargs": dict(
                thinking_config=dict(thinkingLevel=types.ThinkingLevel.MINIMAL)
            ),
            "expect": ("error", None),
        },
        id="minimal-enum",
    ),
    pytest.param(
        {
            "dimension": "thinking_level",
            "dict_cfg": {"thinkingConfig": {"thinkingLevel": "low"}},
            "obj_cfg": dict(thinking_config=types.ThinkingConfig(thinking_level="low")),
            "builder_kwargs": dict(thinking_level="low"),
            "expect": ("ok", "LOW"),
        },
        id="lowercase-low",
    ),
    pytest.param(
        {
            "dimension": "thinking_level",
            "dict_cfg": {"cachedContent": "caches/1"},
            "obj_cfg": dict(cached_content="caches/1"),
            "builder_kwargs": dict(cached_content="caches/1"),
            "expect": ("error", None),
        },
        id="cached-content",
    ),
    pytest.param(
        {
            "dimension": "thinking_level",
            "dict_cfg": {"thinkingConfig": {"thinkingBudget": 512}},
            "obj_cfg": dict(thinking_config=types.ThinkingConfig(thinking_budget=512)),
            "builder_kwargs": dict(
                thinking_config=types.ThinkingConfig(thinking_budget=512)
            ),
            "expect": ("error", None),
        },
        id="thinking-budget",
    ),
    pytest.param(
        {
            "dimension": "seed",
            "dict_cfg": {"seed": 42},
            "obj_cfg": dict(seed=42),
            "builder_kwargs": dict(seed=42),
            "expect": ("ok", 42),
        },
        id="seed-passthrough",
    ),
]


def _make_sync_client(config):
    from greycloud.client import GreyCloudClient

    with patch("greycloud.client.create_client"):
        return GreyCloudClient(config)


def _make_async_client(config):
    from greycloud.async_client import GreyCloudAsyncClient

    with patch("greycloud.async_client.create_client"):
        return GreyCloudAsyncClient(config)


def _builder_verdict(client, case):
    """Verdict of a generate-config builder path."""
    try:
        cfg = client._build_generate_config(model=_G4_MODEL, **case["builder_kwargs"])
    except ValueError as exc:
        return ("error", None, str(exc))
    if case["dimension"] == "thinking_level":
        tc = getattr(cfg, "thinking_config", None)
        level = getattr(tc, "thinking_level", None)
        if level is not None and hasattr(level, "value"):
            level = level.value
        return ("ok", level, None)
    return ("ok", getattr(cfg, "seed", None), None)


def _make_batch(config):
    batch = GreyCloudBatch(config)
    storage_client = MagicMock()
    bucket = MagicMock()
    blob = MagicMock()
    storage_client.bucket.return_value = bucket
    bucket.blob.return_value = blob
    batch._storage_client = storage_client
    return batch, blob


def _request(config_obj):
    return types.InlinedRequest(
        model=_G4_MODEL,
        contents=[{"role": "user", "parts": [{"text": "Test"}]}],
        config=config_obj,
    )


def _dict_branch_request(dict_cfg):
    """A request whose config stays a dict (the batch dict branch).

    newer google-genai coerces plain dicts into GenerateContentConfig when an
    InlinedRequest is built, so the dict branch is forced with a mock whose
    config attribute is the raw dict (same pattern as the sampling-omission
    tests above).
    """
    mock_req = MagicMock()
    mock_req.model = _G4_MODEL
    mock_req.contents = [{"role": "user", "parts": [{"text": "Test"}]}]
    mock_req.config = dict_cfg
    return mock_req


def _batch_verdict(batch, blob, config_obj, force_dict=False):
    """Verdict of one batch upload branch (dict or object config)."""
    if force_dict:
        request = _dict_branch_request(config_obj)
    else:
        request = _request(config_obj)
    try:
        batch.upload_batch_requests_to_gcs([request], bucket_name="test-bucket")
    except ValueError as exc:
        return ("error", None, str(exc))
    uploaded = blob.upload_from_string.call_args[0][0]
    req = json.loads(uploaded.strip().split("\n")[0])["request"]
    gen_cfg = req.get("generationConfig") or {}
    if "generationConfig" not in req or gen_cfg == {}:
        return ("ok", None, None)
    if "seed" in gen_cfg:
        return ("ok", gen_cfg["seed"], None)
    tc = gen_cfg.get("thinkingConfig") or {}
    if tc:
        return ("ok", tc.get("thinkingLevel"), None)
    raise AssertionError(f"unexpected generationConfig {gen_cfg}")


class TestBatchGemini4Parity:
    """Parity between the client builders and both batch upload branches."""

    @pytest.mark.batch
    @pytest.mark.unit
    @pytest.mark.parametrize("case", G4_PARITY_CASES)
    def test_same_verdict_through_all_four_paths(self, sample_config, case):
        sync_client = _make_sync_client(sample_config)
        async_client = _make_async_client(sample_config)
        batch = GreyCloudBatch(sample_config)

        verdicts = {
            "sync_builder": _builder_verdict(sync_client, case),
            "async_builder": _builder_verdict(async_client, case),
        }
        batch, blob = _make_batch(sample_config)
        verdicts["batch_dict_branch"] = _batch_verdict(
            batch, blob, case["dict_cfg"], force_dict=True
        )
        batch2, blob2 = _make_batch(sample_config)
        verdicts["batch_object_branch"] = _batch_verdict(
            batch2, blob2, types.GenerateContentConfig(**case["obj_cfg"])
        )

        dimension, expect_status, expect_value = (
            case["dimension"],
            case["expect"][0],
            case["expect"][1],
        )
        for path, (status, value, msg) in verdicts.items():
            assert (
                status,
                value,
            ) == case["expect"], (
                f"{path} verdict {(status, value)} != expected "
                f'{case["expect"]} (message: {msg})'
            )
        if expect_status == "ok" and dimension == "thinking_level":
            for path, (status, value, _msg) in verdicts.items():
                assert status == "ok"
                assert value == expect_value
        if expect_status == "ok" and dimension == "seed":
            assert verdicts["sync_builder"][1] == 42
            assert verdicts["async_builder"][1] == 42
            assert verdicts["batch_dict_branch"][1] == 42
            assert verdicts["batch_object_branch"][1] == 42
