"""Unit tests for GreyCloudBatch"""

import json
import pytest
from unittest.mock import patch, MagicMock, mock_open
from pathlib import Path
from datetime import datetime
from google.genai import types
from greycloud.batch import GreyCloudBatch, GCS_AVAILABLE
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
        with patch.object(subprocess, 'run') as mock_run:
            mock_run.return_value.returncode = 0
            mock_run.return_value.stdout = "test-project"
            batch = GreyCloudBatch()
            assert batch.config is not None
    
    def test_client_property(self, sample_config):
        """Test client property"""
        with patch('greycloud.batch.create_client') as mock_create:
            mock_client = MagicMock()
            mock_create.return_value = mock_client
            
            batch = GreyCloudBatch(sample_config)
            client = batch.client
            
            assert client == mock_client
            mock_create.assert_called_once()
    
    def test_batch_client_property(self, sample_config):
        """Test batch_client property"""
        with patch('greycloud.batch.create_client') as mock_create:
            mock_client = MagicMock()
            mock_create.return_value = mock_client
            
            batch = GreyCloudBatch(sample_config)
            batch_client = batch.batch_client
            
            assert batch_client == mock_client
            # Should use batch_location (global)
            call_kwargs = mock_create.call_args[1]
            assert call_kwargs["location"] == "global"
    
    @pytest.mark.batch
    @patch('greycloud.batch.GCS_AVAILABLE', True)
    def test_storage_client_property(self, sample_config):
        """Test storage_client property"""
        with patch('greycloud.batch.create_client') as mock_create:
            mock_genai_client = MagicMock()
            mock_create.return_value = mock_genai_client
            
            with patch('greycloud.batch.storage') as mock_storage:
                mock_storage_client = MagicMock()
                mock_storage.Client.return_value = mock_storage_client
                
                batch = GreyCloudBatch(sample_config)
                storage_client = batch.storage_client
                
                assert storage_client == mock_storage_client
    
    @pytest.mark.batch
    @patch('greycloud.batch.GCS_AVAILABLE', False)
    def test_storage_client_not_available(self, sample_config):
        """Test error when GCS not available"""
        batch = GreyCloudBatch(sample_config)
        
        with pytest.raises(ImportError, match="google-cloud-storage"):
            _ = batch.storage_client
    
    @pytest.mark.batch
    @patch('greycloud.batch.GCS_AVAILABLE', True)
    def test_upload_file_to_gcs(self, sample_config, tmp_path):
        """Test file upload to GCS"""
        test_file = tmp_path / "test.txt"
        test_file.write_text("Test content")
        
        with patch('greycloud.batch.create_client') as mock_create:
            mock_create.return_value = MagicMock()
            
            with patch('greycloud.batch.storage') as mock_storage:
                mock_storage_client = MagicMock()
                mock_bucket = MagicMock()
                mock_blob = MagicMock()
                mock_storage.Client.return_value = mock_storage_client
                mock_storage_client.bucket.return_value = mock_bucket
                mock_bucket.blob.return_value = mock_blob
                
                batch = GreyCloudBatch(sample_config)
                gcs_uri = batch.upload_file_to_gcs(
                    file_path=test_file,
                    blob_name="test.txt",
                    bucket_name="test-bucket"
                )
                
                assert gcs_uri == "gs://test-bucket/test.txt"
                mock_blob.upload_from_file.assert_called_once()
    
    @pytest.mark.batch
    @patch('greycloud.batch.GCS_AVAILABLE', True)
    def test_upload_file_to_gcs_file_not_found(self, sample_config):
        """Test error when file not found"""
        with patch('greycloud.batch.create_client') as mock_create:
            mock_create.return_value = MagicMock()
            
            with patch('greycloud.batch.storage'):
                batch = GreyCloudBatch(sample_config)
                
                with pytest.raises(FileNotFoundError):
                    batch.upload_file_to_gcs(
                        file_path="nonexistent.txt",
                        bucket_name="test-bucket"
                    )
    
    @pytest.mark.batch
    @patch('greycloud.batch.GCS_AVAILABLE', True)
    def test_upload_file_to_gcs_default_bucket(self, sample_config, tmp_path):
        """Test file upload with default bucket from config"""
        test_file = tmp_path / "test.txt"
        test_file.write_text("Test content")
        
        config = GreyCloudConfig(project_id="test-project", gcs_bucket="default-bucket")
        
        with patch('greycloud.batch.create_client') as mock_create:
            mock_create.return_value = MagicMock()
            
            with patch('greycloud.batch.storage') as mock_storage:
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
    @patch('greycloud.batch.GCS_AVAILABLE', True)
    def test_upload_string_to_gcs(self, sample_config):
        """Test string upload to GCS"""
        with patch('greycloud.batch.create_client') as mock_create:
            mock_create.return_value = MagicMock()
            
            with patch('greycloud.batch.storage') as mock_storage:
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
                    bucket_name="test-bucket"
                )
                
                assert gcs_uri == "gs://test-bucket/test.txt"
                mock_blob.upload_from_string.assert_called_once_with(
                    "Test content",
                    content_type="text/plain"
                )
    
    @pytest.mark.batch
    @patch('greycloud.batch.GCS_AVAILABLE', True)
    def test_upload_files_to_gcs(self, sample_config):
        """Test multiple file upload"""
        files = [
            {"name": "file1.txt", "content": "Content 1"},
            {"name": "file2.txt", "content": "Content 2"}
        ]
        
        with patch('greycloud.batch.create_client') as mock_create:
            mock_create.return_value = MagicMock()
            
            with patch('greycloud.batch.storage') as mock_storage:
                mock_storage_client = MagicMock()
                mock_bucket = MagicMock()
                mock_blob = MagicMock()
                mock_storage.Client.return_value = mock_storage_client
                mock_storage_client.bucket.return_value = mock_bucket
                mock_bucket.blob.return_value = mock_blob
                
                batch = GreyCloudBatch(sample_config)
                file_uris = batch.upload_files_to_gcs(
                    files=files,
                    bucket_name="test-bucket"
                )
                
                assert len(file_uris) == 2
                assert "file1.txt" in file_uris
                assert "file2.txt" in file_uris
                assert mock_blob.upload_from_string.call_count == 2
    
    @pytest.mark.batch
    @patch('greycloud.batch.GCS_AVAILABLE', True)
    def test_upload_batch_requests_to_gcs(self, sample_config):
        """Test batch requests upload"""
        batch_requests = [
            types.InlinedRequest(
                model="gemini-3-pro-preview",
                contents=[{"role": "user", "parts": [{"text": "Test"}]}],
                config=types.GenerateContentConfig(temperature=0.2)
            )
        ]
        
        with patch('greycloud.batch.create_client') as mock_create:
            mock_create.return_value = MagicMock()
            
            with patch('greycloud.batch.storage') as mock_storage:
                mock_storage_client = MagicMock()
                mock_bucket = MagicMock()
                mock_blob = MagicMock()
                mock_storage.Client.return_value = mock_storage_client
                mock_storage_client.bucket.return_value = mock_bucket
                mock_bucket.blob.return_value = mock_blob
                
                batch = GreyCloudBatch(sample_config)
                gcs_uri = batch.upload_batch_requests_to_gcs(
                    batch_requests=batch_requests,
                    bucket_name="test-bucket"
                )
                
                assert gcs_uri.startswith("gs://test-bucket/batch_requests/")
                mock_blob.upload_from_string.assert_called_once()
                
                # Verify JSONL format
                call_args = mock_blob.upload_from_string.call_args
                uploaded_content = call_args[0][0]
                assert uploaded_content.startswith('{"request":')
    
    @pytest.mark.batch
    @patch('greycloud.batch.GCS_AVAILABLE', True)
    def test_create_batch_job(self, sample_config):
        """Test batch job creation"""
        batch_requests = [
            types.InlinedRequest(
                model="gemini-3-pro-preview",
                contents=[{"role": "user", "parts": [{"text": "Test"}]}],
                config=types.GenerateContentConfig()
            )
        ]
        
        mock_batch_job = MagicMock(spec=types.BatchJob)
        mock_batch_job.name = "test-batch-job"
        
        with patch('greycloud.batch.create_client') as mock_create:
            mock_batch_client = MagicMock()
            mock_batch_client.batches.create.return_value = mock_batch_job
            mock_create.return_value = mock_batch_client
            
            with patch('greycloud.batch.storage'):
                batch = GreyCloudBatch(sample_config)
                
                with patch.object(batch, 'upload_batch_requests_to_gcs', return_value="gs://bucket/requests.jsonl"):
                    job = batch.create_batch_job(
                        batch_requests=batch_requests,
                        bucket_name="test-bucket"
                    )
                    
                    assert job == mock_batch_job
                    mock_batch_client.batches.create.assert_called_once()
    
    @pytest.mark.batch
    @patch('greycloud.batch.GCS_AVAILABLE', True)
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
        
        with patch('greycloud.batch.create_client') as mock_create:
            mock_client = MagicMock()
            mock_client.batches.get.side_effect = [
                mock_batch_job,
                mock_batch_job_complete
            ]
            mock_create.return_value = mock_client
            
            with patch('greycloud.batch.storage'):
                batch = GreyCloudBatch(sample_config)
                
                callback_calls = []
                def callback(job, state):
                    callback_calls.append((job, state))
                
                with patch('time.sleep'):  # Speed up test
                    result = batch.monitor_batch_job(
                        batch_job=mock_batch_job,
                        poll_interval=1,
                        callback=callback
                    )
                
                assert result.done is True
                assert len(callback_calls) >= 1
    
    @pytest.mark.batch
    @patch('greycloud.batch.GCS_AVAILABLE', True)
    def test_monitor_batch_job_failure(self, sample_config, mock_batch_job):
        """Test batch job monitoring with failure"""
        mock_batch_job.done = True
        mock_batch_job.error = MagicMock()
        mock_batch_job.error.message = "Job failed"
        
        with patch('greycloud.batch.create_client') as mock_create:
            mock_client = MagicMock()
            mock_client.batches.get.return_value = mock_batch_job
            mock_create.return_value = mock_client
            
            with patch('greycloud.batch.storage'):
                batch = GreyCloudBatch(sample_config)
                
                with patch('time.sleep'):
                    with pytest.raises(RuntimeError, match="Batch job failed"):
                        batch.monitor_batch_job(batch_job=mock_batch_job)
    
    @pytest.mark.batch
    @patch('greycloud.batch.GCS_AVAILABLE', True)
    def test_download_batch_results(self, sample_config, mock_batch_job, tmp_path):
        """Test batch results download"""
        output_file = tmp_path / "results.jsonl"
        
        mock_predictions_blob = MagicMock()
        mock_predictions_blob.name = "batch_results/prediction-model-123/predictions.jsonl"
        
        mock_other_blob = MagicMock()
        mock_other_blob.name = "batch_results/other-file.txt"
        
        with patch('greycloud.batch.create_client') as mock_create:
            mock_create.return_value = MagicMock()
            
            with patch('greycloud.batch.storage') as mock_storage:
                mock_storage_client = MagicMock()
                mock_bucket = MagicMock()
                mock_storage.Client.return_value = mock_storage_client
                mock_storage_client.bucket.return_value = mock_bucket
                mock_bucket.list_blobs.return_value = [mock_other_blob, mock_predictions_blob]
                
                batch = GreyCloudBatch(sample_config)
                result_path = batch.download_batch_results(
                    batch_job=mock_batch_job,
                    output_file=output_file
                )
                
                assert result_path == output_file
                mock_predictions_blob.download_to_filename.assert_called_once_with(str(output_file))
    
    @pytest.mark.batch
    @patch('greycloud.batch.GCS_AVAILABLE', True)
    def test_download_batch_results_not_found(self, sample_config, mock_batch_job):
        """Test error when predictions file not found"""
        mock_blob = MagicMock()
        mock_blob.name = "batch_results/other-file.txt"
        
        with patch('greycloud.batch.create_client') as mock_create:
            mock_create.return_value = MagicMock()
            
            with patch('greycloud.batch.storage') as mock_storage:
                mock_storage_client = MagicMock()
                mock_bucket = MagicMock()
                mock_storage.Client.return_value = mock_storage_client
                mock_storage_client.bucket.return_value = mock_bucket
                mock_bucket.list_blobs.return_value = [mock_blob]
                
                batch = GreyCloudBatch(sample_config)
                
                with pytest.raises(FileNotFoundError, match="predictions.jsonl"):
                    batch.download_batch_results(
                        batch_job=mock_batch_job,
                        output_file="results.jsonl"
                    )
    
    @pytest.mark.batch
    @patch('greycloud.batch.GCS_AVAILABLE', True)
    def test_upload_file_content_type_detection(self, sample_config, tmp_path):
        """Test automatic content type detection"""
        test_file = tmp_path / "test.json"
        test_file.write_text('{"key": "value"}')
        
        with patch('greycloud.batch.create_client') as mock_create:
            mock_create.return_value = MagicMock()
            
            with patch('greycloud.batch.storage') as mock_storage:
                mock_storage_client = MagicMock()
                mock_bucket = MagicMock()
                mock_blob = MagicMock()
                mock_storage.Client.return_value = mock_storage_client
                mock_storage_client.bucket.return_value = mock_bucket
                mock_bucket.blob.return_value = mock_blob
                
                batch = GreyCloudBatch(sample_config)
                batch.upload_file_to_gcs(
                    file_path=test_file,
                    bucket_name="test-bucket"
                )
                
                call_args = mock_blob.upload_from_file.call_args
                # Should detect JSON content type
                assert call_args is not None
