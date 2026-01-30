"""
Batch processing utilities for GreyCloud
"""

import os
import json
import time
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Dict, Any, Union
from google import genai
from google.genai import types

try:
    from google.cloud import storage
    GCS_AVAILABLE = True
except ImportError:
    # Keep a storage name available for tests to patch, even when the
    # google-cloud-storage package is not installed.
    storage = None  # type: ignore[assignment]
    GCS_AVAILABLE = False

from .config import GreyCloudConfig
from .auth import create_client


class GreyCloudBatch:
    """Batch processing client for Vertex AI"""
    
    def __init__(self, config: Optional[GreyCloudConfig] = None):
        """
        Initialize batch processing client
        
        Args:
            config: GreyCloudConfig instance. If None, creates a new one with defaults.
        """
        self.config = config or GreyCloudConfig()
        self._client: Optional[genai.Client] = None
        self._batch_client: Optional[genai.Client] = None
        self._storage_client: Optional[storage.Client] = None
    
    @property
    def client(self) -> genai.Client:
        """Get the authenticated client"""
        if self._client is None:
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
        return self._client
    
    @property
    def batch_client(self) -> genai.Client:
        """Get batch client (uses global location)"""
        if self._batch_client is None:
            self._batch_client = create_client(
                project_id=self.config.project_id,
                location=self.config.batch_location,  # Batch jobs require global
                sa_email=self.config.sa_email,
                use_api_key=self.config.use_api_key,
                api_key_file=self.config.api_key_file,
                endpoint=self.config.endpoint,
                api_version=self.config.api_version,
                auto_reauth=self.config.auto_reauth
            )
        return self._batch_client
    
    @property
    def storage_client(self) -> storage.Client:
        """Get GCS storage client"""
        if not GCS_AVAILABLE:
            raise ImportError(
                "google-cloud-storage is required for batch mode. Install it with: pip install google-cloud-storage"
            )
        
        if self._storage_client is None:
            # Get credentials from the genai client if available
            credentials = None
            if not self.config.use_api_key and self.client:
                # Try to get credentials from the client
                if hasattr(self.client, '_credentials'):
                    credentials = self.client._credentials
            
            if credentials:
                self._storage_client = storage.Client(
                    project=self.config.project_id,
                    credentials=credentials
                )
            else:
                self._storage_client = storage.Client(project=self.config.project_id)
        
        return self._storage_client
    
    def upload_file_to_gcs(
        self,
        file_path: Union[str, Path],
        blob_name: Optional[str] = None,
        bucket_name: Optional[str] = None,
        content_type: Optional[str] = None
    ) -> str:
        """
        Upload a file to GCS
        
        Args:
            file_path: Path to file to upload
            blob_name: GCS blob name (defaults to filename)
            bucket_name: GCS bucket name (defaults to config.gcs_bucket)
            content_type: Content type (auto-detected if not provided)
        
        Returns:
            str: GCS URI (gs://bucket/blob)
        """
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        bucket_name = bucket_name or self.config.gcs_bucket
        if not bucket_name:
            raise ValueError("bucket_name must be provided or set in config.gcs_bucket")
        
        if blob_name is None:
            blob_name = file_path.name
        
        # Auto-detect content type
        if content_type is None:
            import mimetypes
            content_type, _ = mimetypes.guess_type(str(file_path))
            if content_type is None:
                content_type = "application/octet-stream"
        
        # Upload file
        bucket = self.storage_client.bucket(bucket_name)
        blob = bucket.blob(blob_name)
        
        with open(file_path, 'rb') as f:
            blob.upload_from_file(f, content_type=content_type)
        
        gcs_uri = f"gs://{bucket_name}/{blob_name}"
        return gcs_uri
    
    def upload_string_to_gcs(
        self,
        content: str,
        blob_name: str,
        bucket_name: Optional[str] = None,
        content_type: str = "text/plain"
    ) -> str:
        """
        Upload string content to GCS
        
        Args:
            content: String content to upload
            blob_name: GCS blob name
            bucket_name: GCS bucket name (defaults to config.gcs_bucket)
            content_type: Content type
        
        Returns:
            str: GCS URI (gs://bucket/blob)
        """
        bucket_name = bucket_name or self.config.gcs_bucket
        if not bucket_name:
            raise ValueError("bucket_name must be provided or set in config.gcs_bucket")
        
        bucket = self.storage_client.bucket(bucket_name)
        blob = bucket.blob(blob_name)
        blob.upload_from_string(content, content_type=content_type)
        
        gcs_uri = f"gs://{bucket_name}/{blob_name}"
        return gcs_uri
    
    def upload_files_to_gcs(
        self,
        files: List[Dict[str, Any]],
        bucket_name: Optional[str] = None,
        prefix: str = "batch_files"
    ) -> Dict[str, str]:
        """
        Upload multiple files to GCS
        
        Args:
            files: List of file dicts with 'name' and 'content' keys
            bucket_name: GCS bucket name (defaults to config.gcs_bucket)
            prefix: GCS prefix for uploaded files
        
        Returns:
            Dict mapping filename to GCS URI
        """
        bucket_name = bucket_name or self.config.gcs_bucket
        if not bucket_name:
            raise ValueError("bucket_name must be provided or set in config.gcs_bucket")
        
        file_uris = {}
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        for file_data in files:
            filename = file_data.get('name', f"file_{len(file_uris)}")
            content = file_data.get('content', '')
            
            # Determine content type
            import mimetypes
            content_type, _ = mimetypes.guess_type(filename)
            if content_type is None:
                content_type = "text/plain"
            
            blob_name = f"{prefix}/{timestamp}/{filename}"
            gcs_uri = self.upload_string_to_gcs(
                content=content,
                blob_name=blob_name,
                bucket_name=bucket_name,
                content_type=content_type
            )
            
            file_uris[filename] = gcs_uri
        
        return file_uris
    
    def upload_batch_requests_to_gcs(
        self,
        batch_requests: List[types.InlinedRequest],
        bucket_name: Optional[str] = None
    ) -> str:
        """
        Upload batch requests to GCS in JSONL format
        
        Args:
            batch_requests: List of InlinedRequest objects
            bucket_name: GCS bucket name (defaults to config.batch_gcs_bucket)
        
        Returns:
            str: GCS URI of uploaded JSONL file
        """
        bucket_name = bucket_name or self.config.batch_gcs_bucket
        if not bucket_name:
            raise ValueError("bucket_name must be provided or set in config.batch_gcs_bucket")
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        blob_name = f"batch_requests/batch_{timestamp}.jsonl"
        
        # Convert requests to JSONL format
        jsonl_lines = []
        for request in batch_requests:
            # Convert contents to serializable format
            contents_serializable = []
            for content in request.contents:
                if isinstance(content, dict):
                    contents_serializable.append(content)
                else:
                    # Convert Content object to dict
                    content_dict = {"role": content.role}
                    if hasattr(content, 'parts') and content.parts:
                        parts_list = []
                        for part in content.parts:
                            if isinstance(part, dict):
                                parts_list.append(part)
                            elif hasattr(part, 'text'):
                                parts_list.append({"text": part.text})
                            elif hasattr(part, 'file_data'):
                                parts_list.append({"file_data": part.file_data})
                        content_dict["parts"] = parts_list
                    contents_serializable.append(content_dict)
            
            # Convert config to serializable format
            config_serializable = None
            if request.config:
                if isinstance(request.config, dict):
                    config_serializable = request.config
                else:
                    # Convert GenerateContentConfig to dict
                    config_serializable = {}
                    if hasattr(request.config, 'temperature'):
                        config_serializable['temperature'] = request.config.temperature
                    if hasattr(request.config, 'top_p'):
                        config_serializable['top_p'] = request.config.top_p
                    if hasattr(request.config, 'max_output_tokens'):
                        config_serializable['max_output_tokens'] = request.config.max_output_tokens
                    if hasattr(request.config, 'safety_settings'):
                        config_serializable['safety_settings'] = request.config.safety_settings
            
            # Convert InlinedRequest to dict format
            request_dict = {
                "request": {
                    "model": request.model,
                    "contents": contents_serializable,
                }
            }
            if config_serializable:
                request_dict["request"]["config"] = config_serializable
            if hasattr(request, 'metadata') and request.metadata:
                request_dict["request"]["metadata"] = request.metadata
            
            jsonl_lines.append(json.dumps(request_dict))
        
        jsonl_text = "\n".join(jsonl_lines)
        
        # Upload to GCS
        gcs_uri = self.upload_string_to_gcs(
            content=jsonl_text,
            blob_name=blob_name,
            bucket_name=bucket_name,
            content_type="application/jsonl"
        )
        
        return gcs_uri
    
    def create_batch_job(
        self,
        batch_requests: List[types.InlinedRequest],
        model: Optional[str] = None,
        bucket_name: Optional[str] = None,
        results_prefix: Optional[str] = None
    ) -> types.BatchJob:
        """
        Create a batch job
        
        Args:
            batch_requests: List of InlinedRequest objects
            model: Model name (defaults to config.model)
            bucket_name: GCS bucket name for batch I/O
            results_prefix: Prefix for results in GCS (defaults to batch_results/timestamp)
        
        Returns:
            BatchJob object
        """
        model_name = model or self.config.model
        bucket_name = bucket_name or self.config.batch_gcs_bucket
        
        # Upload batch requests to GCS
        gcs_uri = self.upload_batch_requests_to_gcs(batch_requests, bucket_name)
        
        # Create results destination
        if results_prefix is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            results_prefix = f"batch_results/results_{timestamp}.jsonl"
        
        results_gcs_uri = f"gs://{bucket_name}/{results_prefix}"
        
        # Try different model formats
        model_formats_to_try = [
            f'publishers/google/models/{model_name}',  # Full publisher path
            model_name,  # Short name
        ]
        
        batch_job = None
        last_error = None
        
        for model_format in model_formats_to_try:
            try:
                batch_job = self.batch_client.batches.create(
                    model=model_format,
                    src=types.BatchJobSource(
                        gcs_uri=[gcs_uri],
                        format="jsonl",
                    ),
                    config=types.CreateBatchJobConfig(
                        dest=types.BatchJobDestination(
                            gcs_uri=results_gcs_uri,
                            format="jsonl",
                        )
                    )
                )
                break
            except Exception as e:
                last_error = e
                continue
        
        if batch_job is None:
            raise RuntimeError(
                f"Could not create batch job with any model format. Last error: {last_error}"
            )
        
        return batch_job
    
    def monitor_batch_job(
        self,
        batch_job: types.BatchJob,
        poll_interval: Optional[int] = None,
        callback: Optional[callable] = None
    ) -> types.BatchJob:
        """
        Monitor batch job until completion
        
        Args:
            batch_job: BatchJob object to monitor
            poll_interval: Polling interval in seconds (defaults to config.batch_poll_interval)
            callback: Optional callback function called on each poll with (batch_job, state)
        
        Returns:
            Updated BatchJob object
        """
        poll_interval = poll_interval or self.config.batch_poll_interval
        job_name = batch_job.name
        last_state = batch_job.state
        
        # Try different locations for monitoring
        monitor_locations = ["us-central1", "global", self.config.location]
        
        while True:
            time.sleep(poll_interval)
            
            # Try to get batch job from different locations
            batch_job_updated = None
            for location in monitor_locations:
                try:
                    monitor_client = create_client(
                        project_id=self.config.project_id,
                        location=location,
                        sa_email=self.config.sa_email,
                        use_api_key=self.config.use_api_key,
                        api_key_file=self.config.api_key_file,
                        endpoint=self.config.endpoint,
                        api_version=self.config.api_version,
                        auto_reauth=self.config.auto_reauth
                    )
                    batch_job_updated = monitor_client.batches.get(name=job_name)
                    break
                except Exception:
                    continue
            
            if batch_job_updated is None:
                raise RuntimeError(f"Could not retrieve batch job from any location: {job_name}")
            
            batch_job = batch_job_updated
            current_state = batch_job.state
            
            if current_state != last_state:
                last_state = current_state
                if callback:
                    callback(batch_job, current_state)
            
            if batch_job.done:
                if batch_job.error:
                    raise RuntimeError(f"Batch job failed: {batch_job.error}")
                break
        
        return batch_job
    
    def download_batch_results(
        self,
        batch_job: types.BatchJob,
        output_file: Union[str, Path],
        bucket_name: Optional[str] = None
    ) -> Path:
        """
        Download batch job results from GCS
        
        Args:
            batch_job: Completed BatchJob object
            output_file: Path to save results
            bucket_name: GCS bucket name (defaults to config.batch_gcs_bucket)
        
        Returns:
            Path to downloaded file
        """
        output_file = Path(output_file)
        bucket_name = bucket_name or self.config.batch_gcs_bucket
        
        # Get destination from batch job
        dest = batch_job.dest
        if not dest:
            raise ValueError("Batch job has no destination configured")
        
        gcs_uri = getattr(dest, 'gcs_uri', None) or getattr(dest, 'gcsUri', None)
        if not gcs_uri:
            raise ValueError("Batch job destination has no GCS URI")
        
        # Parse GCS URI
        if not gcs_uri.startswith("gs://"):
            raise ValueError(f"Invalid GCS URI: {gcs_uri}")
        
        gcs_path = gcs_uri.replace("gs://", "")
        bucket_name_from_uri, prefix = gcs_path.split("/", 1)
        
        # Use bucket_name from URI if not provided
        if bucket_name is None:
            bucket_name = bucket_name_from_uri
        
        # The GCS URI is a directory, find the actual results file
        # Format: {prefix}/prediction-model-{timestamp}/predictions.jsonl
        bucket = self.storage_client.bucket(bucket_name)
        
        # List blobs with the prefix
        blobs = list(bucket.list_blobs(prefix=prefix))
        
        # Find predictions.jsonl file
        predictions_blob = None
        for blob in blobs:
            if blob.name.endswith("predictions.jsonl"):
                predictions_blob = blob
                break
        
        if predictions_blob is None:
            raise FileNotFoundError(
                f"predictions.jsonl not found in {gcs_uri}. "
                f"The batch job may have completed but results are still being written."
            )
        
        # Download the file
        predictions_blob.download_to_filename(str(output_file))
        
        return output_file
