"""
GreyCloud - Reusable Google GenAI/Vertex AI Client Module

A comprehensive, configurable module for interacting with Google's Vertex AI
and GenAI services, including authentication, content generation, batch processing,
and file management.
"""

from .config import GreyCloudConfig
from .client import GreyCloudClient
from .batch import GreyCloudBatch

__version__ = "0.1.0"
__all__ = ["GreyCloudConfig", "GreyCloudClient", "GreyCloudBatch"]
