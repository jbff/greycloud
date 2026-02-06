"""Unit tests for package initialization"""

import pytest
from greycloud import (
    GreyCloudConfig,
    GreyCloudClient,
    GreyCloudBatch,
    GreyCloudCache,
    __version__,
    __all__,
)


class TestPackageInit:
    """Test package initialization and exports"""

    def test_version(self):
        """Test package version"""
        assert __version__ == "0.3.1"

    def test_all_exports(self):
        """Test __all__ exports"""
        expected_exports = [
            "GreyCloudConfig",
            "GreyCloudClient",
            "GreyCloudBatch",
            "GreyCloudCache",
            "VertexRateLimiter",
            "GreyCloudAsyncClient",
        ]
        assert set(__all__) == set(expected_exports)

    def test_config_import(self):
        """Test GreyCloudConfig import"""
        from greycloud import GreyCloudConfig

        assert GreyCloudConfig is not None

    def test_client_import(self):
        """Test GreyCloudClient import"""
        from greycloud import GreyCloudClient

        assert GreyCloudClient is not None

    def test_batch_import(self):
        """Test GreyCloudBatch import"""
        from greycloud import GreyCloudBatch

        assert GreyCloudBatch is not None

    def test_cache_import(self):
        """Test GreyCloudCache import"""
        from greycloud import GreyCloudCache

        assert GreyCloudCache is not None

    def test_auth_import(self):
        """Test auth module import"""
        from greycloud.auth import create_client

        assert create_client is not None

    def test_async_client_exported(self):
        """GreyCloudAsyncClient is importable from greycloud"""
        from greycloud import GreyCloudAsyncClient

        assert GreyCloudAsyncClient is not None

    def test_rate_limiter_exported(self):
        """VertexRateLimiter is importable from greycloud"""
        from greycloud import VertexRateLimiter

        assert VertexRateLimiter is not None
