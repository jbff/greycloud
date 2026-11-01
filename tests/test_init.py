"""Unit tests for package initialization"""

from greycloud import __version__, __all__


class TestPackageInit:
    """Test package initialization and exports"""

    def test_version(self):
        """Test package version"""
        assert __version__ == "0.5.0"

    def test_py_typed_marker_is_packaged(self):
        """PEP 561 marker ships with the package (declared in pyproject)."""
        from importlib.resources import files

        assert files("greycloud").joinpath("py.typed").is_file()

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


def test_google_genai_dependency_spec():
    import tomllib

    with open("pyproject.toml", "rb") as f:
        data = tomllib.load(f)
    deps = data["project"]["dependencies"]
    genai_dep = next(d for d in deps if d.startswith("google-genai"))
    assert ">=2.34.0" in genai_dep
    assert "<3.0.0" in genai_dep


class TestCILockGate:
    """F3: the CI test job must gate the uv lock and install it frozen.

    pyproject.toml's google-genai floor is part of the Gemini 4 contract;
    a CI job that pip-installs from a stale lock would silently resolve
    below it. The published CI example (TESTING.md) therefore must run
    `uv lock --check` and install with `uv sync --frozen`.
    """

    def _ci_example(self):
        from pathlib import Path

        testing_md = Path(__file__).parent.parent / "TESTING.md"
        return testing_md.read_text()

    def test_ci_example_gates_the_lock(self):
        example = self._ci_example()
        assert "uv lock --check" in example
        assert "uv sync --frozen" in example
