"""Unit tests for package initialization"""

from pathlib import Path

import pytest

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


class TestWheelShipsPyTyped:
    """F4 guard: a clean-clone (tagged) build must ship greycloud/py.typed.

    The adjudicated finding claimed the wheel omitted the marker; it did not
    reproduce (pyproject declares [tool.setuptools.package-data] and the
    wheel RECORD carries the entry). Kept as a build guard so the package
    config can't silently regress.
    """

    def test_wheel_contains_py_typed(self, tmp_path):
        import shutil
        import subprocess
        import zipfile

        uv = shutil.which("uv")
        if uv is None:
            pytest.skip("uv not available")
        repo_root = Path(__file__).parent.parent
        result = subprocess.run(
            [uv, "build", "--wheel", "-o", str(tmp_path)],
            cwd=repo_root,
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, result.stderr
        wheels = list(tmp_path.glob("greycloud-*.whl"))
        assert len(wheels) == 1
        names = zipfile.ZipFile(wheels[0]).namelist()
        assert "greycloud/py.typed" in names


class TestContractSuiteCollected:
    """F4 guard: the Gemini 4 contract suite must be collectable.

    Adjudication: it already collects and passes (48 tests); kept as a guard
    so a broken contract module fails the suite loudly instead of being
    silently skipped.
    """

    def test_gemini4_contract_module_collects(self):
        import subprocess
        import sys

        repo_root = Path(__file__).parent.parent
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "pytest",
                "tests/test_gemini4_contract.py",
                "--collect-only",
                "-q",
            ],
            cwd=repo_root,
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, result.stderr
        # collected count line, e.g. "48 tests collected in 0.35s"
        assert "tests collected" in result.stdout
        count = int(result.stdout.split(" tests collected")[0].split()[-1])
        assert count > 0


class TestMarkerCoverage:
    """F18: marker selectors must reach the newer suites.

    The four module-level http-options tests in test_auth.py were written
    without @pytest.mark.auth, so `pytest -m auth` skipped them; the Gemini 4
    contract suite carried no marker at all."""

    @staticmethod
    def _collect(expr):
        import subprocess
        import sys

        repo_root = Path(__file__).parent.parent
        result = subprocess.run(
            [sys.executable, "-m", "pytest", "-m", expr, "--co", "-q"],
            cwd=repo_root,
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, result.stderr
        return result.stdout

    def test_auth_marker_reaches_http_options_tests(self):
        collected = self._collect("auth")
        for name in (
            "test_create_client_http_options_timeout_and_api_version",
            "test_create_client_http_options_defaults",
            "test_create_client_http_options_timeout_none",
            "test_create_client_with_api_key_http_options_timeout",
        ):
            assert name in collected, f"{name} not collected under -m auth"

    def test_contract_suite_reachable_by_unit_marker(self):
        collected = self._collect("unit")
        # pytest's -v addopts make --co output a tree; assert the module node.
        assert "test_gemini4_contract.py" in collected
