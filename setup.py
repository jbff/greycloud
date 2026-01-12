"""Setup script for backward compatibility with pip"""

from setuptools import setup

# This file exists for backward compatibility
# Modern Python packages should use pyproject.toml
# This file will be ignored if pyproject.toml is present

if __name__ == "__main__":
    setup()
