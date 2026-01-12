# Installation Guide

## Installing as a Package

### From Local Directory

To install GreyCloud in development mode (editable install):

```bash
# Basic installation
pip install -e .

# With GCS support
pip install -e ".[storage]"

# With development dependencies
pip install -e ".[dev]"

# Everything
pip install -e ".[storage,dev]"
```

### Building a Distribution

To build source and wheel distributions:

```bash
# Install build tools
pip install build

# Build distributions
python -m build

# This creates:
# - dist/greycloud-1.0.0.tar.gz (source distribution)
# - dist/greycloud-1.0.0-py3-none-any.whl (wheel)
```

### Installing from Built Distribution

```bash
# From wheel (recommended)
pip install dist/greycloud-1.0.0-py3-none-any.whl

# From source distribution
pip install dist/greycloud-1.0.0.tar.gz
```

### Installing from Git Repository

```bash
# Install directly from git
pip install git+https://github.com/yourusername/greycloud.git

# Install with specific branch/tag
pip install git+https://github.com/yourusername/greycloud.git@v1.0.0

# Install with extras
pip install "git+https://github.com/yourusername/greycloud.git#egg=greycloud[storage]"
```

## Development Setup

1. Clone the repository:
```bash
git clone https://github.com/yourusername/greycloud.git
cd greycloud
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install in development mode:
```bash
pip install -e ".[storage,dev]"
```

4. Run tests:
```bash
pytest
```

5. Run tests with coverage:
```bash
pytest --cov=greycloud --cov-report=html
```

## Verifying Installation

After installation, verify it works:

```python
python -c "from greycloud import GreyCloudConfig, GreyCloudClient, GreyCloudBatch; print('GreyCloud installed successfully!')"
```

## Uninstalling

```bash
pip uninstall greycloud
```
