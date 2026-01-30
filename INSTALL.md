# Installation Guide

## Installing as a Package

### From Local Directory

To install GreyCloud in development mode (editable install):

```bash
# Basic installation (with runtime deps)
pip install -e .

# With development dependencies
pip install -e ".[dev]"

# Everything (runtime + dev extras)
pip install -e ".[all]"
```

### Building a Distribution

To build source and wheel distributions:

```bash
# Install build tools
pip install build

# Build distributions
python -m build

# This creates:
# - dist/greycloud-0.1.0.tar.gz (source distribution)
# - dist/greycloud-0.1.0-py3-none-any.whl (wheel)
```

### Installing from Built Distribution

```bash
# From wheel (recommended)
pip install dist/greycloud-0.1.0-py3-none-any.whl

# From source distribution
pip install dist/greycloud-0.1.0.tar.gz
```

### Installing from Git Repository

```bash
# Install directly from git
pip install git+https://github.com/jbff/greycloud.git

# Install with specific branch/tag
pip install git+https://github.com/jbff/greycloud.git@v0.1.0
```

## Development Setup

1. Clone the repository:
```bash
git clone https://github.com/jbff/greycloud.git
cd greycloud
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install in development mode:
```bash
pip install -e ".[dev]"
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
