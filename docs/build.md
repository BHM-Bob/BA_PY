# Building and Packaging mbapy

This guide explains how to build and package the mbapy library for distribution.

## Prerequisites

Ensure you have the following tools installed:

```bash
pip install build twine
```

## Building the Package

### Method 1: Using build (Recommended)

```bash
python -m build
```

This command generates both source distribution and wheel packages:
- `dist/mbapy-x.x.x.tar.gz` - Source distribution
- `dist/mbapy-x.x.x-py3-none-any.whl` - Wheel package

### Method 2: Using setuptools (Legacy)

```bash
# Source distribution only
python -m setup.py sdist

# Wheel package only
python -m setup.py bdist_wheel
```

### Method 3: Using pip

```bash
pip wheel . -w dist/
```

## Package Structure

The mbapy package includes two main subpackages:
- `mbapy` - Main package with comprehensive utilities
- `mbapy_lite` - Lightweight version with essential features

Both packages are automatically included in the build process.

## Verifying the Build

After building, you can verify the package contents:

```bash
# Check source package contents
tar -tzf dist/mbapy-*.tar.gz

# Test installation
pip install dist/mbapy-*.whl --force-reinstall
```

## Publishing to PyPI

To upload your package to PyPI:

```bash
# Upload to PyPI
twine upload dist/*

# Upload to TestPyPI (for testing)
twine upload --repository testpypi dist/*
```

## Development Installation

For development purposes, you can install the package in editable mode:

```bash
pip install -e .
```

## Package Configuration

The package configuration is defined in `pyproject.toml` and includes:
- Package metadata and dependencies
- Build system configuration
- Package discovery for both mbapy and mbapy_lite
- Entry points for command-line tools

## Troubleshooting

If you encounter issues:
1. Ensure all required dependencies are installed
2. Check that `pyproject.toml` is properly formatted
3. Verify that both `mbapy` and `mbapy_lite` directories contain valid Python packages
4. Clear build artifacts if needed: `rm -rf build/ dist/`