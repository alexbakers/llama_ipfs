"""
Setup script for llama_ipfs package.
Note: The PyPI package name uses hyphens (llama-ipfs), but the Python module uses underscores (llama_ipfs).
"""

from setuptools import setup

# Read the contents of the README file
with open("README.md", encoding="utf-8") as f:
    long_description = f.read()

# Use a minimal setup.py for compatibility
# Dependencies are defined in pyproject.toml
setup(
    long_description=long_description,
    long_description_content_type="text/markdown",
) 