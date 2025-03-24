"""Basic tests for llama_ipfs package."""

import importlib
import sys

import pytest


def test_import():
    """Test that the package can be imported."""
    try:
        import llama_ipfs

        assert llama_ipfs.__version__ is not None
    except ImportError:
        pytest.fail("Failed to import llama_ipfs")


def test_apply_patch():
    """Test that the apply_patch function exists."""
    import llama_ipfs

    assert hasattr(llama_ipfs, "apply_patch")
    assert callable(llama_ipfs.apply_patch)


def test_is_active():
    """Test that the is_active function exists and can be called."""
    import llama_ipfs

    assert hasattr(llama_ipfs, "is_active")
    assert callable(llama_ipfs.is_active)
    # Just check that it can be called without errors
    llama_ipfs.is_active()


def test_cleanup():
    """Test that cleanup functions exist."""
    import llama_ipfs

    assert hasattr(llama_ipfs, "_cleanup_patch")
    assert callable(llama_ipfs._cleanup_patch)


if __name__ == "__main__":
    # Run the tests
    test_import()
    test_apply_patch()
    test_is_active()
    test_cleanup()
    print("All tests passed!")
