"""
Llama-CPP-Python IPFS Integration

Enable IPFS model loading for llama-cpp-python's Llama.from_pretrained method.
This module is automatically loaded at Python startup and patches llama_cpp.
"""

import atexit
import builtins
import sys

from ._version import __version__
from .patch import _ORIGINAL_IMPORT, _PATCHED_METHODS, CLI_MODE, apply_patch, is_active

# Set CLI_MODE if we're called from the CLI command
script_name = sys.argv[0].lower() if len(sys.argv) > 0 else ""
if "llama-ipfs" in script_name or "llama_ipfs" in script_name:
    CLI_MODE = True


# Cleanup function to restore original functionality when Python exits
def _cleanup_patch():
    """Restore original import function and methods when Python exits."""
    # Only restore if we modified it
    if hasattr(builtins.__import__, "_llama_ipfs_hook"):
        builtins.__import__ = _ORIGINAL_IMPORT

    # Restore patched methods if llama_cpp is loaded
    if "llama_cpp" in sys.modules:
        try:
            llama_module = sys.modules["llama_cpp"]
            if hasattr(llama_module, "Llama"):
                llama_class = llama_module.Llama
                # If we have stored the original from_pretrained method, restore it
                if "llama_from_pretrained" in _PATCHED_METHODS:
                    original = _PATCHED_METHODS["llama_from_pretrained"]
                    if hasattr(llama_class, "from_pretrained"):
                        llama_class.from_pretrained = classmethod(original)
        except Exception:
            # Silently fail during cleanup
            pass


# Register cleanup function
atexit.register(_cleanup_patch)

# Set up the import hook immediately on import
# This will defer actual patching until llama_cpp is fully loaded
# Don't show the general loading message (only show model-specific messages when used)
apply_patch(show_message=False)
