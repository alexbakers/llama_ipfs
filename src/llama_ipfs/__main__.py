#!/usr/bin/env python3
"""
Command line interface for llama_ipfs.

This module provides CLI commands to install, uninstall, and check the status
of the llama_ipfs package.
"""

import argparse
import builtins
import importlib.util
import os
import site
import sys
from pathlib import Path

from ._version import __version__

# Import patch module but suppress the initial message by temporarily redirecting stdout
_original_print = builtins.print
builtins.print = lambda *args, **kwargs: None
from .patch import _PATCH_MESSAGE_PRINTED, apply_patch, is_active

# Restore normal printing
builtins.print = _original_print


def create_pth_file() -> bool:
    """
    Create a .pth file in the site-packages directory to automatically
    load the llama_ipfs module when Python starts.

    Returns:
        bool: True if the file was created successfully, False otherwise.
    """
    import_line = "import llama_ipfs; llama_ipfs.apply_patch()"
    pth_name = "llama_ipfs.pth"  # Use same name as the module for consistency

    # Standard site-packages directory
    std_site_packages = site.getsitepackages()[0]
    user_site_packages = site.getusersitepackages()

    # Try to create .pth file in standard site-packages first
    try:
        pth_path = os.path.join(std_site_packages, pth_name)
        with open(pth_path, "w") as f:
            f.write(import_line)
        print(f"✅ Created {pth_path}")
        print("\n🎉 llama-ipfs integration activated successfully!")
        print("   It will be automatically loaded in all Python sessions.")
        return True
    except (PermissionError, OSError):
        print(
            f"🔶 Could not create .pth file in system site-packages: {std_site_packages}"
        )
        print("   Attempting to use user site-packages instead...")

    # Fall back to user site-packages
    try:
        # Ensure user site-packages directory exists
        os.makedirs(user_site_packages, exist_ok=True)
        pth_path = os.path.join(user_site_packages, pth_name)
        with open(pth_path, "w") as f:
            f.write(import_line)
        print(f"✅ Created {pth_path}")
        print("\n🎉 llama-ipfs integration activated successfully!")
        print("   It will be automatically loaded in all Python sessions.")
        return True
    except (PermissionError, OSError) as e:
        print(f"❌ Failed to create .pth file: {e}")
        print("   You may need to run the command with administrator privileges.")
        return False


def remove_pth_file() -> bool:
    """
    Remove the .pth file that automatically loads the llama_ipfs module.

    Returns:
        bool: True if the file was removed successfully, False otherwise.
    """
    pth_name = "llama_ipfs.pth"

    # Find all site-packages directories
    site_packages_dirs = site.getsitepackages() + [site.getusersitepackages()]
    any_removed = False

    for site_pkg in site_packages_dirs:
        pth_path = os.path.join(site_pkg, pth_name)
        if os.path.exists(pth_path):
            try:
                os.remove(pth_path)
                print(f"✅ Removed {pth_path}")
                any_removed = True
            except (PermissionError, OSError) as e:
                print(f"❌ Could not remove {pth_path}: {e}")

    if any_removed:
        print("\n🎉 llama-ipfs integration deactivated successfully!")
        return True
    else:
        print("\n❌ No llama-ipfs integration files found to deactivate.")
        return False


def find_pth_files():
    """
    Find all .pth files for llama_ipfs in site-packages directories.

    Returns:
        list: List of paths to .pth files.
    """
    pth_name = "llama_ipfs.pth"
    pth_files = []

    # Check all site-packages directories
    site_packages_dirs = site.getsitepackages() + [site.getusersitepackages()]
    for site_pkg in site_packages_dirs:
        pth_path = os.path.join(site_pkg, pth_name)
        if os.path.exists(pth_path):
            pth_files.append(pth_path)

    return pth_files


def status_command() -> bool:
    """
    Check if the llama_ipfs module is properly installed and activated.

    Returns:
        bool: True if the module is properly installed, False otherwise.
    """
    pth_files = find_pth_files()

    # Check for .pth files
    if pth_files:
        print(f"✅ llama-ipfs integration is activated")
        print(f"   Version: {__version__}")
        print("   Activation files:")
        for pth_file in pth_files:
            print(f"   - {pth_file}")
        return True
    else:
        print("❌ llama-ipfs integration is not activated")
        print("   Run 'llama-ipfs activate' to activate it")
        return False


def test_command():
    """Test if llama_ipfs is working properly by loading a small model."""
    # Check if llama-cpp-python is installed
    if importlib.util.find_spec("llama_cpp") is None:
        print("❌ llama-cpp-python is not installed")
        print("   Please install it with: pip install llama-cpp-python")
        return False

    print(f"✅ llama-ipfs version: {__version__}")

    # Try to import and see if our patch is active
    import llama_cpp

    if is_active():
        print("✅ llama-ipfs integration is active")
    else:
        print("❌ llama-ipfs integration is not active")
        print("   Run 'llama-ipfs activate' to enable it")
        return False

    print("\nTesting IPFS functionality...")
    print("Attempting to load a small model from IPFS...")

    try:
        # Try loading a model with IPFS support
        model = llama_cpp.Llama.from_pretrained(
            repo_id="ipfs://bafybeie7quk74kmqg34nl2ewdwmsrlvvt6heayien364gtu2x6g2qpznhq",
            filename="ggml-model-Q4_K_M.gguf",
            verbose=False,
        )

        print("🎉 llama-ipfs is working correctly.")
        return True
    except Exception as e:
        print(f"❌ Error loading from IPFS: {e}")
        print("   This may indicate a problem with the llama-ipfs integration.")
        return False


def main():
    """Main entry point for the llama-ipfs CLI."""
    parser = argparse.ArgumentParser(
        description="llama-ipfs: Enable IPFS model loading for llama-cpp-python"
    )
    parser.add_argument(
        "--version", action="version", version=f"llama-ipfs {__version__}"
    )

    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # Activate command
    activate_parser = subparsers.add_parser(
        "activate", help="Enable auto-loading of llama_ipfs (note: CLI uses hyphens)"
    )

    # Deactivate command
    deactivate_parser = subparsers.add_parser(
        "deactivate", help="Disable auto-loading of llama_ipfs (note: CLI uses hyphens)"
    )

    # Status command
    status_parser = subparsers.add_parser(
        "status", help="Check if llama_ipfs is properly installed"
    )

    # Test command
    test_parser = subparsers.add_parser(
        "test", help="Test if llama_ipfs is working properly"
    )

    # Check command (alias for status)
    check_parser = subparsers.add_parser("check", help="Alias for 'status'")

    args = parser.parse_args()

    # Default to status if no command is provided
    if not args.command:
        return main_status()

    # Execute the appropriate command
    if args.command == "activate":
        return 0 if create_pth_file() else 1
    elif args.command == "deactivate":
        return 0 if remove_pth_file() else 1
    elif args.command in ["status", "check"]:
        return 0 if status_command() else 1
    elif args.command == "test":
        test_command()
        return 0


def main_status():
    """Run the status command directly (called by default with no args)."""
    return 0 if status_command() else 1


if __name__ == "__main__":
    sys.exit(main())
