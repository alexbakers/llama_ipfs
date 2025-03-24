"""
Llama-cpp-python patch implementation.

This module contains the core functionality to patch llama-cpp-python
Llama.from_pretrained method to display information when loading models.
"""

import builtins
import fnmatch
import functools
import inspect
import os
import shutil
import subprocess
import sys
import tarfile
import tempfile
import time
import traceback
from typing import Any, Callable, Dict, List, Optional, Set, Type, Union
from urllib.parse import urlparse

import requests

# Track whether we've patched llama_cpp yet
_LLAMA_CPP_PATCHED = False

# Track whether we've printed the patch loaded message
_PATCH_MESSAGE_PRINTED = False

# Flag to check if we're running from the CLI
CLI_MODE = False

# Store the original import function - keeping a private reference
_ORIGINAL_IMPORT = builtins.__import__

# Track models we've already announced to avoid duplication
_ANNOUNCED_MODELS: Set[str] = set()

# Flag to detect if we're in the middle of an import
_CURRENTLY_IMPORTING = False

# Store the original from_pretrained for when we patch it
_ORIGINAL_FROM_PRETRAINED = None

# Dictionary to track patched methods for cleanup
_PATCHED_METHODS: Dict[str, Callable] = {}

# Cache directory for IPFS models
IPFS_CACHE_DIR = os.path.join(os.path.expanduser("~"), ".cache", "ipfs", "models")
os.makedirs(IPFS_CACHE_DIR, exist_ok=True)


def is_active():
    """Check if the llama_ipfs integration is active."""
    # First check for the import hook
    if hasattr(builtins.__import__, "_llama_ipfs_hook"):
        # Then check for specific patched methods in llama_cpp
        try:
            import llama_cpp

            if hasattr(llama_cpp, "Llama") and hasattr(
                llama_cpp.Llama, "from_pretrained"
            ):
                return True
        except (ImportError, AttributeError):
            pass

    # Check for _PATCHED_METHODS as fallback
    return bool(_PATCHED_METHODS)


def is_transformers_context():
    """Check if we're being called from transformers-related code.

    Returns:
        True if called from transformers context, False otherwise
    """
    # Check for the environment variable first (fastest check)
    if os.environ.get("TRANSFORMERS_PATCH_ACTIVE", "0") == "1":
        return True

    # Check command line arguments for transformers-patch CLI usage
    try:
        script_name = sys.argv[0].lower() if len(sys.argv) > 0 else ""
        if "transformers-patch" in script_name or "transformers_patch" in script_name:
            return True

        # Check for executable script arguments that might contain transformers-patch
        for arg in sys.argv:
            if (
                "transformers-patch" in arg.lower()
                or "transformers_patch" in arg.lower()
            ):
                return True
    except Exception:
        pass

    # Check the call stack as a fallback
    try:
        # Get the call stack
        stack = traceback.extract_stack()
        # Check each frame in the stack
        for frame in stack:
            frame_file = frame.filename.lower()
            # Check if any frame involves transformers code
            if "transformers_patch" in frame_file or "transformers/" in frame_file:
                return True
    except Exception:
        # If we can't determine, better to be conservative
        return False
    return False


def get_class_name(cls_or_obj: Any) -> str:
    """Get the class name from a class or object.

    Args:
        cls_or_obj: Class or object to get name from

    Returns:
        The class name as a string
    """
    if isinstance(cls_or_obj, type):
        return cls_or_obj.__name__
    try:
        return cls_or_obj.__class__.__name__
    except AttributeError:
        return "Unknown"


def download_from_ipfs(
    cid: str, filename: Optional[str] = None, cache_dir: Optional[str] = None
) -> str:
    """Download a model or file from IPFS.

    Args:
        cid: IPFS CID
        filename: Optional specific filename to download
        cache_dir: Directory to store cached models

    Returns:
        Path to the downloaded model directory or file
    """
    if cache_dir is None:
        cache_dir = IPFS_CACHE_DIR

    # Create model directory in cache
    model_dir = os.path.join(cache_dir, cid)

    # If we need a specific file and it exists, return it
    if filename and not filename.startswith("*"):
        file_path = os.path.join(model_dir, filename)
        if (
            os.path.exists(file_path)
            and os.path.isfile(file_path)
            and os.path.getsize(file_path) > 0
        ):
            print(f"🔄 Using cached file: {file_path}")
            # Ensure we handle specific requirements for llama-cpp models
            ensure_llama_model_requirements(model_dir, filename)
            return file_path

    # If model already exists in cache and we don't need a specific file or need a wildcard match, use it
    if os.path.exists(model_dir) and os.path.isdir(model_dir):
        # If we have a wildcard pattern, find matching files
        if filename and filename.startswith("*"):
            matching_files = []
            for root, dirs, files in os.walk(model_dir):
                for file in files:
                    if fnmatch.fnmatch(file, filename):
                        matching_files.append(os.path.join(root, file))

            if matching_files:
                print(
                    f"🔍 Found {len(matching_files)} matching files for pattern '{filename}'"
                )
                # Return the first matching file
                return matching_files[0]
        else:
            print(f"🔄 Using cached model from {model_dir}")
            if filename:
                file_path = os.path.join(model_dir, filename)
                if (
                    os.path.exists(file_path)
                    and os.path.isfile(file_path)
                    and os.path.getsize(file_path) > 0
                ):
                    return file_path
                else:
                    print(
                        f"🔶 Cached file {file_path} is missing or empty, redownloading..."
                    )
            else:
                return model_dir

    # Create a temp directory for the download
    temp_model_dir = f"{model_dir}_temp_{int(time.time())}"
    if os.path.exists(temp_model_dir):
        shutil.rmtree(temp_model_dir)
    os.makedirs(temp_model_dir, exist_ok=True)

    # First check if local IPFS daemon is running
    try:
        # Try to connect to the IPFS API
        ipfs_api_url = "http://127.0.0.1:5001/api/v0/id"
        response = requests.post(ipfs_api_url, timeout=10)

        if response.status_code == 200:
            print(f"🌐 Local IPFS node detected, using it to download {cid}")

            # Use subprocess to call ipfs get
            try:
                if filename and not filename.startswith("*"):
                    # Download specific file if needed
                    cmd = [
                        "ipfs",
                        "get",
                        "-o",
                        os.path.join(temp_model_dir, filename),
                        f"{cid}/{filename}",
                    ]
                    print(f"   Running: {' '.join(cmd)}")
                    result = subprocess.run(
                        cmd, capture_output=True, text=True, timeout=600
                    )

                    if result.returncode == 0:
                        downloaded_path = os.path.join(temp_model_dir, filename)
                        if (
                            os.path.exists(downloaded_path)
                            and os.path.isfile(downloaded_path)
                            and os.path.getsize(downloaded_path) > 0
                        ):
                            # Create model directory first
                            os.makedirs(model_dir, exist_ok=True)
                            # Copy file to final location
                            final_path = os.path.join(model_dir, filename)
                            shutil.copy2(downloaded_path, final_path)
                            # Clean up temp
                            shutil.rmtree(temp_model_dir)
                            print(
                                f"✅ File downloaded using local IPFS node: {final_path}"
                            )
                            return final_path
                        else:
                            print(f"🔶 Downloaded file is empty or missing")
                    else:
                        print(f"🔶 Failed to download specific file")
                        # Fall back to downloading the whole directory
                else:
                    # Download entire directory
                    cmd = ["ipfs", "get", "-o", temp_model_dir, cid]
                    result = subprocess.run(
                        cmd, capture_output=True, text=True, timeout=600
                    )

                    if result.returncode == 0:
                        # Find the downloaded content - should be in a subfolder named with the CID
                        downloaded_path = os.path.join(temp_model_dir, cid)

                        # Verify download was successful
                        if os.path.exists(downloaded_path) and os.path.isdir(
                            downloaded_path
                        ):
                            # Move files to the model directory
                            os.makedirs(model_dir, exist_ok=True)

                            # Copy all files to final location
                            for item in os.listdir(downloaded_path):
                                src = os.path.join(downloaded_path, item)
                                dst = os.path.join(model_dir, item)
                                if os.path.isdir(src):
                                    shutil.copytree(src, dst, dirs_exist_ok=True)
                                else:
                                    shutil.copy2(src, dst)

                            print(
                                f"✅ Model downloaded successfully using local IPFS node"
                            )

                            # Handle wildcard patterns
                            if filename and filename.startswith("*"):
                                matching_files = []
                                for root, dirs, files in os.walk(model_dir):
                                    for file in files:
                                        if fnmatch.fnmatch(file, filename):
                                            matching_files.append(
                                                os.path.join(root, file)
                                            )

                                if matching_files:
                                    print(
                                        f"🔍 Found {len(matching_files)} matching files for pattern '{filename}'"
                                    )
                                    # Clean up temp
                                    shutil.rmtree(temp_model_dir)
                                    return matching_files[0]

                            # Return specific file or directory
                            shutil.rmtree(temp_model_dir)
                            if filename:
                                file_path = os.path.join(model_dir, filename)
                                if (
                                    os.path.exists(file_path)
                                    and os.path.isfile(file_path)
                                    and os.path.getsize(file_path) > 0
                                ):
                                    return file_path
                                else:
                                    print(
                                        f"🔶 Required file {filename} not found or empty after download"
                                    )
                                    raise FileNotFoundError(
                                        f"Required file {filename} not found after IPFS download"
                                    )
                            return model_dir
                        else:
                            print(
                                f"🔶 Local IPFS download succeeded but content appears incomplete"
                            )
                    else:
                        print(f"🔶 Local IPFS download failed")
            except (subprocess.SubprocessError, TimeoutError) as e:
                print(f"🔶 Error using local IPFS node")
    except requests.RequestException:
        print(f"🔵 Local IPFS node not available, using IPFS gateway")

    # Fall back to HTTP gateway download
    if filename and not filename.startswith("*"):
        # Download specific file
        os.makedirs(model_dir, exist_ok=True)
        file_path = os.path.join(model_dir, filename)

        gateways = [
            f"https://ipfs.io/ipfs/{cid}/{filename}",
            f"https://dweb.link/ipfs/{cid}/{filename}",
        ]

        print(f"📥 Downloading file from IPFS: {filename}")

        download_success = False
        for gateway_url in gateways:
            try:
                # Show IPFS URI instead of HTTP URL
                ipfs_uri = f"ipfs://{cid}/{filename}"
                print(f"   Trying gateway via: {ipfs_uri}")

                # Stream download with progress
                with requests.get(gateway_url, stream=True, timeout=300) as r:
                    if r.status_code != 200:
                        print(f"   Gateway returned {r.status_code}, trying next...")
                        continue

                    r.raise_for_status()
                    total_size = int(r.headers.get("content-length", 0))

                    temp_file_path = os.path.join(temp_model_dir, filename)
                    with open(temp_file_path, "wb") as f:
                        downloaded = 0
                        for chunk in r.iter_content(chunk_size=8192 * 1024):
                            f.write(chunk)
                            downloaded += len(chunk)

                            # Print progress
                            if total_size > 0:
                                percent = int(100 * downloaded / total_size)
                                print(
                                    f"\r      {downloaded/1024/1024:.1f}MB / {total_size/1024/1024:.1f}MB ({percent}%)",
                                    end="",
                                )

                        print()  # New line after progress

                # Verify file was downloaded correctly by checking file size
                if os.path.getsize(temp_file_path) > 0:
                    # Move to final location
                    shutil.copy2(temp_file_path, file_path)
                    download_success = True
                    print(f"✅ File download complete: {file_path}")
                    break
                else:
                    print(f"   Downloaded file is empty, trying next gateway...")

            except requests.RequestException as e:
                print(f"   Error with gateway")

        # Clean up temp directory
        shutil.rmtree(temp_model_dir)

        if download_success:
            return file_path
        else:
            raise RuntimeError(
                f"Failed to download file '{filename}' from any IPFS gateway for CID: {cid}"
            )
    else:
        # Download directory listing
        print(
            f"📥 Directory download from IPFS is not supported through gateways without specific filename."
        )
        print(f"   Please specify a filename or use a local IPFS node.")
        # Clean up
        shutil.rmtree(temp_model_dir)
        raise RuntimeError(
            "Directory download from IPFS gateway not implemented - please specify a filename"
        )


def handle_ipfs_model(
    model_id: str, filename: Optional[str] = None, cache_dir: Optional[str] = None
) -> str:
    """Handle IPFS model references for llama-cpp-python.

    Args:
        model_id: Model ID, expected to start with "ipfs://"
        filename: Optional specific model filename to use
        cache_dir: Optional cache directory

    Returns:
        Local path to the model directory or file
    """
    # Extract CID from model_id
    parsed = urlparse(model_id)
    if parsed.scheme != "ipfs":
        raise ValueError(f"Expected 'ipfs://' scheme, got: {model_id}")

    cid = parsed.netloc
    if not cid:
        # Handle ipfs://CID format (without netloc)
        cid = parsed.path.lstrip("/")

    # Download or get from cache
    return download_from_ipfs(cid, filename, cache_dir)


def create_patched_from_pretrained(original_func: Callable) -> Callable:
    """Create a patched version of from_pretrained.

    Args:
        original_func: The original from_pretrained function to patch

    Returns:
        The patched function that adds model loading announcements
    """
    # Store the true original function to avoid recursion
    global _ORIGINAL_FROM_PRETRAINED
    _ORIGINAL_FROM_PRETRAINED = original_func

    # Get the actual function from the classmethod
    if isinstance(original_func, classmethod):
        orig_func = original_func.__func__
    else:
        orig_func = original_func

    @functools.wraps(orig_func)
    def wrapper(cls, *args, **kwargs):
        # Extract repo_id parameter (first positional argument or from kwargs)
        repo_id = None
        if len(args) > 0:
            repo_id = args[0]
        elif "repo_id" in kwargs:
            repo_id = kwargs["repo_id"]

        # Extract filename parameter (second positional argument or from kwargs)
        filename = None
        if len(args) > 1:
            filename = args[1]
        elif "filename" in kwargs:
            filename = kwargs["filename"]

        # Only announce each model once to avoid duplicates
        if repo_id:
            key = f"Llama:{repo_id}"
            if key not in _ANNOUNCED_MODELS:
                print(f"🦙 Loading Llama model: {repo_id}")
                _ANNOUNCED_MODELS.add(key)

        # Check if it's an IPFS model
        if isinstance(repo_id, str) and repo_id.startswith("ipfs://"):
            try:
                # Get cache directory from kwargs
                cache_dir = kwargs.get("cache_dir", None)

                # Download or get from cache
                local_path = handle_ipfs_model(repo_id, filename, cache_dir)

                # Use direct initialization with model_path for IPFS
                direct_kwargs = kwargs.copy()
                direct_kwargs["model_path"] = local_path

                # Remove args that would conflict with direct initialization
                for key in ["repo_id", "local_files_only", "token", "revision"]:
                    if key in direct_kwargs:
                        direct_kwargs.pop(key)

                print(f"✅ Successfully prepared IPFS model: {repo_id}")
                result = cls(**direct_kwargs)
                print(f"✅ Successfully loaded Llama model from IPFS")
                return result
            except Exception as e:
                print(f"❌ Error loading IPFS model: {repo_id}")
                raise
        else:
            # Not an IPFS model, proceed with original from_pretrained
            try:
                # For the case in main.py: Llama.from_pretrained(repo_id=repo_id, filename=filename, verbose=False)
                # We need a clean and direct approach that bypasses any proxy errors from the patching process

                # For simplicity, directly create a clean dictionary of arguments and avoid any potential
                # argument conflicts by not using args at all
                all_args = {}
                if repo_id is not None:
                    all_args["repo_id"] = repo_id
                if filename is not None:
                    all_args["filename"] = filename

                # Add any other kwargs (like verbose)
                for k, v in kwargs.items():
                    if (
                        k not in all_args
                    ):  # don't overwrite repo_id/filename we already set
                        all_args[k] = v

                # Call the original function with only the clean kwargs
                result = orig_func.__get__(None, cls)(**all_args)

                if repo_id:
                    print(f"✅ Successfully loaded Llama model: {repo_id}")
                return result
            except Exception as e:
                if repo_id:
                    print(f"❌ Error loading Llama model: {repo_id}")
                raise

    # Convert back to classmethod and mark as patched
    patched_method = classmethod(wrapper)
    patched_method._is_patched = True
    return patched_method


def _safe_patch(show_message=False):
    """
    Perform the actual patching after all imports are completed.
    This version is designed to be compatible with other patches.

    Args:
        show_message: Whether to show the patch loaded message
    """
    global _LLAMA_CPP_PATCHED, _PATCH_MESSAGE_PRINTED, _PATCHED_METHODS

    # Skip if already patched
    if _LLAMA_CPP_PATCHED:
        return

    # Check if llama_cpp is fully imported
    if "llama_cpp" not in sys.modules:
        return

    # Safely attempt patching
    try:
        # Get module without importing
        llama_module = sys.modules["llama_cpp"]

        # Check if Llama class is available
        if not hasattr(llama_module, "Llama"):
            return

        llama_class = llama_module.Llama

        # Check if from_pretrained is available
        if not hasattr(llama_class, "from_pretrained"):
            return

        # Check if already patched
        if hasattr(llama_class.from_pretrained, "_is_patched"):
            _LLAMA_CPP_PATCHED = True
            return

        # Patch the method - store original method for potential cleanup
        original = llama_class.from_pretrained
        wrapped = create_patched_from_pretrained(original)

        # Set the patched method (already a classmethod)
        setattr(llama_class, "from_pretrained", wrapped)

        # Store the original method for cleanup reference
        _PATCHED_METHODS["llama_from_pretrained"] = original

        # Mark as patched
        _LLAMA_CPP_PATCHED = True

        # Only print the message if explicitly requested, not in CLI mode, and not if called from transformers
        if (
            show_message
            and not _PATCH_MESSAGE_PRINTED
            and not CLI_MODE
            and not is_transformers_context()
        ):
            print(f"✅ Llama-cpp-python patch loaded and active!")
            _PATCH_MESSAGE_PRINTED = True
    except Exception as e:
        # Print diagnostic information for troubleshooting
        print(f"🔶 Failed to patch llama-cpp-python")
        # Silently fail, we'll try again later


def setup_import_hook(show_message=False) -> None:
    """Set up the import hook to automatically patch llama_cpp.

    Args:
        show_message: Whether to show the patch loaded message

    This function patches Python's built-in import system to
    intercept imports and attempt to patch after module loading.
    Modified to be more compatible with other patches.
    """
    global _CURRENTLY_IMPORTING

    # Only install the import hook if it's not already installed
    if hasattr(builtins.__import__, "_llama_patch_hook"):
        return

    def patched_import(name, globals=None, locals=None, fromlist=(), level=0):
        # Get the true original import function to avoid conflicts with other patches
        true_original_import = _ORIGINAL_IMPORT

        # Prevent recursive imports
        global _CURRENTLY_IMPORTING
        if _CURRENTLY_IMPORTING:
            return true_original_import(name, globals, locals, fromlist, level)

        _CURRENTLY_IMPORTING = True
        try:
            # Import normally
            module = true_original_import(name, globals, locals, fromlist, level)

            # After importing llama_cpp, schedule a delayed patching
            if name == "llama_cpp" or name.startswith("llama_cpp."):
                # We'll try to patch, but only after this import completes
                time.sleep(0.01)  # Short delay to ensure module initialization
                _safe_patch(show_message)

            return module
        finally:
            _CURRENTLY_IMPORTING = False

    # Mark our patched import function
    patched_import._llama_patch_hook = True

    # Replace the import function with our patched version
    builtins.__import__ = patched_import


def apply_patch(show_message=False) -> None:
    """Apply the llama-cpp-python patch.

    Args:
        show_message: Whether to show the patch loaded message

    This function sets up the import hook but defers actual patching
    until after modules are fully loaded.
    """
    global _PATCH_MESSAGE_PRINTED, CLI_MODE

    # Check if we're being run from the CLI command
    # Script name will contain llama-ipfs or similar when run from CLI
    script_name = sys.argv[0].lower() if len(sys.argv) > 0 else ""
    CLI_MODE = (
        "llama-ipfs" in script_name
        or "llama_ipfs" in script_name
        or "__main__.py" in script_name
    )

    # Install the import hook
    setup_import_hook()

    # Attempt initial patch if possible
    _safe_patch()

    # Only print the message if explicitly requested or in CLI mode
    # and not if called from transformers context
    if (
        show_message
        and not _PATCH_MESSAGE_PRINTED
        and not CLI_MODE
        and not is_transformers_context()
    ):
        print("✅ Llama-cpp-python IPFS integration active!")
        _PATCH_MESSAGE_PRINTED = True


def ensure_llama_model_requirements(model_dir: str, filename: str) -> None:
    """Ensure all required files exist for llama-cpp model loading.

    Args:
        model_dir: Path to the model directory
        filename: The main model filename
    """
    # For now, llama-cpp doesn't need any additional files beyond the model file itself
    # But we can add specific handling here if needed in the future
    pass
