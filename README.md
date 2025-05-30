# Llama_IPFS

Load models directly from IPFS for llama-cpp-python.

## Features

- 🌐 Direct integration with local IPFS nodes (preferred method)
- 🔄 Automatic fallback to IPFS gateways when local node isn't available
- 🔍 Simple URI format: `ipfs://CID` for easy model sharing
- ⚡ Zero configuration required - works automatically once installed
- 🧩 Compatible with any version of llama-cpp-python

## Installation

```bash
pip install llama-ipfs
llama-ipfs activate
```

Once installed and activated, the `llama_ipfs` integration will be loaded automatically whenever you use Python.

## Usage

- [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/alexbakers/llama_ipfs/blob/main/examples/colab/llama_ipfs_example.ipynb)
- 🤗 [Demo Space](https://huggingface.co/spaces/alexbakers/llama_ipfs)

After installation, use llama-cpp-python with IPFS model URIs:

```python
from llama_cpp import Llama

# Load a model directly from IPFS
model = Llama.from_pretrained(
    # Equivalent HuggingFace model: "aisuko/gpt2-117M-gguf"
    repo_id = "ipfs://bafybeie7quk74kmqg34nl2ewdwmsrlvvt6heayien364gtu2x6g2qpznhq",
    filename = "ggml-model-Q4_K_M.gguf"
)
```

## IPFS Node Connectivity

The `llama_ipfs` package prioritizes connectivity in the following order:

1. **Local IPFS Node** (Recommended): If you have an IPFS daemon running locally (`ipfs daemon`),
   the package will automatically detect and use it. This method:

   - Is much faster for repeated downloads
   - More reliably loads complex model directories
   - Contributes to the IPFS network by providing content to others

2. **IPFS Gateway** (Fallback): If a local node isn't available, the package will fall back to
   public gateways. This method:
   - Works without installing IPFS
   - May be less reliable for complex model directories
   - Downloads can be interrupted more easily

## Command Line Interface

```bash
# Activate the auto-loading
llama-ipfs activate

# Check if the integration is active
llama-ipfs status

# Test the integration
llama-ipfs test

# Deactivate the integration
llama-ipfs deactivate
```

## Dependencies

- Python 3.8+
- llama-cpp-python

## License

MIT License
