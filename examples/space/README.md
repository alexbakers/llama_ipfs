# LLaMA-IPFS Demo

This Streamlit app demonstrates how to use llama-cpp-python with IPFS models using the llama-ipfs integration.

## Features

- Load models directly from IPFS using their CID
- Input custom context and questions
- Adjust generation parameters (temperature, max tokens, etc.)
- Get instant answers from an IPFS-hosted model

## How It Works

The `llama-ipfs` package patches the llama-cpp-python library to:

1. Recognize `ipfs://` URIs as valid model identifiers
2. Download model files from IPFS nodes or gateways
3. Cache models locally for faster loading in subsequent runs

## Model Used

This demo uses a GPT-2 (117M) model hosted on IPFS with CID:
`bafybeie7quk74kmqg34nl2ewdwmsrlvvt6heayien364gtu2x6g2qpznhq`

## Usage

1. Enter or modify the context in the text area
2. Type your question
3. Adjust generation parameters if desired
4. Click "Generate Answer" to get a response

Note: The first run may take longer as the model is downloaded from IPFS.

## Requirements

- streamlit
- llama-cpp-python
- huggingface-hub
- llama-ipfs
