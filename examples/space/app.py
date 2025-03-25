import streamlit as st
import time
import os

# Set page title and description
st.set_page_config(
    page_title="IPFS-LLaMA Demo",
    page_icon="🦙",
    layout="wide"
)

st.title("LLaMA-IPFS Demo")
st.markdown("This app demonstrates using llama-cpp-python with IPFS-hosted models.")

# Install required packages if not already installed
@st.cache_resource
def install_dependencies():
    import subprocess
    packages = ["llama-cpp-python", "huggingface-hub", "llama-ipfs"]
    for package in packages:
        try:
            __import__(package)
        except ImportError:
            st.write(f"Installing {package}...")
            subprocess.check_call(["pip", "install", package, "--quiet"])
    
    # Activate IPFS integration
    import llama_ipfs
    llama_ipfs.activate()
    st.success("✅ Dependencies installed and IPFS integration activated!")

# Display a loading spinner while installing dependencies
with st.spinner("Setting up environment..."):
    install_dependencies()

# Load the LLaMA model
@st.cache_resource
def load_model(repo_id, filename):
    from llama_cpp import Llama
    with st.spinner(f"Loading model from {repo_id}/{filename}..."):
        st.info("This may take a few minutes for the first run as the model is downloaded from IPFS.")
        try:
            model = Llama.from_pretrained(
                repo_id=repo_id, 
                filename=filename, 
                verbose=False
            )
            return model
        except Exception as e:
            st.error(f"Error loading model: {str(e)}")
            return None

# Model selection
model_options = {
    "GPT-2 (117M)": {
        "repo_id": "ipfs://bafybeie7quk74kmqg34nl2ewdwmsrlvvt6heayien364gtu2x6g2qpznhq",
        "filename": "ggml-model-Q4_K_M.gguf"
    }
}

# Sidebar for model selection and parameters
st.sidebar.title("Model Settings")

selected_model = st.sidebar.selectbox(
    "Select Model",
    list(model_options.keys()),
    index=0
)

# Load the selected model
model_info = model_options[selected_model]
llm = load_model(model_info["repo_id"], model_info["filename"])

# Generation parameters
st.sidebar.title("Generation Settings")
temperature = st.sidebar.slider("Temperature", 0.0, 1.0, 0.0, 0.1)
max_tokens = st.sidebar.slider("Max Output Tokens", 10, 100, 30)
top_p = st.sidebar.slider("Top P", 0.1, 1.0, 0.95, 0.05)
repeat_penalty = st.sidebar.slider("Repeat Penalty", 1.0, 2.0, 1.0, 0.1)

# Main content area
st.header("Question & Answer")

# Input fields
context = st.text_area(
    "Context",
    value="France is a country.",
    height=150
)

question = st.text_input("Question", value="What is the capital of France?")

# Generate button
if st.button("Generate Answer"):
    if llm is not None:
        prompt = f"Context: {context}\nQuestion: {question}\nBased solely on the above context, answer the question in one word:"
        
        # Show the prompt
        with st.expander("Prompt"):
            st.text(prompt)
        
        # Generate the answer with progress
        with st.spinner("Generating answer..."):
            start_time = time.time()
            output = llm(
                prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                repeat_penalty=repeat_penalty,
                stop=["\n"]
            )
            end_time = time.time()
            
            # Extract and display the answer
            answer = output['choices'][0]['text'].strip()
            
            st.success(f"Generated in {end_time - start_time:.2f} seconds")
            
            # Display the answer
            st.header("Answer")
            st.write(answer)
    else:
        st.error("Model not loaded. Please check the error message above.")

# Information about the app
st.markdown("---")
st.markdown("""
### How It Works

The `llama-ipfs` package patches the llama-cpp-python library to:

1. Recognize `ipfs://` URIs as valid model identifiers
2. Download model files from IPFS nodes or gateways
3. Cache models locally for faster loading in subsequent runs

This allows you to load models from a decentralized network without changing any of your existing code!
""") 