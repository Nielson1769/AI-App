import streamlit as st
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import os
import re
import logging

# ✅ Fix: Correct logging setup (instead of using st.set_option)
logging.basicConfig(level=logging.ERROR)
logging.getLogger("streamlit").setLevel(logging.ERROR)

# ✅ Fix: Check if running locally or on Streamlit Cloud
IS_LOCAL = os.getenv("STREAMLIT_SERVER") is None  # Streamlit Cloud sets this variable

# ✅ Fix: Only modify os.environ for local use
if IS_LOCAL:
    internet_access = st.sidebar.checkbox("Enable Internet Access", value=False)
    
    def toggle_internet_access(enable: bool):
        """Enable or disable internet access for AI models."""
        if enable:
            if "HTTP_PROXY" in os.environ:
                del os.environ["HTTP_PROXY"]
            if "HTTPS_PROXY" in os.environ:
                del os.environ["HTTPS_PROXY"]
            st.sidebar.success("Internet access enabled.")
        else:
            os.environ["HTTP_PROXY"] = ""
            os.environ["HTTPS_PROXY"] = ""
            st.sidebar.warning("Internet access disabled.")
    
    toggle_internet_access(internet_access)
else:
    st.sidebar.warning("Internet access settings cannot be changed on Streamlit Cloud.")

# 🔹 Model selection dropdown
model_option = st.sidebar.selectbox("Select AI Model", [
    "GPT-4 (OpenAI API)",  
    "DeepSeek-7B",  
    "Mistral-7B",  
    "LLaMA-2-13B",  
    "GPT-J-6B",  
    "Gemma-7B",  
    "Zephyr-7B",  
    "MPT-7B",  
    "StableLM-7B",  
    "RedPajama-7B",  
    "Phi-2"
])

# 🔹 Function to map selected model to Hugging Face model ID
def choose_model(selection: str):
    model_mapping = {
        "GPT-4 (OpenAI API)": None,  # Requires OpenAI API key
        "DeepSeek-7B": "deepseek-ai/deepseek-llm-7b",  
        "Mistral-7B": "mistralai/Mistral-7B",  
        "LLaMA-2-13B": "meta-llama/Llama-2-13b",  
        "GPT-J-6B": "EleutherAI/gpt-j-6B",  
        "Gemma-7B": "google/gemma-7b",  
        "Zephyr-7B": "HuggingFaceH4/zephyr-7b-beta",  
        "MPT-7B": "mosaicml/mpt-7b",  
        "StableLM-7B": "stabilityai/stablelm-7b",  
        "RedPajama-7B": "togethercomputer/RedPajama-INCITE-7B",  
        "Phi-2": "microsoft/phi-2"
    }
    return model_mapping.get(selection, "gpt2")  # Default to GPT-2 if invalid

# 🔹 Load Hugging Face API Token (for private models)
HUGGINGFACE_TOKEN = os.getenv("HUGGINGFACE_TOKEN", "hf_your_token_here")

# 🔹 Load Model with GPU Optimization
@st.cache_resource
def load_model(model_id: str):
    if model_id is None:
        st.error("GPT-4 requires an API key. Support for OpenAI API will be added soon.")
        return None, None

    try:
        tokenizer = AutoTokenizer.from_pretrained(model_id, token=HUGGINGFACE_TOKEN)
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            token=HUGGINGFACE_TOKEN,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,  # Use GPU acceleration
            device_map="auto" if torch.cuda.is_available() else None  # Moves model to GPU if available
        )
        return tokenizer, model
    except Exception as e:
        st.error(f"Error loading {model_id}: {e}")
        return None, None

# Select and Load the Model
model_id = choose_model(model_option)
tokenizer, model = load_model(model_id)

# 🔹 Function to Filter Out Sensitive Data
def filter_sensitive_data(text):
    """Redacts personal information before displaying/sending data."""
    sensitive_patterns = [
        r"\b\d{3}-\d{2}-\d{4}\b",  # Social Security Number (SSN)
        r"\b\d{10}\b",  # Phone number
        r"\b\d{4} \d{4} \d{4} \d{4}\b",  # Credit card numbers
        r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,7}\b",  # Emails
        r"\b\d{1,2}/\d{1,2}/\d{4}\b",  # Dates (e.g., birthdays)
        r"\b\d{5}(-\d{4})?\b"  # ZIP codes
    ]

    for pattern in sensitive_patterns:
        text = re.sub(pattern, "[REDACTED]", text)

    return text

# 🔹 User Input Box
user_input = st.text_area("Enter your prompt:", "Hello, AI!")

# 🔹 Generate AI Response
if st.button("Generate Response"):
    if model and tokenizer:
        inputs = tokenizer(user_input, return_tensors="pt")

        # Move inputs to GPU if available
        if torch.cuda.is_available():
            inputs = {k: v.to("cuda") for k, v in inputs.items()}

        with torch.no_grad():
            output = model.generate(**inputs, max_new_tokens=150)

        response_text = tokenizer.decode(output[0], skip_special_tokens=True)

        # Apply privacy filter before displaying
        response_text = filter_sensitive_data(response_text)

        st.subheader("AI Response:")
        st.write(response_text)
    else:
        st.error("Model failed to load. Please check logs or try a different model.")

# Display Footer
st.sidebar.info("🔹 AI Model Playground - Secure & Private")
