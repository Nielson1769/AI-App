import streamlit as st
import openai
import os
import re
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# ✅ Load API Keys Securely from Environment Variables
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
HUGGINGFACE_TOKEN = os.getenv("HUGGINGFACE_TOKEN")

# ✅ Sidebar toggle to control internet access
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

# 🔹 Model mapping for Hugging Face models
def choose_model(selection: str):
    model_mapping = {
        "GPT-4 (OpenAI API)": None,  
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

# ✅ Function to call GPT-4 API
def call_gpt4(prompt):
    if not OPENAI_API_KEY:
        return "Error: OpenAI API key is missing. Please set OPENAI_API_KEY in environment variables."
    
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}],
            api_key=OPENAI_API_KEY
        )
        return response["choices"][0]["message"]["content"]
    except Exception as e:
        return f"Error calling GPT-4: {e}"

# ✅ Load Model with Authentication
@st.cache_resource
def load_model(model_id: str):
    if model_id is None:
        return None, None

    if model_id == "GPT-4 (OpenAI API)":
        return None, None  # GPT-4 uses OpenAI API, not Hugging Face

    try:
        tokenizer = AutoTokenizer.from_pretrained(model_id, token=HUGGINGFACE_TOKEN)
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            token=HUGGINGFACE_TOKEN,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None
        )
        return tokenizer, model
    except Exception as e:
        st.error(f"Error loading {model_id}: {e}")
        return None, None

# Select and Load the Model
model_id = choose_model(model_option)
tokenizer, model = load_model(model_id)

# ✅ Function to Filter Out Sensitive Data
def filter_sensitive_data(text):
    """Removes sensitive personal data from AI responses."""
    sensitive_patterns = [
        r"\b\d{3}-\d{2}-\d{4}\b",  # SSN
        r"\b\d{10}\b",  # Phone numbers
        r"\b\d{4} \d{4} \d{4} \d{4}\b",  # Credit card numbers
        r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,7}\b",  # Emails
        r"\b\d{1,2}/\d{1,2}/\d{4}\b",  # Dates
        r"\b\d{5}(-\d{4})?\b"  # ZIP codes
    ]
    for pattern in sensitive_patterns:
        text = re.sub(pattern, "[REDACTED]", text)
    return text

# ✅ Generate AI Response
if st.button("Generate Response"):
    if model_option == "GPT-4 (OpenAI API)":
        response_text = call_gpt4(user_input)
    elif model and tokenizer:
        inputs = tokenizer(user_input, return_tensors="pt")
        if torch.cuda.is_available():
            inputs = {k: v.to("cuda") for k, v in inputs.items()}
        with torch.no_grad():
            output = model.generate(**inputs, max_new_tokens=150)
        response_text = tokenizer.decode(output[0], skip_special_tokens=True)
    else:
        st.error("Model failed to load. Please check logs or try a different model.")

    response_text = filter_sensitive_data(response_text)

    st.subheader("AI Response:")
    st.write(response_text)

# ✅ Footer
st.sidebar.info("🔹 AI Model Playground - Secure & Private")
