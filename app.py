import streamlit as st
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# Streamlit UI Title
st.title("🧠 AI Model Playground")
st.sidebar.header("Model Settings")

# 🔹 Model Selection Dropdown
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

# 🔹 Function to Map Model Names to Hugging Face Paths
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

# 🔹 Load Model and Tokenizer (With GPU Optimization)
@st.cache_resource
def load_model(model_id: str):
    if model_id is None:
        st.error("GPT-4 requires an API key. Support for OpenAI API will be added soon.")
        return None, None

    try:
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
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

# 🔹 User Input Box
user_input = st.text_area("Enter your prompt:", "Hello, AI!")

# 🔹 Generate Response
if st.button("Generate Response"):
    if model and tokenizer:
        inputs = tokenizer(user_input, return_tensors="pt")
        
        # Move inputs to GPU if available
        if torch.cuda.is_available():
            inputs = {k: v.to("cuda") for k, v in inputs.items()}

        with torch.no_grad():
            output = model.generate(**inputs, max_new_tokens=150)
        
        response_text = tokenizer.decode(output[0], skip_special_tokens=True)
        st.subheader("AI Response:")
        st.write(response_text)
    else:
        st.error("Model failed to load. Please check logs or try a different model.")

# Display Footer
st.sidebar.info("🔹 AI Model Playground - Powered by Hugging Face & Streamlit")
