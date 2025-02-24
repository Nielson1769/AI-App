import os
import logging
import streamlit as st
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# ------------------------------
# Logging Configuration
# ------------------------------
logging.basicConfig(
    filename='app.log',
    level=logging.INFO,
    format='%(asctime)s %(levelname)s: %(message)s'
)

# ------------------------------
# Simple Authentication
# ------------------------------
def login():
    st.title("Login")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    if st.button("Login"):
        # For demonstration, use fixed credentials. Replace with your own authentication.
        if username == "admin" and password == "password":
            st.session_state['authenticated'] = True
            st.success("Logged in successfully!")
            logging.info("User %s logged in successfully.", username)
        else:
            st.error("Invalid credentials.")
            logging.warning("Failed login attempt for user %s", username)

if 'authenticated' not in st.session_state:
    st.session_state['authenticated'] = False

if not st.session_state['authenticated']:
    login()
    st.stop()

# ------------------------------
# Sidebar: Settings and Controls
# ------------------------------
st.sidebar.header("Settings")

# Internet Access Toggle
internet_enabled = st.sidebar.checkbox("Enable Internet Access", value=False)
if internet_enabled:
    os.environ["ALLOW_INTERNET"] = "true"
    st.sidebar.success("Internet Access Enabled")
else:
    os.environ["ALLOW_INTERNET"] = "false"
    st.sidebar.error("Internet Access Disabled")

# Model Selection: Automatic, GPT-2, or GPT-Neo
model_option = st.sidebar.selectbox("Select Model", ["Automatic", "GPT-2", "GPT-Neo"])

# ------------------------------
# Model Loading (Cached)
# ------------------------------
@st.cache_resource
def load_model(model_id: str):
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForCausalLM.from_pretrained(model_id)
        if torch.cuda.is_available():
            model.to("cuda")
        logging.info("Loaded model: %s", model_id)
        return tokenizer, model
    except Exception as e:
        logging.error("Error loading model %s: %s", model_id, e)
        st.error("Failed to load the model. Check logs for details.")
        return None, None

# ------------------------------
# Helper: Choose Model Based on Prompt & Selection
# ------------------------------
def choose_model(prompt: str, selection: str):
    if selection == "Automatic":
        # Use GPT-2 for short prompts; GPT-Neo for longer ones.
        return "gpt2" if len(prompt.split()) < 20 else "EleutherAI/gpt-neo-125M"
    elif selection == "GPT-2":
        return "gpt2"
    else:
        return "EleutherAI/gpt-neo-125M"

# ------------------------------
# Conversation History Initialization
# ------------------------------
if 'history' not in st.session_state:
    st.session_state['history'] = []

# ------------------------------
# Main Application Interface
# ------------------------------
st.title("Enhanced LM Studio AI Interface")
st.write("Welcome! Enter a prompt to generate a response. All actions are logged, and conversation history is saved below.")

# Input prompt
prompt = st.text_area("Enter your prompt:", height=150)

if st.button("Generate Response"):
    if not prompt.strip():
        st.error("Please enter a prompt.")
    else:
        with st.spinner("Generating response..."):
            try:
                # Determine which model to use.
                model_id = choose_model(prompt, model_option)
                tokenizer, model = load_model(model_id)
                if tokenizer is None or model is None:
                    st.error("Model loading failed. See logs for details.")
                else:
                    # Encode the prompt and generate a response.
                    input_ids = tokenizer.encode(prompt, return_tensors="pt")
                    if torch.cuda.is_available():
                        input_ids = input_ids.to("cuda")
                    output = model.generate(input_ids, max_length=150)
                    response = tokenizer.decode(output[0], skip_special_tokens=True)
                    st.success("Response generated!")
                    st.write("**Response:**")
                    st.write(response)
                    # Log and save the conversation.
                    st.session_state['history'].append((prompt, response))
                    logging.info("Generated response for prompt: %s", prompt)
            except Exception as e:
                st.error("An error occurred during response generation.")
                logging.error("Error during inference: %s", e)

# Display conversation history.
if st.session_state['history']:
    st.subheader("Conversation History")
    for idx, (q, a) in enumerate(reversed(st.session_state['history'])):
        st.markdown(f"**Q: {q}**")
        st.markdown(f"**A: {a}**")
        st.write("---")

# Clear history button.
if st.button("Clear History"):
    st.session_state['history'] = []
    st.success("Conversation history cleared.")
    logging.info("Conversation history cleared.")
