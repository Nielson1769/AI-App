import streamlit as st
import openai
import os

# ✅ Load OpenAI API Key
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

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

# ✅ Define User Input Field
user_input = st.text_area("Enter your prompt:", "Hello, AI!")

# ✅ Submit Button
if st.button("Generate Response"):
    response_text = call_gpt4(user_input)

    st.subheader("AI Response:")
    st.write(response_text)
