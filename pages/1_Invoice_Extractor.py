import os
from dotenv import load_dotenv
import streamlit as st
import google.generativeai as genai

load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
model = genai.GenerativeModel("gemini-pro-vision")

system_message = """
You are a expert in understanding invoices. 
We will upload an image of invoice.
You need to answer any questions based on the uploaded invoice image.
If the answer is not in the image, just say "answer is not available in the image".
"""

def get_gemini_response(input, image, prompt):
    response = model.generate_content([input, image[0], prompt])
    return response.text


def get_image_details(uploaded_file):
    if uploaded_file is None:
        raise FileNotFoundError("No file uploaded")
    
    bytes_data = uploaded_file.getvalue()
    image_parts = [
        {
            "mime_type": uploaded_file.type,
            "data": bytes_data
        }
    ]
    return image_parts


def main():
    st.set_page_config("Multi-Language Invoice Extractor")
    st.header("Multi-Language Invoice Extractor")

    with st.sidebar:
        st.header("Menu")
        uploaded_image = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

    if uploaded_image is not None:
        image_data = get_image_details(uploaded_image)
        with st.chat_message("human"):
            st.image(uploaded_image, caption="Uploaded Image")
    else:
        st.info("Upload an image to begin the conversation")

    question = st.chat_input("Ask a question", disabled=(uploaded_image is None))
    if question:
        with st.chat_message("human"):
            st.write(question)
        with st.chat_message("assistant"):
            with st.spinner(""):
                response = get_gemini_response(system_message, image_data, question)
            st.write(response)


if __name__ == "__main__":
    main()