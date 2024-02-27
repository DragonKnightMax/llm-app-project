import json
from dotenv import load_dotenv
import os
import streamlit as st
from PyPDF2 import PdfReader
import google.generativeai as genai


load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

prompt_template = """
You are a skilled or very experienced resume applicant tracking system (ATS), 
with a deep understanding of tech fields, such as Software Engineering, Data Science, DevOps and Machine Learning. 

Your task is to evaluate the resume provided based on the given job description. 
You must consider the job market is very competitive and you should provide best assistance for improving the resume. 
Assign the percentage matching based on the job description and the missing keywords with high accuracy.

Resume: {resume}
Job Description: {job_description}

I want the response in one single JSON string having the structure without ``` characters: 
{{"percent_match": "XX%", "missing_keywords": [], "profile_summary": ""}}
"""


def get_gemini_response(prompt):
    model = genai.GenerativeModel("gemini-pro")
    response = model.generate_content(prompt)
    return response.text


def extract_pdf_content(uploaded_file):
    if uploaded_file is None:
        raise FileNotFoundError("No file uploaded")
    pdf = PdfReader(uploaded_file)
    text = ""
    for page in pdf.pages:
        text += page.extract_text()
    return text


def main():
    st.set_page_config(page_title="Resume Expert")

    st.header("ATS Resume Scanner")

    with st.sidebar:
        st.header("Menu")
        uploaded_pdf = st.file_uploader("Upload your resume", type=["pdf"])

    job_description = st.chat_input("Enter job description", disabled=(uploaded_pdf is None))

    if uploaded_pdf and job_description:
        with st.chat_message("human"):
            st.write(job_description)

        with st.chat_message("assistant"):
            with st.spinner(""):
                resume = extract_pdf_content(uploaded_pdf)
                input_prompt = prompt_template.format(
                    resume=resume, job_description=job_description
                )
                response = get_gemini_response(input_prompt)
            result = json.loads(response)

            st.subheader("Percentage Match")
            st.write(result["percent_match"])

            st.subheader("Missing Keyword(s)")
            for keyword in result["missing_keywords"]:
                st.markdown(f"- {keyword}")

            st.subheader("Profile Summary")
            st.write(result["profile_summary"])

            
    else:
        st.info("Please upload a resume and provide job description")


if __name__ == "__main__":
    main()