import os
import streamlit as st
from PyPDF2 import PdfReader
from dotenv import load_dotenv

import google.generativeai as genai
from langchain.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.question_answering import load_qa_chain
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI


load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

FAISS_INDEX = "faiss_index"

def extract_pdf_text(pdf_files):
    text = ""
    for pdf in pdf_files:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text


def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    text_chunks = text_splitter.split_text(text)
    return text_chunks


def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    vector_store.save_local(FAISS_INDEX)
    return vector_store


def get_conversation_chain():
    prompt_template = """
    You are a chatbot having a conversation with a human.
    
    Answer the following question as detailed as possible from the provided context, 
    make sure to provide all the details, if the answer is not in provided context 
    just say, "answer is not available in the context", don't provide the wrong answer

    Context: {context}
    Question: {user_question}

    Answer:
    """
    llm = ChatGoogleGenerativeAI(
        model="gemini-pro", temperature=0.3, convert_system_message_to_human=True
    )
    prompt = PromptTemplate(
        template=prompt_template, input_variables=["context", "user_question"]
    )
    conversation_chain = load_qa_chain(
        llm=llm, chain_type="stuff", prompt=prompt, verbose=True
    )
    return conversation_chain


def handle_user_input(user_question):
    with st.chat_message("user"):
        st.write(user_question)

    chatbot_reply = st.chat_message("assistant")
    with chatbot_reply:
        st.write()

    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.load_local(FAISS_INDEX, embeddings, allow_dangerous_deserialization=True)
    input_docs = vector_store.similarity_search(user_question)
    
    conversation_chain = get_conversation_chain()
    response =  conversation_chain(
        {"input_documents": input_docs, "user_question": user_question}, 
        return_only_outputs=True
    )
    
    with chatbot_reply:
        st.write(response['output_text'])


def process_pdf_files(pdf_files):
    with st.spinner("Processing..."):
        raw_text = extract_pdf_text(pdf_files)
        text_chunks = get_text_chunks(raw_text)
        vector_store = get_vector_store(text_chunks)
        st.success("Completed", icon="✅")


def main():
    st.set_page_config("Chat with your PDFs", page_icon=":books:")
    st.header("Chat with your PDFs using Gemini AI 🤖")

    user_question =  st.chat_input("Ask a question:")
    if user_question:
        handle_user_input(user_question)

    with st.sidebar:
        st.title("Menu")
        pdf_files = st.file_uploader("Upload your PDF file(s)", accept_multiple_files=True)
        if st.button("Process"):
            process_pdf_files(pdf_files)


if __name__ == "__main__":
    main()