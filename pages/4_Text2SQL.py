import os
from dotenv import load_dotenv
import streamlit as st
import sqlite3
import google.generativeai as genai

load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))


def get_gemini_response(prompt, question):
    llm = genai.GenerativeModel("gemini-pro")
    response = llm.generate_content([prompt, question])
    return response.text

prompts = [
    """
    You are an expert in converting English questions to SQL query!
    The SQL database has the name STUDENT and has the following columns - NAME, CLASS, SECTION and MARKS.
    
    For example, 
    Example 1 - How many entries of records are present?, the SQL command would be something like this: SELECT COUNT(*) FROM STUDENT;
    Example 2 - Tell me all the students studying in Data Science class?, the SQL command will be something like this: SELECT * FROM STUDENT WHERE CLASS="Data Science";
    also the SQL code should not have ``` at the beginning or end and sql word in the output.
    """
]

def run_query(db, sql):
    connection = sqlite3.connect(db)
    cursor = connection.cursor()
    cursor.execute(sql)
    rows = cursor.fetchall()
    connection.commit()
    connection.close()
    return rows


def main():
    st.set_page_config(page_title="Text to SQL")
    st.header("Text to SQL query using Gemini Pro 🤖")

    question = st.chat_input("Ask a question")
    if question:
        with st.chat_message("human"):
            st.write(question)
        
        with st.chat_message("assistant"):
            with st.spinner(""):
                response = get_gemini_response(prompts[0], question)
                rows = run_query("sql/student.db", response)
            data = "\n".join(map(str, rows))
            st.write(response)
            st.write(data)


if __name__ == "__main__":
    main()