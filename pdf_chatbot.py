import pandas as pd 
import openai 
import langchain
import streamlit as st 

from config import OPENAI_API_KEY, HUGGINGFACE_API_KEY

openai.api_key = OPENAI_API_KEY

def main():
    st.set_page_config(page_title="Chat with multiple pdfs",page_icon=":books:")
    st.header("Chat with multiple pdfs :books:")
    st.text_input("Ask any question about your pdf: ")

    with st.sidebar:
        st.subheader("Your documents: ")
        st.file_uploader("Upload your file here and click on 'Process'.",accept_multiple_files = True)
        st.button("Process")

        st.spinner("Loading your data...")

            #extract the pdf texts



        
            #create embeddings




if __name__ == "__main__":
    main()