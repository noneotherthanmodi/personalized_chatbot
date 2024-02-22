import pandas as pd 
import openai 
import langchain
import streamlit as st 
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter


from config import OPENAI_API_KEY, HUGGINGFACE_API_KEY

openai.api_key = OPENAI_API_KEY



def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        read_text = PdfReader(pdf)
        for page in read_text.pages:
            text += page.extract_text()

    return text 


def get_split_text(text):
    text_splitter = CharacterTextSplitter(separator="\n",
                                          chunk_size = 1000,
                                          chunk_overlap=200,
                                          length_function = len)
    chunks = text_splitter.split_text(text)
    return chunks 



def main():
    st.set_page_config(page_title="Chat with multiple pdfs",page_icon=":books:")
    st.header("Chat with multiple pdfs :books:")
    st.text_input("Ask any question about your pdf: ")

    with st.sidebar:
        st.subheader("Your documents: ")
        pdf_docs = st.file_uploader("Upload your file here and click on 'Process'.",accept_multiple_files = True)
        st.button("Process")

        with st.spinner("Loading your data..."):

            #extract the pdf texts
            raw_texts = get_pdf_text(pdf_docs)
            # st.write(raw_texts)


        
            #create embeddings/text split or text chunk
            splited_texts = get_split_text(raw_texts)
            st.write(splited_texts)




if __name__ == "__main__":
    main()