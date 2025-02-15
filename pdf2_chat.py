import openai 
from dotenv import load_dotenv
import streamlit as st 
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter 
from langchain_openai import OpenAIEmbeddings
from config import OPENAI_API_KEY

from langchain.vectorstores import FAISS
from langchain_openai import ChatOpenAI 
from langchain.memory import ConversationBufferMemory 
from langchain.chains import ConversationalRetrievalChain
# from langchain.document_loaders import PyPDFLoader 



openai.api_key = OPENAI_API_KEY

from htmlTemp import css,bot_template,user_template




def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        read_text = PdfReader(pdf)
        # documents = read_text.load()
        # text += "\n".join([doc.page_content for doc in documents])

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


def get_vectors(text_chunks):
    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
    
    # print(f"Number of text chunks: {len(text_chunks)}")
    # if text_chunks:
    #     print(f"Length of the first text chunk: {len(text_chunks[0])}")


    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore


def get_conversation_chain(vectorstore):
    llm = ChatOpenAI(api_key=OPENAI_API_KEY,temperature=0)
    memory = ConversationBufferMemory(memory_key='chat_history',return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory = memory)
    return conversation_chain

 
def handle_userinput(user_question):
    if st.session_state.conversation is None:
        st.error("Please upload and process PDF documents first before asking questions.")
        return
    
    response = st.session_state.conversation({'question': user_question})
    st.write(response)




def main():
    load_dotenv()
    st.set_page_config(page_title="Chat with multiple pdfs",page_icon=":books:")
    st.write(css,unsafe_allow_html=True)


    if "conversation" not in st.session_state:
        st.session_state.conversation = None 
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    st.header("Chat with multiple pdfs :books:")
    user_question = st.text_input("Ask any question about your pdf: ")
    if user_question:
        handle_userinput(user_question)

    st.write(user_template.replace("{{MSG}}","Hello Future!"),unsafe_allow_html=True)
    st.write(bot_template.replace("{{MSG}}","Hello Human!"),unsafe_allow_html=True)



    with st.sidebar:
        st.subheader("Your documents: ")
        pdf_docs = st.file_uploader("Upload your file here and click on 'Process'.",accept_multiple_files = True)
        
        if st.button("Process"):
            with st.spinner("Loading your data..."):

                #extract the pdf texts
                raw_texts = get_pdf_text(pdf_docs)
                # st.write(raw_texts)


            
                #create embeddings/text split or text chunk
                text_chunks = get_split_text(raw_texts)
                # st.write(splited_texts)



                #create vector 
                vectorstore = get_vectors(text_chunks)



                #create lang chains for conversation
                st.session_state.conversation = get_conversation_chain(vectorstore)

    


if __name__ == "__main__":
    main()