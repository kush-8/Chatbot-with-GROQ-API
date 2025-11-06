import streamlit as st
from langchain_classic.chains import create_history_aware_chain, create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings

import os
from dotenv import load_dotenv
load_dotenv()

os.environ["HF_TOKEN"] = os.getenv("HF_TOKEN")
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# Set up Streamlit app
st.title("Conversational RAG with PDF uploads and chat history")
st.write("Upload PDFs, ask questions, and get answers based on the content of the PDFs.")

## Input the Groq API key
api_key = st.text_input("Enter your Groq API Key:", type="password")

if api_key:
    llm = ChatGroq(api_key=api_key, model="Gemma2-9b-It")
    session_id = st.text_input("Session ID:", value="default_session")

    ## statefully manage chat history
    if 'store' not in st.session_state:
        st.session_state.store = {}

    ## Upload PDFs and create vector store
    uploaded_files = st.file_uploader("Upload PDF files", type=["pdf"], accept_multiple_files=True)
    if uploaded_files:
        docs = []
        for uploaded_file in uploaded_files:
            temppdf = "./temp.pdf"
            with open(temppdf, "wb") as f:
                f.write(uploaded_file.getvalue())
                file_name = uploaded_file.name
            loader = PyPDFLoader(uploaded_file)
            docs.extend(loader.load())

        ### Split documents and create vector store
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        splits = text_splitter.split_documents(docs)
        vector_store = Chroma.from_documents(documents=docs, embedding=embeddings)
        retriever = vector_store.as_retriever()
    