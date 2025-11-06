import streamlit as st
from langchain_classic.chains import create_history_aware_retriever, create_retrieval_chain
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
    llm = ChatGroq(api_key=api_key, model="meta-llama/llama-4-scout-17b-16e-instruct")
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
            loader = PyPDFLoader("temp.pdf")
            docs.extend(loader.load())
            os.remove("temp.pdf")

        ### Split documents and create vector store
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        splits = text_splitter.split_documents(docs)
        vector_store = Chroma.from_documents(documents=docs, embedding=embeddings)
        retriever = vector_store.as_retriever()

        ## create prompt template
        contextualize_q_system_prompt = (
            "Given a chat history and the latest user question"
            "which might reference context in the chat history,"
            "formulate a standalone question which can be understood"
            "without the chat history. Do NOT answer the question,"
            "just reformulate it if needed and otherwise return it as is."
        )
        contextualize_q_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", contextualize_q_system_prompt),
                MessagesPlaceholder(variable_name="chat_history"),
                ("human", "{input}"),
            ]
        )


        ## Q&A prompt template
        system_prompt = (
            "You are an assistant for question answering tasks."
            "Use the following pieces of retrieved context to answer"  
            "the question. If you don't know the answer, say that you don't"
            "know. Use three sentences maximum and keep the"
            "answers concise and to the point."
            "\n\n"
            "{context}"
        )
        qa_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system_prompt),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}"),
            ]
        )

        ## Function to get or create chat history for a session
        def get_session_history(session_id: str) -> BaseChatMessageHistory:
            if session_id not in st.session_state.store:
                st.session_state.store[session_id] = ChatMessageHistory()
            return st.session_state.store[session_id]
        
        ## Create retrieval chain with chat history
        history_aware_retriever = create_history_aware_retriever(llm, retriever, contextualize_q_prompt)
        question_answering_chain = create_stuff_documents_chain(llm, qa_prompt)
        rag_chain = create_retrieval_chain(
            history_aware_retriever,
            question_answering_chain,
        )
        conversational_rag = RunnableWithMessageHistory(
            rag_chain,
            get_session_history=get_session_history,
            input_messages_key="input",
            history_messages_key="chat_history",
            output_messages_key="answer",
        )

        ## User input for questions
        user_input = st.text_input("Enter your question about the uploaded PDFs:")
        if user_input:
            session_history = get_session_history(session_id)
            response = conversational_rag.invoke(
                {"input": user_input},
                config={
                    "configurable": {"session_id": session_id}
                }
            )
            st.write("**Assistant:**", response["answer"])
            st.subheader("Chat History:")
            for message in session_history.messages:
                if message.type == "human":
                    st.markdown(f"**You:** {message.content}")
                elif message.type == "ai":
                    st.markdown(f"**Assistant:** {message.content}")

else:
    st.warning("Please enter your Groq API Key to proceed.")
    