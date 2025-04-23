import streamlit as st
import os
import tempfile
from langchain.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from pinecone import Pinecone
from langchain_pinecone import PineconeVectorStore
from pinecone import ServerlessSpec
import time
from langchain.embeddings.openai import OpenAIEmbeddings

# Streamlit page configuration
st.set_page_config(page_title="RAG Demo", page_icon="📚", layout="wide")

# Initialize session state variables
if "conversation" not in st.session_state:
    st.session_state.conversation = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "processing_complete" not in st.session_state:
    st.session_state.processing_complete = False

# Function to load and process the document
def process_document(uploaded_file):
    with tempfile.NamedTemporaryFile(delete=False, suffix=f".{uploaded_file.name.split('.')[-1]}") as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        tmp_path = tmp_file.name

    if uploaded_file.name.endswith('.pdf'):
        loader = PyPDFLoader(tmp_path)
    elif uploaded_file.name.endswith('.docx'):
        loader = Docx2txtLoader(tmp_path)
    elif uploaded_file.name.endswith('.txt'):
        loader = TextLoader(tmp_path)
    else:
        os.unlink(tmp_path)
        st.error("Unsupported file format. Please upload a PDF, DOCX, or TXT file.")
        return None

    documents = loader.load()
    os.unlink(tmp_path)

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_documents(documents)

    return chunks

# Function to create embeddings and store in Pinecone
def embed_and_store(chunks):
    embeddings = OpenAIEmbeddings()

    pinecone = Pinecone(api_key=os.environ.get("PINECONE_API_KEY"))
    index_name = "rag-demo"

    existing_indexes = [index_info["name"] for index_info in pinecone.list_indexes()]
    if index_name not in existing_indexes:
        pinecone.create_index(
            name=index_name,
            dimension=1536,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1")
        )
        while not pinecone.describe_index(index_name).status["ready"]:
            time.sleep(1)

    index = pinecone.Index(index_name)
    vectorstore = PineconeVectorStore(index=index, embedding=embeddings)

    vectorstore = vectorstore.from_documents(documents=chunks, embedding=embeddings, index_name=index_name)
    return vectorstore

# Function to create conversation chain
def create_conversation_chain(vectorstore):
    llm = ChatOpenAI(temperature=0.0, model_name="gpt-3.5-turbo")
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

    return ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
        memory=memory
    )

# Function to handle user query
def process_query(user_query):
    response = st.session_state.conversation({"question": user_query})
    st.session_state.chat_history.append((user_query, response["answer"]))

# UI Title
st.title("📚 RAG Demo Application")

# Sidebar: API key input
with st.sidebar:
    st.header("Configuration")

    openai_api_key = st.text_input("OpenAI API Key", type="password")
    pinecone_api_key = st.text_input("Pinecone API Key", type="password")

    if openai_api_key:
        os.environ["OPENAI_API_KEY"] = openai_api_key
    if pinecone_api_key:
        os.environ["PINECONE_API_KEY"] = pinecone_api_key

    st.divider()
    st.markdown("### About")
    st.markdown(
        """
        This application demonstrates Retrieval-Augmented Generation (RAG):

        1. Upload a document
        2. It gets split and embedded
        3. Embeddings are stored in Pinecone
        4. Chat with your document!
        """
    )

# Document Upload Section
if not st.session_state.processing_complete:
    st.header("📄 Document Processing")

    uploaded_file = st.file_uploader("Upload a document (PDF, DOCX, TXT)", type=["pdf", "docx", "txt"])

    if uploaded_file is not None:
        with st.spinner("Processing document..."):
            chunks = process_document(uploaded_file)

            if chunks:
                st.success(f"Document processed! Created {len(chunks)} chunks.")

                if not (os.environ.get("OPENAI_API_KEY") and os.environ.get("PINECONE_API_KEY")):
                    st.error("Please provide both OpenAI and Pinecone API keys in the sidebar.")
                else:
                    with st.spinner("Creating embeddings and storing in Pinecone..."):
                        vectorstore = embed_and_store(chunks)
                        st.session_state.conversation = create_conversation_chain(vectorstore)
                        st.session_state.processing_complete = True
                        st.success("Embeddings created and stored successfully!")
                        st.rerun()

# Chat Section
if st.session_state.processing_complete:
    st.header("💬 Chat with your Document")

    for user_msg, ai_msg in st.session_state.chat_history:
        st.chat_message("user").write(user_msg)
        st.chat_message("assistant").write(ai_msg)

    user_query = st.chat_input("Ask something about your document...")
    if user_query:
        st.chat_message("user").write(user_query)
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                process_query(user_query)
                st.write(st.session_state.chat_history[-1][1])

# Reset Button
if st.session_state.processing_complete:
    if st.sidebar.button("🔄 Process New Document"):
        st.session_state.processing_complete = False
        st.session_state.conversation = None
        st.session_state.chat_history = []
        st.rerun()
