"""
Financial Chatbot with Streamlit Interface
Enhanced version with LangChain integration
"""

import streamlit as st
import os
import openai
from datetime import datetime
import time
import langchain

# LangChain imports
from dotenv import load_dotenv, find_dotenv
from langchain.document_loaders import PyPDFLoader, WebBaseLoader, TextLoader
from langchain.document_loaders.generic import GenericLoader
from langchain.document_loaders.parsers import OpenAIWhisperParser
from langchain_community.document_loaders.blob_loaders import YoutubeAudioLoader

# Text splitters
from langchain.text_splitter import CharacterTextSplitter, RecursiveCharacterTextSplitter
from langchain.text_splitter import MarkdownHeaderTextSplitter, TokenTextSplitter

# Embeddings and Vector Stores
from langchain.embeddings import OpenAIEmbeddings, HuggingFaceEmbeddings, SentenceTransformerEmbeddings
from langchain.vectorstores import Chroma

# Chat models and chains
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.chains import RetrievalQA, ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.retrievers.self_query.base import SelfQueryRetriever
from langchain.chains.query_constructor.base import AttributeInfo

# Page configuration
st.set_page_config(
    page_title="Financial Analysis Chatbot",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .stButton > button {
        background-color: #4CAF50;
        color: white;
        border-radius: 5px;
        padding: 0.5rem 1rem;
        border: none;
        cursor: pointer;
        transition: background-color 0.3s;
    }
    .stButton > button:hover {
        background-color: #45a049;
    }
    .chat-message {
        padding: 1rem;
        margin: 0.5rem 0;
        border-radius: 10px;
    }
    .user-message {
        background-color: #e3f2fd;
        text-align: right;
    }
    .bot-message {
        background-color: #f5f5f5;
        text-align: left;
    }
    .sidebar .sidebar-content {
        background-color: #f0f2f6;
    }
    </style>
    """, unsafe_allow_html=True)

# Initialize session state
if 'initialized' not in st.session_state:
    st.session_state.initialized = False
if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'vectordb' not in st.session_state:
    st.session_state.vectordb = None
if 'qa_chain' not in st.session_state:
    st.session_state.qa_chain = None
if 'memory' not in st.session_state:
    st.session_state.memory = None

# Load environment variables
load_dotenv(find_dotenv())

def initialize_openai(api_key):
    """Initialize OpenAI with API key"""
    openai.api_key = api_key
    os.environ["OPENAI_API_KEY"] = api_key

@st.cache_resource
def load_and_process_documents(file_path=None, file_type="pdf", url=None):
    """Load and process documents based on type"""
    try:
        if file_type == "pdf" and file_path:
            loader = PyPDFLoader(file_path)
            pages = loader.load()
        elif file_type == "youtube" and url:
            save_dir = "./youtube_audio"
            os.makedirs(save_dir, exist_ok=True)
            loader = GenericLoader(
                YoutubeAudioLoader([url], save_dir), 
                OpenAIWhisperParser()
            )
            pages = loader.load()
        elif file_type == "web" and url:
            loader = WebBaseLoader(url)
            pages = loader.load()
        else:
            return None
        
        # Split documents
        chunk_size = 150
        chunk_overlap = 10
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size, 
            chunk_overlap=chunk_overlap
        )
        text_split = splitter.split_documents(pages)
        
        return text_split
    except Exception as e:
        st.error(f"Error loading documents: {str(e)}")
        return None

@st.cache_resource
def create_vectorstore(_documents, api_key):
    """Create vector store from documents"""
    try:
        os.environ["OPENAI_API_KEY"] = api_key
        embedding = OpenAIEmbeddings()
        vectordb = Chroma.from_documents(
            documents=_documents, 
            embedding=embedding
        )
        return vectordb
    except Exception as e:
        st.error(f"Error creating vector store: {str(e)}")
        return None

def initialize_qa_chain(vectordb, api_key):
    """Initialize QA chain with memory"""
    try:
        os.environ["OPENAI_API_KEY"] = api_key
        llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)
        memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True
        )
        qa_chain = ConversationalRetrievalChain.from_llm(
            llm,
            retriever=vectordb.as_retriever(search_kwargs={"k": 3}),
            memory=memory
        )
        return qa_chain, memory
    except Exception as e:
        st.error(f"Error initializing QA chain: {str(e)}")
        return None, None

def main():
    # Header
    col1, col2 = st.columns([3, 1])
    with col1:
        st.title("💰 Financial Analysis Chatbot")
        st.markdown("*Powered by LangChain and OpenAI*")
    with col2:
        if st.button("🔄 Clear Chat", use_container_width=True):
            st.session_state.messages = []
            if st.session_state.memory:
                st.session_state.memory.clear()
            st.rerun()
    
    # Sidebar for configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # API Key input
        api_key = st.text_input(
            "OpenAI API Key",
            type="password",
            placeholder="Enter your OpenAI API key",
            help="Get your API key from https://platform.openai.com/api-keys"
        )
        
        if api_key:
            initialize_openai(api_key)
            
            st.divider()
            st.header("📄 Document Upload")
            
            # Document type selection
            doc_type = st.selectbox(
                "Select Document Type",
                ["PDF", "YouTube URL", "Web URL"]
            )
            
            if doc_type == "PDF":
                uploaded_file = st.file_uploader(
                    "Upload PDF Document",
                    type=['pdf'],
                    help="Upload a financial document for analysis"
                )
                
                if uploaded_file is not None:
                    # Save uploaded file temporarily
                    with open("temp.pdf", "wb") as f:
                        f.write(uploaded_file.getbuffer())
                    
                    if st.button("📥 Process Document", use_container_width=True):
                        with st.spinner("Processing document..."):
                            documents = load_and_process_documents(
                                file_path="temp.pdf",
                                file_type="pdf"
                            )
                            if documents:
                                st.session_state.vectordb = create_vectorstore(documents, api_key)
                                if st.session_state.vectordb:
                                    qa_chain, memory = initialize_qa_chain(
                                        st.session_state.vectordb, 
                                        api_key
                                    )
                                    st.session_state.qa_chain = qa_chain
                                    st.session_state.memory = memory
                                    st.session_state.initialized = True
                                    st.success("✅ Document processed successfully!")
            
            elif doc_type == "YouTube URL":
                youtube_url = st.text_input(
                    "Enter YouTube URL",
                    placeholder="https://youtube.com/..."
                )
                if youtube_url and st.button("📥 Process Video", use_container_width=True):
                    with st.spinner("Processing video..."):
                        documents = load_and_process_documents(
                            url=youtube_url,
                            file_type="youtube"
                        )
                        if documents:
                            st.session_state.vectordb = create_vectorstore(documents, api_key)
                            if st.session_state.vectordb:
                                qa_chain, memory = initialize_qa_chain(
                                    st.session_state.vectordb,
                                    api_key
                                )
                                st.session_state.qa_chain = qa_chain
                                st.session_state.memory = memory
                                st.session_state.initialized = True
                                st.success("✅ Video processed successfully!")
            
            elif doc_type == "Web URL":
                web_url = st.text_input(
                    "Enter Web URL",
                    placeholder="https://example.com/..."
                )
                if web_url and st.button("📥 Process Webpage", use_container_width=True):
                    with st.spinner("Processing webpage..."):
                        documents = load_and_process_documents(
                            url=web_url,
                            file_type="web"
                        )
                        if documents:
                            st.session_state.vectordb = create_vectorstore(documents, api_key)
                            if st.session_state.vectordb:
                                qa_chain, memory = initialize_qa_chain(
                                    st.session_state.vectordb,
                                    api_key
                                )
                                st.session_state.qa_chain = qa_chain
                                st.session_state.memory = memory
                                st.session_state.initialized = True
                                st.success("✅ Webpage processed successfully!")
            
            # Quick load default document option
            st.divider()
            if st.button("📊 Load Sample Financial Document", use_container_width=True):
                # This would load your default financial_analysis_bcg.pdf
                st.info("Please upload the financial_analysis_bcg.pdf file")
            
            # Settings
            st.divider()
            st.header("🎯 Settings")
            
            retrieval_method = st.selectbox(
                "Retrieval Method",
                ["Conversational", "Standard QA", "Map-Reduce", "Refine"]
            )
            
            num_results = st.slider(
                "Number of Results",
                min_value=1,
                max_value=10,
                value=3,
                help="Number of relevant documents to retrieve"
            )
            
            temperature = st.slider(
                "Response Temperature",
                min_value=0.0,
                max_value=1.0,
                value=0.0,
                step=0.1,
                help="Higher values make output more creative"
            )
            
            # Display status
            st.divider()
            st.header("📈 Status")
            if st.session_state.initialized:
                st.success("✅ System Ready")
                if st.session_state.vectordb:
                    st.info(f"📚 Documents loaded and indexed")
            else:
                st.warning("⚠️ Please configure and load documents")
    
    # Main chat interface
    if not api_key:
        st.warning("👈 Please enter your OpenAI API key in the sidebar to get started")
        st.info("""
        ### How to use this chatbot:
        1. Enter your OpenAI API key in the sidebar
        2. Upload a financial document (PDF, YouTube URL, or Web URL)
        3. Wait for the document to be processed
        4. Start asking questions about the content!
        
        ### Example questions:
        - Which company do you think is the best to invest in?
        - What are the key financial metrics mentioned?
        - Summarize the main findings of the analysis
        - What are the growth prospects for the companies mentioned?
        """)
    elif not st.session_state.initialized:
        st.info("👈 Please upload and process a document to start chatting")
    else:
        # Chat messages display
        chat_container = st.container()
        with chat_container:
            for message in st.session_state.messages:
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])
        
        # Chat input
        if prompt := st.chat_input("Ask a question about your financial documents..."):
            # Add user message
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)
            
            # Generate response
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    try:
                        if st.session_state.qa_chain:
                            response = st.session_state.qa_chain({"question": prompt})
                            answer = response.get("answer", "I couldn't find an answer to that question.")
                            
                            # Display response with typing effect
                            message_placeholder = st.empty()
                            full_response = ""
                            for chunk in answer.split():
                                full_response += chunk + " "
                                time.sleep(0.05)
                                message_placeholder.markdown(full_response + "▌")
                            message_placeholder.markdown(full_response)
                            
                            # Add to message history
                            st.session_state.messages.append(
                                {"role": "assistant", "content": full_response}
                            )
                        else:
                            st.error("QA chain not initialized. Please reload the document.")
                    except Exception as e:
                        st.error(f"Error generating response: {str(e)}")
        
        # Example questions
        st.divider()
        st.subheader("💡 Suggested Questions")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("📊 Best investment?", use_container_width=True):
                st.session_state.messages.append(
                    {"role": "user", "content": "Which company do you think is the best to invest in?"}
                )
                st.rerun()
        
        with col2:
            if st.button("📈 Growth analysis", use_container_width=True):
                st.session_state.messages.append(
                    {"role": "user", "content": "What are the growth prospects mentioned in the document?"}
                )
                st.rerun()
        
        with col3:
            if st.button("💰 Key metrics", use_container_width=True):
                st.session_state.messages.append(
                    {"role": "user", "content": "What are the key financial metrics discussed?"}
                )
                st.rerun()

if __name__ == "__main__":
    main()