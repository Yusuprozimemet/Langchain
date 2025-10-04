import streamlit as st
import shutil
from pathlib import Path
import os
import time

# Add the project root to path for imports
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

from app.services.vector_store_service import ingest_documents, get_retriever
from app.services.rag_service import query_rag
from app.services.agent_service import query_agent
from app.config.settings import settings  # Loads .env

st.set_page_config(page_title="LangChain RAG & Agent Demo", layout="wide")

st.title("🦜️🔗 LangChain RAG & Agent Demo")
st.write("Upload PDFs, query RAG, or use the agent with tools!")

# Sidebar for session management
st.sidebar.header("Session")
session_id = st.sidebar.text_input("Session ID", value="default", help="Unique ID for document store")

# Tabbed interface
tab1, tab2, tab3 = st.tabs(["📄 Ingest Documents", "🔍 RAG Query", "🤖 Agent Query"])

with tab1:
    st.header("Upload and Index PDFs")
    uploaded_files = st.file_uploader("Choose PDF files", type="pdf", accept_multiple_files=True)
    if st.button("Ingest Documents") and uploaded_files:
        with st.spinner("Indexing..."):
            upload_dir = Path("./data/uploads") / session_id
            upload_dir.mkdir(parents=True, exist_ok=True)
            
            for file in uploaded_files:
                file_path = upload_dir / file.name
                with open(file_path, "wb") as buffer:
                    shutil.copyfileobj(file, buffer)
            
            ingest_documents(session_id, str(upload_dir))
            st.success(f"Indexed {len(uploaded_files)} documents for session '{session_id}'!")
    elif st.button("Load Sample Data"):
        with st.spinner("Loading sample US Census data..."):
            ingest_documents(session_id, "./data/us_census")
            st.success("Loaded sample data!")

with tab2:
    st.header("RAG Query")
    question = st.text_area("Ask a question based on documents:", height=100)
    if st.button("Query RAG") and question:
        if session_id not in st.session_state.get("vector_stores", {}):
            st.warning("No documents ingested. Use the Ingest tab first.")
        else:
            with st.spinner("Querying..."):
                start = time.time()
                result = query_rag(question, session_id)
                end = time.time()
                
                st.subheader("Answer")
                st.write(result["answer"])
                st.subheader("Sources")
                for src in result["sources"]:
                    st.write(f"- {src}")
                st.caption(f"Response time: {result['response_time']:.2f}s")

with tab3:
    st.header("Agent Query")
    question = st.text_area("Ask the agent (uses tools like Wikipedia, ArXiv, and docs):", height=100)
    if st.button("Query Agent") and question:
        if session_id not in st.session_state.get("vector_stores", {}):
            st.warning("No documents ingested. Use the Ingest tab first.")
        else:
            with st.spinner("Agent thinking..."):
                result = query_agent(question, session_id)
                st.write(result["answer"])

# Footer
st.sidebar.markdown("---")
st.sidebar.info("Powered by Groq + OpenAI | Inspired by LangChain docs")
