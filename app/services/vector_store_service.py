from langchain_community.vectorstores import FAISS
# from langchain_objectbox.vectorstores import ObjectBox  # Uncomment if using ObjectBox
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from app.services.llm_service import get_embeddings
from typing import Dict, Any

# In-memory store (use session_id as key)
vector_stores: Dict[str, Any] = {}

def ingest_documents(session_id: str, directory: str = "./data/us_census"):
    if session_id not in vector_stores:
        embeddings = get_embeddings()
        loader = PyPDFDirectoryLoader(directory)
        docs = loader.load()[:20]  # Limit as in demo
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        split_docs = splitter.split_documents(docs)
        
        # Default to FAISS; switch based on config
        vector_stores[session_id] = FAISS.from_documents(split_docs, embeddings)
        # For ObjectBox: vector_stores[session_id] = ObjectBox.from_documents(split_docs, embeddings, embedding_dimensions=768)
    
    return vector_stores[session_id]

def get_retriever(session_id: str):
    store = vector_stores.get(session_id)
    if not store:
        raise ValueError("No vector store for session")
    return store.as_retriever()
