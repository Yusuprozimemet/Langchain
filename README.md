# LangChain RAG & Agent API with FastAPI + Streamlit Demo

This project is a FastAPI-based API service inspired by Streamlit demos. It implements a Retrieval-Augmented Generation (RAG) system for document Q&A, with support for PDF ingestion, multiple vector stores (FAISS and ObjectBox), and an optional agent endpoint using tools (e.g., Wikipedia, ArXiv). It uses Groq (Llama3) as the primary LLM for cost-efficiency, with OpenAI fallback for embeddings.

## Key Features
- **Document Ingestion**: Upload and index PDFs (e.g., US Census data) into vector stores.
- **RAG Query Endpoint**: Retrieve relevant chunks and generate answers using Groq.
- **Agent Endpoint**: Query an agent that can use RAG retriever + external tools (Wikipedia, ArXiv).
- **Session State**: In-memory vector stores per session (extendable to persistent DB).
- **Monitoring**: Basic logging; integrate LangSmith tracing via env vars.
- **Deployment-Ready**: FastAPI with async support, Pydantic validation, and CORS.
- **Streamlit Demo**: Interactive UI for ingestion and querying (runs alongside or independently).

## Setup Instructions
1. Copy `.env.example` to `.env` and fill in keys (e.g., `GROQ_API_KEY`, `OPENAI_API_KEY`).
2. Install deps: `pip install -r requirements.txt`.
3. For API: Run `uvicorn app.main:app --reload` and access docs at `http://localhost:8000/docs`.
4. For Streamlit Demo: Run `streamlit run streamlit_app.py` and access at `http://localhost:8501`.
   - Note: Streamlit uses direct service calls (no auth required for demo; add if needed).

## API Endpoints
- POST /api/v1/rag/ingest: Upload PDFs and index them.
- POST /api/v1/rag/query: Query the RAG system.
- POST /api/v1/agent/query: Query the agent.
