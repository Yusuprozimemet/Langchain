import os
from pathlib import Path

def create_file(filepath: Path, content: str = ""):
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(content)

def main():
    root = Path(__file__).parent

    # README.md
    readme_content = """# LangChain RAG & Agent API with FastAPI

This project is a FastAPI-based API service inspired by Streamlit demos. It implements a Retrieval-Augmented Generation (RAG) system for document Q&A, with support for PDF ingestion, multiple vector stores (FAISS and ObjectBox), and an optional agent endpoint using tools (e.g., Wikipedia, ArXiv). It uses Groq (Llama3) as the primary LLM for cost-efficiency, with OpenAI fallback for embeddings.

## Key Features
- **Document Ingestion**: Upload and index PDFs (e.g., US Census data) into vector stores.
- **RAG Query Endpoint**: Retrieve relevant chunks and generate answers using Groq.
- **Agent Endpoint**: Query an agent that can use RAG retriever + external tools (Wikipedia, ArXiv).
- **Session State**: In-memory vector stores per session (extendable to persistent DB).
- **Monitoring**: Basic logging; integrate LangSmith tracing via env vars.
- **Deployment-Ready**: FastAPI with async support, Pydantic validation, and CORS.

## Setup Instructions
1. Copy `.env.example` to `.env` and fill in keys (e.g., `GROQ_API_KEY`, `OPENAI_API_KEY`).
2. Install deps: `pip install -r requirements.txt`.
3. Run: `uvicorn app.main:app --reload`.
4. Access docs: `http://localhost:8000/docs`.

## API Endpoints
- POST /api/v1/rag/ingest: Upload PDFs and index them.
- POST /api/v1/rag/query: Query the RAG system.
- POST /api/v1/agent/query: Query the agent.
"""
    create_file(root / 'README.md', readme_content)

    # requirements.txt
    reqs_content = """fastapi==0.104.1
uvicorn[standard]==0.24.0
pydantic==2.5.0
pydantic-settings==2.0.3
python-dotenv==1.0.0
langchain==0.1.0
langchain-groq==0.0.2
langchain-openai==0.0.2
langchain-community==0.0.10
langchain-core==0.1.0
langchain-text-splitters==0.0.1
faiss-cpu==1.7.4
python-multipart==0.0.6
arxiv==1.4.2
wikipedia==1.4.0
"""
    create_file(root / 'requirements.txt', reqs_content)

    # .env.example
    env_content = """GROQ_API_KEY=your_groq_key_here
OPENAI_API_KEY=your_openai_key_here
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=your_langsmith_key_here  # Optional for tracing
"""
    create_file(root / '.env.example', env_content)

    # .gitignore
    gitignore_content = """.env
__pycache__/
*.pyc
data/uploads/
.pytest_cache/
"""
    create_file(root / '.gitignore', gitignore_content)

    # app/__init__.py
    create_file(root / 'app' / '__init__.py')

    # app/main.py
    main_content = """from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api.v1.routers import main_router
from app.config.settings import settings

app = FastAPI(title="LangChain RAG & Agent API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(main_router, prefix="/api/v1")

@app.get("/")
def root():
    return {"message": "LangChain FastAPI Demo - See /docs for endpoints"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
"""
    create_file(root / 'app' / 'main.py', main_content)

    # app/config/__init__.py
    create_file(root / 'app' / 'config' / '__init__.py')

    # app/config/settings.py
    settings_content = """from pydantic_settings import BaseSettings
from dotenv import load_dotenv

load_dotenv()

class Settings(BaseSettings):
    groq_api_key: str
    openai_api_key: str
    langchain_tracing_v2: bool = True
    langchain_api_key: str | None = None
    
    class Config:
        env_file = ".env"

settings = Settings()
"""
    create_file(root / 'app' / 'config' / 'settings.py', settings_content)

    # app/core/__init__.py
    create_file(root / 'app' / 'core' / '__init__.py')

    # app/core/security.py
    security_content = """from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from app.config.settings import settings

security = HTTPBearer()

async def get_api_key(credentials: HTTPAuthorizationCredentials = Depends(security)):
    if credentials.credentials != settings.groq_api_key:  # Simple key auth; use JWT in prod
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid API key")
    return credentials
"""
    create_file(root / 'app' / 'core' / 'security.py', security_content)

    # app/api/__init__.py
    create_file(root / 'app' / 'api' / '__init__.py')

    # app/api/deps.py
    deps_content = """from app.api.v1.routers import main_router
"""
    create_file(root / 'app' / 'api' / 'deps.py', deps_content)

    # app/api/v1/__init__.py
    create_file(root / 'app' / 'api' / 'v1' / '__init__.py')

    # app/api/v1/routers.py
    routers_content = """from fastapi import APIRouter
from .endpoints import rag, agent

main_router = APIRouter()
main_router.include_router(rag.router)
main_router.include_router(agent.router)
"""
    create_file(root / 'app' / 'api' / 'v1' / 'routers.py', routers_content)

    # app/api/v1/endpoints/__init__.py
    create_file(root / 'app' / 'api' / 'v1' / 'endpoints' / '__init__.py')

    # app/api/v1/endpoints/rag.py
    rag_endpoint_content = """from fastapi import APIRouter, Depends, UploadFile, File, HTTPException
from app.models.pydantic_models import QueryRequest, QueryResponse
from app.services.rag_service import query_rag
from app.services.vector_store_service import ingest_documents
from app.core.security import get_api_key
import shutil
from pathlib import Path
import os

router = APIRouter(prefix="/rag", tags=["RAG"])

@router.post("/ingest")
async def ingest_docs(
    files: list[UploadFile] = File(...),
    session_id: str = "default",
    api_key: str = Depends(get_api_key)
):
    upload_dir = Path("./data/uploads") / session_id
    upload_dir.mkdir(parents=True, exist_ok=True)
    
    for file in files:
        file_path = upload_dir / file.filename
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
    
    ingest_documents(session_id, str(upload_dir))
    return {"message": f"Indexed {len(files)} documents for session {session_id}"}

@router.post("/query", response_model=QueryResponse)
async def query(
    request: QueryRequest,
    api_key: str = Depends(get_api_key)
):
    try:
        result = query_rag(request.question, request.session_id or "default")
        return QueryResponse(**result)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
"""
    create_file(root / 'app' / 'api' / 'v1' / 'endpoints' / 'rag.py', rag_endpoint_content)

    # app/api/v1/endpoints/agent.py
    agent_endpoint_content = """from fastapi import APIRouter, Depends
from app.models.pydantic_models import QueryRequest
from app.services.agent_service import query_agent
from pydantic import BaseModel
from app.core.security import get_api_key

router = APIRouter(prefix="/agent", tags=["Agent"])

class AgentResponse(BaseModel):
    answer: str

@router.post("/query", response_model=AgentResponse)
async def agent_query(
    request: QueryRequest,
    api_key: str = Depends(get_api_key)
):
    result = query_agent(request.question, request.session_id or "default")
    return AgentResponse(answer=result["answer"])
"""
    create_file(root / 'app' / 'api' / 'v1' / 'endpoints' / 'agent.py', agent_endpoint_content)

    # app/models/__init__.py
    create_file(root / 'app' / 'models' / '__init__.py')

    # app/models/pydantic_models.py
    pydantic_models_content = """from pydantic import BaseModel
from typing import List, Optional

class QueryRequest(BaseModel):
    question: str
    vector_store: str = "faiss"  # 'faiss' or 'objectbox'
    session_id: Optional[str] = None

class QueryResponse(BaseModel):
    answer: str
    sources: List[str]
    response_time: float
"""
    create_file(root / 'app' / 'models' / 'pydantic_models.py', pydantic_models_content)

    # app/models/db_models.py (placeholder)
    create_file(root / 'app' / 'models' / 'db_models.py', "# Placeholder for DB models if needed")

    # app/services/__init__.py
    create_file(root / 'app' / 'services' / '__init__.py')

    # app/services/llm_service.py
    llm_service_content = """from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from app.config.settings import settings

def get_llm(model_name: str = "llama3-8b-8192"):
    return ChatGroq(groq_api_key=settings.groq_api_key, model_name=model_name, temperature=0)

def get_openai_llm():
    return ChatOpenAI(model="gpt-3.5-turbo-0125", temperature=0, openai_api_key=settings.openai_api_key)

def get_embeddings():
    return OpenAIEmbeddings(openai_api_key=settings.openai_api_key)
"""
    create_file(root / 'app' / 'services' / 'llm_service.py', llm_service_content)

    # app/services/vector_store_service.py
    vector_service_content = """from langchain_community.vectorstores import FAISS
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
"""
    create_file(root / 'app' / 'services' / 'vector_store_service.py', vector_service_content)

    # app/services/rag_service.py
    rag_service_content = '''import time
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from app.services.llm_service import get_llm
from app.services.vector_store_service import get_retriever
from typing import Dict, Any

prompt = ChatPromptTemplate.from_template(
    """
Answer the questions based on the provided context only.
Please provide the most accurate response based on the question.
<context>
{context}
</context>
Question: {input}
"""
)

def query_rag(question: str, session_id: str) -> Dict[str, Any]:
    start = time.process_time()
    llm = get_llm()
    retriever = get_retriever(session_id)
    doc_chain = create_stuff_documents_chain(llm, prompt)
    chain = create_retrieval_chain(retriever, doc_chain)
    
    response = chain.invoke({"input": question})
    response_time = time.process_time() - start
    
    sources = [doc.metadata.get('source', 'Unknown') for doc in response["context"]]
    
    return {
        "answer": response['answer'],
        "sources": sources,
        "response_time": response_time
    }
'''
    create_file(root / 'app' / 'services' / 'rag_service.py', rag_service_content)

    # app/services/agent_service.py
    agent_service_content = """from langchain_community.tools import WikipediaQueryRun, ArxivQueryRun
from langchain_community.utilities import WikipediaAPIWrapper, ArxivAPIWrapper
from langchain import hub
from langchain.agents import create_openai_tools_agent, AgentExecutor
from app.services.llm_service import get_openai_llm, get_embeddings
from app.services.vector_store_service import get_retriever
from langchain.tools.retriever import create_retriever_tool
from app.config.settings import settings
from typing import Dict, Any

def create_agent(session_id: str) -> AgentExecutor:
    # Tools
    wiki_wrapper = WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=200)
    wiki_tool = WikipediaQueryRun(api_wrapper=wiki_wrapper)
    
    arxiv_wrapper = ArxivAPIWrapper(top_k_results=1, doc_content_chars_max=200)
    arxiv_tool = ArxivQueryRun(api_wrapper=arxiv_wrapper)
    
    retriever = get_retriever(session_id)
    retriever_tool = create_retriever_tool(
        retriever, "doc_search", "Search for information in uploaded documents."
    )
    
    tools = [wiki_tool, arxiv_tool, retriever_tool]
    
    llm = get_openai_llm()
    prompt = hub.pull("hwchase17/openai-functions-agent")
    agent = create_openai_tools_agent(llm, tools, prompt)
    return AgentExecutor(agent=agent, tools=tools, verbose=True)

def query_agent(question: str, session_id: str) -> Dict[str, Any]:
    agent_executor = create_agent(session_id)
    response = agent_executor.invoke({"input": question})
    return {"answer": response['output']}
"""
    create_file(root / 'app' / 'services' / 'agent_service.py', agent_service_content)

    # app/utils/__init__.py
    create_file(root / 'app' / 'utils' / '__init__.py')

    # app/utils/document_loader.py (placeholder)
    create_file(root / 'app' / 'utils' / 'document_loader.py', "# Additional document loading utilities")

    # app/utils/embedding_helper.py (placeholder)
    create_file(root / 'app' / 'utils' / 'embedding_helper.py', "# Embedding helper functions")

    # app/schemas/__init__.py
    create_file(root / 'app' / 'schemas' / '__init__.py')

    # app/schemas/responses.py (placeholder)
    create_file(root / 'app' / 'schemas' / 'responses.py', "# Additional response schemas")

    # data/
    (root / 'data').mkdir(exist_ok=True)
    (root / 'data' / 'uploads').mkdir(exist_ok=True)
    (root / 'data' / 'us_census').mkdir(exist_ok=True)

    # tests/__init__.py
    create_file(root / 'tests' / '__init__.py')

    # tests/test_rag.py
    test_content = """import pytest
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_root():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "LangChain FastAPI Demo - See /docs for endpoints"}

# Note: For full tests, mock services and add auth headers
"""
    create_file(root / 'tests' / 'test_rag.py', test_content)

    print(f"Project structure generated in '{root}' directory.")

if __name__ == "__main__":
    main()