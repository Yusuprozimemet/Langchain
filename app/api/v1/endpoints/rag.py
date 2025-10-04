from fastapi import APIRouter, Depends, UploadFile, File, HTTPException
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
