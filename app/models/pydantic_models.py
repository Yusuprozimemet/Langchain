from pydantic import BaseModel
from typing import List, Optional

class QueryRequest(BaseModel):
    question: str
    vector_store: str = "faiss"  # 'faiss' or 'objectbox'
    session_id: Optional[str] = None

class QueryResponse(BaseModel):
    answer: str
    sources: List[str]
    response_time: float
