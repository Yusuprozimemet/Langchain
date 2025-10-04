from fastapi import APIRouter, Depends
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
