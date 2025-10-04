from fastapi import APIRouter
from .endpoints import rag, agent

main_router = APIRouter()
main_router.include_router(rag.router)
main_router.include_router(agent.router)
