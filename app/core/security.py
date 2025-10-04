from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from app.config.settings import settings

security = HTTPBearer()

async def get_api_key(credentials: HTTPAuthorizationCredentials = Depends(security)):
    if credentials.credentials != settings.openai_api_key:  # Now checks OpenAI key
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid API key")
    return credentials