from pydantic_settings import BaseSettings
from dotenv import load_dotenv

load_dotenv()

class Settings(BaseSettings):
    openai_api_key: str  # Only OpenAI now
    langchain_tracing_v2: bool = True
    langchain_api_key: str | None = None
    
    class Config:
        env_file = ".env"

settings = Settings()
