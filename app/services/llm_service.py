from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from app.config.settings import settings

def get_llm(model_name: str = "gpt-4o-mini"):  # Or "gpt-3.5-turbo" for cheaper/faster
    return ChatOpenAI(
        model=model_name,
        temperature=0,
        openai_api_key=settings.openai_api_key
    )

# Deprecated now; use get_llm() everywhere
def get_openai_llm():
    return get_llm("gpt-3.5-turbo")  # Keep for agent if preferred

def get_embeddings():
    return OpenAIEmbeddings(openai_api_key=settings.openai_api_key)