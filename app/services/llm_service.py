from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from app.config.settings import settings

def get_llm(model_name: str = "llama3-8b-8192"):
    return ChatGroq(groq_api_key=settings.groq_api_key, model_name=model_name, temperature=0)

def get_openai_llm():
    return ChatOpenAI(model="gpt-3.5-turbo-0125", temperature=0, openai_api_key=settings.openai_api_key)

def get_embeddings():
    return OpenAIEmbeddings(openai_api_key=settings.openai_api_key)
