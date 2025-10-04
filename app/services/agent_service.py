from langchain_community.tools import WikipediaQueryRun, ArxivQueryRun
from langchain_community.utilities import WikipediaAPIWrapper, ArxivAPIWrapper
from langchain import hub
from langchain.agents import create_openai_tools_agent, AgentExecutor
from app.services.llm_service import get_llm, get_embeddings  
from app.services.vector_store_service import get_retriever
from langchain.tools.retriever import create_retriever_tool
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
    
    llm = get_llm("gpt-3.5-turbo")  # Use OpenAI for agent
    prompt = hub.pull("hwchase17/openai-functions-agent")
    agent = create_openai_tools_agent(llm, tools, prompt)
    return AgentExecutor(agent=agent, tools=tools, verbose=True)

def query_agent(question: str, session_id: str) -> Dict[str, Any]:
    agent_executor = create_agent(session_id)
    response = agent_executor.invoke({"input": question})
    return {"answer": response['output']}