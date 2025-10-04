import time
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
