from langchain.tools import tool
from pipeline.vector_store import VectorStoreManager
from langchain_groq import ChatGroq

_VSM = None
_LLM = None

def init_insight_generator(vsm: VectorStoreManager, llm: ChatGroq):
    global _VSM, _LLM
    _VSM = vsm
    _LLM = llm

@tool
def generate_insight(question: str) -> str:
    """
    Useful for answering free-text questions about the business by querying the data store.
    Input should be the question to answer.
    """
    if _VSM is None or _LLM is None:
        return "Error: Insight generator not initialized."

    # Retrieve relevant chunks
    chunks = _VSM.similarity_search(question, k=5)
    
    if not chunks:
        return "I couldn't find relevant data for your question."
        
    context = "\n".join(chunks)
    
    prompt = f"""
    You are a business analyst AI.
    Answer the following question based ONLY on the provided context.
    Use simple language. Provide specific numbers if available.
    
    Context:
    {context}
    
    Question: {question}
    
    Answer:
    """
    
    response = _LLM.invoke(prompt)
    return response.content
