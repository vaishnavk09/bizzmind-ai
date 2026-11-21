from langchain.tools import tool
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pipeline.tools.base import ToolContext
    from pipeline.vector_store import VectorStoreManager
    from langchain_groq import ChatGroq

# Legacy globals kept for backward-compat
_VSM = None
_LLM = None


def init_insight_generator(vsm, llm):
    global _VSM, _LLM
    _VSM = vsm
    _LLM = llm


def _run_insight(vsm, llm, question: str) -> str:
    chunks = vsm.similarity_search(question, k=7)
    if not chunks:
        return "I couldn't find relevant data for your question."

    context = "\n".join(chunks)
    prompt = (
        "You are a business analyst AI helping a small shop owner in India.\n"
        "Answer the question below using ONLY the provided sales data context.\n"
        "Be concise. Use specific numbers with ₹ for currency.\n"
        "If the context doesn't contain enough information, say so.\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {question}\n\n"
        "Answer:"
    )
    response = llm.invoke(prompt)
    return response.content


def make_insight_tool(ctx: "ToolContext", llm):
    """Returns a generate_insight tool bound to the given ToolContext and LLM."""

    @tool
    def generate_insight(question: str) -> str:
        """
        Answers free-text business questions by searching the sales data vector store.
        Use for questions like 'who are my best customers?' or 'what sells on weekends?'.
        Input: a natural language question about the business.
        """
        if ctx.vsm is None or llm is None:
            return "Error: Insight generator not initialized."
        return _run_insight(ctx.vsm, llm, question)

    return generate_insight


@tool
def generate_insight(question: str) -> str:
    """
    Answers free-text business questions by searching the sales data vector store.
    Use for questions like 'who are my best customers?' or 'what sells on weekends?'.
    Input: a natural language question about the business.
    """
    if _VSM is None or _LLM is None:
        return "Error: Insight generator not initialized."
    return _run_insight(_VSM, _LLM, question)
