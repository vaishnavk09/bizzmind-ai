"""
RAG Evaluation Module — BizMind AI
===================================
Measures retrieval quality and answer faithfulness without requiring an external
eval framework. Two metrics are computed:

1. Context Recall  — does the retrieved context contain the keywords expected
                     for each golden question? (token-overlap proxy)
2. Faithfulness    — does the LLM answer stay grounded in the retrieved context?
                     (checks whether key numbers/entities from context appear in answer)

Usage:
    python -m eval.rag_eval

Requires GROQ_API_KEY in .env and the mock dataset to exist.
Run `python data/generate_mock.py` first if needed.
"""

import os
import sys
import json
import logging
from dataclasses import dataclass, field, asdict
from typing import Optional

# Add project root to path so pipeline imports work when run directly
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from dotenv import load_dotenv
load_dotenv()

from pipeline.ingestion import DataIngestionPipeline
from pipeline.vector_store import VectorStoreManager
from langchain_groq import ChatGroq

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger("bizmind.eval")


# ------------------------------------------------------------------
# Golden evaluation set
# Each entry has:
#   question        — the query
#   expected_tokens — keywords that MUST appear in retrieved context
#   expected_answer — reference answer (used for faithfulness check)
# ------------------------------------------------------------------

GOLDEN_SET = [
    {
        "question": "Which customer bought the most Rice?",
        "expected_tokens": ["rice"],
        "expected_answer_keywords": ["rice", "cust"],
    },
    {
        "question": "What payment mode is most popular?",
        "expected_tokens": ["cash", "upi", "credit"],
        "expected_answer_keywords": ["cash", "upi"],
    },
    {
        "question": "Which product generates the highest revenue?",
        "expected_tokens": ["revenue"],
        "expected_answer_keywords": ["₹"],
    },
    {
        "question": "Were there any sales on weekends?",
        "expected_tokens": ["saturday", "sunday"],
        "expected_answer_keywords": ["saturday", "sunday", "weekend"],
    },
    {
        "question": "What is the average unit price of Cooking Oil?",
        "expected_tokens": ["cooking oil"],
        "expected_answer_keywords": ["cooking oil", "₹"],
    },
    {
        "question": "Which category had the most orders?",
        "expected_tokens": ["grocery", "snacks", "personal care"],
        "expected_answer_keywords": ["grocery", "snacks", "personal care"],
    },
    {
        "question": "Show me recent sales for CUST001",
        "expected_tokens": ["cust001"],
        "expected_answer_keywords": ["cust001"],
    },
]


# ------------------------------------------------------------------
# Metric helpers
# ------------------------------------------------------------------

def _token_overlap(text: str, tokens: list[str]) -> float:
    """Fraction of expected tokens found in text (case-insensitive)."""
    lower = text.lower()
    hits = sum(1 for t in tokens if t.lower() in lower)
    return hits / len(tokens) if tokens else 0.0


@dataclass
class EvalResult:
    question: str
    context_recall: float          # 0–1: fraction of expected tokens in context
    answer_faithfulness: float     # 0–1: fraction of expected answer keywords in answer
    retrieved_chunks: list[str] = field(default_factory=list)
    answer: str = ""
    passed: bool = False


def evaluate_single(
    question: str,
    expected_tokens: list[str],
    expected_answer_keywords: list[str],
    vsm: VectorStoreManager,
    llm: ChatGroq,
    k: int = 7,
) -> EvalResult:
    # --- Retrieval ---
    chunks = vsm.similarity_search(question, k=k)
    context = "\n".join(chunks)

    context_recall = _token_overlap(context, expected_tokens)

    # --- Generation ---
    prompt = (
        "You are a business analyst. Answer based ONLY on the context below.\n"
        f"Context:\n{context}\n\nQuestion: {question}\nAnswer:"
    )
    response = llm.invoke(prompt)
    answer = response.content

    faithfulness = _token_overlap(answer, expected_answer_keywords)

    result = EvalResult(
        question=question,
        context_recall=context_recall,
        answer_faithfulness=faithfulness,
        retrieved_chunks=chunks,
        answer=answer,
        passed=(context_recall >= 0.5 and faithfulness >= 0.5),
    )
    return result


def run_eval(output_path: Optional[str] = None) -> dict:
    """
    Runs the full eval suite and returns a summary dict.
    Optionally writes results to a JSON file at output_path.
    """
    # Build pipeline
    csv_path = os.path.join(os.path.dirname(__file__), "..", "data", "mock_store.csv")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(
            f"Mock data not found at {csv_path}. Run: python data/generate_mock.py"
        )

    logger.info("Loading data from %s", csv_path)
    pipeline = DataIngestionPipeline()
    df = pipeline.load_csv(csv_path)
    df = pipeline.clean_data(df)
    df = pipeline.add_features(df)
    chunks = pipeline.to_text_chunks(df)

    vsm = VectorStoreManager(index_path=os.path.join(os.path.dirname(__file__), "..", "faiss_index"))
    vsm.build_store(chunks)

    api_key = os.environ.get("GROQ_API_KEY", "")
    llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0, api_key=api_key)

    results: list[EvalResult] = []
    for entry in GOLDEN_SET:
        logger.info("Evaluating: %s", entry["question"][:60])
        r = evaluate_single(
            question=entry["question"],
            expected_tokens=entry["expected_tokens"],
            expected_answer_keywords=entry["expected_answer_keywords"],
            vsm=vsm,
            llm=llm,
        )
        results.append(r)
        status = "✅ PASS" if r.passed else "❌ FAIL"
        logger.info(
            "  %s | recall=%.2f | faithfulness=%.2f",
            status, r.context_recall, r.answer_faithfulness,
        )

    avg_recall = sum(r.context_recall for r in results) / len(results)
    avg_faith = sum(r.answer_faithfulness for r in results) / len(results)
    pass_rate = sum(1 for r in results if r.passed) / len(results)

    summary = {
        "total_questions": len(results),
        "pass_rate": round(pass_rate, 3),
        "avg_context_recall": round(avg_recall, 3),
        "avg_answer_faithfulness": round(avg_faith, 3),
        "results": [asdict(r) for r in results],
    }

    print("\n" + "=" * 60)
    print("BizMind AI — RAG Evaluation Summary")
    print("=" * 60)
    print(f"  Questions evaluated : {summary['total_questions']}")
    print(f"  Pass rate           : {pass_rate*100:.0f}%")
    print(f"  Avg context recall  : {avg_recall:.2f}")
    print(f"  Avg faithfulness    : {avg_faith:.2f}")
    print("=" * 60)

    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        logger.info("Results written to %s", output_path)

    return summary


if __name__ == "__main__":
    out = os.path.join(os.path.dirname(__file__), "eval_results.json")
    run_eval(output_path=out)
