import os
import logging
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from rank_bm25 import BM25Okapi

logger = logging.getLogger(__name__)


class VectorStoreManager:
    """
    Hybrid retrieval: dense FAISS + sparse BM25, fused via Reciprocal Rank Fusion (RRF).

    Why hybrid?
    - FAISS/dense embeddings excel at semantic similarity ("products that sell well on weekends").
    - BM25/sparse excels at exact keyword match ("CUST042", "Rice", "2024-03-15").
    - RRF merges both ranked lists so neither signal dominates.
    """

    def __init__(self, index_path: str = "faiss_index"):
        self.index_path = index_path
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        self.vector_store: FAISS | None = None
        self._chunks: list[str] = []
        self._bm25: BM25Okapi | None = None

    # ------------------------------------------------------------------
    # Build / persist
    # ------------------------------------------------------------------

    def build_store(self, text_chunks: list[str]) -> None:
        """Builds FAISS index and BM25 corpus from text chunks, then persists."""
        logger.info("Building hybrid vector store with %d chunks…", len(text_chunks))
        self._chunks = text_chunks

        # Dense index
        self.vector_store = FAISS.from_texts(text_chunks, self.embeddings)
        os.makedirs(self.index_path, exist_ok=True)
        self.vector_store.save_local(self.index_path)

        # Sparse index (BM25 on whitespace-tokenised chunks)
        tokenised = [c.lower().split() for c in text_chunks]
        self._bm25 = BM25Okapi(tokenised)
        logger.info("Hybrid store built and saved to %s.", self.index_path)

    def load_store(self) -> None:
        """Loads the persisted FAISS index. BM25 must be rebuilt from chunks."""
        if not os.path.exists(self.index_path):
            logger.warning("Index path %s does not exist.", self.index_path)
            return
        try:
            self.vector_store = FAISS.load_local(
                self.index_path,
                self.embeddings,
                allow_dangerous_deserialization=True,
            )
            logger.info("FAISS index loaded from %s.", self.index_path)
        except Exception:
            logger.exception("Failed to load FAISS index.")

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------

    def similarity_search(self, query: str, k: int = 7) -> list[str]:
        """
        Hybrid search: merges top-k from FAISS and BM25 via RRF, returns top-k unique chunks.
        Falls back to dense-only if BM25 is unavailable (e.g., after load_store).
        """
        if not self.vector_store:
            logger.warning("Vector store not initialised.")
            return []

        # --- Dense retrieval ---
        dense_results = self.vector_store.similarity_search(query, k=k)
        dense_chunks = [r.page_content for r in dense_results]

        if not self._bm25 or not self._chunks:
            # BM25 not available (e.g., loaded from disk without re-building)
            return dense_chunks[:k]

        # --- Sparse retrieval ---
        tokens = query.lower().split()
        bm25_scores = self._bm25.get_scores(tokens)
        # Get indices of top-k BM25 results
        top_bm25_indices = sorted(range(len(bm25_scores)), key=lambda i: bm25_scores[i], reverse=True)[:k]
        sparse_chunks = [self._chunks[i] for i in top_bm25_indices]

        # --- Reciprocal Rank Fusion ---
        rrf_scores: dict[str, float] = {}
        rrf_k = 60  # standard RRF constant

        for rank, chunk in enumerate(dense_chunks):
            rrf_scores[chunk] = rrf_scores.get(chunk, 0.0) + 1.0 / (rrf_k + rank + 1)

        for rank, chunk in enumerate(sparse_chunks):
            rrf_scores[chunk] = rrf_scores.get(chunk, 0.0) + 1.0 / (rrf_k + rank + 1)

        fused = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
        return [chunk for chunk, _ in fused[:k]]
