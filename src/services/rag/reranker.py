"""Reranking of retrieved candidates using a cross-encoder model."""

from sentence_transformers import CrossEncoder

from config import settings

from .logger import get_logger
from .paper_hit import PaperHit

logger = get_logger(__name__)

class Reranker:
    def __init__(self):
        
        self._reranker = CrossEncoder(settings.rerank_model_id)

    def select_top(self, query: str, hits: list[PaperHit], top_k: int) -> list[PaperHit]:
        """Rerank candidate abstracts against a query using a cross-encoder.

        Args:
            query: The search query to rerank candidates against.
            hits: Candidate PaperHit objects to be reranked.
            top_k: Number of top-scoring hits to return.

        Returns:
            The top_k PaperHit objects, ordered by descending relevance score.
        """

        logger.info(f"Selecting the top {top_k}")
        abstracts = [hit.abstract for hit in hits]
        pairs = [(query, doc) for doc in abstracts]

        scores = self._reranker.predict(pairs)
        
        # Sort hits by score descending
        scored_hits = sorted(zip(hits, scores), key=lambda x: x[1], reverse=True)

        return [hit for hit, _ in scored_hits[:top_k]]
