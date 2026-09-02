"""Reranking of retrieved candidates using a cross-encoder model."""

from sentence_transformers import CrossEncoder

from config import settings


class Reranker:
    def __init__(self):
        
        self._reranker = CrossEncoder(settings.rerank_model_id)

    def top(self, query: str, abstracts: list[str], top_k: int) -> list[str]:
        """Rerank candidate abstracts against a query using a cross-encoder.

        Args:
            reranker: A loaded CrossEncoder model used to score (query, doc) pairs.
            query: The search query to rerank candidates against.
            abstracts: Candidate documents (e.g. paper abstracts) to be reranked.
            top_k: Number of top-scoring documents to return.

        Returns:
            The top_k abstracts, ordered by descending relevance score.
        """
        pairs = [(query, doc) for doc in abstracts]

        scores = self._reranker.predict(pairs)
        
        # Sort by score descending
        scored_docs = sorted(zip(abstracts, scores), key=lambda x: x[1], reverse=True)

        return scored_docs[:top_k]
