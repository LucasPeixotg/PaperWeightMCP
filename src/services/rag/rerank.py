"""Reranking of retrieved candidates.

Not implemented yet. This module exists with its real signature so the package
imports cleanly; `RagPipeline.retrieve` only calls it when `use_rerank=True`,
which is not the default.
"""

from .rag_model import RagModel


def rerank(model: RagModel, query: str, abstracts: list[str], top_k: int) -> list[str]:
    """Re-sort `abstracts` by relevance to `query`, best first."""
    raise NotImplementedError("reranking is not implemented yet; call with use_rerank=False")
