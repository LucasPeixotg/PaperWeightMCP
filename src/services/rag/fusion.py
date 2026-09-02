"""Reciprocal Rank Fusion of several ranked candidate lists.

The two retrieval arms score on incomparable scales — FAISS returns inner
products, `ts_rank_cd` returns a length-normalized term-density score — so the
fusion deliberately reads only *position*. Nothing has to be normalized, and a
list that comes back empty contributes nothing rather than skewing a blend.
"""

from __future__ import annotations

from collections import defaultdict

from .paper_hit import PaperHit


def reciprocal_rank_fusion(
    ranked_lists: list[list[PaperHit]], *, k: int, limit: int
) -> list[PaperHit]:
    """Merge ranked lists into one, scoring each paper by sum of 1 / (k + rank).

    Args:
        ranked_lists: Candidate lists, each already ordered best-first.
        k: Damping constant. Larger values flatten the weight of top ranks.
        limit: Maximum number of fused candidates to return.

    Returns:
        Deduplicated hits ordered by descending fused score, truncated to limit.
    """
    scores: dict[str, float] = defaultdict(float)
    hits: dict[str, PaperHit] = {}

    for ranked in ranked_lists:
        for rank, hit in enumerate(ranked):
            scores[hit.paper_id] += 1 / (k + rank)
            # Both arms select the same columns, so the first hit seen for an id
            # is as good as any later duplicate.
            hits.setdefault(hit.paper_id, hit)

    if limit <= 0:
        return []

    best = sorted(scores, key=scores.__getitem__, reverse=True)[:limit]
    return [hits[paper_id] for paper_id in best]
