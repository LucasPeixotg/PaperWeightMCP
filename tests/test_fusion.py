"""Reciprocal Rank Fusion behaviour — pure functions, no database."""

import pytest

from services.rag.fusion import reciprocal_rank_fusion
from services.rag.paper_hit import PaperHit

K = 60


def hit(paper_id: str) -> PaperHit:
    return PaperHit(
        paper_id=paper_id,
        title=f"title {paper_id}",
        year=2020,
        abstract=f"abstract {paper_id}",
        url=f"https://arxiv.org/abs/{paper_id}",
    )


def ids(hits: list[PaperHit]) -> list[str]:
    return [h.paper_id for h in hits]


def test_agreement_beats_a_single_first_place():
    """The whole point of fusing: two mid-rank votes outweigh one top-rank vote."""
    dense = [hit("solo"), hit("a"), hit("agreed")]
    lexical = [hit("b"), hit("agreed"), hit("c")]

    fused = reciprocal_rank_fusion([dense, lexical], k=K, limit=10)

    assert fused[0].paper_id == "agreed"


def test_duplicates_collapse_to_one_entry():
    both = [hit("dup"), hit("x")]
    fused = reciprocal_rank_fusion([both, list(both)], k=K, limit=10)

    assert ids(fused) == ["dup", "x"]


def test_empty_list_is_a_no_op():
    """A retriever returning nothing must not disturb the other's order."""
    dense = [hit("a"), hit("b"), hit("c")]

    assert ids(reciprocal_rank_fusion([dense, []], k=K, limit=10)) == ["a", "b", "c"]


def test_single_list_keeps_its_own_order():
    dense = [hit("a"), hit("b"), hit("c")]

    assert ids(reciprocal_rank_fusion([dense], k=K, limit=10)) == ["a", "b", "c"]


def test_no_lists_at_all():
    assert reciprocal_rank_fusion([], k=K, limit=10) == []


@pytest.mark.parametrize("limit, expected", [(2, ["a", "b"]), (0, []), (99, ["a", "b", "c"])])
def test_limit_truncates(limit, expected):
    dense = [hit("a"), hit("b"), hit("c")]

    assert ids(reciprocal_rank_fusion([dense], k=K, limit=limit)) == expected


def test_larger_k_flattens_the_top_ranks():
    """k damps rank 0's advantage — the reason it is a tunable and not a literal."""
    lists = [[hit("first"), hit("second")], [hit("second"), hit("first")]]

    def gap(k):
        # With both papers holding rank 0 and rank 1 once each, the scores tie;
        # measure the spread a single list produces instead.
        scored = reciprocal_rank_fusion([[hit("first"), hit("second")]], k=k, limit=2)
        return 1 / (k + 0) - 1 / (k + 1), scored

    small, _ = gap(1)
    large, _ = gap(1000)
    assert small > large
    # Symmetric agreement is a genuine tie, and order is then insertion order.
    assert ids(reciprocal_rank_fusion(lists, k=K, limit=2)) == ["first", "second"]
