"""rerank — pure logic, with the cross-encoder stubbed out.

The real model is ~90MB of weights fetched from HuggingFace and a torch install
the rest of this tree does not need, so every test here injects a scorer instead.
That is not only about speed: what the function actually owns is the ordering
contract around the scores, and a fake scorer is the only way to state a case
like "these two papers tie" and know it holds.

The cases worth the file are the ones the wiring can get wrong. Sorting the
(paper, score) pairs directly rather than by key reads fine and works right up
until two papers tie, where it compares PaperData and raises — so both the tie
and the ordering are pinned. The guards are checked for *not loading the model*
rather than only for their return value, since skipping that download on a blank
query is the whole reason they run first. And the document handed to the model is
asserted field by field: a paper missing an abstract is common enough in OpenAlex
that a stray separator would be the normal case, not the edge one.
"""

import pytest

from common import PaperData
from services.models import reranker
from services.models.reranker import rerank

ABSTRACT = (
    "We propose a new simple network architecture, the Transformer, based solely "
    "on attention mechanisms, dispensing with recurrence and convolutions entirely."
)


def paper(**overrides) -> PaperData:
    """A complete PaperData so a test can name only the field it cares about."""
    base = {
        "paper_id": "W1",
        "doi": "10.48550/arxiv.1706.03762",
        "title": "Attention Is All You Need",
        "year": 2017,
        "abstract": ABSTRACT,
        "url": "https://arxiv.org/abs/1706.03762",
        "license": "",
    }

    return PaperData(**(base | overrides))


class StubModel:
    """A cross-encoder that returns scores chosen by the test.

    Records the pairs it was given, so a test can assert both what the model was
    asked and — by an empty record — that it was never asked at all.
    """

    def __init__(self, scores=()):
        self.scores = list(scores)
        self.calls = []

    def predict(self, pairs, batch_size=None):
        pairs = list(pairs)
        self.calls.append(pairs)

        # A test that cares about ordering supplies scores; one that does not
        # gets a flat zero per pair rather than an index error.
        return self.scores or [0.0] * len(pairs)


@pytest.fixture
def model(monkeypatch):
    """Install a stub in place of the lazily-loaded singleton.

    `_get_model` returns `_model` untouched when it is already set, so seeding it
    is enough to keep sentence-transformers out of the test run entirely.
    """

    def install(*scores):
        stub = StubModel(scores)
        monkeypatch.setattr(reranker, "_model", stub)
        return stub

    return install


def test_papers_come_back_in_descending_score_order(model):
    """The reordering is the entire point: input order must not survive scoring."""
    first, second, third = paper(paper_id="A"), paper(paper_id="B"), paper(paper_id="C")
    model(0.1, 0.9, 0.5)

    ranked = rerank("attention mechanisms", [first, second, third], top_k=3)

    assert [p.paper_id for p in ranked] == ["B", "C", "A"]


def test_only_top_k_are_returned(model):
    """The pool is deliberately wider than the answer — the surplus is dropped here."""
    papers = [paper(paper_id=str(index)) for index in range(5)]
    model(0.1, 0.2, 0.3, 0.4, 0.5)

    ranked = rerank("attention", papers, top_k=2)

    assert [p.paper_id for p in ranked] == ["4", "3"]


def test_a_top_k_larger_than_the_pool_returns_every_paper(model):
    """Asking for more than exists is not an error; dedup can shrink the pool."""
    papers = [paper(paper_id="A"), paper(paper_id="B")]
    model(0.2, 0.8)

    assert len(rerank("attention", papers, top_k=10)) == 2


def test_papers_scoring_equally_keep_their_input_order(model):
    """Ties fall back to the upstream ranking, which position in the list carries.

    This is also what catches a sort that compares the (paper, score) pairs
    themselves: PaperData has no ordering, so a tie would raise TypeError rather
    than quietly misorder.
    """
    papers = [paper(paper_id="A"), paper(paper_id="B"), paper(paper_id="C")]
    model(0.5, 0.5, 0.5)

    ranked = rerank("attention", papers, top_k=3)

    assert [p.paper_id for p in ranked] == ["A", "B", "C"]


def test_the_model_scores_title_and_abstract_together(model):
    """Both fields are one passage — a title alone rarely settles relevance."""
    stub = model(0.5)

    rerank("attention mechanisms", [paper()], top_k=1)

    (query, document), = stub.calls[0]
    assert query == "attention mechanisms"
    assert document == f"Attention Is All You Need\n\n{ABSTRACT}"


@pytest.mark.parametrize(
    "overrides, expected",
    [
        ({"abstract": ""}, "Attention Is All You Need"),
        ({"title": ""}, ABSTRACT),
        ({"title": "", "abstract": ""}, ""),
    ],
    ids=["no abstract", "no title", "neither"],
)
def test_a_missing_field_leaves_no_stray_separator(model, overrides, expected):
    """PaperData coerces absent values to "", and OpenAlex omits abstracts often.

    Joining unconditionally would open most real documents with a blank line.
    """
    stub = model(0.5)

    rerank("attention", [paper(**overrides)], top_k=1)

    (_, document), = stub.calls[0]
    assert document == expected


def test_a_blank_query_keeps_the_upstream_order_without_loading_the_model(model):
    """There is nothing to score against, and loading weights to learn that is waste."""
    stub = model(0.9, 0.1)
    papers = [paper(paper_id="A"), paper(paper_id="B"), paper(paper_id="C")]

    ranked = rerank("   ", papers, top_k=2)

    assert [p.paper_id for p in ranked] == ["A", "B"]
    assert stub.calls == []


@pytest.mark.parametrize(
    "papers, top_k",
    [([], 5), ([paper()], 0), ([paper()], -1)],
    ids=["empty pool", "top_k of zero", "negative top_k"],
)
def test_nothing_to_rank_returns_empty_without_loading_the_model(model, papers, top_k):
    """A cold cache would otherwise pay a ~90MB download to return an empty list."""
    stub = model(0.5)

    assert rerank("attention", papers, top_k) == []
    assert stub.calls == []


def test_the_input_list_is_not_mutated(model):
    """Callers hold the pool afterwards — the deduplicator hands over its own list."""
    papers = [paper(paper_id="A"), paper(paper_id="B")]
    model(0.1, 0.9)

    rerank("attention", papers, top_k=2)

    assert [p.paper_id for p in papers] == ["A", "B"]
