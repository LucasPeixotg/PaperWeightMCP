"""Full-text retrieval against the live `papers` table.

These tests need a loaded database with the search_vec column, so they skip
rather than fail on a checkout where `python src/loader.py --only-fts` has not
been run yet.
"""

import psycopg
import pytest

import fts
from config import settings
from services.db import read_only_connection
from services.rag.lexicalstore import SEARCH_SQL, LexicalStore

# The embed job pages through `papers` in id order, so every arXiv id above this
# one is in Postgres but not in FAISS. Read off embed_state at the time these
# tests were written; the point it makes holds for any id the vector index has
# not reached yet.
UNEMBEDDED_ID_FLOOR = "1408.1509"


def _fts_ready() -> bool:
    try:
        with psycopg.connect(settings.dsn, connect_timeout=settings.db_connect_timeout) as conn:
            return fts.is_built(conn)
    except psycopg.Error:
        return False


pytestmark = pytest.mark.skipif(
    not _fts_ready(),
    reason="needs a loaded database with search_vec (python src/loader.py --only-fts)",
)


@pytest.fixture(scope="module")
def store() -> LexicalStore:
    return LexicalStore()


@pytest.fixture(scope="module")
def sample_paper() -> dict:
    """A real row whose title carries a distinctive term, for an exact lookup."""
    with read_only_connection() as conn:
        return conn.execute(
            """
            SELECT id, title FROM papers
            WHERE title ILIKE %s AND id > %s
            ORDER BY id LIMIT 1
            """,
            ("%renormalization%", UNEMBEDDED_ID_FLOOR),
        ).fetchone()


def test_finds_a_paper_by_its_own_title(store, sample_paper):
    hits = store.search(sample_paper["title"], 10)

    assert sample_paper["id"] in [h.paper_id for h in hits]


def test_reaches_papers_faiss_has_not_embedded(store, sample_paper):
    """The coverage win: lexical search sees rows the vector index does not."""
    assert sample_paper["id"] > UNEMBEDDED_ID_FLOOR

    hits = store.search(sample_paper["title"], 10)

    assert hits, "a title copied verbatim out of the table must match something"


def test_hits_are_fully_populated(store):
    hits = store.search("quantum entanglement entropy", 5)

    assert hits
    for hit in hits:
        assert hit.paper_id and hit.title and hit.abstract
        assert hit.url == f"https://arxiv.org/abs/{hit.paper_id}"
        assert hit.year > 1980


def test_stopwords_only_returns_nothing(store):
    """websearch_to_tsquery reduces this to an empty query; it must not error."""
    assert store.search("the of and", 5) == []


@pytest.mark.parametrize("query", ["", "   "])
def test_blank_query_short_circuits(store, query):
    assert store.search(query, 5) == []


def test_non_positive_k_returns_nothing(store):
    assert store.search("neural network", 0) == []


def test_quoted_phrase_is_honoured(store):
    """websearch_to_tsquery, not plainto_: quoting must mean adjacency."""
    hits = store.search('"neural network"', 5)

    assert all("neural" in (h.title + h.abstract).lower() for h in hits)


def test_query_uses_the_gin_index(store):
    """Guards against an edit that stops the planner matching search_vec."""
    with read_only_connection() as conn:
        plan = "\n".join(
            row["QUERY PLAN"]
            for row in conn.execute(
                "EXPLAIN " + SEARCH_SQL, (settings.fts_config, "contrastive learning", 50)
            ).fetchall()
        )

    assert fts.INDEX_NAME in plan, plan
    assert "Seq Scan on papers" not in plan, plan
