"""Postgres full-text search over paper titles and abstracts.

The dense arm in `vectorstore.py` smears literal tokens — author names, model
names, dataset names, acronyms — into a 768-dim vector, and HyDE compounds it by
embedding a *generated* abstract rather than the user's own words. This arm
matches the query verbatim instead, ranked with `ts_rank_cd` over the stored
`papers.search_vec` column.

It also sees the whole table, whereas FAISS only covers the ids embedded so far,
so it reaches papers the vector index does not know about yet.
"""

from __future__ import annotations

import psycopg

from config import settings
from services.db import read_only_connection

from .logger import get_logger
from .paper_hit import PaperHit

# `search_vec` is a stored generated column (see src/fts.py), so ts_rank_cd reads
# a precomputed tsvector instead of tokenizing each candidate row.
#
# The tsquery sits in the FROM clause so it is parsed once and still drives a
# bitmap index scan on papers_search_vec_idx. Normalization flag 1 divides the
# rank by 1 + log(document length), so a long abstract cannot outrank a short one
# on term count alone.
SEARCH_SQL = """
    SELECT p.id, p.title, p.abstract, p.versions->0->>'created' AS created
    FROM papers p, websearch_to_tsquery(%s::regconfig, %s) AS q
    WHERE p.search_vec @@ q
    ORDER BY ts_rank_cd(p.search_vec, q, 1) DESC
    LIMIT %s
"""

logger = get_logger(__name__)


class LexicalStore:
    """Serves keyword queries against the full-text index on `papers`."""

    def search(self, query: str, k: int) -> list[PaperHit]:
        """Return the k papers whose title/abstract best match `query` lexically.

        `query` is the raw user text, not the HyDE document: the point of this
        arm is the terms the user actually typed.
        """
        if k <= 0 or not query.strip():
            return []

        logger.info(f"Full-text searching for the top {k} matches")

        try:
            with read_only_connection() as conn, conn.transaction():
                # Tighter than the session-wide ceiling read_only_connection
                # sets: this is one arm of a search a user is waiting on, and
                # SET LOCAL keeps the override inside this transaction.
                # Interpolated, not bound: SET takes no parameters. The value
                # is an int from Settings, exactly as db.py does it.
                conn.execute(
                    f"SET LOCAL statement_timeout = {settings.lexical_timeout_ms}"
                )
                rows = conn.execute(
                    SEARCH_SQL, (settings.fts_config, query, k)
                ).fetchall()
        except psycopg.errors.QueryCanceled:
            # Expected for a query broad enough to match a large share of the
            # corpus, not a fault. Fusion carries on with the dense arm alone.
            logger.warning(
                f"full-text search exceeded {settings.lexical_timeout_ms}ms — "
                "too many matches to rank; skipping the lexical arm"
            )
            return []

        # websearch_to_tsquery returns an empty query for stopword-only input,
        # which matches nothing. That is correct — fusion then falls back to the
        # dense arm on its own.
        return [PaperHit.from_row(row) for row in rows]
