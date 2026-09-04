"""Add the full-text search index over paper titles and abstracts.

The lexical arm of the hybrid retriever (`services/rag/lexicalstore.py`) ranks
with `ts_rank_cd` over `papers.search_vec`, a stored generated column. Storing
the tsvector rather than recomputing it per candidate is what makes the ranking
affordable: tokenizing a row costs ~58 microseconds, reading a stored tsvector
~1, and a broad query can match tens of thousands of rows.

A fresh load gets the column for free — `loader.CREATE_PAPERS_SQL` declares it,
so COPY computes it inline. This module exists for a database that was loaded
before the column did, where adding it means an `ALTER TABLE` that rewrites the
heap. Progress is reported per step because that rewrite runs for many minutes.

State is read from the system catalog rather than a marker table: unlike the
loader's `loader_state`, there is nothing here a catalog query cannot answer,
and a catalog cannot disagree with the schema it describes.
"""

import shutil

from config import ROOT_DIR, settings

COLUMN_NAME = "search_vec"
INDEX_NAME = "papers_search_vec_idx"

# The two-argument to_tsvector is IMMUTABLE, which a generated column requires;
# the one-argument form reads default_text_search_config and is only STABLE.
# Weighting title above abstract lets ts_rank_cd favour a term in the title:
# with the default weights that is 1.0 against 0.4.
GENERATED_EXPR = """GENERATED ALWAYS AS (
        setweight(to_tsvector('{config}', coalesce(title, '')),    'A') ||
        setweight(to_tsvector('{config}', coalesce(abstract, '')), 'B')
    ) STORED"""


def column_ddl(config: str | None = None) -> str:
    """The column definition, shared by CREATE TABLE and ALTER TABLE.

    `loader.CREATE_PAPERS_SQL` declares it inline so a fresh load computes the
    tsvector during COPY; this module's ALTER adds it to a table loaded before
    the column existed. Both have to produce byte-identical tsvectors, so the
    definition is written once.
    """
    config = settings.fts_config if config is None else config
    return f"{COLUMN_NAME} tsvector {GENERATED_EXPR.format(config=config)}"


CREATE_INDEX_SQL = f"CREATE INDEX {INDEX_NAME} ON papers USING gin ({COLUMN_NAME})"

HAS_COLUMN_SQL = """
SELECT EXISTS (
    SELECT 1 FROM information_schema.columns
    WHERE table_schema = 'public' AND table_name = 'papers' AND column_name = %s
)
"""

# indisvalid is the point of asking pg_index rather than pg_indexes: an index
# left behind by an interrupted build still has a name and a definition, but
# the planner will not use it. Treat it as absent so it gets rebuilt.
INDEX_STATE_SQL = """
SELECT i.indisvalid
FROM pg_class c
JOIN pg_namespace n ON n.oid = c.relnamespace
JOIN pg_index i ON i.indexrelid = c.oid
WHERE n.nspname = 'public' AND c.relname = %s
"""

# The ALTER rewrites the table, so the old and new heaps coexist at peak. Two
# and a half times the current heap covers both plus the tsvector column and
# the GIN index that follows.
REWRITE_HEADROOM = 2.5


def has_column(conn) -> bool:
    """True if `papers.search_vec` exists."""
    return conn.execute(HAS_COLUMN_SQL, (COLUMN_NAME,)).fetchone()[0]


def index_state(conn) -> str:
    """One of 'missing', 'invalid' or 'ready'."""
    row = conn.execute(INDEX_STATE_SQL, (INDEX_NAME,)).fetchone()
    if row is None:
        return "missing"
    return "ready" if row[0] else "invalid"


def is_built(conn) -> bool:
    """True if nothing is left to do — column present and index usable."""
    return has_column(conn) and index_state(conn) == "ready"


def preflight(conn) -> None:
    """Fail before a multi-minute rewrite if the disk cannot hold the result."""
    heap_bytes = conn.execute("SELECT pg_relation_size('papers')").fetchone()[0]
    needed_gb = heap_bytes * REWRITE_HEADROOM / 1e9

    # Only meaningful when Postgres stores its data on this machine, which is
    # what docker-compose.yml sets up (a bind mount under ./data). A managed
    # database has its own disk and the local free space says nothing about it.
    if settings.database_url or settings.postgres_host not in ("localhost", "127.0.0.1", "::1"):
        print(f"  note: the rewrite needs roughly {needed_gb:.0f} GB on the database host")
        return

    free_gb = shutil.disk_usage(ROOT_DIR).free / 1e9
    if free_gb < needed_gb:
        raise SystemExit(
            f"only {free_gb:.1f} GB free at {ROOT_DIR}, and adding {COLUMN_NAME} "
            f"rewrites the papers heap — that needs about {needed_gb:.0f} GB at its "
            "peak. Free some space and re-run."
        )


def build(conn, progress) -> None:
    """Add the column and its GIN index, resuming from whatever already exists.

    `conn` must be in autocommit: each statement then stands on its own, so a
    completed ALTER is not thrown away by a later index failure — the same
    reasoning that puts the loader's index builds after its COPY commits — and
    VACUUM cannot run inside a transaction block at all.
    """
    with progress.phase("full-text index"):
        # 64 MB is the server default and would make the GIN build crawl; the
        # rewrite must not be killed by a statement timeout inherited from
        # anywhere. Session-scoped, so both statements below see them.
        conn.execute("SET statement_timeout = 0")
        conn.execute("SET maintenance_work_mem = '1GB'")

        if has_column(conn):
            print(f"  {COLUMN_NAME} column already present")
        else:
            with progress.step(f"add {COLUMN_NAME} column (rewrites the table)"):
                conn.execute(f"ALTER TABLE papers ADD COLUMN {column_ddl()}")

        state = index_state(conn)
        if state == "invalid":
            with progress.step(f"drop invalid {INDEX_NAME}"):
                conn.execute(f"DROP INDEX {INDEX_NAME}")
            state = "missing"

        if state == "ready":
            print(f"  {INDEX_NAME} already built")
        else:
            with progress.step(f"build {INDEX_NAME}"):
                conn.execute(CREATE_INDEX_SQL)

        # ANALYZE because without statistics on the new column the planner has
        # no idea how selective a tsquery is. VACUUM because the rewrite left
        # every page's visibility bit clear, so until it runs the first read of
        # each page dirties it again — measured as several thousand extra block
        # writes on every ranking query.
        with progress.step("VACUUM ANALYZE papers"):
            conn.execute("VACUUM (ANALYZE) papers")
