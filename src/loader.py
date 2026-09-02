"""One-time bulk load of the arXiv metadata snapshot into Postgres.

Run this once, by hand, after the database is up:

    python src/loader.py

It creates the schema, streams `data/arxiv-metadata-oai-snapshot.json` into the
`papers` table with COPY, builds the secondary indexes and records a marker row
in `loader_state`. Every later run sees that marker and exits without touching
anything, so it is safe to re-run.

The snapshot is ~5.5 GB / ~3.1M records, so the load takes minutes and writes
several GB of WAL. It is deliberately a setup step rather than something
`server.py` does at startup.
"""

import argparse
import json
import sys
import time
from collections import Counter
from datetime import date
from pathlib import Path

from tqdm import tqdm

from config import masked_dsn, settings
from progress import Progress

# Column order for the COPY. The tuple built in `iter_rows` must match this
# exactly — keep the two adjacent when editing.
COLUMNS = (
    "id",
    "submitter",
    "authors",
    "title",
    "comments",
    "journal_ref",
    "doi",
    "report_no",
    "categories",
    "license",
    "abstract",
    "update_date",
    "versions",
    "authors_parsed",
)

# The primary key is added after the COPY, not declared here: maintaining a
# btree across 3.1M inserts is the classic bulk-load tax. Adding it afterwards
# is one scan plus a sort, and it still rejects duplicate ids.
CREATE_PAPERS_SQL = """
CREATE TABLE papers (
    id              TEXT,                  -- arXiv id, e.g. "0704.0001"
    submitter       TEXT,
    authors         TEXT,                  -- raw author string
    title           TEXT,
    comments        TEXT,
    journal_ref     TEXT,
    doi             TEXT,
    report_no       TEXT,
    categories      TEXT,                  -- space-separated category codes
    license         TEXT,
    abstract        TEXT,
    update_date     DATE,
    versions        JSONB,                 -- list of {version, created}
    authors_parsed  JSONB                  -- list of [last, first, suffix, ...]
)
"""

CREATE_STATE_SQL = """
CREATE TABLE IF NOT EXISTS loader_state (
    name         TEXT PRIMARY KEY,
    status       TEXT NOT NULL,            -- 'completed' | 'partial'
    rows_loaded  BIGINT NOT NULL,
    rows_skipped BIGINT NOT NULL,
    source_file  TEXT,
    source_bytes BIGINT,                   -- lets a newer snapshot be spotted
    finished_at  TIMESTAMPTZ NOT NULL DEFAULT now()
)
"""

UPSERT_STATE_SQL = """
INSERT INTO loader_state
    (name, status, rows_loaded, rows_skipped, source_file, source_bytes, finished_at)
VALUES (%s, %s, %s, %s, %s, %s, now())
ON CONFLICT (name) DO UPDATE SET
    status       = EXCLUDED.status,
    rows_loaded  = EXCLUDED.rows_loaded,
    rows_skipped = EXCLUDED.rows_skipped,
    source_file  = EXCLUDED.source_file,
    source_bytes = EXCLUDED.source_bytes,
    finished_at  = EXCLUDED.finished_at
"""

# The snapshot ships the same paper more than once (90 ids at the time of
# writing), differing only in update_date — so a plain ADD PRIMARY KEY would
# abort the whole load. Keep the freshest row per id. ctid is stable here
# because nothing else touches the table inside this transaction.
DEDUPE_SQL = """
DELETE FROM papers p
USING (
    SELECT ctid,
           row_number() OVER (
               PARTITION BY id
               ORDER BY update_date DESC NULLS LAST, ctid DESC
           ) AS rn
    FROM papers
) dupes
WHERE p.ctid = dupes.ctid AND dupes.rn > 1
"""

# Targets the filters `query_paper_metadata` advertises: update_date ranges and
# substring matches on categories/title. A leading-wildcard LIKE cannot use a
# btree, hence the trigram GIN indexes.
INDEX_SQL = (
    "CREATE EXTENSION IF NOT EXISTS pg_trgm",
    "CREATE INDEX IF NOT EXISTS papers_update_date_idx ON papers (update_date)",
    "CREATE INDEX IF NOT EXISTS papers_categories_trgm_idx "
    "ON papers USING gin (categories gin_trgm_ops)",
    "CREATE INDEX IF NOT EXISTS papers_title_trgm_idx "
    "ON papers USING gin (title gin_trgm_ops)",
)

# Arbitrary but stable key for the advisory lock that serializes loader runs
LOCK_KEY = 0x9A7E_9E16


def _jsonb(value) -> str | None:
    """Serialize to a JSON text literal for a jsonb column.

    Returns None (SQL NULL) rather than the string "null", so a missing field is
    distinguishable from a stored JSON null.
    """
    if value is None:
        return None
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def _parse_date(value) -> date | None:
    """Parse an ISO date, degrading to NULL instead of aborting the COPY."""
    if not value:  # covers None and ""
        return None
    try:
        return date.fromisoformat(value)
    except (ValueError, TypeError):
        return None


def iter_rows(path: Path, limit: int | None, stats: Counter):
    """Stream the JSONL snapshot as COPY-ready tuples.

    COPY is all-or-nothing — one bad row aborts the whole stream after gigabytes
    have already crossed the socket — so every line is validated here and bad
    ones are counted and skipped rather than raised.
    """
    bad_lines: list[str] = []

    # Binary mode: a bad byte sequence becomes a skipped line rather than a dead
    # run, and summing len(raw) gives byte progress for free. (f.tell() cannot be
    # used while iterating — the read-ahead buffer makes it wrong or raising.)
    with open(path, "rb") as fh:
        for lineno, raw in enumerate(fh, start=1):
            stats["bytes"] += len(raw)

            if not raw.strip():
                continue

            # Postgres text columns cannot hold \x00, in either escaped or raw
            # form. These checks are fast C substring scans and almost never hit.
            if b"\\u0000" in raw:
                raw = raw.replace(b"\\u0000", b"")
            if b"\x00" in raw:
                raw = raw.replace(b"\x00", b"")

            try:
                record = json.loads(raw.decode("utf-8"))
            except UnicodeDecodeError:
                stats["bad_unicode"] += 1
                _note_bad(bad_lines, lineno, raw)
                continue
            except json.JSONDecodeError:
                stats["bad_json"] += 1
                _note_bad(bad_lines, lineno, raw)
                continue

            paper_id = record.get("id")
            if not paper_id:
                stats["missing_id"] += 1
                _note_bad(bad_lines, lineno, raw)
                continue

            # .get() throughout: a KeyError at row 2.5M would throw away the run.
            # Note journal-ref / report-no are hyphenated in the JSON.
            yield (
                paper_id,
                record.get("submitter"),
                record.get("authors"),
                record.get("title"),
                record.get("comments"),
                record.get("journal-ref"),
                record.get("doi"),
                record.get("report-no"),
                record.get("categories"),
                record.get("license"),
                record.get("abstract"),
                _parse_date(record.get("update_date")),
                _jsonb(record.get("versions")),
                _jsonb(record.get("authors_parsed")),
            )

            stats["ok"] += 1
            if limit is not None and stats["ok"] >= limit:
                break

    for message in bad_lines:
        print(message, file=sys.stderr)


def _note_bad(bad_lines: list[str], lineno: int, raw: bytes) -> None:
    """Record the first few malformed lines; a broken file must not spam stderr."""
    if len(bad_lines) < settings.loader_log_bad_lines:
        bad_lines.append(f"  skipped line {lineno}: {raw[:120]!r}")


def _consume(rows, stats: Counter, total_bytes: int, write) -> None:
    """Drive the row generator, updating a byte-based progress bar as it goes."""
    bar = tqdm(
        total=total_bytes,
        unit="B",
        unit_scale=True,
        unit_divisor=1024,
        desc="loading papers",
        smoothing=0.05,
    )
    seen_bytes = 0
    try:
        for n, row in enumerate(rows, start=1):
            write(row)
            # Updating the bar per row costs real time at 3.1M rows
            if n % settings.loader_progress_every == 0:
                bar.update(stats["bytes"] - seen_bytes)
                seen_bytes = stats["bytes"]
        bar.update(stats["bytes"] - seen_bytes)
    finally:
        bar.close()


def is_loaded(conn) -> bool:
    """True only if a *completed* load is recorded.

    The marker lives in the database rather than in a file on disk because the
    fact it asserts is about this database: dropping the Docker volume must
    invalidate it. A `--limit` run records 'partial', which does not count.
    """
    row = conn.execute(
        "SELECT status FROM loader_state WHERE name = %s", (settings.loader_name,)
    ).fetchone()
    return row is not None and row[0] == "completed"


def has_papers(conn) -> bool:
    """True if a usable `papers` table exists, however it was loaded.

    Distinct from `is_loaded`, which demands a *completed* full load: a
    deliberate `--limit` load is recorded as 'partial' but is still perfectly
    valid to embed against.
    """
    if conn.execute("SELECT to_regclass('papers')").fetchone()[0] is None:
        return False
    return conn.execute("SELECT EXISTS(SELECT 1 FROM papers)").fetchone()[0]


def load(conn, path: Path, limit: int | None, build_indexes: bool, progress: Progress) -> Counter:
    """Create the schema and bulk-load the snapshot, atomically."""
    stats: Counter = Counter()
    total_bytes = path.stat().st_size
    copy_sql = f"COPY papers ({', '.join(COLUMNS)}) FROM STDIN"

    started = time.monotonic()

    # One transaction for schema + data + marker. Postgres has transactional
    # DDL, so a crash or Ctrl-C rolls the whole thing back: no half-populated
    # table to reason about, and no marker, so the next run just starts over.
    with conn.transaction():
        with conn.cursor() as cur:
            cur.execute("SET LOCAL statement_timeout = 0")  # a long COPY must not be killed
            cur.execute("SET LOCAL synchronous_commit = off")  # one commit at the end
            cur.execute("SET LOCAL maintenance_work_mem = '1GB'")  # for the PK build

            # The phase boundaries below are print boundaries, not commit
            # boundaries: schema, data and marker all stay inside this one
            # transaction.
            with progress.phase("copy into postgres"):
                # Safe to drop unconditionally: the guard already established that
                # no completed load exists, so anything here is untrusted leftovers.
                cur.execute("DROP TABLE IF EXISTS papers")
                cur.execute(CREATE_PAPERS_SQL)

                with cur.copy(copy_sql) as copy:
                    _consume(iter_rows(path, limit, stats), stats, total_bytes, copy.write_row)

            with progress.phase("dedupe + primary key"):
                with progress.step("dedupe duplicate ids"):
                    cur.execute(DEDUPE_SQL)
                    stats["deduped"] = cur.rowcount

                # Now guaranteed to succeed; it is still the backstop that proves
                # the dedupe was complete.
                with progress.step("add primary key"):
                    cur.execute("ALTER TABLE papers ADD PRIMARY KEY (id)")

            cur.execute(
                UPSERT_STATE_SQL,
                (
                    settings.loader_name,
                    "partial" if limit is not None else "completed",
                    stats["ok"] - stats["deduped"],  # rows actually in the table
                    _skipped(stats),
                    str(path),
                    total_bytes,
                ),
            )

    # Indexes go after the commit, so a failed index build cannot discard a
    # ten-minute COPY. They are built after the data for the same reason as the
    # primary key.
    failures: list[str] = []
    with progress.phase("secondary indexes + analyze" if build_indexes else "analyze"):
        if build_indexes:
            for statement in INDEX_SQL:
                with progress.step(statement.split(" ON ")[0]):
                    try:
                        conn.execute(statement)
                    except Exception as exc:  # pg_trgm needs superuser on some setups
                        # Collected rather than printed here: the step is mid-line
                        # until its duration lands.
                        failures.append(f"  index step failed: {exc}")

        # A fresh table has no statistics, so without this the planner
        # sequential-scans everything on the very first query.
        with progress.step("ANALYZE papers"):
            conn.execute("ANALYZE papers")

    for message in failures:
        print(message, file=sys.stderr)

    # Measured across the whole Postgres phase, indexes included, so the summary
    # rate is what a future run can actually be projected from.
    stats["seconds"] = time.monotonic() - started

    return stats


def _skipped(stats: Counter) -> int:
    return stats["bad_json"] + stats["bad_unicode"] + stats["missing_id"]


def _report(stats: Counter) -> None:
    seconds = stats.get("seconds", 0.0)
    rate = stats["ok"] / seconds if seconds else 0.0
    print(f"loaded {stats['ok']:,} rows in {seconds:.1f}s ({rate:,.0f} rows/s)")
    if stats["deduped"]:
        print(
            f"removed {stats['deduped']:,} duplicate ids "
            f"(kept the latest update_date) -> {stats['ok'] - stats['deduped']:,} in papers"
        )
    print(
        f"skipped {_skipped(stats):,}  "
        f"(bad_json={stats['bad_json']:,}  "
        f"bad_unicode={stats['bad_unicode']:,}  "
        f"missing_id={stats['missing_id']:,})"
    )


def dry_run(path: Path, limit: int | None) -> Counter:
    """Exercise the full parse/transform pipeline without touching Postgres."""
    stats: Counter = Counter()
    started = time.monotonic()
    _consume(iter_rows(path, limit, stats), stats, path.stat().st_size, lambda row: None)
    stats["seconds"] = time.monotonic() - started
    return stats


def main() -> int:
    parser = argparse.ArgumentParser(
        description="One-time bulk load of the arXiv snapshot into Postgres."
    )
    parser.add_argument("--data-path", type=Path, default=settings.data_path)
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Load only the first N rows (smoke test). Records the load as 'partial'.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Reload even if a completed load is already recorded.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Parse and transform every row without connecting to Postgres.",
    )
    parser.add_argument(
        "--skip-embeddings",
        action="store_true",
        help="Load Postgres only; do not build the FAISS abstract index.",
    )
    parser.add_argument(
        "--only-embeddings",
        action="store_true",
        help="Build the FAISS index only; assume Postgres is already loaded.",
    )
    parser.add_argument(
        "--no-indexes",
        action="store_true",
        help="Skip the secondary indexes built after the load.",
    )
    parser.add_argument(
        "--force-embeddings",
        action="store_true",
        help="Rebuild the FAISS index even if one is already recorded as complete.",
    )
    parser.add_argument(
        "--nlist",
        type=int,
        default=None,
        help="FAISS IVF list count. Lower it for small test builds.",
    )
    args = parser.parse_args()

    path: Path = args.data_path
    if not path.exists():
        print(f"snapshot not found: {path}", file=sys.stderr)
        return 1

    if args.dry_run:
        print(f"dry run over {path} (no database connection)")
        progress = Progress(["parse snapshot"])
        with progress.phase("parse snapshot"):
            stats = dry_run(path, args.limit)
        _report(stats)
        return 0

    # Imported here, not at module scope, so --dry-run works before psycopg
    # is installed and before any database exists.
    import psycopg

    from services.db import connection

    # Pre-flight, so a database that simply isn't up yet reports that plainly
    # instead of a traceback.
    try:
        with connection() as probe:
            probe.execute("SELECT 1")
    except psycopg.OperationalError as exc:
        print(f"could not connect to Postgres: {exc}".rstrip(), file=sys.stderr)
        print(
            "Is the database running? Connection settings come from .env "
            f"(currently {masked_dsn(settings.dsn)}).",
            file=sys.stderr,
        )
        return 1

    with connection(autocommit=True) as conn:
        # Session-level advisory lock, released automatically on disconnect, so
        # two concurrent runs cannot both drop and recreate the table.
        locked = conn.execute("SELECT pg_try_advisory_lock(%s)", (LOCK_KEY,)).fetchone()[0]
        if not locked:
            print("another loader is already running", file=sys.stderr)
            return 1

        conn.execute(CREATE_STATE_SQL)

        # Work out every phase this run will execute before announcing the first
        # one: the "[2/4]" denominator has to account for phases that a flag or
        # an already-finished load will skip.
        run_postgres = not args.only_embeddings and (args.force or not is_loaded(conn))

        # Imported here, not at module scope, so --dry-run and --skip-embeddings
        # work without numpy and faiss installed.
        embed_built = False
        if not args.skip_embeddings:
            import embeddings

            conn.execute(embeddings.CREATE_EMBED_STATE_SQL)
            embed_built = embeddings.is_built(conn) and not args.force_embeddings

        labels = []
        if run_postgres:
            labels += ["copy into postgres", "dedupe + primary key"]
            labels.append("secondary indexes + analyze" if not args.no_indexes else "analyze")
        if not args.skip_embeddings and not embed_built:
            # Two phases, not one: the quantizer trains on a sample orders of
            # magnitude smaller than the corpus, and folding them under a single
            # header made that sample look like the entire index.
            if embeddings.needs_training():
                labels.append("train IVF-PQ quantizer")
            labels.append("embed abstracts")
        progress = Progress(labels)

        # --- Postgres ---
        stats = None
        if args.only_embeddings:
            if not has_papers(conn):
                print("papers table is empty; run without --only-embeddings first",
                      file=sys.stderr)
                return 1
        elif not run_postgres:
            print("papers already loaded — nothing to do (use --force to reload)")
        else:
            print(f"loading {path} into {masked_dsn(settings.dsn)}")
            stats = load(conn, path, args.limit, not args.no_indexes, progress)

        if stats is not None:
            _report(stats)

        # --- FAISS abstract index ---
        if args.skip_embeddings:
            return 0
        if embed_built:
            print("abstract index already built — nothing to do "
                  "(use --force-embeddings to rebuild)")
            return 0

        embeddings.preflight()
        print(f"building abstract index in {settings.faiss_dir}")
        # build() opens its own phases — it is the only caller that knows the
        # sample size and the corpus size the labels have to carry.
        result = embeddings.build(
            conn, limit=args.limit, nlist=args.nlist, progress=progress
        )
        print(f"abstract index holds {result['total']:,} vectors")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
