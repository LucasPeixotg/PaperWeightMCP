"""Postgres connection helpers.

The one-time bulk loader and the metadata tools talk to the same database, so
connection handling lives here — one place that knows the DSN, the timeouts and
the read-only policy.
"""

from contextlib import contextmanager

import psycopg
from psycopg.rows import dict_row

from config import settings


@contextmanager
def connection(*, autocommit: bool = False):
    """A read-write connection, used by the loader.

    Commits on clean exit, rolls back if the block raises.
    """
    with psycopg.connect(
        settings.dsn,
        connect_timeout=settings.db_connect_timeout,
        autocommit=autocommit,
    ) as conn:
        yield conn


@contextmanager
def read_only_connection():
    """A read-only connection for `query_paper_metadata`.

    `conn.read_only` is enforced by the server, so it backs up the SELECT-only
    string validation the tool does: a query that slips past the parser still
    cannot write. Rows come back as dicts, which is the shape the tool needs to
    serialize.
    """
    with psycopg.connect(
        settings.dsn,
        connect_timeout=settings.db_connect_timeout,
        row_factory=dict_row,
    ) as conn:
        conn.read_only = True  # must be set before the first transaction opens
        conn.execute(f"SET statement_timeout = {settings.db_statement_timeout_ms}")
        yield conn
