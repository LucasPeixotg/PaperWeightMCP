"""Tests for the SELECT-only guard in services.sql_guard.

The rejection cases are the point of the module, so they carry the most weight
here: anything that gets through reaches Postgres with a language model's SQL
in it.
"""

import pytest
import sqlglot
from fastmcp.exceptions import ToolError

from services.sql_guard import ensure_read_only

ACCEPTED = [
    "SELECT id, title FROM papers",
    "SELECT id FROM papers WHERE update_date >= '2024-01-01'",
    "SELECT 1;",  # a trailing semicolon is not a second statement
    "WITH recent AS (SELECT * FROM papers WHERE update_date > '2024-01-01') "
    "SELECT id FROM recent",
    "SELECT id FROM papers UNION SELECT id FROM papers",
    "(SELECT id FROM papers)",
    "SELECT id FROM papers ORDER BY update_date DESC FETCH FIRST 5 ROWS ONLY",
    # The word DELETE inside a literal is not a DELETE — the case a keyword
    # blocklist over the raw string gets wrong.
    "SELECT id FROM papers WHERE title LIKE '%DROP TABLE%'",
]

REJECTED = [
    "DROP TABLE papers",
    "INSERT INTO papers (id) VALUES ('x')",
    "UPDATE papers SET title = 'x'",
    "DELETE FROM papers",
    "TRUNCATE TABLE papers",
    "ALTER TABLE papers ADD COLUMN x TEXT",
    "CREATE TABLE evil (id TEXT)",
    "GRANT ALL ON papers TO public",
    "REVOKE ALL ON papers FROM public",
    "SET statement_timeout = 0",
    "COPY papers TO '/tmp/leak.csv'",
    "VACUUM papers",  # sqlglot models this as a Command
    "SELECT 1; DROP TABLE papers",
    # Postgres data-modifying CTE: the top level really is a SELECT, so only a
    # tree walk catches the DELETE.
    "WITH gone AS (DELETE FROM papers RETURNING *) SELECT * FROM gone",
    "WITH added AS (INSERT INTO papers (id) VALUES ('x') RETURNING *) "
    "SELECT * FROM added",
    "SELECT id FROM papers FOR UPDATE",
    "",
]


@pytest.mark.parametrize("sql", ACCEPTED)
def test_accepts_read_only_queries(sql):
    assert ensure_read_only(sql, 10)


@pytest.mark.parametrize("sql", REJECTED)
def test_rejects_everything_else(sql):
    with pytest.raises(ToolError):
        ensure_read_only(sql, 10)


def test_rejects_unparseable_sql():
    with pytest.raises(ToolError):
        ensure_read_only("SELECT FROM WHERE ((", 10)


def test_applies_limit_when_absent():
    out = ensure_read_only("SELECT id FROM papers", 7)
    assert _limit_of(out) == "7"


def test_keeps_the_callers_own_limit():
    out = ensure_read_only("SELECT id FROM papers LIMIT 3", 100)
    assert _limit_of(out) == "3"


def test_limit_applies_to_a_union():
    out = ensure_read_only("SELECT id FROM papers UNION SELECT id FROM papers", 5)
    assert _limit_of(out) == "5"


@pytest.mark.parametrize(
    "sql",
    [
        "SELECT id FROM papers -- ; DROP TABLE papers",
        # A comment that tries to close its own /* */ block once sqlglot
        # re-emits it.
        "SELECT id FROM papers -- */ ; DROP TABLE papers",
    ],
)
def test_strips_comments(sql):
    # The statement is regenerated from the tree with comments dropped, so
    # nothing a comment contains reaches Postgres.
    out = ensure_read_only(sql, 10)
    assert "DROP" not in out.upper()
    assert len([s for s in sqlglot.parse(out, read="postgres") if s]) == 1


def _limit_of(sql: str) -> str:
    """The LIMIT value of a generated statement, as text."""
    parsed = sqlglot.parse_one(sql, read="postgres")
    return parsed.args["limit"].expression.sql()
