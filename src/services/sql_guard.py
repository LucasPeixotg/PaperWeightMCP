"""SELECT-only validation for caller-supplied SQL.

`query_paper_metadata` hands Postgres a statement written by a language model,
so the statement has to be proven read-only before it runs. That proof is two
layers deep:

1. this module, which parses the SQL and inspects the syntax tree, and
2. `read_only_connection()` in `services.db`, which sets `conn.read_only` so a
   statement that somehow slips past layer 1 still cannot write.

Layer 1 parses rather than pattern-matches. A keyword blocklist run over the
raw string cannot tell a real `DELETE` from the word appearing inside a title
literal, and it misses the interesting case entirely: Postgres allows
data-modifying CTEs, so `WITH x AS (DELETE FROM papers RETURNING *) SELECT * FROM x`
is a statement whose top level really is a SELECT. Only walking the tree finds
the DELETE hiding inside it.
"""

from __future__ import annotations

import sqlglot
from fastmcp.exceptions import ToolError
from sqlglot import exp

DIALECT = "postgres"

# Only these may appear at the top level. Note that `WITH ... SELECT` parses as
# a Select carrying a `with` arg, not as a top-level exp.With, so it is already
# covered here.
ALLOWED_ROOTS = (exp.Select, exp.SetOperation, exp.Subquery)

# Anything in the tree matching one of these rejects the whole statement.
#
# exp.DML and exp.DDL are base classes and cover Insert, Update, Delete, Merge,
# Create and Copy between them. The rest are statement types sqlglot models
# outside that hierarchy — they have to be named individually, which is why this
# list is derived from the classes sqlglot actually defines rather than from the
# keyword list in the tool's docstring. exp.Command is the parser's fallback for
# syntax it does not model (VACUUM, CALL, ...), so it is hostile by default.
#
# Deliberately absent: exp.Fetch, which is the legitimate `FETCH FIRST n ROWS
# ONLY` clause of a SELECT, not a statement.
FORBIDDEN = (
    exp.DML,
    exp.DDL,
    exp.Command,
    exp.Drop,
    exp.Alter,
    exp.TruncateTable,
    exp.Grant,
    exp.Revoke,
    exp.Set,
    exp.SetItem,
    exp.Use,
    exp.Transaction,
    exp.Commit,
    exp.Rollback,
    exp.Analyze,
    exp.Attach,
    exp.Detach,
    exp.Refresh,
    exp.Export,
    exp.Describe,
    exp.Show,
    exp.Cache,
    exp.Uncache,
    exp.Pragma,
    exp.Kill,
    exp.Comment,
    exp.Execute,
    exp.Declare,
    exp.Lock,  # SELECT ... FOR UPDATE takes write locks
)


def ensure_read_only(sql: str, limit: int) -> str:
    """Validate `sql` as a single read-only SELECT and return it ready to run.

    Applies `limit` as a LIMIT clause unless the statement already carries one.
    The string returned is regenerated from the parsed tree rather than passed
    through: whatever the parser normalized away — comments, odd quoting — never
    reaches Postgres.

    Raises ToolError, whose message goes back to the calling model, so every
    rejection says enough for it to correct the query and retry.
    """
    try:
        statements = sqlglot.parse(sql, read=DIALECT)
    except sqlglot.errors.ParseError as exc:
        raise ToolError(f"could not parse SQL: {exc}") from exc

    # parse() yields None for empty segments, so a trailing semicolon is fine.
    statements = [s for s in statements if s is not None]

    if not statements:
        raise ToolError("no SQL statement provided")
    if len(statements) > 1:
        raise ToolError(
            f"expected a single statement, got {len(statements)}; "
            "semicolon-separated statements are not permitted"
        )

    statement = statements[0]

    if not isinstance(statement, ALLOWED_ROOTS):
        raise ToolError(
            f"only SELECT statements are permitted, got {_describe(statement)}"
        )

    for node in statement.walk():
        if isinstance(node, FORBIDDEN):
            raise ToolError(
                f"only read-only statements are permitted, but the query "
                f"contains {_describe(node)}"
            )

    if statement.args.get("limit") is None:
        statement = statement.limit(limit)

    # comments=False drops them rather than re-emitting them as /* ... */.
    # sqlglot does escape a `*/` inside a comment so it cannot close the block
    # early, but a validated statement has no use for the caller's comments and
    # not carrying them across is one less thing resting on that escaping.
    return statement.sql(dialect=DIALECT, comments=False)


def _describe(node: exp.Expression) -> str:
    """Name a rejected node the way the caller wrote it, e.g. 'DELETE'."""
    return type(node).__name__.upper()
