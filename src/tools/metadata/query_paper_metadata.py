def query_paper_metadata(sql: str, limit: int = 10) -> str:
    """
    Executes precise structured filtering on paper metadata by running a
    read-only SQL query directly against the `papers` table.

    Use this tool instead of semantic vector search when answering queries
    that require exact numerical, temporal, or categorical constraints—such
    as filtering by date ranges, specific categories, or specific author
    names/fields present in the schema below.

    Only read-only statements are permitted. The query must be a single
    SELECT statement (CTEs beginning with WITH ... SELECT are allowed).
    Statements containing INSERT, UPDATE, DELETE, DROP, ALTER, CREATE,
    TRUNCATE, GRANT, REVOKE, or multiple semicolon-separated statements
    are rejected before execution.

    Schema:
        CREATE TABLE IF NOT EXISTS papers (
            id              TEXT PRIMARY KEY,      -- arXiv id, e.g. "0704.0001"
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
            authors_parsed  JSONB,                 -- list of [last, first, suffix]
            search_vec      TSVECTOR               -- full-text index payload; never SELECT it
        );

    Args:
        sql: A single read-only SQL SELECT statement targeting the `papers`
             table (e.g. "SELECT id, title FROM papers WHERE categories
             LIKE '%cs.AI%' AND update_date >= '2024-01-01'").
        limit: Maximum number of matching metadata records to return
               (default is 10). If the SQL query does not already include
               its own LIMIT clause, this value is applied.

    Returns:
        JSON string containing structured metadata records matching the
        query (e.g. id, title, authors, categories, update_date, and any
        other selected columns).
    """
    return 'cannot return right now'