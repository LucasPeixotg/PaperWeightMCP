"""The shape every retriever returns.

Both retrieval arms — FAISS in `vectorstore.py` and Postgres full-text in
`lexicalstore.py` — select the same columns, so the row -> hit conversion lives
here rather than being duplicated on each side.
"""

from dataclasses import dataclass
from email.utils import parsedate_to_datetime

ARXIV_URL = "https://arxiv.org/abs/{paper_id}"


def _year(created: str | None) -> int:
    """Year from an RFC-2822 version date, e.g. 'Mon, 2 Apr 2007 19:18:42 GMT'."""
    if not created:
        return 0
    try:
        return parsedate_to_datetime(created).year
    except (TypeError, ValueError):
        return 0


@dataclass
class PaperHit:
    """A single retrieval result."""

    paper_id: str
    title: str
    year: int
    abstract: str
    url: str

    @classmethod
    def from_row(cls, row: dict) -> "PaperHit":
        """Build a hit from a row of id, title, abstract and `created`.

        `created` is the first version's date — see the note on HYDRATE_SQL in
        `vectorstore.py` for why that is asked for instead of update_date.
        """
        paper_id = row["id"]
        return cls(
            paper_id=paper_id,
            title=row["title"],
            year=_year(row["created"]),
            abstract=row["abstract"],
            url=ARXIV_URL.format(paper_id=paper_id),
        )
