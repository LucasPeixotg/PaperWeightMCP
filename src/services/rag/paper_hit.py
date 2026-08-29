from dataclasses import dataclass


@dataclass
class PaperHit:
    """A single retrieval result."""

    paper_id: str
    title: str
    year: int
    abstract: str
    url: str