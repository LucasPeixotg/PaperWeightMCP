from dataclasses import dataclass


@dataclass
class PaperData:
    paper_id: str
    doi: str
    title: str
    year: int
    abstract: str
    url: str
    license: str
