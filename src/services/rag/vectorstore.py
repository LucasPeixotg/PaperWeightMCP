"""FAISS-backed nearest-neighbour search over paper abstracts.

The index stores only vectors; every hit's metadata is read back from Postgres,
so abstracts are not duplicated on disk. Ordinals returned by FAISS are
positions in the id map written alongside the index by `src/embeddings.py`.
"""

from __future__ import annotations

from email.utils import parsedate_to_datetime

import faiss
import numpy as np

from config import settings
from services.db import read_only_connection

from .logger import get_logger
from .paper_hit import PaperHit

ARXIV_URL = "https://arxiv.org/abs/{paper_id}"
IVFPQ_FILENAME = "abstracts.ivfpq"
NPY_FILENAME = "paper_ids.npy"

# Ask Postgres for the first version's date rather than update_date: update_date
# is when the metadata last changed (2008 for a 2007 paper), not publication.
HYDRATE_SQL = """
    SELECT id, title, abstract, versions->0->>'created' AS created
    FROM papers
    WHERE id = ANY(%s)
"""

logger = get_logger(__name__)


def _year(created: str | None) -> int:
    """Year from an RFC-2822 version date, e.g. 'Mon, 2 Apr 2007 19:18:42 GMT'."""
    if not created:
        return 0
    try:
        return parsedate_to_datetime(created).year
    except (TypeError, ValueError):
        return 0


def _l2_normalize(vectors: np.ndarray) -> None:
    """L2-normalize in place so inner product equals cosine similarity."""
    faiss.normalize_L2(vectors)


class VectorStore:
    """Loads the abstract index once, then serves nearest-neighbour queries."""

    def __init__(self):
        self._index_path = settings.faiss_dir / IVFPQ_FILENAME
        self._ids_path = settings.faiss_dir / NPY_FILENAME
        self._index: faiss.Index | None = None
        self._ids: np.ndarray | None = None

        self._load()

    def _load(self) -> None:
        if self._index is not None:
            return

        if not self._index_path.exists():
            raise FileNotFoundError(
                f"no abstract index at {self._index_path}. "
                "Build it with: python src/loader.py --only-embeddings"
            )

        logger.info("loading FAISS index from %s", self._index_path)
        index = faiss.read_index(str(self._index_path))
        index.nprobe = settings.faiss_nprobe
        ids = np.load(self._ids_path)

        if len(ids) != index.ntotal:
            logger.error(
                f"index/id-map mismatch: {index.ntotal:,} vectors vs "
                f"{len(ids):,} ids — rebuild the index."
            )

            raise RuntimeError("Could not load vectorstore")

        self._index, self._ids = index, ids

    def search(self, embedding: np.ndarray, k: int) -> list[PaperHit]:
        """Return the k papers whose abstracts sit closest to `embedding`."""
        logger.info(f"Searching nearest {k} embeddings")

        if k <= 0:
            return []

        ranked_ids = self._nearest_ids(embedding, k)
        if not ranked_ids:
            return []

        rows_by_id = self._hydrate(ranked_ids)
        return [
            self._to_hit(pid, rows_by_id[pid]) for pid in ranked_ids if pid in rows_by_id
        ]

    def _nearest_ids(self, embedding: np.ndarray, k: int) -> list[str]:
        """Query FAISS and translate ordinals back into paper ids, ranked by similarity."""
        query = np.ascontiguousarray(embedding, dtype="float32").reshape(1, -1)
        _l2_normalize(query)

        _, ordinals = self._index.search(query, k)
        return [
            self._ids[o].decode()
            for o in ordinals[0]
            if o != -1  # FAISS pads with -1 when fewer than k results exist
        ]

    @staticmethod
    def _hydrate(paper_ids: list[str]) -> dict[str, dict]:
        """Fetch title/abstract/date for a batch of paper ids. Order is not preserved."""
        with read_only_connection() as conn:
            rows = conn.execute(HYDRATE_SQL, (paper_ids,)).fetchall()
        return {row["id"]: row for row in rows}

    @staticmethod
    def _to_hit(paper_id: str, row: dict) -> PaperHit:
        return PaperHit(
            paper_id=paper_id,
            title=row["title"],
            year=_year(row["created"]),
            abstract=row["abstract"],
            url=ARXIV_URL.format(paper_id=paper_id),
        )