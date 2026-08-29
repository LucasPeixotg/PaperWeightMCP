"""FAISS-backed nearest-neighbour search over paper abstracts.

The index stores only vectors; every hit's metadata is read back from Postgres,
so abstracts are not duplicated on disk. Ordinals returned by FAISS are
positions in the id map written alongside the index by `src/embeddings.py`.
"""

from __future__ import annotations

from email.utils import parsedate_to_datetime

import numpy as np

from config import settings
from services.db import read_only_connection

from .paper_hit import PaperHit

ARXIV_URL = "https://arxiv.org/abs/{paper_id}"

# Ask Postgres for the first version's date rather than update_date: update_date
# is when the metadata last changed (2008 for a 2007 paper), not publication.
HYDRATE_SQL = """
SELECT id, title, abstract, versions->0->>'created' AS created
FROM papers
WHERE id = ANY(%s)
"""


def _year(created: str | None) -> int:
    """Year from an RFC-2822 version date, e.g. 'Mon, 2 Apr 2007 19:18:42 GMT'."""
    if not created:
        return 0
    try:
        return parsedate_to_datetime(created).year
    except (TypeError, ValueError):
        return 0


class VectorStore:
    """Loads the abstract index once, then serves nearest-neighbour queries."""

    def __init__(self):
        self._index_path = settings.faiss_dir / "abstracts.ivfpq"
        self._ids_path = settings.faiss_dir / "paper_ids.npy"
        self._index = None
        self._ids: np.ndarray | None = None

    def _load(self):
        """Load lazily so importing this module never costs a 300 MB read."""
        if self._index is not None:
            return
        import faiss

        if not self._index_path.exists():
            raise FileNotFoundError(
                f"no abstract index at {self._index_path}. "
                "Build it with: python src/loader.py --only-embeddings"
            )
        self._index = faiss.read_index(str(self._index_path))
        self._index.nprobe = settings.faiss_nprobe
        self._ids = np.load(self._ids_path)
        if len(self._ids) != self._index.ntotal:
            raise RuntimeError(
                f"index/id-map mismatch: {self._index.ntotal:,} vectors vs "
                f"{len(self._ids):,} ids — rebuild the index."
            )

    def search(self, embedding: np.ndarray, k: int) -> list:
        """Return the k papers whose abstracts sit closest to `embedding`."""

        self._load()

        query = np.ascontiguousarray(embedding, dtype="float32").reshape(1, -1)
        faiss_norm(query)
        _, ordinals = self._index.search(query, k)

        ranked = [
            self._ids[o].decode()
            for o in ordinals[0]
            if o != -1  # FAISS pads with -1 when fewer than k results exist
        ]
        if not ranked:
            return []

        with read_only_connection() as conn:
            rows = conn.execute(HYDRATE_SQL, (ranked,)).fetchall()

        by_id = {row["id"]: row for row in rows}
        # SQL returns no particular order, so restore the FAISS ranking
        return [
            PaperHit(
                paper_id=pid,
                title=by_id[pid]["title"],
                year=_year(by_id[pid]["created"]),
                abstract=by_id[pid]["abstract"],
                url=ARXIV_URL.format(paper_id=pid),
            )
            for pid in ranked
            if pid in by_id
        ]


def faiss_norm(vectors: np.ndarray) -> None:
    """L2-normalize in place so inner product equals cosine similarity."""
    import faiss

    faiss.normalize_L2(vectors)
