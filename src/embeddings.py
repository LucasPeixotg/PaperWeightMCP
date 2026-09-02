"""Build the FAISS index over paper abstracts.

This is the second loader phase. It reads every row from the `papers` table
Postgres already holds, embeds `title[SEP]abstract` with the configured
SentenceTransformer, and writes a compressed IVF-PQ index into `./data/faiss`.

At roughly 70 abstracts/s on Apple silicon the full corpus takes ~12 hours, so
the build checkpoints itself and can be interrupted and resumed freely.
"""

import math
import os
import shutil
import sys
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import psycopg
from tqdm import tqdm

from config import settings

# arXiv ids are at most 16 characters ("supr-con/9609002"), verified against the
# whole table, so a fixed-width byte array maps ordinals back to ids compactly.
ID_DTYPE = "S16"

CREATE_EMBED_STATE_SQL = """
CREATE TABLE IF NOT EXISTS embed_state (
    name           TEXT PRIMARY KEY,
    status         TEXT NOT NULL,        -- 'running' | 'completed'
    vectors_added  BIGINT NOT NULL,
    last_id        TEXT,
    model_id       TEXT,
    updated_at     TIMESTAMPTZ NOT NULL DEFAULT now()
)
"""

UPSERT_EMBED_STATE_SQL = """
INSERT INTO embed_state (name, status, vectors_added, last_id, model_id, updated_at)
VALUES (%s, %s, %s, %s, %s, now())
ON CONFLICT (name) DO UPDATE SET
    status        = EXCLUDED.status,
    vectors_added = EXCLUDED.vectors_added,
    last_id       = EXCLUDED.last_id,
    model_id      = EXCLUDED.model_id,
    updated_at    = EXCLUDED.updated_at
"""


class ResilientConn:
    """Postgres access that survives the connection dropping mid-build.

    A multi-hour embed outlives things a short query never meets: a laptop
    sleeping, Docker restarting, a network blip. Any of those kills the socket,
    and losing hours of GPU work to a dead bookkeeping connection is not a
    trade worth making — so reads and writes reconnect once and retry.
    """

    def __init__(self, conn):
        self._conn = conn

    def execute(self, sql, params=None):
        try:
            return self._conn.execute(sql, params)
        except psycopg.OperationalError:
            print("  postgres connection lost — reconnecting", flush=True)
            self._conn = psycopg.connect(settings.dsn, autocommit=True)
            return self._conn.execute(sql, params)


def index_path() -> Path:
    return settings.faiss_dir / "abstracts.ivfpq"


def ids_path() -> Path:
    return settings.faiss_dir / "paper_ids.npy"


def trained_path() -> Path:
    return settings.faiss_dir / "abstracts.trained"


def needs_training() -> bool:
    """True when the next build will have to embed a training sample.

    Both a checkpoint and a saved quantizer let `train` return without embedding
    anything, so only their absence means the multi-minute sample pass actually
    runs. `loader.py` asks before the run starts, to decide whether training is
    a phase of its own in the "[2/4]" counter.
    """
    if index_path().exists() and ids_path().exists():
        return False
    return not trained_path().exists()


def _phase(progress, label):
    """`progress.phase(label)`, or a no-op when the caller kept no Progress."""
    return nullcontext() if progress is None else progress.phase(label)


def _atomic_write(path: Path, write) -> None:
    """Write via a temp file and rename, so a crash never leaves a torn file."""
    tmp = path.with_suffix(path.suffix + ".tmp")
    write(tmp)
    os.replace(tmp, path)


def check_openmp() -> None:
    """Guard against the faiss/torch OpenMP clash on macOS.

    Both wheels vendor their own libomp.dylib. Loading both into one process
    aborts with "OMP: Error #15" and a SIGSEGV — and this build needs both at
    once, since it embeds with torch and adds to the index in the same loop.
    Pointing faiss at torch's runtime fixes it (verified: 100% agreement with
    exact search afterwards). Detected here so a faiss reinstall produces this
    message instead of a crash 20 minutes in.
    """
    if sys.platform != "darwin":
        return
    from importlib.util import find_spec

    try:
        faiss_omp = Path(find_spec("faiss").origin).parent / ".dylibs" / "libomp.dylib"
        torch_omp = Path(find_spec("torch").origin).parent / "lib" / "libomp.dylib"
    except (AttributeError, ValueError, ModuleNotFoundError):
        return

    if faiss_omp.exists() and torch_omp.exists() and not faiss_omp.is_symlink():
        raise SystemExit(
            "faiss and torch each bundle their own libomp, which segfaults when both "
            "load. Point faiss at torch's copy:\n"
            f"  ln -sf {torch_omp} {faiss_omp}"
        )


def preflight(min_free_gb: float | None = None) -> None:
    """Fail before loading a model or embedding anything if disk is short."""
    check_openmp()
    settings.faiss_dir.mkdir(parents=True, exist_ok=True)
    floor = settings.embed_min_free_gb if min_free_gb is None else min_free_gb
    free_gb = shutil.disk_usage(settings.faiss_dir).free / 1e9
    if free_gb < floor:
        raise SystemExit(
            f"only {free_gb:.1f} GB free on {settings.faiss_dir}, need {floor:.1f} GB. "
            "Free some space and re-run."
        )


def load_embedder():
    """The SentenceTransformer, assembled explicitly rather than by auto-config.

    SPECTER2 ships as a plain transformers checkpoint, so SentenceTransformer
    would infer *mean* pooling. It is trained for CLS: mean measured 82.8% vs
    99.0% rank-1 on title->own-paper retrieval, so the pooling mode is set here
    rather than inherited.
    """
    import torch
    from sentence_transformers import SentenceTransformer, models

    device = settings.embed_device
    if device == "mps" and not torch.backends.mps.is_available():
        device = "cpu"

    word = models.Transformer(
        settings.embed_model_id,
        max_seq_length=settings.embed_max_seq_length,
        model_kwargs={"torch_dtype": getattr(torch, settings.embed_dtype)},
    )
    pooling = models.Pooling(
        word.get_word_embedding_dimension(), pooling_mode=settings.embed_pooling
    )
    return SentenceTransformer(modules=[word, pooling], device=device)


def encode(model, rows) -> np.ndarray:
    """Embed (title, abstract) pairs into L2-normalized float32 vectors.

    "title[SEP]abstract" is the input format SPECTER2 expects. Vectors are
    normalized so FAISS inner product is cosine similarity.
    """
    texts = [f"{title}[SEP]{abstract}" for _, title, abstract in rows]
    vectors = model.encode(
        texts,
        batch_size=settings.embed_batch_size,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=False,
    )
    return np.ascontiguousarray(vectors, dtype="float32")


def iter_pages(conn, after_id: str | None, limit: int | None):
    """Yield pages of papers in `id` order via keyset pagination.

    Keyset rather than OFFSET so resuming is a single indexed seek, and so the
    3.1M rows never materialize in Python at once.
    """
    cursor_id = after_id or ""
    yielded = 0
    while True:
        page_size = settings.embed_page_size
        if limit is not None:
            page_size = min(page_size, limit - yielded)
            if page_size <= 0:
                return
        rows = conn.execute(
            "SELECT id, title, abstract FROM papers WHERE id > %s ORDER BY id LIMIT %s",
            (cursor_id, page_size),
        ).fetchall()
        if not rows:
            return
        yield rows
        yielded += len(rows)
        cursor_id = rows[-1][0]


def train(model, conn, nlist: int, sample_size: int):
    """Train the IVF-PQ quantizer on a random sample, then persist it.

    IVF-PQ cannot accept vectors until it has learned its centroids and
    codebooks, so this pass runs first. The trained-but-empty index is saved so
    a resumed build never repeats it.
    """
    import faiss

    if trained_path().exists():
        print(f"  reusing trained index at {trained_path().name}")
        return faiss.read_index(str(trained_path()))

    # TABLESAMPLE is a cheap random scan; ORDER BY random() would sort 3.1M rows.
    pct = min(100.0, 100.0 * sample_size / max(1, _count(conn)) * 1.3)
    print(f"  sampling ~{sample_size:,} papers for training (TABLESAMPLE {pct:.1f}%)")
    rows = conn.execute(
        "SELECT id, title, abstract FROM papers TABLESAMPLE SYSTEM (%s) LIMIT %s",
        (pct, sample_size),
    ).fetchall()

    print(f"  embedding {len(rows):,} training vectors ...")
    chunks = []
    with tqdm(total=len(rows), unit=" vec", desc="  training sample",
              smoothing=0.05) as bar:
        for i in range(0, len(rows), settings.embed_page_size):
            chunk = rows[i : i + settings.embed_page_size]
            chunks.append(encode(model, chunk))
            bar.update(len(chunk))
    training = np.vstack(chunks)

    quantizer = faiss.IndexFlatIP(settings.embed_dim)
    index = faiss.IndexIVFPQ(
        quantizer, settings.embed_dim, nlist, settings.faiss_pq_m, 8,
        faiss.METRIC_INNER_PRODUCT,
    )
    print(f"  training IVF-PQ (nlist={nlist}, m={settings.faiss_pq_m}) ...")
    index.train(training)
    _atomic_write(trained_path(), lambda p: faiss.write_index(index, str(p)))
    return index


def _count(conn) -> int:
    return conn.execute("SELECT count(*) FROM papers").fetchone()[0]


def _load_checkpoint(index_file: Path, ids_file: Path):
    """Reload a previous run, reconciling the index against the id map.

    Checkpoints write the id map *before* the index, so a crash between the two
    can only leave more ids than vectors. Truncating the ids back to
    `index.ntotal` restores a consistent pair; the reverse would be
    unrecoverable, which is why the write order matters.
    """
    import faiss

    if not (index_file.exists() and ids_file.exists()):
        return None, []
    index = faiss.read_index(str(index_file))
    ids = list(np.load(ids_file))
    if len(ids) > index.ntotal:
        print(f"  torn checkpoint: {len(ids):,} ids vs {index.ntotal:,} vectors — truncating ids")
        ids = ids[: index.ntotal]
    elif index.ntotal > len(ids):
        raise SystemExit(
            f"index has {index.ntotal:,} vectors but only {len(ids):,} ids — cannot map "
            f"results back to papers. Delete {settings.faiss_dir} and rebuild."
        )
    return index, ids


def build(raw_conn, limit: int | None = None, nlist: int | None = None,
          progress=None) -> dict:
    """Embed every paper and write the FAISS index. Resumable.

    Runs as two `progress` phases — training the quantizer on a sample, then
    embedding the corpus — because the two counts are wildly different and a
    single header made the sample look like the whole job.
    """
    import faiss

    preflight()
    conn = ResilientConn(raw_conn)
    conn.execute(CREATE_EMBED_STATE_SQL)
    total = limit or _count(conn)

    # FAISS needs >= 39 * nlist training points to fit its centroids, and warns
    # but proceeds when short — yielding a quietly under-trained index. Scale
    # both knobs to the corpus so a --limit build is correct too;
    # settings.faiss_nlist is the ceiling, and an explicit --nlist still wins.
    if nlist is None:
        nlist = max(64, min(settings.faiss_nlist, int(4 * math.sqrt(total))))
    train_target = min(settings.faiss_train_sample, total, max(10_000, 40 * nlist))

    # Asked before `_load_checkpoint`, though neither writes, so the phase list
    # `loader.py` built from the same question cannot disagree with this run.
    wants_training = needs_training()

    index, ids = _load_checkpoint(index_path(), ids_path())
    model = None

    if index is None:
        model = load_embedder()
        # Not a phase when `train` will just reload the saved quantizer: an
        # instant step announced as a phase reads like something went wrong.
        label = (f"train IVF-PQ (nlist={nlist:,}) "
                 f"on {train_target:,} sampled abstracts")
        with _phase(progress if wants_training else None, label):
            index = train(model, conn, nlist, train_target)
    else:
        print(f"  resuming from {len(ids):,} vectors already indexed")

    embed_label = f"embed {total:,} abstracts"
    if ids:
        embed_label += f" ({len(ids):,} already done)"

    with _phase(progress, embed_label):
        return _embed(conn, index, ids, model, total, limit)


def _embed(conn, index, ids, model, total: int, limit: int | None) -> dict:
    """The corpus pass: embed from `ids` to `total`, checkpointing as it goes.

    Split out of `build` only so the phase it runs under is a single `with`
    rather than an indent wrapped around the whole function.
    """
    import faiss

    if len(ids) >= total:
        print("  index already complete")
        return {"added": len(ids), "total": total, "resumed": True}

    if model is None:
        model = load_embedder()

    after_id = ids[-1].decode() if ids else None
    since_checkpoint = 0
    added = 0

    def checkpoint():
        # ids first, then the index — see _load_checkpoint for why the order matters
        # np.save appends ".npy" to a path lacking it, which would defeat the
        # atomic rename — write through an open handle instead.
        def _save_ids(target):
            with open(target, "wb") as fh:
                np.save(fh, np.array(ids, dtype=ID_DTYPE))

        _atomic_write(ids_path(), _save_ids)
        _atomic_write(index_path(), lambda p: faiss.write_index(index, str(p)))
        conn.execute(
            UPSERT_EMBED_STATE_SQL,
            ("abstracts", "running", len(ids), ids[-1].decode(), settings.embed_model_id),
        )

    # `initial` is what keeps a resumed build honest: tqdm measures its rate over
    # (n - initial), i.e. over this session only, and so projects the vectors
    # still outstanding — otherwise resuming at 3.0M/3.1M would keep predicting
    # the full 12 hours. The bar itself still counts from 0 to the whole corpus.
    bar = tqdm(
        total=total,
        initial=len(ids),
        unit=" vec",
        desc="embedding abstracts",
        smoothing=0.05,
    )
    try:
        for rows in iter_pages(conn, after_id, None if limit is None else limit - len(ids)):
            vectors = encode(model, rows)
            index.add(vectors)  # ids are implicit ordinals == position in `ids`
            ids.extend(r[0].encode() for r in rows)
            added += len(rows)
            since_checkpoint += len(rows)
            bar.update(len(rows))

            if since_checkpoint >= settings.embed_checkpoint_every:
                checkpoint()
                since_checkpoint = 0
                # Through the bar, so the note scrolls above it instead of
                # cutting the bar in half.
                bar.write(f"  checkpointed at {len(ids):,}")
    finally:
        if ids:
            checkpoint()
        bar.close()

    complete = len(ids) >= total
    conn.execute(
        UPSERT_EMBED_STATE_SQL,
        (
            "abstracts",
            "completed" if complete else "running",
            len(ids),
            ids[-1].decode() if ids else None,
            settings.embed_model_id,
        ),
    )
    return {"added": added, "total": len(ids), "resumed": False}


def is_built(conn) -> bool:
    row = conn.execute(
        "SELECT status FROM embed_state WHERE name = 'abstracts'"
    ).fetchone()
    return row is not None and row[0] == "completed"
