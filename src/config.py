"""Tunables for the services.

Values below are defaults; any of them can be overridden via environment
variables or the ``.env`` file at the repo root (e.g. ``DEFAULT_TOP_K=5`` in the
environment takes precedence over the default here).
"""

from pathlib import Path
from urllib.parse import urlparse

from pydantic_settings import BaseSettings, SettingsConfigDict

# Repo root, resolved from this file so paths work from any working directory
ROOT_DIR = Path(__file__).resolve().parents[1]


class Settings(BaseSettings):
    # Models
    gen_model_id: str = "Qwen/Qwen3-1.7B"
    embed_model_id: str = "allenai/specter2_base"
    rerank_model_id: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    # How the generative model is placed on the available hardware
    dtype: str = "bfloat16"
    device_map: str = "auto"

    # Generation defaults
    default_max_new_tokens: int = 150
    default_temperature: float = 0.3

    # HyDE writes a hypothetical abstract, so it wants room and some creativity
    hyde_temperature: float = 0.7
    hyde_max_new_tokens: int = 200

    # SQL generation is deterministic
    sql_temperature: float = 0.0
    sql_max_new_tokens: int = 256

    # Retrieval
    default_top_k: int = 3

    # Postgres — these mirror the POSTGRES_* vars docker-compose.yml interpolates,
    # so a single .env drives the container and the client identically.
    postgres_user: str = "paperweht"
    postgres_password: str = "paperweight"
    postgres_db: str = "paperweight"
    postgres_host: str = "localhost"
    postgres_port: int = 5432

    # Set DATABASE_URL to override the composed DSN outright (e.g. a managed database)
    database_url: str | None = None

    db_connect_timeout: int = 10
    db_statement_timeout_ms: int = 30_000  # ceiling for tool queries, not for the loader

    # FAISS abstract index, built by src/embeddings.py into ./data
    faiss_dir: Path = ROOT_DIR / "data" / "faiss"
    embed_dim: int = 768
    embed_batch_size: int = 128
    # SPECTER2 is trained for CLS pooling; SentenceTransformer would default to
    # mean, which measured 82.8% vs 99.0% rank-1 on title->paper retrieval.
    embed_pooling: str = "cls"
    # 256 tokens truncates 17.8% of abstracts with no measured quality loss,
    # and is ~1.5x faster than 512.
    embed_max_seq_length: int = 256
    embed_device: str = "mps"
    embed_dtype: str = "float16"
    embed_page_size: int = 2_048       # rows per keyset page; also the progress interval
    embed_checkpoint_every: int = 50_000   # ~12 min of work at ~70 vectors/s
    embed_min_free_gb: float = 3.0

    faiss_nlist: int = 4096
    faiss_pq_m: int = 96               # 768 / 96 = 8 dims per subquantizer
    faiss_train_sample: int = 262_144
    # Lists probed per query. IVF defaults to 1, which would cripple recall.
    faiss_nprobe: int = 32

    # Bulk loader — runs once, the first time this MCP is set up
    data_path: Path = ROOT_DIR / "data" / "arxiv-metadata-oai-snapshot.json"
    loader_name: str = "arxiv_metadata"  # key in the loader_state table
    loader_progress_every: int = 5_000   # lines between progress-bar refreshes
    loader_log_bad_lines: int = 20       # log this many malformed lines, then just count

    @property
    def dsn(self) -> str:
        """The connection string to use. DATABASE_URL wins if set."""
        if self.database_url:
            return self.database_url
        return (
            f"postgresql://{self.postgres_user}:{self.postgres_password}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )

    model_config = SettingsConfigDict(
        # Anchored to the repo root so the same config works from any working
        # directory — an MCP client launches src/server.py with a cwd of its own.
        env_file=ROOT_DIR / ".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )


def masked_dsn(dsn: str) -> str:
    """A DSN with the credentials stripped, safe to print in logs and errors."""
    parsed = urlparse(dsn)
    host = parsed.hostname or ""
    port = f":{parsed.port}" if parsed.port else ""
    user = f"{parsed.username}@" if parsed.username else ""
    return f"{parsed.scheme}://{user}{host}{port}{parsed.path}"


# Singleton instance — import this everywhere
settings = Settings()
