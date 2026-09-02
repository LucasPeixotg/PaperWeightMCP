"""Retrieval pipeline: query in, ranked papers out.

RagPipeline is the entry point the ``search_paper_abstracts`` tool calls. It
owns the end-to-end flow (model + vectorstore + lexical store + HyDE + fusion +
rerank) so that the tool itself stays a thin MCP wrapper.
"""

from config import settings

from .fusion import reciprocal_rank_fusion
from .hyde import hyde_embed
from .lexicalstore import LexicalStore
from .paper_hit import PaperHit
from .rag_model import RagModel
from .reranker import Reranker
from .vectorstore import VectorStore  # owns index load + nearest-neighbor search
from.logger import get_logger

logger = get_logger(__name__)

class RagPipeline:
    """Loads the model and vectorstore once, then serves retrieval queries."""

    def __init__(self):
        self.model = RagModel()
        self.vectorstore = VectorStore()
        self.lexicalstore = LexicalStore()
        self.reranker = Reranker()

        logger.info("Pipeline loaded and ready to run")

    def retrieve(
        self,
        query: str,
        top_k: int = settings.default_top_k,
    ) -> list[PaperHit]:
        """Find the papers whose abstracts best answer a natural language query.

        Runs two retrievers over the same corpus — semantic search on a HyDE
        translation of the query, and full-text search on the query verbatim —
        fuses their rankings, then reranks the pool before truncating to top_k.
        """
        embedding = hyde_embed(self.model, query)
        dense = self.vectorstore.search(embedding, settings.dense_k)
        lexical = self._lexical(query)

        # Over-fetch before reranking so the cross-encoder has a pool to re-sort.
        pool = reciprocal_rank_fusion(
            [dense, lexical], k=settings.rrf_k, limit=settings.rerank_pool
        )
        logger.info(
            f"Fused {len(dense)} dense and {len(lexical)} lexical candidates "
            f"into a pool of {len(pool)}"
        )

        most_relevant = self.reranker.select_top(query, pool, top_k)

        return most_relevant

    def _lexical(self, query: str) -> list[PaperHit]:
        """Full-text candidates, or none if Postgres is unreachable.

        A database that is down or missing the full-text index should degrade
        the search to dense-only, not fail the tool call outright.
        """
        try:
            return self.lexicalstore.search(query, settings.lexical_k)
        except Exception:
            logger.exception("lexical search failed — falling back to dense-only")
            return []
