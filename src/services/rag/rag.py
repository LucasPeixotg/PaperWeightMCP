"""Retrieval pipeline: query in, ranked papers out.

RagPipeline is the entry point the ``search_paper_abstracts`` tool calls. It
owns the end-to-end flow (model + vectorstore + HyDE + rerank) so
that the tool itself stays a thin MCP wrapper.
"""

from config import settings

from .hyde import hyde_embed
from .paper_hit import PaperHit
from .rag_model import RagModel
from .reranker import Reranker
from .vectorstore import VectorStore  # owns index load + nearest-neighbor search


class RagPipeline:
    """Loads the model and vectorstore once, then serves retrieval queries."""

    def __init__(self):
        self.model = RagModel()
        self.vectorstore = VectorStore()
        self.reranker = Reranker()

    def retrieve(
        self,
        query: str,
        top_k: int = settings.default_top_k,
    ) -> list[PaperHit]:
        """Find the papers whose abstracts best answer a natural language query.

        Translates the query with HyDE, searches the abstract index, and
        reranks candidates before truncating to top_k.
        """
        embedding = hyde_embed(self.model, query)

        # Over-fetch when reranking so there's a pool to re-sort from.
        fetch_k = top_k * 4
        hits = self.vectorstore.search(embedding, fetch_k)

        abstracts = [hit.abstract for hit in hits]
        most_relevant = self.reranker.top(query, abstracts, top_k)

        return most_relevant
