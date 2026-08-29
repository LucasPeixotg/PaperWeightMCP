"""Retrieval pipeline: query in, ranked papers out.

RagPipeline is the entry point the ``search_paper_abstracts`` tool calls. It
owns the end-to-end flow (model + vectorstore + HyDE + rerank) so
that the tool itself stays a thin MCP wrapper.
"""

from config import settings

from .hyde import hyde_embed
from .paper_hit import PaperHit
from .rag_model import RagModel
from .rerank import rerank as rerank_fn
from .vectorstore import VectorStore  # owns index load + nearest-neighbor search


class RagPipeline:
    """Loads the model and vectorstore once, then serves retrieval queries."""

    def __init__(
        self,
        model: RagModel | None = None,
        vectorstore: VectorStore | None = None,
    ):
        self.model = model or RagModel()
        self.vectorstore = vectorstore or VectorStore()

    def retrieve(
        self,
        query: str,
        top_k: int = settings.default_top_k,
        use_rerank: bool = False,
    ) -> list[PaperHit]:
        """Find the papers whose abstracts best answer a natural language query.

        Translates the query with HyDE, searches the abstract index, and
        optionally reranks candidates before truncating to top_k.
        """
        embedding = hyde_embed(self.model, query)

        # Over-fetch when reranking so there's a pool to re-sort from.
        fetch_k = top_k * 4 if use_rerank else top_k
        hits = self.vectorstore.search(embedding, fetch_k)

        if use_rerank and hits:
            abstracts = [hit.abstract for hit in hits]
            reranked_abstracts = rerank_fn(self.model, query, abstracts, top_k=top_k)
            hits_by_abstract = {hit.abstract: hit for hit in hits}
            hits = [hits_by_abstract[a] for a in reranked_abstracts]

        return hits[:top_k]