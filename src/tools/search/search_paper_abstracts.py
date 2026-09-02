import json
from dataclasses import asdict

from services.rag import RagPipeline

rag = RagPipeline()

def search_paper_abstracts(query: str, top_k: int = 3) -> str:
    """
    Performs a hybrid search over pre-indexed paper abstracts to retrieve the
    most relevant publications that answer a query.

    Two retrievers run over the same corpus and their rankings are merged:
    semantic search, which finds papers about the same idea in different words,
    and full-text search, which matches the query's terms verbatim. Literal
    tokens therefore work as well as paraphrases — quote a model name, a dataset
    name, an author surname or an acronym and it will be matched exactly.

    Use this tool as the first step to discover relevant research papers based on
    a natural language question, topic, or search prompt.

    Args:
        query: The natural language question or topic description to search for.
            Supports quoted phrases ("attention is all you need") and a leading
            minus to exclude a term (-survey).
        top_k: The maximum number of top matching paper abstracts to retrieve (default is 3).

    Returns:
        JSON string containing a list of top-k paper matches with their paper IDs, 
        titles, publication years, abstracts, and direct paper URLs.
    """
    papers = rag.retrieve(query, top_k)
    response = json.dumps([asdict(paper) for paper in papers], indent=2)

    return response