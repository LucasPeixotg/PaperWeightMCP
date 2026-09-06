import dataclasses
import json

from common import PaperData
from config import settings
from services.api import OpenAlexClient, SemanticScholarClient
from services.models import rerank

from .paper_deduplicator import remove_duplicates
from .query_cleaner import remove_wildcards

semantic_scholar_client = SemanticScholarClient()
open_alex_client = OpenAlexClient()

def find_revelant_papers(query: str, top_k: int = 5) -> str:
    """
    Finds the most relevant papers through multiple sources that could answer a natural language query.

    Args:
        query: The natural language query to search for.
        top_k: The maximum number of top matching paper abstracts to retrieve (default is 5).

    Returns:
        JSON string containing a list of top-k paper matches with their paper IDs,
        DOIs, titles, publication years, abstracts, direct paper URLs and licenses.
    """

    relevant_papers: list[PaperData] = []

    ## searching through multiple sources (parallel in near future)
    # semantic_scholar_papers = semantic_scholar_client.search_papers(query)

    cleaned_query = remove_wildcards(query)

    # open_alex_papers = open_alex_client.search_papers(cleaned_query)
    # Deliberately over-fetched: the reranker can only improve on the order it is
    # given, so it needs a pool wider than the `top_k` slots it fills.
    open_alex_semantic_papers = open_alex_client.semantic_search_papers(
        cleaned_query, limit=max(top_k*2, settings.RERANK_CANDIDATE_POOL)
    )

    ## extending all results
    # relevant_papers.extend(semantic_scholar_papers)
    # relevant_papers.extend(open_alex_papers)
    relevant_papers.extend(open_alex_semantic_papers)

    ## post processing (deduplicate, complement)
    post_processed = remove_duplicates(relevant_papers)

    ## reranking — scored against the raw query, since wildcard cleaning exists
    ## for search APIs that read `?` and `*` as operators, not for a model that
    ## reads the punctuation as language.
    top_papers = rerank(query, post_processed, top_k)

    result = json.dumps([dataclasses.asdict(paper) for paper in top_papers])
    return result