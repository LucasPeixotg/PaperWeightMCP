import dataclasses
import json

from common import PaperData
from services.api import OpenAlexClient, SemanticScholarClient
from utils.query_cleaner import remove_wildcards
from utils.paper_deduplicator import remove_duplicates

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
    open_alex_semantic_papers = open_alex_client.semantic_search_papers(cleaned_query, limit=top_k)

    ## extending all results
    # relevant_papers.extend(semantic_scholar_papers)
    # relevant_papers.extend(open_alex_papers)
    relevant_papers.extend(open_alex_semantic_papers)

    ## post processing (deduplicate, complement)
    post_processed = remove_duplicates(relevant_papers)

    ## reranking in the future
    top_papers = post_processed[:top_k]

    result = json.dumps([dataclasses.asdict(paper) for paper in top_papers])
    return result

    # raise ToolError("Not Yet Implemented")