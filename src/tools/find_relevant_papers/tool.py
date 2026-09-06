import dataclasses
import json

from common import PaperData
from services.api import SemanticScholarClient

semantic_scholar_client = SemanticScholarClient()

def find_revelant_papers(query: str, top_k: int = 5) -> str:
    """
    Finds the most relevant papers through multiple sources that could answer a natural language query.

    Args:
        query: The natural language query to search for.
        top_k: The maximum number of top matching paper abstracts to retrieve (default is 5).

    Returns:
        JSON string containing a list of top-k paper matches with their paper IDs, 
        titles, publication years, abstracts, and direct paper URLs.
    """

    relevant_papers: list[PaperData] = []

    semantic_scholar_papers = semantic_scholar_client.search_papers(query)

    relevant_papers.extend(semantic_scholar_papers)
    
    return json.dumps([dataclasses.asdict(paper) for paper in semantic_scholar_papers])

    # raise ToolError("Not Yet Implemented")