def get_citations(paper_id: str, limit: int = 10) -> str:
    """
    Retrieves a list of papers that have cited the target publication (forward citation search).
    
    Use this tool when you need to track how a paper's ideas evolved, find recent 
    benchmarks building on it, or discover modern follow-up studies.
    
    Args:
        paper_id: The canonical paper identifier (e.g., arXiv ID '2301.12345', 
                  DOI '10.1038/s41586-020...', or Semantic Scholar ID).
        limit: Maximum number of citing papers to return (default is 10).
        
    Returns:
        JSON string containing titles, paper IDs, publication years, venues, 
        and citation counts for citing works.
    """
    return 'cannot return right now'