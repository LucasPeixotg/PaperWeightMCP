
def get_references(paper_id: str, limit: int = 10) -> str:
    """
    Retrieves the bibliography and reference list cited by the target paper (backward citation search).
    
    Use this tool to inspect foundational work, verify theoretical baselines, 
    or identify the core prior models the authors relied upon.
    
    Args:
        paper_id: The canonical paper identifier (e.g., arXiv ID '2301.12345', 
                  DOI '10.1038/s41586-020...', or Semantic Scholar ID).
        limit: Maximum number of referenced papers to return (default is 10).
        
    Returns:
        JSON string containing titles, lead authors, paper IDs, publication years, 
        and direct PDF links where available.
    """
    return 'cannot return right now'
