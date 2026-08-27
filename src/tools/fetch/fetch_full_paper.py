
def fetch_full_paper(paper_id_or_url: str) -> str:
    """
    Downloads and extracts the complete full-text content of a specific paper on the fly.
    
    Use this tool AFTER running `search_paper_abstracts` when you need granular 
    technical details not available in the abstract—such as specific equations, 
    detailed methodology, hyperparameter grids, or code implementations.
    
    Args:
        paper_id_or_url: The canonical paper identifier (e.g., arXiv ID '2301.12345', 
                         DOI) or the direct HTTP URL to the paper's PDF or HTML page.
            
    Returns:
        JSON string containing the full extracted plain-text body of the paper, 
        section headers (if parsed), and fetch metadata.
    """
    return 'cannot return right now'