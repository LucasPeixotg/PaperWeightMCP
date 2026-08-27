def query_paper_metadata(query: str, limit: int = 10) -> str:
    """
    Executes precise structured filtering on paper metadata (e.g., publication year, 
    author name, citation count, category, or venue) using SQL under the hood.
    
    Use this tool instead of semantic vector search when answering queries that require 
    exact numerical, temporal, or categorical constraints—such as filtering by date 
    ranges, minimum citation thresholds, or specific author names.
    
    Args:
        query: Natural language description of the metadata filter or criteria 
               (e.g., "papers published in 2024 with over 50 citations in cs.AI").
        limit: Maximum number of matching metadata records to return (default is 10).
        
    Returns:
        JSON string containing structured metadata records matching the criteria 
        (IDs, titles, authors, publication years, citation counts, venues, categories).
    """
    return 'cannot return right now'