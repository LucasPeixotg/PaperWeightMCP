def search_paper_abstracts(query: str, top_k: int = 3) -> str:
    """
    Performs a semantic vector search over pre-indexed paper abstracts to retrieve 
    the most relevant publications that answer a query.
    
    Use this tool as the first step to discover relevant research papers based on 
    a natural language question, topic, or search prompt.
    
    Args:
        query: The natural language question or topic description to search for.
        top_k: The maximum number of top matching paper abstracts to retrieve (default is 3).
        
    Returns:
        JSON string containing a list of top-k paper matches with their paper IDs, 
        titles, publication years, abstracts, and direct paper URLs.
    """
    return 'cannot return right now'