"""
Query Helper Functions for Neo4j Database
"""

from .neo4j_connection import graph

def get_documents_by_source(source: str, limit: int = None):
    """
    Retrieve all documents from a specific source.
    
    Args:
        source (str): Source identifier (e.g., 'uc1', 'uc2')
        limit (int): Optional limit for number of results
    
    Returns:
        list: List of document nodes
    """
    query = f"""
    MATCH (n:Document)
    WHERE n.subdirectory = '{source}'
    RETURN n
    """
    
    if limit:
        query += f" LIMIT {limit}"
    
    result = graph.query(query)
    return result

def get_document_count_by_source(source: str = None):
    """
    Get document count by source.
    
    Args:
        source (str): Optional source filter. If None, returns count for all sources.
    
    Returns:
        dict or int: Document counts
    """
    if source:
        query = f"""
        MATCH (n:Document)
        WHERE n.subdirectory = '{source}'
        RETURN count(n) as count
        """
        result = graph.query(query)
        return result[0]['count'] if result else 0
    else:
        query = """
        MATCH (n:Document)
        RETURN n.subdirectory as source, count(n) as count
        ORDER BY source
        """
        result = graph.query(query)
        return {row['source']: row['count'] for row in result}

def get_all_sources():
    """
    Get all available sources in the database.
    
    Returns:
        list: List of source identifiers
    """
    query = """
    MATCH (n:Document)
    RETURN DISTINCT n.subdirectory as source
    ORDER BY source
    """
    result = graph.query(query)
    return [row['source'] for row in result]

def search_documents_by_content(search_term: str, source: str = None, limit: int = 10):
    """
    Search documents by content text.
    
    Args:
        search_term (str): Text to search for
        source (str): Optional source filter
        limit (int): Maximum number of results
    
    Returns:
        list: List of matching documents
    """
    where_clause = f"n.text CONTAINS '{search_term}'"
    if source:
        where_clause += f" AND n.subdirectory = '{source}'"
    
    query = f"""
    MATCH (n:Document)
    WHERE {where_clause}
    RETURN n.text, n.subdirectory, n.file_name, n.page
    LIMIT {limit}
    """
    
    result = graph.query(query)
    return result

def get_document_sample(source: str, sample_size: int = 5):
    """
    Get a sample of documents from a specific source.
    
    Args:
        source (str): Source identifier
        sample_size (int): Number of sample documents to return
    
    Returns:
        list: Sample documents
    """
    query = f"""
    MATCH (n:Document)
    WHERE n.subdirectory = '{source}'
    RETURN n.text, n.subdirectory, n.file_name, n.page, n.total_pages
    LIMIT {sample_size}
    """
    
    result = graph.query(query)
    return result

def get_nodes_by_source(source: str, node_type: str = None, limit: int = 20):
    """
    Get knowledge graph nodes (not Document nodes) by source.
    
    Args:
        source (str): Source identifier (e.g., 'uc1', 'uc2')
        node_type (str): Optional node type filter (e.g., 'uc1:Machine')
        limit (int): Maximum number of results
    
    Returns:
        list: List of nodes
    """
    if node_type:
        query = f"""
        MATCH (n:{node_type})
        WHERE n.subdirectory = '{source}' OR n.source = '{source}'
        RETURN n
        LIMIT {limit}
        """
    else:
        # Get all nodes that have the source prefix in their type or properties
        query = f"""
        MATCH (n)
        WHERE (n.subdirectory = '{source}' OR n.source = '{source}') AND NOT n:Document
        RETURN n, labels(n) as node_labels
        LIMIT {limit}
        """
    
    result = graph.query(query)
    return result
