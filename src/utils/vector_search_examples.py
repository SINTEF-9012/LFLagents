"""
Vector Search Examples for Multilingual Document Retrieval
Updated for unified vector index with source filtering
"""

from .neo4j_connection import load_vector_index

def search_uc1_documents(query: str, top_k: int = 5):
    """
    Search UC1 documents using multilingual vector similarity.
    
    Args:
        query (str): Search query (can be in English for Spanish docs)
        top_k (int): Number of top results to return
    
    Returns:
        list: Similar documents from UC1
    """
    # Load unified vector index with UC1 source filter
    uc1_vector = load_vector_index(source_filter="uc1")
    
    # Perform similarity search with multilingual embeddings
    results = uc1_vector.similarity_search(query, k=top_k)
    
    return results

def search_uc2_documents(query: str, top_k: int = 5):
    """
    Search UC2 documents using multilingual vector similarity.
    
    Args:
        query (str): Search query (can be in English for Spanish docs)
        top_k (int): Number of top results to return
    
    Returns:
        list: Similar documents from UC2
    """
    # Load unified vector index with UC2 source filter
    uc2_vector = load_vector_index(source_filter="uc2")
    
    # Perform similarity search with multilingual embeddings
    results = uc2_vector.similarity_search(query, k=top_k)
    
    return results

def search_all_documents(query: str, top_k: int = 5):
    """
    Search across all documents (UC1 + UC2) using multilingual vector similarity.
    
    Args:
        query (str): Search query (works cross-language)
        top_k (int): Number of top results to return
    
    Returns:
        list: Similar documents from all sources
    """
    # Load unified vector index without source filter
    all_vector = load_vector_index()
    
    # Perform similarity search across all documents
    results = all_vector.similarity_search(query, k=top_k)
    
    return results

def search_with_scores(query: str, source_filter=None, top_k: int = 5):
    """
    Search documents with similarity scores.
    
    Args:
        query (str): Search query
        source_filter (str): Filter by source ("uc1", "uc2", or None for all)
        top_k (int): Number of top results to return
    
    Returns:
        list: Tuples of (document, score)
    """
    vector_index = load_vector_index(source_filter=source_filter)
    
    # Get results with similarity scores
    results = vector_index.similarity_search_with_score(query, k=top_k)
    
    return results

def compare_sources(query: str, top_k: int = 3):
    """
    Compare search results between UC1 and UC2 with multilingual support.
    
    Args:
        query (str): Search query (works in English for Spanish docs)
        top_k (int): Number of top results per source
    
    Returns:
        dict: Results from both sources with scores
    """
    uc1_results = search_with_scores(query, source_filter="uc1", top_k=top_k)
    uc2_results = search_with_scores(query, source_filter="uc2", top_k=top_k)
    
    return {
        'uc1': uc1_results,
        'uc2': uc2_results,
        'query': query,
        'embedding_type': 'multilingual'
    }

def demo_cross_language_search():
    """
    Demonstrate cross-language search capabilities.
    """
    print("🌍 Multilingual Vector Search Demo")
    print("=" * 50)
    
    # English queries that should find Spanish content
    english_queries = [
        "machine maintenance procedures",
        "safety instructions", 
        "technical specifications",
        "installation guide"
    ]
    
    for query in english_queries:
        print(f"\n🔍 Query: '{query}'")
        
        # Search UC2 (Spanish documents)
        uc2_results = search_with_scores(query, source_filter="uc2", top_k=2)
        
        for i, (doc, score) in enumerate(uc2_results, 1):
            print(f"   {i}. Score: {score:.4f}")
            print(f"      File: {doc.metadata.get('file_name', 'Unknown')}")
            print(f"      Content: {doc.page_content[:100]}...")
    
    print("\n✅ Cross-language search working!")

# Example usage functions
def example_uc1_search():
    """Example: Search UC1 documents"""
    results = search_uc1_documents("process model", top_k=3)
    for i, doc in enumerate(results, 1):
        print(f"{i}. {doc.metadata.get('file_name', 'Unknown')}")
        print(f"   Content: {doc.page_content[:150]}...")

def example_uc2_search():
    """Example: Search UC2 documents (Spanish) with English query"""
    results = search_uc2_documents("machine maintenance", top_k=3)
    for i, doc in enumerate(results, 1):
        print(f"{i}. {doc.metadata.get('file_name', 'Unknown')}")
        print(f"   Content: {doc.page_content[:150]}...")

def example_compare_sources():
    """Example: Compare results across sources"""
    comparison = compare_sources("safety procedures", top_k=2)
    
    print("UC1 Results:")
    for i, (doc, score) in enumerate(comparison['uc1'], 1):
        print(f"  {i}. Score: {score:.4f} - {doc.metadata.get('file_name', 'Unknown')}")
    
    print("UC2 Results:")
    for i, (doc, score) in enumerate(comparison['uc2'], 1):
        print(f"  {i}. Score: {score:.4f} - {doc.metadata.get('file_name', 'Unknown')}")

if __name__ == "__main__":
    # Run demo
    demo_cross_language_search()
