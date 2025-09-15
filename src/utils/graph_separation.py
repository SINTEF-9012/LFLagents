# -*- coding: utf-8 -*-
"""Neo4j Graph Separation Utilities.
This module provides utilities for working with separated graphs in Neo4j
when you have multiple data sources creating distinct knowledge graphs.
"""

from utils.neo4j_connection import graph

class GraphSeparationQueries:
    """
    Collection of Cypher queries to work with separated graphs in Neo4j.
    """
    
    @staticmethod
    def get_nodes_by_source(source_name: str):
        """Get all nodes from a specific source."""
        query = """
        MATCH (n)
        WHERE n.source = $source_name
        RETURN n
        """
        return graph.query(query, {"source_name": source_name})
    
    @staticmethod
    def get_all_sources():
        """Get list of all data sources in the database."""
        query = """
        MATCH (n)
        WHERE n.source IS NOT NULL
        RETURN DISTINCT n.source as source, count(n) as node_count
        ORDER BY source
        """
        return graph.query(query)
    
    @staticmethod
    def get_relationships_by_source(source_name: str):
        """Get all relationships involving nodes from a specific source."""
        query = """
        MATCH (a)-[r]->(b)
        WHERE a.source = $source_name OR b.source = $source_name
        RETURN a, r, b
        """
        return graph.query(query, {"source_name": source_name})
    
    @staticmethod
    def get_cross_source_relationships():
        """Find relationships that connect different data sources."""
        query = """
        MATCH (a)-[r]->(b)
        WHERE a.source IS NOT NULL AND b.source IS NOT NULL AND a.source <> b.source
        RETURN a.source as source_a, b.source as source_b, type(r) as relationship_type, count(*) as count
        ORDER BY count DESC
        """
        return graph.query(query)
    
    @staticmethod
    def delete_source_data(source_name: str):
        """Delete all nodes and relationships from a specific source."""
        query = """
        MATCH (n)
        WHERE n.source = $source_name
        DETACH DELETE n
        """
        return graph.query(query, {"source_name": source_name})
    
    @staticmethod
    def get_source_statistics():
        """Get detailed statistics for each data source."""
        query = """
        MATCH (n)
        WHERE n.source IS NOT NULL
        WITH n.source as source, 
             count(n) as node_count,
             collect(DISTINCT labels(n)) as label_types
        OPTIONAL MATCH (a)-[r]->(b)
        WHERE a.source = source
        WITH source, node_count, label_types, count(r) as relationship_count
        RETURN source, node_count, relationship_count, label_types
        ORDER BY source
        """
        return graph.query(query)
    
    @staticmethod
    def get_vector_indexes():
        """Get information about all vector indexes."""
        query = "SHOW VECTOR INDEXES"
        try:
            return graph.query(query)
        except Exception as e:
            print(f"Error getting vector indexes: {e}")
            return []
    
    @staticmethod
    def search_vector_by_source(source_name: str, query_text: str, k: int = 5):
        """Perform vector similarity search within a specific source."""
        from utils.neo4j_connection import load_vector_index
        
        try:
            # Load source-specific vector index
            index_name = f"documents_{source_name}"
            vector_index = load_vector_index(index_name, source_filter=source_name)
            
            # Perform similarity search
            results = vector_index.similarity_search(query_text, k=k)
            return results
        except Exception as e:
            print(f"Error in vector search for {source_name}: {e}")
            return []
        """Search for nodes containing a term within a specific source."""
        query = """
        MATCH (n)
        WHERE n.source = $source_name
        AND (n.id CONTAINS $search_term OR 
             any(prop in keys(n) WHERE toString(n[prop]) CONTAINS $search_term))
        RETURN n
        LIMIT 20
        """
        return graph.query(query, {"source_name": source_name, "search_term": search_term})

def print_source_summary():
    """Print a summary of all data sources in the database."""
    print("DATA SOURCE SUMMARY")
    print("=" * 50)
    
    sources = GraphSeparationQueries.get_all_sources()
    if not sources:
        print("No data sources found in the database.")
        return
    
    for source_info in sources:
        source = source_info['source']
        count = source_info['node_count']
        print(f"🔹 {source}: {count} nodes")
    
    print("\nDETAILED STATISTICS")
    print("=" * 50)
    
    stats = GraphSeparationQueries.get_source_statistics()
    for stat in stats:
        source = stat['source']
        nodes = stat['node_count']
        rels = stat['relationship_count']
        labels = stat['label_types']
        
        print(f"\nSource: {source}")
        print(f"   Nodes: {nodes}")
        print(f"   Relationships: {rels}")
        print(f"   Node Types: {labels}")
    
    print("\nCROSS-SOURCE CONNECTIONS")
    print("=" * 50)
    
    cross_connections = GraphSeparationQueries.get_cross_source_relationships()
    if cross_connections:
        for conn in cross_connections:
            print(f"   {conn['source_a']} → {conn['source_b']}: {conn['count']} {conn['relationship_type']} relationships")
    else:
        print("   No cross-source relationships found.")

def separate_graph_examples():
    """Print example queries for working with separated graphs."""
    print("\nEXAMPLE QUERIES FOR SEPARATED GRAPHS")
    print("=" * 60)
    
    examples = [
        ("Get all UC1 nodes", "MATCH (n) WHERE n.source = 'uc1' RETURN n"),
        ("Get all UC2 nodes", "MATCH (n) WHERE n.source = 'uc2' RETURN n"),
        ("Count nodes by source", "MATCH (n) RETURN n.source, count(n) as node_count"),
        ("Find UC1 machines", "MATCH (n) WHERE n.source = 'uc1' AND n.id CONTAINS 'machine' RETURN n"),
        ("Get UC2 relationships", "MATCH (a)-[r]->(b) WHERE a.source = 'uc2' RETURN a, r, b"),
        ("Compare sources", "MATCH (n) RETURN n.source, labels(n), count(*) as count ORDER BY n.source"),
    ]
    
    for i, (description, query) in enumerate(examples, 1):
        print(f"\n{i}. {description}:")
        print(f"   {query}")

if __name__ == "__main__":
    print_source_summary()
    separate_graph_examples()
