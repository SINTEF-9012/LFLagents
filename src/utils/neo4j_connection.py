# -*- coding: utf-8 -*-
"""Neo4j Connection Utilities.
This module provides utilities for connecting to a Neo4j database.
It includes functions to establish a connection and perform basic operations.
"""

import os
from dotenv import load_dotenv, find_dotenv
from langchain_community.graphs import Neo4jGraph
from langchain_community.vectorstores import Neo4jVector
from langchain_community.embeddings import HuggingFaceEmbeddings

import warnings
warnings.filterwarnings("ignore")

# Load environment variables from .env file
load_dotenv()
CHAT_MODEL = os.getenv("CHAT_MODEL")

# Load environment variables
load_dotenv(find_dotenv(), override=True)

# Initialize multilingual embedding model for cross-language retrieval
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    model_kwargs={'device': 'cpu'},  # Use 'cuda' if you have GPU
    encode_kwargs={'normalize_embeddings': True}
)
print("🌍 Using multilingual embeddings for cross-language retrieval")

# Neo4j setup
graph = Neo4jGraph(
    url=os.environ["NEO4J_URL"],
    username=os.environ["NEO4J_USERNAME"],
    password=os.environ["NEO4J_PASSWORD"],
    # enhanced_schema=True
)

def clean_graph():
    query = """
    MATCH (n)
    DETACH DELETE n
    """
    graph.query(query)
    
def create_vector_index(index_name="documents"):
    """
    Create a unified vector index for Document nodes.
    
    Args:
        index_name (str): Name of the vector index. Defaults to "documents".
    """
    query = f"""
    CREATE VECTOR INDEX {index_name} IF NOT EXISTS FOR (n:Document) ON (n.embedding)
    """
    graph.query(query)
    print(f"Vector index '{index_name}' created (if it did not already exist) for Document nodes.")

def update_vector_index(index_name="documents", source_filter=None):
    """
    Update the Neo4j vector index for Document nodes.
    This will create embeddings for documents that don't have them yet.
    
    Args:
        index_name (str): Name of the vector index to update.
        source_filter (str): Optional source filter to update only specific source documents.
    """
    print(f"Updating vector index '{index_name}'...")
    
    # Configure node label filter if source is specified
    if source_filter:
        print(f"  - Filtering for source: {source_filter}")
        retrieval_query = f"""
        MATCH (n:Document)
        WHERE n.source = '{source_filter}'
        RETURN n.text AS text, score, n {{.*}} AS metadata
        """
    else:
        retrieval_query = None
    
    try:
        # Use from_existing_graph to create embeddings if they don't exist
        # This method handles token limits automatically by batching
        neo4j_vector = Neo4jVector.from_existing_graph(
            embedding=embedding_model,
            index_name=index_name,
            node_label="Document",
            text_node_properties=["text"],
            embedding_node_property="embedding",  # This will be created if it doesn't exist
            retrieval_query=retrieval_query,
        )
        print(f"Vector index '{index_name}' updated successfully.")
        return neo4j_vector
    except Exception as e:
        if "max_tokens_per_request" in str(e):
            print(f"Token limit exceeded. Using batch processing for '{index_name}'...")
            return update_vector_index_batch(index_name, source_filter)
        else:
            print(f"Error updating vector index '{index_name}': {e}")
            # Try alternative approach if the main one fails
            try:
                print(f"Attempting alternative setup for '{index_name}'...")
                neo4j_vector = Neo4jVector.from_existing_index(
                    embedding=embedding_model,
                    index_name=index_name,
                    url=os.environ["NEO4J_URL"],
                    username=os.environ["NEO4J_USERNAME"],
                    password=os.environ["NEO4J_PASSWORD"],
                )
                print(f"Alternative setup successful for '{index_name}'")
                return neo4j_vector
            except Exception as e2:
                print(f"Alternative setup also failed: {e2}")
                raise e

def update_vector_index_batch(index_name="documents", source_filter=None, batch_size=50):
    """
    Update vector index in smaller batches to handle token limits.
    
    Args:
        index_name (str): Name of the vector index to update.
        source_filter (str): Optional source filter.
        batch_size (int): Number of documents to process at once.
    """
    print(f"Batch processing vector index '{index_name}' with batch size {batch_size}...")
    
    # Use the global embedding model (already initialized above)
    
    # Get documents that need embeddings
    if source_filter:
        fetch_query = f"""
        MATCH (n:Document)
        WHERE n.source = '{source_filter}' AND n.embedding IS NULL
        RETURN elementId(n) AS id, n.text AS text
        ORDER BY n.page
        """
    else:
        fetch_query = """
        MATCH (n:Document)
        WHERE n.embedding IS NULL
        RETURN elementId(n) AS id, n.text AS text
        ORDER BY n.page
        """
    
    documents = graph.query(fetch_query)
    total_docs = len(documents)
    print(f"Found {total_docs} documents needing embeddings")
    
    if total_docs == 0:
        print("All documents already have embeddings")
        return Neo4jVector.from_existing_index(
            embedding=embedding_model,
            index_name=index_name,
            url=os.environ["NEO4J_URL"],
            username=os.environ["NEO4J_USERNAME"],
            password=os.environ["NEO4J_PASSWORD"],
        )
    
    # Process in batches
    processed = 0
    for i in range(0, total_docs, batch_size):
        batch = documents[i:i + batch_size]
        
        try:
            # Truncate text if too long (roughly 250K characters = ~62K tokens)
            texts = []
            for doc in batch:
                text = doc['text']
                if len(text) > 250000:  # Truncate very long documents
                    text = text[:250000] + "... [truncated]"
                texts.append(text)
            
            # Create embeddings for this batch
            embeddings = embedding_model.embed_documents(texts)
            
            # Update Neo4j with embeddings
            batch_data = [
                {"id": doc["id"], "embedding": emb}
                for doc, emb in zip(batch, embeddings)
            ]
            
            update_query = """
            UNWIND $batch AS row
            MATCH (n:Document)
            WHERE elementId(n) = row.id
            CALL db.create.setNodeVectorProperty(n, 'embedding', row.embedding)
            RETURN count(*) as updated
            """
            
            result = graph.query(update_query, params={"batch": batch_data})
            batch_processed = result[0]['updated'] if result else 0
            processed += batch_processed
            
            print(f"  Processed batch {i//batch_size + 1}/{(total_docs + batch_size - 1)//batch_size}: {batch_processed} documents")
            
        except Exception as e:
            print(f"Error processing batch {i//batch_size + 1}: {e}")
            # Try with smaller batch if token limit still exceeded
            if "max_tokens_per_request" in str(e) and batch_size > 10:
                print(f"Retrying with smaller batch size...")
                smaller_batch = batch[:batch_size//2]
                try:
                    texts = [doc['text'][:100000] for doc in smaller_batch]  # Further truncate
                    embeddings = embedding_model.embed_documents(texts)
                    
                    batch_data = [
                        {"id": doc["id"], "embedding": emb}
                        for doc, emb in zip(smaller_batch, embeddings)
                    ]
                    
                    result = graph.query(update_query, params={"batch": batch_data})
                    batch_processed = result[0]['updated'] if result else 0
                    processed += batch_processed
                    print(f"  Smaller batch processed: {batch_processed} documents")
                except Exception as e2:
                    print(f"Even smaller batch failed: {e2}")
            continue
    
    print(f"Batch processing completed. {processed} documents updated with embeddings.")
    
    # Return the vector index
    return Neo4jVector.from_existing_index(
        embedding=embedding_model,
        index_name=index_name,
        url=os.environ["NEO4J_URL"],
        username=os.environ["NEO4J_USERNAME"],
        password=os.environ["NEO4J_PASSWORD"],
    )

def setup_single_source_vector_index(source: str):
    """
    Set up unified vector index and verify documents for a specific source.
    This function is deprecated - use create_vector_index("documents") instead.
    
    Args:
        source (str): Source identifier (e.g., 'uc1', 'uc2')
    
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        print(f"Setting up unified vector index for source: {source}")
        
        # Check if documents exist for this source
        check_query = f"""
        MATCH (n:Document)
        WHERE n.source = '{source}'
        RETURN count(n) as count
        """
        result = graph.query(check_query)
        doc_count = result[0]['count'] if result else 0
        
        if doc_count == 0:
            print(f"No documents found for source: {source}")
            return False
        
        print(f"Found {doc_count} documents for {source}")
        
        # Create unified index instead of source-specific
        create_vector_index("documents")
        update_vector_index("documents")
        
        print(f"Unified vector index setup completed for {source}")
        return True
        
    except Exception as e:
        print(f"Error setting up vector index for {source}: {e}")
        return False

def load_vector_index(
    index_name=None,
    embedding=None,
    source_filter=None,
):
    """
    Load an existing vector index for Document nodes with error handling and fallbacks.

    Args:
        index_name (str): The name of the vector index to retrieve. If None, defaults to "documents".
                         For source-specific searches, use "documents" with source_filter.
        embedding (OpenAIEmbeddings): The embedding model to use.
        source_filter (str): Optional source filter for retrieval (e.g., 'uc1', 'uc2').
                           This filters results at query time, not index level.

    Returns:
        Neo4jVector: An instance of the Neo4jVector for the specified index.
    
    Raises:
        Exception: If the index cannot be loaded or created.
    """
    # Default to unified index name
    if index_name is None:
        index_name = "documents"
    
    print(f"Loading vector index: '{index_name}'" + (f" (filtering for: {source_filter})" if source_filter else " (no filtering)"))
    
    # Use default embedding if none is provided
    if embedding is None:
        embedding = embedding_model  # Use the global multilingual embedding model

    # Configure retrieval query for source filtering
    if source_filter:
        retrieval_query = f"""
        MATCH (n:Document)
        WHERE n.source = '{source_filter}'
        RETURN n.text AS text, score, n {{.*}} AS metadata
        """
    else:
        retrieval_query = None

    try:
        # Try to load the vector index directly
        neo4j_vector = Neo4jVector.from_existing_index(
            embedding=embedding,
            index_name=index_name,
            url=os.environ["NEO4J_URL"],
            username=os.environ["NEO4J_USERNAME"],
            password=os.environ["NEO4J_PASSWORD"],
            retrieval_query=retrieval_query,
        )
        
        return neo4j_vector

    except Exception as e:
        error_msg = str(e).lower()
        
        # Handle specific error cases
        if "does not exist" in error_msg or "not found" in error_msg:
            print(f"Index '{index_name}' not found. Attempting to create and populate...")
            
            try:
                # Try using from_existing_graph as a fallback
                neo4j_vector = Neo4jVector.from_existing_graph(
                    embedding=embedding,
                    index_name=index_name,
                    node_label="Document",
                    text_node_properties=["text"],
                    embedding_node_property="embedding",
                    retrieval_query=retrieval_query,
                )
                print(f"Successfully created vector index '{index_name}' using from_existing_graph")
                return neo4j_vector
                
            except Exception as e2:
                print(f"from_existing_graph also failed: {e2}")
                
                # Last resort: create index manually and try again
                try:
                    create_vector_index(index_name)
                    update_vector_index(index_name, source_filter)
                    
                    # Try loading again after creation
                    neo4j_vector = Neo4jVector.from_existing_index(
                        embedding=embedding,
                        index_name=index_name,
                        url=os.environ["NEO4J_URL"],
                        username=os.environ["NEO4J_USERNAME"],
                        password=os.environ["NEO4J_PASSWORD"],
                        retrieval_query=retrieval_query,
                    )
                    print(f"Successfully loaded vector index '{index_name}' after manual creation")
                    return neo4j_vector
                    
                except Exception as e3:
                    print(f"Manual creation also failed: {e3}")
                    raise Exception(f"Failed to load or create vector index '{index_name}': {e3}")
        else:
            print(f"Unexpected error loading vector index '{index_name}': {e}")
            raise e

def get_vector_index_info():
    """Get information about available vector indexes."""
    query = "SHOW VECTOR INDEX"
    result = graph.query(query)
    if result:
        return result
    else:
        return "No vector index found."