# -*- coding: utf-8 -*-
"""Neo4j Connection Utilities.
This module provides utilities for connecting to a Neo4j database.
It includes functions to establish a connection and perform basic operations.
"""

import os
from dotenv import load_dotenv, find_dotenv
from langchain_community.graphs import Neo4jGraph
from langchain_community.vectorstores import Neo4jVector

import warnings
warnings.filterwarnings("ignore")

# Load environment variables from .env file
load_dotenv()
CHAT_MODEL = os.getenv("CHAT_MODEL")

# Load environment variables
load_dotenv(find_dotenv(), override=True)

# Embedding model for vector retrieval.
# EMBEDDING_PROVIDER=idun (default): IDUN-hosted embeddings via the
# OpenAI-compatible /v1/embeddings endpoint (e.g. Qwen/Qwen3-Embedding-8B).
# EMBEDDING_PROVIDER=local_hf: the original local HuggingFace model.
EMBEDDING_PROVIDER = os.getenv("EMBEDDING_PROVIDER", "idun").strip('"').lower()

if EMBEDDING_PROVIDER == "local_hf":
    from langchain_community.embeddings import HuggingFaceEmbeddings
    embedding_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        model_kwargs={'device': 'cpu'},  # Use 'cuda' if you have GPU
        encode_kwargs={'normalize_embeddings': True}
    )
else:
    from langchain_openai import OpenAIEmbeddings
    embedding_model = OpenAIEmbeddings(
        model=os.getenv("EMBEDDING_MODEL", "Qwen/Qwen3-Embedding-8B").strip('"'),
        openai_api_base=os.getenv("LLM_BASE_URL"),
        openai_api_key=os.getenv("IDUN_API_KEY"),
        check_embedding_ctx_length=False,  # non-OpenAI model: skip tiktoken batching
        chunk_size=16,  # small batches: one /v1/embeddings request per 16 chunks
    )

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

    The index dimension is probed from the configured embedding model so the
    index always matches the vectors that will be written into it.

    Args:
        index_name (str): Name of the vector index. Defaults to "documents".
    """
    dim = len(embedding_model.embed_query("vector index dimension probe"))
    query = f"""
    CREATE VECTOR INDEX {index_name} IF NOT EXISTS FOR (n:Document) ON (n.embedding)
    OPTIONS {{
      indexConfig: {{
        `vector.dimensions`: {dim},
        `vector.similarity_function`: 'cosine'
      }}
    }}
    """
    graph.query(query)
    print(f"Vector index '{index_name}' created (dim {dim}, cosine similarity) for Document nodes.")

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
        RETURN n.text AS text, n {{.*}} AS metadata
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
    index_name="documents",
    embedding=embedding_model,
):
    """
    Call the existing vector index for Document nodes.

    Args:
        index_name (str): The name of the vector index to retrieve. Defaults to "documents".
        embedding (HuggingFaceEmbeddings): The embedding model to use. Defaults to free HuggingFace multilingual embeddings.

    Returns:
        Neo4jVector: An instance of the Neo4jVector for the specified index.
    """
    # Use default embedding if none is provided
    if embedding is None:
        embedding = embedding_model

    # Initialize the Neo4j vector index
    neo4j_vector = Neo4jVector.from_existing_index(
        embedding=embedding,
        index_name=index_name,
        url=os.environ["NEO4J_URL"],
        username=os.environ["NEO4J_USERNAME"],
        password=os.environ["NEO4J_PASSWORD"],
    )

    return neo4j_vector

def get_vector_index_info():
    """Get information about available vector indexes."""
    query = "SHOW VECTOR INDEX"
    result = graph.query(query)
    if result:
        return result
    else:
        return "No vector index found."