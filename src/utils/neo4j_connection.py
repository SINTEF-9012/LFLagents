# -*- coding: utf-8 -*-
"""Neo4j Connection Utilities.
This module provides utilities for connecting to a Neo4j database.
It includes functions to establish a connection and perform basic operations.
"""

import os
from dotenv import load_dotenv, find_dotenv
from langchain_community.graphs import Neo4jGraph
from langchain_community.vectorstores import Neo4jVector
from langchain_openai import OpenAIEmbeddings

import warnings
warnings.filterwarnings("ignore")

# Load environment variables from .env file
load_dotenv()
CHAT_MODEL = os.getenv("CHAT_MODEL")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL")

# Load environment variables
load_dotenv(find_dotenv(), override=True)

# Neo4j setup
graph = Neo4jGraph(
    url=os.environ["NEO4J_URL"],
    username=os.environ["NEO4J_USERNAME"],
    password=os.environ["NEO4J_PASSWORD"],
    enhanced_schema=True
)

def clean_graph():
    query = """
    MATCH (n)
    DETACH DELETE n
    """
    graph.query(query)
    
def create_vector_index():
    query = """
    CREATE VECTOR INDEX documents IF NOT EXISTS FOR (n:Document) ON (n.id)
    """
    graph.query(query)
    print("Vector index created (if it did not already exist) for Document nodes.")
    
def update_vector_index():
    """
    Update the Neo4j vector index for Document nodes.
    """
    print("Updating vector index for documents...")
    neo4j_vector = Neo4jVector.from_existing_graph(
        embedding=OpenAIEmbeddings(model=EMBEDDING_MODEL),
        index_name="documents",
        node_label="Document",
        text_node_properties=["text"],
        embedding_node_property="embedding",
    )
    print("Vector index updated successfully.")

def load_vector_index(
    index_name="documents",
    embedding=None,
):
    """
    Call the existing vector index for Document nodes.

    Args:
        index_name (str): The name of the vector index to retrieve. Defaults to "documents".
        embedding (OpenAIEmbeddings): The embedding model to use. Defaults to OpenAIEmbeddings with "text-embedding-3-small".

    Returns:
        Neo4jVector: An instance of the Neo4jVector for the specified index.
    """
    # Use default embedding if none is provided
    if embedding is None:
        embedding = OpenAIEmbeddings(model=EMBEDDING_MODEL)

    # Initialize the Neo4j vector index
    neo4j_vector = Neo4jVector.from_existing_index(
        embedding=embedding,
        index_name=index_name,
        url=os.environ["NEO4J_URL"],
        username=os.environ["NEO4J_USERNAME"],
        password=os.environ["NEO4J_PASSWORD"],
    )

    return neo4j_vector