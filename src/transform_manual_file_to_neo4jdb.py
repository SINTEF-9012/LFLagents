import os
from utils.pdf_processing import load_and_chunk_pdf
from utils.graph_transformer import extract_and_store_graph
from utils.neo4j_connection import clean_graph, create_vector_index, update_vector_index
from tqdm import tqdm

def process_pdf_file(pdf_path: str):
    """
    Process a single PDF file, extract its content, and create a graph.
    
    Args:
        pdf_path (str): Path to the PDF file.
    """
    # Check if the file exists
    if not os.path.isfile(pdf_path):
        raise ValueError(f"File path {pdf_path} is not a valid file.")

    # Load and chunk the PDF file
    print(f"Loading and chunking PDF file: {pdf_path}")
    documents = load_and_chunk_pdf(pdf_path)

    # Process each document and create a graph
    print("Processing document chunks...")
    distinct_nodes = set()
    relations = []

    for i, doc in tqdm(enumerate(documents), total=len(documents)):
        graph_document = extract_and_store_graph(doc)

        if graph_document:
            # Ensure properties are always a list
            for node in graph_document.nodes:
                distinct_nodes.add(node.id)

            # Get all relations
            for relation in graph_document.relationships:
                relations.append(relation.type)

    print(f"Distinct nodes: {len(distinct_nodes)}")
    print(f"Relations: {len(relations)}")

if __name__ == "__main__":
    # Path to the PDF file
    PDF_PATH = "./manual/fabrik_2020_S7-1500_EN_korrigiert_final.pdf"

    # Clean the Neo4j graph database
    print("Cleaning the Neo4j graph database...")
    clean_graph()

    # Process the PDF file
    process_pdf_file(PDF_PATH)
   
     # Create vector index for documents
    print("Creating vector index for documents...")
    create_vector_index()
   
    # Update vector index for documents
    update_vector_index()