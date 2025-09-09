import os
from utils.pdf_processing import (
    load_and_chunk_pdf, 
    process_pdf_directory_individually,
    process_subdirectories_individually
)
from utils.graph_transformer import extract_and_store_graph
from utils.neo4j_connection import (
    clean_graph, 
    create_vector_index, 
    update_vector_index
)
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

    # Process documents and create graph
    process_documents_to_graph(documents)

def process_pdf_directory_individual(directory_path: str):
    """
    Process all PDF files in a directory individually.
    
    Args:
        directory_path (str): Path to the directory containing PDF files.
    """
    print(f"Processing all PDF files in directory: {directory_path}")
    documents, file_mapping = process_pdf_directory_individually(directory_path)
    
    print(f"File mapping: {file_mapping}")
    process_documents_to_graph(documents)

def process_subdirectories_with_separate_graphs(directory_path: str):
    """
    Process subdirectories individually, creating separate graphs for each subdirectory.
    
    Args:
        directory_path (str): Path to the directory containing subdirectories with PDF files.
    """
    print(f"Processing subdirectories in: {directory_path}")
    subdirectory_results = process_subdirectories_individually(directory_path)
    
    if not subdirectory_results:
        print("No subdirectories found or processed.")
        return
    
    # Collect processed sources
    processed_sources = []
    
    # Process each subdirectory's documents separately
    for subdir_name, result in subdirectory_results.items():
        if 'error' in result:
            print(f"Skipping {subdir_name} due to error: {result['error']}")
            continue
            
        documents = result['documents']
        file_mapping = result['file_mapping']
        
        if not documents:
            print(f"No documents found in subdirectory: {subdir_name}")
            continue
        
        print(f"\n{'='*50}")
        print(f"Creating graph for subdirectory: {subdir_name}")
        print(f"Documents to process: {len(documents)}")
        print(f"File mapping: {file_mapping}")
        print(f"{'='*50}")
        
        # Create a separate graph for this subdirectory
        process_documents_to_graph(documents, subdir_name)
        processed_sources.append(subdir_name)
    
    # Create source-specific vector indexes for all processed sources
    if processed_sources:
        print(f"\nCreating vector index...")
        print(f"Processed sources: {processed_sources}")
        
        # Create single unified vector index for all documents
        create_vector_index("documents")
        update_vector_index("documents")
        
        print("Unified vector index 'documents' created successfully!")
        print("\n Usage:")
        for source in processed_sources:
            print(f"   {source.upper()}: load_vector_index(index_name='documents', source_filter='{source}')")
        print(f"   ALL: load_vector_index(index_name='documents')")
    
    return processed_sources

def process_documents_to_graph(documents, graph_name: str = "default"):
    """
    Process document chunks and create a graph.
    
    Args:
        documents: List of document chunks to process.
        graph_name (str): Name identifier for the graph (useful for tracking different graphs).
    """
    print(f"Processing document chunks for graph: {graph_name}")
    
    # Track language detection for summary
    language_count = {}
    distinct_nodes = set()
    relations = []

    for doc in tqdm(documents, desc=f"Processing documents for {graph_name}"):
        # Pass the graph_name as source_label to distinguish nodes by source
        graph_document = extract_and_store_graph(doc, source_label=graph_name)

        if graph_document:
            # Track language from document metadata if available
            for node in graph_document.nodes:
                if hasattr(node, 'properties') and node.properties and 'original_language' in node.properties:
                    lang = node.properties['original_language']
                    language_count[lang] = language_count.get(lang, 0) + 1
                
                distinct_nodes.add(node.id)

            # Get all relations
            for relation in graph_document.relationships:
                relations.append(relation.type)

    # Show language summary
    if language_count:
        print(f"Graph '{graph_name}' - Languages detected: {dict(language_count)}")
    
    print(f"Graph '{graph_name}' - Distinct nodes: {len(distinct_nodes)}")
    print(f"Graph '{graph_name}' - Relations: {len(relations)}")
    print(f"Graph '{graph_name}' - Completed processing")
    
    return {
        'graph_name': graph_name,
        'node_count': len(distinct_nodes),
        'relation_count': len(relations),
        'distinct_nodes': distinct_nodes,
        'relations': relations
    }

if __name__ == "__main__":
    # Configuration - Choose your processing mode
    PROCESSING_MODE = "subdirectories"  # Options: "single", "individual", "subdirectories"
    
    # Path to the PDF file or directory
    PDF_PATH = "./manual"  # Main directory containing subdirectories (uc1, uc2, etc.)
    # PDF_PATH = "./manual/fabrik_2020_S7-1500_EN_korrigiert_final.pdf"  # Single file
    # PDF_PATH = "./manual/uc2"  # Directory with multiple PDFs

    # Clean the Neo4j graph database
    print("Cleaning the Neo4j graph database...")
    clean_graph()

    # Choose processing method
    processed_sources = []
    
    # Process based on the selected mode
    if PROCESSING_MODE == "single":
        # Process a single PDF file
        if os.path.isfile(PDF_PATH):
            process_pdf_file(PDF_PATH)
            processed_sources = ["default"]
        else:
            raise ValueError(f"Error: {PDF_PATH} is not a valid file for single processing mode")
            
    elif PROCESSING_MODE == "individual":
        # Process all PDFs in a directory individually (without subdirectories)
        if os.path.isdir(PDF_PATH):
            process_pdf_directory_individual(PDF_PATH)
            processed_sources = ["default"]
        else:
            raise ValueError(f"Error: {PDF_PATH} is not a valid directory for individual processing mode")
            
    elif PROCESSING_MODE == "subdirectories":
        # Process subdirectories individually, creating separate graphs for each
        if os.path.isdir(PDF_PATH):
            result = process_subdirectories_with_separate_graphs(PDF_PATH)
            if result:
                processed_sources = result
        else:
            raise ValueError(f"Error: {PDF_PATH} is not a valid directory for subdirectory processing mode")
            
    else:
        raise ValueError(f"Invalid processing mode: {PROCESSING_MODE}. Use 'single', 'individual', or 'subdirectories'.")
   
    # Create and update vector indexes based on processing results
    if not processed_sources:
        print("No sources processed successfully. Creating default vector index...")
        create_vector_index("documents")
        update_vector_index("documents")
    elif len(processed_sources) == 1 and processed_sources[0] == "default":
        print("Creating unified vector index for single source...")
        create_vector_index("documents")
        update_vector_index("documents")
    # Note: For subdirectories mode, unified vector index is already created in process_subdirectories_with_separate_graphs()
    
    print("\nProcessing completed! Vector index 'documents' is ready for use.")
    print("Search examples:")
    if processed_sources and len(processed_sources) > 1:
        for source in processed_sources:
            print(f"   {source.upper()}: load_vector_index(source_filter='{source}').similarity_search('query')")
    print("   ALL: load_vector_index().similarity_search('query')")