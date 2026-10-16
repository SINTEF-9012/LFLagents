# -*- coding: utf-8 -*-
"""PDF Processing Utilities.
This mod    # Use semantic chunking with RecursiveCharacterTextSplitter
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size * 4,  # Convert tokens to approximate characters
        chunk_overlap=chunk_overlap * 4,
        length_function=len,
        separators=[
            "\n\n",    # Paragraph breaks
            "\n",      # Line breaks
            ". ",      # Sentence endings
            "! ",      # Exclamation sentences
            "? ",      # Question sentences
            "; ",      # Semicolons
            ", ",      # Commas
            " ",       # Spaces
            "",        # Characters
        ]
    )
    return text_splitter.split_documents(pages)ties for loading and processing PDF files.
It includes functions to load PDF files, split them into pages,
and chunk the text into smaller segments for further processing.
"""

import os
import glob
from typing import List, Optional, Tuple
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain.schema import Document

def is_blank_page(text: str) -> bool:
    """
    Detect if a page is intentionally blank or contains minimal content.
    
    Args:
        text (str): Page content to analyze
        
    Returns:
        bool: True if page should be considered blank/skipped
    """
    # Clean the text for analysis
    clean_text = text.strip().lower()
    
    # Very short content
    if len(clean_text) < 20:
        return True
    
    # Common blank page indicators in multiple languages
    blank_indicators = [
        # Spanish
        "página intencionadamente en blanco",
        "página en blanco",
        "esta página está en blanco", 
        "hoja en blanco",
        # English
        "this page intentionally left blank",
        "intentionally blank",
        "blank page",
        "this page is blank",
        # German
        "diese seite wurde absichtlich leer gelassen",
        "leerseite",
        # French
        "page intentionnellement laissée vierge",
        "page blanche",
        # General patterns
        "intentionally",
        "absichtlich",
        "intencionadamente"
    ]
    
    # Check if text contains blank page indicators
    for indicator in blank_indicators:
        if indicator in clean_text:
            return True
    
    # Enhanced logic for pages with headers/footers but minimal content
    lines = [line.strip() for line in clean_text.split('\n') if line.strip()]
    
    # Remove common header/footer patterns
    content_lines = []
    for line in lines:
        # Skip page numbers
        if line.isdigit() or (len(line) <= 3 and any(c.isdigit() for c in line)):
            continue
        # Skip section headers that are just numbers/letters followed by dot
        if len(line) <= 20 and ('.' in line or line.isupper()) and len(line.split()) <= 3:
            continue
        # Skip very short lines (likely headers/footers)
        if len(line) <= 5:
            continue
        content_lines.append(line)
    
    # If after filtering headers/footers we have very little content, consider it blank
    if len(content_lines) <= 1:
        return True
        
    # Check if remaining content is minimal
    total_content = ' '.join(content_lines)
    if len(total_content) < 30:
        return True
    
    # Check for very repetitive content (like page numbers only)
    words = clean_text.split()
    if len(words) <= 3 and all(word.isdigit() or len(word) <= 2 for word in words):
        return True
    
    return False

def load_and_chunk_pdf(pdf_path: str, chunk_size: int = 2048, chunk_overlap: int = 100):
    """
    Load and chunk a PDF file using semantic-aware chunking.
    
    Args:
        pdf_path (str): Path to the PDF file
        chunk_size (int): Size of each chunk in tokens (default: 2048)
        chunk_overlap (int): Overlap between chunks in tokens (default: 100)
    
    Returns:
        List[Document]: List of chunked documents with semantic boundaries
    """
    # Load PDF pages
    loader = PyPDFLoader(pdf_path)
    pages = loader.load()
    
    # Filter out blank pages early
    filtered_pages = []
    skipped_count = 0
    
    for page in pages:
        if is_blank_page(page.page_content):
            skipped_count += 1
            print(f"⏭️ Skipping blank page (content: {page.page_content[:50]}...)")
        else:
            # Add source information to metadata for valid pages
            page.metadata['source'] = pdf_path
            filtered_pages.append(page)
    
    if skipped_count > 0:
        print(f"🗑️ Filtered out {skipped_count} blank pages from {pdf_path}")
    
    if not filtered_pages:
        print(f"❌ No valid content found after filtering in {pdf_path}")
        return []
    
    # Use semantic chunking with RecursiveCharacterTextSplitter
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size * 4,  # Convert tokens to approximate characters
        chunk_overlap=chunk_overlap * 4,
        length_function=len,
        separators=[
            "\n\n",    # Paragraph breaks
            "\n",      # Line breaks
            ". ",      # Sentence endings
            "! ",      # Exclamation sentences
            "? ",      # Question sentences
            "; ",      # Semicolons
            ", ",      # Commas
            " ",       # Spaces
            "",        # Characters
        ]
    )
    return text_splitter.split_documents(filtered_pages)

def get_pdf_files_from_directory(directory_path: str) -> List[str]:
    """
    Get all PDF files from a directory and its subdirectories.
    
    Args:
        directory_path (str): Path to the directory to search for PDF files.
        
    Returns:
        List[str]: List of paths to PDF files found in the directory.
    """
    if not os.path.isdir(directory_path):
        raise ValueError(f"Directory path {directory_path} is not a valid directory.")
    
    # Use glob to find all PDF files recursively
    pdf_pattern = os.path.join(directory_path, "**", "*.pdf")
    pdf_files = glob.glob(pdf_pattern, recursive=True)
    
    # Sort the files for consistent processing order
    pdf_files.sort()
    
    print(f"Found {len(pdf_files)} PDF files in {directory_path}")
    for pdf_file in pdf_files:
        print(f"  - {pdf_file}")
    
    return pdf_files

def process_pdf_directory_individually(directory_path: str, chunk_size: int = 2048, chunk_overlap: int = 100) -> Tuple[List[Document], dict]:
    """
    Process all PDF files in a directory individually, keeping track of source files.
    
    Args:
        directory_path (str): Path to the directory containing PDF files.
        chunk_size (int): Size of text chunks for splitting.
        chunk_overlap (int): Overlap between chunks.
        
    Returns:
        Tuple[List[Document], dict]: 
            - List of all document chunks from all PDF files
            - Dictionary mapping document indices to source file paths
    """
    pdf_files = get_pdf_files_from_directory(directory_path)
    
    if not pdf_files:
        print(f"No PDF files found in directory: {directory_path}")
        return [], {}
    
    all_documents = []
    file_mapping = {}
    
    for pdf_file in pdf_files:
        print(f"Processing PDF file: {pdf_file}")
        try:
            # Load and chunk the current PDF file
            documents = load_and_chunk_pdf(pdf_file, chunk_size, chunk_overlap)
            
            # Add source file information to each document's metadata
            for doc in documents:
                if doc.metadata is None:
                    doc.metadata = {}
                doc.metadata['source_file'] = pdf_file
                doc.metadata['file_name'] = os.path.basename(pdf_file)
            
            # Track which documents came from which file
            start_idx = len(all_documents)
            all_documents.extend(documents)
            end_idx = len(all_documents)
            
            file_mapping[pdf_file] = {
                'start_index': start_idx,
                'end_index': end_idx,
                'document_count': len(documents)
            }
            
            print(f"  - Loaded {len(documents)} document chunks from {pdf_file}")
            
        except (FileNotFoundError, PermissionError, ValueError) as e:
            print(f"Error processing {pdf_file}: {str(e)}")
            continue
        except Exception as e:
            print(f"Unexpected error processing {pdf_file}: {str(e)}")
            continue
    
    print(f"Total documents loaded: {len(all_documents)}")
    return all_documents, file_mapping

def process_all_pdfs_in_directory_batch(directory_path: str, chunk_size: int = 2048, chunk_overlap: int = 100) -> List[Document]:
    """
    Process all PDF files in a directory as a single batch.
    
    Args:
        directory_path (str): Path to the directory containing PDF files.
        chunk_size (int): Size of text chunks for splitting.
        chunk_overlap (int): Overlap between chunks.
        
    Returns:
        List[Document]: List of all document chunks from all PDF files combined.
    """
    pdf_files = get_pdf_files_from_directory(directory_path)
    
    if not pdf_files:
        print(f"No PDF files found in directory: {directory_path}")
        return []
    
    all_documents = []
    
    for pdf_file in pdf_files:
        print(f"Processing PDF file: {pdf_file}")
        try:
            # Load and chunk the current PDF file
            documents = load_and_chunk_pdf(pdf_file, chunk_size, chunk_overlap)
            
            # Add source file information to each document's metadata
            for doc in documents:
                if doc.metadata is None:
                    doc.metadata = {}
                doc.metadata['source_file'] = pdf_file
                doc.metadata['file_name'] = os.path.basename(pdf_file)
                doc.metadata['batch_processing'] = True
            
            all_documents.extend(documents)
            print(f"  - Loaded {len(documents)} document chunks from {pdf_file}")
            
        except (FileNotFoundError, PermissionError, ValueError) as e:
            print(f"Error processing {pdf_file}: {str(e)}")
            continue
        except Exception as e:
            print(f"Unexpected error processing {pdf_file}: {str(e)}")
            continue
    
    print(f"Total documents loaded in batch: {len(all_documents)}")
    return all_documents

def get_subdirectories(directory_path: str, include: Optional[List[str]] = None) -> List[str]:
    """
    Get all subdirectories from a directory, optionally filtered by name.

    Args:
        directory_path (str): Path to the directory to search for subdirectories.
        include (List[str], optional): Subdirectory names to keep (case-insensitive).
            If None, all subdirectories are returned.

    Returns:
        List[str]: List of paths to subdirectories found in the directory.
    """
    if not os.path.isdir(directory_path):
        raise ValueError(f"Directory path {directory_path} is not a valid directory.")

    # Normalize the filter names for case-insensitive matching
    include_lower = [name.lower() for name in include] if include else None

    subdirectories = []
    for item in os.listdir(directory_path):
        item_path = os.path.join(directory_path, item)
        if os.path.isdir(item_path):
            if include_lower is not None and item.lower() not in include_lower:
                continue
            subdirectories.append(item_path)

    # Fail early if the filter matched nothing (e.g. a typo like "uc3")
    if include_lower is not None and not subdirectories:
        available = [d for d in os.listdir(directory_path) if os.path.isdir(os.path.join(directory_path, d))]
        raise ValueError(
            f"No subdirectories matching {include} in {directory_path}. Available: {available}"
        )
    
    # Sort the subdirectories for consistent processing order
    subdirectories.sort()
    
    print(f"Found {len(subdirectories)} subdirectories in {directory_path}")
    for subdir in subdirectories:
        print(f"  - {subdir}")
    
    return subdirectories

def process_subdirectories_individually(directory_path: str, chunk_size: int = 2048, chunk_overlap: int = 100, include: Optional[List[str]] = None) -> dict:
    """
    Process each subdirectory individually, treating each as a separate document collection.

    Args:
        directory_path (str): Path to the directory containing subdirectories with PDF files.
        chunk_size (int): Size of text chunks for splitting.
        chunk_overlap (int): Overlap between chunks.
        include (List[str], optional): Subdirectory names to process (case-insensitive).
            If None, all subdirectories are processed.

    Returns:
        dict: Dictionary mapping subdirectory names to their processed documents and file mappings.
    """
    subdirectories = get_subdirectories(directory_path, include=include)
    
    if not subdirectories:
        print(f"No subdirectories found in directory: {directory_path}")
        return {}
    
    results = {}
    
    for subdir in subdirectories:
        subdir_name = os.path.basename(subdir)
        print(f"\n{'='*50}")
        print(f"Processing subdirectory: {subdir_name}")
        print(f"{'='*50}")
        
        try:
            # Process all PDFs in this subdirectory
            documents, file_mapping = process_pdf_directory_individually(subdir, chunk_size, chunk_overlap)
            
            # Add subdirectory information to each document's metadata
            for doc in documents:
                if doc.metadata is None:
                    doc.metadata = {}
                doc.metadata['subdirectory'] = subdir_name
                doc.metadata['subdirectory_path'] = subdir
                # Add consistent source identifier for filtering
                doc.metadata['source'] = subdir_name
            
            results[subdir_name] = {
                'documents': documents,
                'file_mapping': file_mapping,
                'subdirectory_path': subdir,
                'document_count': len(documents)
            }
            
            print(f"Completed processing {subdir_name}: {len(documents)} documents")
            
        except Exception as e:
            print(f"Error processing subdirectory {subdir_name}: {str(e)}")
            results[subdir_name] = {
                'documents': [],
                'file_mapping': {},
                'subdirectory_path': subdir,
                'document_count': 0,
                'error': str(e)
            }
            continue
    
    total_documents = sum(result['document_count'] for result in results.values())
    print(f"\n{'='*50}")
    print(f"Total documents processed across all subdirectories: {total_documents}")
    print(f"{'='*50}")
    
    return results