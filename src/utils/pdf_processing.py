# -*- coding: utf-8 -*-
"""PDF Processing Utilities.
This module provides utilities for loading and processing PDF files.
It includes functions to load PDF files, split them into pages,
and chunk the text into smaller segments for further processing.
"""

from langchain.text_splitter import TokenTextSplitter
from langchain_community.document_loaders import PyPDFLoader

def load_and_chunk_pdf(pdf_path: str, chunk_size: int = 2048, chunk_overlap: int = 100):
    """Load and chunk a PDF file."""
    loader = PyPDFLoader(pdf_path)
    pages = loader.load_and_split()
    text_splitter = TokenTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return text_splitter.split_documents(pages)