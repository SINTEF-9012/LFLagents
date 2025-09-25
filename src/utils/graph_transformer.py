# -*- coding: utf-8 -*-
"""Graph Transformer Utilities.
This module provides utilities for transforming and extracting knowledge graphs
from documents. It includes functions to create extraction chains,
map nodes and relationships to base types, and store the extracted graphs
in a Neo4j database.
"""

import os
from dotenv import load_dotenv
from langchain.chains.openai_functions import create_structured_output_chain
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain_community.graphs.graph_document import GraphDocument

from utils.graph_models import KnowledgeGraph
from utils.graph_mapping import map_to_base_node, map_to_base_relationship
from utils.neo4j_connection import graph
from langchain.schema import Document
from typing import Optional, List
from pydantic import ValidationError
from langdetect import detect

# Load environment variables from .env file
load_dotenv()
CHAT_MODEL = os.getenv("CHAT_MODEL")

from utils.llm_config import get_llm_instance

llm = get_llm_instance(model_name=CHAT_MODEL, temperature=0.0)

def detect_language(text: str) -> str:
    """
    Detect the language of the input text.
    
    Args:
        text (str): Input text to analyze
        
    Returns:
        str: Detected language code (e.g., 'en', 'es', 'de')
    """
    try:
        # Take a sample of text for detection (first 500 characters should be enough)
        sample_text = text[:500].strip()
        if len(sample_text) < 10:
            return 'en'  # Default to English for very short texts
        
        detected = detect(sample_text)
        return detected
    except Exception:
        # Default to English if detection fails
        return 'en'

def get_language_name(lang_code: str) -> str:
    """Convert language code to language name."""
    language_map = {
        'en': 'English',
        'es': 'Spanish', 
        'de': 'German',
        'fr': 'French',
        'it': 'Italian',
        'pt': 'Portuguese'
    }
    return language_map.get(lang_code, 'English')

def get_extraction_chain(
    allowed_nodes: Optional[List[str]] = None,
    allowed_rels: Optional[List[str]] = None,
    source_language: Optional[str] = None
    ):
    
    # Language-specific instructions - always extract in English for consistency
    language_instruction = ""
    if source_language and source_language.lower() != "english":
        language_instruction = f"""
                ## Language Processing Instructions
                - **Source Language**: The input text is in {source_language}
                - **Extract in English**: Always extract node names and relationships in English for consistency
                - **Translation**: Translate {source_language} terms to their English equivalents
                - **Node IDs**: Use clear English terms (e.g., "Welding_Machine", "Process_Control", "Safety_System")
                - **Relationships**: All relationships must be in English (e.g., "CONTROLS", "PART_OF", "CONNECTS_TO")
                - **Properties**: Keep original {source_language} text in properties for reference
                - **Example Translations**:
                  - Spanish "Máquina de Soldadura" → English "Welding_Machine"
                  - Spanish "Sistema de Control" → English "Control_System"
                  - German "Steuerungssystem" → English "Control_System"
                """
    else:
        language_instruction = """
                ## Language Processing Instructions
                - **Source Language**: The input text is in English
                - **Node Names**: Use clear, descriptive English terms
                - **Relationships**: Extract relationships in English
                - **Consistency**: Maintain consistent English naming across all extractions
                """
    
    # Build the system prompt
    system_prompt = """# Knowledge Graph Instructions for GPT-4
                ## 1. Overview
                You are a top-tier algorithm designed for extracting information in structured formats to build a knowledge graph.
                - **Nodes** represent entities and concepts. They're akin to Wikipedia nodes.
                - **IMPORTANT**: ALL node names and relationship types must be in ENGLISH regardless of source language.
                - The aim is to achieve simplicity and clarity in the knowledge graph, making it accessible for a vast audience.
                """ + language_instruction + """
                ## 2. Labeling Nodes
                - **Consistency**: Ensure you use basic or elementary types for node labels.
                - **English Only**: ALL node labels must be in English (e.g., "Machine", "Process", "Component", "System").
                - **Translation Required**: If source text is in another language, translate to English equivalents.
                - For example, when you identify an entity representing a person, always label it as **"Person"**. Avoid using more specific terms like "mathematician" or "scientist".
                - **Node IDs**: Never utilize integers as node IDs. Node IDs should be English names or human-readable identifiers.
                - **Standard English Terms**: Use standard English technical terminology (e.g., "Welding_Machine", "Control_System", "Safety_Protocol").
                """
    
    if allowed_nodes:
        system_prompt += f"- **Allowed Node Labels:** {', '.join(allowed_nodes)}\n"
    if allowed_rels:
        system_prompt += f"- **Allowed Relationship Types:** {', '.join(allowed_rels)}\n"
        
    system_prompt += """
                ## 3. Handling Numerical Data and Dates
                - Numerical data, like age or other related information, should be incorporated as attributes or properties of the respective nodes.
                - **No Separate Nodes for Dates/Numbers**: Do not create separate nodes for dates or numerical values. Always attach them as attributes or properties of nodes.
                - **Property Format**: Properties must be in a key-value format.
                - **Quotation Marks**: Never use escaped single or double quotes within property values.
                - **Naming Convention**: Use camelCase for property keys, e.g., `birthDate`.
                ## 4. Coreference Resolution
                - **Maintain Entity Consistency**: When extracting entities, it's vital to ensure consistency.
                If an entity, such as "John Doe", is mentioned multiple times in the text but is referred to by different names or pronouns (e.g., "Joe", "he"), 
                always use the most complete identifier for that entity throughout the knowledge graph. In this example, use "John Doe" as the entity ID.  
                Remember, the knowledge graph should be coherent and easily understandable, so maintaining consistency in entity references is crucial. 
                ## 5. Translation Examples
                - Spanish "Máquina" → English "Machine"
                - Spanish "Sistema de Control" → English "Control_System"  
                - Spanish "Proceso" → English "Process"
                - German "Maschine" → English "Machine"
                - German "Steuerung" → English "Control"
                ## 6. Output Format Requirements
                CRITICAL: Your response MUST include both 'nodes' and 'rels' fields:
                - 'nodes': Array of node objects (can be empty array [])
                - 'rels': Array of relationship objects (can be empty array [])
                - Node properties MUST be simple key-value pairs, not nested objects
                - Relationship properties MUST be simple key-value pairs, not nested objects
                ## 7. Strict Compliance
                Adhere to the rules strictly. Non-compliance will result in termination."""
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "Use the given format to extract information from the following input: {input}"),
        ("human", "Tip: Make sure to answer in the correct format"),
    ])
    return create_structured_output_chain(KnowledgeGraph, llm, prompt, verbose=False)

def extract_and_store_graph(
    document: Document,
    nodes:Optional[List[str]] = None,
    rels:Optional[List[str]] = None,
    source_label: Optional[str] = None) -> None:

    # Detect language automatically using langdetect
    lang_code = detect_language(document.page_content)
    detected_language = get_language_name(lang_code)
    
    # Only print language detection for the first document or when language changes
    # This reduces verbose logging during bulk processing

    # Extract graph data using OpenAI functions with language awareness
    extract_chain = get_extraction_chain(nodes, rels, detected_language)
    
    max_retries = 3
    for attempt in range(max_retries):
        try:
            # Pass the content with the correct key name 'input'
            data = extract_chain.invoke({"input": document.page_content})['function']
            
            # Ensure rels field exists and is a list
            if not hasattr(data, 'rels') or data.rels is None:
                data.rels = []
            
            # Ensure nodes field exists and is a list  
            if not hasattr(data, 'nodes') or data.nodes is None:
                data.nodes = []
            
            # If we get here, the extraction was successful
            break
                
        except ValidationError as e:
            if attempt < max_retries - 1:
                print(f"ValidationError on attempt {attempt + 1}: {e}. Retrying...")
                continue
            else:
                print(f"ValidationError after {max_retries} attempts: {e}. Returning empty graph...")
                return GraphDocument(
                    nodes=[],
                    relationships=[],
                    source=document
                )
        except Exception as e:
            print(f"Unexpected error during extraction: {e}. Returning empty graph...")
            return GraphDocument(
                nodes=[],
                relationships=[],
                source=document
            )

    # Construct a graph document
    nodes_with_source = []
    for node in data.nodes:
        base_node = map_to_base_node(node)
        
        # Add source label if provided
        if source_label:
            # Change from "name:uc1" to "uc1:name" format
            if hasattr(base_node, 'type') and base_node.type:
                # Add the source as a prefix to the original type
                original_type = base_node.type
                base_node.type = f"{source_label}:{original_type}"
            else:
                # If no existing type, use source label as type
                base_node.type = source_label
            
            # Add source information to properties
            if not hasattr(base_node, 'properties') or base_node.properties is None:
                base_node.properties = {}
                
            # Ensure properties is a dict (not a list) for Neo4j compatibility
            if isinstance(base_node.properties, list):
                # Convert list of properties to dict
                props_dict = {}
                for prop in base_node.properties:
                    if isinstance(prop, dict) and 'key' in prop and 'value' in prop:
                        props_dict[prop['key']] = prop['value']
                base_node.properties = props_dict
            
            base_node.properties['source'] = source_label
            base_node.properties['original_language'] = detected_language
            base_node.properties['language_code'] = lang_code
            
        nodes_with_source.append(base_node)
    
    # Handle relationships and fix their properties
    relationships_with_fixed_props = []
    for rel in data.rels:
        base_rel = map_to_base_relationship(rel)
        
        # Ensure relationship properties is a dict (not a list) for Neo4j compatibility
        if hasattr(base_rel, 'properties') and isinstance(base_rel.properties, list):
            # Convert list of properties to dict
            props_dict = {}
            for prop in base_rel.properties:
                if isinstance(prop, dict) and 'key' in prop and 'value' in prop:
                    props_dict[prop['key']] = prop['value']
            base_rel.properties = props_dict
        elif not hasattr(base_rel, 'properties') or base_rel.properties is None:
            base_rel.properties = {}
        
        # Add consistent source information to relationships
        if source_label:
            base_rel.properties['source'] = source_label
            base_rel.properties['original_language'] = detected_language
            base_rel.properties['language_code'] = lang_code
            
        relationships_with_fixed_props.append(base_rel)
    
    graph_document = GraphDocument(
      nodes = nodes_with_source,
      relationships = relationships_with_fixed_props,
      source = document
    )
    
    # Store information in a graph
    # print(graph_document)
    graph.add_graph_documents([graph_document], include_source=True)

    return graph_document
