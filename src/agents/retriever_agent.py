"""
Retriever Agent for Manufacturing Knowledge Graph Querying.

This module provides retrieval capabilities for manufacturing knowledge graphs,
including semantic search, graph database querying, and knowledge extraction
from structured manufacturing data.
"""

import os
import json
from datetime import datetime

from utils.neo4j_connection import graph, load_vector_index
from utils.llm_config import get_llm_instance, get_temperature, strip_reasoning  # Use flexible LLM config

from langchain_community.chains.graph_qa.cypher import GraphCypherQAChain

from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import ConversationChain, LLMChain
from langchain.chains.conversation.memory import ConversationBufferWindowMemory

from langchain.pydantic_v1 import BaseModel
from langchain.tools import BaseTool
from typing import Type, Dict, Any, List, Optional

import streamlit as st
from langchain.prompts import (
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
    PromptTemplate
)

# Load environment variables
# CHAT_MODEL is read at LLM-construction time (os.getenv) so callers that
# switch models via the environment after import are respected.
TEMPERATURE = get_temperature()

# Load the graph db
neo4j_graph = graph

class Question(BaseModel):
    question: str
    source_filter: str = None

class Neo4jVectorTool(BaseTool):
    name: str = "neo4j_vector_tool"
    description: str = "Tool for performing similarity-based retrieval from the Neo4j vector index."
    args_schema: Type[Question] = Question

    def _run(self, question: str, source_filter: str = None):
        # Initialize Neo4j vector index
        neo4j_vector = load_vector_index()

        # Perform similarity search
        results = neo4j_vector.similarity_search(question,
                                                 k=5,
                                                 filter={"source": source_filter} if source_filter else None)
        
        all_pages = "\n".join([result.page_content for result in results])

        return all_pages

class CypherGenerationChain(LLMChain):
    """LLMChain that strips leaked reasoning text from the generated Cypher
    before the QA chain parses it"""

    def run(self, *args, **kwargs):
        return strip_reasoning(super().run(*args, **kwargs))[0]


class Neo4jCypherTool(BaseTool):
    name: str = "neo4j_cypher_tool"
    description: str = "Tool for retrieving structured, interconnected, and relational knowledge from the Neo4j graph."
    args_schema: Type[Question] = Question

    def _run(self, question: str, source_filter: str = None):
        """
        Execute a Cypher query based on the user's question.

        Args:
            question (str): The user's natural language question.
            source_filter (str): Optional source filter (e.g., 'uc1', 'uc2').

        Returns:
            dict: The results of the Cypher query executed on the Neo4j graph.
        """

        # # Initialize Cypher validation tool
        # corrector_schema = [
        #     Schema(el["start"], el["type"], el["end"])
        #     for el in neo4j_graph.structured_schema.get("relationships")
        # ]
        # cypher_validation = CypherQueryCorrector(corrector_schema)

        # LLM for Cypher query generation
        CYPHER_LLM = get_llm_instance(model_name=os.getenv('CHAT_MODEL'), temperature=TEMPERATURE)

        # Generate Cypher query with source filtering
        source_instruction = ""
        if source_filter:
            source_instruction = f"- Filter results to only include nodes with source = '{source_filter}'"

        CYPHER_GENERATION_TEMPLATE = f"""
        Task: Generate Cypher statement to query a graph database.
        Instructions:
        - Use only the provided relationship types and properties in the schema
        - Do not use any other relationship types or properties that are not provided
        - Use case-insensitive matching with `toLower()` function for text comparisons
        - For configuration/detail queries, return full context including neighboring nodes
        - Use patterns like (n)-[r]-(m) to include related nodes and relationships
        - Limit results appropriately to avoid overwhelming responses
        - Use OPTIONAL MATCH for relationships that may not exist
        - Always include node labels and key properties in results
        {source_instruction}

        Schema:
        {{schema}}

        Question: {{question}}

        Generate only the Cypher statement without explanations.
        """

        CYPHER_GENERATION_PROMPT = PromptTemplate(
            input_variables=["schema", "question"], template=CYPHER_GENERATION_TEMPLATE
        )

        # Create the chain
        chain = GraphCypherQAChain.from_llm(
            llm=CYPHER_LLM,
            graph=neo4j_graph,
            prompt=CYPHER_GENERATION_PROMPT,
            validate_cypher=True,  # Enable Cypher validation
            verbose=True,
            top_k=5,
            allow_dangerous_requests=True,
        )
        # Strip leaked reasoning from the generated Cypher before the QA
        # chain parses it
        chain.cypher_generation_chain = CypherGenerationChain(
            llm=CYPHER_LLM, prompt=CYPHER_GENERATION_PROMPT
        )

        return strip_reasoning(chain.run(question))[0]

class RetrieverAgent:

    def __init__(self):

        # Initialize tools
        self.cypher_tool = Neo4jCypherTool()
        self.vector_tool = Neo4jVectorTool()

        # Initialize chat model with flexible LLM config
        self.llm = get_llm_instance(model_name=os.getenv('CHAT_MODEL'), temperature=TEMPERATURE)
        # Initialize session state variables
        self.initialize_session_state()

        # Initialize conversation chain
        self.initialize_conversation_chain()


    def initialize_session_state(self):
        if 'responses' not in st.session_state:
            st.session_state['responses'] = ["How can I help you today?"]

        if 'requests' not in st.session_state:
            st.session_state['requests'] = []

        if 'messages' not in st.session_state:
            st.session_state['messages'] = []

        if 'buffer_memory' not in st.session_state:
            st.session_state['buffer_memory'] = ConversationBufferWindowMemory(k=1000000000, return_messages=True)

        if 'chat_history' not in st.session_state:
            st.session_state['chat_history'] = []


    def initialize_conversation_chain(self, source_filter: str = None):
        # Create context-aware system message based on source filter
        if source_filter == "uc1":
            system_template = """
            You are an expert assistant specialized in the fischertechnik Training Factory Industry 4.0 system.
            Your goal is to answer questions by integrating knowledge from both structured data (Cypher results) and unstructured content (Vector results) relevant to the Training Factory.

            Based on the question, Cypher results, and Vector results, write a clear and informative natural language response that is technically accurate and aligned with the Training Factory system.
            You may cover aspects such as:
            - Factory components (e.g., high-bay warehouse, suction gripper, sorting line)
            - SIMATIC S7-1500 PLC configuration and TIA Portal usage
            - TXT controller and MQTT/OPC UA communication
            - IoT Gateway and Node-RED dashboards
            - Cloud integration and dashboard usage
            - Programming tasks and educational simulations
            - System setup, networking, and environmental monitoring

            Be concise, avoid speculation, and if information is missing, clearly indicate what is unknown.
            """
        elif source_filter == "uc2":
            system_template = """
            You are an expert assistant specialized in industrial milling machines and manufacturing equipment.
            Your goal is to answer questions by integrating knowledge from both structured data (Cypher results) and unstructured content (Vector results) relevant to industrial milling machines.

            The documentation may be in Spanish, but you should respond in English unless specifically asked otherwise.
            
            Based on the question, Cypher results, and Vector results, write a clear and informative natural language response that is technically accurate.
            You may cover aspects such as:
            - Machine operation procedures and specifications
            - Safety protocols and maintenance instructions
            - Technical documentation for machines like 7152, 11007, and PM-10005
            - Operating parameters and configurations
            - Troubleshooting and repair procedures
            - Machine programming and setup instructions

            Be concise, avoid speculation, and if information is missing, clearly indicate what is unknown.
            """
        else:
            # Default template for all sources or when no filter is specified
            system_template = """
            You are an expert assistant trained on manufacturing systems and industrial documentation.
            Your goal is to answer questions by integrating knowledge from both structured data (Cypher results) and unstructured content (Vector results).

            The documentation covers two main areas:
            1. Fischertechnik Training Factory Industry 4.0 system (UC1)
            2. Industrial milling machines and manufacturing equipment (UC2)

            Based on the question, Cypher results, and Vector results, write a clear and informative natural language response that is technically accurate.
            You may cover aspects such as:
            - Factory automation components and PLC systems
            - Industrial machine operation and maintenance
            - Communication protocols (MQTT/OPC UA)
            - Manufacturing processes and procedures
            - Safety protocols and technical specifications
            - System setup, programming, and troubleshooting

            Be concise, avoid speculation, and if information is missing, clearly indicate what is unknown.
            """

        self.system_msg_template = SystemMessagePromptTemplate.from_template(template=system_template)
        self.human_msg_template = HumanMessagePromptTemplate.from_template(template="{input}")
        self.prompt_template = ChatPromptTemplate.from_messages(
            [self.system_msg_template, MessagesPlaceholder(variable_name="history"), self.human_msg_template]
        )
        self.conversation = ConversationChain(
            memory=st.session_state.buffer_memory,
            prompt=self.prompt_template,
            llm=self.llm,
            verbose=True
        )

    # def save_chat_history(self,chat_history, folder_name, file_name):
    #     folder_path = os.path.join(os.getcwd(), folder_name)
    #     if not os.path.exists(folder_path):
    #         os.makedirs(folder_path)
    #     file_path = os.path.join(folder_path, file_name)
    #     with open(file_path, "w", encoding="utf-8") as file:
    #         for query, response in chat_history:
    #             file.write(f"User: {query}\nBot: {response}\n\n")

    def get_documentation_context(self, query: str, source_filter: str = None) -> Dict[str, Any]:
        """
        Get relevant documentation context using retriever tools.

        Args:
            query: User's question
            source_filter: Optional source filter (e.g., 'uc1', 'uc2')

        Returns:
            dict: Documentation context
        """
        # Get vector similarity results
        vector_results = self.vector_tool.run(query, source_filter=source_filter)

        # Get cypher query results
        cypher_results = self.cypher_tool.run(query, source_filter=source_filter)

        return {
            "vector_results": vector_results,
            "cypher_results": cypher_results,
            "source": "RetrieverAgent",
            "source_filter": source_filter,
            "timestamp": datetime.now().isoformat()
        }

    def classify_documentation_query(self, query: str) -> Dict[str, bool]:
        """
        Classify if a query needs documentation retrieval.

        Args:
            query: User's question

        Returns:
            dict: Classification of documentation requirements
        """
        query_lower = query.lower()

        # Keywords that indicate documentation needs
        documentation_keywords = [
            'how to', 'setup', 'configure', 'install', 'procedure', 'manual',
            'guide', 'steps', 'tutorial', 'programming', 'troubleshoot',
            'documentation', 'explain', 'what is', 'describe'
        ]

        # Component-specific keywords
        component_keywords = [
            # UC1 - Fischertechnik Training Factory keywords
            'plc', 'tia portal', 'simatic', 's7-1500', 'txt controller',
            'mqtt', 'opc ua', 'node-red', 'iot gateway', 'dashboard',
            'fischertechnik', 'training factory', 'high-bay warehouse',
            'suction gripper', 'sorting line', 'vacuum gripper', 'vgr',
            'hbw', 'sld', 'mpo', 'dso', 'dsi', 'factory components',
            
            # UC2 - Industrial Milling Machines keywords  
            'machine 7152', 'machine 11007', 'pm-10005', 'milling machine',
            'industrial machine', 'machine operation', 'machine programming',
            'machine maintenance', 'machine safety', 'machine specifications',
            'cutting parameters', 'tool change', 'calibration', 'cnc',
            'machining', 'manufacturing equipment', 'industrial equipment'
        ]

        needs_documentation = any(keyword in query_lower for keyword in documentation_keywords)
        is_component_query = any(keyword in query_lower for keyword in component_keywords)

        return {
            'needs_documentation': needs_documentation or is_component_query,
            'is_how_to_query': 'how to' in query_lower or 'how do' in query_lower,
            'is_component_query': is_component_query,
            'is_setup_query': any(word in query_lower for word in ['setup', 'configure', 'install']),
            'is_troubleshoot_query': 'troubleshoot' in query_lower or 'problem' in query_lower
        }

    def query(self, user_query: str, source_filter: str = None) -> str:
        """
        Process a user query with full documentation context.

        Args:
            user_query: User's question
            source_filter: Optional source filter (e.g., 'uc1', 'uc2')

        Returns:
            str: Response with integrated documentation context
        """
        try:
            # Reinitialize conversation chain with appropriate context for the source filter
            self.initialize_conversation_chain(source_filter=source_filter)
            
            # Get documentation context
            doc_context = self.get_documentation_context(user_query, source_filter=source_filter)

            # Classify the query
            query_classification = self.classify_documentation_query(user_query)

            # Prepare enhanced input for the conversation
            enhanced_input = f"""
            User Query: {user_query}

            Query Classification: {json.dumps(query_classification, indent=2)}

            Documentation Context:
            Vector Search Results: {doc_context.get('vector_results', 'No vector results')}

            Cypher Query Results: {doc_context.get('cypher_results', 'No cypher results')}

            Based on the above context and documentation, please provide a comprehensive answer.
            """

            # Use the conversation chain with the appropriate system context
            response = self.conversation.predict(input=enhanced_input)

            # Reasoning models may leak reasoning-tag blocks into the content
            return strip_reasoning(response)[0]

        except Exception as e:
            return f"Error processing documentation query: {str(e)}"

    def get_available_sources(self) -> List[str]:
        """
        Get list of available sources in the Neo4j database.

        Returns:
            List[str]: List of available source values
        """
        try:
            query = """
            MATCH (n:Document)
            WHERE n.source IS NOT NULL
            RETURN DISTINCT n.source AS source
            ORDER BY source
            """
            result = neo4j_graph.query(query)
            sources = [record['source'] for record in result]
            return sources
        except Exception as e:
            print(f"Error retrieving available sources: {e}")
            return []

    def get_neo4j_status(self) -> Dict[str, Any]:
        """
        Get a comprehensive overview of the current Neo4j system status.

        Returns:
            dict: Complete system overview with all available data
        """
        # Return the status Neo4jDB connection

        index_info = graph.query("SHOW INDEXES")

        neo4j_status = {}

        for index in index_info:
            index_key = index['name']
            neo4j_status[index_key] = {
                'provider': index['indexProvider'],
                'type': index['entityType'],
                'state': index['state'],
                'last_read': index['lastRead'].to_native().strftime("%Y-%m-%d %H:%M:%S UTC") if index['lastRead'] else None
            }

        return neo4j_status
