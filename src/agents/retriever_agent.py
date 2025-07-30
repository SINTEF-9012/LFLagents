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

from langchain_openai import ChatOpenAI
from langchain_community.chains.graph_qa.cypher import GraphCypherQAChain

from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import ConversationChain
from langchain.chains.conversation.memory import ConversationBufferWindowMemory

from langchain.pydantic_v1 import BaseModel
from langchain.tools import BaseTool
from typing import Type, Dict, Any

import streamlit as st
from langchain.prompts import (
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
    PromptTemplate
)

# Load environment variables
CHAT_MODEL = os.getenv('CHAT_MODEL')
TEMPERATURE = os.getenv('TEMPERATURE')

# Load the graph db
neo4j_graph = graph

class Question(BaseModel):
    question: str

class Neo4jVectorTool(BaseTool):
    name: str = "neo4j_vector_tool"
    description: str = "Tool for performing similarity-based retrieval from the Neo4j vector index."
    args_schema: Type[Question] = Question

    def _run(self, question):
        # Initialize Neo4j vector index
        neo4j_vector = load_vector_index()

        # Perform similarity search
        results = neo4j_vector.similarity_search(question, k=5)
        all_pages = "\n".join([result.page_content for result in results])

        return all_pages

class Neo4jCypherTool(BaseTool):
    name: str = "neo4j_cypher_tool"
    description: str = "Tool for retrieving structured, interconnected, and relational knowledge from the Neo4j graph."
    args_schema: Type[Question] = Question

    def _run(self, question: str):
        """
        Execute a Cypher query based on the user's question.

        Args:
            question (str): The user's natural language question.

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
        CYPHER_LLM = ChatOpenAI(model_name=CHAT_MODEL, temperature=TEMPERATURE)

        # Generate Cypher query
        CYPHER_GENERATION_TEMPLATE = """
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

        Schema:
        {schema}

        Question: {question}

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

        return chain.run(question)

class RetrieverAgent:

    def __init__(self):

        # Initialize tools
        self.cypher_tool = Neo4jCypherTool()
        self.vector_tool = Neo4jVectorTool()

        # Initialize chat model
        self.llm = ChatOpenAI(model_name=CHAT_MODEL, temperature=TEMPERATURE)
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


    def initialize_conversation_chain(self):
        self.system_msg_template = SystemMessagePromptTemplate.from_template(
            template="""
            You are an expert assistant trained on the fischertechnik Training Factory Industry 4.0 system.
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
        )
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

    def get_documentation_context(self, query: str) -> Dict[str, Any]:
        """
        Get relevant documentation context using retriever tools.

        Args:
            query: User's question

        Returns:
            dict: Documentation context
        """
        # Get vector similarity results
        vector_results = self.vector_tool.run(query)

        # Get cypher query results
        cypher_results = self.cypher_tool.run(query)

        return {
            "vector_results": vector_results,
            "cypher_results": cypher_results,
            "source": "RetrieverAgent",
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
            'plc', 'tia portal', 'simatic', 's7-1500', 'txt controller',
            'mqtt', 'opc ua', 'node-red', 'iot gateway', 'dashboard'
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

    def query(self, user_query: str) -> str:
        """
        Process a user query with full documentation context.

        Args:
            user_query: User's question

        Returns:
            str: Response with integrated documentation context
        """
        try:
            # Get documentation context
            doc_context = self.get_documentation_context(user_query)

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

            # Use the existing conversation chain
            response = self.conversation.predict(input=enhanced_input)

            return response

        except Exception as e:
            return f"Error processing documentation query: {str(e)}"

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
