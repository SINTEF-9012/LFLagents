import json
import os
import sys
from datetime import datetime
from typing import Any, Dict, List, Optional

from langchain.chains import LLMChain
from langchain.chat_models import ChatOpenAI
from langchain.prompts import (ChatPromptTemplate, HumanMessagePromptTemplate,
                               SystemMessagePromptTemplate)
from langchain_openai import ChatOpenAI

from agents.monitoring_agent import MonitoringAgent
from agents.retriever_agent import RetrieverAgent

# Load environment variables
CHAT_MODEL = os.getenv('CHAT_MODEL')
TEMPERATURE = os.getenv('TEMPERATURE')

class IntegratedManufacturingAgent:
    """
    Integrated Manufacturing Agent that combines retriever and monitoring capabilities.
    
    This agent provides comprehensive manufacturing support by:
    1. Retrieving documentation and manuals from Neo4j (via RetrieverAgent)
    2. Monitoring real-time manufacturing data from SINDIT API (via MonitoringAgent)
    3. Providing unified responses that combine both static knowledge and live data
    """
    
    def __init__(self):
        """Initialize the Integrated Agent with both retriever and monitoring capabilities."""
        
        self.monitoring_agent = MonitoringAgent()
        self.retriever_agent = RetrieverAgent()
        self.llm = ChatOpenAI(model_name=CHAT_MODEL, temperature=TEMPERATURE)
        self.initialize_conversation_chain()
    
    def initialize_conversation_chain(self):
        """Initialize the conversation chain for the integrated agent."""
        
        self.system_msg_template = SystemMessagePromptTemplate.from_template(
            template="""
            You are an Integrated Manufacturing Assistant for the fischertechnik Training Factory Industry 4.0 system.
            You combine both comprehensive documentation knowledge and real-time monitoring capabilities.

            Your capabilities include:

            DOCUMENTATION & KNOWLEDGE (via Neo4j):
            - Factory components and configuration details
            - SIMATIC S7-1500 PLC programming and TIA Portal usage
            - TXT controller setup and communication protocols (MQTT/OPC UA)
            - IoT Gateway configuration and Node-RED dashboards
            - Cloud integration and dashboard management
            - Educational simulations and programming tasks
            - System setup, networking, and troubleshooting guides

            REAL-TIME MONITORING (via SINDIT API):
            - Live sensor data (temperature, humidity, air quality, brightness)
            - Manufacturing equipment status (VGR, HBW, SLD, MPO, DSO, DSI)
                - VGR: Vacuum Gripper Robot
                - HBW: High Bay Warehouse
                - SLD: Sorting Line with Detection
                - MPO: Multi Processing Station with Oven
                - DSI/DSO: Sensor for incoming/outgoing pieces
            - Data connection health and streaming properties
            - Factory operational status and performance metrics
            - Environmental condition monitoring
            - System connectivity and health assessment

            INTEGRATION APPROACH:
            When answering questions:
            1. First determineide  if the question requires static knowledge, real-time data, or both
            2. Retrieve relevant documentation/manuals if needed for context
            3. Get current real-time data if the question involves current status or monitoring
            4. Combine both sources to provcomprehensive, accurate responses
            5. Clearly distinguish between documented procedures and current operational status
            6. Provide related information that bridge theory and practice

            RESPONSE STRUCTURE:
            - Start with current status/data if relevant
            - Provide contextual documentation/procedures as needed
            - Indicate data sources (documentation vs. real-time) for transparency

            If either data source is unavailable, clearly indicate this and work with available information.
            Always prioritize accuracy and practical utility in manufacturing contexts.
            """
        )
        
        self.human_msg_template = HumanMessagePromptTemplate.from_template(template="{input}")
        
        self.prompt_template = ChatPromptTemplate.from_messages([
            self.system_msg_template,
            self.human_msg_template
        ])
        
        self.conversation_chain = LLMChain(
            llm=self.llm,
            prompt=self.prompt_template,
            verbose=True
        )
    
    def classify_query_type(self, query: str) -> Dict[str, bool]:
        """
        Classify what type of information the query requires.
        
        Args:
            query: User's question
        
        Returns:
            dict: Classification of query requirements
        """
        query_lower = query.lower()
        
        # Define keyword/phrase lists
        realtime_keywords = [
            'current', 'now', 'status', 'temperature', 'humidity', 'sensor',
            'monitoring', 'real-time', 'live', 'today', 'latest', 'present',
            'reading', 'value', 'measurement', 'data', 'active', 'connected'
        ]
        documentation_keywords = [
            'how to', 'setup', 'configure', 'install', 'procedure', 'manual',
            'guide', 'steps', 'tutorial', 'programming', 'troubleshoot', 'documentation',
            'instruction', 'explain', 'reference'
        ]
        
        needs_realtime = any(keyword in query_lower for keyword in realtime_keywords)
        needs_documentation = any(keyword in query_lower for keyword in documentation_keywords)
        
        # If neither is clearly indicated, assume both might be useful
        if not needs_realtime and not needs_documentation:
            needs_realtime = True
            needs_documentation = True
        
        return {
            'MonitoringAgent': needs_realtime,
            'RetrieverAgent': needs_documentation,
        }
    

    def query(self, user_query: str) -> str:
        """
        Process a user query using integrated knowledge and monitoring.
        
        Args:
            user_query: User's question
        
        Returns:
            str: Comprehensive integrated response
        """
        # Classify the query type
        query_classification = self.classify_query_type(user_query)
        
        # Gather context based on query requirements
        context = {
            "user_query": user_query,
            "timestamp": datetime.now().isoformat(),
            "query_classification": query_classification
        }
        
        # Get documentation context if needed
        if query_classification['RetrieverAgent']:
            doc_context = self.retriever_agent.get_documentation_context(user_query)
            context["documentation"] = doc_context
        
        # Get real-time context if needed
        if query_classification['MonitoringAgent']:
            realtime_context = self.monitoring_agent.get_realtime_context(user_query)
            context["realtime"] = realtime_context
                
        # Format the input for the conversation chain
        formatted_input = f"""
        User Query: {user_query}
        
        Query Classification: 
        {json.dumps(query_classification, indent=2)}
        
        Available Context:
        {json.dumps(context, indent=3)}
        
        Please provide a comprehensive response that integrates both documentation knowledge 
        and real-time monitoring data as appropriate for this query.
        """
        
        # Generate response using the conversation chain
        response = self.conversation_chain.run(input=formatted_input)
        
        return response
    
    def get_system_status(self) -> Dict[str, Any]:
        """
        Get comprehensive system status combining all data sources.
        
        Returns:
            dict: Complete system status
        """
        status = {
            "system_components": {}
        }
        
        # Get monitoring status
        monitoring_overview = self.monitoring_agent.get_sindit_status()
        status["system_components"]["monitoring"] = monitoring_overview
            
        # Get documentation system status
        doc_status = self.retriever_agent.get_neo4j_status()
        status["system_components"]["documentation"] = doc_status
            
        return status
    
    def get_manufacturing_insights(self) -> str:
        """
        Generate comprehensive manufacturing insights.
        
        Returns:
            str: Detailed insights about current manufacturing status
        """
        insights_query = """
            "Provide comprehensive insights about the current manufacturing system including:\n"
            "1. Current sensor readings and environmental conditions\n"
            "2. Equipment status and operational efficiency\n"
            "3. System health and connectivity status\n"
        """
        
        return self.query(insights_query)
    