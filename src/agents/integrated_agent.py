"""
Integrated Manufacturing Agent for Comprehensive Manufacturing Intelligence.

This module provides an integrated agent that combines retrieval, monitoring,
and analytics capabilities to provide comprehensive manufacturing intelligence
and decision support.
"""

import json
import os
import numpy as np
import re
from datetime import datetime
from typing import Any, Dict, Optional, Tuple, List

from langchain.chains import LLMChain
from langchain.prompts import (ChatPromptTemplate, HumanMessagePromptTemplate,
                               SystemMessagePromptTemplate)
from utils.llm_config import get_llm_instance
from utils.neo4j_connection import embedding_model

from agents.monitoring_agent import MonitoringAgent
from agents.retriever_agent import RetrieverAgent
from agents.analytics_agent import AnalyticsAgent

# Load environment variables
CHAT_MODEL = os.getenv('CHAT_MODEL')
TEMPERATURE = os.getenv('TEMPERATURE')

class IntegratedManufacturingAgent:
    """
    Integrated Manufacturing Agent that combines retriever, monitoring, and analytics capabilities.

    This agent provides comprehensive manufacturing support by:
    1. Retrieving documentation and manuals from Neo4j (via RetrieverAgent)
    2. Monitoring real-time manufacturing data from SINDIT DT platform (via MonitoringAgent)
    3. Analyzing historical manufacturing data with charts and insights (via AnalyticsAgent)
    4. Providing unified responses that combine static knowledge, live data, and historical analytics
    """

    def __init__(self):
        """Initialize the Integrated Agent with retriever, monitoring, and analytics capabilities."""

        self.monitoring_agent = MonitoringAgent()
        self.retriever_agent = RetrieverAgent()
        self.analytics_agent = AnalyticsAgent()
        self.llm = get_llm_instance(model_name=CHAT_MODEL, temperature=TEMPERATURE)
        self.embeddings = embedding_model
        self.last_parsed_time_range = None  # Store the last parsed time range
        
        # Initialize semantic classification system
        self._initialize_semantic_classification()
        self.initialize_conversation_chain()

    def _initialize_semantic_classification(self):
        """Initialize the semantic classification system with example questions and embeddings."""
        
        # Curated taxonomy of manufacturing-related use cases with example questions
        self.example_questions = {
            'documentation': [
                # UC1 - Fischertechnik Training Factory documentation examples
                "How to configure the SIMATIC S7-1500 PLC for the training factory?",
                "What are the setup steps for TIA Portal?",
                "How do I program the TXT controller?",
                "What is the procedure for installing the IoT Gateway?",
                "How to configure MQTT communication protocols?",
                "What are the networking requirements for the system?",
                "How to troubleshoot PLC connection issues?",
                "What is the definition of OPC UA in this context?",
                "Explain the factory components and their functions",
                "What are the programming steps for Node-RED dashboards?",
                "How to set up cloud integration?",
                "What manual should I follow for system maintenance?",
                "Guide me through the educational simulation setup",
                "What are the safety procedures for the training factory?",
                "How to configure the vacuum gripper robot programming?",
                # UC2 - Industrial Milling Machines documentation examples
                "How to operate machine 7152?",
                "What are the specifications for machine 11007?",
                "How to program the PM-10005 machine?",
                "Where can I find technical documentation for the machines?",
                "How to perform maintenance on industrial machines?",
                "What are the safety procedures for the machines?",
                "Machine programming and setup instructions",
                "Milling machine operation manual",
                "Industrial equipment maintenance procedures",
                "How to troubleshoot machine 7152?",
                "What is the operating temperature range for machine 11007?",
                "PM-10005 calibration procedures",
                "Machine safety protocols and guidelines",
                "How to change tools on the milling machine?",
                "What are the cutting parameters for different materials?"
            ],
            'monitoring': [
                "What is the current temperature in the factory?",
                "Show me the live humidity readings",
                "What is the current status of the VGR robot?",
                "Are all sensors connected and active?",
                "What is the present air quality measurement?",
                "Show me the real-time brightness sensor data",
                "What is the current operational status of HBW?",
                "Are there any active alarms or alerts now?",
                "What is the latest data from the factory sensors?",
                "Show me the current position of the vacuum gripper",
                "What is the live status of the sorting line?",
                "Is the multi-processing station oven currently active?",
                "What are the present environmental conditions?",
                "Show me today's sensor readings",
                "What is the current connectivity status of all equipment?"
            ],
            'analytics': [
                "Analyze the temperature trends over the last week",
                "Show me the process efficiency patterns from yesterday",
                "What were the humidity levels between December 1st and 15th?",
                "Compare the performance metrics from last month",
                "Generate a chart showing cycle time trends",
                "Analyze the historical data for equipment downtime",
                "What patterns can you identify in the manufacturing process?",
                "Show me the correlation between temperature and humidity over time",
                "Visualize the performance trends for the past quarter",
                "What insights can you derive from the process data history?",
                "Analyze the operational efficiency from January to March",
                "Generate a report on equipment utilization patterns",
                "Show me the historical analysis of environmental conditions",
                "Compare cycle times between different time periods"
            ]
        }
        
        # Generate embeddings for all example questions
        self.question_embeddings = {}
        self.category_mapping = {}
        
        try:
            for category, questions in self.example_questions.items():
                embeddings = []
                for question in questions:
                    embedding = self.embeddings.embed_query(question)
                    embeddings.append(embedding)
                    self.category_mapping[question] = category
                    self.question_embeddings[category] = np.array(embeddings)
        except Exception as e:
            # Fallback: use simple keyword matching if embeddings fail
            print(f"Warning: Semantic classification initialization failed: {e}")
            self.question_embeddings = None

    def _cosine_similarity(self, vec1, vec2):
        """Calculate cosine similarity between two vectors."""
        vec1 = np.array(vec1)
        vec2 = np.array(vec2)
        return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

    def _semantic_classify_query(self, query: str, similarity_threshold: float = 0.7) -> Dict[str, float]:
        """
        Use semantic similarity to classify query against example questions.
        
        Args:
            query: User's question
            similarity_threshold: Minimum similarity score to consider a match
            
        Returns:
            dict: Category confidence scores
        """
        if self.question_embeddings is None:
            # Fallback to equal weights if semantic search is unavailable
            return {'documentation': 0.33, 'monitoring': 0.33, 'analytics': 0.33}
        
        try:
            # Get embedding for the user query
            query_embedding = self.embeddings.embed_query(query)
            
            category_scores = {}
            
            for category, embeddings in self.question_embeddings.items():
                # Calculate cosine similarity with all examples in this category
                similarities = []
                for embedding in embeddings:
                    similarity = self._cosine_similarity(query_embedding, embedding)
                    similarities.append(similarity)
                
                # Use the maximum similarity score for this category
                max_similarity = max(similarities) if similarities else 0
                category_scores[category] = max_similarity
            
            return category_scores
            
        except Exception as e:
            print(f"Warning: Semantic classification failed: {e}")
            # Fallback to equal weights
            return {'documentation': 0.33, 'monitoring': 0.33, 'analytics': 0.33}

    def initialize_conversation_chain(self, source_filter: str = None):
        """Initialize the conversation chain for the integrated agent with use case-specific context."""

        # Create context-aware system message based on source filter
        if source_filter == "uc1":
            system_template = """
            You are an Integrated Manufacturing Assistant specialized in the fischertechnik Training Factory Industry 4.0 system.
            You combine comprehensive documentation knowledge, real-time monitoring capabilities, and advanced analytics.

            Your capabilities include:

            DOCUMENTATION & KNOWLEDGE (via Neo4j):
            - Factory components and configuration details (high-bay warehouse, suction gripper, sorting line)
            - SIMATIC S7-1500 PLC programming and TIA Portal usage
            - TXT controller setup and communication protocols (MQTT/OPC UA)
            - IoT Gateway configuration and Node-RED dashboards
            - Cloud integration and dashboard management
            - Educational simulations and programming tasks
            - System setup, networking, and troubleshooting guides

            REAL-TIME MONITORING (via SINDIT DT Platform):
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

            HISTORICAL ANALYTICS & INSIGHTS (via Analytics Agent):
            - Process cycle analysis and optimization
            - Historical data trends and patterns
            - Component performance analysis
            - Environmental condition analysis
            - Interactive charts and visualizations
            - Data-driven manufacturing insights
            - Process efficiency metrics
            - Predictive maintenance indicators

            Focus on fischertechnik Training Factory specific information and provide detailed technical guidance for Industry 4.0 learning scenarios.
            """
        elif source_filter == "uc2":
            system_template = """
            You are an Integrated Manufacturing Assistant specialized in industrial milling machines and manufacturing equipment.
            You combine comprehensive documentation knowledge, real-time monitoring capabilities, and advanced analytics.

            Your capabilities include:

            DOCUMENTATION & KNOWLEDGE (via Neo4j):
            - Industrial milling machine operation and specifications (machines 7152, 11007, PM-10005)
            - Machine programming and setup instructions
            - Safety protocols and maintenance procedures
            - Technical documentation and operating parameters
            - Troubleshooting and repair procedures
            - Manufacturing process optimization

            Note: Documentation may be in Spanish, but you should respond in English unless specifically asked otherwise.

            REAL-TIME MONITORING (via SINDIT DT Platform):
            - Live sensor data (temperature, humidity, air quality, brightness)
            - Manufacturing equipment status monitoring
            - Environmental condition monitoring
            - System connectivity and health assessment

            HISTORICAL ANALYTICS & INSIGHTS (via Analytics Agent):
            - Process cycle analysis and optimization
            - Historical data trends and patterns
            - Component performance analysis
            - Environmental condition analysis
            - Interactive charts and visualizations
            - Data-driven manufacturing insights
            - Process efficiency metrics
            - Predictive maintenance indicators

            Focus on industrial milling machine operations, maintenance, and manufacturing processes. Provide practical guidance for machine operation and maintenance.
            """
        else:
            # Default template for all sources or when no filter is specified
            system_template = """
            You are an Integrated Manufacturing Assistant that combines comprehensive documentation knowledge, real-time monitoring capabilities, and advanced analytics.

            Your capabilities cover two main areas:
            1. Fischertechnik Training Factory Industry 4.0 system (UC1)
            2. Industrial milling machines and manufacturing equipment (UC2)

            DOCUMENTATION & KNOWLEDGE (via Neo4j):
            - Factory automation components and PLC systems (UC1)
            - Industrial machine operation and maintenance (UC2)
            - Communication protocols (MQTT/OPC UA)
            - Manufacturing processes and procedures
            - Safety protocols and technical specifications
            - System setup, programming, and troubleshooting

            REAL-TIME MONITORING (via SINDIT DT Platform):
            - Live sensor data (temperature, humidity, air quality, brightness)
            - Manufacturing equipment status monitoring
            - Environmental condition monitoring
            - System connectivity and health assessment

            HISTORICAL ANALYTICS & INSIGHTS (via Analytics Agent):
            - Process cycle analysis and optimization
            - Historical data trends and patterns
            - Component performance analysis
            - Environmental condition analysis
            - Interactive charts and visualizations
            - Data-driven manufacturing insights
            - Process efficiency metrics
            - Predictive maintenance indicators

            Provide comprehensive assistance across both training factory automation and industrial manufacturing domains.
            """

        # Common integration approach for all templates
        integration_approach = """
        INTEGRATION APPROACH:
        When answering questions:
        1. First determine if the question requires static knowledge, real-time data, historical analytics, or combinations
        2. Retrieve relevant documentation/manuals if needed for context
        3. Get current real-time data if the question involves current status or monitoring
        4. Perform historical analysis if the question involves trends, patterns, or performance optimization
        5. Combine all sources to provide comprehensive, accurate responses
        6. Clearly distinguish between documented procedures, current operational status, and historical insights
        7. Provide related information that bridges theory, current practice, and historical performance

        RESPONSE STRUCTURE:
        - Start with current status/data if relevant
        - Provide contextual documentation/procedures as needed
        - Include historical insights and trends when applicable
        - Indicate data sources (documentation vs. real-time vs. historical) for transparency

        If any data source is unavailable, clearly indicate this and work with available information.
        Always prioritize accuracy and practical utility in manufacturing contexts.
        """

        full_template = system_template + integration_approach

        self.system_msg_template = SystemMessagePromptTemplate.from_template(template=full_template)

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
        Classify what type of information the query requires using semantic understanding.
        
        This simplified approach relies on:
        1. Semantic vector similarity against curated examples
        2. LLM classification for edge cases
        3. Time pattern detection for analytics override

        Args:
            query: User's question

        Returns:
            dict: Classification of query requirements
        """
        query_lower = query.lower()

        # STEP 1: Semantic vector search (primary classification method)
        semantic_scores = self._semantic_classify_query(query)
        
        # Find the most confident semantic category
        max_semantic_category = max(semantic_scores.items(), key=lambda x: x[1])
        max_semantic_score = max_semantic_category[1]
        
        # STEP 2: High confidence semantic classification
        if max_semantic_score > 0.8:
            # High confidence - use semantic classification directly
            needs_realtime = max_semantic_category[0] == 'monitoring'
            needs_documentation = max_semantic_category[0] == 'documentation'
            needs_analytics = max_semantic_category[0] == 'analytics'
            
            # For very high confidence (>0.95), skip time pattern detection to avoid overrides
            time_override_allowed = max_semantic_score <= 0.95
        else:
            # STEP 3: Low confidence - use LLM classification
            llm_classification = self._llm_classify_query(query)
            needs_realtime = llm_classification.get('realtime', False)
            needs_documentation = llm_classification.get('documentation', True)  # Default fallback
            needs_analytics = llm_classification.get('analytics', False)
            
            # Allow time pattern override for LLM classifications
            time_override_allowed = True

        # STEP 4: Time pattern detection (only for analytics enhancement)
        if time_override_allowed and self._has_time_patterns(query_lower):
            needs_analytics = True

        # STEP 5: Ensure at least one agent is selected (safety net)
        if not needs_realtime and not needs_documentation and not needs_analytics:
            # Use the best semantic score as fallback
            if max_semantic_category[0] == 'monitoring':
                needs_realtime = True
            elif max_semantic_category[0] == 'documentation':
                needs_documentation = True
            elif max_semantic_category[0] == 'analytics':
                needs_analytics = True
            else:
                # Ultimate fallback - default to documentation for unknown queries
                needs_documentation = True

        return {
            'MonitoringAgent': bool(needs_realtime),
            'RetrieverAgent': bool(needs_documentation),
            'AnalyticsAgent': bool(needs_analytics),
            # # Include diagnostic information for debugging
            # '_classification_debug': {
            #     'semantic_scores': {k: float(v) for k, v in semantic_scores.items()},
            #     'max_semantic_category': (max_semantic_category[0], float(max_semantic_category[1])),
            #     'max_semantic_score': float(max_semantic_score),
            #     'classification_method': 'semantic' if max_semantic_score > 0.8 else 'llm',            
            #     }
        }

    def _has_time_patterns(self, query_lower: str) -> bool:
        """
        Detect time-based patterns that indicate analytics queries.
        
        Args:
            query_lower: Lowercase query string
            
        Returns:
            bool: True if time patterns are detected
        """
        time_patterns = [
            # Time range indicators
            ' from ', ' to ', ' between ', 
            # Date indicators (months)
            'december', 'january', 'february', 'march', 'april', 'may', 
            'june', 'july', 'august', 'september', 'october', 'november',
            # Time indicators  
            ' on ', ' at ', ' during ', ':', ' am', ' pm', 
            # Relative time indicators
            'yesterday', ' last ', 'previous', 'past ', 'over time',
            # Specific time phrases (precise to avoid false positives)
            'for the last ', 'for the past ', 'in the last ', 'in the past ',
            'over the last ', 'over the past ', 'during the last ', 'during the past '
        ]
        
        # Check for time patterns
        has_time_pattern = any(time_pattern in query_lower for time_pattern in time_patterns)
        
        # Check for time + duration combinations
        time_durations = ['hour', 'day', 'week', 'month', 'year', 'quarter']
        has_duration = any(duration in query_lower for duration in time_durations)
        
        return has_time_pattern or (has_duration and any(word in query_lower for word in ['last', 'past', 'previous', 'ago']))

    def _llm_classify_query(self, query: str) -> Dict[str, bool]:
        """
        Use LLM to classify query when keyword matching is insufficient.

        Args:
            query: User's question

        Returns:
            dict: LLM-based classification
        """
        try:
            classification_prompt = f"""
            Analyze this query and determine what type of information sources are needed:

            Query: "{query}"

            Please respond with a JSON object indicating whether each source is needed (true/false):
            {{
                "realtime": true/false,  // Current live data, sensors, active status
                "documentation": true/false,  // Manuals, procedures, setup guides
                "analytics": true/false  // Historical data, trends, time-based analysis
            }}

            Only respond with the JSON object, no additional text.
            """

            response = self.llm.predict(classification_prompt)

            # Parse the JSON response
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                classification = json.loads(json_match.group())
                return classification

        except Exception:
            pass

        # Fallback to default
        return {"realtime": True, "documentation": True, "analytics": True}

    def query(self, user_query: str, source_filter: str = None) -> str:
        """
        Process a user query using integrated knowledge, monitoring, and analytics.

        Args:
            user_query: User's question
            source_filter: Optional source filter for documentation retrieval (e.g., 'uc1', 'uc2')

        Returns:
            str: Comprehensive integrated response
        """
        # Reinitialize conversation chain with appropriate context for the source filter
        self.initialize_conversation_chain(source_filter=source_filter)
        
        # Classify the query type
        query_classification = self.classify_query_type(user_query)

        # Gather context based on query requirements
        context = {
            "user_query": user_query,
            "timestamp": datetime.now().isoformat(),
            "query_classification": query_classification,
            "source_filter": source_filter
        }

        # Get documentation context if needed
        if query_classification['RetrieverAgent']:
            try:
                doc_context = self.retriever_agent.get_documentation_context(user_query, source_filter=source_filter)
                context["documentation"] = doc_context
            except Exception as e:
                context["documentation"] = {"error": f"Documentation unavailable: {str(e)}"}

        # Get real-time context if needed
        if query_classification['MonitoringAgent']:
            try:
                realtime_context = self.monitoring_agent.get_realtime_context(user_query)
                context["realtime"] = realtime_context
            except Exception as e:
                context["realtime"] = {"error": f"Real-time monitoring unavailable: {str(e)}"}

        # Get analytics context if needed
        if query_classification['AnalyticsAgent']:
            try:
                # Parse time range from the user query first
                parsed_time_range = None
                if hasattr(self.analytics_agent, 'parse_time_range_from_query'):
                    parsed_time_range = self.analytics_agent.parse_time_range_from_query(user_query)
                    # Store the parsed time range for potential use by other components
                    self.last_parsed_time_range = parsed_time_range
                
                # Pass the parsed time range to the analytics agent
                analytics_result = self.analytics_agent.query(user_query, parsed_time_range)
                context["analytics"] = analytics_result
            except Exception as e:
                context["analytics"] = {"error": f"Analytics unavailable: {str(e)}"}

        # Format the input for the conversation chain - handle datetime and other serialization
        def json_serialize_helper(obj):
            if isinstance(obj, datetime):
                return obj.isoformat()
            elif isinstance(obj, (bool, int, float, str, list, dict)):
                return obj
            return str(obj)

        try:
            context_json = json.dumps(context, indent=2, default=json_serialize_helper)
            classification_json = json.dumps(query_classification, indent=2, default=json_serialize_helper)
        except Exception as e:
            # Fallback to string representation if JSON serialization fails
            context_json = str(context)
            classification_json = str(query_classification)

        formatted_input = f"""
        User Query: {user_query}

        Query Classification:
        {classification_json}

        Available Context:
        {context_json}

        Please provide a comprehensive response that integrates documentation knowledge,
        real-time monitoring data, and historical analytics as appropriate for this query.
        """

        # Generate response using the conversation chain
        response = self.conversation_chain.run(input=formatted_input)

        return response

    def get_available_sources(self) -> List[str]:
        """
        Get list of available sources from the RetrieverAgent.

        Returns:
            List[str]: List of available source values
        """
        try:
            return self.retriever_agent.get_available_sources()
        except Exception as e:
            print(f"Error retrieving available sources: {e}")
            return []

    def get_process_analysis(self, query: str, time_range: Optional[Tuple[datetime, datetime]] = None) -> Dict[str, Any]:
        """
        Get process analysis combining real-time monitoring and historical analytics.

        Args:
            query: User's process analysis query
            time_range: Optional time range for analysis

        Returns:
            dict: Combined process analysis results
        """
        try:
            # Parse time range from query if not explicitly provided
            if time_range is None and hasattr(self.analytics_agent, 'parse_time_range_from_query'):
                time_range = self.analytics_agent.parse_time_range_from_query(query)
            
            # Get analytics results with parsed time range
            analytics_result = self.analytics_agent.query(query, time_range)

            # Get current monitoring data for context
            current_context = self.monitoring_agent.get_realtime_context(query)

            # Combine results
            combined_analysis = {
                "query": query,
                "time_range": time_range,
                "historical_analysis": analytics_result,
                "current_status": current_context,
                "timestamp": datetime.now().isoformat()
            }

            return combined_analysis

        except Exception as e:
            return {"error": f"Process analysis failed: {str(e)}"}

    def analyze_cycles(self, time_range: Optional[Tuple[datetime, datetime]] = None) -> Dict[str, Any]:
        """
        Analyze manufacturing cycles using the analytics agent.

        Args:
            time_range: Optional time range for cycle analysis

        Returns:
            dict: Cycle analysis results
        """
        try:
            return self.analytics_agent.analyze_cycles(time_range)
        except Exception as e:
            return {"error": f"Cycle analysis failed: {str(e)}"}

    def get_component_metrics(self, component: str, time_range: Optional[Tuple[datetime, datetime]] = None) -> Dict[str, Any]:
        """
        Get detailed metrics for a specific component.

        Args:
            component: Component name (e.g., 'VGR', 'HBW', 'FACTORY')
            time_range: Optional time range for metrics

        Returns:
            dict: Component metrics and analysis
        """
        try:
            # Get analytics metrics
            metrics = self.analytics_agent.get_component_metrics(component)

            # Get current status from monitoring agent
            current_query = f"What is the current status of {component}?"
            current_status = self.monitoring_agent.get_realtime_context(current_query)

            return {
                "component": component,
                "time_range": time_range,
                "metrics": metrics,
                "current_status": current_status,
                "timestamp": datetime.now().isoformat()
            }

        except Exception as e:
            return {"error": f"Component metrics failed: {str(e)}"}

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
        try:
            monitoring_overview = self.monitoring_agent.get_sindit_status()
            status["system_components"]["monitoring"] = monitoring_overview
        except Exception as e:
            status["system_components"]["monitoring"] = {"status": "error", "error": str(e)}

        # Get documentation system status
        try:
            doc_status = self.retriever_agent.get_neo4j_status()
            status["system_components"]["documentation"] = doc_status
        except Exception as e:
            status["system_components"]["documentation"] = {"status": "error", "error": str(e)}

        # Get analytics system status
        try:
            analytics_status = {
                "status": "available",
                "loaded_data": list(self.analytics_agent.loaded_data.keys()) if hasattr(self.analytics_agent, 'loaded_data') else [],
                "default_time_range": str(self.analytics_agent.default_time_range) if hasattr(self.analytics_agent, 'default_time_range') else "unknown"
            }
        except Exception as e:
            analytics_status = {"status": "error", "error": str(e)}
        status["system_components"]["analytics"] = analytics_status

        return status

    def get_available_sources(self) -> List[str]:
        """
        Get list of available sources from the documentation system.

        Returns:
            List[str]: List of available source values
        """
        try:
            return self.retriever_agent.get_available_sources()
        except Exception as e:
            print(f"Error retrieving available sources: {e}")
            return []