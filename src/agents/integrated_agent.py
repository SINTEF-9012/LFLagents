import json
import os
import re
from datetime import datetime
from typing import Any, Dict, Optional, Tuple

from langchain.chains import LLMChain
from langchain.prompts import (ChatPromptTemplate, HumanMessagePromptTemplate,
                               SystemMessagePromptTemplate)
from langchain_openai import ChatOpenAI

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
        self.llm = ChatOpenAI(model_name=CHAT_MODEL, temperature=TEMPERATURE)
        self.last_parsed_time_range = None  # Store the last parsed time range
        self.initialize_conversation_chain()
    
    def initialize_conversation_chain(self):
        """Initialize the conversation chain for the integrated agent."""
        
        self.system_msg_template = SystemMessagePromptTemplate.from_template(
            template="""
            You are an Integrated Manufacturing Assistant for the fischertechnik Training Factory Industry 4.0 system.
            You combine comprehensive documentation knowledge, real-time monitoring capabilities, and advanced analytics.

            Your capabilities include:

            DOCUMENTATION & KNOWLEDGE (via Neo4j):
            - Factory components and configuration details
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
        Classify what type of information the query requires using both keyword matching and LLM analysis.
        
        Args:
            query: User's question
        
        Returns:
            dict: Classification of query requirements
        """
        query_lower = query.lower()
        
        # Keyword-based classification
        realtime_keywords = [
            'current', 'now', 'status', 'temperature', 'humidity', 'sensor',
            'monitoring', 'real-time', 'live', 'today', 'latest', 'present',
            'reading', 'value', 'measurement', 'data', 'active', 'connected'
        ]
        documentation_keywords = [
            'how to', 'setup', 'configure', 'install', 'procedure', 'manual',
            'guide', 'steps', 'tutorial', 'programming', 'troubleshoot', 'documentation',
            'instruction', 'explain', 'reference', 'what is', 'definition'
        ]
        analytics_keywords = [
            'analyze', 'analysis', 'trend', 'pattern', 'history', 'historical',
            'chart', 'graph', 'visualize', 'performance', 'efficiency', 'optimize',
            'cycle', 'process', 'compare', 'insight', 'report', 'summary',
            'over time', 'past', 'last hour', 'last day', 'yesterday', 'previous',
            'metrics', 'statistics', 'correlation', 'anomaly', 'december', 'january',
            'february', 'march', 'april', 'may', 'june', 'july', 'august',
            'september', 'october', 'november', 'from', 'to', 'between'
        ]
        
        # Basic keyword matching
        needs_realtime = any(keyword in query_lower for keyword in realtime_keywords)
        needs_documentation = any(keyword in query_lower for keyword in documentation_keywords)
        needs_analytics = any(keyword in query_lower for keyword in analytics_keywords)
        
        # Enhanced LLM-based classification for ambiguous cases
        if not needs_realtime and not needs_documentation and not needs_analytics:
            llm_classification = self._llm_classify_query(query)
            needs_realtime = llm_classification.get('realtime', True)
            needs_documentation = llm_classification.get('documentation', True)
            needs_analytics = llm_classification.get('analytics', True)
        
        # Special handling for time-based queries
        if any(time_indicator in query_lower for time_indicator in [
            'on ', 'at ', 'during ', 'from ', 'to ', 'between ', 'december', 'january',
            'february', 'march', 'april', 'may', 'june', 'july', 'august',
            'september', 'october', 'november', ':', 'am', 'pm', 'yesterday', 'last'
        ]):
            needs_analytics = True
        
        # If still no clear classification, use conservative approach
        if not needs_realtime and not needs_documentation and not needs_analytics:
            needs_realtime = True
            needs_documentation = True
            needs_analytics = True
        
        return {
            'MonitoringAgent': needs_realtime,
            'RetrieverAgent': needs_documentation,
            'AnalyticsAgent': needs_analytics,
        }
    
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
    
    def query(self, user_query: str) -> str:
        """
        Process a user query using integrated knowledge, monitoring, and analytics.
        
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
            try:
                doc_context = self.retriever_agent.get_documentation_context(user_query)
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
                # Use the analytics agent's time parsing capability directly
                time_range = self.analytics_agent.parse_time_range_from_query(user_query) if hasattr(self.analytics_agent, 'parse_time_range_from_query') else None
                analytics_result = self.analytics_agent.query(user_query, time_range)
                context["analytics"] = analytics_result
            except Exception as e:
                context["analytics"] = {"error": f"Analytics unavailable: {str(e)}"}
                
        # Format the input for the conversation chain - handle datetime serialization
        def json_serialize_helper(obj):
            if isinstance(obj, datetime):
                return obj.isoformat()
            return str(obj)
        
        try:
            context_json = json.dumps(context, indent=2, default=json_serialize_helper)
        except Exception as e:
            # Fallback to string representation if JSON serialization fails
            context_json = str(context)
        
        formatted_input = f"""
        User Query: {user_query}
        
        Query Classification: 
        {json.dumps(query_classification, indent=2)}
        
        Available Context:
        {context_json}
        
        Please provide a comprehensive response that integrates documentation knowledge, 
        real-time monitoring data, and historical analytics as appropriate for this query.
        """
        
        # Generate response using the conversation chain
        response = self.conversation_chain.run(input=formatted_input)
        
        return response
    
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
            # Get analytics results
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
    
    # def get_manufacturing_insights(self, time_range: Optional[Tuple[datetime, datetime]] = None) -> str:
    #     """
    #     Generate comprehensive manufacturing insights combining all data sources.
        
    #     Args:
    #         time_range: Optional time range for insights
            
    #     Returns:
    #         str: Detailed insights about manufacturing system
    #     """
    #     insights_query = """
    #     Provide comprehensive insights about the manufacturing system including:
    #     1. Current sensor readings and environmental conditions
    #     2. Equipment status and operational efficiency
    #     3. Historical performance trends and patterns
    #     4. Process optimization recommendations
    #     5. System health and connectivity status
    #     """
        
    #     if time_range:
    #         insights_query += f"\nAnalyze data from {time_range[0]} to {time_range[1]}"
        
    #     return self.query(insights_query)
    
    # def get_environmental_analysis(self, time_range: Optional[Tuple[datetime, datetime]] = None) -> Dict[str, Any]:
    #     """
    #     Get environmental analysis combining current and historical data.
        
    #     Args:
    #         time_range: Optional time range for environmental analysis
            
    #     Returns:
    #         dict: Environmental analysis results
    #     """
    #     try:
    #         # Get current environmental data
    #         current_env = self.monitoring_agent.get_realtime_context("environmental conditions")
            
    #         # Get historical environmental analysis
    #         env_query = "analyze environmental conditions including temperature, humidity, air quality"
    #         historical_env = self.analytics_agent.query(env_query, time_range)
            
    #         return {
    #             "current_conditions": current_env,
    #             "historical_analysis": historical_env,
    #             "time_range": time_range,
    #             "timestamp": datetime.now().isoformat()
    #         }
            
    #     except Exception as e:
    #         return {"error": f"Environmental analysis failed: {str(e)}"}
    
    # def get_analytics_insights(self, user_query: str, time_range: Optional[Tuple[datetime, datetime]] = None) -> Dict[str, Any]:
    #     """
    #     Get analytics insights directly from the analytics agent.
    #     This method provides a clean interface for pure analytics operations.
        
    #     Args:
    #         user_query: Analytics query
    #         time_range: Optional time range for analysis
            
    #     Returns:
    #         dict: Analytics insights with chart and analysis
    #     """
    #     try:
    #         # Parse time range if not provided
    #         if time_range is None:
    #             time_range = self.analytics_agent._parse_time_range_from_query(user_query) if hasattr(self.analytics_agent, '_parse_time_range_from_query') else None
            
    #         # Get analytics result
    #         analytics_result = self.analytics_agent.query(user_query, time_range)
            
    #         # Try to generate interactive chart
    #         chart = None
    #         chart_error = None
    #         try:
    #             chart = self.analytics_agent.get_interactive_chart(user_query, time_range)
    #         except Exception:
    #             chart_error = "Chart generation failed"
            
    #         # Try to get process analysis if relevant
    #         process_analysis = None
    #         if any(keyword in user_query.lower() for keyword in ['process', 'cycle', 'efficiency', 'performance']):
    #             try:
    #                 process_analysis = self.get_process_analysis(user_query, time_range)
    #             except Exception:
    #                 pass
            
    #         return {
    #             "query": user_query,
    #             "time_range": {
    #                 "start": time_range[0].isoformat() if time_range else None,
    #                 "end": time_range[1].isoformat() if time_range else None
    #             },
    #             "analytics_result": analytics_result,
    #             "chart_available": chart is not None,
    #             "chart": chart,
    #             "chart_error": chart_error,
    #             "process_analysis": process_analysis,
    #             "timestamp": datetime.now().isoformat()
    #         }
            
    #     except Exception as e:
    #         return {
    #             "query": user_query,
    #             "error": f"Analytics insights failed: {str(e)}",
    #             "timestamp": datetime.now().isoformat()
    #         }
