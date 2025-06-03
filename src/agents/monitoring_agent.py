import json
import os
import sys
from datetime import datetime
from typing import Any, Dict, Optional

# Add the src directory to the path to import utils
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langchain.chains import LLMChain
from langchain.prompts import (ChatPromptTemplate, HumanMessagePromptTemplate,
                               SystemMessagePromptTemplate)
from langchain.tools import BaseTool
from langchain_community.chat_models import ChatOpenAI
from pydantic import BaseModel

from utils.sindit_api import query_get_api, query_get_connections_info

# Load environment variables
CHAT_MODEL = os.getenv('CHAT_MODEL')
TEMPERATURE = os.getenv('TEMPERATURE')

class MonitoringQuery(BaseModel):
    query: str
    asset_type: Optional[str] = None
    property_type: Optional[str] = None

class SINDITDataTool(BaseTool):
    name: str = "sindit_data_tool"
    description: str = "Tool for retrieving real-time manufacturing data from SINDIT Knowledge Graph API."
    args_schema: type[MonitoringQuery] = MonitoringQuery
    
    def classify_query(self, query: str) -> Dict[str, Any]:
        key_assets = [
            "http://sindit.sintef.no/2.0#factory-sensor",
            "http://sindit.sintef.no/2.0#camera",
            "http://sindit.sintef.no/2.0#vgr",
            "http://sindit.sintef.no/2.0#hbw",
            "http://sindit.sintef.no/2.0#sld",
            "http://sindit.sintef.no/2.0#mpo",
            "http://sindit.sintef.no/2.0#dso",
            "http://sindit.sintef.no/2.0#dsi",
            "http://sindit.sintef.no/2.0#order",
        ]
        
        asset_keywords = {
            "vgr": ["vgr", "vacuum gripper", "gripper", "robot"],
            "hbw": ["hbw", "high bay warehouse", "warehouse", "storage"],
            "sld": ["sld", "sorting line", "detection", "sorting"],
            "mpo": ["mpo", "multi processing", "oven", "processing station"],
            "dso": ["dso", "outgoing", "output", "piece out"],
            "dsi": ["dsi", "incoming", "input", "piece in"],
            "factory-sensor": ["temperature", "humidity", "air quality", "brightness", "sensor", "environmental", "environment"],
            "camera": ["camera", "vision", "image", "picture"],
            "order": ["order", "production order", "manufacturing order", "order status"],
            "Stock": ["stock", "inventory", "material"]
        }
        
        query_lower = query.lower()
        
        # Find specific assets mentioned in the query
        relevant_assets = []
        for asset_key, keywords in asset_keywords.items():
            if any(keyword in query_lower for keyword in keywords):
                asset_uri = f"http://sindit.sintef.no/2.0#{asset_key}"
                if asset_uri in key_assets:
                    relevant_assets.append(asset_uri)
                    
        # If no specific assets found, default to key manufacturing assets
        if not relevant_assets:
            return key_assets
        
        return relevant_assets

    def _run(self, query: str, **kwargs):
        """
        Retrieve real-time data from SINDIT API based on the query.
        
        Args:
            query: Natural language query about manufacturing data
        
        Returns:
            dict: Real-time data from SINDIT API
        """
        
        relevant_assets = self.classify_query(query)
        
        # Get available data using correct endpoints
        streaming_data = {}
        
        # Get all node types to understand available data structure
        node_types = query_get_api('kg/node_types')
        streaming_data['node_types'] = node_types
        
        assets = []
        
        for asset_uri in relevant_assets:
            asset_data = query_get_api('kg/node', asset_uri, 'node_uri')
            assets.append(asset_data)
        
        streaming_data['data'] = assets
        
        # Get connection information     
        connection_data = query_get_api('kg/node', "http://sindit.sintef.no/2.0#mqtt-connection", 'node_uri')
        streaming_data['connections'] = [connection_data] if connection_data else []
        
        return streaming_data


# class ManufacturingStatusAnalyzer(BaseTool):
#     name: str = "manufacturing_status_analyzer"
#     description: str = "Tool for analyzing manufacturing status based on real-time data and providing insights."
#     args_schema: type[MonitoringQuery] = MonitoringQuery

#     def _run(self, query: str, **kwargs):
#         """
#         Analyze manufacturing status and provide insights.
        
#         Args:
#             query: Natural language query about manufacturing status
#             **kwargs: Additional parameters like asset_type, property_type
        
#         Returns:
#             dict: Analysis results
#         """
#         # Initialize the SINDIT data tool
#         data_tool = SINDITDataTool()
#         raw_data = data_tool.run(query)
        
#         # Analyze the data
#         analysis_results = {
#             "timestamp": datetime.now().isoformat(),
#             # "query": query,
#             # "status": "analyzing",
#             "summary": []
#         }
        
#         # Analyze streaming properties
#         data = raw_data['data']
        
#         # Extract types and topics of assets and sensors
#         assets_topics = []
#         assets_types = []
#         sensor_topics = []
#         sensor_types = []

#         for asset in data:
#             if asset['class_uri'] == "urn:samm:sindit.sintef.no:1.0.0#AbstractAsset":
#                 assets_topics.append(asset['uri'])
#                 assets_types.append(asset['label'])
#                 if 'assetProperties' in asset:
#                     for prop in asset['assetProperties']:
#                         sensor_topics.append(prop['uri'])
#                         sensor_types.append(prop['label'])
        
#         analysis_results["assets_topics"] = list(set(assets_topics))
#         analysis_results["assets_types"] = list(set(assets_types))
#         analysis_results["sensor_topics"] = list(set(sensor_topics))
#         analysis_results["sensor_types"] = list(set(sensor_types))
        
        
#         # Analyze connections
#         if 'connections' in raw_data:
#             connections = raw_data['connections'][0]
#             analysis_results["active_connections"] = connections['isConnected']
        
#         # Generate insights based on analysis
#         summary = []
        
#         if analysis_results.get("active_connections") is True:
#             summary.append(f"Mqtt is active with {analysis_results['active_connections']} live connections.")
#         else:
#             summary.append("Mqtt is not connected or has no active connections.")
        
#         if analysis_results.get("sensor_types"):
#             summary.append(f"Available sensor data includes: {', '.join(analysis_results['sensor_types'])}.")
        
#         analysis_results["summary"] = summary
#         # analysis_results["status"] = "completed"
        
#         return analysis_results

class MonitoringAgent:
    """
    Monitoring Agent for real-time manufacturing data retrieval and analysis.
    
    This agent connects to the SINDIT Knowledge Graph API to retrieve real-time
    manufacturing data and provides insights about factory operations, sensor status,
    and equipment monitoring.
    """
    
    def __init__(self):
        """Initialize the Monitoring Agent with necessary tools and LLM."""
        
        # Initialize tools
        self.sindit_data_tool = SINDITDataTool()
        # self.status_analyzer = ManufacturingStatusAnalyzer()
        
        # Initialize chat model
        self.llm = ChatOpenAI(model_name=CHAT_MODEL, temperature=TEMPERATURE)
        
        # Initialize conversation chain
        self.initialize_conversation_chain()
    
    def initialize_conversation_chain(self):
        """Initialize the conversation chain for the monitoring agent."""
        
        self.system_msg_template = SystemMessagePromptTemplate.from_template(
            template="""
            You are a specialized Monitoring Agent for the fischertechnik Training Factory Industry 4.0 system.
            Your expertise lies in real-time data monitoring, sensor analysis, and manufacturing status assessment.

            Your capabilities include:
            - Retrieving real-time sensor data from the SINDIT Knowledge Graph
            - Analyzing manufacturing equipment status (VGR, HBW, SLD, MPO, DSO, DSI)
                - Manufacturing equipment status (VGR, HBW, SLD, MPO, DSO, DSI)
                - VGR: Vacuum Gripper Robot
                - HBW: High Bay Warehouse
                - SLD: Sorting Line with Detection
                - MPO: Multi Processing Station with Oven
                - DSI/DSO: Sensor for incoming/outgoing pieces
            - Monitoring environmental conditions (temperature, humidity, air quality, brightness)
            - Assessing data connection health and streaming properties
            - Providing insights on factory operational status

            When answering questions about manufacturing status or sensor data:
            1. Always retrieve the latest real-time data first
            2. Analyze the data for patterns, anomalies, or status indicators
            3. Provide related information about factory operations
            4. Reference specific sensor readings and asset statuses when available

            Be technical but accessible, focusing on operational insights that help with:
            - Production monitoring and control
            - Preventive maintenance planning
            - Environmental condition assessment
            - System health and connectivity status
            - Performance optimization opportunities

            If real-time data is unavailable or incomplete, clearly indicate what information is missing
            and suggest alternative monitoring approaches.
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
    
    def get_realtime_context(self, query: str) -> Dict[str, Any]:
        """
        Get real-time monitoring context.
        
        Args:
            query: User's question
        
        Returns:
            dict: Real-time monitoring context
        """
        
        # Get real-time data
        realtime_data = self.sindit_data_tool.run(query)
        
        return {
            "realtime_data": realtime_data,
            "source": "Monitoringgent"
        }
    
    # def analyze_manufacturing_status(self, query: str, asset_type: str = None, property_type: str = None) -> Dict[str, Any]:
    #     """
    #     Analyze current manufacturing status and provide insights.
        
    #     Args:
    #         query: Natural language query about manufacturing status
    #         asset_type: Specific asset type to analyze (optional)
    #         property_type: Specific property type to analyze (optional)
        
    #     Returns:
    #         dict: Analysis results and insights
    #     """
    #     return self.status_analyzer.run(query)
    
    def query(self, user_query: str) -> str:
        """
        Process a user query about manufacturing monitoring.
        
        Args:
            user_query: User's question about manufacturing status or real-time data
        
        Returns:
            str: Comprehensive response with real-time data analysis
        """
        # Retrieve real-time data relevant to the query by using the SINDIT data tool
        real_time_context = self.get_realtime_context(user_query)
        real_time_data = real_time_context.get("realtime_data", {})
        
        # # Then analyze the manufacturing status by using the status analyzer tool
        # status_analysis = self.analyze_manufacturing_status(user_query)
                
        # Format the input for the conversation chain
        formatted_input = f"""
        User Query: {user_query}
        
        Real-time Data Retrieved:
        {json.dumps(real_time_data, indent=2)}
        
        Please provide a comprehensive response based on this real-time manufacturing data.
        """
        
        # Generate response using the conversation chain
        response = self.conversation_chain.run(input=formatted_input)
        
        return response
    
    def get_sindit_status(self) -> Dict[str, Any]:
        """
        Retrieve connection info of MQTT connections from SINDIT Knowledge Graph API.
        
        Returns:
            dict: Real-time data from SINDIT API
        """

        return query_get_connections_info('kg/node', "http://sindit.sintef.no/2.0#mqtt-connection")
        