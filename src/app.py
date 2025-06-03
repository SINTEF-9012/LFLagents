import streamlit as st
import os
import sys
from datetime import datetime
import json

# Add the src directory to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import agents
from agents.integrated_agent import IntegratedManufacturingAgent
from agents.monitoring_agent import MonitoringAgent
from agents.retriever_agent import RetrieverAgent


# Page configuration
st.set_page_config(
    page_title="Manufacturing Assistant - fischertechnik Training Factory",
    page_icon="🏭",
    layout="wide",
    initial_sidebar_state="expanded"
)

def initialize_session_state():
    """Initialize session state variables."""
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    
    if 'current_agent' not in st.session_state:
        st.session_state.current_agent = "integrated"
    
    if 'agent_instances' not in st.session_state:
        st.session_state.agent_instances = {}

def get_agent_instance(agent_type):
    """Get or create an agent instance."""
    if agent_type not in st.session_state.agent_instances:
        try:
            if agent_type == "integrated":
                st.session_state.agent_instances[agent_type] = IntegratedManufacturingAgent()
            elif agent_type == "monitoring":
                st.session_state.agent_instances[agent_type] = MonitoringAgent()
            elif agent_type == "retriever":
                st.session_state.agent_instances[agent_type] = RetrieverAgent()
        except Exception as e:
            st.error(f"Error initializing {agent_type} agent: {str(e)}")
            return None
    
    return st.session_state.agent_instances.get(agent_type)

def display_chat_history():
    """Display chat history."""
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            if message["role"] == "assistant":
                st.markdown(message["content"])
                if "metadata" in message:
                    with st.expander("Response Details"):
                        st.json(message["metadata"])
            else:
                st.markdown(message["content"])

def display_system_status():
    """Display current system status."""
    st.subheader("🌐 Connection Status")
    
    try:
        integrated_agent = get_agent_instance("integrated")
        if integrated_agent:
            status = integrated_agent.get_system_status()
            
            col1, col2 = st.columns(2)
                    
            with col1:
                monitoring_status = status.get("system_components", {}).get("monitoring", {}).get("isConnected")
                
                # Color-coded status
                if monitoring_status:
                    st.success("🟢 MQTT ONLINE")
                else:
                    st.error("🔴 MQTT OFFLINE")
            
            with col2:
                doc_status = status.get("system_components", {}).get("documentation", {})
                all_states = [index_info.get('state', 'unknown') for index_info in doc_status.values()]
                
                if all(state == 'ONLINE' for state in all_states):
                    st.success("🟢 Neo4j ONLINE")
                elif any(state == 'OFFLINE' for state in all_states):
                    st.error("🔴 Neo4j OFFLINE")
                else:
                    st.warning("🟡 Neo4j UNKNOW")
                
            # Detailed status
            with st.expander("Detailed System Information"):
                st.json(status)
        else:
            st.error("Unable to get system status")
    
    except Exception as e:
        st.error(f"Error getting system status: {str(e)}")

def display_quick_actions():
    """Display enhanced quick action buttons."""
    st.subheader("🚀 Quick Actions")
    
    # First row - Monitoring
    st.markdown("**🔍 Monitoring**")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("🌡️ Temperature", use_container_width=True, help="Check current temperature and humidity"):
            query = "What is the current temperature and humidity in the factory?"
            process_query(query)
    
    with col2:
        if st.button("📊 System Overview", use_container_width=True, help="Get comprehensive system status"):
            query = "Provide a comprehensive overview of all manufacturing systems and their current status"
            process_query(query)
    
    with col3:
        if st.button("⚙️ Components", use_container_width=True, help="Check all component statuses"):
            query = "Check the status of all manufacturing components including VGR, HBW, SLD, MPO, DSO, and DSI"
            process_query(query)
    
    # Second row - Production
    st.markdown("**🏭 Production**")
    col4, col5, col6 = st.columns(3)
    
    with col4:
        if st.button("📈 Production Status", use_container_width=True, help="Current production progress"):
            query = "What is the current production status and order progress?"
            process_query(query)
    
    with col5:
        if st.button("🔧 Equipment Health", use_container_width=True, help="Check for maintenance alerts"):
            query = "Check for any equipment issues or maintenance alerts across all manufacturing stations"
            process_query(query)
    
    with col6:
        if st.button("🌍 Environment", use_container_width=True, help="Environmental sensor readings"):
            query = "Show all environmental sensor readings including air quality, brightness, and pressure"
            process_query(query)
    
    # Component-specific queries
    st.markdown("**🎯 Component Analysis**")
    component = st.selectbox(
        "Select a component:",
        ["Select...", "VGR (Vacuum Gripper Robot)", "HBW (High Bay Warehouse)", 
         "SLD (Sorting Line with Color Detection)", "MPO (Multi Processing with Oven)", 
         "DSI (Input Sensor)", "DSO (Output Sensor)", "Camera System"],
        help="Get detailed status for a specific component"
    )
    
    if component != "Select...":
        component_queries = {
            "VGR (Vacuum Gripper Robot)": "What is the current status of the vacuum gripper robot?",
            "HBW (High Bay Warehouse)": "Show the current status of the High Bay Warehouse including storage capacity and operations",
            "SLD (Sorting Line with Color Detection)": "What is the current status of the Sorting Line with Detection system?",
            "MPO (Multi Processing with Oven)": "Show the Multi Processing Oven status including temperature and current operations",
            "DSI (Input Sensor)": "What is the status of the DSI input sensor and piece detection?",
            "DSO (Output Sensor)": "What is the status of the DSO output sensor and piece tracking?",
            "Camera System": "Show camera system status including position and captured data"
        }
        
        if st.button(f"📋 Get {component} Status", use_container_width=True):
            process_query(component_queries[component])

def process_query(query):
    """Process a user query with the selected agent."""
    agent = get_agent_instance(st.session_state.current_agent)
    
    if not agent:
        st.error(f"Agent '{st.session_state.current_agent}' not available")
        return
    
    # Add user message to chat history
    st.session_state.chat_history.append({
        "role": "user",
        "content": query,
        "timestamp": datetime.now().isoformat()
    })
    
    # Process the query with progress indication
    progress_bar = st.progress(0, text="Processing query...")
    
    try:
        progress_bar.progress(25, text="Initializing...")
        
        if st.session_state.current_agent == "integrated":
            progress_bar.progress(50, text="Retrieving data...")
            response = agent.query(query)
            metadata = {
                "agent_type": "integrated",
                "timestamp": datetime.now().isoformat(),
                "query_classification": agent.classify_query_type(query)
            }
        elif st.session_state.current_agent == "monitoring":
            progress_bar.progress(50, text="Getting real-time data...")
            response = agent.query(query)
            metadata = {
                "agent_type": "MonitoringAgent",
                "connection_status": agent.get_sindit_status(),
            }
        elif st.session_state.current_agent == "retriever":
            progress_bar.progress(50, text="Searching documentation...")
            response = agent.query(query)
            metadata = {
                "agent_type": "RetrieverAgent",
                "connection_status": agent.get_neo4j_status(),
            }
        
        progress_bar.progress(100, text="Complete!")
        
        # Add assistant response to chat history
        st.session_state.chat_history.append({
            "role": "assistant",
            "content": response,
            "timestamp": datetime.now().isoformat(),
            "metadata": metadata
        })
        
    except Exception as e:
        error_message = f"❌ Error processing query: {str(e)}"
        st.session_state.chat_history.append({
            "role": "assistant",
            "content": error_message,
            "timestamp": datetime.now().isoformat(),
            "metadata": {"error": True, "error_type": type(e).__name__}
        })
    finally:
        progress_bar.empty()
        
def main():
    """Main application function."""
    
    # Initialize session state
    initialize_session_state()
    
    # Header
    st.title("Manufacturing Assistant")
    st.markdown("**fischertechnik Training Factory Industry 4.0**")
    
    # Sidebar
    with st.sidebar:
        st.header("Configuration")
        
        # Agent selection
        agent_options = {
            "integrated": "Integrated Agent",
            "monitoring": "Monitoring Agent",
            "retriever": "Documentation Agent"
        }
        
        selected_agent = st.selectbox(
            "Select Agent",
            options=list(agent_options.keys()),
            format_func=lambda x: agent_options[x],
            index=0
        )
        st.session_state.current_agent = selected_agent
        
        # Agent descriptions
        st.markdown("### Agent Capabilities")
        
        if selected_agent == "integrated":
            st.info("""
            **Integrated Agent**
            - Real-time monitoring data
            - User manuals
            - Comprehensive insights
            - Best for most queries
            """)
        elif selected_agent == "monitoring":
            st.info("""
            **Monitoring Agent**
            - Real-time sensor data
            - Equipment status
            - Performance metrics
            - Manufacturing insights
            """)
        elif selected_agent == "retriever":
            st.info("""
            **Documentation Agent**
            - User manuals
            - Setup procedures
            - Technical documentation
            - Historical knowledge
            """)
        
        # Clear chat history
        if st.button("Clear Chat History"):
            st.session_state.chat_history = []
            st.rerun()
        
        # Export chat history
        if st.session_state.chat_history and st.button("💾 Export Chat"):
            chat_data = json.dumps(st.session_state.chat_history, indent=2)
            st.download_button(
                "Download Chat History",
                chat_data,
                file_name=f"chat_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json"
            )
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Chat interface
        st.subheader("💬 Chat Interface")
        
        # Display chat history
        chat_container = st.container()
        with chat_container:
            display_chat_history()
        
        # Chat input with suggestions
        placeholder_text = {
            "integrated": "Ask about manufacturing status, procedures, or get help...",
            "monitoring": "Ask about current sensor data, equipment status...",
            "retriever": "Ask about setup procedures, documentation..."
        }
        
        if prompt := st.chat_input(placeholder_text.get(st.session_state.current_agent, "How can I help you?")):
            process_query(prompt)
            st.rerun()
        
        # Example queries
        with st.expander("💡 Example Queries"):
            examples = {
                "integrated": [
                    "What's the current factory status?",
                    "How do I set up the MQTT connection?",
                ],
                "monitoring": [
                    "Current temperature readings?",
                    "VGR status?",
                    "Equipment health check"
                ],
                "retriever": [
                    "How to configure TIA Portal?",
                    "Troubleshooting guide",
                    "System architecture overview"
                ]
            }
            
            for example in examples.get(st.session_state.current_agent, []):
                if st.button(f"💭 {example}", key=f"example_{example}"):
                    process_query(example)
                    st.rerun()
    
    with col2:
        
        # Display current system status
        display_system_status()
        
        st.markdown("---")
        
        # Display quick actions
        display_quick_actions()
        
        # Manufacturing insights
        if st.button("🔮 Get Manufacturing Insights", use_container_width=True):
            integrated_agent = get_agent_instance("integrated")
            if integrated_agent:
                with st.spinner("Generating insights..."):
                    try:
                        insights = integrated_agent.get_manufacturing_insights()
                        st.session_state.chat_history.append({
                            "role": "assistant",
                            "content": f"**🔮 Manufacturing Insights:**\n\n{insights}",
                            "timestamp": datetime.now().isoformat(),
                            "metadata": {"type": "insights", "generated": True}
                        })
                        st.rerun()
                    except Exception as e:
                        st.error(f"Error generating insights: {str(e)}")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666;'>
        <small>Manufacturing Assistant for fischertechnik Training Factory Industry 4.0<br>
        Powered by SINDIT Knowledge Graph & Neo4j Documentation System</small>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()