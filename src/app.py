import streamlit as st
import os
import sys
from datetime import datetime, timedelta
import json

# Add the src directory to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import agents
from agents.integrated_agent import IntegratedManufacturingAgent
from agents.monitoring_agent import MonitoringAgent
from agents.retriever_agent import RetrieverAgent
from agents.analytics_agent import AnalyticsAgent


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
    
    if 'selected_viz_component' not in st.session_state:
        st.session_state.selected_viz_component = "FACTORY"
    
    if 'selected_time_range' not in st.session_state:
        # Default to factory data time range (2022-12-02 14:00 to 14:30)
        start_time = datetime(2022, 12, 2, 14, 0, 0)
        end_time = datetime(2022, 12, 2, 14, 30, 0)
        st.session_state.selected_time_range = (start_time, end_time)

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
            elif agent_type == "analytics":
                st.session_state.agent_instances[agent_type] = AnalyticsAgent()
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
    
    start_time = datetime.now()
    # Process the query with progress indication
    progress_bar = st.progress(0, text="Processing query...")
    
    try:
        progress_bar.progress(25, text="Initializing...")
        
        if st.session_state.current_agent == "integrated":
            progress_bar.progress(50, text="Retrieving data...")
            response = agent.query(query)
            response_time = (datetime.now() - start_time).total_seconds()
            metadata = {
                "agent_type": "integrated",
                "timestamp": datetime.now().isoformat(),
                "query_classification": agent.classify_query_type(query),
                "response_time_sec": response_time
            }
        elif st.session_state.current_agent == "monitoring":
            progress_bar.progress(50, text="Getting real-time data...")
            response = agent.query(query)
            response_time = (datetime.now() - start_time).total_seconds()
            metadata = {
                "agent_type": "MonitoringAgent",
                "connection_status": agent.get_sindit_status(),
                "response_time_sec": response_time
            }
        elif st.session_state.current_agent == "retriever":
            progress_bar.progress(50, text="Searching documentation...")
            response = agent.query(query)
            response_time = (datetime.now() - start_time).total_seconds()
            metadata = {
                "agent_type": "RetrieverAgent",
                "connection_status": agent.get_neo4j_status(),
                "response_time_sec": response_time
            }
        elif st.session_state.current_agent == "analytics":
            progress_bar.progress(50, text="Analyzing factory data...")
            response = agent.query(query)
            response_time = (datetime.now() - start_time).total_seconds()
            metadata = {
                "agent_type": "AnalyticsAgent",
                "response_time_sec": response_time
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
        
def display_analytics_dashboard():
    """Display the analytics dashboard with visualizations from the viz.py module."""
        
    # Initialize the analytics agent
    analytics_agent = get_agent_instance("analytics")
    if not analytics_agent:
        st.error("Analytics Agent not available")
        return
    
    # Get the selected component from session state
    selected_component = st.session_state.selected_viz_component
    
    # Get resample interval from session state (default to 5S if not set)
    resample_interval = st.session_state.get('resample_interval', '5S')
    
    # Main visualization area
    if selected_component:
        st.subheader(f"{selected_component} - {analytics_agent.get_component_description(selected_component)}")
        
        # Show metrics at the top
        try:
            metrics = analytics_agent.get_component_metrics(selected_component)
            
            if metrics and not isinstance(metrics, dict) or "error" in metrics:
                st.warning("No metrics available. Please load data first.")
            else:
                # Display metrics in a nice format
                if selected_component == "FACTORY":
                    # For factory overview, show summary metrics
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.metric("Avg Activity", f"{metrics.get('avg_activity_pct', 0)}%")
                    
                    with col2:
                        st.metric("Active Time", f"{metrics.get('active_time_minutes', 0)} min")
                    
                    with col3:
                        st.metric("VGR Targets", metrics.get("vgr_target_count", 0))
                    
                    with col4:
                        # Find the component with highest activity
                        activity_metrics = {k: v for k, v in metrics.items() if k.endswith('_active_pct') and not k.startswith('avg')}
                        if activity_metrics:
                            max_component = max(activity_metrics.items(), key=lambda x: x[1])
                            component_name = max_component[0].replace('_active_pct', '')
                            st.metric("Most Active", f"{component_name} ({max_component[1]}%)")
                
                elif selected_component == "ENVIRONMENTAL":
                    # For environmental data, show temperature, humidity, etc.
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        if "factory_temperature_raw_avg" in metrics:
                            st.metric("Temperature", f"{metrics['factory_temperature_raw_avg']}°C")
                    
                    with col2:
                        if "factory_humidity_raw_avg" in metrics:
                            st.metric("Humidity", f"{metrics['factory_humidity_raw_avg']}%")
                    
                    with col3:
                        if "factory_index_air_quality_avg" in metrics:
                            st.metric("Air Quality", metrics['factory_index_air_quality_avg'])
                    
                    with col4:
                        if "factory_air_pressure_avg" in metrics:
                            st.metric("Air Pressure", f"{metrics['factory_air_pressure_avg']} hPa")
                
                else:
                    # For other components, show status and positions
                    status_metrics = {k: v for k, v in metrics.items() if '_active_pct' in k}
                    
                    cols = st.columns(len(status_metrics) if status_metrics else 4)
                    
                    if status_metrics:
                        for i, (key, value) in enumerate(status_metrics.items()):
                            with cols[i % len(cols)]:
                                name = key.replace('_active_pct', '').replace('_', ' ').title()
                                st.metric(f"{name} Activity", f"{value}%")
        
        except Exception as e:
            st.error(f"Error displaying metrics: {str(e)}")
        
        # Import visualization functions directly from viz.py
        try:
            # Check if we have data loaded in the session state
            if 'component_data' in st.session_state and selected_component in st.session_state.component_data:
                component_data = st.session_state.component_data[selected_component]
                
                # Import visualization functions here to avoid circular imports
                import sys
                import os
                sys.path.append(os.path.dirname(os.path.abspath(__file__)))
                from viz import visualize_component_activity, visualize_factory_overview, visualize_environmental_data
                
                # Create appropriate visualization based on component type
                if selected_component == 'FACTORY':
                    fig = visualize_factory_overview(component_data, resample_interval)
                elif selected_component == 'ENVIRONMENTAL':
                    fig = visualize_environmental_data(component_data)
                else:
                    fig = visualize_component_activity(component_data, selected_component, resample_interval)
                
                if fig:
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.warning("No visualization available for the selected data.")
            else:
                st.info("Please select a component and load data to view visualizations.")
        except ImportError as e:
            st.error(f"Error importing visualization functions: {str(e)}")
            st.info("Please ensure viz.py is in the correct location and all dependencies are installed.")
        except Exception as e:
            st.error(f"Error generating visualization: {str(e)}")
            st.exception(e)
        
        # Display raw data and insights
        col1, col2 = st.columns(2)
        
        with col1:
            with st.expander("Insights"):
                try:
                    insights = analytics_agent.get_anomaly_insights(selected_component)
                    if insights:
                        for insight in insights:
                            st.markdown(f"**{insight['type'].replace('_', ' ').title()}**")
                            st.markdown(f"{insight['description']}")
                            if 'count' in insight:
                                st.markdown(f"Occurrences: {insight['count']}")
                            if 'duration_seconds' in insight:
                                st.markdown(f"Duration: {insight['duration_seconds']} seconds")
                            st.markdown("---")
                    else:
                        st.info("No significant insights detected in the current data.")
                except Exception as e:
                    st.warning(f"Could not generate insights: {str(e)}")
        
        with col2:
            with st.expander("Raw Data"):
                if 'component_data' in st.session_state and selected_component in st.session_state.component_data:
                    st.dataframe(st.session_state.component_data[selected_component].head(100))
                    
                    # Download option
                    csv = st.session_state.component_data[selected_component].to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label="Download CSV",
                        data=csv,
                        file_name=f"{selected_component.lower()}_data.csv",
                        mime='text/csv'
                    )
                else:
                    st.info("No data loaded. Use the sidebar controls to load data first.")

def main():
    """Main application function."""
    
    # Initialize session state
    initialize_session_state()
    
    # Header
    st.title("🏭 Manufacturing Assistant")
    st.markdown("**fischertechnik Training Factory Industry 4.0**")
    
    # Sidebar
    with st.sidebar:
        st.header("Configuration")
        
        # Agent selection
        agent_options = {
            "integrated": "Integrated Agent",
            "monitoring": "Monitoring Agent",
            "retriever": "Documentation Agent",
            "analytics": "Analytics Agent"
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
        elif selected_agent == "analytics":
            st.info("""
            **Analytics Agent**
            - Data visualization
            - Performance metrics
            - Historical trends
            - Component analysis
            - Anomaly detection
            """)
            
            # Add dashboard toggle for analytics agent
            st.session_state.show_dashboard = st.checkbox(
                "Show Analytics Dashboard", 
                value=st.session_state.get('show_dashboard', False),
                help="Toggle between chat interface and visual dashboard"
            )
            
            if not st.session_state.get('show_dashboard', False):
                st.info("💡 Check 'Show Analytics Dashboard' above to view interactive visualizations!")
            
            # Dashboard Controls (when dashboard is enabled)
            if st.session_state.get('show_dashboard', False):
                st.markdown("---")
                st.header("Dashboard Controls")
                
                # Get analytics agent for dashboard controls
                analytics_agent = get_agent_instance("analytics")
                if analytics_agent:
                    # Get available components for visualization
                    available_components = analytics_agent.get_available_components()
                    
                    # Component selection
                    selected_component = st.selectbox(
                        "Select Component",
                        available_components,
                        index=available_components.index(st.session_state.selected_viz_component) if st.session_state.selected_viz_component in available_components else 0,
                        format_func=lambda x: f"{x} - {analytics_agent.get_component_description(x)}"
                    )
                    st.session_state.selected_viz_component = selected_component
                    
                    # Time range selection
                    st.subheader("Time Range")
                    
                    # Default to a time range that matches the available data (2022-12-02 14:00 to 14:30)
                    default_start = datetime(2022, 12, 2, 14, 0, 0)  # Start of available data
                    default_end = datetime(2022, 12, 2, 14, 30, 0)  # 30 minutes from start
                    
                    start_time = st.date_input(
                        "Start Date",
                        value=st.session_state.selected_time_range[0].date() if 'selected_time_range' in st.session_state else default_start.date()
                    )
                    
                    start_hour = st.time_input(
                        "Start Time",
                        value=st.session_state.selected_time_range[0].time() if 'selected_time_range' in st.session_state else default_start.time()
                    )
                    
                    end_time = st.date_input(
                        "End Date",
                        value=st.session_state.selected_time_range[1].date() if 'selected_time_range' in st.session_state else default_end.date()
                    )
                    
                    end_hour = st.time_input(
                        "End Time",
                        value=st.session_state.selected_time_range[1].time() if 'selected_time_range' in st.session_state else default_end.time()
                    )
                    
                    # Combine date and time
                    start_datetime = datetime.combine(start_time, start_hour)
                    end_datetime = datetime.combine(end_time, end_hour)
                    
                    # Update session state
                    st.session_state.selected_time_range = (start_datetime, end_datetime)
                    
                    # Resampling interval for data aggregation
                    resample_interval = st.selectbox(
                        "Resampling Interval",
                        ["1S", "5S", "10S", "30S", "1Min", "5Min", "10Min"],
                        index=1  # Default to 5S
                    )
                    st.session_state.resample_interval = resample_interval
                    
                    # Load data button
                    if st.button("Load Data", use_container_width=True):
                        with st.spinner(f"Loading data for {selected_component}..."):
                            try:
                                # Load data using the analytics agent
                                st.session_state.component_data = analytics_agent.load_component_data(
                                    [selected_component], 
                                    (start_datetime, end_datetime),
                                    resample_interval
                                )
                                
                                if st.session_state.component_data and selected_component in st.session_state.component_data:
                                    data_shape = st.session_state.component_data[selected_component].shape
                                    st.success(f"Data loaded successfully! Shape: {data_shape}")
                                else:
                                    st.warning("No data found for the selected component and time range.")
                                    st.info("Try adjusting the time range or selecting a different component.")
                                    
                            except Exception as e:
                                st.error(f"Error loading data: {str(e)}")
                                st.exception(e)
        
        st.markdown("---")
        
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
    # Check if we're in dashboard mode
    if st.session_state.current_agent == "analytics" and st.session_state.get('show_dashboard', False):
        # Display the analytics dashboard in full width
        display_analytics_dashboard()
    else:
        # Regular chat interface
        
        # Display chat history
        display_chat_history()
        
        # Chat input - positioned after main content area
        placeholder_text = {
            "integrated": "Ask about manufacturing status, procedures, or get help...",
            "monitoring": "Ask about current sensor data, equipment status...",
            "retriever": "Ask about setup procedures, documentation...",
            "analytics": "Ask for data analysis, visualizations, or metrics..."
        }
        
        if prompt := st.chat_input(placeholder_text.get(st.session_state.current_agent, "How can I help you?")):
            process_query(prompt)
            st.rerun()
        
        # Example queries section (only show in chat mode)
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
                ],
                "analytics": [
                    "Show VGR performance metrics",
                    "Analyze factory component activity",
                    "How are the environmental conditions in the factory?"
                ]
            }
            
            for example in examples.get(st.session_state.current_agent, []):
                if st.button(f"💭 {example}", key=f"example_{example}"):
                    process_query(example)
                    st.rerun()
    
        # System status section (appears after chat input)
        with st.expander("🔌 System Status"):
            try:
                integrated_agent = get_agent_instance("integrated")
                if integrated_agent:
                    status = integrated_agent.get_system_status()
                    
                    # Create a compact status display
                    status_col1, status_col2 = st.columns(2)
                    
                    with status_col1:
                        st.markdown("**🌐 MQTT Connection**")
                        monitoring_status = status.get("system_components", {}).get("monitoring", {}).get("isConnected")
                        if monitoring_status:
                            st.success("🟢 Online")
                            # Show detailed MQTT info
                            mqtt_info = status.get("system_components", {}).get("monitoring", {})
                            if mqtt_info.get("lastUpdate"):
                                st.caption(f"📅 Last update: {mqtt_info['lastUpdate']}")
                        else:
                            st.error("🔴 Offline")
                    
                    with status_col2:
                        st.markdown("**🗄️ Neo4j Database**")
                        doc_status = status.get("system_components", {}).get("documentation", {})
                        all_states = [index_info.get('state', 'unknown') for index_info in doc_status.values()]
                        
                        if all(state == 'ONLINE' for state in all_states):
                            st.success("🟢 Online")
                        elif any(state == 'OFFLINE' for state in all_states):
                            st.error("🔴 Offline")
                        else:
                            st.warning("🟡 Unknown")
                    
                    # Detailed system information toggle
                    show_details = st.toggle("🔍 Show detailed system info", value=False)
                    if show_details:
                        st.json(status)
                            
            except Exception as e:
                st.error("⚠️ Unable to get system status")
                st.caption(f"Error: {str(e)}")
    
    # Enhanced footer
    st.markdown("---")
    
    # Footer with better styling
    footer_col1, footer_col2, footer_col3 = st.columns([1, 2, 1])
    
    with footer_col2:
        st.markdown("""
        <div style='text-align: center; color: #666; padding: 20px;'>
            <p style='margin: 5px 0; font-size: 14px;'>fischertechnik Training Factory Industry 4.0</p>
            <p style='margin: 5px 0; font-size: 12px;'>Powered by SINDIT Knowledge Graph & Neo4j Documentation System</p>
            <div style='margin-top: 10px; font-size: 12px; color: #888;'>
                Built with Streamlit
            </div>
        </div>
        """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()