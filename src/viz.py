"""
Factory Data Visualization Dashboard

A Streamlit app for visualizing factory component data with interactive charts.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import sys
import os

# Add the src directory to the path to import utilities
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from utils.ts_preprocessing import (
    load_resampled_requested_components,
    FACTORY_COMPONENT_CONFIG
)

# Set page configuration
st.set_page_config(
    page_title="Factory Data Dashboard",
    page_icon="🏭",
    layout="wide"
)

def visualize_component_activity(component_data, component_name, resample_interval='5S'):
    """
    Visualize component activity with position data, status highlighting, and target movements
    
    Parameters:
    component_data: DataFrame with component data
    component_name: str, name of the component for titles
    resample_interval: str, resampling interval for shaded regions
    """
    
    if component_data is None or component_data.empty:
        st.warning(f"No data available for {component_name}")
        return None
    
    # Define position columns mapping for different components
    position_mappings = {
        'VGR': {
            'horizontal': 'actual_pos_horizontal',
            'vertical': 'actual_pos_vertical', 
            'rotation': 'actual_pos_rotation'
        },
        'HBW': {
            'horizontal': 'actual_pos_horizontal',
            'vertical': 'actual_pos_vertical'
        },
        'SSC': {
            'horizontal': 'actual_pos_horizontal',
            'vertical': 'actual_pos_vertical'
        },
        'CAMERA': {
            'pan': 'relative_position_pan',
            'tilt': 'relative_position_tilt'
        }
    }
    
    # Status columns mapping
    status_mappings = {
        'VGR': 'status_code',
        'HBW': 'status_code', 
        'MPO': 'status_code',
        'SLD': 'status_code',
        'DPS': ['dsi_status_code', 'dso_status_code']  # DPS has multiple status codes
    }
    
    # Target/movement columns
    target_mappings = {
        'VGR': 'target_status',
        'HBW': None,  # Add if HBW has target data
        'SSC': None
    }
    
    fig = go.Figure()
    position_traces = []
    
    # Get position columns for this component
    pos_cols = position_mappings.get(component_name, {})
    
    # Add position traces if available
    for pos_name, col_name in pos_cols.items():
        if col_name in component_data.columns:
            fig.add_trace(go.Scatter(
                x=component_data['time'], 
                y=component_data[col_name], 
                mode='lines', 
                name=f'{pos_name.title()} Position',
                line=dict(width=2)
            ))
            position_traces.append(col_name)
    
    # If no position data, plot only status codes
    if not position_traces:
        # Add status codes for components like DPS
        status_cols = status_mappings.get(component_name, [])
        if isinstance(status_cols, list):
            for status_col in status_cols:
                if status_col in component_data.columns:
                    fig.add_trace(go.Scatter(
                        x=component_data['time'], 
                        y=component_data[status_col], 
                        mode='lines', 
                        name=status_col.replace('_', ' ').title(),
                        line=dict(width=2)
                    ))
                    position_traces.append(status_col)
        elif isinstance(status_cols, str) and status_cols in component_data.columns:
            fig.add_trace(go.Scatter(
                x=component_data['time'], 
                y=component_data[status_cols], 
                mode='lines', 
                name='Status Code',
                line=dict(width=2)
            ))
            position_traces.append(status_cols)
    
    if not position_traces:
        st.warning(f"No suitable data columns found for visualization of {component_name}")
        return None
    
    # Find y-axis range for annotations and shading
    y_values = []
    for col in position_traces:
        if col in component_data.columns:
            y_values.extend(component_data[col].dropna().tolist())
    
    if y_values:
        y_min = min(y_values)
        y_max = max(y_values)
    else:
        y_min, y_max = 0, 1
    
    # Add non-idle time highlighting
    status_col = status_mappings.get(component_name, 'status_code')
    if isinstance(status_col, str) and status_col in component_data.columns:
        # Single status column
        highlighted_times = component_data[component_data[status_col] != 1]['time']
        
        for time_point in highlighted_times:
            fig.add_vrect(
                x0=time_point, 
                x1=time_point + pd.Timedelta(seconds=int(resample_interval[:-1])),
                fillcolor="red", opacity=0.2,
                layer="below", line_width=0,
            )
    
    elif isinstance(status_col, list):
        # Multiple status columns (like DPS)
        for col in status_col:
            if col in component_data.columns:
                highlighted_times = component_data[component_data[col] != 1]['time']
                color = "red" if "dsi" in col else "orange"
                
                for time_point in highlighted_times:
                    fig.add_vrect(
                        x0=time_point, 
                        x1=time_point + pd.Timedelta(seconds=int(resample_interval[:-1])),
                        fillcolor=color, opacity=0.15,
                        layer="below", line_width=0,
                    )
    
    # Add target movement annotations (mainly for VGR)
    target_col = target_mappings.get(component_name)
    if target_col and target_col in component_data.columns:
        target_movements = component_data[component_data[target_col].notna()].copy()
        
        if not target_movements.empty:
            for idx, row in target_movements.iterrows():
                target = row[target_col]
                time_point = row['time']
                
                # Check for start of movement
                prev_idx = idx - 1 if idx > 0 else None
                if (prev_idx is None or 
                    pd.isna(component_data.loc[prev_idx, target_col]) or 
                    component_data.loc[prev_idx, target_col] != target):
                    
                    fig.add_annotation(
                        x=time_point, y=y_max * 0.9,
                        text=f"→ {str(target).upper()}",
                        showarrow=True, arrowhead=2,
                        arrowcolor="green", bgcolor="lightgreen",
                        bordercolor="green", borderwidth=1,
                        font=dict(size=9, color="darkgreen")
                    )
                
                # Check for end of movement
                next_idx = idx + 1 if idx < len(component_data) - 1 else None
                if (next_idx is None or 
                    pd.isna(component_data.loc[next_idx, target_col]) or 
                    component_data.loc[next_idx, target_col] != target):
                    
                    fig.add_annotation(
                        x=time_point, y=y_max * 0.8,
                        text=f"✓ {str(target).upper()}",
                        showarrow=True, arrowhead=2,
                        arrowcolor="red", bgcolor="lightcoral",
                        bordercolor="red", borderwidth=1,
                        font=dict(size=9, color="darkred")
                    )
    
    # Add legend for non-idle periods
    fig.add_trace(go.Scatter(
        x=[None], y=[None], mode='markers',
        marker=dict(color='red', size=10), 
        name='Non-Idle Periods', showlegend=True
    ))
    
    # Update layout
    fig.update_layout(
        title=f'{component_name} Activity Analysis - Position and Status Over Time',
        xaxis_title='Time',
        yaxis_title='Value',
        height=600,
        hovermode='x unified',
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    
    return fig

def visualize_environmental_data(env_data):
    """Create environmental sensors visualization"""
    if env_data is None or env_data.empty:
        st.warning("No environmental data available")
        return None
    
    # Create subplots for all environmental parameters
    fig = make_subplots(
        rows=5, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.02,
        subplot_titles=[
            'Air Pressure (hPa)',
            'Air Quality Index',
            'Gas Resistance (Ω)',
            'Humidity (%)',
            'Temperature (°C)'
        ]
    )
    
    # Environmental sensors mapping
    env_sensors = {
        'factory_air_pressure': {'row': 1, 'color': 'blue', 'name': 'Air Pressure'},
        'factory_index_air_quality': {'row': 2, 'color': 'orange', 'name': 'Air Quality Index'},
        'factory_gas_resistance': {'row': 3, 'color': 'green', 'name': 'Gas Resistance'},
        'factory_humidity_raw': {'row': 4, 'color': 'cyan', 'name': 'Humidity'},
        'factory_temperature_raw': {'row': 5, 'color': 'red', 'name': 'Temperature'}
    }
    
    for sensor, config in env_sensors.items():
        if sensor in env_data.columns:
            fig.add_trace(
                go.Scatter(
                    x=env_data['time'],
                    y=env_data[sensor],
                    mode='lines+markers',
                    name=config['name'],
                    line=dict(color=config['color'], width=2),
                    marker=dict(size=4)
                ),
                row=config['row'], col=1
            )
    
    # Update layout
    fig.update_layout(
        title='Environmental Sensors - Time Series Analysis',
        height=800,
        showlegend=False,
        hovermode='x unified'
    )
    
    # Update axes
    fig.update_xaxes(title_text='Time', row=5, col=1)
    fig.update_yaxes(title_text='hPa', row=1, col=1)
    fig.update_yaxes(title_text='Index', row=2, col=1)
    fig.update_yaxes(title_text='Ω', row=3, col=1)
    fig.update_yaxes(title_text='%', row=4, col=1)
    fig.update_yaxes(title_text='°C', row=5, col=1)
    
    return fig

def main():
    st.title("🏭 Factory Data Visualization Dashboard")
    st.markdown("Interactive visualization of factory component data and environmental sensors")
    
    # Sidebar for configuration
    st.sidebar.header("Configuration")
    
    # Time range selection
    st.sidebar.subheader("Time Range")
    default_start_date = datetime(2022, 12, 2).date()
    default_start_time = datetime(2022, 12, 2, 14, 0, 0).time()
    default_end_date = datetime(2022, 12, 2).date()
    default_end_time = datetime(2022, 12, 2, 15, 0, 0).time()
    
    # Start datetime
    start_date = st.sidebar.date_input("Start Date", value=default_start_date)
    start_time_input = st.sidebar.time_input("Start Time", value=default_start_time)
    start_time = datetime.combine(start_date, start_time_input)
    
    # End datetime
    end_date = st.sidebar.date_input("End Date", value=default_end_date)
    end_time_input = st.sidebar.time_input("End Time", value=default_end_time)
    end_time = datetime.combine(end_date, end_time_input)
    
    # Resample interval
    resample_interval = st.sidebar.selectbox(
        "Resample Interval",
        options=['1S', '5S', '10S', '30S', '1T', '5T'],
        index=1
    )
    
    # Component selection
    st.sidebar.subheader("Components")
    available_components = list(FACTORY_COMPONENT_CONFIG.keys())
    selected_components = st.sidebar.multiselect(
        "Select Components to Analyze",
        options=available_components,
        default=['VGR', 'SLD', 'ENVIRONMENTAL']
    )
    
    if not selected_components:
        st.warning("Please select at least one component to analyze.")
        return
    
    # Load data button
    if st.sidebar.button("Load Data", type="primary"):
        with st.spinner("Loading and processing data..."):
            # Create time range as [start, end] list
            time_range = [start_time, end_time]
            
            # Load data
            try:
                data = load_resampled_requested_components(
                    selected_components,
                    time_range,
                    resample_interval=resample_interval
                )
                
                if data:
                    st.session_state['data'] = data
                    st.session_state['time_range'] = time_range
                    st.session_state['resample_interval'] = resample_interval
                    st.success(f"Successfully loaded data for {len(data)} components!")
                else:
                    st.error("No data could be loaded. Please check your time range and component selection.")
                    
            except Exception as e:
                st.error(f"Error loading data: {str(e)}")
    
    # Display data if available
    if 'data' in st.session_state:
        data = st.session_state['data']
        resample_interval = st.session_state.get('resample_interval', '5S')
        
        st.header("Component Analysis")
        
        # Create tabs for each component
        tab_names = []
        for component in data.keys():
            description = FACTORY_COMPONENT_CONFIG[component]['description']
            tab_names.append(f"{component} - {description}")
        
        tabs = st.tabs(tab_names)
        
        for i, (component, component_data) in enumerate(data.items()):
            with tabs[i]:
                st.subheader(f"{component} - {FACTORY_COMPONENT_CONFIG[component]['description']}")
                
                # Display data info
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Records", len(component_data))
                with col2:
                    st.metric("Columns", len(component_data.columns))
                with col3:
                    time_span = component_data['time'].max() - component_data['time'].min()
                    st.metric("Time Span", f"{time_span.total_seconds():.0f}s")
                
                # Show data preview
                with st.expander("Data Preview"):
                    st.dataframe(component_data.head())
                
                # Create and display visualization
                if component == 'ENVIRONMENTAL':
                    fig = visualize_environmental_data(component_data)
                else:
                    fig = visualize_component_activity(
                        component_data, 
                        component, 
                        resample_interval
                    )
                
                if fig:
                    st.plotly_chart(fig, use_container_width=True)
                
                # Display statistics
                st.subheader("Statistics")
                numeric_cols = component_data.select_dtypes(include=[np.number]).columns
                if len(numeric_cols) > 1:  # Exclude time column
                    stats_df = component_data[numeric_cols].describe()
                    st.dataframe(stats_df)
    
    else:
        st.info("Configure settings in the sidebar and click 'Load Data' to begin analysis.")

if __name__ == "__main__":
    main()
