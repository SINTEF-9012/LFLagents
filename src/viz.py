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

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

# Add the src directory to the path to import utilities
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from utils.ts_preprocessing import (
    load_resampled_requested_components,
    FACTORY_COMPONENT_CONFIG
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

def visualize_factory_overview(factory_data, resample_interval='5S'):
    """
    Create a specialized factory overview visualization that shows synchronized activity
    across all major factory subsystems (VGR, HBW, MPO, SLD, DSO)
    
    Parameters:
    factory_data: DataFrame with merged data from all factory components
    resample_interval: str, resampling interval for processing
    """
    
    if factory_data is None or factory_data.empty:
        st.warning("No factory overview data available")
        return None
    
    # Define position columns for the overview plot
    position_columns = [
        'vgr_actual_pos_horizontal', 'vgr_actual_pos_rotation', 'vgr_actual_pos_vertical',
        'hbw_actual_pos_horizontal', 'hbw_actual_pos_vertical'
    ]
    
    # Status columns for activity highlighting
    status_columns = ['mpo_status_code', 'sld_status_code', 'hbw_status_code', 'vgr_status_code']
    
    # Create the main line plot for positions
    fig = px.line(
        factory_data, 
        x='time', 
        y=[col for col in position_columns if col in factory_data.columns],
        title='Factory Overview: Synchronized Multi-System Process View',
        labels={'value': 'Position', 'time': 'Time'},
        # markers=True
    )
    
    fig.update_layout(
    legend=dict(
        title_text='',  # Remove legend title
        orientation="h",  # Optional: horizontal legend
        yanchor="bottom",
        y=1.02,
        xanchor="right",
        x=1)
    )
    
    # Define colors for status highlighting - using contrasting colors
    colors = ['#FF7792', "#7777FF", "#77FF92", '#FFAE77', '#77C9FF'] 
    opacities = [0.6, 0.6, 0.6, 0.6, 0.6]
    
    # Find y-axis range for shading and annotations
    y_values = []
    for col in position_columns:
        if col in factory_data.columns:
            y_values.extend(factory_data[col].dropna().tolist())
    
    if y_values:
        y_min = min(y_values)
        y_max = max(y_values)
    else:
        y_min, y_max = 0, 1
    
    # Add shading for each status code component
    for i, status_col in enumerate(status_columns):
        if status_col in factory_data.columns:
            # Find non-idle periods (status != 1)
            non_idle_periods = factory_data[factory_data[status_col] != 1]
            
            if not non_idle_periods.empty:
                # Group consecutive time periods
                time_groups = []
                current_group = []
                
                for idx, row in non_idle_periods.iterrows():
                    current_time = row['time']
                    
                    if not current_group:
                        current_group = [current_time]
                    else:
                        # Check if this time is consecutive (within resample interval)
                        last_time = current_group[-1]
                        time_diff = (current_time - last_time).total_seconds()
                        
                        if time_diff <= 10:  # Within 2 resample intervals
                            current_group.append(current_time)
                        else:
                            # Start new group
                            if len(current_group) > 0:
                                time_groups.append(current_group)
                            current_group = [current_time]
                
                # Add the last group
                if len(current_group) > 0:
                    time_groups.append(current_group)
                
                # Add shading for each time group
                for j, group in enumerate(time_groups):
                    if len(group) > 0:
                        start_time = min(group)
                        end_time = max(group) + pd.Timedelta(seconds=5)  # Add one interval
                        
                        # Add the shaded rectangle
                        fig.add_vrect(
                            x0=start_time,
                            x1=end_time,
                            fillcolor=colors[i % len(colors)],
                            opacity=opacities[i % len(opacities)],
                            layer="below",
                            line_width=0
                        )
                        
                        # Add text annotation within the shaded area for longer periods
                        if len(group) > 5:  # Only for periods longer than 5 data points
                            mid_time = start_time + (end_time - start_time) / 2  # Center of the shaded area
                            
                            # Calculate vertical position based on component index to avoid overlap
                            y_position = y_max * (0.9 - i * 0.05)  # Stagger vertically
                            
                            component_name = status_col.replace('_status_code', '').upper()
                            
                            fig.add_annotation(
                                x=mid_time,
                                y=y_position,
                                text=f"{component_name} Active",
                                showarrow=False,
                                font=dict(size=9, color='black'),
                                bgcolor=colors[i % len(colors)],
                                bordercolor='white',
                                borderwidth=1,
                                opacity=0.8,
                                xanchor='center',
                                yanchor='middle'
                            )
    
    # Add target movement annotations for VGR
    if 'vgr_target_status' in factory_data.columns:
        target_movements = factory_data[factory_data['vgr_target_status'].notna()].copy()
        
        if not target_movements.empty:
            for idx, row in target_movements.iterrows():
                target = row['vgr_target_status']
                time_point = row['time']
                
                # Check for start of movement (target changes or starts)
                if idx == 0 or pd.isna(factory_data.loc[idx-1, 'vgr_target_status']) or factory_data.loc[idx-1, 'vgr_target_status'] != target:
                    fig.add_annotation(
                        x=time_point, y=y_max * 0.95,
                        text=f"VGR → {str(target).upper()}",
                        showarrow=True, arrowhead=2,
                        arrowcolor="green", bgcolor="lightgreen",
                        bordercolor="green", borderwidth=1,
                        font=dict(size=9, color="darkgreen")
                    )
                
                # Check for end of movement (target ends or changes)
                if idx == len(factory_data)-1 or pd.isna(factory_data.loc[idx+1, 'vgr_target_status']) or factory_data.loc[idx+1, 'vgr_target_status'] != target:
                    fig.add_annotation(
                        x=time_point, y=y_max * 0.85,
                        text=f"VGR at {str(target).upper()}",
                        showarrow=True, arrowhead=2,
                        arrowcolor="red", bgcolor="lightcoral",
                        bordercolor="red", borderwidth=1,
                        font=dict(size=9, color="darkred")
                    )
    
    # Add MPO -> SLD movement annotations (when SLD status changes)
    if 'sld_status_code' in factory_data.columns:
        # Create a copy for analysis
        sld_df = factory_data[['time', 'sld_status_code']].copy()
        sld_df = sld_df.dropna()
        
        if not sld_df.empty:
            # Track status transitions
            mpo_to_sld_transfers = []  # 1 -> 2 transitions
            sld_deliveries = []        # 2 -> 1 transitions
            
            for i in range(1, len(sld_df)):
                prev_status = sld_df.iloc[i-1]['sld_status_code']
                curr_status = sld_df.iloc[i]['sld_status_code']
                curr_time = sld_df.iloc[i]['time']
                
                # MPO -> SLD: status changes from 1 to 2
                if prev_status == 1 and curr_status == 2:
                    mpo_to_sld_transfers.append(curr_time)
                    
                # SLD delivery complete: status changes from 2 to 1
                elif prev_status == 2 and curr_status == 1:
                    sld_deliveries.append(curr_time)
            
            # Add annotations for MPO -> SLD transfers (1→2)
            for i, transfer_time in enumerate(mpo_to_sld_transfers):
                fig.add_annotation(
                    x=transfer_time, y=y_max * 0.75,
                    text=f"MPO → SLD",
                    showarrow=True, arrowhead=2,
                    arrowcolor="#8A2BE2", bgcolor="#DDA0DD",  # Purple colors
                    bordercolor="#8A2BE2", borderwidth=1,
                    font=dict(size=9, color="#4B0082")
                )
            
            # Add annotations for SLD delivery completions (2→1)
            for i, delivery_time in enumerate(sld_deliveries):
                fig.add_annotation(
                    x=delivery_time, y=y_max * 0.65,
                    text=f"SLD Complete",
                    showarrow=True, arrowhead=2,
                    arrowcolor="#228B22", bgcolor="#90EE90",  # Green colors
                    bordercolor="#228B22", borderwidth=1,
                    font=dict(size=9, color="#006400")
                )
    
    # Update layout for the factory overview
    fig.update_layout(
        height=700,  # Increased height for more annotations
        hovermode='x unified',
        legend=dict(
            orientation="h", 
            yanchor="bottom", 
            y=1.02, 
            xanchor="right", 
            x=1
        )
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

def display_factory_overview_summary(factory_data):
    """Display summary statistics for the factory overview"""
    if factory_data is None or factory_data.empty:
        return
    
    # st.subheader("📊 Factory Process Summary")
    
    # # Create columns for metrics
    # col1, col2, col3, col4 = st.columns(4)
    
    # # VGR target movements
    # if 'vgr_target_status' in factory_data.columns:
    #     vgr_targets = factory_data[factory_data['vgr_target_status'].notna()]['vgr_target_status'].unique()
    #     with col1:
    #         st.metric("VGR Targets", len(vgr_targets))
    #         with st.expander("VGR Target Details"):
    #             for target in vgr_targets:
    #                 target_count = len(factory_data[factory_data['vgr_target_status'] == target])
    #                 st.write(f"→ {target}: {target_count} periods")
    
    # # SLD activity periods (MPO -> SLD transfers)
    # if 'sld_status_code' in factory_data.columns:
    #     sld_active_count = len(factory_data[factory_data['sld_status_code'] != 1])
    #     total_periods = len(factory_data[factory_data['sld_status_code'].notna()])
    #     sld_percentage = (sld_active_count / total_periods * 100) if total_periods > 0 else 0
        
    #     with col2:
    #         st.metric("SLD Active", f"{sld_percentage:.1f}%")
    #         st.caption(f"{sld_active_count}/{total_periods} periods")
    
    # Overall activity summary
    status_columns = ['mpo_status_code', 'sld_status_code', 'hbw_status_code', 'vgr_status_code']
    
    # total_activity = 0
    # active_components = 0

    # for status_col in status_columns:
    #     if status_col in factory_data.columns:
    #         non_idle_count = len(factory_data[factory_data[status_col] != 1])
    #         total_count = len(factory_data[factory_data[status_col].notna()])
    #         if total_count > 0:
    #             percentage = (non_idle_count / total_count * 100)
    #             total_activity += percentage
    #             active_components += 1

    # avg_activity = total_activity / active_components if active_components > 0 else 0

    # with col3:
    #     st.metric("Avg Activity", f"{avg_activity:.1f}%")
    #     st.caption("Across all components")

    # # Active time
    # # Calculate active time: any row where any status_col != 1
    # active_mask = pd.Series(False, index=factory_data.index)
    # for status_col in status_columns:
    #     if status_col in factory_data.columns:
    #         active_mask = active_mask | (factory_data[status_col] != 1)
    # active_times = factory_data.loc[active_mask, 'time']
    # if not active_times.empty:
    #     active_time = active_times.max() - active_times.min()
    #     active_seconds = active_time.total_seconds()
    # else:
    #     active_seconds = 0

    # with col4:
    #     st.metric("Active Time", f"{active_seconds:.0f}s")
    #     st.caption(f"{active_seconds/60:.1f} minutes")
    
    # Detailed activity breakdown
    with st.expander("🔍 Detailed Component Activity Breakdown"):
        for status_col in status_columns:
            if status_col in factory_data.columns:
                non_idle_count = len(factory_data[factory_data[status_col] != 1])
                total_count = len(factory_data[factory_data[status_col].notna()])
                percentage = (non_idle_count / total_count * 100) if total_count > 0 else 0
                component_name = status_col.replace('_status_code', '').upper()
                
                col_a, col_b, col_c = st.columns([2, 1, 1])
                with col_a:
                    st.write(f"**{component_name}**")
                with col_b:
                    st.write(f"{non_idle_count}/{total_count} periods")
                with col_c:
                    st.write(f"{percentage:.1f}% active")

def main():
    st.title("Factory Dashboard")
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
        default=['FACTORY', 'VGR', 'ENVIRONMENTAL']
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
                    if component == 'ENVIRONMENTAL':
                        # For ENVIRONMENTAL, show total duration in minutes excluding periods with null data
                        if not component_data.empty and 'time' in component_data.columns:
                            valid_times = component_data['time'][component_data.notnull().all(axis=1)]
                            if not valid_times.empty:
                                interval = pd.to_timedelta(resample_interval)
                                total_minutes = (len(valid_times) * interval.total_seconds()) / 60
                            else:
                                total_minutes = 0
                        else:
                            total_minutes = 0
                        st.metric("Active Time", f"{total_minutes:.2f} min")
                    else:
                        active_mask = pd.Series(False, index=component_data.index)
                        status_cols = [col for col in component_data.columns if col.endswith('status_code')]
                        for status_col in status_cols:
                            active_mask = active_mask | (component_data[status_col] != 1)
                        active_times = component_data.loc[active_mask, 'time']
                        if not active_times.empty:
                            active_time = active_times.max() - active_times.min()
                            active_minutes = active_time.total_seconds() / 60
                        else:
                            active_minutes = 0
                        st.metric("Active Time", f"{active_minutes:.2f} min")
                
                # Show data preview
                with st.expander("Data Preview"):
                    st.dataframe(component_data.head())
                    csv = component_data.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label="Download full CSV",
                        data=csv,
                        file_name=f"{component.lower()}_data.csv",
                        mime='text/csv'
                    )
                
                # Create and display visualization
                if component == 'FACTORY':
                    # display_factory_overview_summary(component_data)
                    fig = visualize_factory_overview(component_data, resample_interval)
                elif component == 'ENVIRONMENTAL':
                    fig = visualize_environmental_data(component_data)
                else:
                    fig = visualize_component_activity(
                        component_data, 
                        component, 
                        resample_interval
                    )
                
                if fig:
                    st.plotly_chart(fig, use_container_width=True)
                
                # Display statistics (skip for FACTORY as it has custom summary)
                if component != 'FACTORY':
                    numeric_cols = component_data.select_dtypes(include=[np.number]).columns
                    with st.expander("Statistics Summary"):
                        if len(numeric_cols) > 1:  # Exclude time column
                            stats_df = component_data[numeric_cols].describe()
                            st.dataframe(stats_df)
                else:
                    display_factory_overview_summary(component_data)
    
    else:
        st.info("Configure settings in the sidebar and click 'Load Data' to begin analysis.")

if __name__ == "__main__":
    main()
