import os
import sys
import pandas as pd
import numpy as np
import re
import tempfile
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple, Union

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(root_dir)

from utils.ts_preprocessing import load_resampled_requested_components, FACTORY_COMPONENT_CONFIG
from utils.llm_image_explanation import (
    generate_comprehensive_process_explanation,
    LLMImageAnalyzer,
    ChartImageSaver
)

# Import visualization functions
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from viz import visualize_factory_overview, visualize_component_activity, visualize_environmental_data

class AnalyticsAgent:
    """
    Refactored Analytics Agent for manufacturing data visualization and analysis.
    
    This agent provides essential data processing capabilities:
    1. Loading and processing historical factory data
    2. Generating metrics for dashboard display
    3. Processing natural language queries with chart generation
    4. Providing LLM-powered process analysis
    """
    
    def __init__(self):
        """Initialize the Analytics Agent with data processing capabilities."""
        self.component_config = FACTORY_COMPONENT_CONFIG
        self.available_components = list(self.component_config.keys())
        self.default_time_range = self._get_default_time_range()
        self.loaded_data = {}
        self.charts_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'generated_charts')
        os.makedirs(self.charts_dir, exist_ok=True)
        
        # Initialize LLM and Chart utilities
        self.llm_analyzer = LLMImageAnalyzer()
        self.chart_saver = ChartImageSaver()
    
    def _get_default_time_range(self) -> Tuple[datetime, datetime]:
        """Get a default time range for data loading."""
        # Default to factory data time range (2022-12-02 14:00 to 14:30)
        start_time = datetime(2022, 12, 2, 14, 0, 0)
        end_time = datetime(2022, 12, 2, 14, 30, 0)
        return start_time, end_time
    
    def get_available_components(self) -> List[str]:
        """Get list of available components."""
        return self.available_components
    
    def get_component_description(self, component: str) -> str:
        """Get description for a component."""
        if component in self.component_config:
            return self.component_config[component]['description']
        return f"Unknown component: {component}"
    
    def load_component_data(self, components: List[str], time_range: Optional[Tuple[datetime, datetime]] = None,
                           resample_interval: str = '5S') -> Dict[str, pd.DataFrame]:
        """
        Load component data for the specified time range.
        
        Args:
            components: List of component names to load
            time_range: Optional time range tuple (start, end)
            resample_interval: Resampling interval (default: '5S')
            
        Returns:
            Dictionary with component names as keys and DataFrames as values
        """
        if time_range is None:
            time_range = self.default_time_range
        
        try:
            # Load data for requested components
            data_dict = load_resampled_requested_components(
                components, 
                time_range, 
                resample_interval
            )
            
            # Store loaded data
            self.loaded_data.update(data_dict)
            
            return data_dict
            
        except Exception as e:
            print(f"Error loading data: {e}")
            return {}

    def get_component_metrics(self, component: str) -> Dict[str, Any]:
        """
        Get metrics for a specific component.
        
        Args:
            component: Component name
            
        Returns:
            Dictionary containing component metrics
        """
        if component not in self.loaded_data:
            return {"error": f"No data loaded for component {component}"}
        
        data = self.loaded_data[component]
        
        if data.empty:
            return {"error": f"No data available for component {component}"}
        
        metrics = {}
        
        try:
            # Calculate metrics based on component type
            if component == 'ENVIRONMENTAL':
                # Environmental metrics
                env_columns = [col for col in data.columns if any(env_type in col.lower() for env_type in 
                              ['temperature', 'humidity', 'air_quality', 'pressure', 'brightness', 'gas'])]
                
                for col in env_columns:
                    if data[col].notna().any():
                        metrics[f"{col}_avg"] = round(data[col].mean(), 2)
                        metrics[f"{col}_min"] = round(data[col].min(), 2)
                        metrics[f"{col}_max"] = round(data[col].max(), 2)
                        metrics[f"{col}_std"] = round(data[col].std(), 2)
                        
            elif component == 'FACTORY':
                # Factory overview metrics
                all_columns = [col for col in data.columns if not col.startswith('factory_')]
                
                if all_columns:
                    # Calculate activity percentages using status codes
                    activity_data = []
                    for col in all_columns:
                        if 'status_code' in col:
                            col_data = data[col]
                            # For status codes: != 1 means active
                            activity_pct = (col_data != 1).mean() * 100
                            activity_data.append(activity_pct)
                    
                    if activity_data:
                        metrics['avg_activity_pct'] = round(sum(activity_data) / len(activity_data), 2)
                        metrics['active_time_minutes'] = round(len(data) * 5 / 60, 2)  # 5-second intervals
                
            else:
                # Component-specific metrics
                # Status columns - use status codes preferentially
                status_columns = [col for col in data.columns if 'status_code' in col]
                for col in status_columns:
                    col_data = data[col]
                    # For status codes: != 1 means active
                    active_pct = (col_data != 1).mean() * 100
                    metrics[f"{col}_active_pct"] = round(active_pct, 2)
                
                # Position columns
                position_columns = [col for col in data.columns if any(pos_type in col for pos_type in 
                                   ['pos_', 'position', 'actual_pos', 'target_pos'])]
                
                for col in position_columns:
                    if data[col].notna().any():
                        metrics[f"{col}_avg"] = round(data[col].mean(), 2)
                        metrics[f"{col}_min"] = round(data[col].min(), 2)
                        metrics[f"{col}_max"] = round(data[col].max(), 2)
                        metrics[f"{col}_range"] = round(data[col].max() - data[col].min(), 2)
                        
        except Exception as e:
            metrics['error'] = f"Error calculating metrics: {str(e)}"
            
        return metrics

    def generate_data_summary(self, component: str, data: pd.DataFrame, time_range: Tuple[datetime, datetime]) -> str:
        """
        Generate a comprehensive data summary for any component.
        
        Args:
            component: Component name (ENVIRONMENTAL, FACTORY, VGR, etc.)
            data: Component data DataFrame
            time_range: Time range for the analysis
            
        Returns:
            Formatted string with component data summary
        """
        try:
            start_time, end_time = time_range
            duration_minutes = (end_time - start_time).total_seconds() / 60
            
            base_summary = f"""**{component} Analysis ({start_time.strftime('%H:%M:%S')} to {end_time.strftime('%H:%M:%S')})**

**Time Range Analysis:**
- Duration: {duration_minutes:.1f} minutes
- Data Points: {len(data)}
- Sampling Rate: 5-second intervals

"""
            
            if component == 'ENVIRONMENTAL':
                return base_summary + self._analyze_environmental_metrics(data)
            elif component == 'FACTORY':
                return base_summary + self._analyze_factory_metrics(data)
            else:
                return base_summary + self._analyze_component_metrics(data)
                
        except Exception as e:
            return f"Error generating {component} summary: {str(e)}"
    
    def _analyze_environmental_metrics(self, data: pd.DataFrame) -> str:
        """Analyze environmental data metrics."""
        metrics_text = "**Environmental Metrics:**"
        env_columns = [col for col in data.columns if any(env_type in col.lower() for env_type in 
                      ['temperature', 'humidity', 'air_quality', 'pressure', 'brightness', 'gas'])]
        
        for col in env_columns:
            if data[col].notna().any():
                avg_val = data[col].mean()
                min_val = data[col].min()
                max_val = data[col].max()
                std_val = data[col].std()
                metric_name = col.replace('factory_', '').replace('_', ' ').title()
                metrics_text += f"\n- {metric_name}: Avg {avg_val:.2f}, Range {min_val:.2f}-{max_val:.2f}, Std {std_val:.2f}"
        
        return metrics_text
    
    def _analyze_factory_metrics(self, data: pd.DataFrame) -> str:
        """Analyze factory overview metrics with cycle analysis."""
        metrics_text = "**Component Activity Analysis:**"
        
        # Component activity analysis
        component_activity = {}
        for col in data.columns:
            if 'status_code' in col:
                component_name = col.split('_')[0].upper()
                if component_name not in component_activity:
                    activity_pct = (data[col] != 1).mean() * 100
                    component_activity[component_name] = activity_pct
        
        for comp, pct in component_activity.items():
            metrics_text += f"\n- {comp}: {pct:.1f}% active time"
        
        # Position movement analysis
        metrics_text += "\n\n**Position Movement Analysis:**"
        position_movements = {}
        for col in data.columns:
            if any(pos_type in col for pos_type in ['actual_pos', 'target_pos']):
                component_name = col.split('_')[0].upper()
                if component_name not in position_movements:
                    position_movements[component_name] = {
                        'min_pos': data[col].min(),
                        'max_pos': data[col].max(),
                        'movement_range': data[col].max() - data[col].min()
                    }
        
        for comp, pos_data in position_movements.items():
            metrics_text += f"\n- {comp}: Range {pos_data['min_pos']:.0f}-{pos_data['max_pos']:.0f}, Movement: {pos_data['movement_range']:.0f}"
        
        # Process cycle analysis
        cycle_analysis = self._analyze_process_cycles(data)
        metrics_text += f"\n\n**Process Cycle Analysis:**"
        metrics_text += f"\n- Detected Process Cycles: {cycle_analysis['cycle_count']}"
        metrics_text += f"\n- Average Cycle Duration: {cycle_analysis['avg_cycle_duration']:.1f} minutes"
        metrics_text += f"\n- Overall Process Type: {cycle_analysis['cycle_type']}"
        
        if cycle_analysis['cycle_details']:
            metrics_text += "\n\n**Individual Cycle Details:**"
            for cycle in cycle_analysis['cycle_details']:
                metrics_text += f"\n- Cycle {cycle['cycle_number']}: ({cycle['duration_minutes']}min) - {cycle['process_type']}"
                if cycle['active_components']:
                    metrics_text += f" [Components: {', '.join(cycle['active_components'])}]"
        
        return metrics_text
    
    def _analyze_component_metrics(self, data: pd.DataFrame) -> str:
        """Analyze individual component metrics."""
        metrics_text = "**Component Activity:**"
        
        # Status code analysis
        status_columns = [col for col in data.columns if 'status_code' in col]
        for col in status_columns:
            if data[col].notna().any():
                activity_pct = (data[col] != 1).mean() * 100
                metrics_text += f"\n- {col.replace('_', ' ').title()}: {activity_pct:.1f}% active time"
        
        # Position analysis
        position_columns = [col for col in data.columns if any(pos_type in col for pos_type in 
                           ['pos_', 'position', 'actual_pos', 'target_pos'])]
        
        if position_columns:
            metrics_text += "\n\n**Position Analysis:**"
            for col in position_columns:
                if data[col].notna().any():
                    avg_pos = data[col].mean()
                    min_pos = data[col].min()
                    max_pos = data[col].max()
                    movement_range = max_pos - min_pos
                    pos_name = col.replace('_', ' ').title()
                    metrics_text += f"\n- {pos_name}: Avg {avg_pos:.0f}, Range {min_pos:.0f}-{max_pos:.0f}, Movement {movement_range:.0f}"
        
        return metrics_text

    def _analyze_process_cycles(self, factory_data: pd.DataFrame) -> Dict[str, Any]:
        """
        Enhanced cycle detection considering multiple components and their coordination.
        Now includes per-cycle process type identification.
        
        Args:
            factory_data: Factory data DataFrame
            
        Returns:
            Dictionary with cycle analysis results including per-cycle types
        """
        cycle_analysis = {
            'cycle_count': 0,
            'avg_cycle_duration': 0,
            'cycle_type': 'Unknown',
            'cycle_details': []  # New: detailed info for each cycle
        }
        
        try:
            # Enhanced cycle detection: consider multiple components working together
            # Key components in the manufacturing process
            component_columns = {
                'vgr': 'vgr_status_code',
                'hbw': 'hbw_status_code', 
                'mpo': 'mpo_status_code',
                'sld': 'sld_status_code'
            }
            
            # Find available components
            available_components = {}
            for comp_name, col_name in component_columns.items():
                if col_name in factory_data.columns:
                    available_components[comp_name] = col_name
            
            if not available_components:
                return cycle_analysis
            
            # Create a combined activity signal considering all components
            # Any component active (status != 1) means the process is active
            combined_activity = pd.Series(False, index=factory_data.index)
            
            for comp_name, col_name in available_components.items():
                component_active = factory_data[col_name] != 1
                combined_activity = combined_activity | component_active
            
            # Find process start transitions with more flexible logic
            # Option 1: Find transitions from idle to active
            process_start_mask = (combined_activity.shift(1) == False) & (combined_activity == True)
            
            # CRITICAL FIX: Extract proper timestamps from the dataframe index
            # Instead of using list() which can return positions, directly access the timestamps
            process_start_indices = []
            for i, is_start in enumerate(process_start_mask):
                if is_start:
                    process_start_indices.append(factory_data.index[i])
            
            # If no transitions found, try alternative approach: 
            # Find continuous active periods and treat each as a cycle
            if len(process_start_indices) == 0 and combined_activity.sum() > 0:
                # Find all active periods
                active_periods = []
                in_active_period = False
                period_start = None
                
                for i, is_active in enumerate(combined_activity):
                    if is_active and not in_active_period:
                        # Start of new active period - use timestamp
                        period_start = factory_data.index[i]
                        in_active_period = True
                    elif not is_active and in_active_period:
                        # End of active period
                        period_end = factory_data.index[i-1] if i > 0 else factory_data.index[i]
                        active_periods.append((period_start, period_end))
                        in_active_period = False
                
                # Handle case where data ends while still active
                if in_active_period and period_start is not None:
                    active_periods.append((period_start, factory_data.index[-1]))
                
                # Use active periods as cycles - use timestamps, not indices
                if active_periods:
                    process_start_indices = [start for start, end in active_periods]
            
            # Verify all process start indices are timestamps
            if process_start_indices:
                # Additional validation - ensure all indices are timestamps
                for idx in process_start_indices:
                    if not isinstance(idx, (pd.Timestamp, datetime)):
                        # This should not happen now, but if it does, we'll convert
                        if isinstance(idx, (int, np.integer)):
                            # This is a fallback - should not be needed
                            converted_idx = factory_data.index[idx]
            
            if len(process_start_indices) == 0:
                cycle_analysis['cycle_count'] = 0
            else:
                # Filter transitions that are too close together (< 1 minute)
                # Since we have 5-second intervals, minimum cycle duration should be 1 minute
                # 60 seconds / 5 seconds per interval = 12 intervals minimum
                
                filtered_transitions = []
                min_cycle_duration_intervals = 12  # 60 seconds / 5 seconds per interval (1 minute minimum)
                
                if process_start_indices:
                    # Always keep the first transition
                    filtered_transitions.append(process_start_indices[0])
                    
                    # Check subsequent transitions
                    for i in range(1, len(process_start_indices)):
                        current_idx = process_start_indices[i]
                        last_kept_idx = filtered_transitions[-1]
                        
                        # Calculate time difference in intervals
                        # Find the position of current and last indices in the dataframe
                        current_pos = factory_data.index.get_loc(current_idx)
                        last_pos = factory_data.index.get_loc(last_kept_idx)
                        time_diff_intervals = current_pos - last_pos
                        
                        # Only keep this transition if it's far enough from the last one
                        if time_diff_intervals >= min_cycle_duration_intervals:
                            filtered_transitions.append(current_idx)
                
                # ENSURE ALL FILTERED TRANSITIONS ARE TIMESTAMPS
                validated_filtered_transitions = []
                for idx in filtered_transitions:
                    if isinstance(idx, (int, np.integer)):
                        validated_filtered_transitions.append(factory_data.index[idx])
                    else:
                        validated_filtered_transitions.append(idx)
                filtered_transitions = validated_filtered_transitions
                
                cycle_analysis['cycle_count'] = len(filtered_transitions)
                
                # Analyze each individual cycle
                cycle_details = []
                for i, cycle_start in enumerate(filtered_transitions):
                    cycle_num = i + 1
                    
                    # Determine cycle end (next cycle start or end of data)
                    if i + 1 < len(filtered_transitions):
                        next_cycle_start = filtered_transitions[i + 1]
                        try:
                            next_pos = factory_data.index.get_loc(next_cycle_start)
                            if next_pos > 0:
                                cycle_end = factory_data.index[next_pos - 1]
                            else:
                                cycle_end = next_cycle_start
                        except KeyError:
                            cycle_end = factory_data.index[-1]
                    else:
                        cycle_end = factory_data.index[-1]
                    
                    # Extract data for this specific cycle
                    cycle_data = factory_data.loc[cycle_start:cycle_end]
                    
                    # Calculate cycle duration
                    cycle_duration_minutes = len(cycle_data) * 5 / 60  # 5-second intervals
                    
                    # Determine process type for this cycle
                    cycle_type = self._determine_cycle_type(cycle_data, available_components)
                    
                    cycle_info = {
                        'cycle_number': cycle_num,
                        'duration_minutes': round(cycle_duration_minutes, 1),
                        'process_type': cycle_type,
                        'active_components': self._get_active_components_for_cycle(cycle_data, available_components)
                    }
                    
                    cycle_details.append(cycle_info)
                
                cycle_analysis['cycle_details'] = cycle_details
                
                # Calculate average cycle duration if cycles detected
                if len(filtered_transitions) > 1:
                    # Calculate duration based on filtered transitions
                    total_duration_minutes = len(factory_data) * 5 / 60  # 5-second intervals
                    cycle_analysis['avg_cycle_duration'] = round(total_duration_minutes / len(filtered_transitions), 1)
                elif len(filtered_transitions) == 1:
                    # Single cycle - use total duration
                    total_duration_minutes = len(factory_data) * 5 / 60
                    cycle_analysis['avg_cycle_duration'] = round(total_duration_minutes, 1)
                
                # Determine overall process type based on which components are active across all cycles
                if cycle_analysis['cycle_count'] > 0:
                    hbw_active = 'hbw' in available_components and (factory_data[available_components['hbw']] != 1).any()
                    mpo_active = 'mpo' in available_components and (factory_data[available_components['mpo']] != 1).any()
                    sld_active = 'sld' in available_components and (factory_data[available_components['sld']] != 1).any()
                    vgr_active = 'vgr' in available_components and (factory_data[available_components['vgr']] != 1).any()
                    
                    if mpo_active or sld_active:
                        cycle_analysis['cycle_type'] = 'Production Process'
                    elif hbw_active and vgr_active:
                        cycle_analysis['cycle_type'] = 'Storage Process'
                    elif vgr_active:
                        cycle_analysis['cycle_type'] = 'Transfer/Movement Process'
                    else:
                        cycle_analysis['cycle_type'] = 'Mixed/Unknown Process'
            
            return cycle_analysis
            
        except Exception as e:
            cycle_analysis['cycle_count'] = 0
            return cycle_analysis

    def _determine_cycle_type(self, cycle_data: pd.DataFrame, available_components: Dict[str, str]) -> str:
        """
        Determine the process type for a specific cycle based on active components.
        
        Args:
            cycle_data: DataFrame containing data for a specific cycle
            available_components: Dictionary of available component columns
            
        Returns:
            String describing the process type for this cycle
        """
        try:
            # Check which components are active during this cycle
            hbw_active = 'hbw' in available_components and (cycle_data[available_components['hbw']] != 1).any()
            mpo_active = 'mpo' in available_components and (cycle_data[available_components['mpo']] != 1).any()
            sld_active = 'sld' in available_components and (cycle_data[available_components['sld']] != 1).any()
            vgr_active = 'vgr' in available_components and (cycle_data[available_components['vgr']] != 1).any()
            
            # Determine process type based on active components
            if mpo_active or sld_active:
                return 'Production Process'
            elif hbw_active and vgr_active:
                return 'Storage Process'
            elif vgr_active:
                return 'Transfer/Movement Process'
            else:
                return 'Unknown Process'
                
        except Exception as e:
            return 'Unknown Process'

    def _get_active_components_for_cycle(self, cycle_data: pd.DataFrame, available_components: Dict[str, str]) -> List[str]:
        """
        Get list of active components for a specific cycle.
        
        Args:
            cycle_data: DataFrame containing data for a specific cycle
            available_components: Dictionary of available component columns
            
        Returns:
            List of active component names
        """
        try:
            active_components = []
            
            for comp_name, col_name in available_components.items():
                if (cycle_data[col_name] != 1).any():
                    active_components.append(comp_name.upper())
            
            return active_components
            
        except Exception as e:
            return []

    def analyze_cycles(self, time_range_or_start: Optional[Union[Tuple[datetime, datetime], datetime]] = None, 
                       end_time: Optional[datetime] = None) -> Dict[str, Any]:
        """
        Public method to analyze manufacturing cycles in the factory data for a given time range.
        If no time range is provided, uses the default time range.
        
        Args:
            time_range_or_start: Either a tuple (start_time, end_time) or just the start_time
            end_time: If time_range_or_start is a start time, this must be the end time
        Returns:
            Dictionary with cycle analysis results including per-cycle types
        """
        # Handle both calling conventions:
        # 1. analyze_cycles((start, end))
        # 2. analyze_cycles(start, end)
        if end_time is not None and isinstance(time_range_or_start, datetime):
            # Called as analyze_cycles(start, end)
            time_range = (time_range_or_start, end_time)
        else:
            # Called as analyze_cycles(time_range)
            time_range = time_range_or_start
            
        if time_range is None:
            time_range = self.default_time_range
            
        # Load factory data for the given time range if not already loaded
        if 'FACTORY' not in self.loaded_data or self.loaded_data['FACTORY'].empty:
            self.load_component_data(['FACTORY'], time_range)
        factory_data = self.loaded_data.get('FACTORY', None)
        if factory_data is None or factory_data.empty:
            return {'error': 'No factory data available for cycle analysis'}
        return self._analyze_process_cycles(factory_data)

    def generate_process_data_summary(self, time_range: Optional[Tuple[datetime, datetime]] = None) -> str:
        """Generate process data summary - wrapper for backwards compatibility."""
        if time_range is None:
            time_range = self.default_time_range
        
        if 'FACTORY' not in self.loaded_data:
            self.load_component_data(['FACTORY'], time_range)
        
        factory_data = self.loaded_data.get('FACTORY')
        if factory_data is None or factory_data.empty:
            return "No factory data available for analysis"
        
        return self.generate_data_summary('FACTORY', factory_data, time_range)
    

    
    def query(self, query_text: str, time_range: Optional[Tuple[datetime, datetime]] = None) -> Dict[str, Any]:
        """
        Process a natural language query about the factory data.
        
        Args:
            query_text: The user's natural language query
            time_range: Optional time range for the query
            
        Returns:
            Dictionary containing query results with charts and analysis
        """
        try:
            # Parse time range from query if not provided
            if time_range is None:
                time_range = self._parse_time_range_from_query(query_text)
            
            # Detect query type and components
            query_type = self._classify_query_type(query_text)
            components_to_load = self._detect_components_from_query(query_text)
            
            # Load the appropriate data
            self.load_component_data(components_to_load, time_range)
            
            # Get the primary data for analysis
            primary_component = components_to_load[0]
            primary_data = self.loaded_data.get(primary_component)
            
            if primary_data is None or primary_data.empty:
                return {
                    'error': f'No {primary_component.lower()} data available for the specified time range',
                    'query': query_text,
                    'time_range': time_range,
                    'success': False
                }
            
            # Generate data summary
            data_summary = self.generate_data_summary(primary_component, primary_data, time_range)
            
            # Generate chart path for LLM analysis (process analysis only)
            chart_path = None
            if query_type == 'process_analysis':
                chart_path = self._save_chart_for_llm(primary_component, primary_data)
            
            # Generate response based on query type
            if query_type == 'environmental':
                return {
                    'query': query_text,
                    'time_range': time_range,
                    'primary_component': primary_component,
                    'data_summary': data_summary,
                    'chart_path': chart_path,
                    'query_type': 'environmental',
                    'llm_analysis': data_summary,  # For environmental, just return data summary
                    'success': True
                }
            else:  # process_analysis
                # Generate LLM analysis for process queries
                llm_analysis = self._generate_llm_analysis(chart_path, data_summary, query_text, primary_component)
                
                return {
                    'query': query_text,
                    'time_range': time_range,
                    'primary_component': primary_component,
                    'data_summary': data_summary,
                    'chart_path': chart_path,
                    'query_type': 'process_analysis',
                    'llm_analysis': llm_analysis,
                    'success': True
                }
            
        except Exception as e:
            return {
                'error': f'Error processing query: {str(e)}',
                'query': query_text,
                'time_range': time_range,
                'success': False
            }
    
    def _classify_query_type(self, query_text: str) -> str:
        """Classify query as environmental or process analysis."""
        query_lower = query_text.lower()
        
        # Environmental queries take priority
        if any(phrase in query_lower for phrase in [
            'environment', 'environmental', 'temperature', 'humidity', 'air', 
            'pressure', 'gas', 'quality', 'sensor', 'condition'
        ]):
            return 'environmental'
        
        return 'process_analysis'
    
    def _save_chart_for_llm(self, component: str, data: pd.DataFrame) -> Optional[str]:
        """Save chart to file for LLM analysis."""
        try:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            chart_path = os.path.join(self.charts_dir, f'{component.lower()}_{timestamp}.png')
            
            fig = self._create_chart_by_component(component, data)
            if fig is not None:
                success = self.chart_saver.save_chart(fig, chart_path)
                return chart_path if success else None
            return None
        except Exception as e:
            print(f"Error saving chart for LLM: {e}")
            return None
    
    def _generate_llm_analysis(self, chart_path: Optional[str], data_summary: str, 
                              query_text: str, primary_component: str) -> str:
        """Generate LLM analysis for process queries."""
        if chart_path and os.path.exists(chart_path):
            return generate_comprehensive_process_explanation(
                chart_path, 
                data_summary,
                f"User query: {query_text}\n\nPlease provide a comprehensive analysis addressing the user's specific question about the {primary_component.lower()} data."
            )
        else:
            return f"Based on the query '{query_text}', here's the {primary_component.lower()} analysis:\n\n{data_summary}"
    
    def _parse_time_range_from_query(self, query: str) -> Tuple[datetime, datetime]:
        """
        Parse date and time information from a natural language query.
        
        For relative time expressions (e.g., "last 10 minutes", "past hour"), 
        this method uses the current time as reference to support future 
        integration with real-time databases.
        
        For absolute dates (e.g., "2/12/2022"), it uses the specified date.
        
        Args:
            query: Natural language query that may contain date/time information
            
        Returns:
            Tuple of (start_time, end_time) if found, default time range otherwise
        """
        query_lower = query.lower()

        # Handle relative time expressions first (e.g., "last 10 minutes", "past hour")
        relative_patterns = [
            # "last X minutes/hours/days"
            (r'last\s+(\d+)\s+minutes?', 'minutes'),
            (r'last\s+(\d+)\s+hours?', 'hours'),
            (r'last\s+(\d+)\s+days?', 'days'),
            # "past X minutes/hours/days"
            (r'past\s+(\d+)\s+minutes?', 'minutes'),
            (r'past\s+(\d+)\s+hours?', 'hours'),
            (r'past\s+(\d+)\s+days?', 'days'),
            # "X minutes/hours ago"
            (r'(\d+)\s+minutes?\s+ago', 'minutes'),
            (r'(\d+)\s+hours?\s+ago', 'hours'),
            (r'(\d+)\s+days?\s+ago', 'days'),
        ]

        for pattern, unit in relative_patterns:
            match = re.search(pattern, query_lower)
            if match:
                amount = int(match.group(1))

                # Use current time as reference for relative queries
                # This allows for future integration with real-time databases
                now = datetime.now()

                if unit == 'minutes':
                    start_time = now - timedelta(minutes=amount)
                    end_time = now
                elif unit == 'hours':
                    start_time = now - timedelta(hours=amount)
                    end_time = now
                elif unit == 'days':
                    start_time = now - timedelta(days=amount)
                    end_time = now

                return (start_time, end_time)

        # Common date patterns
        date_patterns = [
            # DD/MM/YYYY or D/M/YYYY formats
            r'(\d{1,2})/(\d{1,2})/(\d{4})',
            # YYYY-MM-DD format
            r'(\d{4})-(\d{1,2})-(\d{1,2})',
            # DD-MM-YYYY format
            r'(\d{1,2})-(\d{1,2})-(\d{4})',
            # Handle ordinal dates like "2nd december 2022"
            r'(\d{1,2})(?:st|nd|rd|th)?\s+(?:of\s+)?([a-z]+)\s+(\d{4})',
        ]

        # Try to find date
        date_match = None
        date_format = None
        for i, pattern in enumerate(date_patterns):
            match = re.search(pattern, query_lower)
            if match:
                date_match = match
                date_format = i
                break

        if not date_match:
            # If no specific date found, return default
            return self.default_time_range

        # Parse the date based on the pattern matched
        groups = date_match.groups()

        if date_format == 3:  # Ordinal date format (e.g., "2nd december 2022")
            day = int(groups[0])
            month_name = groups[1].lower()
            year = int(groups[2])

            # Convert month name to number
            month_names = {
                'january': 1, 'jan': 1, 'february': 2, 'feb': 2, 'march': 3, 'mar': 3,
                'april': 4, 'apr': 4, 'may': 5, 'june': 6, 'jun': 6, 'july': 7, 'jul': 7,
                'august': 8, 'aug': 8, 'september': 9, 'sep': 9, 'october': 10, 'oct': 10,
                'november': 11, 'nov': 11, 'december': 12, 'dec': 12
            }
            month = month_names.get(month_name)
            if not month:
                return self.default_time_range
        elif len(groups) == 3:
            # Check if it's DD/MM/YYYY or YYYY-MM-DD format
            if len(groups[0]) == 4:  # YYYY-MM-DD
                year, month, day = int(groups[0]), int(groups[1]), int(groups[2])
            else:  # DD/MM/YYYY or DD-MM-YYYY
                day, month, year = int(groups[0]), int(groups[1]), int(groups[2])
        else:
            return self.default_time_range

        try:
            base_date = datetime(year, month, day)
        except ValueError:
            # Invalid date
            return self.default_time_range

        # Look for time ranges
        start_time = None
        end_time = None

        # Pattern for "from HH:MM to HH:MM" or "from HHpm to HH:MMpm"
        time_range_patterns = [
            r'from\s+(\d{1,2}):(\d{2})\s+to\s+(\d{1,2}):(\d{2})',
            r'from\s+(\d{1,2})(?::(\d{2}))?\s*pm\s+to\s+(\d{1,2}):(\d{2})\s*pm',
            r'from\s+(\d{1,2})\s*pm\s+to\s+(\d{1,2}):(\d{2})\s*pm',
        ]

        for pattern in time_range_patterns:
            time_range_match = re.search(pattern, query_lower)
            if time_range_match:
                groups = time_range_match.groups()

                if pattern.endswith('pm'):
                    # Handle PM times
                    if len(groups) == 4 and groups[1] is not None:  # HH:MM pm to HH:MM pm
                        start_hour, start_min, end_hour, end_min = map(int, groups)
                        if start_hour < 12:
                            start_hour += 12
                        if end_hour < 12:
                            end_hour += 12
                    elif len(groups) == 3:  # H pm to HH:MM pm
                        start_hour, end_hour, end_min = int(groups[0]), int(groups[1]), int(groups[2])
                        start_min = 0
                        if start_hour < 12:
                            start_hour += 12
                        if end_hour < 12:
                            end_hour += 12
                    else:
                        continue  # Skip this pattern if groups don't match expected format
                else:
                    # Regular 24-hour format
                    if len(groups) == 4:
                        start_hour, start_min, end_hour, end_min = map(int, groups)
                    else:
                        continue  # Skip this pattern if groups don't match expected format

                start_time = base_date.replace(hour=start_hour, minute=start_min, second=0)
                end_time = base_date.replace(hour=end_hour, minute=end_min, second=0)
                break

        if not start_time:
            # Pattern for "at HH:MM" - use 30 minute window
            at_time_match = re.search(r'at\s+(\d{1,2}):(\d{2})', query_lower)
            if at_time_match:
                hour, minute = map(int, at_time_match.groups())
                start_time = base_date.replace(hour=hour, minute=minute, second=0)
                end_time = start_time + timedelta(minutes=30)
            else:
                # Look for individual times and try to infer range
                times_found = []
                for match in re.finditer(r'(\d{1,2}):(\d{2})', query):
                    hour, minute = int(match.group(1)), int(match.group(2))
                    times_found.append(base_date.replace(hour=hour, minute=minute, second=0))

                if len(times_found) >= 2:
                    start_time = min(times_found)
                    end_time = max(times_found)
                elif len(times_found) == 1:
                    # Single time, use 1 hour window
                    start_time = times_found[0]
                    end_time = start_time + timedelta(hours=1)
                else:
                    # No specific time found, use the whole day
                    start_time = base_date
                    end_time = base_date + timedelta(days=1)

        if start_time and end_time:
            return (start_time, end_time)

        return self.default_time_range
    
    def _detect_components_from_query(self, query_text: str) -> List[str]:
        """
        Detect what components to load based on the query content.
        
        Args:
            query_text: The user's natural language query
            
        Returns:
            List of component names to load, ordered by priority
        """
        query_lower = query_text.lower()
        components_to_load = []
        
        # Environmental keywords
        if any(keyword in query_lower for keyword in ['environmental', 'environment', 'temperature', 'humidity', 'air quality', 'pressure', 'gas', 'brightness']):
            components_to_load.append('ENVIRONMENTAL')
        
        # Specific component keywords
        if any(keyword in query_lower for keyword in ['vgr', 'vacuum gripper', 'gripper']):
            components_to_load.append('VGR')
        elif any(keyword in query_lower for keyword in ['hbw', 'high bay warehouse', 'warehouse', 'storage']):
            components_to_load.append('HBW')
        elif any(keyword in query_lower for keyword in ['mpo', 'multi processing', 'processing']):
            components_to_load.append('MPO')
        elif any(keyword in query_lower for keyword in ['sld', 'sorting line', 'sorting']):
            components_to_load.append('SLD')
        
        # Default to factory overview if no specific component mentioned
        if not components_to_load:
            components_to_load.append('FACTORY')
        
        return components_to_load
    

    
    def get_interactive_chart(self, query_text: str, time_range: Optional[Tuple[datetime, datetime]] = None):
        """
        Generate interactive chart based on query.
        
        Args:
            query_text: The user's natural language query
            time_range: Optional time range for the query
            
        Returns:
            Plotly figure object or None if generation fails
        """
        try:
            if time_range is None:
                time_range = self.default_time_range
            
            # Detect and load components
            components_to_load = self._detect_components_from_query(query_text)
            self.load_component_data(components_to_load, time_range)
            
            # Get primary component and data
            primary_component = components_to_load[0]
            primary_data = self.loaded_data.get(primary_component)
            
            if primary_data is None or primary_data.empty:
                return None
            
            # Generate appropriate chart
            return self._create_chart_by_component(primary_component, primary_data)
                
        except Exception as e:
            print(f"Error generating interactive chart: {e}")
            return None
    
    def _create_chart_by_component(self, component: str, data: pd.DataFrame):
        """Create chart based on component type."""
        if component == 'ENVIRONMENTAL':
            return visualize_environmental_data(data)
        elif component == 'FACTORY':
            return visualize_factory_overview(data, '5S')
        else:
            return visualize_component_activity(data, component, '5S')
        

