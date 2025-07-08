import os
import sys
import pandas as pd
import numpy as np
import re
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple

# Add the src directory to the path to import utils
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Also add the root directory to access factory_data
root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(root_dir)

try:
    from utils.ts_preprocessing import (
        load_resampled_requested_components,
        FACTORY_COMPONENT_CONFIG
    )
except ImportError as e:
    print(f"Warning: Could not import ts_preprocessing: {e}")
    # Fallback configuration
    FACTORY_COMPONENT_CONFIG = {
        'FACTORY': {'description': 'Factory Overview'},
        'VGR': {'description': 'Vacuum Gripper Robot'},
        'HBW': {'description': 'High-bay Warehouse'},
        'MPO': {'description': 'Multi-Processing Station'},
        'SLD': {'description': 'Sorting Line'},
        'DPS': {'description': 'Delivery Station'},
        'SSC': {'description': 'Surveillance Camera'},
        'ENVIRONMENTAL': {'description': 'Environmental Sensors'}
    }
    
    def load_resampled_requested_components(components, time_range, resample_interval='5S'):
        """Fallback function when ts_preprocessing is not available"""
        return {}

class AnalyticsAgent:
    """
    Analytics Agent for manufacturing data visualization and analysis.
    
    This agent provides data visualization and analytics capabilities by:
    1. Loading and processing historical factory data
    2. Generating interactive visualizations for different components
    3. Calculating performance metrics and statistics
    4. Providing insights based on historical trends
    """
    
    def __init__(self):
        """Initialize the Analytics Agent with data processing capabilities."""
        self.component_config = FACTORY_COMPONENT_CONFIG
        self.available_components = list(self.component_config.keys())
        self.default_time_range = self._get_default_time_range()
        self.loaded_data = {}
        self.metrics_cache = {}
    
    def _get_default_time_range(self) -> Tuple[datetime, datetime]:
        """Get a default time range for data loading."""
        # Use the actual time range from the factory data (starts 2022-12-01)
        start_time = datetime(2022, 12, 1, 0, 0, 0)
        end_time = datetime(2022, 12, 1, 1, 0, 0)  # 1 hour window
        return (start_time, end_time)
    
    def get_available_components(self) -> List[str]:
        """Return a list of available factory components for visualization."""
        return self.available_components
    
    def get_component_description(self, component: str) -> str:
        """Get the description for a specific component."""
        if component in self.component_config:
            return self.component_config[component]['description']
        return "Component not found"
    
    def load_component_data(self, components: List[str], time_range: Optional[Tuple[datetime, datetime]] = None,
                           resample_interval: str = '5S') -> Dict[str, pd.DataFrame]:
        """
        Load data for requested components within the specified time range.
        
        Args:
            components: List of component names to load data for
            time_range: Tuple of (start_time, end_time) or None to use default
            resample_interval: Resampling interval for time series data
            
        Returns:
            Dictionary of component names mapped to their dataframes
        """
        if time_range is None:
            time_range = self.default_time_range
            
        # Validate components
        valid_components = [c for c in components if c in self.available_components]
        if not valid_components:
            print(f"No valid components found. Available: {self.available_components}")
            return {}
            
        # Load data using the utility function
        try:
            print(f"Loading data for components: {valid_components}")
            print(f"Time range: {time_range}")
            print(f"Resample interval: {resample_interval}")
            
            self.loaded_data = load_resampled_requested_components(
                valid_components, time_range, resample_interval
            )
            
            # Debug output
            if self.loaded_data:
                for comp, data in self.loaded_data.items():
                    if data is not None:
                        print(f"Loaded {comp}: {data.shape} rows/cols")
                    else:
                        print(f"No data for {comp}")
            else:
                print("No data loaded at all")
                
        except Exception as e:
            print(f"Error loading data: {str(e)}")
            import traceback
            traceback.print_exc()
            self.loaded_data = {}
        
        return self.loaded_data
    
    def get_component_metrics(self, component: str) -> Dict[str, Any]:
        """
        Calculate and return key metrics for a component.
        
        Args:
            component: Component name to calculate metrics for
            
        Returns:
            Dictionary of metrics with their values
        """
        if component not in self.loaded_data or self.loaded_data[component] is None:
            return {"error": f"No data loaded for {component}"}
            
        # If metrics are already calculated, return from cache
        if component in self.metrics_cache:
            return self.metrics_cache[component]
            
        df = self.loaded_data[component]
        metrics = {}
        
        # Calculate basic metrics based on component type
        if component == 'FACTORY':
            # For factory overview, calculate activity metrics across components
            status_columns = [col for col in df.columns if 'status_code' in col]
            
            # Overall activity percentage
            active_percentages = []
            for col in status_columns:
                if col in df.columns:
                    non_idle = df[df[col] != 1][col].count()
                    total = df[col].count()
                    percentage = (non_idle / total * 100) if total > 0 else 0
                    active_percentages.append(percentage)
                    
                    component_name = col.replace('_status_code', '').upper()
                    metrics[f"{component_name}_active_pct"] = round(percentage, 1)
            
            # Average activity across all components
            metrics["avg_activity_pct"] = round(sum(active_percentages) / len(active_percentages), 1) if active_percentages else 0
            
            # Calculate total active time (any component active)
            active_mask = pd.Series(False, index=df.index)
            for col in status_columns:
                active_mask = active_mask | (df[col] != 1)
            
            active_times = df.loc[active_mask, 'time']
            if not active_times.empty:
                active_time = active_times.max() - active_times.min()
                active_seconds = active_time.total_seconds()
                metrics["active_time_seconds"] = round(active_seconds, 1)
                metrics["active_time_minutes"] = round(active_seconds / 60, 1)
            else:
                metrics["active_time_seconds"] = 0
                metrics["active_time_minutes"] = 0
                
            # Count target movements for VGR if available
            if 'vgr_target_status' in df.columns:
                vgr_targets = df[df['vgr_target_status'].notna()]['vgr_target_status'].unique()
                metrics["vgr_target_count"] = len(vgr_targets)
                
                # Target details
                target_counts = {}
                for target in vgr_targets:
                    target_count = len(df[df['vgr_target_status'] == target])
                    target_counts[str(target)] = target_count
                metrics["vgr_target_details"] = target_counts
                
        elif component == 'ENVIRONMENTAL':
            # For environmental data, calculate statistics for each sensor
            env_mapping = {
                'factory_air_pressure': ('Air Pressure', 'hPa'),
                'factory_index_air_quality': ('Air Quality Index', 'index'),
                'factory_gas_resistance': ('Gas Resistance', 'Ω'),
                'factory_humidity_raw': ('Humidity', '%'),
                'factory_temperature_raw': ('Temperature', '°C'),
            }
            
            for col, (name, unit) in env_mapping.items():
                if col in df.columns:
                    values = df[col].dropna()
                    if not values.empty:
                        metrics[f"{col}_avg"] = round(values.mean(), 2)
                        metrics[f"{col}_min"] = round(values.min(), 2)
                        metrics[f"{col}_max"] = round(values.max(), 2)
                        metrics[f"{col}_unit"] = unit
                        
        else:
            # For other components, focus on status codes and positions
            # Status analysis
            status_cols = [col for col in df.columns if 'status_code' in col]
            for col in status_cols:
                if col in df.columns:
                    non_idle = df[df[col] != 1][col].count()
                    total = df[col].count()
                    percentage = (non_idle / total * 100) if total > 0 else 0
                    metrics[f"{col}_active_pct"] = round(percentage, 1)
            
            # Position analysis for components with position data
            position_cols = [col for col in df.columns if any(pos in col for pos in 
                            ['pos_horizontal', 'pos_vertical', 'pos_rotation', 'position_pan', 'position_tilt'])]
            
            for col in position_cols:
                if col in df.columns:
                    values = df[col].dropna()
                    if not values.empty:
                        metrics[f"{col}_avg"] = round(values.mean(), 2)
                        metrics[f"{col}_min"] = round(values.min(), 2)
                        metrics[f"{col}_max"] = round(values.max(), 2)
                        metrics[f"{col}_range"] = round(values.max() - values.min(), 2)
        
        # Cache the calculated metrics
        self.metrics_cache[component] = metrics
        return metrics
    
    def get_anomaly_insights(self, component: str) -> List[Dict[str, Any]]:
        """
        Identify potential anomalies or interesting patterns in the data.
        
        Args:
            component: Component name to analyze
            
        Returns:
            List of anomaly insights with descriptions and timestamps
        """
        if component not in self.loaded_data or self.loaded_data[component] is None:
            return []
        
        df = self.loaded_data[component]
        insights = []
        
        # Component-specific anomaly detection
        if component == 'FACTORY':
            # Look for concurrent activity across multiple components
            status_columns = [col for col in df.columns if 'status_code' in col]
            if len(status_columns) >= 2:
                concurrent_mask = pd.Series(False, index=df.index)
                for col in status_columns:
                    concurrent_mask = concurrent_mask | (df[col] != 1)
                
                concurrent_times = df.loc[concurrent_mask, 'time']
                if not concurrent_times.empty:
                    insights.append({
                        "type": "concurrent_activity",
                        "description": f"Concurrent activity detected across multiple components",
                        "start_time": concurrent_times.min(),
                        "end_time": concurrent_times.max(),
                        "duration_seconds": (concurrent_times.max() - concurrent_times.min()).total_seconds()
                    })
        
        elif component == 'ENVIRONMENTAL':
            # Check for temperature spikes or drops
            if 'factory_temperature_raw' in df.columns:
                temp = df['factory_temperature_raw'].dropna()
                if not temp.empty:
                    # Calculate rolling mean and standard deviation
                    rolling_mean = temp.rolling(window=10, min_periods=1).mean()
                    rolling_std = temp.rolling(window=10, min_periods=1).std()
                    
                    # Find values that deviate significantly
                    threshold = 2.0  # Standard deviations
                    anomalies = df[(temp > rolling_mean + threshold * rolling_std) | 
                                  (temp < rolling_mean - threshold * rolling_std)]
                    
                    if not anomalies.empty:
                        insights.append({
                            "type": "temperature_anomaly",
                            "description": f"Temperature anomalies detected",
                            "count": len(anomalies),
                            "max_deviation": round(max(abs(temp - rolling_mean)), 2),
                            "timestamps": anomalies['time'].tolist()
                        })
        
        elif component in ['VGR', 'HBW']:
            # Check for unusual movement patterns
            position_cols = [col for col in df.columns if any(pos in col for pos in 
                            ['pos_horizontal', 'pos_vertical', 'pos_rotation'])]
            
            for col in position_cols:
                if col in df.columns:
                    values = df[col].dropna()
                    if len(values) > 10:
                        # Look for rapid changes (jerky movements)
                        deltas = values.diff().abs()
                        mean_delta = deltas.mean()
                        std_delta = deltas.std()
                        
                        rapid_changes = df[deltas > mean_delta + 2 * std_delta]
                        if len(rapid_changes) > 0:
                            insights.append({
                                "type": "rapid_movement",
                                "component": component,
                                "axis": col,
                                "description": f"Unusually rapid movement detected in {col}",
                                "count": len(rapid_changes),
                                "timestamps": rapid_changes['time'].tolist()
                            })
        
        return insights
    
    def query(self, query: str) -> str:
        """
        Process a natural language query about visualization and analytics.
        
        Args:
            query: Natural language query about data visualization or analytics
            
        Returns:
            Response with visualization insights and metrics
        """
        # Parse time range from query first
        parsed_time_range = self._parse_time_range_from_query(query)
        
        # Debug output to show what time range is being used
        if parsed_time_range:
            print(f"Using parsed time range: {parsed_time_range}")
        else:
            print(f"No time range parsed from query, using default: {self.default_time_range}")
        
        # Load data for all components with the parsed or default time range
        self.load_component_data(self.available_components, time_range=parsed_time_range)
        
        # Parse query to identify relevant components and metrics
        components_mentioned = []
        for component in self.available_components:
            if component.lower() in query.lower():
                components_mentioned.append(component)
                
        # If no specific components mentioned, include the factory overview
        if not components_mentioned:
            components_mentioned = ['FACTORY']
            
        # Collect metrics for mentioned components
        all_metrics = {}
        insights = []
        
        for component in components_mentioned:
            metrics = self.get_component_metrics(component)
            all_metrics[component] = metrics
            
            # Get insights for this component
            component_insights = self.get_anomaly_insights(component)
            insights.extend(component_insights)
            
        # Format response based on the metrics and insights
        time_range_info = ""
        if parsed_time_range:
            start_time, end_time = parsed_time_range
            time_range_info = f"\n**Time Range:** {start_time.strftime('%Y-%m-%d %H:%M:%S')} to {end_time.strftime('%Y-%m-%d %H:%M:%S')}\n"
        
        response = f"## Analytics for {', '.join(components_mentioned)}{time_range_info}\n"
        
        for component in components_mentioned:
            response += f"### {component} - {self.get_component_description(component)}\n\n"
            
            if component in all_metrics:
                metrics = all_metrics[component]
                
                if component == 'FACTORY':
                    response += f"**Overall Factory Metrics:**\n"
                    response += f"- Average Activity: {metrics.get('avg_activity_pct', 0)}%\n"
                    response += f"- Active Time: {metrics.get('active_time_minutes', 0)} minutes\n\n"
                    
                    response += "**Component Activity:**\n"
                    for key, value in metrics.items():
                        if key.endswith('_active_pct') and not key.startswith('avg'):
                            component_name = key.replace('_active_pct', '')
                            response += f"- {component_name}: {value}% active\n"
                    
                elif component == 'ENVIRONMENTAL':
                    response += "**Environmental Conditions:**\n"
                    
                    # Temperature
                    if 'factory_temperature_raw_avg' in metrics:
                        response += (f"- Temperature: {metrics['factory_temperature_raw_avg']}°C "
                                   f"(Range: {metrics['factory_temperature_raw_min']}°C - {metrics['factory_temperature_raw_max']}°C)\n")
                    
                    # Humidity
                    if 'factory_humidity_raw_avg' in metrics:
                        response += (f"- Humidity: {metrics['factory_humidity_raw_avg']}% "
                                   f"(Range: {metrics['factory_humidity_raw_min']}% - {metrics['factory_humidity_raw_max']}%)\n")
                    
                    # Air Quality
                    if 'factory_index_air_quality_avg' in metrics:
                        response += (f"- Air Quality Index: {metrics['factory_index_air_quality_avg']} "
                                   f"(Range: {metrics['factory_index_air_quality_min']} - {metrics['factory_index_air_quality_max']})\n")
                    
                    # Air Pressure
                    if 'factory_air_pressure_avg' in metrics:
                        response += (f"- Air Pressure: {metrics['factory_air_pressure_avg']} hPa "
                                   f"(Range: {metrics['factory_air_pressure_min']} - {metrics['factory_air_pressure_max']} hPa)\n")
                
                else:
                    # For other components, show status and position data
                    status_metrics = {k: v for k, v in metrics.items() if 'active_pct' in k}
                    position_metrics = {k: v for k, v in metrics.items() if any(x in k for x in ['_avg', '_min', '_max', '_range'])}
                    
                    if status_metrics:
                        response += "**Activity Status:**\n"
                        for key, value in status_metrics.items():
                            status_name = key.replace('_active_pct', '').replace('_', ' ').title()
                            response += f"- {status_name}: {value}% active\n"
                    
                    if position_metrics:
                        response += "\n**Position Data:**\n"
                        # Group metrics by position type
                        pos_groups = {}
                        for key, value in position_metrics.items():
                            base_name = '_'.join(key.split('_')[:-1])  # Remove _avg, _min, etc.
                            metric_type = key.split('_')[-1]  # avg, min, max, range
                            
                            if base_name not in pos_groups:
                                pos_groups[base_name] = {}
                            pos_groups[base_name][metric_type] = value
                        
                        # Format each position metric
                        for base_name, values in pos_groups.items():
                            if 'avg' in values:
                                display_name = base_name.replace('_', ' ').title()
                                response += f"- {display_name}: {values.get('avg')} "
                                
                                if 'min' in values and 'max' in values:
                                    response += f"(Range: {values.get('min')} - {values.get('max')})"
                                
                                response += "\n"
            
            # Add insights
            component_insights = [insight for insight in insights if insight.get('component', component) == component]
            if component_insights:
                response += "\n**Insights & Anomalies:**\n"
                for insight in component_insights:
                    response += f"- {insight['description']}\n"
                    if 'count' in insight:
                        response += f"  - Occurrences: {insight['count']}\n"
                    if 'duration_seconds' in insight:
                        response += f"  - Duration: {insight['duration_seconds']} seconds\n"
            
            response += "\n"
            
        return response
    
    def get_visualization_urls(self, component: str) -> Dict[str, str]:
        """
        Get URLs for visualizations that can be embedded in the UI.
        
        In a real implementation, this might generate URLs to visualization endpoints
        or return configuration for embedded visualizations.
        
        Args:
            component: Component name to visualize
            
        Returns:
            Dictionary of visualization names mapped to their URLs/configs
        """
        # This is a placeholder that would be replaced with actual visualization URLs
        # In a real implementation, these might point to Streamlit components or external dashboards
        visualizations = {
            "activity_chart": f"/visualize?component={component}&chart=activity",
            "metrics_dashboard": f"/visualize?component={component}&chart=metrics",
            "anomaly_detection": f"/visualize?component={component}&chart=anomalies"
        }
        
        return visualizations
    
    def _parse_time_range_from_query(self, query: str) -> Optional[Tuple[datetime, datetime]]:
        """
        Parse date and time information from a natural language query.
        
        For relative time expressions (e.g., "last 10 minutes", "past hour"), 
        this method uses the current time as reference to support future 
        integration with real-time databases.
        
        For absolute dates (e.g., "2/12/2022"), it uses the specified date.
        
        Args:
            query: Natural language query that may contain date/time information
            
        Returns:
            Tuple of (start_time, end_time) if found, None otherwise
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
            # If no specific date found, return None to use default
            return None
            
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
                return None
        elif len(groups) == 3:
            # Check if it's DD/MM/YYYY or YYYY-MM-DD format
            if len(groups[0]) == 4:  # YYYY-MM-DD
                year, month, day = int(groups[0]), int(groups[1]), int(groups[2])
            else:  # DD/MM/YYYY or DD-MM-YYYY
                day, month, year = int(groups[0]), int(groups[1]), int(groups[2])
        else:
            return None
            
        try:
            base_date = datetime(year, month, day)
        except ValueError:
            # Invalid date
            return None
        
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
        
        return None
