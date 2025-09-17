"""
Analytics Agent for Manufacturing Data Analysis.

This module provides comprehensive analytics capabilities for manufacturing data,
including time-series analysis, component performance monitoring, environmental
data processing, and advanced visualization features with LLM-powered explanations.
"""

import os
import sys
import pandas as pd
import numpy as np
import re
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple, Union
from functools import lru_cache
import logging

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(root_dir)

from utils.ts_preprocessing import (
    load_resampled_requested_components,
    FACTORY_COMPONENT_CONFIG,
)
from utils.llm_image_explanation import (
    generate_comprehensive_process_explanation,
    LLMImageAnalyzer,
    ChartImageSaver,
)
from utils.llm_config import get_llm_instance

# Import visualization functions
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from viz import (
    visualize_factory_overview,
    visualize_component_activity,
    visualize_environmental_data,
)


class AnalyticsAgent:
    """
    Performance-optimized Analytics Agent for manufacturing data visualization and analysis.

    This agent provides essential data processing capabilities:
    1. Loading and processing historical factory data with caching
    2. Generating metrics for dashboard display with optimized calculations
    3. Processing natural language queries with chart generation
    4. Providing LLM-powered process analysis

    Performance improvements:
    - Data caching to avoid redundant loading
    - Vectorized operations for metrics calculations
    - Pre-compiled regex patterns for query parsing
    - Memory-efficient data processing
    """

    # Pre-compile regex patterns for better performance
    _TIME_PATTERNS = [
        (re.compile(r'last\s+(\d+)\s+minutes?', re.IGNORECASE), "minutes"),
        (re.compile(r'last\s+(\d+)\s+hours?', re.IGNORECASE), "hours"),
        (re.compile(r'last\s+(\d+)\s+days?', re.IGNORECASE), "days"),
        (re.compile(r'past\s+(\d+)\s+minutes?', re.IGNORECASE), "minutes"),
        (re.compile(r'past\s+(\d+)\s+hours?', re.IGNORECASE), "hours"),
        (re.compile(r'past\s+(\d+)\s+days?', re.IGNORECASE), "days"),
        (re.compile(r'(\d+)\s+minutes?\s+ago', re.IGNORECASE), "minutes"),
        (re.compile(r'(\d+)\s+hours?\s+ago', re.IGNORECASE), "hours"),
        (re.compile(r'(\d+)\s+days?\s+ago', re.IGNORECASE), "days"),
    ]
    
    _DATE_PATTERNS = [
        re.compile(r'(\d{1,2})/(\d{1,2})/(\d{4})'),
        re.compile(r'(\d{4})-(\d{1,2})-(\d{1,2})'),
        re.compile(r'(\d{1,2})-(\d{1,2})-(\d{4})'),
        re.compile(r'(\d{1,2})(?:st|nd|rd|th)?\s+(?:of\s+)?([a-z]+)\s+(\d{4})', re.IGNORECASE),
        re.compile(r'([a-z]+)\s+(\d{1,2})(?:st|nd|rd|th)\s+(\d{4})', re.IGNORECASE),  # "december 1st 2022"
    ]

    # Pre-define column types for fast lookup
    _ENV_TYPES = frozenset(['temperature', 'humidity', 'air_quality', 'pressure', 'brightness', 'gas'])
    _POS_TYPES = frozenset(['pos_', 'position', 'actual_pos', 'target_pos'])
    
    # Environmental query keywords
    _ENV_KEYWORDS = frozenset([
        'environment', 'environmental', 'temperature', 'humidity', 'air',
        'pressure', 'gas', 'quality', 'sensor', 'condition'
    ])

    def __init__(self):
        """Initialize the Analytics Agent with performance optimizations."""
        self.component_config = FACTORY_COMPONENT_CONFIG
        self.available_components = list(self.component_config.keys())
        self.default_time_range = self._get_default_time_range()
        self.loaded_data = {}
        self._data_cache = {}  # Cache for processed data
        self._metrics_cache = {}  # Cache for calculated metrics
        self._cache_timestamps = {}  # Track cache validity
        
        self.charts_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "generated_charts",
        )
        os.makedirs(self.charts_dir, exist_ok=True)

        # Initialize LLM and Chart utilities (lazy loading)
        self._llm_analyzer = None
        self._chart_saver = None
        
        # Configure logging for performance monitoring
        self.logger = logging.getLogger(__name__)

    @property
    def llm_analyzer(self):
        """Lazy load LLM analyzer to improve startup time."""
        if self._llm_analyzer is None:
            self._llm_analyzer = LLMImageAnalyzer()
        return self._llm_analyzer

    @property
    def chart_saver(self):
        """Lazy load chart saver to improve startup time."""
        if self._chart_saver is None:
            self._chart_saver = ChartImageSaver()
        return self._chart_saver

    def clear_cache(self) -> None:
        """Clear all caches to free memory."""
        self._data_cache.clear()
        self._metrics_cache.clear()
        self._cache_timestamps.clear()
        
        # Clear LRU caches
        self.parse_time_range_from_query.cache_clear()
        self._detect_components_from_query_cached.cache_clear()
        
        self.logger.info("All caches cleared")

    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics for performance monitoring."""
        return {
            "data_cache_size": len(self._data_cache),
            "metrics_cache_size": len(self._metrics_cache),
            "parse_time_cache_info": self.parse_time_range_from_query.cache_info()._asdict(),
            "detect_components_cache_info": self._detect_components_from_query_cached.cache_info()._asdict(),
        }

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
            return self.component_config[component]["description"]
        return f"Unknown component: {component}"

    def load_component_data(
        self,
        components: List[str],
        time_range: Optional[Tuple[datetime, datetime]] = None,
        resample_interval: str = "5S",
        force_reload: bool = False,
    ) -> Dict[str, pd.DataFrame]:
        """
        Load component data for the specified time range with caching.

        Args:
            components: List of component names to load
            time_range: Optional time range tuple (start, end)
            resample_interval: Resampling interval (default: '5S')
            force_reload: Force reload data ignoring cache

        Returns:
            Dictionary with component names as keys and DataFrames as values
        """
        if time_range is None:
            time_range = self.default_time_range

        # Create cache key
        cache_key = (tuple(components), time_range, resample_interval)
        
        # Check cache first unless force_reload is True
        if not force_reload and cache_key in self._data_cache:
            cache_time = self._cache_timestamps.get(cache_key, 0)
            # Cache is valid for 1 hour for performance
            if time.time() - cache_time < 3600:
                cached_data = self._data_cache[cache_key]
                self.loaded_data.update(cached_data)
                return cached_data

        try:
            start_time = time.time()
            
            # Load data for requested components
            data_dict = load_resampled_requested_components(
                components, time_range, resample_interval
            )

            # Cache the result
            self._data_cache[cache_key] = data_dict
            self._cache_timestamps[cache_key] = time.time()
            
            # Store loaded data
            self.loaded_data.update(data_dict)
            
            load_time = time.time() - start_time
            self.logger.debug(f"Loaded data for {components} in {load_time:.3f}s")

            return data_dict

        except Exception as e:
            self.logger.error(f"Error loading component data: {e}")
            return {}

    def get_component_metrics(self, component: str) -> Dict[str, Any]:
        """
        Get metrics for a specific component with caching and optimized calculations.

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

        # Check cache first
        cache_key = (component, id(data))  # Use data object ID for cache key
        if cache_key in self._metrics_cache:
            return self._metrics_cache[cache_key]

        start_time = time.time()
        metrics = {}

        try:
            # Use vectorized operations for better performance
            if component == "ENVIRONMENTAL":
                metrics = self._calculate_environmental_metrics_vectorized(data)
            elif component == "FACTORY":
                metrics = self._calculate_factory_metrics_vectorized(data)
            else:
                metrics = self._calculate_component_metrics_vectorized(data)

            # Cache the result
            self._metrics_cache[cache_key] = metrics
            
            calc_time = time.time() - start_time
            self.logger.debug(f"Calculated metrics for {component} in {calc_time:.3f}s")

        except Exception as e:
            self.logger.error(f"Error calculating metrics for {component}: {e}")
            metrics["error"] = "Error calculating metrics"

        return metrics

    def _calculate_environmental_metrics_vectorized(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Calculate environmental metrics using vectorized operations."""
        metrics = {}
        
        # Find environmental columns efficiently
        env_columns = [col for col in data.columns 
                      if any(env_type in col.lower() for env_type in self._ENV_TYPES)]

        if not env_columns:
            return metrics

        # Vectorized calculations for all columns at once
        for col in env_columns:
            col_data = data[col]
            if col_data.notna().any():
                # Calculate all statistics in one pass
                stats = col_data.describe()
                metrics[f"{col}_avg"] = round(stats['mean'], 2)
                metrics[f"{col}_min"] = round(stats['min'], 2)
                metrics[f"{col}_max"] = round(stats['max'], 2)
                metrics[f"{col}_std"] = round(stats['std'], 2)

        return metrics

    def _calculate_factory_metrics_vectorized(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Calculate factory metrics using vectorized operations."""
        metrics = {}
        
        # Filter columns efficiently
        all_columns = [col for col in data.columns if not col.startswith("factory_")]

        if all_columns:
            # Find status code columns
            status_columns = [col for col in all_columns if "status_code" in col]
            
            if status_columns:
                # Vectorized activity calculation
                status_data = data[status_columns]
                activity_percentages = ((status_data != 1).mean() * 100).values
                
                if len(activity_percentages) > 0:
                    metrics["avg_activity_pct"] = round(np.mean(activity_percentages), 2)
                    metrics["active_time_minutes"] = round(len(data) * 5 / 60, 2)

        return metrics

    def _calculate_component_metrics_vectorized(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Calculate component-specific metrics using vectorized operations."""
        metrics = {}

        # Status columns - vectorized processing
        status_columns = [col for col in data.columns if "status_code" in col]
        if status_columns:
            status_data = data[status_columns]
            activity_percentages = (status_data != 1).mean() * 100
            
            for col in status_columns:
                metrics[f"{col}_active_pct"] = round(activity_percentages[col], 2)

        # Position columns - vectorized processing
        position_columns = [col for col in data.columns 
                           if any(pos_type in col for pos_type in self._POS_TYPES)]

        if position_columns:
            position_data = data[position_columns].select_dtypes(include=[np.number])
            
            for col in position_data.columns:
                if position_data[col].notna().any():
                    stats = position_data[col].describe()
                    metrics[f"{col}_avg"] = round(stats['mean'], 2)
                    metrics[f"{col}_min"] = round(stats['min'], 2)
                    metrics[f"{col}_max"] = round(stats['max'], 2)
                    metrics[f"{col}_range"] = round(stats['max'] - stats['min'], 2)

        return metrics

    def generate_data_summary(
        self, component: str, data: pd.DataFrame, time_range: Tuple[datetime, datetime]
    ) -> str:
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

            if component == "ENVIRONMENTAL":
                return base_summary + self._analyze_environmental_metrics(data)
            elif component == "FACTORY":
                return base_summary + self._analyze_factory_metrics(data)
            else:
                return base_summary + self._analyze_component_metrics(data)

        except Exception:
            return f"Error generating {component} summary"

    def _analyze_environmental_metrics(self, data: pd.DataFrame) -> str:
        """Analyze environmental data metrics."""
        metrics_text = "**Environmental Metrics:**"
        env_columns = [
            col
            for col in data.columns
            if any(
                env_type in col.lower()
                for env_type in [
                    "temperature",
                    "humidity",
                    "air_quality",
                    "pressure",
                    "brightness",
                    "gas",
                ]
            )
        ]

        for col in env_columns:
            if data[col].notna().any():
                avg_val = data[col].mean()
                min_val = data[col].min()
                max_val = data[col].max()
                std_val = data[col].std()
                metric_name = col.replace("factory_", "").replace("_", " ").title()
                metrics_text += f"\n- {metric_name}: Avg {avg_val:.2f}, Range {min_val:.2f}-{max_val:.2f}, Std {std_val:.2f}"

        return metrics_text

    def _analyze_factory_metrics(self, data: pd.DataFrame) -> str:
        """Analyze factory overview metrics with cycle analysis."""
        metrics_text = "**Component Activity Analysis:**"

        # Component activity analysis
        component_activity = {}
        for col in data.columns:
            if "status_code" in col:
                component_name = col.split("_")[0].upper()
                if component_name not in component_activity:
                    activity_pct = (data[col] != 1).mean() * 100
                    component_activity[component_name] = activity_pct

        for comp, pct in component_activity.items():
            metrics_text += f"\n- {comp}: {pct:.1f}% active time"

        # Position movement analysis
        metrics_text += "\n\n**Position Movement Analysis:**"
        position_movements = {}
        for col in data.columns:
            if any(pos_type in col for pos_type in ["actual_pos", "target_pos"]):
                component_name = col.split("_")[0].upper()
                if component_name not in position_movements:
                    position_movements[component_name] = {
                        "min_pos": data[col].min(),
                        "max_pos": data[col].max(),
                        "movement_range": data[col].max() - data[col].min(),
                    }

        for comp, pos_data in position_movements.items():
            metrics_text += f"\n- {comp}: Range {pos_data['min_pos']:.0f}-{pos_data['max_pos']:.0f}, Movement: {pos_data['movement_range']:.0f}"

        # Process cycle analysis
        cycle_analysis = self._analyze_process_cycles(data)
        metrics_text += "\n\n**Process Cycle Analysis:**"
        metrics_text += f"\n- Detected Process Cycles: {cycle_analysis['cycle_count']}"
        metrics_text += f"\n- Average Cycle Duration: {cycle_analysis['avg_cycle_duration']:.1f} minutes"
        metrics_text += f"\n- Overall Process Type: {cycle_analysis['cycle_type']}"

        if cycle_analysis["cycle_details"]:
            metrics_text += "\n\n**Individual Cycle Details:**"
            for cycle in cycle_analysis["cycle_details"]:
                metrics_text += f"\n- Cycle {cycle['cycle_number']}: ({cycle['duration_minutes']}min) - {cycle['process_type']}"
                if cycle["active_components"]:
                    metrics_text += (
                        f" [Components: {', '.join(cycle['active_components'])}]"
                    )

        return metrics_text

    def _analyze_component_metrics(self, data: pd.DataFrame) -> str:
        """Analyze individual component metrics."""
        metrics_text = "**Component Activity:**"

        # Status code analysis
        status_columns = [col for col in data.columns if "status_code" in col]
        for col in status_columns:
            if data[col].notna().any():
                activity_pct = (data[col] != 1).mean() * 100
                metrics_text += f"\n- {col.replace('_', ' ').title()}: {activity_pct:.1f}% active time"

        # Position analysis
        position_columns = [
            col
            for col in data.columns
            if any(
                pos_type in col
                for pos_type in ["pos_", "position", "actual_pos", "target_pos"]
            )
        ]

        if position_columns:
            metrics_text += "\n\n**Position Analysis:**"
            for col in position_columns:
                if data[col].notna().any():
                    avg_pos = data[col].mean()
                    min_pos = data[col].min()
                    max_pos = data[col].max()
                    movement_range = max_pos - min_pos
                    pos_name = col.replace("_", " ").title()
                    metrics_text += f"\n- {pos_name}: Avg {avg_pos:.0f}, Range {min_pos:.0f}-{max_pos:.0f}, Movement {movement_range:.0f}"

        return metrics_text

    def _analyze_process_cycles(self, factory_data: pd.DataFrame) -> Dict[str, Any]:
        """
        Optimized cycle detection with vectorized operations and improved algorithms.

        Args:
            factory_data: Factory data DataFrame

        Returns:
            Dictionary with cycle analysis results including per-cycle types
        """
        cycle_analysis = {
            "cycle_count": 0,
            "avg_cycle_duration": 0,
            "cycle_type": "Unknown",
            "cycle_details": [],
        }

        try:
            # Pre-define component columns for faster lookup
            component_columns = {
                "vgr": "vgr_status_code",
                "hbw": "hbw_status_code",
                "mpo": "mpo_status_code",
                "sld": "sld_status_code",
            }

            # Find available components efficiently
            available_components = {
                comp_name: col_name 
                for comp_name, col_name in component_columns.items()
                if col_name in factory_data.columns
            }

            if not available_components:
                return cycle_analysis

            # Vectorized activity detection - much faster than individual column processing
            status_cols = list(available_components.values())
            status_data = factory_data[status_cols]
            
            # Create combined activity signal using vectorized operations
            combined_activity = (status_data != 1).any(axis=1)

            # Fast transition detection using diff and boolean indexing
            activity_changes = combined_activity.diff()
            process_starts = factory_data.index[activity_changes & combined_activity].tolist()

            # If no clear transitions, find continuous active periods
            if not process_starts and combined_activity.sum() > 0:
                process_starts = self._find_active_periods_vectorized(combined_activity, factory_data.index)

            # Filter close transitions efficiently (< 1 minute = 12 intervals)
            if process_starts:
                filtered_transitions = self._filter_close_transitions_vectorized(
                    process_starts, factory_data.index, min_intervals=12
                )

                cycle_analysis["cycle_count"] = len(filtered_transitions)

                if filtered_transitions:
                    # Vectorized cycle analysis
                    cycle_details = self._analyze_cycles_vectorized(
                        filtered_transitions, factory_data, available_components
                    )
                    cycle_analysis["cycle_details"] = cycle_details

                    # Calculate average duration
                    if len(filtered_transitions) > 1:
                        total_duration_minutes = len(factory_data) * 5 / 60
                        cycle_analysis["avg_cycle_duration"] = round(
                            total_duration_minutes / len(filtered_transitions), 1
                        )
                    elif len(filtered_transitions) == 1:
                        cycle_analysis["avg_cycle_duration"] = round(len(factory_data) * 5 / 60, 1)

                    # Determine overall process type using vectorized operations
                    cycle_analysis["cycle_type"] = self._determine_overall_process_type_vectorized(
                        factory_data, available_components
                    )

            return cycle_analysis

        except Exception as e:
            self.logger.error(f"Error in cycle analysis: {e}")
            cycle_analysis["cycle_count"] = 0
            return cycle_analysis

    def _find_active_periods_vectorized(self, combined_activity: pd.Series, index: pd.Index) -> List:
        """Find active periods using vectorized operations."""
        # Find start and end points of active periods
        activity_diff = combined_activity.astype(int).diff()
        starts = index[activity_diff == 1].tolist()
        
        # Handle case where data starts active
        if combined_activity.iloc[0]:
            starts.insert(0, index[0])
            
        return starts

    def _filter_close_transitions_vectorized(self, transitions: List, index: pd.Index, min_intervals: int) -> List:
        """Filter transitions that are too close together using vectorized operations."""
        if not transitions:
            return transitions
            
        filtered = [transitions[0]]
        
        # Convert to positions for faster computation
        positions = [index.get_loc(t) for t in transitions]
        
        for i in range(1, len(positions)):
            if positions[i] - positions[filtered[-1] if len(filtered) == 1 else 
                                      index.get_loc(filtered[-1])] >= min_intervals:
                filtered.append(transitions[i])
                
        return filtered

    def _analyze_cycles_vectorized(self, transitions: List, factory_data: pd.DataFrame, 
                                 available_components: Dict[str, str]) -> List[Dict[str, Any]]:
        """Analyze individual cycles using vectorized operations."""
        cycle_details = []
        
        for i, cycle_start in enumerate(transitions):
            cycle_num = i + 1
            
            # Determine cycle end efficiently
            if i + 1 < len(transitions):
                try:
                    next_pos = factory_data.index.get_loc(transitions[i + 1])
                    cycle_end = factory_data.index[max(0, next_pos - 1)]
                except (KeyError, IndexError):
                    cycle_end = factory_data.index[-1]
            else:
                cycle_end = factory_data.index[-1]
            
            # Extract cycle data
            cycle_data = factory_data.loc[cycle_start:cycle_end]
            
            # Calculate duration
            cycle_duration_minutes = len(cycle_data) * 5 / 60
            
            # Vectorized component activity detection
            active_components = []
            process_type = "Unknown Process"
            
            if available_components:
                status_cols = list(available_components.values())
                cycle_status = cycle_data[status_cols]
                
                # Check which components are active in this cycle
                for comp_name, col_name in available_components.items():
                    if (cycle_status[col_name] != 1).any():
                        active_components.append(comp_name.upper())
                
                # Determine process type based on active components
                process_type = self._determine_cycle_type_fast(active_components)
            
            cycle_info = {
                "cycle_number": cycle_num,
                "duration_minutes": round(cycle_duration_minutes, 1),
                "process_type": process_type,
                "active_components": active_components,
            }
            
            cycle_details.append(cycle_info)
        
        return cycle_details

    def _determine_cycle_type_fast(self, active_components: List[str]) -> str:
        """Fast cycle type determination using set operations."""
        active_set = set(comp.lower() for comp in active_components)
        
        if {'mpo', 'sld'}.intersection(active_set):
            return "Production Process"
        elif {'hbw', 'vgr'}.issubset(active_set):
            return "Storage Process"
        elif 'vgr' in active_set:
            return "Transfer/Movement Process"
        else:
            return "Unknown Process"

    def _determine_overall_process_type_vectorized(self, factory_data: pd.DataFrame, 
                                                  available_components: Dict[str, str]) -> str:
        """Determine overall process type using vectorized operations."""
        if not available_components:
            return "Unknown Process"
            
        # Vectorized activity check for all components
        activity_status = {}
        for comp_name, col_name in available_components.items():
            activity_status[comp_name] = (factory_data[col_name] != 1).any()
        
        # Use the same logic as cycle type determination
        if activity_status.get('mpo', False) or activity_status.get('sld', False):
            return "Production Process"
        elif activity_status.get('hbw', False) and activity_status.get('vgr', False):
            return "Storage Process"
        elif activity_status.get('vgr', False):
            return "Transfer/Movement Process"
        else:
            return "Mixed/Unknown Process"

    def _determine_cycle_type(
        self, cycle_data: pd.DataFrame, available_components: Dict[str, str]
    ) -> str:
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
            hbw_active = (
                "hbw" in available_components
                and (cycle_data[available_components["hbw"]] != 1).any()
            )
            mpo_active = (
                "mpo" in available_components
                and (cycle_data[available_components["mpo"]] != 1).any()
            )
            sld_active = (
                "sld" in available_components
                and (cycle_data[available_components["sld"]] != 1).any()
            )
            vgr_active = (
                "vgr" in available_components
                and (cycle_data[available_components["vgr"]] != 1).any()
            )

            # Determine process type based on active components
            if mpo_active or sld_active:
                return "Production Process"
            elif hbw_active and vgr_active:
                return "Storage Process"
            elif vgr_active:
                return "Transfer/Movement Process"
            else:
                return "Unknown Process"

        except Exception:
            return "Unknown Process"

    def _get_active_components_for_cycle(
        self, cycle_data: pd.DataFrame, available_components: Dict[str, str]
    ) -> List[str]:
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

        except Exception:
            return []

    def analyze_cycles(
        self,
        time_range_or_start: Optional[
            Union[Tuple[datetime, datetime], datetime]
        ] = None,
        end_time: Optional[datetime] = None,
    ) -> Dict[str, Any]:
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
        if "FACTORY" not in self.loaded_data or self.loaded_data["FACTORY"].empty:
            self.load_component_data(["FACTORY"], time_range)
        factory_data = self.loaded_data.get("FACTORY", None)
        if factory_data is None or factory_data.empty:
            return {"error": "No factory data available for cycle analysis"}
        return self._analyze_process_cycles(factory_data)

    def generate_process_data_summary(
        self, time_range: Optional[Tuple[datetime, datetime]] = None
    ) -> str:
        """Generate process data summary - wrapper for backwards compatibility."""
        if time_range is None:
            time_range = self.default_time_range

        if "FACTORY" not in self.loaded_data:
            self.load_component_data(["FACTORY"], time_range)

        factory_data = self.loaded_data.get("FACTORY")
        if factory_data is None or factory_data.empty:
            return "No factory data available for analysis"

        return self.generate_data_summary("FACTORY", factory_data, time_range)

    def query(
        self, query_text: str, time_range: Optional[Tuple[datetime, datetime]] = None
    ) -> Dict[str, Any]:
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
                time_range = self.parse_time_range_from_query(query_text)
                
            # Check if the parsed time range is outside our historical data range
            historical_start = datetime(2022, 12, 1)
            historical_end = datetime(2022, 12, 31)
            
            if time_range[0] > historical_end or time_range[1] < historical_start:
                # Time range is completely outside our historical data
                start_time, end_time = time_range
                time_range_str = f"{start_time.strftime('%Y-%m-%d %H:%M:%S')} to {end_time.strftime('%Y-%m-%d %H:%M:%S')}"
                
                return {
                    "error": "No data available for the requested time period.",
                    "guidance": f"The requested time range ({time_range_str}) is outside our historical data range (December 1-31, 2022). Please specify a time period within the available data range.",
                    "time_range_searched": time_range_str,
                    "available_data_range": "2022-12-01 to 2022-12-31",
                    "query": query_text,
                    "time_range": time_range,
                    "success": False,
                }

            # Detect query type and components
            query_type = self._classify_query_type(query_text)
            components_to_load = self._detect_components_from_query(query_text)

            # Load the appropriate data
            self.load_component_data(components_to_load, time_range)

            # Get the primary data for analysis
            primary_component = components_to_load[0]
            primary_data = self.loaded_data.get(primary_component)

            if primary_data is None or primary_data.empty:
                # Format time range for user-friendly display
                start_time, end_time = time_range
                time_range_str = f"{start_time.strftime('%Y-%m-%d %H:%M:%S')} to {end_time.strftime('%Y-%m-%d %H:%M:%S')}"

                # Create a more helpful error message
                component_name = primary_component.lower().replace("_", " ")
                error_message = f"No {component_name} activity detected during the specified time range."

                # Add guidance based on query type
                if query_type == "environmental":
                    guidance = "Try checking a different time period or verify that environmental sensors are active."
                else:
                    guidance = "Try expanding the time range or checking if the factory was operational during this period."

                return {
                    "error": error_message,
                    "guidance": guidance,
                    "time_range_searched": time_range_str,
                    "query": query_text,
                    "time_range": time_range,
                    "primary_component": primary_component,
                    "query_type": query_type,
                    "success": False,
                }

            # Generate data summary
            data_summary = self.generate_data_summary(
                primary_component, primary_data, time_range
            )

            # Only generate chart if there is data
            chart_path = None
            if (
                query_type == "process_analysis"
                and primary_data is not None
                and not primary_data.empty
            ):
                chart_path = self._save_chart_for_llm(primary_component, primary_data)

            # Generate response based on query type
            if query_type == "environmental":
                return {
                    "query": query_text,
                    "time_range": time_range,
                    "primary_component": primary_component,
                    "data_summary": data_summary,
                    "chart_path": chart_path,
                    "query_type": "environmental",
                    "llm_analysis": data_summary,  # For environmental, just return data summary
                    "success": True,
                }
            else:  # process_analysis
                # Generate LLM analysis for process queries
                llm_analysis = self._generate_llm_analysis(
                    chart_path, data_summary, query_text, primary_component
                )

                return {
                    "query": query_text,
                    "time_range": time_range,
                    "primary_component": primary_component,
                    "data_summary": data_summary,
                    "chart_path": chart_path,
                    "query_type": "process_analysis",
                    "llm_analysis": llm_analysis,
                    "success": True,
                }

        except Exception:
            return {
                "error": "Error processing query",
                "query": query_text,
                "time_range": time_range,
                "success": False,
            }

    def _classify_query_type(self, query_text: str) -> str:
        """Classify query as environmental or process analysis using optimized keyword matching."""
        query_lower = query_text.lower()

        # Use set intersection for faster keyword matching
        query_words = set(query_lower.split())
        if query_words.intersection(self._ENV_KEYWORDS):
            return "environmental"

        return "process_analysis"

    @lru_cache(maxsize=128)
    def _detect_components_from_query_cached(self, query_text: str) -> Tuple[str, ...]:
        """
        Cached version of component detection for repeated queries.
        Returns tuple for hashability in cache.
        """
        return tuple(self._detect_components_from_query_impl(query_text))

    def _detect_components_from_query(self, query_text: str) -> List[str]:
        """
        Detect what components to load based on the query content.
        Uses caching for better performance.
        """
        return list(self._detect_components_from_query_cached(query_text))

    def _detect_components_from_query_impl(self, query_text: str) -> List[str]:
        """
        Implementation of component detection with optimized keyword matching.

        Args:
            query_text: The user's natural language query

        Returns:
            List of component names to load, ordered by priority
        """
        query_lower = query_text.lower()
        query_words = set(query_lower.split())
        components_to_load = []

        # Environmental keywords - use set intersection for faster matching
        env_keywords = {'environmental', 'environment', 'temperature', 'humidity', 
                       'air', 'quality', 'pressure', 'gas', 'brightness'}
        if query_words.intersection(env_keywords):
            components_to_load.append("ENVIRONMENTAL")

        # Specific component keywords - optimized with direct lookups
        component_mappings = {
            'vgr': ["VGR"],
            'vacuum': ["VGR"],
            'gripper': ["VGR"],
            'hbw': ["HBW"],
            'warehouse': ["HBW"],
            'storage': ["HBW"],
            'mpo': ["MPO"],
            'processing': ["MPO"],
            'sld': ["SLD"],
            'sorting': ["SLD"],
        }

        for keyword, components in component_mappings.items():
            if keyword in query_lower:
                components_to_load.extend(components)

        # Default to factory overview if no specific component mentioned
        if not components_to_load:
            components_to_load.append("FACTORY")

        # Remove duplicates while preserving order
        seen = set()
        unique_components = []
        for comp in components_to_load:
            if comp not in seen:
                seen.add(comp)
                unique_components.append(comp)

        return unique_components

    def _save_chart_for_llm(self, component: str, data: pd.DataFrame) -> Optional[str]:
        """Save chart to file for LLM analysis."""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            chart_path = os.path.join(
                self.charts_dir, f"{component.lower()}_{timestamp}.png"
            )

            fig = self._create_chart_by_component(component, data)
            if fig is not None:
                success = self.chart_saver.save_chart(fig, chart_path)
                return chart_path if success else None
            return None
        except Exception:
            return None

    def _generate_llm_analysis(
        self,
        chart_path: Optional[str],
        data_summary: str,
        query_text: str,
        primary_component: str,
    ) -> str:
        """Generate LLM analysis for process queries."""
        if chart_path and os.path.exists(chart_path):
            return generate_comprehensive_process_explanation(
                chart_path, data_summary, None  # No custom prompt for standard queries
            )
        else:
            # Provide a more helpful response when no chart is available
            component_name = primary_component.lower().replace("_", " ")

            if "no data available" in data_summary.lower() or not data_summary.strip():
                return (
                    f"No {component_name} activity was detected for the requested time period. This could indicate:\n\n"
                    f"• The factory was not operational during this time\n"
                    f"• The {component_name} components were in idle state\n"
                    f"• Data collection may have been interrupted\n\n"
                    f"Consider checking a different time range or verifying system status."
                )
            else:
                return f"Based on the query '{query_text}', here's the {component_name} analysis:\n\n{data_summary}"

    @lru_cache(maxsize=64)
    def parse_time_range_from_query(self, query: str) -> Tuple[datetime, datetime]:
        """
        Optimized time range parsing with caching and pre-compiled patterns.

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

        # Handle relative time expressions using pre-compiled patterns
        for pattern, unit in self._TIME_PATTERNS:
            match = pattern.search(query_lower)
            if match:
                amount = int(match.group(1))

                # Use current time as reference for relative queries
                now = datetime.now()

                if unit == "minutes":
                    start_time = now - timedelta(minutes=amount)
                    end_time = now
                elif unit == "hours":
                    start_time = now - timedelta(hours=amount)
                    end_time = now
                elif unit == "days":
                    start_time = now - timedelta(days=amount)
                    end_time = now

                return (start_time, end_time)

        # Handle month-day range patterns
        month_day_pattern = re.compile(r'(\w+)\s+(\d{1,2})\s+to\s+(\w+)\s+(\d{1,2})', re.IGNORECASE)
        match = month_day_pattern.search(query_lower)

        if match:
            start_month = match.group(1)
            start_day = int(match.group(2))
            end_month = match.group(3)
            end_day = int(match.group(4))

            # Pre-defined month mapping for better performance
            month_map = {
                'january': 1, 'jan': 1, 'february': 2, 'feb': 2, 'march': 3, 'mar': 3,
                'april': 4, 'apr': 4, 'may': 5, 'june': 6, 'jun': 6,
                'july': 7, 'jul': 7, 'august': 8, 'aug': 8, 'september': 9, 'sep': 9,
                'october': 10, 'oct': 10, 'november': 11, 'nov': 11, 'december': 12, 'dec': 12
            }

            start_month_num = month_map.get(start_month)
            end_month_num = month_map.get(end_month)

            if start_month_num and end_month_num:
                try:
                    start_time = datetime(2022, start_month_num, start_day, 0, 0, 0)
                    end_time = datetime(2022, end_month_num, end_day, 23, 59, 59)
                    return (start_time, end_time)
                except ValueError:
                    pass

        # Use pre-compiled date patterns
        date_match = None
        date_format = None
        for i, pattern in enumerate(self._DATE_PATTERNS):
            match = pattern.search(query_lower)
            if match:
                date_match = match
                date_format = i
                break

        if not date_match:
            return self.default_time_range

        # Parse the date based on the pattern matched
        groups = date_match.groups()

        if date_format == 3:  # Ordinal date format "1st december 2022"
            day = int(groups[0])
            month_name = groups[1].lower()
            year = int(groups[2])

            month_names = {
                "january": 1, "jan": 1, "february": 2, "feb": 2, "march": 3, "mar": 3,
                "april": 4, "apr": 4, "may": 5, "june": 6, "jun": 6,
                "july": 7, "jul": 7, "august": 8, "aug": 8, "september": 9, "sep": 9,
                "october": 10, "oct": 10, "november": 11, "nov": 11, "december": 12, "dec": 12,
            }
            month = month_names.get(month_name)
            if not month:
                return self.default_time_range
        elif date_format == 4:  # "december 1st 2022" format
            month_name = groups[0].lower()
            day = int(groups[1])
            year = int(groups[2])

            month_names = {
                "january": 1, "jan": 1, "february": 2, "feb": 2, "march": 3, "mar": 3,
                "april": 4, "apr": 4, "may": 5, "june": 6, "jun": 6,
                "july": 7, "jul": 7, "august": 8, "aug": 8, "september": 9, "sep": 9,
                "october": 10, "oct": 10, "november": 11, "nov": 11, "december": 12, "dec": 12,
            }
            month = month_names.get(month_name)
            if not month:
                return self.default_time_range
        elif len(groups) == 3:
            if len(groups[0]) == 4:  # YYYY-MM-DD
                year, month, day = int(groups[0]), int(groups[1]), int(groups[2])
            else:  # DD/MM/YYYY or DD-MM-YYYY
                day, month, year = int(groups[0]), int(groups[1]), int(groups[2])
        else:
            return self.default_time_range

        try:
            base_date = datetime(year, month, day)
        except ValueError:
            return self.default_time_range

        # Optimized time range parsing
        start_time, end_time = self._parse_time_ranges_optimized(query_lower, base_date)
        
        if start_time and end_time:
            return (start_time, end_time)

        return self.default_time_range

    def _parse_time_ranges_optimized(self, query_lower: str, base_date: datetime) -> Tuple[Optional[datetime], Optional[datetime]]:
        """Optimized time range parsing with pre-compiled patterns."""
        # Pre-compiled time range patterns with more comprehensive coverage
        time_range_patterns = [
            # Standard "from HH:MM to HH:MM" format (24-hour)
            re.compile(r"from\s+(\d{1,2}):(\d{2})\s+to\s+(\d{1,2}):(\d{2})"),
            # Alternative "HH:MM to HH:MM" format
            re.compile(r"(\d{1,2}):(\d{2})\s+to\s+(\d{1,2}):(\d{2})"),
            # "between HH:MM and HH:MM" format
            re.compile(r"between\s+(\d{1,2}):(\d{2})\s+and\s+(\d{1,2}):(\d{2})"),
            # PM patterns
            re.compile(r"from\s+(\d{1,2})(?::(\d{2}))?\s*pm\s+to\s+(\d{1,2}):(\d{2})\s*pm"),
            re.compile(r"from\s+(\d{1,2})\s*pm\s+to\s+(\d{1,2}):(\d{2})\s*pm"),
        ]

        for i, pattern in enumerate(time_range_patterns):
            time_range_match = pattern.search(query_lower)
            if time_range_match:
                groups = time_range_match.groups()

                if i >= 3:  # PM patterns (indices 3+)
                    if len(groups) == 4 and groups[1] is not None:
                        start_hour, start_min, end_hour, end_min = map(int, groups)
                        if start_hour < 12:
                            start_hour += 12
                        if end_hour < 12:
                            end_hour += 12
                    elif len(groups) == 3:
                        start_hour, end_hour, end_min = int(groups[0]), int(groups[1]), int(groups[2])
                        start_min = 0
                        if start_hour < 12:
                            start_hour += 12
                        if end_hour < 12:
                            end_hour += 12
                    else:
                        continue
                else:  # 24-hour format patterns (indices 0-2)
                    if len(groups) == 4:
                        start_hour, start_min, end_hour, end_min = map(int, groups)
                    else:
                        continue

                start_time = base_date.replace(hour=start_hour, minute=start_min, second=0)
                end_time = base_date.replace(hour=end_hour, minute=end_min, second=0)
                return start_time, end_time

        # Handle "at HH:MM" pattern
        at_time_pattern = re.compile(r"at\s+(\d{1,2}):(\d{2})")
        at_time_match = at_time_pattern.search(query_lower)
        if at_time_match:
            hour, minute = map(int, at_time_match.groups())
            start_time = base_date.replace(hour=hour, minute=minute, second=0)
            end_time = start_time + timedelta(minutes=30)
            return start_time, end_time

        # Handle individual times
        time_pattern = re.compile(r"(\d{1,2}):(\d{2})")
        times_found = []
        for match in time_pattern.finditer(query_lower):
            hour, minute = int(match.group(1)), int(match.group(2))
            times_found.append(base_date.replace(hour=hour, minute=minute, second=0))

        if len(times_found) >= 2:
            return min(times_found), max(times_found)
        elif len(times_found) == 1:
            return times_found[0], times_found[0] + timedelta(hours=1)
        else:
            # No specific time found, use the whole day
            return base_date, base_date + timedelta(days=1)

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
        if any(
            keyword in query_lower
            for keyword in [
                "environmental",
                "environment",
                "temperature",
                "humidity",
                "air quality",
                "pressure",
                "gas",
                "brightness",
            ]
        ):
            components_to_load.append("ENVIRONMENTAL")

        # Specific component keywords
        if any(
            keyword in query_lower for keyword in ["vgr", "vacuum gripper", "gripper"]
        ):
            components_to_load.append("VGR")
        elif any(
            keyword in query_lower
            for keyword in ["hbw", "high bay warehouse", "warehouse", "storage"]
        ):
            components_to_load.append("HBW")
        elif any(
            keyword in query_lower
            for keyword in ["mpo", "multi processing", "processing"]
        ):
            components_to_load.append("MPO")
        elif any(
            keyword in query_lower for keyword in ["sld", "sorting line", "sorting"]
        ):
            components_to_load.append("SLD")

        # Default to factory overview if no specific component mentioned
        if not components_to_load:
            components_to_load.append("FACTORY")

        return components_to_load

    def get_interactive_chart(
        self, query_text: str, time_range: Optional[Tuple[datetime, datetime]] = None
    ):
        """
        Generate interactive chart based on query.

        Args:
            query_text: The user's natural language query
            time_range: Optional time range for the query

        Returns:
            Plotly figure object or None if generation fails
        """
        try:
            # Parse time range from query if not provided
            if time_range is None:
                time_range = self.parse_time_range_from_query(query_text)

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

        except Exception:
            return None

    def _create_chart_by_component(self, component: str, data: pd.DataFrame):
        """Create chart based on component type."""
        if component == "ENVIRONMENTAL":
            return visualize_environmental_data(data)
        elif component == "FACTORY":
            return visualize_factory_overview(data, "5S")
        else:
            return visualize_component_activity(data, component, "5S")
