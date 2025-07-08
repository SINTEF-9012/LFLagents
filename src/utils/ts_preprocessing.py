import os
import pandas as pd
import numpy as np

# Path to the factory data directory - use absolute path relative to this file
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
FACTORY_DIR = os.path.join(project_root, 'factory_data')

FACTORY_COMPONENT_CONFIG = {
    'FACTORY': {  # Factory overview
        'files': [
            'vgr_status_code.csv', 'vgr_actual_pos_horizontal.csv', 'vgr_actual_pos_rotation.csv', 'vgr_actual_pos_vertical.csv', 'vgr_target_status.csv',
            'hbw_status_code.csv', 'hbw_actual_pos_horizontal.csv', 'hbw_actual_pos_vertical.csv',
            'mpo_status_code.csv',
            'sld_status_code.csv',
            'dps_dsi_status_code.csv',
            'dps_dso_status_code.csv',
        ],
        'description': 'Factory Overview (VGR, HBW, MPO, SLD, DSO)'
    },
    'DPS': {  # Deliver and Pick-up Station
        'files': ['dps_dsi_active_status.csv', 'dps_dsi_status_code.csv', 
                 'dps_dso_active_status.csv', 'dps_dso_status_code.csv'],
        'description': 'Deliver and Pick-up Station'
    },
    'SSC': {  # Environmental station with surveillance camera
        'files': ['ssc_actual_pos_horizontal.csv', 'ssc_actual_pos_vertical.csv',
                 'ssc_target_pos_horizontal.csv', 'ssc_target_pos_vertical.csv'],
        'description': 'Environmental Station with Surveillance Camera'
    },
    'VGR': {  # Vacuum Gripper Robot
        'files': ['vgr_status_code.csv',
                 'vgr_actual_pos_horizontal.csv', 'vgr_actual_pos_rotation.csv', 'vgr_actual_pos_vertical.csv',
                 'vgr_target_pos_horizontal.csv', 'vgr_target_pos_rotation.csv', 'vgr_target_pos_vertical.csv',
                 'vgr_target_status.csv'],
        'description': 'Vacuum Gripper Robot'
    },
    'HBW': {  # Automated High-bay Warehouse
        'files': ['hbw_status_code.csv',
                 'hbw_actual_pos_horizontal.csv', 'hbw_actual_pos_vertical.csv',
                 'hbw_target_pos_horizontal.csv', 'hbw_target_pos_vertical.csv'],
        'description': 'Automated High-bay Warehouse'
    },
    'MPO': {  # Multi-Processing Station with kiln
        'files': ['mpo_status_code.csv'],
        'description': 'Multi-Processing Station with Kiln'
    },
    'SLD': {  # Sorting Line with Color Detection
        'files': ['sld_status_code.csv'],
        'description': 'Sorting Line with Color Detection'
    },
    'ENVIRONMENTAL': {  # Environmental sensors
        'files': ['factory_air_pressure.csv',
                  'factory_index_air_quality.csv',
                  'factory_gas_resistance.csv', 
                  'factory_humidity_raw.csv',
                  'factory_temperature_raw.csv'],
        'description': 'Environmental Sensors'
    },
    'CAMERA': {  # Camera positioning
        'files': ['camera_relative_position_pan.csv', 'camera_relative_position_tilt.csv'],
        'description': 'Camera Positioning System'
    },
}

def load_and_filter_data(file_path, time_range):
    """Load CSV data and filter by time range"""
    try:
        df = pd.read_csv(file_path)
        # Convert time column
        if 'time' in df.columns:
            df['time'] = pd.to_datetime(df['time'], format='mixed', utc=True)
            # Convert time_range to timezone-aware if needed
            start_time = pd.to_datetime(time_range[0]).tz_localize('UTC') if pd.to_datetime(time_range[0]).tz is None else pd.to_datetime(time_range[0])
            end_time = pd.to_datetime(time_range[1]).tz_localize('UTC') if pd.to_datetime(time_range[1]).tz is None else pd.to_datetime(time_range[1])
            # Filter by time range
            df = df[(df['time'] >= start_time) & (df['time'] <= end_time)]
        
        return df
    except Exception as e:
        print(f"Error loading {file_path}: {str(e)}")
        return None
    
def load_requested_component_data(component, time_range):
    """Load data for requested factory components"""
    data = {}
    if component not in FACTORY_COMPONENT_CONFIG:
        print(f"Component {component} not found in configuration.")
        print("Available components:", list(FACTORY_COMPONENT_CONFIG.keys()))
        return None
    
    info = FACTORY_COMPONENT_CONFIG[component]
    for file_name in info['files']:
        file_path = os.path.join(FACTORY_DIR, file_name)
        df = load_and_filter_data(file_path, time_range)
        if df is not None:
            file_key = file_name.replace('.csv', '').replace(f'{component.lower()}_', '')
            data[file_key] = df
    return data

def merge_and_resample_component_data(data, resample_interval='5S'):
    """Resample data and merge all data on time"""
    
    resampled_dfs = []
    
    for component, df in data.items():
        if df is not None and not df.empty and 'time' in df.columns and 'value' in df.columns:
            # Extract time and value columns
            clean_df = df[['time', 'value']].copy()
            
            # Set time as index and resample
            df_indexed = clean_df.set_index('time')
            
            # Get aggregation method for this data type using pattern matching
            if 'target_status' in component:
                agg_method = 'mode'
            elif 'status_code' in component or 'active_status' in component:
                agg_method = 'max'
            else:
                agg_method = 'mean'
            
            if agg_method == 'mode':
                # Use mode for categorical data (like target_status)
                resampled = df_indexed.resample(resample_interval).agg(
                    lambda x: x.mode()[0] if not x.mode().empty else np.nan
                ).reset_index()
            else:
                # Use specified aggregation method
                resampled = df_indexed.resample(resample_interval).agg(agg_method).reset_index()
            
            # Rename value column to component name
            resampled = resampled.rename(columns={'value': component})
            resampled_dfs.append(resampled)
    
    # Merge all resampled dataframes on time
    if resampled_dfs:
        merged_resampled = resampled_dfs[0]
        for df in resampled_dfs[1:]:
            merged_resampled = merged_resampled.merge(df, on='time', how='outer')
        return merged_resampled
    else:
        return None
   
def load_resampled_requested_components(components, time_range, resample_interval='5S'):
    """Load and process final data for a list of requested components"""
   
    data = {}
    for component in components:
        print(f"Loading data for component: {component}")
        component_data = load_requested_component_data(component, time_range)
        if component_data:
            print("Merging and resampling data...")
            resampled_data = merge_and_resample_component_data(component_data, resample_interval)
            if resampled_data is not None:
                data[component] = resampled_data
                print(f"Data for {component} loaded and resampled successfully.")
            else:
                print(f"No valid data found for component {component}.")
        else:
            print(f"Failed to load data for component {component}.")
    return data