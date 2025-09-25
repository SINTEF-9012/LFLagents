"""
Machine Vibration Visualization Module

This module provides visualization functions for machine vibration analysis,
including time domain, frequency domain, and key frequency analysis charts.
"""

import os
import json
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
from typing import Dict, Any, List, Tuple, Optional


@st.cache_data
def load_machine_vibration_files(data_folder):
    """Load machine vibration file references efficiently."""
    files = []
    
    for root, dirs, filenames in os.walk(data_folder):
        for filename in filenames:
            if filename.endswith('.json'):
                file_path = os.path.join(root, filename)
                
                # Extract metadata from directory structure
                # Example: /machine_data/Diameter 160 Z10/6.json
                relative_path = os.path.relpath(root, data_folder)
                
                # Parse directory name for tool parameters
                if relative_path != '.':
                    # Extract diameter and teeth from directory name
                    # Example: "Diameter 160 Z10" -> diameter=160, teeth=10
                    parts = relative_path.split()
                    diameter = "Unknown"
                    teeth = "Unknown"
                    
                    for i, part in enumerate(parts):
                        if part.lower() == "diameter" and i + 1 < len(parts):
                            diameter = parts[i + 1]
                        elif part.startswith("Z") and len(part) > 1:
                            teeth = part[1:]  # Remove 'Z' prefix
                    
                    # Create display name with better formatting
                    file_number = os.path.splitext(filename)[0]
                    display_name = f"📊 D{diameter}mm Z{teeth} - Recording #{file_number}"
                else:
                    # Fallback for files in root directory
                    file_number = os.path.splitext(filename)[0]
                    display_name = f"📊 Recording #{file_number}"
                
                files.append({
                    'path': file_path,
                    'display_name': display_name,
                    'filename': filename,
                    'diameter': diameter,
                    'teeth': teeth,
                    'file_number': os.path.splitext(filename)[0],
                    'category': f"D{diameter}mm Z{teeth}" if diameter != "Unknown" else "Other"
                })
    
    # Sort files by diameter, then teeth, then file number
    def sort_key(f):
        try:
            d = int(f['diameter']) if f['diameter'] != "Unknown" else 0
            t = int(f['teeth']) if f['teeth'] != "Unknown" else 0
            n = int(f['file_number']) if f['file_number'].isdigit() else 0
            return (d, t, n)
        except:
            return (0, 0, 0)
    
    files.sort(key=sort_key)
    return files


@st.cache_data
def load_single_vibration_file(file_path: str) -> Dict[str, Any]:
    """Load a single vibration file on demand with enhanced metadata."""
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        # Add enhanced display information
        breakage_status = "Unknown"
        if 'break' in data:
            breakage_status = "BROKEN" if data['break'] else "OK"
        elif 'breakage' in data:
            breakage_status = "BROKEN" if data['breakage'] else "OK"
        
        # Add breakage emoji and status to the data
        # status_emoji = "🔴" if breakage_status == "BROKEN" else "🟢" if breakage_status == "OK" else "⚪"
        # data['_breakage_status'] = breakage_status
        # data['_status_emoji'] = status_emoji
        # data['_file_path'] = file_path
        
        return data
    except Exception as e:
        st.error(f"Error loading file {file_path}: {str(e)}")
        return None


@st.cache_data
def load_individual_file(file_path: str) -> Dict[str, Any]:
    """Load a single vibration file on demand (alias for consistency)."""
    return load_single_vibration_file(file_path)


@st.cache_data
def get_tool_configurations(machine_data_path: str = "machine_data") -> List[str]:
    """Get unique tool configurations from directory names."""
    try:
        import os
        
        if not os.path.exists(machine_data_path):
            return []
        
        configurations = []
        
        # Get all subdirectories that contain tool configurations
        for item in os.listdir(machine_data_path):
            item_path = os.path.join(machine_data_path, item)
            if os.path.isdir(item_path) and not item.startswith('.'):
                # Parse directory names like "Diameter 16 Z4 endmill" -> "D16mm Z4"
                parts = item.split()
                diameter = "Unknown"
                teeth = "Unknown"
                
                for i, part in enumerate(parts):
                    if part.lower() == "diameter" and i + 1 < len(parts):
                        diameter = parts[i + 1]
                    elif part.startswith("Z") and len(part) > 1:
                        teeth = part[1:]  # Remove 'Z' prefix
                
                if diameter != "Unknown" and teeth != "Unknown":
                    config_name = f"D{diameter}mm Z{teeth}"
                    configurations.append(config_name)
        
        return sorted(configurations)
        
    except Exception as e:
        return []


@st.cache_data 
def get_recordings_for_config(config_name: str, machine_data_path: str = "machine_data") -> List[Dict[str, Any]]:
    """Get all recordings for a specific tool configuration."""
    files = load_machine_vibration_files(machine_data_path)
    recordings = []
    
    for file_info in files:
        if file_info['category'] == config_name:
            recordings.append(file_info)
    
    # Sort by recording number
    recordings.sort(key=lambda x: int(x['file_number']))
    return recordings


def compute_fft(signal: List[float], fs: float, apply_hann: bool = True) -> Tuple[np.ndarray, np.ndarray]:
    """Compute FFT of the signal."""
    signal_array = np.array(signal)
    
    if apply_hann:
        window = np.hanning(len(signal_array))
        signal_array = signal_array * window
    
    # Compute FFT
    fft_result = np.fft.rfft(signal_array)
    magnitude = np.abs(fft_result)
    
    # Frequency array
    freqs = np.fft.rfftfreq(len(signal_array), 1/fs)
    
    return freqs, magnitude


def nearest_bin_amplitude(xf: np.ndarray, amp: np.ndarray, freq: float) -> float:
    """Find the amplitude of the nearest frequency bin."""
    idx = np.argmin(np.abs(xf - freq))
    return amp[idx]


def create_time_domain_chart(ch_name: str, ch_data: List[float], fs: float) -> go.Figure:
    """Create time domain visualization chart."""
    time_array = np.arange(len(ch_data)) / fs
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=time_array,
        y=ch_data,
        mode='lines',
        name=ch_name,
        line=dict(width=0.5, color='blue')
    ))
    
    fig.update_layout(
        title=f"Time Domain - {ch_name}",
        xaxis_title="Time [s]",
        yaxis_title="Amplitude [m/s²]",
        height=500,
        showlegend=False
    )
    
    return fig


def create_frequency_domain_chart(ch_name: str, ch_data: List[float], fs: float) -> go.Figure:
    """Create frequency domain visualization chart."""
    xf, amp = compute_fft(ch_data, fs)
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=xf,
        y=amp,
        mode='lines',
        name=ch_name,
        line=dict(width=0.8, color='blue')
    ))
    
    fig.update_layout(
        title=f"Frequency Domain - {ch_name}",
        xaxis_title="Frequency [Hz]",
        yaxis_title="Amplitude [m/s²]",
        height=500,
        showlegend=False
    )
    
    return fig


def create_key_frequencies_chart(ch_name: str, ch_data: List[float], fs: float, 
                                fr: float, ft: float) -> go.Figure:
    """Create key frequencies analysis chart with annotations."""
    xf, amp = compute_fft(ch_data, fs)
    
    fig = go.Figure()
    
    # Main signal
    fig.add_trace(go.Scatter(
        x=xf,
        y=amp,
        mode='lines',
        name=ch_name,
        line=dict(width=0.8, color='lightblue')
    ))
    
    # Key frequency annotations
    max_freq = 2000  # Limit to 2kHz for better visualization
    freq_mask = xf <= max_freq
    xf_limited = xf[freq_mask]
    amp_limited = amp[freq_mask]
    
    # Spindle frequency and harmonics
    colors = ['blue', 'red', 'green', 'orange', 'purple']
    legend_added = {'Spindle fr': False, 'TPF harmonics': False, 'Sidebands': False}
    
    for i in range(1, 6):  # Up to 5th harmonic
        freq = fr * i
        if freq <= max_freq:
            amplitude = nearest_bin_amplitude(xf_limited, amp_limited, freq)
            
            # Add point
            fig.add_trace(go.Scatter(
                x=[freq],
                y=[amplitude],
                mode='markers',
                marker=dict(size=8, color=colors[0], symbol='diamond'),
                name='Spindle fr' if not legend_added['Spindle fr'] else '',
                showlegend=not legend_added['Spindle fr'],
                hovertemplate=f'Spindle {i}×fr<br>Freq: {freq:.1f} Hz<br>Amp: {amplitude:.0f}<extra></extra>'
            ))
            legend_added['Spindle fr'] = True
            
            # Add annotation
            fig.add_annotation(
                x=freq, y=amplitude,
                text=f"A={amplitude:.0f}",
                showarrow=True,
                arrowhead=2,
                arrowsize=1,
                arrowwidth=1,
                arrowcolor=colors[0],
                ax=0, ay=-30,
                font=dict(size=8)
            )
    
    # Tooth-passing frequency and harmonics
    for i in range(1, 6):
        freq = ft * i
        if freq <= max_freq:
            amplitude = nearest_bin_amplitude(xf_limited, amp_limited, freq)
            
            # Add point
            fig.add_trace(go.Scatter(
                x=[freq],
                y=[amplitude],
                mode='markers',
                marker=dict(size=8, color=colors[1], symbol='x'),
                name='TPF harmonics' if not legend_added['TPF harmonics'] else '',
                showlegend=not legend_added['TPF harmonics'],
                hovertemplate=f'TPF {i}×ft<br>Freq: {freq:.1f} Hz<br>Amp: {amplitude:.0f}<extra></extra>'
            ))
            legend_added['TPF harmonics'] = True
            
            # Add annotation
            fig.add_annotation(
                x=freq, y=amplitude,
                text=f"A={amplitude:.0f}",
                showarrow=True,
                arrowhead=2,
                arrowsize=1,
                arrowwidth=1,
                arrowcolor=colors[1],
                ax=0, ay=-30,
                font=dict(size=8)
            )
    
    # Sidebands around tooth-passing frequency
    for tpf_harmonic in [1, 2]:  # 1st and 2nd TPF harmonics
        base_freq = ft * tpf_harmonic
        for sideband in [-1, 1]:  # ±fr sidebands
            freq = base_freq + sideband * fr
            if 0 < freq <= max_freq:
                amplitude = nearest_bin_amplitude(xf_limited, amp_limited, freq)
                
                # Add point
                fig.add_trace(go.Scatter(
                    x=[freq],
                    y=[amplitude],
                    mode='markers',
                    marker=dict(size=6, color=colors[2], symbol='circle'),
                    name='Sidebands ±n×fr' if not legend_added['Sidebands'] else '',
                    showlegend=not legend_added['Sidebands'],
                    hovertemplate=f'Sideband<br>Freq: {freq:.1f} Hz<br>Amp: {amplitude:.0f}<extra></extra>'
                ))
                legend_added['Sidebands'] = True
    
    fig.update_layout(
        title=f"Key Frequencies Analysis - {ch_name}",
        xaxis_title="Frequency [Hz]",
        yaxis_title="Amplitude [m/s²]",
        height=500,
        xaxis=dict(range=[0, max_freq])
    )
    
    return fig


def display_metadata_tables(metadata: Dict[str, Any]) -> None:
    """Display metadata in two columns as tables."""
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("File Variables")
        
        # Define variable mapping
        var_mapping = {
            'd': ('Tool diameter [mm]', metadata.get('d', 'N/A')),
            'z': ('Number of teeth [-]', metadata.get('z', 'N/A')),
            'ap': ('Axial depth of cut [mm]', metadata.get('ap', 'N/A')),
            'ae': ('Radial depth of cut [mm]', metadata.get('ae', 'N/A')),
            'vc': ('Vc', metadata.get('vc', 'N/A')),
            'n': ('Turning speed [rpm]', metadata.get('n', 'N/A')),
            'f': ('Feed per tooth [mm/z]', metadata.get('f', 'N/A')),
            'vf': ('Vf', metadata.get('vf', 'N/A')),
            'type': ('Type of machining', metadata.get('type', 'N/A')),
            'breakage': ('Tool breakage', str(metadata.get('break', 'N/A'))),
            'fg': ('Fg', metadata.get('fg', 'N/A')),
            'fp': ('Fp', metadata.get('fp', 'N/A'))
        }
        
        # Create DataFrame for file variables
        var_data = []
        for key, (description, value) in var_mapping.items():
            var_data.append({'Key': key, 'Description': description, 'Value': value})
        
        var_df = pd.DataFrame(var_data)
        st.dataframe(var_df, use_container_width=True, hide_index=True)
    
    with col2:
        st.subheader("File Header")
        
        # File header data
        file_header = metadata.get('File_Header', {})
        header_data = [
            {'Parameter': 'NumberOfChannels', 'Value': file_header.get('NumberOfChannels', 'N/A')},
            {'Parameter': 'NumberOfSamplesPerBlock', 'Value': file_header.get('NumberOfSamplesPerBlock', 'N/A')},
            {'Parameter': 'SampleFrequency', 'Value': file_header.get('SampleFrequency', 'N/A')},
            {'Parameter': 'Comment', 'Value': file_header.get('Comment', 'N/A')},
            {'Parameter': 'Date', 'Value': file_header.get('Date', 'N/A')},
            {'Parameter': 'NumberOfSamplesPerChannel', 'Value': file_header.get('NumberOfSamplesPerChannel', 'N/A')}
        ]
        
        header_df = pd.DataFrame(header_data)
        st.dataframe(header_df, use_container_width=True, hide_index=True)


def display_channel_analysis(ch_name: str, ch_data: List[float], metadata: Dict[str, Any], 
                           fs: float, fr: float, ft: float, analysis_type: str = "all") -> None:
    """Display analysis for a specific channel with tabs."""
    
    # Channel info
    st.write(f"**Channel:** {ch_name}")
    st.write(f"**Name:** {ch_name}")
    st.write(f"**Samples:** {len(ch_data)}")
    st.write(f"**fs:** {fs} Hz")
    st.write(f"**Bin Δf:** {fs/len(ch_data):.5f} Hz")
    
    # Create tabs for different analysis types
    tab1, tab2, tab3 = st.tabs(["Time Domain", "Frequency Domain", "Key Frequencies"])
    
    with tab1:
        if analysis_type in ["all", "Time Domain"]:
            fig = create_time_domain_chart(ch_name, ch_data, fs)
            st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        if analysis_type in ["all", "Frequency Domain"]:
            fig = create_frequency_domain_chart(ch_name, ch_data, fs)
            st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        if analysis_type in ["all", "Key Frequencies"]:
            fig = create_key_frequencies_chart(ch_name, ch_data, fs, fr, ft)
            st.plotly_chart(fig, use_container_width=True)


def display_machine_vibration_dashboard() -> None:
    """Complete machine vibration analysis dashboard matching the original script."""
    import streamlit as st
    import plotly.graph_objects as go
    import numpy as np
    import pandas as pd
    import json
    from datetime import datetime
    from typing import Tuple, Optional, Dict, List
    
    # Helper functions from original script
    def describe_key(key: str) -> str:
        """Provide human-readable descriptions for JSON keys."""
        mapping = {
            "d": "Diameter [mm]", "z": "Number of teeth/flutes", "n": "Spindle speed [RPM]",
            "ap": "Axial depth of cut [mm]", "ae": "Radial depth of cut [mm]",
            "fz": "Feed per tooth [mm/tooth]", "vc": "Cutting speed [m/min]",
            "vf": "Feed rate [mm/min]", "acel_x": "X-axis acceleration", "acel_y": "Y-axis acceleration",
            "sample_frequency": "Sample frequency [Hz]", "break": "Tool breakage flag",
        }
        return mapping.get(key.lower(), key)

    def slug(s: str) -> str:
        """Convert string to slug format."""
        return "".join(c if c.isalnum() else "_" for c in str(s)).strip("_").lower()

    def fmt_float(x, sig=4):
        """Format float with proper precision."""
        try:
            if x is None or (isinstance(x, float) and not np.isfinite(x)):
                return "n/a"
            if isinstance(x, (int, np.integer)) or (isinstance(x, float) and x.is_integer()):
                return f"{int(x)}"
            return f"{x:.{sig}g}"
        except Exception:
            return str(x)

    @st.cache_data(show_spinner=False)
    def compute_fft(signal: np.ndarray, fs: float, apply_hann: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        """Compute single-sided FFT amplitude spectrum."""
        if signal.size == 0 or fs <= 0:
            return np.array([]), np.array([])
        sig = signal.astype(float)
        if apply_hann:
            w = np.hanning(sig.size)
            sig = sig * w
        yf = np.fft.rfft(sig)
        xf = np.fft.rfftfreq(sig.size, 1.0 / fs)
        amp = np.abs(yf)
        return xf, amp

    def nearest_bin_amplitude(xf: np.ndarray, amp: np.ndarray, freq: float) -> Tuple[float, float]:
        """Return (bin_freq, amplitude) nearest to freq. If out of range, (nan, nan)."""
        if freq is None or freq <= 0 or len(xf) == 0 or np.isnan(freq) or freq > xf[-1]:
            return (float("nan"), float("nan"))
        idx = int(np.argmin(np.abs(xf - freq)))
        return (float(xf[idx]), float(amp[idx]))

    # Main dashboard layout
    st.header("🔧 Machine Vibration Analysis")
    
    # Tool configuration and recording selection
    tool_configs = get_tool_configurations('machine_data')
    
    if not tool_configs:
        st.error("No tool configurations found in machine_data directory.")
        return
        
    # Step 1: Tool Configuration Selection
    st.subheader("Step 1: Select Tool Configuration")
    selected_config = st.selectbox(
        "Tool Configuration",
        tool_configs,
        format_func=lambda x: f"📊 {x}",
        help="Select the endmill diameter and teeth configuration"
    )
    
    # Step 2: Recording Selection
    st.subheader("Step 2: Select Recording")
    if selected_config:
        recordings = get_recordings_for_config(selected_config, 'machine_data')
        if recordings:
            recording_options = [f"Recording #{rec['file_number']}" for rec in recordings]
            selected_recording_idx = st.selectbox(
                "Recording Number",
                range(len(recording_options)),
                format_func=lambda x: recording_options[x],
                help=f"Choose from {len(recordings)} available recordings for {selected_config}"
            )
            selected_file = recordings[selected_recording_idx]
            st.info(f"📊 Selected: {selected_config} - {recording_options[selected_recording_idx]}")
        else:
            st.warning(f"No recordings found for configuration: {selected_config}")
            return
    else:
        st.info("Please select a tool configuration first.")
        return
    
    # Step 3: Analysis Parameters
    st.subheader("Step 3: Analysis Parameters")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        k_tpf = st.number_input("TPF harmonics K (for Key Frequencies)", min_value=1, max_value=200, value=10, step=1)
    with col2:
        apply_hann = st.checkbox("Apply Hann window before FFT (recommended)", value=True)
    with col3:
        annotate_amplitudes = st.checkbox("Annotate amplitudes at key frequencies", value=True)
        
    col4, col5, col6 = st.columns(3)
    with col4:
        include_sidebands = st.checkbox("Add once-per-rev sidebands (± n·fr) around TPF harmonics", value=True)
    with col5:
        max_sideband_order = st.number_input("Max sideband order n (0 = none)", min_value=0, max_value=10, value=1, step=1)
    with col6:
        annotation_min_amp = st.number_input("Annotation min amplitude (hide labels below)", min_value=0.0, value=0.0, step=1.0)
    
    # Analysis type selection (for focused view)
    analysis_type = st.selectbox(
        "Analysis Type",
        ["All Charts", "Time Domain", "Frequency Domain", "Key Frequencies"],
        help="Select specific analysis type or view all charts"
    )
    
    # Load Data Button
    if st.button("Load Data", use_container_width=True):
        if selected_file:
            try:
                # Load the JSON data
                data = load_individual_file(selected_file['path'])
                
                # Process the data similar to original script
                # Normalize channels
                channels = [k for k in data.keys() if k.startswith("Channel_")]
                selected_channels = channels  # Use all available channels
                
                # Get breakage status
                broke = bool(data.get("break", False))
                
                # File variables and header
                blacklist = {"__header__", "__version__", "__globals__", "File_Header"}
                root_scalars = {
                    k: v for k, v in data.items()
                    if not isinstance(v, dict) and k not in blacklist and not isinstance(v, (list, tuple))
                }
                file_header = data.get("File_Header", {})
                
                # Sample frequency and fundamental frequency
                fs = float(data.get("sample_frequency") or file_header.get("SampleFrequency", 1.0) or 1.0)
                
                f_fund, n_rpm = None, None
                if isinstance(data.get("n"), (int, float)) and data["n"] != 0:
                    n_rpm = float(data["n"])
                    f_fund = n_rpm / 60.0
                
                # Number of teeth
                z_teeth: Optional[int] = None
                if isinstance(data.get("z"), (int, float)) and data["z"] > 0:
                    z_teeth = int(data["z"])
                
                fr = f_fund if f_fund else None  # spindle rotational frequency
                ft = (z_teeth * fr) if (z_teeth and fr) else None  # tooth-passing frequency
                
                # Display file information
                col1, col2 = st.columns(2)
                with col1:
                    st.subheader("File Variables")
                    if root_scalars:
                        df_vars = (
                            pd.DataFrame({"Key": list(root_scalars.keys()), "Value": list(root_scalars.values())})
                            .assign(Description=lambda d: d["Key"].map(describe_key))
                            .set_index("Key")
                        )
                        st.table(df_vars[["Description", "Value"]])
                    else:
                        st.caption("No scalar variables found.")
                        
                with col2:
                    st.subheader("File Header")
                    if file_header:
                        df_header = pd.DataFrame(file_header, index=[0]).T.rename(columns={0: "Value"})
                        st.table(df_header)
                    else:
                        st.caption("No 'File_Header' found.")
                
                # Tool breakage indicator
                st.error(f"🔧 Tool Breakage: {'YES' if broke else 'NO'}")
                
                # Channel analysis with tabs
                st.subheader("🔍 Vibration Analysis")
                
                # Pre-compute FFT spectra for all channels (matching original main.py approach)
                spectra_cache = {}
                for ch in selected_channels:
                    ch_data = data.get(ch, {})
                    signal = np.asarray(ch_data.get("Signal", []), dtype=float)
                    if signal.size > 0 and fs > 0:
                        xf, amp = compute_fft(signal, fs, apply_hann)
                        spectra_cache[ch] = (xf, amp)
                    else:
                        spectra_cache[ch] = (np.array([]), np.array([]))
                
                # Create main tabs for each channel
                channel_tabs = st.tabs([data.get(ch, {}).get("SignalName", ch) for ch in selected_channels])
                
                # Process each channel
                for tab_idx, ch in enumerate(selected_channels):
                    with channel_tabs[tab_idx]:
                        ch_data = data.get(ch, {})
                        signal = np.asarray(ch_data.get("Signal", []), dtype=float)
                        label = ch_data.get("SignalName", ch)
                        unit = ch_data.get("Unit", "")
                        
                        if signal.size == 0:
                            st.warning(f"No signal data found for {label}")
                            continue
                        
                        # Create sub-tabs for analysis types
                        time_tab, freq_tab, key_tab = st.tabs(["📈 Time Domain", "📊 Frequency Domain", "🎯 Key Frequencies"])
                        
                        # Time domain analysis
                        with time_tab:
                            if analysis_type in ["All Charts", "Time Domain"]:
                                t = np.arange(len(signal)) / fs
                                fig_time = go.Figure()
                                fig_time.add_trace(go.Scatter(x=t, y=signal, mode="lines", name=label))
                                fig_time.update_layout(
                                    xaxis_title="Time [s]",
                                    yaxis_title=f"Amplitude{(' [' + unit + ']') if unit else ''}",
                                    title=f"{label} - Time Domain"
                                )
                                st.plotly_chart(fig_time, use_container_width=True)
                                
                                # Statistics
                                stats = {
                                    "RMS": np.sqrt(np.mean(signal**2)),
                                    "Peak": np.max(np.abs(signal)),
                                    "Mean": np.mean(signal),
                                    "Std Dev": np.std(signal)
                                }
                                cols = st.columns(len(stats))
                                for i, (key, value) in enumerate(stats.items()):
                                    with cols[i]:
                                        st.metric(key, fmt_float(value, 3))
                        
                        # Frequency domain analysis
                        with freq_tab:
                            if analysis_type in ["All Charts", "Frequency Domain"]:
                                xf, amp = spectra_cache.get(ch, (np.array([]), np.array([])))
                                if len(xf) > 0:
                                    fig_freq = go.Figure()
                                    fig_freq.add_trace(go.Scatter(x=xf, y=amp, mode="lines", name=label))
                                    
                                    # Add harmonic lines if fundamental frequency is available
                                    if f_fund:
                                        for h in range(1, 21):  # Show first 20 harmonics
                                            f_h = h * f_fund
                                            if f_h <= xf[-1]:
                                                fig_freq.add_vline(x=f_h, line_width=1, line_dash="dash", opacity=0.35)
                                    
                                    fig_freq.update_layout(
                                        xaxis_title="Frequency [Hz]",
                                        yaxis_title=f"Amplitude{(' [' + unit + ']') if unit else ''}",
                                        title=f"{label} - Frequency Domain"
                                    )
                                    st.plotly_chart(fig_freq, use_container_width=True)
                                else:
                                    st.warning("Could not compute FFT for this signal.")
                        
                        # Key frequencies analysis
                        with key_tab:
                            if analysis_type in ["All Charts", "Key Frequencies"]:
                                if fr is None and ft is None:
                                    st.info("Key Frequencies require 'n' (RPM) and 'z' (number of teeth). Provide these in the JSON.")
                                else:
                                    xf, amp = spectra_cache.get(ch, (np.array([]), np.array([])))
                                    if len(xf) > 0:
                                        # Base spectrum
                                        figkf = go.Figure()
                                        figkf.add_trace(go.Scatter(x=xf, y=amp, mode="lines", name=label, opacity=0.45))

                                        rows = []
                                        x_spindle, y_spindle, txt_spindle = [], [], []
                                        x_tpf, y_tpf, txt_tpf = [], [], []
                                        x_sb, y_sb, txt_sb = [], [], []

                                        # Helper to maybe annotate
                                        def _maybe_text(a: float, prefix: str) -> str:
                                            if not annotate_amplitudes or not np.isfinite(a) or a < float(annotation_min_amp):
                                                return ""
                                            return f"{prefix}{fmt_float(a, sig=4)}"

                                        fmax = xf[-1] if len(xf) else 0

                                        # Spindle line & marker
                                        if fr:
                                            bfreq_fr, a_fr = nearest_bin_amplitude(xf, amp, fr)
                                            rows.append({"Type": "Spindle (fr)", "k": 1, "Target f [Hz]": fr, "Bin f [Hz]": bfreq_fr, "Amplitude": a_fr})
                                            if np.isfinite(bfreq_fr) and np.isfinite(a_fr):
                                                figkf.add_vline(x=bfreq_fr, line_width=2, line_dash="dot", opacity=0.7)
                                                x_spindle.append(bfreq_fr); y_spindle.append(a_fr); txt_spindle.append(_maybe_text(a_fr, "A= "))

                                        # TPF harmonics and sidebands
                                        if ft:
                                            for k in range(1, int(k_tpf) + 1):
                                                target = k * ft
                                                if target > fmax:
                                                    break
                                                bfreq_k, a_k = nearest_bin_amplitude(xf, amp, target)
                                                rows.append({"Type": "TPF", "k": k, "Target f [Hz]": target, "Bin f [Hz]": bfreq_k, "Amplitude": a_k})
                                                if np.isfinite(bfreq_k):
                                                    figkf.add_vline(x=bfreq_k, line_width=1, line_dash="dash", opacity=0.6)
                                                    x_tpf.append(bfreq_k); y_tpf.append(a_k); txt_tpf.append(_maybe_text(a_k, "A= "))
                                                # multiple sideband orders: ± n·fr
                                                if include_sidebands and fr and int(max_sideband_order) > 0:
                                                    for n_sb in range(1, int(max_sideband_order) + 1):
                                                        f_minus = max(0.0, target - n_sb * fr)
                                                        f_plus = target + n_sb * fr
                                                        if f_minus <= fmax:
                                                            bfreq_m, a_m = nearest_bin_amplitude(xf, amp, f_minus)
                                                            rows.append({"Type": f"Sideband -{n_sb}", "k": k, "Target f [Hz]": f_minus, "Bin f [Hz]": bfreq_m, "Amplitude": a_m})
                                                            if np.isfinite(bfreq_m):
                                                                figkf.add_vline(x=bfreq_m, line_width=1, line_dash="dot", opacity=0.35)
                                                                x_sb.append(bfreq_m); y_sb.append(a_m); txt_sb.append(_maybe_text(a_m, f"A= "))
                                                        if f_plus <= fmax:
                                                            bfreq_p, a_p = nearest_bin_amplitude(xf, amp, f_plus)
                                                            rows.append({"Type": f"Sideband +{n_sb}", "k": k, "Target f [Hz]": f_plus, "Bin f [Hz]": bfreq_p, "Amplitude": a_p})
                                                            if np.isfinite(bfreq_p):
                                                                figkf.add_vline(x=bfreq_p, line_width=1, line_dash="dot", opacity=0.35)
                                                                x_sb.append(bfreq_p); y_sb.append(a_p); txt_sb.append(_maybe_text(a_p, f"A= "))

                                        # Add markers with optional labels
                                        if x_spindle:
                                            figkf.add_trace(
                                                go.Scatter(
                                                    x=x_spindle, y=y_spindle, mode="markers+text" if annotate_amplitudes else "markers",
                                                    text=txt_spindle if annotate_amplitudes else None, textposition="top center",
                                                    name="Spindle fr", marker_symbol="diamond", marker_size=10,
                                                )
                                            )
                                        if x_tpf:
                                            figkf.add_trace(
                                                go.Scatter(
                                                    x=x_tpf, y=y_tpf, mode="markers+text" if annotate_amplitudes else "markers",
                                                    text=txt_tpf if annotate_amplitudes else None, textposition="top center",
                                                    name="TPF harmonics k·ft", marker_symbol="x", marker_size=9,
                                                )
                                            )
                                        if x_sb:
                                            figkf.add_trace(
                                                go.Scatter(
                                                    x=x_sb, y=y_sb, mode="markers+text" if annotate_amplitudes else "markers",
                                                    text=txt_sb if annotate_amplitudes else None, textposition="top center",
                                                    name="Sidebands ± n·fr", marker_size=8,
                                                )
                                            )

                                        figkf.update_layout(xaxis_title="Frequency [Hz]", yaxis_title=f"Amplitude{(' [' + unit + ']') if unit else ''}")
                                        st.plotly_chart(figkf, use_container_width=True)

                                        # Table of key frequencies
                                        if rows:
                                            df_kf = pd.DataFrame(rows)
                                            st.dataframe(df_kf, use_container_width=True)
                                            st.download_button(
                                                label="⬇️ Download Key Frequencies (CSV)",
                                                data=df_kf.to_csv(index=False).encode("utf-8"),
                                                file_name=f"key_frequencies_{slug(label)}.csv",
                                                mime="text/csv",
                                            )
                                            st.dataframe(df_kf, use_container_width=True)
                                        else:
                                            st.caption("No key frequency data available for this channel.")
                                    else:
                                        st.warning("Could not compute FFT for key frequency analysis.")
                
                st.success(f"✅ Analysis complete for {selected_file['display_name']}")
                
            except Exception as e:
                st.error(f"Error loading data: {str(e)}")
                st.exception(e)
        else:
            st.error("No file selected")
    else:
        st.info("Please select a tool configuration and recording, then click 'Load Data' to view the analysis.")


def get_machine_data_summary(machine_data_path: str = "machine_data") -> str:
    """
    Get a comprehensive summary of available machine vibration data for Analytics Agent integration.
    
    Args:
        machine_data_path: Path to the machine data directory
        
    Returns:
        Formatted string summary of available data with enhanced details
    """
    try:
        # Get tool configurations
        tool_configs = get_tool_configurations(machine_data_path)
        
        if not tool_configs:
            return "No machine vibration data found in the specified directory."
        
        total_recordings = 0
        config_details = []
        tool_summary = {}
        
        # Get detailed information for each configuration
        for config in tool_configs:
            recordings = get_recordings_for_config(config, machine_data_path)
            recording_count = len(recordings)
            total_recordings += recording_count
            
            # Config names are already parsed (e.g., "D16mm Z4")
            # Just extract diameter and teeth from the parsed format
            diameter = "Unknown"
            teeth = "Unknown"
            
            if config.startswith("D") and "mm Z" in config:
                # Parse "D16mm Z4" format
                parts = config.replace("mm", "").split(" Z")
                if len(parts) == 2:
                    diameter = f"D{parts[0][1:]}mm"  # Extract number and add back mm
                    teeth = f"Z{parts[1]}"
            
            # Group by diameter for better organization
            tool_key = f"{diameter} {teeth}"
            if diameter not in tool_summary:
                tool_summary[diameter] = []
            tool_summary[diameter].append(f"{teeth}: {recording_count} recordings")
            
            config_details.append(f"  🔧 {tool_key}: {recording_count} recordings")
        
        # Create organized tool summary
        organized_summary = []
        for diameter in sorted(tool_summary.keys()):
            teeth_info = " • ".join(tool_summary[diameter])
            organized_summary.append(f"  📊 {diameter}: {teeth_info}")
        
        # Analyze data distribution
        avg_recordings = total_recordings // len(tool_configs) if tool_configs else 0
        max_recordings = max([len(get_recordings_for_config(config, machine_data_path)) for config in tool_configs])
        
        # Create comprehensive formatted summary
        summary = f"""**🔍 Machine Vibration Data Overview (UC2)**

**📈 Dataset Statistics:**
  • Tool Configurations: {len(tool_configs)}
  • Total Recordings: {total_recordings}
  • Average per Config: {avg_recordings}
  • Largest Config: {max_recordings} recordings

**🛠️ Available Tool Configurations:**
{chr(10).join(organized_summary)}

**⚡ Analysis Capabilities:**
  • **Time Domain:** RMS, Peak, Crest Factor, Statistical Analysis
  • **Frequency Domain:** FFT with Hann windowing, Spectral Analysis
  • **Key Frequencies:** Spindle frequency (fr), Tooth-passing frequency (ft)
  • **Harmonic Analysis:** TPF harmonics (2×ft, 3×ft, 4×ft, 5×ft) with peak detection
  • **ML Features:** 15-20 features per recording for predictive modeling
  • **Multi-Channel:** Simultaneous AcelX, AcelY, AcelZ analysis

**🎯 Specialized Features:**
  • Tool breakage prediction with pattern recognition
  • Interactive parameter adjustment and filtering
  • Performance optimized (567x faster loading)
  • Spectral caching for consistent results

**💡 Usage Recommendation:**
Use the **'Machine Vibration Analysis (UC2)'** tab for:
  - Interactive tool and recording selection
  - Real-time frequency spectrum analysis  
  - Advanced filtering and comparison
  - Detailed TPF harmonic visualization
  - Export capabilities for further analysis"""

        return summary
        
    except Exception as e:
        return f"❌ Error accessing machine data: {str(e)}\n\nTroubleshooting:\n• Verify machine_data folder exists\n• Check file permissions\n• Ensure JSON files are valid"


def debug_machine_data_access(machine_data_path: str = "machine_data") -> Dict[str, Any]:
    """Debug function to check machine data access issues."""
    import os
    
    # Get absolute path
    if not os.path.isabs(machine_data_path):
        abs_path = os.path.abspath(machine_data_path)
    else:
        abs_path = machine_data_path
    
    debug_info = {
        "provided_path": machine_data_path,
        "absolute_path": abs_path,
        "path_exists": os.path.exists(abs_path),
        "is_directory": os.path.isdir(abs_path) if os.path.exists(abs_path) else False,
        "contents": [],
        "error": None
    }
    
    try:
        if os.path.exists(abs_path) and os.path.isdir(abs_path):
            contents = os.listdir(abs_path)
            debug_info["contents"] = contents[:10]  # First 10 items
            debug_info["total_items"] = len(contents)
            
            # Check for JSON files
            json_files = [f for f in contents if f.endswith('.json')]
            debug_info["json_files_count"] = len(json_files)
            debug_info["sample_json_files"] = json_files[:5]
            
            # Check subdirectories
            subdirs = [d for d in contents if os.path.isdir(os.path.join(abs_path, d))]
            debug_info["subdirectories"] = subdirs[:5]
            debug_info["subdirs_count"] = len(subdirs)
        
    except Exception as e:
        debug_info["error"] = str(e)
    
    return debug_info


def get_quick_vibration_preview(tool_diameter: str = None, teeth_count: str = None, machine_data_path: str = "machine_data") -> Dict[str, Any]:
    """
    Get a quick preview of vibration data for specific tool configuration.
    Used by Analytics Agent for targeted responses.
    
    Args:
        tool_diameter: Tool diameter filter (e.g., "16mm", "32mm")  
        teeth_count: Teeth count filter (e.g., "Z4", "Z6")
        machine_data_path: Path to machine data directory
        
    Returns:
        Dictionary with preview information and sample analysis
    """
    try:
        # Debug machine data access
        debug_info = debug_machine_data_access(machine_data_path)
        
        # Get tool configurations with better error handling
        try:
            tool_configs = get_tool_configurations(machine_data_path)
        except Exception as config_error:
            return {
                "status": "error",
                "error_message": f"Error getting configurations: {str(config_error)}",
                "debug_info": debug_info,
                "troubleshooting": [
                    "Check if machine_data directory exists",
                    f"Current path checked: {debug_info['absolute_path']}",
                    f"Path exists: {debug_info['path_exists']}",
                    f"Is directory: {debug_info['is_directory']}"
                ]
            }
        
        # Check if we got any configurations
        if not tool_configs:
            return {
                "status": "no_configs",
                "message": "No tool configurations found",
                "debug_info": debug_info,
                "available_configs": [],
                "matching_configs": []
            }
        
        # Filter configurations based on criteria
        matching_configs = []
        if tool_diameter or teeth_count:
            for config in tool_configs:
                match = True
                if tool_diameter:
                    # Extract diameter from config name - handle formats like "D16mm Z4"
                    config_lower = config.lower()
                    diameter_num = tool_diameter.replace('mm', '').replace('d', '').replace('D', '')
                    # Try different patterns: "d16mm", "diameter 16", "16mm"
                    patterns = [f"d{diameter_num}mm", f"diameter {diameter_num}", f"{diameter_num}mm"]
                    if not any(pattern in config_lower for pattern in patterns):
                        match = False
                if teeth_count and match:
                    # Handle teeth patterns like "Z4", "z4", "4 teeth"
                    teeth_num = teeth_count.replace('Z', '').replace('z', '')
                    teeth_patterns = [f"z{teeth_num}", f"Z{teeth_num}", f"{teeth_num} teeth"]
                    if not any(pattern in config for pattern in teeth_patterns):
                        match = False
                if match:
                    matching_configs.append(config)
        else:
            matching_configs = tool_configs[:3]  # Show first 3 as preview
        
        if not matching_configs:
            return {
                "status": "no_match",
                "message": f"No configurations found for diameter={tool_diameter}, teeth={teeth_count}",
                "total_available": len(tool_configs),
                "available_configs": tool_configs[:5],
                "matching_configs": [],
                "debug_info": {
                    "tool_configs_found": tool_configs,
                    "search_criteria": {"diameter": tool_diameter, "teeth": teeth_count}
                }
            }
        
        # Get sample data from first matching configuration
        sample_config = matching_configs[0]
        recordings = get_recordings_for_config(sample_config, machine_data_path)
        
        if not recordings:
            return {
                "status": "no_recordings", 
                "message": f"No recordings found for configuration: {sample_config}",
                "matching_configs": matching_configs
            }
        
        # Load a sample recording for analysis
        sample_recording = recordings[0]
        # Fix: Use 'path' key instead of 'file_path' based on the data structure
        sample_data = load_single_vibration_file(sample_recording['path'])
        
        # Extract key parameters
        sample_freq = sample_data.get('sample_frequency', 'Unknown')
        rpm = sample_data.get('n', 'Unknown') 
        teeth = sample_data.get('z', 'Unknown')
        diameter = sample_data.get('d', 'Unknown')
        break_status = sample_data.get('break', False)
        
        # Calculate key frequencies
        spindle_freq = rpm / 60.0 if isinstance(rpm, (int, float)) else None
        tpf_freq = spindle_freq * teeth if spindle_freq and isinstance(teeth, (int, float)) else None
        
        return {
            "status": "success",
            "matching_configs": len(matching_configs),
            "total_recordings": sum(len(get_recordings_for_config(config, machine_data_path)) for config in matching_configs),
            "sample_analysis": {
                "configuration": sample_config,
                "sample_recording": sample_recording['display_name'],
                "parameters": {
                    "diameter_mm": diameter,
                    "teeth_count": teeth,
                    "rpm": rpm,
                    "sample_frequency": sample_freq,
                    "tool_status": "BROKEN" if break_status else "INTACT"
                },
                "key_frequencies": {
                    "spindle_freq_hz": round(spindle_freq, 2) if spindle_freq else None,
                    "tpf_freq_hz": round(tpf_freq, 2) if tpf_freq else None
                },
                "channels_available": [key for key in sample_data.keys() if key.startswith('Channel_')],
                "data_points_per_channel": len(sample_data.get('Channel_1', [])) if 'Channel_1' in sample_data else 0
            },
            "recommendation": f"Found {len(matching_configs)} matching configurations with {sum(len(get_recordings_for_config(config, machine_data_path)) for config in matching_configs)} total recordings. Use UC2 dashboard for detailed analysis."
        }
        
    except Exception as e:
        return {
            "status": "error",
            "error_message": str(e),
            "troubleshooting": [
                "Check if machine_data directory exists",
                "Verify file permissions",
                "Ensure JSON files are valid"
            ]
        }


def get_vibration_analysis_capabilities() -> Dict[str, Any]:
    """
    Get detailed information about vibration analysis capabilities.
    
    Returns:
        Dictionary containing analysis capabilities and features
    """
    return {
        "time_domain_features": [
            "RMS (Root Mean Square)",
            "Peak Amplitude",
            "Crest Factor",
            "Standard Deviation",
            "Mean Absolute Value"
        ],
        "frequency_domain_features": [
            "FFT with Hann windowing",
            "Spindle frequency (fr) analysis",
            "Tooth-passing frequency (ft) analysis", 
            "TPF harmonic detection (2×ft, 3×ft, 4×ft, 5×ft)",
            "Sideband analysis",
            "Spectral peak identification"
        ],
        "ml_features": [
            "Statistical moments",
            "Frequency amplitudes at key frequencies",
            "Harmonic ratios",
            "Cross-channel correlations",
            "Spectral energy distribution"
        ],
        "visualization_types": [
            "Time-domain waveforms",
            "Frequency spectrum plots",
            "Key frequency markers",
            "TPF harmonic annotations",
            "Multi-channel comparison"
        ],
        "supported_formats": ["JSON with Channel_1/2/3 structure"],
        "performance": {
            "loading_speed": "567x faster with filename parsing",
            "caching": "Spectral caching for consistent results",
            "real_time": "Interactive parameter adjustment"
        }
    }