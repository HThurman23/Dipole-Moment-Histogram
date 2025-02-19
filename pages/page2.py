# pages/page2.py
import streamlit as st
import plotly.graph_objects as go
from pyopenms import MSExperiment, MzMLFile
import tempfile
import numpy as np
import json
import os
import pandas as pd
from io import BytesIO
from typing import Tuple, List


MZ_RANGES_FILE = "saved_mz_ranges.json"

def save_mz_ranges(ranges, name):
    """Save m/z ranges to JSON file"""
    saved_ranges = load_all_saved_ranges()
    saved_ranges[name] = ranges
    
    with open(MZ_RANGES_FILE, 'w') as f:
        json.dump(saved_ranges, f, indent=4)

def load_all_saved_ranges():
    """Load all saved m/z range sets"""
    if os.path.exists(MZ_RANGES_FILE):
        with open(MZ_RANGES_FILE, 'r') as f:
            return json.load(f)
    return {}

@st.cache_resource
def load_mzml_file(file_path: str) -> MSExperiment:
    """Cache the mzML file loading to avoid reloading the same file"""
    experiment = MSExperiment()
    MzMLFile().load(file_path, experiment)
    return experiment

def extract_chromatogram(exp: MSExperiment, mz_min: float, mz_max: float) -> Tuple[List[float], List[float]]:
    """
    Vectorized and optimized version of chromatogram extraction
    """
    # Pre-allocate arrays for better memory efficiency
    num_spectra = exp.size()
    rt_list = np.zeros(num_spectra)
    intensity_list = np.zeros(num_spectra)
    
    # Process only MS1 spectra
    for i, spectrum in enumerate(exp):
        if spectrum.getMSLevel() == 1:
            rt_list[i] = spectrum.getRT() / 60.0  # Convert to minutes
            
            # Get all peaks at once as numpy arrays
            mz_array, int_array = spectrum.get_peaks()
            
            # Vectorized m/z range filtering
            mask = (mz_array >= mz_min) & (mz_array <= mz_max)
            intensity_list[i] = np.sum(int_array[mask])
    
    # Remove any zero entries if spectra were skipped
    mask = rt_list != 0
    return rt_list[mask].tolist(), intensity_list[mask].tolist()

def create_excel_download(rt_values, intensities, mz_ranges=None, is_summed=False):
    """Create Excel file with chromatogram data"""
    output = BytesIO()
    
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        if is_summed:
            # Summed chromatogram
            df = pd.DataFrame({
                'Retention Time (min)': rt_values,
                'Summed Intensity': intensities
            })
            df.to_excel(writer, sheet_name='Summed_Chromatogram', index=False)
        else:
            # Individual chromatograms
            for i, intensity in enumerate(intensities):
                df = pd.DataFrame({
                    'Retention Time (min)': rt_values,
                    f'Intensity ({mz_ranges[i]["min"]}-{mz_ranges[i]["max"]} m/z)': intensity
                })
                df.to_excel(writer, sheet_name=f'XIC_{i+1}', index=False)
    
    return output.getvalue()

def app():
    st.title("FAIMS XIC Calculator")
    
    # FAIMS Parameters in a dropdown
    with st.sidebar.expander("FAIMS Parameters", expanded=True):
        bias = st.number_input("Bias (V)", value=30.0)
        gap = st.number_input("Gap (mm)", value=0.188, format="%.3f")  # 3 decimal places
        scan_rate = st.number_input("Scan Rate (V/min)", value=1.0)
        cv_start = st.number_input("CV Start (V)", value=50.0)
        cv_end = st.number_input("CV End (V)", value=40.0)
        waveform_polarity = st.radio("Waveform Polarity", options=["Positive", "Negative"])

    def convert_rt_to_ec(retention_time, bias, cv_start, scan_rate, gap, polarity):
        if polarity == "Positive":
            return (bias - (cv_start - (retention_time * scan_rate))) / gap
        else:  # Negative polarity
            return ((cv_start - (retention_time * scan_rate)) - bias) / gap

    if 'mz_ranges' not in st.session_state:
        st.session_state.mz_ranges = [{'min': 1308.08, 'max': 1309.08}]
    
    if 'show_edit' not in st.session_state:
        st.session_state.show_edit = False
    
    uploaded_file = st.file_uploader("Upload mzML file", type="mzML")
    
    # m/z Range Controls (without dropdown)
    if st.sidebar.button("Toggle m/z Range Editor"):
        st.session_state.show_edit = not st.session_state.show_edit
    
    if st.session_state.show_edit:
        st.sidebar.markdown("### Edit m/z Ranges")
        
        col1, col2 = st.sidebar.columns([1, 1])
        with col1:
            if st.button("Add m/z Range"):
                st.session_state.mz_ranges.append({'min': 1308.08, 'max': 1309.08})
        with col2:
            if st.button("Remove Last Range") and len(st.session_state.mz_ranges) > 1:
                st.session_state.mz_ranges.pop()
        
        for i, mz_range in enumerate(st.session_state.mz_ranges):
            st.sidebar.markdown(f"#### Range {i+1}")
            col1, col2 = st.sidebar.columns(2)
            with col1:
                mz_range['min'] = st.number_input(
                    f"Min m/z #{i+1}", 
                    value=float(mz_range['min']), 
                    format="%.4f",
                    key=f"min_{i}"
                )
            with col2:
                mz_range['max'] = st.number_input(
                    f"Max m/z #{i+1}", 
                    value=float(mz_range['max']), 
                    format="%.4f",
                    key=f"max_{i}"
                )
    
    st.sidebar.markdown("### Save/Load m/z Ranges")
    
    save_name = st.sidebar.text_input("Name for current ranges:")
    if st.sidebar.button("Save Current Ranges") and save_name:
        save_mz_ranges(st.session_state.mz_ranges, save_name)
        st.sidebar.success(f"Saved ranges as '{save_name}'")
    
    saved_ranges = load_all_saved_ranges()
    if saved_ranges:
        selected_ranges = st.sidebar.selectbox(
            "Load saved ranges:",
            options=list(saved_ranges.keys()),
            key="load_ranges"
        )
        if st.sidebar.button("Load Selected Ranges"):
            st.session_state.mz_ranges = saved_ranges[selected_ranges]
            st.rerun()
    
        # Delete saved ranges
        to_delete = st.sidebar.selectbox(
            "Delete saved ranges:",
            options=list(saved_ranges.keys()),
            key="delete_ranges"
        )
        if st.sidebar.button("Delete Selected Ranges"):
            saved_ranges.pop(to_delete)
            with open(MZ_RANGES_FILE, 'w') as f:
                json.dump(saved_ranges, f, indent=4)
            st.rerun()

    if uploaded_file is not None:
        try:
            progress_text = st.empty()
            progress_bar = st.progress(0)
            
            # Step 1: File Loading
            progress_text.text("Loading mzML file...")
            progress_bar.progress(0.2)
            
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mzML") as tmp_file:
                tmp_file.write(uploaded_file.getbuffer())
                tmp_file_path = tmp_file.name

            # Use cached file loading
            experiment = load_mzml_file(tmp_file_path)
            
            # Pre-allocate lists for better memory efficiency
            all_rt = None
            all_intensities = [None] * len(st.session_state.mz_ranges)
            
            # Step 2: Extracting Chromatograms
            progress_text.text("Extracting ion chromatograms...")
            progress_bar.progress(0.4)
            
            # Process each m/z range
            for i, mz_range in enumerate(st.session_state.mz_ranges):
                rt, intensity = extract_chromatogram(experiment, mz_range['min'], mz_range['max'])
                if all_rt is None:
                    # Convert retention time to EC values
                    all_rt = np.array(rt)  # Convert to numpy array
                    all_ec = convert_rt_to_ec(
                        all_rt, 
                        bias, 
                        cv_start, 
                        scan_rate, 
                        gap, 
                        waveform_polarity
                    )
                all_intensities[i] = intensity
                
                # Update progress within extraction step
                sub_progress = 0.4 + (0.2 * (i + 1) / len(st.session_state.mz_ranges))
                progress_bar.progress(sub_progress)
            
            # Step 3: Creating Individual Plots
            progress_text.text("Creating individual chromatogram plots...")
            progress_bar.progress(0.6)
            
            fig_individual = go.Figure()
            for i, mz_range in enumerate(st.session_state.mz_ranges):
                fig_individual.add_trace(go.Scatter(
                    x=all_ec,  # Use EC values instead of retention time
                    y=all_intensities[i],
                    mode='lines',
                    name=f'XIC {i+1} ({mz_range["min"]}-{mz_range["max"]} m/z)'
                ))
            
            fig_individual.update_layout(
                template='plotly_dark',
                title='Individual Extracted Ion Chromatograms (XIC)',
                xaxis_title='EC (V/cm)',  # Updated x-axis label
                yaxis_title='Intensity',
                showlegend=True,
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(color='white')
            )
            
            # Step 4: Displaying Results
            progress_text.text("Displaying results...")
            progress_bar.progress(0.8)
            
            st.plotly_chart(fig_individual, use_container_width=True)
            
            excel_data_individual = create_excel_download(all_ec, all_intensities, st.session_state.mz_ranges)
            st.download_button(
                label="Download Individual XICs Data",
                data=excel_data_individual,
                file_name="individual_chromatograms.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
            
            if all_intensities:
                # Step 5: Creating Summed Plot
                progress_text.text("Creating summed chromatogram...")
                progress_bar.progress(0.9)
                
                fig_summed = go.Figure()
                summed_intensity = np.sum(all_intensities, axis=0)
                
                fig_summed.add_trace(go.Scatter(
                    x=all_ec,  # Use EC values instead of retention time
                    y=summed_intensity,
                    mode='lines',
                    name='Summed XIC'
                ))
                
                fig_summed.update_layout(
                    template='plotly_dark',
                    title='Summed Extracted Ion Chromatogram',
                    xaxis_title='EC (V/cm)',  # Updated x-axis label
                    yaxis_title='Intensity',
                    showlegend=True,
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    font=dict(color='white')
                )
                
                st.plotly_chart(fig_summed, use_container_width=True)
                
                excel_data_summed = create_excel_download(all_ec, summed_intensity, is_summed=True)
                st.download_button(
                    label="Download Summed XIC Data",
                    data=excel_data_summed,
                    file_name="summed_chromatogram.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )
            
            # Complete
            progress_text.text("Processing complete!")
            progress_bar.progress(1.0)
            
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    app()
