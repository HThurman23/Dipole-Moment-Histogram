import streamlit as st
import pandas as pd
import numpy as np
import statsmodels.api as sm
import tempfile
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from numpy import trapz
import io

# Constants
FAIMS_ELECTRODE_GAP = 0.188  
K_B = 1.380649e-23  
T_ION = 298.5 
DEBYE_CONVERSION = 3.33564e-30  

# Theme
plotly_template = "ggplot2"

# Information
def show_calculation_info():
    with st.expander("Click here to learn how each calculation is made"):
        st.markdown("""
        **1. Electric Field Compensation (Ec):**
        - Calculation: `Ec = (bias - (cv_start - (time * scan_rate))) / FAIMS_ELECTRODE_GAP`
        - Explanation: This calculates the electric field compensation required based on input parameters.

        **2. Normalize Intensity:**
        - Calculation: `intensity / max(intensity)`
        - Explanation: This normalizes the intensity data to a scale from 0 to 1.

        **3. Smooth Data:**
        - Calculation: LOWESS smoothing using `statsmodels`
        - Explanation: Applies local regression smoothing to the intensity data.

        **4. Linear Regression:**
        - Calculation: `LinearRegression().fit(X, y)`
        - Explanation: Fits a linear model to the data points to understand the relationship between Ed and threshold Ec.

        **5. Dipole Moment Calculation:**
        - Calculation: `D_moment = (K_B * T_ION) / (2 * Ed * 1e8) / DEBYE_CONVERSION`
        - Explanation: Calculates the dipole moment based on temperature, Boltzmann constant, and Ed.
        """)


def calculate_ec(time, bias, cv_start, scan_rate):
    return (bias - (cv_start - (time * scan_rate))) / FAIMS_ELECTRODE_GAP


def normalize_intensity(intensity):
    return intensity / max(intensity)


def smooth_data(intensity, frac=0.03):
    return sm.nonparametric.lowess(intensity, np.arange(len(intensity)), frac=frac, it=0, return_sorted=False)


def find_rightmost_index(ec_values, intensity):
    half_max = max(intensity) / 2
    indices_above_half = np.where(intensity >= half_max)[0]
    if indices_above_half.any():
        return ec_values[indices_above_half[-1]]
    return None


def adjust_ec_values(ec_values, current_index, target_index, manual_shift):
    if current_index is not None:
        shift_value = target_index - current_index
        return ec_values + shift_value - manual_shift
    return ec_values


def process_data(file_path, num_sets, manual_shift, background_data=None):
    max_right_index = -np.inf
    processed_data = []

    for i, kv in enumerate(num_sets):
        metadata = pd.read_excel(file_path, nrows=3, header=None, usecols=[i*2, i*2+1])
        bias, cv_start, scan_rate = metadata.iloc[0, 1], metadata.iloc[1, 1], metadata.iloc[2, 1]

        data = pd.read_excel(file_path, skiprows=3, usecols=[i*2, i*2+1])
        ec_values = calculate_ec(data.iloc[:, 0], bias, cv_start, scan_rate)
        
        intensity = data.iloc[:, 1]
        
        if background_data is not None:
            background_intensity = background_data[i]
            intensity -= background_intensity

        smoothed_intensity = smooth_data(intensity)  

        normalized_intensity = normalize_intensity(smoothed_intensity)

        current_right_index = find_rightmost_index(ec_values, normalized_intensity)
        if current_right_index is not None and current_right_index > max_right_index:
            max_right_index = current_right_index

        processed_data.append((kv, ec_values, normalized_intensity, current_right_index))

    for i, (kv, ec_values, normalized_intensity, current_right_index) in enumerate(processed_data):
        adjusted_ec_values = adjust_ec_values(ec_values, current_right_index, max_right_index, manual_shift)
        processed_data[i] = (kv, adjusted_ec_values, normalized_intensity)

    return processed_data

def load_background_data(background_files, num_sets):
    background_data = []
    for i, bg_file in enumerate(background_files):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".xlsx") as tmp_file:
            tmp_file.write(bg_file.getbuffer())
            bg_file_path = tmp_file.name

        bg_intensities = []
        for j, kv in enumerate(num_sets):
            data = pd.read_excel(bg_file_path, skiprows=3, usecols=[j*2, j*2+1])
            smoothed_intensity = smooth_data(data.iloc[:, 1])
            bg_intensities.append(smoothed_intensity)

        background_data.append(bg_intensities)


    combined_background_data = []
    for j in range(len(num_sets)):
        combined_intensity = np.mean([bg[j] for bg in background_data], axis=0)
        combined_background_data.append(combined_intensity)

    return combined_background_data


def plot_data(processed_data):
    fig = go.Figure()

    for kv, ec_values, normalized_intensity in processed_data:
        fig.add_trace(go.Scatter(x=ec_values, y=normalized_intensity, mode='lines', name=f'{kv} kV'))

    fig.update_layout(
        title='Aligned Peak Data',
        xaxis_title='Ec (V/cm)',
        yaxis_title='Normalized Intensity',
        legend_title='Voltage (kV)',
        font=dict(
            family="Times New Roman",
            size=14
        ),
        template=plotly_template
    )
    st.plotly_chart(fig)
    return fig


def save_processed_data_to_excel(processed_data, df_intersections_filtered, df_fraction_aligned, df_histogram):
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
       
        for idx, (kv, ec_values, normalized_intensity) in enumerate(processed_data):
            df = pd.DataFrame({
                'E_C (V/cm)': ec_values,
                f'Normalized Intensity ({kv} kV)': normalized_intensity
            })
            df.to_excel(writer, sheet_name='Spectra', index=False, startcol=idx * 2)

       
        df_intersections_filtered.to_excel(writer, sheet_name='Linear_Ec_Ed_Plot', index=False)

       
        df_fraction_aligned.to_excel(writer, sheet_name='FA_Plot', index=False)

    
        df_histogram.to_excel(writer, sheet_name='Histogram_Plot', index=False)

    output.seek(0)
    return output


def find_intersections(processed_data, threshold):
    intersections = []
    for kv, ec_values, normalized_intensity in processed_data:
        for i in range(1, len(ec_values)):
            if normalized_intensity[i-1] < threshold and normalized_intensity[i] >= threshold:
                x1, x2 = ec_values[i-1], ec_values[i]
                y1, y2 = normalized_intensity[i-1], normalized_intensity[i]
                x_intersect = x1 + (threshold - y1) * (x2 - x1) / (y2 - y1)
                intersections.append((kv, x_intersect))
                break
    return intersections


def store_intersections_in_dataframe(intersections):
    df = pd.DataFrame(intersections, columns=["num_set", "threshold_Ec"])
    return df


def calculate_ED_values(df):
    df["E_D"] = df["num_set"] / FAIMS_ELECTRODE_GAP
    return df


def calculate_dipole_values(df):
    df["D_moment"] = (K_B * T_ION) / (2 * df["E_D"] * 1e8) / DEBYE_CONVERSION
    return df 


def filter_by_ED(df, start_ED):
    return df[df["E_D"] >= start_ED]


def plot_intersections(df):
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=df["E_D"],
        y=df["threshold_Ec"],
        mode='markers',
        name='Threshold Intersections'
    ))

    fig.update_layout(
        title='Threshold Intersections',
        xaxis_title='Ed (kV/cm)',
        yaxis_title='Threshold Ec (V/cm)',
        font=dict(
            family="Times New Roman",
            size=14
        ),
        template=plotly_template
    )
    st.plotly_chart(fig)
    return fig


def perform_linear_regression(df):
    X = df["E_D"].values.reshape(-1, 1)
    y = df["threshold_Ec"].values
    reg = LinearRegression().fit(X, y)
    y_pred = reg.predict(X)
    slope = reg.coef_[0]
    intercept = reg.intercept_
    r2 = r2_score(y, y_pred)

    return slope, intercept, r2


def calculate_fraction_of_aligned(processed_data, last_spectrum_voltage):
    areas = []
    for kv, ec_values, normalized_intensity in processed_data:
        valid_indices = ~np.isnan(normalized_intensity)
        ec_values = ec_values[valid_indices]
        normalized_intensity = normalized_intensity[valid_indices]

        if len(ec_values) > 1:
            area = trapz(normalized_intensity, ec_values)
        else:
            area = 0

        areas.append((kv, area))

    last_spectrum_area = next((area for kv, area in areas if kv == last_spectrum_voltage), None)
    
    if last_spectrum_area is None or last_spectrum_area == 0:
        st.error("Last spectrum voltage not found in the dataset or area is zero.")
        return pd.DataFrame(columns=["num_set", "fraction_aligned"])

    fractions = [(kv, (area - last_spectrum_area) / area) for kv, area in areas if area != 0]

    df_fractions = pd.DataFrame(fractions, columns=["num_set", "fraction_aligned"])

    return df_fractions


def filter_fa_by_start_ED(df, start_ED):
    df["fraction_aligned"] = df.apply(lambda row: 0 if row["E_D"] < start_ED else row["fraction_aligned"], axis=1)
    return df


def plot_fraction_aligned(df):
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=df["E_D"],
        y=df["fraction_aligned"],
        mode='markers',
        name='Fraction Aligned'
    ))

    fig.update_layout(
        title='Fraction Aligned vs E_D',
        xaxis_title='E_D (kV/cm)',
        yaxis_title='Fraction Aligned',
        font=dict(
            family="Times New Roman",
            size=14
        ),
        template=plotly_template
    )
    st.plotly_chart(fig)
    return fig


def calculate_and_plot_histogram(df_fraction_aligned, exclude_negative_density):
    bin_edges = []
    bin_heights = []

    unaligned_fraction = 1 - df_fraction_aligned['fraction_aligned'].iloc[0]
    first_bin_width = df_fraction_aligned['D_moment'].iloc[0]
    first_bin_height = unaligned_fraction / first_bin_width

    bin_edges.append((0, df_fraction_aligned['D_moment'].iloc[0]))
    bin_heights.append(first_bin_height)

    for i in range(1, len(df_fraction_aligned)):
        increment_FA = df_fraction_aligned['fraction_aligned'].iloc[i-1] - df_fraction_aligned['fraction_aligned'].iloc[i]
        bin_width = df_fraction_aligned['D_moment'].iloc[i] - df_fraction_aligned['D_moment'].iloc[i-1]

        if bin_width != 0:
            bin_height = increment_FA / bin_width
            bin_edges.append((df_fraction_aligned['D_moment'].iloc[i-1], df_fraction_aligned['D_moment'].iloc[i]))
            bin_heights.append(bin_height)

    total_area = sum(bin_heights[i] * (bin_edges[i][1] - bin_edges[i][0]) for i in range(len(bin_heights)))
    bin_heights = [height / total_area for height in bin_heights]

    if exclude_negative_density:
        bin_heights = [max(0, height) for height in bin_heights]

    bin_centers = [(edge[0] + edge[1]) / 2 for edge in bin_edges]
    bin_widths = [edge[1] - edge[0] for edge in bin_edges]

    df_histogram = pd.DataFrame({
        'Bin Center': bin_centers,
        'Bin Width': bin_widths,
        'Bin Height': bin_heights
    })

    st.write("Histogram Raw Data:")
    st.write(df_histogram)

    x_data = []
    y_data = []

    for i in range(len(df_histogram)):
        center = df_histogram['Bin Center'].iloc[i]
        width = df_histogram['Bin Width'].iloc[i]
        height = df_histogram['Bin Height'].iloc[i]
        
        x_data.extend([center - width/2, center + width/2])
        y_data.extend([height, height])

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=x_data,
        y=y_data,
        mode='lines',
        name='Dipole Moment Histogram'
    ))

    fig.update_layout(
        title=dict(
            text='Dipole Moment Histogram (Horizontal Line Plot)',
            font=dict(
                family="Times New Roman",
                size=16
            ),
            x=0.5,
            xanchor='center'
        ),
        xaxis=dict(
            title=dict(
                text='Dipole Moment (kD)',
                font=dict(
                    family="Times New Roman",
                    size=14
                )
            ),
            showgrid=False,
            linewidth=2,
            ticks='outside',
            tickwidth=2,
            tickfont=dict(family='Times New Roman', size=14)
        ),
        yaxis=dict(
            title=dict(
                text='Density of Species',
                font=dict(
                    family="Times New Roman",
                    size=14
                )
            ),
            showgrid=False,
            linewidth=2,
            ticks='outside',
            tickwidth=2,
            tickfont=dict(family='Times New Roman', size=14)
        ),
        margin=dict(t=50, b=40, l=50, r=50)
    )

    st.plotly_chart(fig)

    df_xy = pd.DataFrame({
        'x': x_data,
        'y': y_data
    })

    st.write("X/Y Data for Horizontal Line Plot:")
    st.write(df_xy)

    return fig, df_histogram


def app():
    st.title("Data Processing and Plotting")
    
   
    show_calculation_info()
    
    uploaded_file = st.file_uploader("Choose an Excel file", type="xlsx")
    manual_shift = st.number_input("Manual Shift", value=2.2)
    num_sets = st.text_input("Number of Sets (comma separated)", "1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1")
    num_sets = [float(n) for n in num_sets.split(",")]
    threshold = st.number_input("Threshold for Alignment (0 to 1)", value=0.05, min_value=0.0, max_value=1.0)
    start_ED = st.number_input("Start Alignment Point (ED)", value=0.5, step=0.01, format="%.2f")
    last_spectrum_voltage = st.number_input("Last Spectrum Voltage (ED)", value=0.1)
    exclude_negative_density = st.checkbox("Exclude Negative Density", value=False)
    
   
    background_files = st.file_uploader("Upload Background Spectra (optional)", type="xlsx", accept_multiple_files=True)

    if uploaded_file is not None:
        if st.button("Run All Calculations and Generate Plots"):
            with tempfile.NamedTemporaryFile(delete=False, suffix=".xlsx") as tmp_file:
                file_path = tmp_file.name
                tmp_file.write(uploaded_file.getbuffer())

            try:
                background_data = None
                if background_files:
                    background_data = load_background_data(background_files, num_sets)

                processed_data = process_data(file_path, num_sets, manual_shift, background_data)
                fig1 = plot_data(processed_data)

                intersections = find_intersections(processed_data, threshold)
                df_intersections = store_intersections_in_dataframe(intersections)
                df_intersections = calculate_ED_values(df_intersections)
                df_intersections_filtered = filter_by_ED(df_intersections, start_ED)
                st.write(df_intersections_filtered)
                fig2 = plot_intersections(df_intersections_filtered)
                
                if not df_intersections_filtered.empty:
                    slope, intercept, r2 = perform_linear_regression(df_intersections_filtered)
                    st.markdown(f"### Linear Regression Results")
                    st.markdown(f"- **Slope**: {slope:.4f}")
                    st.markdown(f"- **Intercept**: {intercept:.4f}")
                    st.markdown(f"- **R²**: {r2:.4f}")
                else:
                    st.write("No data points above the specified start alignment point.")

                df_fraction_aligned = calculate_fraction_of_aligned(processed_data, last_spectrum_voltage)
                df_fraction_aligned = calculate_ED_values(df_fraction_aligned)
                df_fraction_aligned = calculate_dipole_values(df_fraction_aligned)
                df_fraction_aligned = filter_fa_by_start_ED(df_fraction_aligned, start_ED)
                st.write("Fraction Aligned DataFrame:")
                st.write(df_fraction_aligned)
                fig3 = plot_fraction_aligned(df_fraction_aligned)
                
                fig4, df_histogram = calculate_and_plot_histogram(df_fraction_aligned, exclude_negative_density)

                excel_data = save_processed_data_to_excel(processed_data, df_intersections_filtered, df_fraction_aligned, df_histogram)
                st.download_button(
                    label="Download Processed Data",
                    data=excel_data,
                    file_name=f"{uploaded_file.name.split('.')[0]} Adjusted.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )

            except Exception as e:
                st.error(f"An error occurred: {e}")


if __name__ == "__main__":
    app()
