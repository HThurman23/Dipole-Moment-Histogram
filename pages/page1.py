import streamlit as st
import pandas as pd
import numpy as np
import statsmodels.api as sm
from scipy.integrate import simps
import tempfile
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# Constants
FAIMS_ELECTRODE_GAP = 0.188  # cm
K_B = 1.380649e-23  # Boltzmann constant in J/K
T_ION = 298.5  # Ion temperature in Kelvin
DEBYE_CONVERSION = 3.33564e-30  # C·m in one Debye

# Set Plotly theme
plotly_template = "plotly_dark"

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

def process_data(file_path, num_sets, manual_shift):
    max_right_index = -np.inf
    processed_data = []

    for i, kv in enumerate(num_sets):
        metadata = pd.read_excel(file_path, nrows=3, header=None, usecols=[i*2, i*2+1])
        bias, cv_start, scan_rate = metadata.iloc[0, 1], metadata.iloc[1, 1], metadata.iloc[2, 1]

        data = pd.read_excel(file_path, skiprows=3, usecols=[i*2, i*2+1])
        ec_values = calculate_ec(data.iloc[:, 0], bias, cv_start, scan_rate)
        smoothed_intensity = smooth_data(data.iloc[:, 1])
        normalized_intensity = normalize_intensity(smoothed_intensity)

        current_right_index = find_rightmost_index(ec_values, normalized_intensity)
        if current_right_index is not None and current_right_index > max_right_index:
            max_right_index = current_right_index

        processed_data.append((kv, ec_values, normalized_intensity, current_right_index))

    # Adjust Ec values to align to the maximum rightmost index
    for i, (kv, ec_values, normalized_intensity, current_right_index) in enumerate(processed_data):
        adjusted_ec_values = adjust_ec_values(ec_values, current_right_index, max_right_index, manual_shift)
        processed_data[i] = (kv, adjusted_ec_values, normalized_intensity)

    return processed_data

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

def save_processed_data_to_excel(processed_data, filename):
    with pd.ExcelWriter(filename, engine='openpyxl') as writer:
        for kv, ec_values, normalized_intensity in processed_data:
            df = pd.DataFrame({
                'E_C (V/cm)': ec_values,
                'Normalized Intensity': normalized_intensity
            })
            df.to_excel(writer, sheet_name=f'{kv} kV', index=False)

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
        # Remove NaN values from intensity and corresponding ec_values
        valid_indices = ~np.isnan(normalized_intensity)
        ec_values = ec_values[valid_indices]
        normalized_intensity = normalized_intensity[valid_indices]

        if len(ec_values) > 1:  # Ensure there's enough data to integrate
            area = simps(normalized_intensity, ec_values)
        else:
            area = 0

        areas.append((kv, area))

    # Find the area under the last spectrum
    last_spectrum_area = next((area for kv, area in areas if kv == last_spectrum_voltage), None)
    
    if last_spectrum_area is None or last_spectrum_area == 0:
        st.error("Last spectrum voltage not found in the dataset or area is zero.")
        return pd.DataFrame(columns=["num_set", "fraction_aligned"])

    fractions = [(kv, (area - last_spectrum_area) / area) for kv, area in areas if area != 0]

    return pd.DataFrame(fractions, columns=["num_set", "fraction_aligned"])

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
    # Initialize lists to store bin edges and heights
    bin_edges = []
    bin_heights = []

    # Calculate the first bin for the unaligned fraction
    unaligned_fraction = 1 - df_fraction_aligned['fraction_aligned'].iloc[0]
    first_bin_width = df_fraction_aligned['D_moment'].iloc[0]
    first_bin_height = unaligned_fraction / first_bin_width

    # Add the first bin
    bin_edges.append((0, df_fraction_aligned['D_moment'].iloc[0]))
    bin_heights.append(first_bin_height)

    # Loop through the rest of the DataFrame to calculate increments and bin heights
    for i in range(1, len(df_fraction_aligned)):
        increment_FA = df_fraction_aligned['fraction_aligned'].iloc[i-1] - df_fraction_aligned['fraction_aligned'].iloc[i]
        bin_width = df_fraction_aligned['D_moment'].iloc[i] - df_fraction_aligned['D_moment'].iloc[i-1]

        if bin_width != 0:
            bin_height = increment_FA / bin_width
            bin_edges.append((df_fraction_aligned['D_moment'].iloc[i-1], df_fraction_aligned['D_moment'].iloc[i]))
            bin_heights.append(bin_height)

    # Normalize the bin heights so that their total area sums to 1
    total_area = sum(bin_heights[i] * (bin_edges[i][1] - bin_edges[i][0]) for i in range(len(bin_heights)))
    bin_heights = [height / total_area for height in bin_heights]

    normalized_total_area = sum(bin_heights[i] * (bin_edges[i][1] - bin_edges[i][0]) for i in range(len(bin_heights)))
    st.write(f"Normalized Total Area: {normalized_total_area}")

    # Exclude negative density if the checkbox is checked
    if exclude_negative_density:
        bin_heights = [max(0, height) for height in bin_heights]

    # Prepare data for plotting
    bin_centers = [(edge[0] + edge[1]) / 2 for edge in bin_edges]
    bin_widths = [edge[1] - edge[0] for edge in bin_edges]

    # Create a DataFrame for the histogram raw data
    df_histogram = pd.DataFrame({
        'Bin Center': bin_centers,
        'Bin Width': bin_widths,
        'Bin Height': bin_heights
    })

    # Display the histogram raw data
    st.write("Histogram Raw Data:")
    st.write(df_histogram)

    # Generate x/y data for the horizontal line plot
    x_data = []
    y_data = []

    for i in range(len(df_histogram)):
        center = df_histogram['Bin Center'].iloc[i]
        width = df_histogram['Bin Width'].iloc[i]
        height = df_histogram['Bin Height'].iloc[i]
        
        x_data.extend([center - width/2, center + width/2])
        y_data.extend([height, height])

    # Create the horizontal line plot
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=x_data,
        y=y_data,
        mode='lines',
        name='Dipole Moment Histogram',
        #line=dict(color='blue')
    ))

    fig.update_layout(
        title=dict(
            text='Dipole Moment Histogram (Horizontal Line Plot)',
            font=dict(
                family="Times New Roman",
                size=16,
                #color='black'
            ),
            x=0.5,
            xanchor='center'
        ),
        xaxis=dict(
            title=dict(
                text='Dipole Moment (kD)',
                font=dict(
                    family="Times New Roman",
                    size=14,
                    #color='black'
                )
            ),
            showgrid=False,
            #linecolor='black',  # Bolden axis lines
            linewidth=2,
            ticks='outside',
            tickwidth=2,
            tickfont=dict(family='Times New Roman', 
                          #color='black', 
                          size=14)
        ),
        yaxis=dict(
            title=dict(
                text='Density of Species',
                font=dict(
                    family="Times New Roman",
                    size=14,
                    #color='White'
                )
            ),
            showgrid=False,
            #linecolor='black',  # Bolden axis lines
            linewidth=2,
            ticks='outside',
            tickwidth=2,
            tickfont=dict(family='Times New Roman', 
                          #color='black', 
                          size=14)
        ),
        # plot_bgcolor='white',
        # paper_bgcolor='white',
        margin=dict(t=50, b=40, l=50, r=50)  # Adjust margins to reduce white space
    )

    # Display the plot using Streamlit
    st.plotly_chart(fig)

    # Create DataFrame for x/y data
    df_xy = pd.DataFrame({
        'x': x_data,
        'y': y_data
    })

    # Display x/y data
    st.write("X/Y Data for Horizontal Line Plot:")
    st.write(df_xy)

    return fig

def app():
    st.title("Data Processing and Plotting")
    
    uploaded_file = st.file_uploader("Choose an Excel file", type="xlsx")
    manual_shift = st.number_input("Manual Shift", value=2.2)
    num_sets = st.text_input("Number of Sets (comma separated)", "1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1")
    num_sets = [float(n) for n in num_sets.split(",")]
    threshold = st.number_input("Threshold for Alignment (0 to 1)", value=0.05, min_value=0.0, max_value=1.0)
    start_ED = st.number_input("Start Alignment Point (ED)", value=1)
    last_spectrum_voltage = st.number_input("Last Spectrum Voltage (ED)", value=0.1)
    exclude_negative_density = st.checkbox("Exclude Negative Density", value=False)

    if uploaded_file is not None:
        if st.button("Run All Calculations and Generate Plots"):
            with tempfile.NamedTemporaryFile(delete=False, suffix=".xlsx") as tmp_file:
                file_path = tmp_file.name
                tmp_file.write(uploaded_file.getbuffer())

            file_name = f"{uploaded_file.name.split('.')[0]} Adjusted.xlsx"

            try:
                processed_data = process_data(file_path, num_sets, manual_shift)
                fig1 = plot_data(processed_data)
                save_processed_data_to_excel(processed_data, file_name)

                st.success(f"Processed data saved to {file_name}")
                st.download_button(
                    label="Download Processed Data",
                    data=open(file_name, "rb").read(),
                    file_name=file_name,
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )

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
                st.write("Fraction Aligned DataFrame:")
                st.write(df_fraction_aligned)
                fig3 = plot_fraction_aligned(df_fraction_aligned)
                
                fig4 = calculate_and_plot_histogram(df_fraction_aligned, exclude_negative_density)
                
                # Add download buttons for each plot
                st.download_button(
                    label="Download Aligned Peak Data Plot",
                    data=fig1.to_image(format="png"),
                    file_name="aligned_peak_data_plot.png",
                    mime="image/png"
                )
                st.download_button(
                    label="Download Threshold Intersections Plot",
                    data=fig2.to_image(format="png"),
                    file_name="threshold_intersections_plot.png",
                    mime="image/png"
                )
                st.download_button(
                    label="Download Fraction Aligned Plot",
                    data=fig3.to_image(format="png"),
                    file_name="fraction_aligned_plot.png",
                    mime="image/png"
                )


            except Exception as e:
                st.error(f"An error occurred: {e}")

# Ensure this is run when the script is called
if __name__ == "__main__":
    app()