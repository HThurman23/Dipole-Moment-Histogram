import streamlit as st
import dipole_moment_calculator  # Adjust to your actual file names
import data_analysis
import settings

st.set_page_config(page_title="Modular Dashboard", layout="wide")

# Sidebar navigation
st.sidebar.title("Navigation")
page = st.sidebar.selectbox("Go to", [
    "Dipole Moment Calculator",
    "Data Analysis",
    "Settings"
])

# Load the selected page
if page == "Dipole Moment Calculator":
    dipole_moment_calculator.app()
elif page == "Data Analysis":
    data_analysis.app()
elif page == "Settings":
    settings.app()
