import streamlit as st
import pages.page1 as page1
import pages.page2 as page2
import pages.page3 as page3

st.set_page_config(page_title="Modular Dashboard", layout="wide")

# Sidebar navigation
st.sidebar.title("Navigation")
page = st.sidebar.selectbox("Go to", ["LODIMS Dipole Moment Calculator", "Page 2", "Page 3"])

# Load the selected page
if page == "LODIMS Dipole Moment Calculator":
    page1.app()
elif page == "Page 2":
    page2.app()
elif page == "Page 3":
    page3.app()
