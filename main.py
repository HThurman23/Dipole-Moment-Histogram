import streamlit as st

st.set_page_config(page_title="Modular Dashboard", layout="wide")

# Sidebar navigation
st.sidebar.title("Navigation")
page = st.sidebar.selectbox("Go to", ["LODIMS Dipole Moment Calculator", "Page 2", "Page 3"])

# Load the selected page
if page == "LODIMS Dipole Moment Calculator":
    import pages.page1 as page1
    page1.app()
elif page == "Page 2":
    st.write("Page 2 content here")
elif page == "Page 3":
    st.write("Page 3 content here")