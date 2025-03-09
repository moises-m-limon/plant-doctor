import streamlit as st

st.title("Logout")

if "user" in st.session_state:
    del st.session_state["user"]
    st.success("You have logged out.")

st.page_link("../ui/login.py", label="Return to Home 🏠")
