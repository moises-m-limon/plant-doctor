"""
User logout page for Dr. Plant.

Handles user session termination and cleanup.
"""

import streamlit as st

st.title("Logout")

# Authentication guard
if not st.session_state.get("authenticated", False):
    st.warning("Please log in to access this page.")
    st.switch_page("login.py")
    st.stop()

# Clear user session
if "user" in st.session_state:
    del st.session_state["user"]
    st.success("You have logged out.")

# Return to home link
st.page_link("../app/login.py", label="Return to Home 🏠")
