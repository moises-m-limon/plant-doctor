"""
User settings page for Dr. Plant.

Displays user information and API key settings.
"""

import streamlit as st

st.set_page_config(page_title="Settings", layout="centered")
st.title("⚙️ Settings")

# Authentication guard
if not st.session_state.get("authenticated", False):
    st.warning("Please log in to access this page.")
    st.switch_page("login.py")
    st.stop()

# Display authenticated user information
user = st.session_state.get("user")

st.success(f"Logged in as {user['username']}")

# Logout link
st.page_link("pages/logout.py", label="Logout", icon="🚪")

st.divider()

# API key display
st.subheader("🔐 API Settings")

if "gemini_key" in st.session_state:
    # Mask API key for security (show only first 6 characters)
    masked: str = st.session_state["gemini_key"][:6] + "••••••"
    st.code(masked)
else:
    st.info("No Gemini API key found in session.")
