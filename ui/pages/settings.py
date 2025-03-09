import streamlit as st

st.set_page_config(page_title="Settings", layout="centered")
st.title("Settings")

if "user" in st.session_state:
    st.success(f"Logged in as {st.session_state.user['username']}")
    st.page_link("pages/logout.py", label="Logout", icon="🚪")
else:
    st.page_link("login.py", label="Login", icon="🔑")
