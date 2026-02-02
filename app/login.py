"""
User authentication module for Dr. Plant.

Handles user login, signup, password hashing, and API key encryption.
"""

import sys
import os

# Add app directory to path for imports
sys.path.insert(0, os.path.dirname(__file__))

from typing import Optional, Tuple, Dict, Any
import streamlit as st
from pymongo import MongoClient
from pymongo.collection import Collection
import bcrypt
from cryptography.fernet import Fernet
import time
from config import ENCRYPTION_KEY, MONGO_URI, DB_NAME, USERS_COLLECTION

# Initialize encryption cipher
cipher: Fernet = Fernet(ENCRYPTION_KEY.encode() if isinstance(ENCRYPTION_KEY, str) else ENCRYPTION_KEY)

# MongoDB Setup
client: MongoClient = MongoClient(MONGO_URI)
db = client[DB_NAME]
users_collection: Collection = db[USERS_COLLECTION]


def hash_password(password: str) -> bytes:
    """
    Hash a password using bcrypt.

    Args:
        password: Plain text password to hash

    Returns:
        Hashed password as bytes
    """
    return bcrypt.hashpw(password.encode(), bcrypt.gensalt())


def verify_password(password: str, hashed_password: bytes) -> bool:
    """
    Verify a password against its hash.

    Args:
        password: Plain text password to verify
        hashed_password: Previously hashed password

    Returns:
        True if password matches, False otherwise
    """
    return bcrypt.checkpw(password.encode(), hashed_password)


def encrypt_gemini_key(api_key: str) -> str:
    """
    Encrypt a Gemini API key for secure storage.

    Args:
        api_key: Plain text Gemini API key

    Returns:
        Encrypted API key as string
    """
    return cipher.encrypt(api_key.encode()).decode()


def decrypt_gemini_key(encrypted_key: str) -> str:
    """
    Decrypt a Gemini API key from storage.

    Args:
        encrypted_key: Encrypted API key string

    Returns:
        Decrypted API key as plain text
    """
    return cipher.decrypt(encrypted_key.encode()).decode()


def sign_up(
    username: str,
    email: str,
    password: str,
    company: str,
    gemini_key: str
) -> Tuple[str, Optional[str]]:
    """
    Register a new user account.

    Args:
        username: Desired username
        email: User's email address
        password: Plain text password
        company: Company or farm name
        gemini_key: User's Gemini API key

    Returns:
        Tuple of (message, error_field) where error_field is None on success
        or the field name ("username" or "email") that caused an error
    """
    # Check for duplicate username
    if users_collection.find_one({"username": username}):
        return "Username is already taken!", "username"

    # Check for duplicate email
    if users_collection.find_one({"email": email}):
        return "Email is already registered!", "email"

    # Hash password and encrypt API key
    hashed_pw = hash_password(password)
    encrypted_gemini_key = encrypt_gemini_key(gemini_key)
    st.session_state["gemini_key"] = encrypted_gemini_key

    # Create user document
    user_data = {
        "username": username,
        "email": email,
        "password": hashed_pw,
        "company": company,
        "gemini_key": encrypted_gemini_key
    }

    users_collection.insert_one(user_data)
    return "Account created successfully!", None


def login(email: str, password: str) -> Optional[Dict[str, Any]]:
    """
    Authenticate a user with email and password.

    Args:
        email: User's email address
        password: Plain text password

    Returns:
        User document dict if authentication succeeds, None otherwise
    """
    user = users_collection.find_one({"email": email})
    if user:
        stored_password = user["password"]

        # Handle password stored as string (legacy compatibility)
        if isinstance(stored_password, str):
            stored_password = stored_password.encode()

        if verify_password(password, stored_password):
            return user
    return None

# Streamlit UI
st.set_page_config(page_title="Login / Sign-Up", layout="centered")

st.markdown(
    """
    <style>
    .stApp {
        background-color: #E8F5E9;
    }
    .stButton>button {
        background-color: #2E7D32 !important;
        color: white !important;
        border-radius: 10px;
        font-weight: bold;
    }
    .stTextInput>div>div>input {
        background-color: #A5D6A7;
        color: #1B5E20;
        border-radius: 5px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

menu = st.pills("Choose how to access:", ["Login", "Sign Up"])

if menu == "Sign Up":
    st.subheader("Create a New Account")

    username = st.text_input("Username", key="signup_username")
    email = st.text_input("Email", key="signup_email")
    password = st.text_input("Password", type="password", key="signup_password")
    company = st.text_input("Company / Farm", key="signup_company")
    gemini_key = st.text_input("Google Gemini API Key", type="password", key="signup_gemini_key")

    username_error, email_error = "", ""

    if username and users_collection.find_one({"username": username}):
        username_error = "❌ Username is already taken!"

    if email and users_collection.find_one({"email": email}):
        email_error = "❌ Email is already registered!"

    if username_error:
        st.error(username_error)
    if email_error:
        st.error(email_error)

    if st.button("Sign Up", disabled=bool(username_error) or bool(email_error)):
        if username and email and password and company and gemini_key:
            message, error_field = sign_up(username, email, password, company, gemini_key)
            if not error_field:
                st.success(message)
                # Set up session for authenticated user
                st.session_state["user"] = {"username": username, "email": email}
                st.session_state["authenticated"] = True
                st.session_state["gemini_key"] = gemini_key
                time.sleep(1)
                # Redirect to doctor page
                st.switch_page("pages/doctor.py")
        else:
            st.error("All fields are required!")

elif menu == "Login":
    st.subheader("Log In to Your Account")

    email = st.text_input("Email", key="login_email")
    password = st.text_input("Password", type="password", key="login_password")

    if st.button("Login"):
        user = login(email, password)
        if user:
            st.session_state["user"] = user
            decrypted_gemini_key = decrypt_gemini_key(user["gemini_key"])

            # Store decrypted key and set authenticated
            st.session_state["authenticated"] = True
            st.session_state["gemini_key"] = decrypted_gemini_key
            st.success(f"Welcome, {user['username']}!")
            time.sleep(1)
            # Redirect to doctor page
            st.switch_page("pages/doctor.py")

        else:
            st.error("❌ Invalid email or password.")
