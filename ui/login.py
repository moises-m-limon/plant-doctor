import streamlit as st
from pymongo import MongoClient
import bcrypt
from cryptography.fernet import Fernet
import time
from config import ENCRYPTION_KEY, MONGO_URI

cipher = Fernet(ENCRYPTION_KEY)

# MongoDB Setup
DB_NAME = "plant_doctor"
COLLECTION_NAME = "users"

client = MongoClient(MONGO_URI)
db = client[DB_NAME]
users_collection = db[COLLECTION_NAME]

def hash_password(password):
    return bcrypt.hashpw(password.encode(), bcrypt.gensalt())

def verify_password(password, hashed_password):
    return bcrypt.checkpw(password.encode(), hashed_password)

def encrypt_gemini_key(api_key):
    return cipher.encrypt(api_key.encode()).decode()

def decrypt_gemini_key(encrypted_key):
    return cipher.decrypt(encrypted_key.encode()).decode()

def sign_up(username, email, password, company, gemini_key):
    if users_collection.find_one({"username": username}):
        return "Username is already taken!", "username"
    if users_collection.find_one({"email": email}):
        return "Email is already registered!", "email"

    hashed_pw = hash_password(password)

    encrypted_gemini_key = encrypt_gemini_key(gemini_key)
    st.session_state["gemini_key"] = encrypted_gemini_key 

    user_data = {
        "username": username,
        "email": email,
        "password": hashed_pw, 
        "company": company,
        "gemini_key": encrypted_gemini_key
    }

    users_collection.insert_one(user_data)
    return "Account created successfully!", None

def login(email, password):
    user = users_collection.find_one({"email": email})
    if user:
        stored_password = user["password"]
        
        if isinstance(stored_password, str):
            stored_password = stored_password.encode()
        
        if verify_password(password, stored_password):
            return user
    return None

# Streamlit UI
st.set_page_config(page_title="Login / Sign-Up", layout="centered")

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
            st.session_state["gemini_key"] = gemini_key 
            message, error_field = sign_up(username, email, password, company, gemini_key)
            if not error_field:
                st.success(message)
                time.sleep(1)  # Small delay before redirecting
                st.session_state["user"] = {"username": username, "email": email}
                st.rerun()  # Refresh to trigger login session
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
            
            # Store decrypted key in session state for use in settings
            st.success(f"Welcome, {user['username']}!")
            time.sleep(1)
            st.session_state["authenticated"] = True
            st.session_state["gemini_key"] = decrypted_gemini_key  # Ensure Gemini key is stored
            st.rerun()  # Ensures session state is set before navigation

        else:
            st.error("❌ Invalid email or password.")

# If user is logged in, redirect to main page immediately
if "user" in st.session_state:
    st.session_state["authenticated"] = True
    st.switch_page("pages/doctor.py")

# If user is logged in, show API Key settings
if "user" in st.session_state:
    st.sidebar.subheader("Settings")
    st.sidebar.text("Google Gemini API Key:")
    st.sidebar.code(st.session_state["gemini_key"], language="plaintext")
