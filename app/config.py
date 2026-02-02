"""
Configuration module for Dr. Plant application.

Loads environment variables and imports application constants.
"""

import os
import sys
from dotenv import load_dotenv

# Add project root to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Load environment variables from .env file
load_dotenv()

# Environment variables
ENCRYPTION_KEY: str = os.getenv('ENCRYPTION_KEY', '')
GEMINI_KEY: str = os.getenv("GEMINI_KEY", '')
MONGO_URI: str = os.getenv('MONGO_URI', 'mongodb://localhost:27017/')

# Import constants from centralized location
from core.constants import (
    CLASS_LABELS,
    DB_NAME,
    USERS_COLLECTION,
    DISEASES_COLLECTION,
    NUM_CLASSES,
    IMAGE_SIZE,
    MODEL_PATH,
    GEMINI_MODEL,
    GEMINI_TEMPERATURE,
    GEMINI_MAX_TOKENS
)