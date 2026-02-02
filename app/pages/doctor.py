"""
Plant disease diagnosis page for Dr. Plant.

Handles image upload, CNN-based disease prediction, and AI-powered
diagnosis and treatment recommendations using Gemini.
"""

from typing import Dict, Any
import streamlit as st
from PIL import Image
import torch
import torchvision.transforms as transforms
import google.generativeai as genai
import json
import pymongo
from pymongo.collection import Collection
import sys
import os
# Add the project root to sys.path to allow absolute imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from app.config import (
    CLASS_LABELS,
    MONGO_URI,
    DB_NAME,
    DISEASES_COLLECTION,
    NUM_CLASSES,
    IMAGE_SIZE,
    MODEL_PATH,
    GEMINI_MODEL,
    GEMINI_TEMPERATURE,
    GEMINI_MAX_TOKENS
)

from models.plant_disease_cnn import PlantDiseaseCNN

# Check authentication
if not st.session_state.get("authenticated", False):
    st.warning("Please log in to access this page.")
    st.switch_page("login.py")
    st.stop()

# Get user's Gemini API key from session (stored during login)
user_gemini_key = st.session_state.get("gemini_key")
if not user_gemini_key:
    st.error("Gemini API key not found. Please log in again.")
    st.switch_page("login.py")
    st.stop()

# Configure Gemini API with user's key
genai.configure(api_key=user_gemini_key)

# Connect to MongoDB
client: pymongo.MongoClient = pymongo.MongoClient(MONGO_URI)
db = client[DB_NAME]
disease_collection: Collection = db[DISEASES_COLLECTION]

# Load trained model
device: torch.device = torch.device("cpu")
model: PlantDiseaseCNN = PlantDiseaseCNN(NUM_CLASSES)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()
model.to(device)


def preprocess_image(image: Image.Image) -> torch.Tensor:
    """
    Preprocess an image for model inference.

    Resizes the image to 128x128, converts to tensor, and normalizes.

    Args:
        image: PIL Image object

    Returns:
        Preprocessed image tensor ready for model input
    """
    transform = transforms.Compose([
        transforms.Resize(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    return transform(image).unsqueeze(0).to(device)

# Streamlit UI
st.title("🌱 Dr. Plant: Plant Disease Diagnosis AI")

uploaded_file = st.file_uploader("Upload an image of the plant", type=["jpg", "png", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", width=600)

    # Preprocess image and make prediction using your model
    img_tensor = preprocess_image(image)
    with torch.no_grad():
        output = model(img_tensor)
        prediction = torch.argmax(output, dim=1).item()
        disease_label = CLASS_LABELS[prediction]

    # Check if disease is already in MongoDB
    existing_disease = disease_collection.find_one({"disease_label": disease_label})

    if existing_disease:
        diagnosis = existing_disease.get("disease_detail", "No diagnosis found.")
        treatment = existing_disease.get("treatment_detail", "No treatment found.")

        st.subheader("🩺 Diagnosis")
        st.write(diagnosis.strip())

        st.subheader("💊 Recommended Treatment")
        st.write(treatment.strip())

    else:
        # Query Gemini if disease is not found in cache
        model_gemini = genai.GenerativeModel(
            GEMINI_MODEL,
            system_instruction="""
            You are a plant disease expert.
            Return ONLY valid JSON with exactly two fields:
            - diagnosis
            - treatment

            Each field should be a detailed paragraph.
            Explain why the disease is harmful in the diagnosis.
            Treatment should include actionable steps and resources.
            Do not include markdown or extra text.
            """
        )

        try:
            response = model_gemini.generate_content(
                disease_label,
                generation_config=genai.types.GenerationConfig(
                    temperature=GEMINI_TEMPERATURE,
                    response_mime_type="application/json",
                    max_output_tokens=GEMINI_MAX_TOKENS
                )
            )

            if response and response.candidates:
                candidate = response.candidates[0]

                # Check if response was truncated
                if candidate.finish_reason.name == "MAX_TOKENS":
                    st.warning("⚠️ Response was truncated due to length. Trying again with adjusted settings...")
                    # Retry with higher token limit
                    response = model_gemini.generate_content(
                        disease_label,
                        generation_config=genai.types.GenerationConfig(
                            temperature=GEMINI_TEMPERATURE,
                            response_mime_type="application/json",
                            max_output_tokens=GEMINI_MAX_TOKENS * 2
                        )
                    )
                    candidate = response.candidates[0]

                raw_text = candidate.content.parts[0].text

                try:
                    parsed_response = json.loads(raw_text)
                    diagnosis = parsed_response.get("diagnosis", "No diagnosis found.")
                    treatment = parsed_response.get("treatment", "No treatment found.")

                    # Store result in MongoDB
                    disease_collection.insert_one({
                        "disease_label": disease_label,
                        "disease_detail": diagnosis,
                        "treatment_detail": treatment
                    })

                    st.subheader("🩺 Diagnosis")
                    st.write(diagnosis.strip())

                    st.subheader("💊 Recommended Treatment")
                    st.write(treatment.strip())

                except json.JSONDecodeError as e:
                    st.error(f"Failed to parse AI response: {str(e)}")
                    st.subheader("Raw Response from AI:")
                    st.code(raw_text)
            else:
                st.error("No valid response from Gemini.")

        except Exception as e:
            st.error(f"Error generating diagnosis: {str(e)}")
