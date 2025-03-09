import streamlit as st
from PIL import Image
import torch
import torchvision.transforms as transforms
import google.generativeai as genai
import json
import pymongo
import sys
import os
from config import GEMINI_KEY, CLASS_LABELS, MONGO_URI

# Add the project root to sys.path to allow absolute imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from modules.plant_disease_cnn import PlantDiseaseCNN

# Configure Gemini API key
genai.configure(api_key=GEMINI_KEY)

# Connect to MongoDB
client = pymongo.MongoClient(MONGO_URI)
db = client["plant_diagnosis"]
disease_collection = db["plant_diseases"]

# Load trained model
device = torch.device("cpu")
num_classes = 38
model = PlantDiseaseCNN(num_classes)
model.load_state_dict(torch.load("modules/plant_disease_cnn.pth", map_location=device))
model.eval()
model.to(device)

# Define image preprocessing
def preprocess_image(image):
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
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
        # Query Gemini if disease is not found
        model_gemini = genai.GenerativeModel(
            "gemini-1.5-flash",
            system_instruction=f"""
            You are a plant disease expert providing AI-powered diagnoses. 
            The identified disease is **{disease_label}**. 
            Provide a structured JSON response with two fields:`diagnosis` and `treatment`.
            Ensure responses are verbose and add any helpful resources (max 1000 characters).
            With diagnosis, make sure to give context to why it is harmful.
            """
        )

        try:
            response = model_gemini.generate_content(
                disease_label,
                generation_config=genai.types.GenerationConfig(
                    temperature=0.7,
                    response_mime_type="application/json",
                    max_output_tokens=200
                )
            )

            if response and response.candidates:
                raw_text = response.candidates[0].content.parts[0].text
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

                except json.JSONDecodeError:
                    st.subheader("Response from AI:")
                    st.write(raw_text)
            else:
                st.error("No valid response from Gemini.")

        except Exception as e:
            st.error(f"Error: {str(e)}")
