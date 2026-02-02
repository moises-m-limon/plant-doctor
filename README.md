# 🌱 Dr. Plant: AI-Powered Plant Disease Diagnosis

## Overview

Dr. Plant is an AI-driven platform designed to empower farmers and plant enthusiasts by providing real-time plant disease diagnosis and treatment recommendations. By leveraging deep learning and generative AI, it offers instant insights to help users take proactive measures for plant health. The service is built to be accessible, user-friendly, and impactful, ensuring that anyone, from small-scale gardeners to large agricultural producers, can make informed decisions to protect their crops. With Dr. Plant, we bring cutting-edge technology to agriculture, making expert knowledge available at the fingertips of those who need it most.

## Table of Contents

- [Dataset](#dataset)
- [Features](#features)
- [Model Training](#model-training)
- [Development Setup](#development-setup)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
  - [Running the Application](#running-the-application)
  - [Project Structure](#project-structure)
  - [Development Workflow](#development-workflow)
  - [Key Technologies](#key-technologies)
  - [Common Development Tasks](#common-development-tasks)
  - [Troubleshooting](#troubleshooting)

## Dataset

https://www.kaggle.com/datasets/vipoooool/new-plant-diseases-dataset/data

## Features

Plant Disease Classification: Uses a CNN model to classify plant diseases from uploaded images.

AI-Powered Diagnosis & Treatment: Integrates Gemini AI to provide structured insights on disease management.

Streamlit-Based UI: Provides an intuitive web interface for easy interaction.

## Model Training

The CNN model can be trained using the Jupyter notebook in the `notebooks/` directory.

1. Ensure the dataset is downloaded and extracted to the `data/` directory:
```
data/
└── New Plant Diseases Dataset(Augmented)/
    └── New Plant Diseases Dataset(Augmented)/
        ├── train/
        └── valid/
```

2. Open and run the training notebook:
```bash
jupyter notebook notebooks/dev.ipynb
```

3. The notebook will:
   - Load and preprocess the image dataset
   - Initialize the PlantDiseaseCNN model
   - Train for 10 epochs with progress output
   - Evaluate on validation set
   - Save the trained model as `plant_disease_cnn.pth`

4. The trained model can then be loaded in the application for inference.

The training script automatically detects and uses available hardware acceleration (CUDA, MPS, or CPU).

## Demo

![image](https://github.com/user-attachments/assets/dd013b8e-bb93-46c4-be0c-f84a46b9e777)

## Development Setup

### Prerequisites

- [Anaconda](https://www.anaconda.com/download) or [Miniconda](https://docs.conda.io/en/latest/miniconda.html)
- [Python 3.11](https://www.python.org/downloads/)
- [MongoDB](https://www.mongodb.com/try/download/community) instance (local) or [MongoDB Atlas](https://www.mongodb.com/cloud/atlas/register) (cloud)
- [Google Gemini API key](https://aistudio.google.com/apikey)

### Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/plant-doctor.git
cd plant-doctor
```

2. Create and activate the conda environment:
```bash
conda env create -f environment.yml
conda activate plantdoctor
```

3. Set up environment variables:

Copy the example environment file and fill in your values:
```bash
cp .env.example .env
```

Then edit `.env` with your actual credentials:
```bash
GEMINI_KEY=your_gemini_api_key_here
ENCRYPTION_KEY=your_fernet_encryption_key_here
MONGO_URI=your_mongodb_connection_string_here
```

To generate an encryption key, run:
```python
python -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())"
```

4. Download the dataset (optional, for model training):
- Download from [Kaggle](https://www.kaggle.com/datasets/vipoooool/new-plant-diseases-dataset/data)
- Extract to `data/` directory

### Running the Application

Start the Streamlit application:
```bash
streamlit run app/login.py
```

The application will open in your browser at `http://localhost:8501`.

### Project Structure

```
plant-doctor/
├── app/                       # Streamlit application
│   ├── login.py              # Main entry point & authentication
│   ├── config.py             # Configuration & environment setup
│   └── pages/                # Application pages
│       ├── doctor.py         # Disease diagnosis interface
│       ├── settings.py       # User settings
│       └── logout.py         # Logout functionality
├── models/                    # ML models
│   ├── plant_disease_cnn.py  # CNN model architecture
│   └── plant_disease_cnn.pth # Trained model weights
├── core/                      # Core utilities
│   └── constants.py          # Application constants
├── notebooks/                 # Jupyter notebooks
│   └── dev.ipynb             # Model training notebook
├── data/                      # Dataset directory (gitignored)
├── .streamlit/                # Streamlit configuration
│   └── config.toml           # UI customization
├── environment.yml            # Conda dependencies
├── .env                      # Environment variables (gitignored)
├── .env.example              # Example environment file
└── README.md
```

### Development Workflow

1. Make sure your conda environment is activated:
```bash
conda activate plantdoctor
```

2. Make your changes to the code

3. Test the application locally:
```bash
streamlit run app/login.py
```

4. For development with auto-reload, Streamlit automatically watches for file changes

### Key Technologies

- **Frontend**: Streamlit
- **ML Framework**: PyTorch
- **AI Integration**: Google Gemini AI
- **Database**: MongoDB with PyMongo
- **Authentication**: bcrypt for password hashing
- **Encryption**: cryptography (Fernet) for API key storage

### Common Development Tasks

#### Updating the conda environment

When `environment.yml` changes (new dependencies added), update your environment:

```bash
# Update the environment with new dependencies
conda env update -f environment.yml --prune

# Or if you prefer, remove and recreate the environment
conda env remove -n plantdoctor
conda env create -f environment.yml
```

#### Adding a new dependency

1. Install the package:
```bash
conda activate plantdoctor
conda install package_name
# or for pip packages
pip install package_name
```

2. Update `environment.yml`:
```bash
# Export without builds for cross-platform compatibility
conda env export --no-builds > environment.yml
```

3. Commit the updated `environment.yml` to version control

#### Updating the CNN model
Edit `models/plant_disease_cnn.py` to modify the model architecture.

#### Adding new UI pages
Create a new file in `app/pages/` and it will automatically appear in the Streamlit navigation.

### Troubleshooting

- **Module not found errors**: Make sure the conda environment is activated
- **MongoDB connection issues**: Verify your `MONGO_URI` in `.env`
- **Gemini API errors**: Check your `GEMINI_KEY` is valid and has proper permissions. Test your key with:
  ```python
  import google.generativeai as genai
  import os
  from config import GEMINI_KEY

  genai.configure(api_key=GEMINI_KEY)

  model = genai.GenerativeModel("models/gemini-2.5-flash")
  resp = model.generate_content("Say OK")
  print(resp.text)
  ```
- **Port already in use**: Use `streamlit run app/login.py --server.port 8502` to run on a different port
