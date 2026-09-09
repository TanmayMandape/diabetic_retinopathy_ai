"""
Central place for every path used in the project.
Every other script imports from here so training and inference
always agree on where things live (this was one of the bugs in
the original draft: train.py saved to ../models/... while
predict.py loaded from models/...).
"""

import os

# Root of the project = one level above this file's src/ folder
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_DIR = os.path.join(BASE_DIR, "data")
IMG_DIR = os.path.join(DATA_DIR, "train_images")
TRAIN_CSV = os.path.join(DATA_DIR, "train.csv")

MODELS_DIR = os.path.join(BASE_DIR, "models")
MODEL_PATH = os.path.join(MODELS_DIR, "dr_model.pth")

APP_DIR = os.path.join(BASE_DIR, "app")
UPLOAD_DIR = os.path.join(APP_DIR, "uploads")
GRADCAM_DIR = os.path.join(APP_DIR, "static", "gradcam")

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(GRADCAM_DIR, exist_ok=True)

CLASS_NAMES = [
    "No Diabetic Retinopathy",
    "Mild Diabetic Retinopathy",
    "Moderate Diabetic Retinopathy",
    "Severe Diabetic Retinopathy",
    "Proliferative Diabetic Retinopathy",
]

IMAGE_SIZE = 224
