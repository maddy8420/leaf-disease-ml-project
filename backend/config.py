import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_PATH = os.path.join(BASE_DIR, "data", "diseases.json")
MODEL_PATH = os.path.join(BASE_DIR, "models", "disease_cnn_model.keras")
CLASS_PATH = os.path.join(BASE_DIR, "models", "disease_classes.txt")

UPLOAD_FOLDER = "uploads"
RESULT_FOLDER = "results"
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"}