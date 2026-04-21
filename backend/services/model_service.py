from tensorflow import keras
import tensorflow as tf
from config import MODEL_PATH, CLASS_PATH

cnn_model = None
disease_classes = []

def load_model_data():
    global cnn_model, disease_classes

    cnn_model = keras.models.load_model(MODEL_PATH)
    cnn_model(tf.zeros((1, 224, 224, 3)))

    with open(CLASS_PATH) as f:
        disease_classes = [line.strip() for line in f]

    return cnn_model, disease_classes