from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import numpy as np
import cv2
import os


from tensorflow.keras.models import load_model
from vit_keras import vit

app = Flask(__name__)
CORS(app)

# ================= LOAD MODEL =================
print("🔄 Loading ViT model...")
vit_model = load_model("models/vit_model.h5", compile=False, safe_mode=False)
print("✅ Model loaded")

# ================= LOAD CLASSES =================
with open("models/disease_classes.txt") as f:
    class_names = [line.strip() for line in f.readlines()]

print("✅ Classes loaded:", class_names)


# ================= FRONTEND =================
@app.route("/")
def home():
    return render_template("vit.html")


# ================= PREPROCESS =================
def preprocess_vit(image_bytes):
    img = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)

    if img is None:
        raise ValueError("Invalid image")

    img = cv2.resize(img, (224, 224))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)
    return img


# ================= PREDICT =================
def predict_vit(image_bytes):
    img = preprocess_vit(image_bytes)

    preds = vit_model.predict(img)
    idx = int(np.argmax(preds[0]))
    confidence = float(preds[0][idx]) * 100

    disease = class_names[idx]

    print(f"Prediction: {disease}, Confidence: {confidence:.2f}%")

    return disease, confidence


# ================= API =================
@app.route("/api/predict-vit", methods=["POST"])
def predict_vit_api():
    file = request.files.get("image")

    if not file:
        return jsonify({"error": "No image uploaded"}), 400

    try:
        disease, confidence = predict_vit(file.read())

        return jsonify({
            "model": "Vision Transformer",
            "disease": disease.replace("___", " - ").replace("_", " "),
            "confidence": f"{confidence:.2f}%"
        })

    except Exception as e:
        print("Error:", e)
        return jsonify({"error": "Prediction failed"}), 500


# ================= START =================
if __name__ == "__main__":
    print("🚀 ViT Server running on http://127.0.0.1:5001")
    app.run(host="0.0.0.0", port=5001, debug=True)