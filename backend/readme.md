# 🌿 Leaf Disease Detection — CNN & Vision Transformer

> **1st Prize — Project Expo**

[![Python](https://img.shields.io/badge/Python-3.10.11-blue?logo=python)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange?logo=tensorflow)](https://www.tensorflow.org/)
[![Flask](https://img.shields.io/badge/Flask-Web%20Framework-black?logo=flask)](https://flask.palletsprojects.com/)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

---

##  Overview

An end-to-end AI-powered plant pathology system that classifies leaf diseases from images using a dual-model architecture — **MobileNetV2 (CNN)** and **Vision Transformer (ViT)**. The system provides model explainability via **Grad-CAM heatmaps** and integrates an **LLM-based agricultural chatbot** (LLaMA3 via Ollama) for contextual disease guidance.

---

##  Architecture

### Model Comparison

| Model | Type | Backbone | Strengths | Limitations |
|---|---|---|---|---|
| MobileNetV2 | CNN | Depthwise Separable Convolutions | Lightweight, high accuracy on small datasets, fast inference | Limited global context modeling |
| Vision Transformer (ViT) | Transformer | Multi-Head Self-Attention + Patch Embeddings | Captures global spatial relationships | Requires large datasets & high compute |



##  Features

-  **Dual-model inference** — switch between CNN and ViT backends at runtime
-  **Explainable AI (XAI)** — Grad-CAM visualizations for CNN; attention rollout for ViT
-  **Agricultural chatbot** — LLaMA3 (via Ollama) for disease-specific natural language queries
-  **Model benchmarking** — side-by-side accuracy and confidence score comparison
-  **REST API** — Flask endpoints for prediction, visualization, and chatbot interaction

---

##  Tech Stack

| Layer | Technology |
|---|---|
| Language | Python 3.10.11 |
| Deep Learning | TensorFlow 2.x / Keras |
| CNN Model | MobileNetV2 (ImageNet pretrained, fine-tuned) |
| Transformer Model | ViT (vit-keras) |
| Explainability | Grad-CAM (custom implementation via TF GradientTape) |
| Image Processing | OpenCV, NumPy |
| Web Framework | Flask |
| LLM Chatbot | LLaMA3 via Ollama (local inference) |

---

##  Project Structure

```
backend/
├── app.py                
├── app_vit.py              
├── train_cnn.py
├── tarin_crop_model.py 
├── config.py                
├── requirements.txt
│
├── data/
│   ├── diseases.json     
│ 
│
├── models/
│   ├── disease_cnn_model.h5     
│   └── vit_model.h5      
│   └── disease_classes     
├── services/
│   ├── chatbot_service.py         
│   ├── disease_service.py          
│   └── gradcam_service.py
│   └── image_service.py    
│   └── model_service.py            
└── templates/
    └── index.html 
     └── leaf.html
      └── climate.html 
       └── vit.html    

    
```

---

##  Setup & Installation

### Prerequisites

- Python `3.10.11`
- [Ollama](https://ollama.com/) installed locally (for chatbot)
- GPU recommended for ViT training/inference

### 1. Clone the Repository

```bash
git clone https://github.com/<your-username>/leaf-disease-detection.git
cd leaf-disease-detection/backend
```

### 2. Create Virtual Environment

```bash
python -m venv venv
source venv/bin/activate        # Linux/macOS
venv\Scripts\activate           # Windows
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Pull LLaMA3 Model (for chatbot)

```bash
ollama pull llama3
```

### 5. Run the Application

**CNN (MobileNetV2) backend:**
```bash
python app.py
```

**Vision Transformer backend:**
```bash
python app_vit.py
```

Access the web interface at `http://localhost:5000`

---

##  Results

| Model | Accuracy | Avg. Inference Time | Notes |
|---|---|---|---|
| CNN (MobileNetV2) | High | ~50ms | Stable on limited datasets; suitable for deployment |
| ViT | Moderate | ~200ms | Improves significantly with larger datasets |

> **Note:** Quantitative metrics depend on dataset size and hardware. Results above reflect training on the PlantVillage dataset subset.

##  Limitations

- **ViT** requires substantially more training data and compute to outperform CNNs on small agricultural datasets
- **Chatbot** performance is constrained by available system RAM (LLaMA3 ~4–8GB VRAM/RAM)
- **Model weights** are excluded from the repository due to file size; see training scripts to reproduce

---

##  Model Weights

Pre-trained weights are not included in this repository. To obtain them:

1. **Retrain** using the provided dataset structure and `config.py` hyperparameters
2. Or **contact the author** for direct access to trained `.h5` files

---

##  Author

**Madhu Sudhan**
-  Madhu Suhan (madhusudhancbtech24@rvu.edu.in)
-  [LinkedIn / GitHub]


##  Conclusion

This project empirically evaluates CNN and Transformer architectures for agricultural image classification. MobileNetV2 proves more practical for constrained environments, while ViT demonstrates superior scalability potential. The integrated Grad-CAM XAI module and LLM chatbot bridge the gap between model predictions and actionable agronomic insights.