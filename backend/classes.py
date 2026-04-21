from tensorflow import keras
import tensorflow as tf

# Load model
model = keras.models.load_model("models/disease_cnn_model.h5")
model(tf.zeros((1, 224, 224, 3)))  # force build

# Load classes
with open("models/disease_classes.txt") as f:
    disease_classes = [line.strip() for line in f]

print("\n📌 Disease class index mapping:\n")
for i, name in enumerate(disease_classes):
    print(f"{i}: {name}")

print("\n✅ Total classes:", len(disease_classes))
