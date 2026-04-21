# train_cnn.py
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
import matplotlib.pyplot as plt
import os

# ==================== CONFIG ====================

DATASET_PATH = "dataset"
IMG_SIZE = (224, 224)
BATCH_SIZE = 32

EPOCHS_PHASE1 = 10
EPOCHS_PHASE2 = 10

MODEL_PATH = "models/disease_cnn_model.h5"
CLASSES_FILE = "models/disease_classes.txt"
HISTORY_PLOT = "models/training_history.png"

os.makedirs("models", exist_ok=True)

# ==================== DATA LOADING ====================

train_datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2,
    rotation_range=25,
    zoom_range=0.25,
    horizontal_flip=True,
    brightness_range=[0.8, 1.2]
)

val_datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2
)

train_gen = train_datagen.flow_from_directory(
    DATASET_PATH,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    subset="training"
)

val_gen = val_datagen.flow_from_directory(
    DATASET_PATH,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    subset="validation"
)

NUM_CLASSES = train_gen.num_classes

class_indices = train_gen.class_indices
class_names = sorted(class_indices, key=class_indices.get)

with open(CLASSES_FILE, "w") as f:
    for c in class_names:
        f.write(c + "\n")

print("\n✓ Classes detected:", class_names)

# ==================== MODEL ====================

base_model = MobileNetV2(
    weights="imagenet",
    include_top=False,
    input_shape=(224, 224, 3)
)

base_model.trainable = False  

model = keras.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(256, activation="relu"),
    layers.Dropout(0.5),
    layers.Dense(NUM_CLASSES, activation="softmax")
])

# ==================== CALLBACKS ====================

callbacks = [
    EarlyStopping(patience=5, restore_best_weights=True),
    ModelCheckpoint(MODEL_PATH, save_best_only=True),
    ReduceLROnPlateau(factor=0.3, patience=3, verbose=1)
]

# ==================== PHASE 1 TRAINING ====================

print("\n🔹 Phase 1: Training classifier head...")

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-4),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

history1 = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=EPOCHS_PHASE1,
    callbacks=callbacks
)

# ==================== PHASE 2 FINE-TUNING ====================

print("\n🔹 Phase 2: Fine-tuning last layers...")

base_model.trainable = True


for layer in base_model.layers[:-30]:
    layer.trainable = False

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-5),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

history2 = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=EPOCHS_PHASE2,
    callbacks=callbacks
)

# ==================== MERGE HISTORY ====================

history = {}
for key in history1.history.keys():
    history[key] = history1.history[key] + history2.history[key]

# ==================== SAVE TRAINING PLOTS ====================

plt.figure(figsize=(10,4))

plt.subplot(1,2,1)
plt.plot(history["accuracy"], label="Train")
plt.plot(history["val_accuracy"], label="Val")
plt.title("Accuracy")
plt.legend()

plt.subplot(1,2,2)
plt.plot(history["loss"], label="Train")
plt.plot(history["val_loss"], label="Val")
plt.title("Loss")
plt.legend()

plt.tight_layout()
plt.savefig(HISTORY_PLOT)
plt.close()

print("\n TRAINING COMPLETE")
print(f"✓ Model saved to: {MODEL_PATH}")
print(f"✓ Classes saved to: {CLASSES_FILE}")
print(f"✓ Training plot saved to: {HISTORY_PLOT}")
