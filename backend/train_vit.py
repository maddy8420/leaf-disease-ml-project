from vit_keras import vit
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt   
import os

# ================= CONFIG =================
DATASET_PATH = "dataset"
IMG_SIZE = (224, 224)
BATCH_SIZE = 8
EPOCHS = 2

# ================= DATA =================
datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2
)

train_gen = datagen.flow_from_directory(
    DATASET_PATH,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    subset="training"
)

val_gen = datagen.flow_from_directory(
    DATASET_PATH,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    subset="validation"
)

NUM_CLASSES = train_gen.num_classes

# ================= MODEL =================
model = vit.vit_b16(
    image_size=224,
    activation='softmax',
    pretrained=True,
    include_top=True,
    pretrained_top=False,
    classes=NUM_CLASSES
)

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-4),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

# ================= TRAIN =================
print(" Training ViT on partial dataset (~40%)")
print("Steps per epoch: 80 | Validation steps: 20")


history = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=EPOCHS,
    steps_per_epoch=80,
    validation_steps=20,
    callbacks=[EarlyStopping(patience=1)]
)

# ================= SAVE =================
os.makedirs("models", exist_ok=True)
model.save("models/vit_model.h5", include_optimizer=False)

print(" ViT Training Completed")

# ================= GRAPH =================
plt.figure(figsize=(10,4))

# Accuracy
plt.subplot(1,2,1)
plt.plot(history.history["accuracy"], label="Train")
plt.plot(history.history["val_accuracy"], label="Val")
plt.title("ViT Accuracy")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()

# Loss
plt.subplot(1,2,2)
plt.plot(history.history["loss"], label="Train")
plt.plot(history.history["val_loss"], label="Val")
plt.title("ViT Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()

plt.tight_layout()
plt.savefig("vit_training.png")  
plt.show()