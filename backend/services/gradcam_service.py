import cv2
import numpy as np
import os
import tensorflow as tf
from tensorflow import keras
from config import RESULT_FOLDER

def generate_gradcam(model, img_array, class_index):
    base_model = model.layers[0]
    classifier_head = model.layers[1:]
    last_conv_layer = base_model.get_layer("Conv_1")

    feature_extractor = keras.Model(base_model.input, last_conv_layer.output)

    with tf.GradientTape() as tape:
        conv_outputs = feature_extractor(img_array)
        tape.watch(conv_outputs)

        x = conv_outputs
        for layer in classifier_head:
            x = layer(x)

        loss = x[:, class_index]

    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    heatmap = tf.reduce_sum(conv_outputs[0] * pooled_grads, axis=-1)
    heatmap = tf.maximum(heatmap, 0)
    heatmap /= tf.reduce_max(heatmap) + 1e-8

    return heatmap.numpy()

def save_original_and_gradcam(original_img, heatmap, name):
    original_path = os.path.join(RESULT_FOLDER, f"original_{name}.jpg")
    cv2.imwrite(original_path, cv2.cvtColor(np.array(original_img), cv2.COLOR_RGB2BGR))

    heatmap = cv2.resize(heatmap, original_img.size)
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    original_bgr = cv2.cvtColor(np.array(original_img), cv2.COLOR_RGB2BGR)
    overlay = cv2.addWeighted(original_bgr, 0.6, heatmap, 0.4, 0)

    gradcam_path = os.path.join(RESULT_FOLDER, f"gradcam_{name}.jpg")
    cv2.imwrite(gradcam_path, overlay)

    return f"results/original_{name}.jpg", f"results/gradcam_{name}.jpg"