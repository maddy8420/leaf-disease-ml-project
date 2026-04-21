from PIL import Image, ImageOps
import numpy as np
import io

def preprocess_image(image_bytes):
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    image = ImageOps.exif_transpose(image)

    w, h = image.size
    min_dim = min(w, h)
    left = (w - min_dim) // 2
    top = (h - min_dim) // 2

    image = image.crop((left, top, left + min_dim, top + min_dim))
    image = image.resize((224, 224))

    img_array = np.array(image).astype("float32") / 255.0
    return image, np.expand_dims(img_array, axis=0)