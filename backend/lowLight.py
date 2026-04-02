import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
import io
import zipfile
import os


def load_lol_model(path="models/LOW_LIGHT_MODEL.h5"):
    if not os.path.exists(path):
        return None
    return tf.keras.models.load_model(path, compile=False)

# Preprocess image
def preprocess_image(img: Image.Image) -> np.ndarray:
    img = img.convert("RGB").resize((512, 512))
    arr = np.asarray(img).astype(np.float32) / 255.0
    return np.expand_dims(arr, axis=0)

# Postprocess and return enhanced image
def enhance_image(model, image: Image.Image, intensity: float = 3.0) -> Image.Image:
    input_tensor = preprocess_image(image)
    curve = model.predict(input_tensor)
    curve = curve * intensity  # Boost enhancement strength

    x = tf.convert_to_tensor(input_tensor)
    for i in range(8):
        a = curve[..., i*3:(i+1)*3]
        x = x + a * (tf.square(x) - x)

    enhanced = x[0].numpy()
    enhanced = np.clip(enhanced * 255.0, 0, 255).astype(np.uint8)
    return Image.fromarray(enhanced)