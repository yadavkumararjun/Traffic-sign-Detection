from pathlib import Path

import numpy as np
import tensorflow as tf
from PIL import Image


# ============================================================
# PATHS
# ============================================================

SRC_DIR = Path(__file__).resolve().parent
ROOT = SRC_DIR.parent

MODEL_PATH = (
    ROOT
    / "models"
    / "traffic_sign_robust_model.keras"
)

CLASS_NAMES_PATH = (
    ROOT
    / "data"
    / "semantic_class_names.txt"
)


# ============================================================
# SETTINGS
# ============================================================

IMAGE_SIZE = 32


# ============================================================
# LOAD MODEL
# ============================================================

def load_model():

    print("Loading traffic sign model...")

    model = tf.keras.models.load_model(
        MODEL_PATH
    )

    print("Model loaded successfully!")

    return model


# ============================================================
# LOAD CLASS NAMES
# ============================================================

def load_class_names():

    class_names = []

    with open(
        CLASS_NAMES_PATH,
        "r",
        encoding="utf-8"
    ) as file:

        for line in file:

            line = line.strip()

            if not line:
                continue

            parts = line.split(
                ",",
                1
            )

            if len(parts) == 2:

                class_names.append(
                    parts[1].strip()
                )

    return class_names


# ============================================================
# PREPROCESS IMAGE
# ============================================================

def preprocess_image(image):

    # PIL -> RGB
    image = image.convert("RGB")

    # Resize
    image = image.resize(
        (
            IMAGE_SIZE,
            IMAGE_SIZE
        )
    )

    # Convert to numpy
    image_array = np.array(
        image,
        dtype=np.float32
    )

    # Normalize
    image_array = (
        image_array / 255.0
    )

    # Add batch dimension
    image_array = np.expand_dims(
        image_array,
        axis=0
    )

    return image_array


# ============================================================
# PREDICT IMAGE
# ============================================================

def predict_image(
    model,
    image
):

    processed_image = (
        preprocess_image(image)
    )

    predictions = model.predict(
        processed_image,
        verbose=0
    )

    return predictions