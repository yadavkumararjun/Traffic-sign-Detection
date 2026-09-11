import cv2
import numpy as np
import tensorflow as tf


# MODEL_PATH = "models/traffic_sign_targeted_model.keras"
MODEL_PATH = "models/traffic_sign_semantic_model.keras"
CLASS_NAMES_PATH = "data/semantic_class_names.txt"


# Load model once
model = tf.keras.models.load_model(MODEL_PATH)


# Load class names
class_names = []

with open(CLASS_NAMES_PATH, "r") as file:
    for line in file:
        class_id, name = line.strip().split(",", 1)
        class_names.append(name)


def preprocess_image(image):
    """
    Preprocess an image for the CNN model.
    """

    # OpenCV uses BGR, convert to RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Resize to model input size
    image_resized = cv2.resize(image_rgb, (32, 32))

    # Normalize pixels
    image_resized = image_resized.astype(np.float32) / 255.0

    # Add batch dimension
    image_input = np.expand_dims(image_resized, axis=0)

    return image_input


def predict(image, top_k=5):
    """
    Predict traffic sign and return top-k results.
    """

    processed_image = preprocess_image(image)

    predictions = model.predict(processed_image, verbose=0)[0]

    top_indices = np.argsort(predictions)[-top_k:][::-1]

    results = []

    for index in top_indices:
        results.append({
            "class_id": int(index),
            "name": class_names[index],
            "confidence": float(predictions[index] * 100)
        })

    return results