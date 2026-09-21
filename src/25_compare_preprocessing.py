import cv2
import numpy as np
import tensorflow as tf


MODEL_PATH = "models/traffic_sign_targeted_model.keras"
X_TEST_PATH = "data/X_test.npy"
Y_TEST_PATH = "data/y_test_semantic.npy"
CLASS_NAMES_PATH = "data/semantic_class_names.txt"


# -----------------------------
# Load model
# -----------------------------
print("Loading model...")

model = tf.keras.models.load_model(MODEL_PATH)


# -----------------------------
# Load data
# -----------------------------
X_test = np.load(X_TEST_PATH)
y_test = np.load(Y_TEST_PATH)


# -----------------------------
# Load class names
# -----------------------------
class_names = []

with open(CLASS_NAMES_PATH, "r") as file:
    for line in file:
        class_id, name = line.strip().split(",", 1)
        class_names.append(name)


# -----------------------------
# Current preprocessing
# -----------------------------
def current_preprocessing(image):
    image = image.astype(np.float32) / 255.0
    return image


# -----------------------------
# Aspect-ratio preserving
# -----------------------------
def padded_preprocessing(image):
    h, w = image.shape[:2]

    scale = min(32 / w, 32 / h)

    new_w = int(w * scale)
    new_h = int(h * scale)

    resized = cv2.resize(image, (new_w, new_h))

    canvas = np.zeros((32, 32, 3), dtype=np.uint8)

    x_offset = (32 - new_w) // 2
    y_offset = (32 - new_h) // 2

    canvas[
        y_offset:y_offset + new_h,
        x_offset:x_offset + new_w
    ] = resized

    canvas = canvas.astype(np.float32) / 255.0

    return canvas


# -----------------------------
# Compare selected classes
# -----------------------------
classes_to_check = [
    "Turn left",
    "Turn right",
    "Steep ascent",
    "Steep descent"
]

class_ids = []

for name in classes_to_check:
    class_ids.append(class_names.index(name))


print("\nTesting preprocessing methods...")
print("=" * 70)


for class_id in class_ids:

    indices = np.where(y_test == class_id)[0]

    correct_current = 0
    correct_padded = 0

    print(f"\nClass: {class_names[class_id]}")
    print(f"Test images: {len(indices)}")

    for index in indices:

        image = X_test[index]

        # Current
        current = current_preprocessing(image)
        current_input = np.expand_dims(current, axis=0)

        current_prediction = model.predict(
            current_input,
            verbose=0
        )[0]

        current_class = np.argmax(current_prediction)

        # Padded
        padded = padded_preprocessing(
            (image * 255).astype(np.uint8)
        )

        padded_input = np.expand_dims(padded, axis=0)

        padded_prediction = model.predict(
            padded_input,
            verbose=0
        )[0]

        padded_class = np.argmax(padded_prediction)

        if current_class == class_id:
            correct_current += 1

        if padded_class == class_id:
            correct_padded += 1

    print(
        f"Current resize accuracy: "
        f"{correct_current}/{len(indices)} "
        f"({correct_current / len(indices) * 100:.1f}%)"
    )

    print(
        f"Padded resize accuracy:  "
        f"{correct_padded}/{len(indices)} "
        f"({correct_padded / len(indices) * 100:.1f}%)"
    )