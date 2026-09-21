import os

import cv2
import numpy as np
import tensorflow as tf


# ==========================================
# 1. Paths
# ==========================================

DATASET_PATH = "dataset/Indian-Traffic Sign-Dataset"
IMAGE_PATH = os.path.join(DATASET_PATH, "Images")

MODEL_PATH = "models/traffic_sign_targeted_model.keras"
CLASS_NAMES_PATH = "data/semantic_class_names.txt"


# ==========================================
# 2. Settings
# ==========================================

IMAGE_SIZE = 32

# Classes we want to investigate
TARGET_CLASSES = [
    "Turn left",
    "Turn right",
    "Steep ascent",
    "Steep descent"
]


# ==========================================
# 3. Load model
# ==========================================

print("\nLoading model...")

model = tf.keras.models.load_model(MODEL_PATH)

print("Model loaded successfully!")


# ==========================================
# 4. Load semantic class names
# ==========================================

class_names = []

with open(CLASS_NAMES_PATH, "r") as file:

    for line in file:

        class_id, name = line.strip().split(",", 1)

        class_names.append(name)


# Create:
# name -> semantic class ID

name_to_semantic_id = {
    name: index
    for index, name in enumerate(class_names)
}


print("\nSemantic classes loaded:", len(class_names))


# ==========================================
# 5. Current preprocessing
# ==========================================

def current_preprocess(image):

    # BGR -> RGB
    image = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2RGB
    )

    # Direct resize
    image = cv2.resize(
        image,
        (IMAGE_SIZE, IMAGE_SIZE)
    )

    # Normalize
    image = image.astype(
        np.float32
    ) / 255.0

    return image


# ==========================================
# 6. Aspect-ratio preserving preprocessing
# ==========================================

def padded_preprocess(image):

    # BGR -> RGB
    image = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2RGB
    )

    original_height, original_width = image.shape[:2]

    # Calculate scale
    scale = min(
        IMAGE_SIZE / original_width,
        IMAGE_SIZE / original_height
    )

    new_width = int(
        original_width * scale
    )

    new_height = int(
        original_height * scale
    )

    # Resize while preserving aspect ratio
    resized = cv2.resize(
        image,
        (new_width, new_height)
    )

    # Create 32x32 black image
    padded = np.zeros(
        (IMAGE_SIZE, IMAGE_SIZE, 3),
        dtype=np.uint8
    )

    # Center the image
    x_offset = (
        IMAGE_SIZE - new_width
    ) // 2

    y_offset = (
        IMAGE_SIZE - new_height
    ) // 2

    padded[
        y_offset:y_offset + new_height,
        x_offset:x_offset + new_width
    ] = resized

    # Normalize
    padded = padded.astype(
        np.float32
    ) / 255.0

    return padded


# ==========================================
# 7. Get original class folders
# ==========================================

class_folders = [
    folder
    for folder in os.listdir(IMAGE_PATH)
    if os.path.isdir(
        os.path.join(IMAGE_PATH, folder)
    )
]

class_ids = sorted(
    int(folder)
    for folder in class_folders
)


# ==========================================
# 8. Build original ClassId -> Name mapping
# ==========================================

import pandas as pd

df = pd.read_csv(
    os.path.join(
        DATASET_PATH,
        "traffic_sign.csv"
    )
)

class_to_name = dict(
    zip(
        df["ClassId"],
        df["Name"]
    )
)


# ==========================================
# 9. Find original ClassIds for target classes
# ==========================================

target_class_ids = {}

for class_id in class_ids:

    name = class_to_name[class_id]

    if name in TARGET_CLASSES:

        if name not in target_class_ids:

            target_class_ids[name] = []

        target_class_ids[name].append(class_id)


print("\nTarget class mapping:")

for name in TARGET_CLASSES:

    print(
        f"{name}: "
        f"{target_class_ids.get(name, [])}"
    )


# ==========================================
# 10. Prediction function
# ==========================================

def predict_image(image, preprocessing_function):

    processed = preprocessing_function(image)

    # Add batch dimension
    processed = np.expand_dims(
        processed,
        axis=0
    )

    predictions = model.predict(
        processed,
        verbose=0
    )[0]

    predicted_id = np.argmax(predictions)

    predicted_name = class_names[
        predicted_id
    ]

    confidence = predictions[
        predicted_id
    ] * 100

    return predicted_name, confidence


# ==========================================
# 11. Compare preprocessing
# ==========================================

print("\n")
print("=" * 60)
print("PREPROCESSING COMPARISON")
print("=" * 60)


for target_name in TARGET_CLASSES:

    correct_current = 0
    correct_padded = 0

    total_images = 0

    class_ids_for_name = target_class_ids.get(
        target_name,
        []
    )

    # --------------------------------------
    # Loop through all original ClassIds
    # having this semantic name
    # --------------------------------------

    for class_id in class_ids_for_name:

        class_path = os.path.join(
            IMAGE_PATH,
            str(class_id)
        )

        image_files = [
            file
            for file in os.listdir(class_path)
            if file.lower().endswith(
                (".png", ".jpg", ".jpeg")
            )
        ]

        for image_file in image_files:

            image_path = os.path.join(
                class_path,
                image_file
            )

            image = cv2.imread(
                image_path
            )

            if image is None:
                continue

            total_images += 1

            # ----------------------------------
            # Current preprocessing
            # ----------------------------------

            current_prediction, _ = predict_image(
                image,
                current_preprocess
            )

            if current_prediction == target_name:

                correct_current += 1

            # ----------------------------------
            # Padded preprocessing
            # ----------------------------------

            padded_prediction, _ = predict_image(
                image,
                padded_preprocess
            )

            if padded_prediction == target_name:

                correct_padded += 1

    # ======================================
    # Accuracy
    # ======================================

    if total_images > 0:

        current_accuracy = (
            correct_current / total_images
        ) * 100

        padded_accuracy = (
            correct_padded / total_images
        ) * 100

    else:

        current_accuracy = 0
        padded_accuracy = 0


    # ======================================
    # Results
    # ======================================

    print("\nClass:", target_name)

    print(
        "Original images:",
        total_images
    )

    print(
        "Current resize:",
        f"{correct_current}/{total_images}",
        f"({current_accuracy:.2f}%)"
    )

    print(
        "Padded resize:",
        f"{correct_padded}/{total_images}",
        f"({padded_accuracy:.2f}%)"
    )


print("\n")
print("=" * 60)
print("COMPARISON COMPLETE")
print("=" * 60)