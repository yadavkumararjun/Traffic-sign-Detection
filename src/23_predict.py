import sys
import cv2
import numpy as np
import tensorflow as tf


# ==========================================
# 1. LOAD MODEL
# ==========================================

# MODEL_PATH = "traffic_sign_targeted_model.keras"
# MODEL_PATH = "traffic_sign_improved_model.keras"
MODEL_PATH = "traffic_sign_semantic_model.keras"
# MODEL_PATH = "traffic_sign_model_augmented.keras"

model = tf.keras.models.load_model(MODEL_PATH)

print("Model loaded successfully.")


# ==========================================
# 2. LOAD CLASS NAMES
# ==========================================

class_names = []

with open("data/semantic_class_names.txt", "r") as file:

    for line in file:

        class_id, name = line.strip().split(",", 1)

        class_names.append(name)


print("Number of classes:", len(class_names))


# ==========================================
# 3. GET IMAGE PATH
# ==========================================

if len(sys.argv) < 2:

    print("\nUsage:")
    print("python src/23_predict.py <image_path>")

    sys.exit()


image_path = sys.argv[1]


# ==========================================
# 4. LOAD IMAGE
# ==========================================

image = cv2.imread(image_path)

if image is None:

    print("ERROR: Could not load image.")

    sys.exit()


print("Original image shape:", image.shape)


# ==========================================
# 5. PREPROCESS IMAGE
# ==========================================

# OpenCV loads images as BGR
# Convert to RGB

image_rgb = cv2.cvtColor(
    image,
    cv2.COLOR_BGR2RGB
)


# Resize to model input size

image_resized = cv2.resize(
    image_rgb,
    (32, 32)
)


# Convert to float

image_resized = image_resized.astype(
    np.float32
)


# Normalize pixels to 0-1

image_resized = image_resized / 255.0


# Add batch dimension

input_image = np.expand_dims(
    image_resized,
    axis=0
)


print("Input shape:", input_image.shape)


# ==========================================
# 6. MAKE PREDICTION
# ==========================================

predictions = model.predict(
    input_image,
    verbose=0
)


probabilities = predictions[0]


# ==========================================
# 7. TOP-5 PREDICTIONS
# ==========================================

top_5_indices = np.argsort(
    probabilities
)[-5:][::-1]


print("\n==============================")
print("TRAFFIC SIGN PREDICTION")
print("==============================")


for rank, index in enumerate(
    top_5_indices,
    start=1
):

    sign_name = class_names[index]

    confidence = probabilities[index] * 100

    print(
        f"{rank}. {sign_name:<40} "
        f"{confidence:.2f}%"
    )


# ==========================================
# 8. FINAL PREDICTION
# ==========================================

predicted_index = top_5_indices[0]

predicted_name = class_names[
    predicted_index
]

predicted_confidence = (
    probabilities[predicted_index] * 100
)


print("\n==============================")
print("FINAL PREDICTION")
print("==============================")

print("Sign       :", predicted_name)

print(
    f"Confidence : {predicted_confidence:.2f}%"
)