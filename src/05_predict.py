import os
import sys
import numpy as np
import pandas as pd
import cv2
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import tensorflow as tf


# ============================================
# CONFIGURATION
# ============================================

MODEL_PATH = "traffic_sign_model.keras"

CSV_PATH = "dataset/Indian-Traffic Sign-Dataset/traffic_sign.csv"


# ============================================
# 1. LOAD MODEL
# ============================================

print("================================")
print("      TRAFFIC SIGN PREDICTION")
print("================================")

print("\nLoading trained model...")

model = tf.keras.models.load_model(MODEL_PATH)

print("Model loaded successfully!")


# ============================================
# 2. LOAD CLASS NAMES
# ============================================

print("\nLoading class names...")

df = pd.read_csv(CSV_PATH)

# Remove Class 39 because its image folder is missing
df = df[df["ClassId"] != 39]

# Reset row numbers
df = df.reset_index(drop=True)

print("Classes loaded:", len(df))


# ============================================
# 3. GET IMAGE PATH
# ============================================

if len(sys.argv) < 2:
    print("\nUsage:")
    print("python src/05_predict.py <image_path>")
    sys.exit()

image_path = sys.argv[1]

print("\nImage:", image_path)


# ============================================
# 4. CHECK IMAGE EXISTS
# ============================================

if not os.path.exists(image_path):
    print("ERROR: Image not found!")
    sys.exit()


# ============================================
# 5. LOAD IMAGE
# ============================================

image = cv2.imread(image_path)

if image is None:
    print("ERROR: Could not read image!")
    sys.exit()


print("\nOriginal image shape:", image.shape)


# ============================================
# 6. RESIZE IMAGE
# ============================================

image = cv2.resize(image, (32, 32))

print("Resized image shape:", image.shape)


# ============================================
# 7. CONVERT BGR → RGB
# ============================================

image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


# ============================================
# 8. NORMALIZE PIXELS
# ============================================

image = image.astype("float32") / 255.0


# ============================================
# 9. ADD BATCH DIMENSION
# ============================================

image = np.expand_dims(image, axis=0)

print("Input shape for model:", image.shape)


# ============================================
# 10. MAKE PREDICTION
# ============================================

predictions = model.predict(image, verbose=0)

# Get probabilities for this image
probabilities = predictions[0]

# Get indexes of top 5 predictions
top_5 = np.argsort(probabilities)[-5:][::-1]


# ============================================
# 11. DISPLAY TOP 5 PREDICTIONS
# ============================================

print("\n================================")
print("       TOP 5 PREDICTIONS")
print("================================")

for rank, model_label in enumerate(top_5, start=1):

    class_id = df.iloc[model_label]["ClassId"]
    class_name = df.iloc[model_label]["Name"]

    confidence = probabilities[model_label] * 100

    print(
        f"{rank}. {class_name:<35} "
        f"{confidence:.2f}%"
    )

print("================================")
# ============================================
# 12. DISPLAY RESULT
# ============================================

# print("\n================================")
# print("          PREDICTION")
# print("================================")

# print("Model label :", predicted_label)
# print("Class ID    :", int(class_id))
# print("Sign        :", class_name)
# print("Confidence  :", f"{confidence * 100:.2f}%")

# print("================================")