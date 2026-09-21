import numpy as np
import tensorflow as tf

from sklearn.metrics import classification_report


# -----------------------------
# Paths
# -----------------------------
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
# Load test data
# -----------------------------
print("Loading test data...")

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
# Predictions
# -----------------------------
print("Making predictions...")

predictions = model.predict(X_test, verbose=1)

y_pred = np.argmax(predictions, axis=1)


# -----------------------------
# Classification report
# -----------------------------
print("\nClassification Report")
print("=" * 80)

report = classification_report(
    y_test,
    y_pred,
    target_names=class_names,
    digits=3
)

print(report)