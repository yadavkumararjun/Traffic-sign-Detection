import numpy as np
import tensorflow as tf

from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    accuracy_score
)


# ============================================
# HEADER
# ============================================

print("================================")
print("      MODEL EVALUATION")
print("================================")


# ============================================
# 1. LOAD TEST DATA
# ============================================

print("\nLoading test data...")

X_test = np.load("data/X_test.npy")
y_test = np.load("data/y_test.npy")

print("X_test shape :", X_test.shape)
print("y_test shape :", y_test.shape)


# ============================================
# 2. REMAP TEST LABELS
# ============================================

print("\nRemapping test labels...")

# Original dataset has 59 class IDs,
# but class 39 has no image folder.

# Create mapping:
# 0 -> 0
# 1 -> 1
# ...
# 38 -> 38
# 40 -> 39
# 41 -> 40
# ...
# 58 -> 57

label_mapping = {}

new_label = 0

for original_label in range(59):

    if original_label == 39:
        continue

    label_mapping[original_label] = new_label

    new_label += 1


# Apply mapping to y_test

y_test_remapped = np.array([
    label_mapping[label]
    for label in y_test
])

print("Original labels :", len(np.unique(y_test)))
print("Remapped labels :", len(np.unique(y_test_remapped)))

print(
    "Minimum label :",
    y_test_remapped.min()
)

print(
    "Maximum label :",
    y_test_remapped.max()
)


# ============================================
# 3. LOAD TRAINED MODEL
# ============================================

print("\nLoading trained model...")

model = tf.keras.models.load_model(
    "traffic_sign_model.keras"
)

print("Model loaded successfully!")


# ============================================
# 4. MAKE PREDICTIONS
# ============================================

print("\nMaking predictions...")

predictions = model.predict(
    X_test,
    verbose=1
)

print(
    "Prediction shape :",
    predictions.shape
)


# ============================================
# 5. CONVERT PROBABILITIES
#    INTO CLASS LABELS
# ============================================

y_pred = np.argmax(
    predictions,
    axis=1
)

print(
    "Prediction label shape :",
    y_pred.shape
)


# ============================================
# 6. ACCURACY
# ============================================

accuracy = accuracy_score(
    y_test_remapped,
    y_pred
)

print("\n================================")
print("           ACCURACY")
print("================================")

print(
    f"Accuracy: {accuracy * 100:.2f}%"
)


# ============================================
# 7. CLASSIFICATION REPORT
# ============================================

print("\n================================")
print("      CLASSIFICATION REPORT")
print("================================")

print(
    classification_report(
        y_test_remapped,
        y_pred,
        zero_division=0
    )
)


# ============================================
# 8. CONFUSION MATRIX
# ============================================

cm = confusion_matrix(
    y_test_remapped,
    y_pred
)

print("\n================================")
print("      CONFUSION MATRIX")
print("================================")

print(
    "Confusion matrix shape:",
    cm.shape
)


# ============================================
# COMPLETE
# ============================================

print("\nEvaluation completed successfully!")