import os 
import numpy as np
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import tensorflow as tf
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import classification_report


print("================================")
print("       CONFUSION MATRIX")
print("================================")


# ============================================
# 1. LOAD TEST DATA
# ============================================

print("\nLoading test data...")

X_test = np.load("data/X_test.npy")
y_test = np.load("data/y_test.npy")

print("Test data loaded!")

print("X_test shape:", X_test.shape)
print("y_test shape:", y_test.shape)


# ============================================
# 2. REMAP CLASS LABELS
# ============================================

print("\nRemapping test labels...")

# Original dataset contains Class IDs 0-58.
# Class 39 does not have an image folder.

existing_classes = [
    class_id
    for class_id in range(59)
    if class_id != 39
]

# Original ID → Model ID
label_mapping = {
    original_id: new_id
    for new_id, original_id in enumerate(existing_classes)
}

# Convert original test labels
# to the same labels used by the CNN.

y_test = np.array([
    label_mapping[label]
    for label in y_test
])

print("Test labels remapped successfully!")

print("Minimum test label:", y_test.min())
print("Maximum test label:", y_test.max())

print("Number of classes:", len(existing_classes))


# ============================================
# 3. LOAD TRAINED MODEL
# ============================================

print("\nLoading trained model...")

model = tf.keras.models.load_model(
    "traffic_sign_model.keras"
)

print("Model loaded successfully!")


# ============================================
# 4. PREDICT TEST DATA
# ============================================

print("\nMaking predictions...")

predictions = model.predict(
    X_test,
    verbose=1
)

# Get the class with the highest probability
y_pred = np.argmax(
    predictions,
    axis=1
)

print("\nPredictions generated!")

print("Prediction shape:", y_pred.shape)


# ============================================
# 5. CREATE CONFUSION MATRIX
# ============================================

print("\nCreating confusion matrix...")

cm = confusion_matrix(
    y_test,
    y_pred,
    labels=range(58)
)

print("Confusion matrix shape:", cm.shape)
# ============================================
# 5.1 FIND TOP CONFUSION PAIRS
# ============================================

print("\n================================")
print("       TOP CONFUSION PAIRS")
print("================================")

confusion_pairs = []

for actual in range(58):

    for predicted in range(58):

        # Ignore correct predictions
        if actual != predicted:

            confusion_pairs.append(
                (
                    cm[actual][predicted],
                    actual,
                    predicted
                )
            )

# Sort by number of mistakes
confusion_pairs.sort(
    reverse=True
)

print("\nMost common classification mistakes:\n")

count = 0

for mistakes, actual, predicted in confusion_pairs:

    if mistakes > 0:

        actual_class = existing_classes[actual]
        predicted_class = existing_classes[predicted]

        print(
            f"Actual Class {actual_class} "
            f"→ Predicted Class {predicted_class} "
            f": {mistakes} mistakes"
        )

        count += 1

        if count == 15:
            break

# ============================================
# 6. DISPLAY CONFUSION MATRIX
# ============================================

print("\nDisplaying confusion matrix...")

fig, ax = plt.subplots(figsize=(18, 18))

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=existing_classes
)

disp.plot(
    ax=ax,
    xticks_rotation=90,
    cmap="Blues",
    values_format="d",
    colorbar=True
)

plt.title(
    "Traffic Sign Confusion Matrix",
    fontsize=16
)

plt.xlabel("Predicted Class")
plt.ylabel("Actual Class")

plt.tight_layout()

plt.show()


# ============================================
# 7. CLASSIFICATION REPORT
# ============================================

print("\n================================")
print("      CLASSIFICATION REPORT")
print("================================")

report = classification_report(
    y_test,
    y_pred,
    labels=range(58),
    target_names=[
        str(class_id)
        for class_id in existing_classes
    ],
    zero_division=0
)

print(report)


# ============================================
# 8. FINAL RESULT
# ============================================

print("\n================================")
print("   CONFUSION MATRIX COMPLETE")
print("================================")