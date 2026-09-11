import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay,
    classification_report
)


# ==========================================
# 1. Load test data
# ==========================================

X_test = np.load("data/X_test.npy")
y_test = np.load("data/y_test_semantic.npy")

print("X_test shape:", X_test.shape)
print("y_test shape:", y_test.shape)


# ==========================================
# 2. Load semantic class names
# ==========================================

class_names = []

with open(
    "data/semantic_class_names.txt",
    "r"
) as file:

    for line in file:

        class_id, name = line.strip().split(",", 1)

        class_names.append(name)


print(
    "Number of classes:",
    len(class_names)
)


# ==========================================
# 3. Load semantic model
# ==========================================

model = tf.keras.models.load_model(
    "traffic_sign_semantic_model.keras"
)


# ==========================================
# 4. Make predictions
# ==========================================

predictions = model.predict(
    X_test,
    verbose=1
)

y_pred = np.argmax(
    predictions,
    axis=1
)

print(
    "Prediction shape:",
    y_pred.shape
)


# ==========================================
# 5. Confusion Matrix
# ==========================================

cm = confusion_matrix(
    y_test,
    y_pred,
    labels=range(len(class_names))
)


print(
    "Confusion matrix shape:",
    cm.shape
)


# ==========================================
# 6. Display confusion matrix
# ==========================================

plt.figure(figsize=(16, 16))

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=class_names
)

disp.plot(
    xticks_rotation=90,
    values_format="d"
)

plt.title(
    "Semantic Traffic Sign Classification - Confusion Matrix"
)

plt.tight_layout()
plt.show()


# ==========================================
# 7. Classification Report
# ==========================================

print("\n==============================")
print("CLASSIFICATION REPORT")
print("==============================")

print(
    classification_report(
        y_test,
        y_pred,
        labels=range(len(class_names)),
        target_names=class_names,
        zero_division=0
    )
)


# ==========================================
# 8. Find biggest confusion pairs
# ==========================================

confusion_pairs = []

for actual in range(len(class_names)):

    for predicted in range(len(class_names)):

        if actual != predicted:

            mistakes = cm[
                actual,
                predicted
            ]

            if mistakes > 0:

                confusion_pairs.append(
                    (
                        mistakes,
                        actual,
                        predicted
                    )
                )


# Sort largest mistakes first
confusion_pairs.sort(
    reverse=True
)


print("\n==============================")
print("TOP CONFUSION PAIRS")
print("==============================")


count = 0

for mistakes, actual, predicted in confusion_pairs:

    print(
        f"{class_names[actual]} "
        f"→ "
        f"{class_names[predicted]} "
        f": "
        f"{mistakes} mistakes"
    )

    count += 1

    if count == 15:
        break