import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from sklearn.metrics import confusion_matrix


# ==========================================
# 1. LOAD DATA
# ==========================================

X_test = np.load("data/X_test.npy")
y_test = np.load("data/y_test_semantic.npy")


# ==========================================
# 2. LOAD CLASS NAMES
# ==========================================

class_names = []

with open("data/semantic_class_names.txt", "r") as file:
    for line in file:
        class_id, name = line.strip().split(",", 1)
        class_names.append(name)


# ==========================================
# 3. LOAD MODEL
# ==========================================

model = tf.keras.models.load_model(
    "traffic_sign_improved_model.keras"
)


# ==========================================
# 4. PREDICTIONS
# ==========================================

predictions = model.predict(
    X_test,
    verbose=1
)

y_pred = np.argmax(predictions, axis=1)


# ==========================================
# 5. CONFUSION MATRIX
# ==========================================

cm = confusion_matrix(
    y_test,
    y_pred,
    labels=range(len(class_names))
)


# ==========================================
# 6. FIND TOP CONFUSIONS
# ==========================================

confusion_pairs = []

for actual in range(len(class_names)):

    for predicted in range(len(class_names)):

        if actual != predicted:

            mistakes = cm[actual, predicted]

            if mistakes > 0:

                confusion_pairs.append(
                    (
                        mistakes,
                        actual,
                        predicted
                    )
                )


confusion_pairs.sort(reverse=True)

top_pairs = confusion_pairs[:15]


# ==========================================
# 7. PRINT TOP CONFUSIONS
# ==========================================

print("\n==============================")
print("TOP 15 CONFUSIONS")
print("==============================")

labels = []
values = []

for mistakes, actual, predicted in top_pairs:

    label = (
        f"{class_names[actual]}\n"
        f"→ {class_names[predicted]}"
    )

    labels.append(label)
    values.append(mistakes)

    print(
        f"{class_names[actual]} → "
        f"{class_names[predicted]} : "
        f"{mistakes}"
    )


# ==========================================
# 8. PLOT
# ==========================================

plt.figure(figsize=(14, 8))

plt.barh(
    range(len(values)),
    values
)

plt.yticks(
    range(len(values)),
    labels
)

plt.xlabel("Number of Misclassifications")

plt.ylabel("Actual → Predicted")

plt.title(
    "Top 15 Confusion Pairs - Improved CNN"
)

plt.gca().invert_yaxis()

plt.tight_layout()

plt.show()
