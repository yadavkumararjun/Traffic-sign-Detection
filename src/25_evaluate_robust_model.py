import numpy as np
import tensorflow as tf

from sklearn.metrics import (
    classification_report,
    confusion_matrix
)


# ==========================================
# 1. SETTINGS
# ==========================================

MODEL_PATH = "models/traffic_sign_robust_model.keras"

X_TEST_PATH = "data/X_test.npy"
Y_TEST_PATH = "data/y_test_semantic.npy"

CLASS_NAMES_PATH = "data/semantic_class_names.txt"


# ==========================================
# 2. LOAD CLASS NAMES
# ==========================================

class_names = []

with open(CLASS_NAMES_PATH, "r") as file:

    for line in file:

        line = line.strip()

        if line:

            class_id, name = line.split(",", 1)

            class_names.append(name)


print("Number of classes:", len(class_names))


# ==========================================
# 3. LOAD TEST DATA
# ==========================================

print("\nLoading test data...")

X_test = np.load(X_TEST_PATH)
y_test = np.load(Y_TEST_PATH)

print("X_test:", X_test.shape)
print("y_test:", y_test.shape)


# ==========================================
# 4. LOAD MODEL
# ==========================================

print("\nLoading robust model...")

model = tf.keras.models.load_model(
    MODEL_PATH
)

print("Model loaded successfully!")


# ==========================================
# 5. MAKE PREDICTIONS
# ==========================================

print("\nMaking predictions...")

predictions = model.predict(
    X_test,
    verbose=1
)

y_pred = np.argmax(
    predictions,
    axis=1
)

print("\nPredictions completed!")


# ==========================================
# 6. OVERALL ACCURACY
# ==========================================

accuracy = np.mean(
    y_pred == y_test
)

print("\n========================================")
print("OVERALL RESULTS")
print("========================================")

print(
    f"Accuracy: {accuracy * 100:.2f}%"
)


# ==========================================
# 7. CLASSIFICATION REPORT
# ==========================================

print("\n========================================")
print("CLASSIFICATION REPORT")
print("========================================")

print(

    classification_report(

        y_test,

        y_pred,

        target_names=class_names,

        digits=3
    )
)


# ==========================================
# 8. CONFUSION MATRIX
# ==========================================

cm = confusion_matrix(
    y_test,
    y_pred
)

print("\n========================================")
print("CONFUSION MATRIX")
print("========================================")

print(
    "Shape:",
    cm.shape
)


# ==========================================
# 9. IMPORTANT CLASSES
# ==========================================

important_classes = {

    "Turn left": 46,

    "Turn right": 47,

    "Steep ascent": 42,

    "Steep descent": 43
}


print("\n========================================")
print("TARGET CLASS ANALYSIS")
print("========================================")


for class_name, class_id in important_classes.items():

    mask = (
        y_test == class_id
    )

    total = np.sum(mask)

    correct = np.sum(
        y_pred[mask] == class_id
    )

    wrong = total - correct

    class_accuracy = (
        correct / total * 100
        if total > 0
        else 0
    )

    print("\nClass:", class_name)

    print(
        f"Total images : {total}"
    )

    print(
        f"Correct      : {correct}"
    )

    print(
        f"Wrong        : {wrong}"
    )

    print(
        f"Accuracy     : {class_accuracy:.2f}%"
    )

    # Show where wrong predictions went

    if wrong > 0:

        wrong_predictions = y_pred[mask]

        wrong_predictions = (
            wrong_predictions[
                wrong_predictions != class_id
            ]
        )

        unique, counts = np.unique(
            wrong_predictions,
            return_counts=True
        )

        print("Predicted as:")

        sorted_pairs = sorted(
            zip(unique, counts),
            key=lambda x: x[1],
            reverse=True
        )

        for predicted_class, count in sorted_pairs:

            if predicted_class < len(class_names):

                predicted_name = class_names[
                    predicted_class
                ]

            else:

                predicted_name = (
                    f"Class {predicted_class}"
                )

            print(
                f"  {predicted_name}: {count}"
            )


# ==========================================
# 10. TOP CONFUSION PAIRS
# ==========================================

print("\n========================================")
print("TOP CONFUSION PAIRS")
print("========================================")


confusion_pairs = []

for true_class in range(len(class_names)):

    for predicted_class in range(len(class_names)):

        if true_class == predicted_class:

            continue

        mistakes = cm[
            true_class,
            predicted_class
        ]

        if mistakes > 0:

            confusion_pairs.append(

                (
                    mistakes,
                    true_class,
                    predicted_class
                )
            )


confusion_pairs.sort(
    reverse=True
)


for mistakes, true_class, predicted_class in confusion_pairs[:20]:

    true_name = class_names[
        true_class
    ]

    predicted_name = class_names[
        predicted_class
    ]

    print(
        f"{true_name} → {predicted_name} : "
        f"{mistakes} mistakes"
    )


# ==========================================
# 11. FINAL SUMMARY
# ==========================================

print("\n========================================")
print("EVALUATION COMPLETE")
print("========================================")

print(
    f"Robust model accuracy: "
    f"{accuracy * 100:.2f}%"
)

print("\nImportant classes checked:")

for name in important_classes:

    print(
        f"  ✓ {name}"
    )