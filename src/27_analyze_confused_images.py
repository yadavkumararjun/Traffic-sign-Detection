import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

# ============================================================
# 1. SETTINGS
# ============================================================

MODEL_PATH = "models/traffic_sign_targeted_model.keras"

X_TEST_PATH = "data/X_test.npy"
Y_TEST_PATH = "data/y_test_semantic.npy"

OUTPUT_DIR = "results/confused_classes"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Original class IDs from your dataset
TARGET_CLASSES = {
    46: "Turn left",
    47: "Turn right",
    42: "Steep ascent",
    43: "Steep descent"
}

# ============================================================
# 2. LOAD DATA
# ============================================================

print("\nLoading test data...")

X_test = np.load(X_TEST_PATH)
y_test = np.load(Y_TEST_PATH)

print("X_test shape:", X_test.shape)
print("y_test shape:", y_test.shape)

# ============================================================
# 3. LOAD MODEL
# ============================================================

print("\nLoading model...")

model = tf.keras.models.load_model(MODEL_PATH)

print("Model loaded successfully!")

# ============================================================
# 4. MAKE PREDICTIONS
# ============================================================

print("\nMaking predictions...")

predictions = model.predict(X_test, verbose=1)

y_pred = np.argmax(predictions, axis=1)

print("Predictions completed!")

# ============================================================
# 5. FIND CONFUSIONS
# ============================================================

print("\n" + "=" * 60)
print("CONFUSION ANALYSIS")
print("=" * 60)

for true_id, true_name in TARGET_CLASSES.items():

    print(f"\nTrue class: {true_name}")

    # Find all images belonging to this class
    class_indices = np.where(y_test == true_id)[0]

    # Find incorrect predictions
    wrong_indices = [
        index
        for index in class_indices
        if y_pred[index] != true_id
    ]

    print("Total images:", len(class_indices))
    print("Wrong predictions:", len(wrong_indices))

    if len(wrong_indices) == 0:
        print("No incorrect predictions.")
        continue

    # Count predicted classes
    prediction_counts = {}

    for index in wrong_indices:

        predicted_id = y_pred[index]

        prediction_counts[predicted_id] = (
            prediction_counts.get(predicted_id, 0) + 1
        )

    print("\nPredicted as:")

    sorted_predictions = sorted(
        prediction_counts.items(),
        key=lambda x: x[1],
        reverse=True
    )

    for predicted_id, count in sorted_predictions:

        predicted_name = TARGET_CLASSES.get(
            predicted_id,
            f"Class {predicted_id}"
        )

        print(
            f"  {predicted_name}: {count}"
        )

    # ========================================================
    # 6. SAVE EXAMPLES
    # ========================================================

    examples = wrong_indices[:20]

    if len(examples) == 0:
        continue

    plt.figure(figsize=(12, 10))

    number_of_examples = len(examples)

    rows = int(np.ceil(number_of_examples / 4))

    for position, index in enumerate(examples):

        plt.subplot(rows, 4, position + 1)

        image = X_test[index]

        # X_test is already normalized
        plt.imshow(image)

        true_label = TARGET_CLASSES.get(
            y_test[index],
            f"Class {y_test[index]}"
        )

        predicted_label = TARGET_CLASSES.get(
            y_pred[index],
            f"Class {y_pred[index]}"
        )

        confidence = predictions[index][y_pred[index]] * 100

        plt.title(
            f"True: {true_label}\n"
            f"Pred: {predicted_label}\n"
            f"{confidence:.1f}%"
        )

        plt.axis("off")

    plt.tight_layout()

    output_file = os.path.join(
        OUTPUT_DIR,
        f"{true_id}_{true_name.replace(' ', '_')}.png"
    )

    plt.savefig(
        output_file,
        dpi=150,
        bbox_inches="tight"
    )

    plt.close()

    print(
        f"\nSaved examples to:\n{output_file}"
    )

# ============================================================
# 7. FINISHED
# ============================================================

print("\n" + "=" * 60)
print("ANALYSIS COMPLETE")
print("=" * 60)

print("\nCheck this folder:")
print(OUTPUT_DIR)