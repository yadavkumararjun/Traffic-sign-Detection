
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

MODEL_PATH = "models/traffic_sign_robust_model.keras"
X_TEST_PATH = "data/X_test.npy"
Y_TEST_PATH = "data/y_test_semantic.npy"
CLASS_NAMES_PATH = "data/semantic_class_names.txt"

# Load the trained model and test data.
model = tf.keras.models.load_model(MODEL_PATH)

X_test = np.load(X_TEST_PATH)
y_test = np.load(Y_TEST_PATH).astype(int)

with open(CLASS_NAMES_PATH, "r", encoding="utf-8") as file:
    class_names = [line.strip() for line in file if line.strip()]

num_classes = len(class_names)

print("Test images:", len(X_test))
print("Test labels:", len(y_test))
print("Classes:", num_classes)
print("Model output classes:", model.output_shape[-1])

if len(X_test) != len(y_test):
    raise ValueError("Test images and labels have different lengths.")

if model.output_shape[-1] != num_classes:
    raise ValueError(
        "Model output size does not match the class-name list."
    )

if y_test.min() < 0 or y_test.max() >= num_classes:
    raise ValueError("Test labels are outside the class-name range.")

# Predict in batches to limit memory usage.
probabilities = model.predict(X_test, batch_size=32, verbose=1)
y_pred = np.argmax(probabilities, axis=1)

# Build the matrix in the fixed order of all 52 classes.
labels = np.arange(num_classes)
cm = confusion_matrix(y_test, y_pred, labels=labels)

# Use short numeric tick labels to keep the plot readable.
display = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=labels
)

fig, ax = plt.subplots(figsize=(14, 14))
display.plot(
    ax=ax,
    cmap="Blues",
    values_format="d",
    colorbar=True,
    xticks_rotation="vertical"
)

ax.set_title("Confusion Matrix — Indian Traffic Sign Classification")
ax.set_xlabel("Predicted class ID")
ax.set_ylabel("True class ID")

plt.tight_layout()
plt.savefig("fig4_confusion_matrix.png",
            dpi=300, bbox_inches="tight")
plt.show()

accuracy = np.mean(y_test == y_pred)
print(f"Accuracy from this test run: {accuracy * 100:.2f}%")
print("Saved: fig4_confusion_matrix.png")
