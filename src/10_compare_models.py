import json
import matplotlib.pyplot as plt


# =========================
# Load baseline history
# =========================

with open("data/training_history.json", "r") as file:
    baseline_history = json.load(file)


# =========================
# Load augmented history
# =========================

with open("data/augmented_training_history.json", "r") as file:
    augmented_history = json.load(file)


# =========================
# Number of epochs
# =========================

epochs_baseline = range(
    1,
    len(baseline_history["accuracy"]) + 1
)

epochs_augmented = range(
    1,
    len(augmented_history["accuracy"]) + 1
)


# =========================
# Training Accuracy
# =========================

plt.figure(figsize=(8, 5))

plt.plot(
    epochs_baseline,
    baseline_history["accuracy"],
    label="Baseline Training Accuracy"
)

plt.plot(
    epochs_augmented,
    augmented_history["accuracy"],
    label="Augmented Training Accuracy"
)

plt.title("Training Accuracy Comparison")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.grid()

plt.show()


# =========================
# Validation Accuracy
# =========================

plt.figure(figsize=(8, 5))

plt.plot(
    epochs_baseline,
    baseline_history["val_accuracy"],
    label="Baseline Validation Accuracy"
)

plt.plot(
    epochs_augmented,
    augmented_history["val_accuracy"],
    label="Augmented Validation Accuracy"
)

plt.title("Validation Accuracy Comparison")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.grid()

plt.show()


# =========================
# Training Loss
# =========================

plt.figure(figsize=(8, 5))

plt.plot(
    epochs_baseline,
    baseline_history["loss"],
    label="Baseline Training Loss"
)

plt.plot(
    epochs_augmented,
    augmented_history["loss"],
    label="Augmented Training Loss"
)

plt.title("Training Loss Comparison")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.grid()

plt.show()


# =========================
# Validation Loss
# =========================

plt.figure(figsize=(8, 5))

plt.plot(
    epochs_baseline,
    baseline_history["val_loss"],
    label="Baseline Validation Loss"
)

plt.plot(
    epochs_augmented,
    augmented_history["val_loss"],
    label="Augmented Validation Loss"
)

plt.title("Validation Loss Comparison")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.grid()

plt.show()


# =========================
# Final Results
# =========================

baseline_final_accuracy = baseline_history["val_accuracy"][-1]
augmented_final_accuracy = augmented_history["val_accuracy"][-1]

print("\n==============================")
print("MODEL COMPARISON")
print("==============================")

print(
    f"Baseline final validation accuracy  : "
    f"{baseline_final_accuracy * 100:.2f}%"
)

print(
    f"Augmented final validation accuracy : "
    f"{augmented_final_accuracy * 100:.2f}%"
)

difference = (
    augmented_final_accuracy -
    baseline_final_accuracy
)

print(
    f"Difference : {difference * 100:.2f}%"
)

if difference > 0:
    print("\nAugmented model performed better.")
else:
    print("\nBaseline model performed better.")