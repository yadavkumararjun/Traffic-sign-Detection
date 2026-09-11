import json
import matplotlib.pyplot as plt

print("================================")
print("     MODEL TRAINING RESULTS")
print("================================")

# Load training history
print("\nLoading training history...")

with open("data/training_history.json", "r") as file:
    history = json.load(file)

print("Training history loaded successfully!")

# Extract metrics
accuracy = history["accuracy"]
val_accuracy = history["val_accuracy"]

loss = history["loss"]
val_loss = history["val_loss"]

# Number of epochs
epochs = range(1, len(accuracy) + 1)


# ============================================
# 1. TRAINING VS VALIDATION ACCURACY
# ============================================

plt.figure(figsize=(8, 5))

plt.plot(
    epochs,
    accuracy,
    label="Training Accuracy"
)

plt.plot(
    epochs,
    val_accuracy,
    label="Validation Accuracy"
)

plt.title("Training vs Validation Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")

plt.legend()
plt.grid()

plt.show()


# ============================================
# 2. TRAINING VS VALIDATION LOSS
# ============================================

plt.figure(figsize=(8, 5))

plt.plot(
    epochs,
    loss,
    label="Training Loss"
)

plt.plot(
    epochs,
    val_loss,
    label="Validation Loss"
)

plt.title("Training vs Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")

plt.legend()
plt.grid()

plt.show()


print("\n================================")
print("     VISUALIZATION COMPLETE")
print("================================")