
import json
import matplotlib.pyplot as plt

history_path = "data/robust_training_history.json"

with open(history_path, "r") as file:
    history = json.load(file)

# Support either a direct Keras history dictionary
# or a dictionary containing a "history" entry.
if "history" in history and isinstance(history["history"], dict):
    history = history["history"]

print("Available history keys:", list(history.keys()))

# Find the available accuracy and loss metrics.
acc_key = next(
    (k for k in ("accuracy", "sparse_categorical_accuracy")
     if k in history),
    None
)
val_acc_key = next(
    (k for k in ("val_accuracy", "val_sparse_categorical_accuracy")
     if k in history),
    None
)

if not all(k in history for k in
           ("loss", "val_loss")) or acc_key is None or val_acc_key is None:
    raise ValueError(
        "Training history must contain accuracy, val_accuracy, "
        "loss and val_loss. Check the printed keys."
    )

epochs = range(1, len(history["loss"]) + 1)

fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

# Accuracy graph
axes[0].plot(epochs, history[acc_key], label="Training accuracy")
axes[0].plot(epochs, history[val_acc_key],
             label="Validation accuracy")
axes[0].set_title("Accuracy During Training")
axes[0].set_xlabel("Epoch")
axes[0].set_ylabel("Accuracy")
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Loss graph
axes[1].plot(epochs, history["loss"], label="Training loss")
axes[1].plot(epochs, history["val_loss"], label="Validation loss")
axes[1].set_title("Loss During Training")
axes[1].set_xlabel("Epoch")
axes[1].set_ylabel("Loss")
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("fig3_training_curves.png",
            dpi=300, bbox_inches="tight")
plt.show()

print("Saved: fig3_training_curves.png")
