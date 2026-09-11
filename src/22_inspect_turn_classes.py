import numpy as np
import matplotlib.pyplot as plt


# Load data
X_train = np.load("data/X_train.npy")
y_train = np.load("data/y_train_semantic.npy")


TURN_LEFT = 46
TURN_RIGHT = 47


# Get indices
left_indices = np.where(y_train == TURN_LEFT)[0]
right_indices = np.where(y_train == TURN_RIGHT)[0]


# Take first 10 from each
left_indices = left_indices[:10]
right_indices = right_indices[:10]


# Create figure
fig, axes = plt.subplots(
    4,
    5,
    figsize=(12, 9)
)


# ------------------------------
# Turn Left
# ------------------------------

for i, idx in enumerate(left_indices):

    ax = axes[0, i % 5]

    ax.imshow(X_train[idx])

    ax.set_title("Turn Left")

    ax.axis("off")


    if i == 4:
        break


# ------------------------------
# More Turn Left
# ------------------------------

for i, idx in enumerate(left_indices[5:10]):

    ax = axes[1, i]

    ax.imshow(X_train[idx])

    ax.set_title("Turn Left")

    ax.axis("off")


# ------------------------------
# Turn Right
# ------------------------------

for i, idx in enumerate(right_indices):

    ax = axes[2, i % 5]

    ax.imshow(X_train[idx])

    ax.set_title("Turn Right")

    ax.axis("off")


    if i == 4:
        break


# ------------------------------
# More Turn Right
# ------------------------------

for i, idx in enumerate(right_indices[5:10]):

    ax = axes[3, i]

    ax.imshow(X_train[idx])

    ax.set_title("Turn Right")

    ax.axis("off")


plt.suptitle(
    "Actual Training Images: Turn Left vs Turn Right"
)

plt.tight_layout()

plt.savefig(
    "data/turn_left_right_inspection.png",
    dpi=200
)

plt.show()