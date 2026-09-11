import numpy as np
import tensorflow as tf

layers = tf.keras.layers
models = tf.keras.modelsc


# ==========================================
# 1. LOAD DATA
# ==========================================

X_train = np.load("data/X_train.npy")
X_test = np.load("data/X_test.npy")

y_train = np.load("data/y_train_semantic.npy")
y_test = np.load("data/y_test_semantic.npy")


print("Original training data:")
print("X_train:", X_train.shape)
print("y_train:", y_train.shape)


# ==========================================
# 2. TURN LEFT / TURN RIGHT CLASS IDs
# ==========================================

TURN_LEFT = 46
TURN_RIGHT = 47


# ==========================================
# 3. FIND TURN LEFT / RIGHT IMAGES
# ==========================================

left_indices = np.where(
    y_train == TURN_LEFT
)[0]

right_indices = np.where(
    y_train == TURN_RIGHT
)[0]


print("\nOriginal class counts:")
print("Turn left :", len(left_indices))
print("Turn right:", len(right_indices))


# ==========================================
# 4. HORIZONTAL FLIP
# ==========================================

# Flip Turn Left images
flipped_left = np.flip(
    X_train[left_indices],
    axis=2
)

# They become Turn Right
flipped_left_labels = np.full(
    len(flipped_left),
    TURN_RIGHT
)


# Flip Turn Right images
flipped_right = np.flip(
    X_train[right_indices],
    axis=2
)

# They become Turn Left
flipped_right_labels = np.full(
    len(flipped_right),
    TURN_LEFT
)


# ==========================================
# 5. ADD AUGMENTED DATA
# ==========================================

X_train_targeted = np.concatenate(
    [
        X_train,
        flipped_left,
        flipped_right
    ],
    axis=0
)

y_train_targeted = np.concatenate(
    [
        y_train,
        flipped_left_labels,
        flipped_right_labels
    ],
    axis=0
)


print("\nAfter targeted augmentation:")
print("X_train:", X_train_targeted.shape)
print("y_train:", y_train_targeted.shape)


# ==========================================
# 6. SHUFFLE TRAINING DATA
# ==========================================

indices = np.random.permutation(
    len(X_train_targeted)
)

X_train_targeted = X_train_targeted[
    indices
]

y_train_targeted = y_train_targeted[
    indices
]


# ==========================================
# 7. BUILD IMPROVED CNN
# ==========================================

num_classes = len(
    np.unique(y_train)
)


model = models.Sequential([

    layers.Input(
        shape=(32, 32, 3)
    ),

    # ------------------------------
    # Block 1
    # ------------------------------

    layers.Conv2D(
        32,
        (3, 3),
        activation="relu",
        padding="same"
    ),

    layers.BatchNormalization(),

    layers.MaxPooling2D(
        (2, 2)
    ),


    # ------------------------------
    # Block 2
    # ------------------------------

    layers.Conv2D(
        64,
        (3, 3),
        activation="relu",
        padding="same"
    ),

    layers.BatchNormalization(),

    layers.MaxPooling2D(
        (2, 2)
    ),


    # ------------------------------
    # Block 3
    # ------------------------------

    layers.Conv2D(
        128,
        (3, 3),
        activation="relu",
        padding="same"
    ),

    layers.BatchNormalization(),

    layers.MaxPooling2D(
        (2, 2)
    ),


    # ------------------------------
    # Classification
    # ------------------------------

    layers.Flatten(),

    layers.Dense(
        128,
        activation="relu"
    ),

    layers.Dropout(
        0.5
    ),

    layers.Dense(
        num_classes,
        activation="softmax"
    )
])


# ==========================================
# 8. MODEL SUMMARY
# ==========================================

model.summary()


# ==========================================
# 9. COMPILE
# ==========================================

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)


# ==========================================
# 10. TRAIN
# ==========================================

history = model.fit(

    X_train_targeted,

    y_train_targeted,

    epochs=10,

    batch_size=32,

    validation_split=0.2,

    shuffle=True
)


# ==========================================
# 11. TEST
# ==========================================

test_loss, test_accuracy = model.evaluate(
    X_test,
    y_test,
    verbose=1
)


print("\n==============================")
print("TARGETED MODEL RESULTS")
print("==============================")

print(
    f"Test Loss     : {test_loss:.4f}"
)

print(
    f"Test Accuracy : {test_accuracy:.4f}"
)

print(
    f"Test Accuracy : {test_accuracy * 100:.2f}%"
)


# ==========================================
# 12. SAVE MODEL
# ==========================================

model.save(
    "traffic_sign_targeted_model.keras"
)

print(
    "\nModel saved as:"
)

print(
    "traffic_sign_targeted_model.keras"
)


# ==========================================
# 13. SAVE HISTORY
# ==========================================

import json

with open(
    "data/targeted_training_history.json",
    "w"
) as file:

    json.dump(
        history.history,
        file
    )


print(
    "Training history saved."
)