import numpy as np
import tensorflow as tf

layers = tf.keras.layers
models = tf.keras.models


# ==========================================
# 1. LOAD DATA
# ==========================================

X_train = np.load("data/X_train.npy")
X_test = np.load("data/X_test.npy")

y_train = np.load("data/y_train_semantic.npy")
y_test = np.load("data/y_test_semantic.npy")

print("X_train:", X_train.shape)
print("X_test :", X_test.shape)
print("y_train:", y_train.shape)
print("y_test :", y_test.shape)


# ==========================================
# 2. NUMBER OF CLASSES
# ==========================================

num_classes = len(
    np.unique(y_train)
)

print("Number of classes:", num_classes)


# ==========================================
# 3. BUILD IMPROVED CNN
# ==========================================

model = models.Sequential([

    layers.Input(shape=(32, 32, 3)),

    # Block 1
    layers.Conv2D(
        32,
        (3, 3),
        activation="relu",
        padding="same"
    ),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),

    # Block 2
    layers.Conv2D(
        64,
        (3, 3),
        activation="relu",
        padding="same"
    ),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),

    # Block 3
    layers.Conv2D(
        128,
        (3, 3),
        activation="relu",
        padding="same"
    ),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),

    # Classification
    layers.Flatten(),

    layers.Dense(
        128,
        activation="relu"
    ),

    layers.Dropout(0.5),

    layers.Dense(
        num_classes,
        activation="softmax"
    )
])


# ==========================================
# 4. DISPLAY MODEL
# ==========================================

model.summary()


# ==========================================
# 5. COMPILE
# ==========================================

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)


# ==========================================
# 6. TRAIN
# ==========================================

history = model.fit(
    X_train,
    y_train,
    epochs=10,
    batch_size=32,
    validation_split=0.2,
    shuffle=True
)


# ==========================================
# 7. TEST
# ==========================================

test_loss, test_accuracy = model.evaluate(
    X_test,
    y_test,
    verbose=1
)

print("\n==============================")
print("IMPROVED MODEL RESULTS")
print("==============================")

print(f"Test Loss     : {test_loss:.4f}")
print(f"Test Accuracy : {test_accuracy:.4f}")
print(f"Test Accuracy : {test_accuracy * 100:.2f}%")


# ==========================================
# 8. SAVE MODEL
# ==========================================

model.save(
    "traffic_sign_improved_model.keras"
)

print("\nModel saved as:")
print("traffic_sign_improved_model.keras")


# ==========================================
# 9. SAVE TRAINING HISTORY
# ==========================================

import json

with open(
    "data/improved_training_history.json",
    "w"
) as file:

    json.dump(
        history.history,
        file
    )

print("Training history saved.")