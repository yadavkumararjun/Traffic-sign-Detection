import numpy as np
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import tensorflow as tf

# Access Keras through TensorFlow to avoid unresolved tensorflow.keras imports
# in environments where TensorFlow's package source is not exposed to Pylance.
layers = tf.keras.layers
models = tf.keras.models
import json

# =========================
# 1. Load preprocessed data
# =========================

X_train = np.load("data/X_train.npy")
X_test = np.load("data/X_test.npy")

y_train = np.load("data/y_train.npy")
y_test = np.load("data/y_test.npy")

print("X_train:", X_train.shape)
print("X_test :", X_test.shape)
print("y_train:", y_train.shape)
print("y_test :", y_test.shape)


# =========================
# 2. Remap labels
# =========================

# Class 39 does not have images
existing_classes = [
    class_id for class_id in range(59)
    if class_id != 39
]

label_mapping = {
    original_id: new_id
    for new_id, original_id in enumerate(existing_classes)
}

y_train = np.array([
    label_mapping[label]
    for label in y_train
])

y_test = np.array([
    label_mapping[label]
    for label in y_test
])

print("Number of classes:", len(existing_classes))


# =========================
# 3. Data Augmentation
# =========================

data_augmentation = tf.keras.Sequential([
    layers.RandomRotation(0.02),
    layers.RandomTranslation(
        height_factor=0.05,
        width_factor=0.05
    ),
    layers.RandomZoom(0.1),
], name="data_augmentation")


# =========================
# 4. Build CNN
# =========================

model = models.Sequential([

    layers.Input(shape=(32, 32, 3)),

    # Augmentation
    data_augmentation,

    # CNN Layer 1
    layers.Conv2D(
        32,
        (3, 3),
        activation="relu"
    ),

    layers.MaxPooling2D((2, 2)),

    # CNN Layer 2
    layers.Conv2D(
        64,
        (3, 3),
        activation="relu"
    ),

    layers.MaxPooling2D((2, 2)),

    # Convert feature maps to vector
    layers.Flatten(),

    # Fully connected layer
    layers.Dense(
        128,
        activation="relu"
    ),

    # Prevent overfitting
    layers.Dropout(0.5),

    # 58 classes
    layers.Dense(
        58,
        activation="softmax"
    )
])


# =========================
# 5. Compile
# =========================

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)


# =========================
# 6. Model summary
# =========================

model.summary()


# =========================
# 7. Train
# =========================

history = model.fit(
    X_train,
    y_train,

    epochs=10,
    batch_size=32,

    validation_split=0.2,

    shuffle=True
)


# =========================
# 8. Save model
# =========================

model.save(
    "traffic_sign_model_augmented.keras"
)

print("\nAugmented model saved successfully!")


# =========================
# 9. Save training history
# =========================

with open(
    "data/augmented_training_history.json",
    "w"
) as f:

    json.dump(
        history.history,
        f
    )

print("Training history saved successfully!")


# =========================
# 10. Evaluate on TEST data
# =========================

test_loss, test_accuracy = model.evaluate(
    X_test,
    y_test,
    verbose=1
)

print("\n==============================")
print("Augmented Model Test Results")
print("==============================")

print("Test Loss     :", test_loss)
print("Test Accuracy :", test_accuracy)
print("Test Accuracy % :", test_accuracy * 100)