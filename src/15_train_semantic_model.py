import numpy as np
import tensorflow as tf
import json

layers = tf.keras.layers
models = tf.keras.models


# ==========================================
# 1. Load image data
# ==========================================

X_train = np.load("data/X_train.npy")
X_test = np.load("data/X_test.npy")

# Load semantic labels
y_train = np.load("data/y_train_semantic.npy")
y_test = np.load("data/y_test_semantic.npy")


print("X_train:", X_train.shape)
print("X_test :", X_test.shape)

print("y_train:", y_train.shape)
print("y_test :", y_test.shape)


# ==========================================
# 2. Number of semantic classes
# ==========================================

num_classes = len(
    np.unique(y_train)
)

print(
    "Number of semantic classes:",
    num_classes
)


# ==========================================
# 3. Build CNN
# ==========================================

model = models.Sequential([

    layers.Input(
        shape=(32, 32, 3)
    ),

    # CNN Layer 1
    layers.Conv2D(
        32,
        (3, 3),
        activation="relu"
    ),

    layers.MaxPooling2D(
        (2, 2)
    ),

    # CNN Layer 2
    layers.Conv2D(
        64,
        (3, 3),
        activation="relu"
    ),

    layers.MaxPooling2D(
        (2, 2)
    ),

    # Flatten feature maps
    layers.Flatten(),

    # Fully connected layer
    layers.Dense(
        128,
        activation="relu"
    ),

    # Reduce overfitting
    layers.Dropout(0.5),

    # 52 semantic classes
    layers.Dense(
        num_classes,
        activation="softmax"
    )
])


# ==========================================
# 4. Compile
# ==========================================

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)


# ==========================================
# 5. Display model
# ==========================================

model.summary()


# ==========================================
# 6. Train
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
# 7. Save model
# ==========================================

model.save(
    "traffic_sign_semantic_model.keras"
)

print(
    "\nSemantic model saved successfully!"
)


# ==========================================
# 8. Save training history
# ==========================================

with open(
    "data/semantic_training_history.json",
    "w"
) as file:

    json.dump(
        history.history,
        file
    )

print(
    "Semantic training history saved successfully!"
)


# ==========================================
# 9. Evaluate on untouched test data
# ==========================================

test_loss, test_accuracy = model.evaluate(
    X_test,
    y_test,
    verbose=1
)


# ==========================================
# 10. Results
# ==========================================

print("\n==============================")
print("SEMANTIC MODEL RESULTS")
print("==============================")

print(
    "Test Loss      :",
    test_loss
)

print(
    "Test Accuracy  :",
    test_accuracy
)

print(
    "Test Accuracy %:",
    test_accuracy * 100
)