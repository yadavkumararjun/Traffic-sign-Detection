import os
import json
import numpy as np
import tensorflow as tf

layers = tf.keras.layers
models = tf.keras.models


# ==========================================
# 1. SETTINGS
# ==========================================

IMAGE_SIZE = 32
BATCH_SIZE = 32
EPOCHS = 20

MODEL_PATH = "models/traffic_sign_robust_model.keras"
HISTORY_PATH = "data/robust_training_history.json"


# ==========================================
# 2. LOAD DATA
# ==========================================

print("\nLoading training data...")

X_train = np.load("data/X_train.npy")
X_test = np.load("data/X_test.npy")

y_train = np.load("data/y_train_semantic.npy")
y_test = np.load("data/y_test_semantic.npy")


print("X_train:", X_train.shape)
print("y_train:", y_train.shape)
print("X_test :", X_test.shape)
print("y_test :", y_test.shape)


# ==========================================
# 3. NUMBER OF CLASSES
# ==========================================

num_classes = len(
    np.unique(y_train)
)

print("\nNumber of classes:", num_classes)


# ==========================================
# 4. DATA AUGMENTATION
# ==========================================

augmentation = tf.keras.Sequential([

    layers.RandomRotation(
        factor=0.08
    ),

    layers.RandomZoom(
        height_factor=(-0.10, 0.10),
        width_factor=(-0.10, 0.10)
    ),

    layers.RandomTranslation(
        height_factor=0.08,
        width_factor=0.08
    ),

    layers.RandomContrast(
        factor=0.15
    )

], name="traffic_sign_augmentation")


# ==========================================
# 5. BUILD CNN
# ==========================================

model = models.Sequential([

    layers.Input(
        shape=(IMAGE_SIZE, IMAGE_SIZE, 3)
    ),

    # Augmentation
    augmentation,


    # ======================================
    # Block 1
    # ======================================

    layers.Conv2D(
        32,
        (3, 3),
        padding="same",
        activation="relu"
    ),

    layers.BatchNormalization(),

    layers.MaxPooling2D(
        (2, 2)
    ),


    # ======================================
    # Block 2
    # ======================================

    layers.Conv2D(
        64,
        (3, 3),
        padding="same",
        activation="relu"
    ),

    layers.BatchNormalization(),

    layers.MaxPooling2D(
        (2, 2)
    ),


    # ======================================
    # Block 3
    # ======================================

    layers.Conv2D(
        128,
        (3, 3),
        padding="same",
        activation="relu"
    ),

    layers.BatchNormalization(),

    layers.MaxPooling2D(
        (2, 2)
    ),


    # ======================================
    # Block 4
    # ======================================

    layers.Conv2D(
        256,
        (3, 3),
        padding="same",
        activation="relu"
    ),

    layers.BatchNormalization(),

    layers.MaxPooling2D(
        (2, 2)
    ),


    # ======================================
    # CLASSIFIER
    # ======================================

    layers.GlobalAveragePooling2D(),

    layers.Dense(
        256,
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
# 6. MODEL SUMMARY
# ==========================================

print("\n========================================")
print("MODEL ARCHITECTURE")
print("========================================")

model.summary()


# ==========================================
# 7. COMPILE
# ==========================================

model.compile(

    optimizer=tf.keras.optimizers.Adam(
        learning_rate=0.001
    ),

    loss="sparse_categorical_crossentropy",

    metrics=["accuracy"]
)


# ==========================================
# 8. CALLBACKS
# ==========================================

callbacks = [

    tf.keras.callbacks.EarlyStopping(

        monitor="val_accuracy",

        patience=5,

        restore_best_weights=True
    ),

    tf.keras.callbacks.ReduceLROnPlateau(

        monitor="val_loss",

        factor=0.5,

        patience=2,

        min_lr=0.00001
    ),

    tf.keras.callbacks.ModelCheckpoint(

        MODEL_PATH,

        monitor="val_accuracy",

        save_best_only=True
    )
]


# ==========================================
# 9. TRAIN
# ==========================================

print("\n========================================")
print("STARTING TRAINING")
print("========================================")

history = model.fit(

    X_train,

    y_train,

    validation_split=0.20,

    epochs=EPOCHS,

    batch_size=BATCH_SIZE,

    shuffle=True,

    callbacks=callbacks
)


# ==========================================
# 10. EVALUATE
# ==========================================

print("\n========================================")
print("FINAL TEST EVALUATION")
print("========================================")

test_loss, test_accuracy = model.evaluate(

    X_test,

    y_test,

    verbose=1
)


print("\nTest Loss:")
print(f"{test_loss:.4f}")

print("\nTest Accuracy:")
print(f"{test_accuracy:.4f}")

print(
    f"\nTest Accuracy: {test_accuracy * 100:.2f}%"
)


# ==========================================
# 11. SAVE MODEL
# ==========================================

os.makedirs(
    "models",
    exist_ok=True
)

model.save(
    MODEL_PATH
)

print("\nModel saved:")
print(MODEL_PATH)


# ==========================================
# 12. SAVE HISTORY
# ==========================================

os.makedirs(
    "data",
    exist_ok=True
)

with open(
    HISTORY_PATH,
    "w"
) as file:

    json.dump(
        history.history,
        file
    )


print("\nTraining history saved:")
print(HISTORY_PATH)


print("\n========================================")
print("TRAINING COMPLETE")
print("========================================")