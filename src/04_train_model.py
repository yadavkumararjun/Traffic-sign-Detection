import numpy as np
import tensorflow as tf

layers = tf.keras.layers
models = tf.keras.models
import json

print("================================")
print("      TRAFFIC SIGN CNN")
print("================================")

# 1. LOAD PREPROCESSED DATA 
print("\nLoading data...")

X_train = np.load("data/X_train.npy")
X_test = np.load("data/X_test.npy")

y_train = np.load("data/y_train.npy")
y_test = np.load("data/y_test.npy")

print("Data loaded successfully!")

print(f"X_train shape: {X_train.shape}")
print(f"X_test shape: {X_test.shape}")
print(f"y_train shape: {y_train.shape}")    
print(f"y_test shape: {y_test.shape}")

# 2. REMAP class labels 
print("\nRemapping class labels...")
existing_classes = [
    class_id
    for class_id in range(59)
    if class_id != 39 
]

# Create mapping:
# Original ID → New ID

label_mapping = {
    original_id: new_id 
    for  new_id , original_id in enumerate(existing_classes)
}
# print("Original → New label mapping:")
# print(label_mapping)

# Apply mapping to training and testing labels

y_train = np.array([
    label_mapping[label]
    for label in y_train
])

y_test = np.array([
    label_mapping[label]
    for label in y_test
])

print("\nLabels remapped successfully!")

# print("Minimum label:", y_train.min())
# print("Maximum label:", y_train.max())

# print("\nUnique training labels:")
# print(np.unique(y_train))

# print("\nNumber of classes:", len(np.unique(y_train)))

# ============================================
# 3. BUILD CNN MODEL
# ============================================

print("\nBuilding CNN model...")

model = models.Sequential([

    # Input image: 32 × 32 × 3
    layers.Input(shape=(32, 32, 3)),

    # First convolution layer
    layers.Conv2D(
        32,
        (3, 3),
        activation="relu"
    ),

    # Reduce spatial dimensions
    layers.MaxPooling2D(
        (2, 2)
    ),

    # Second convolution layer
    layers.Conv2D(
        64,
        (3, 3),
        activation="relu"
    ),

    # Reduce spatial dimensions again
    layers.MaxPooling2D(
        (2, 2)
    ),

    # Convert feature maps into one vector
    layers.Flatten(),

    # Fully connected layer
    layers.Dense(
        128,
        activation="relu"
    ),

    # Reduce overfitting
    layers.Dropout(0.5),

    # 58 traffic-sign classes
    layers.Dense(
        58,
        activation="softmax"
    )
])
print("\nBuilding successfull!")
# ============================================
# 4. COMPILE MODEL
# ============================================
print("Initializing complietion")
model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

print("\nModel compiled successfully!")

# ============================================
# 5. MODEL SUMMARY
# ============================================

print("\n================================")
print("          MODEL SUMMARY")
print("================================")

model.summary()

# ============================================
# 6. TRAIN THE MODEL
# ============================================
print("Traning started...")
history = model.fit(
    X_train ,
    y_train,
    epochs = 10 ,
    batch_size = 32 ,
    validation_split = 0.2 ,
    shuffle = True

)

print("Model trained succesfully!") ;

print("\nSaving training history...")

with open("data/training_history.json", "w") as f:
    json.dump(history.history, f)

print("Training history saved successfully!")
# ============================================
# 7. EVALUATE MODEL ON TEST DATA
# ============================================

print("\n================================")
print("          MODEL TESTING")
print("================================")

test_loss, test_accuracy = model.evaluate(
    X_test,
    y_test,
    verbose=1
)

print("\nTest Loss:", test_loss)
print("Test Accuracy:", test_accuracy)

# ============================================
# 8. SAVE TRAINED MODEL
# ============================================

print("\n================================")
print("        SAVING MODEL")
print("================================")

model.save("traffic_sign_model.keras")

print("Model saved successfully!")