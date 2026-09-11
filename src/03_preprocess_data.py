import os 
import cv2
import numpy as np 
import pandas as pd 
from sklearn.model_selection import train_test_split

# 1.Dataset paths

DATASET_PATH = "dataset/Indian-Traffic Sign-Dataset"
IMAGE_PATH = os.path.join(DATASET_PATH , "Images")
CSV_PATH = os.path.join(DATASET_PATH , "traffic_sign.csv") ;


# 2. Settings 

IMAGE_SIZE = 32
TEST_SIZE = 0.20 
RANDOM_STATE = 42 

# 3. Load csv 

df = pd.read_csv(CSV_PATH )

class_folders =[
    folder
    for folder in os.listdir(IMAGE_PATH)
    if os.path.isdir(os.path.join(IMAGE_PATH , folder))
]
class_ids = sorted(int(folder) for folder in class_folders)

print("\n===========================")
print("          DATA PREPROCESSING")
print("\n===========================")
print(f"Classes found :{len(class_ids)}")

# 5. Load images and label 

images = []
labels = []
print("\n Loading images...")

for class_id in class_ids:
    class_path = os.path.join(
        IMAGE_PATH,
        str(class_id)
    )
    image_files = [
        file 
        for file in os.listdir(class_path)
        if file.lower().endswith(
            (".png" , ".jpg" , ".jpeg")
        )
    ]
    for image_file in image_files:
        image_path = os.path.join(
            class_path ,
            image_file
        )

        # Read image 
        image = cv2.imread(image_path)

        # skip corrupted images 
        if image is None:
            print(f"Warning:could not read {image_path}")
            continue 



        # convert BGR to RGB 
        image = cv2.cvtColor(
            image ,
            cv2.COLOR_BGR2RGB 
        )
        image = cv2.resize(
            image,
            (IMAGE_SIZE, IMAGE_SIZE)
        )

        # Normalize pixel values
        image = image.astype(np.float32) / 255.0

        images.append(image)
        labels.append(class_id)


# ============================================
# 6. Convert to NumPy arrays
# ============================================

X = np.array(images)
y = np.array(labels)


print("\nData loaded successfully!")

print(f"X shape: {X.shape}")
print(f"y shape: {y.shape}")


# ============================================
# 7. Train/Test split
# ============================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
    stratify=y
)


# ============================================
# 8. Final information
# ============================================

print("\n================================")
print("          DATA SPLIT")
print("================================")

print(f"Training images : {len(X_train)}")
print(f"Testing images  : {len(X_test)}")

print("\nImage shape:")
print(X_train.shape[1:])

print("\nPixel range:")
print(f"Minimum: {X_train.min()}")
print(f"Maximum: {X_train.max()}")

print("\n================================")
print("    PREPROCESSING COMPLETE")
print("================================")

# ============================================
# 9. Save processed data
# ============================================

DATA_DIR = "data"

os.makedirs(DATA_DIR, exist_ok=True)

np.save(
    os.path.join(DATA_DIR, "X_train.npy"),
    X_train
)

np.save(
    os.path.join(DATA_DIR, "X_test.npy"),
    X_test
)

np.save(
    os.path.join(DATA_DIR, "y_train.npy"),
    y_train
)

np.save(
    os.path.join(DATA_DIR, "y_test.npy"),
    y_test
)

print("\nProcessed data saved successfully!")
print(f"Saved inside: {DATA_DIR}/")