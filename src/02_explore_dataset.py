import os
import cv2
import pandas as pd

# ============================================
# 1. Dataset paths
# ============================================

DATASET_PATH = "dataset/Indian-Traffic Sign-Dataset"
IMAGE_PATH = os.path.join(DATASET_PATH, "Images")
CSV_PATH = os.path.join(DATASET_PATH, "traffic_sign.csv")


# ============================================
# 2. Load CSV
# ============================================

df = pd.read_csv(CSV_PATH)

print("\n================================")
print("     INDIAN TRAFFIC SIGN DATASET")
print("================================")

print(f"\nCSV columns: {list(df.columns)}")
print(f"Classes listed in CSV: {len(df)}")


# ============================================
# 3. Find actual class folders
# ============================================

class_folders = [
    folder
    for folder in os.listdir(IMAGE_PATH)
    if os.path.isdir(os.path.join(IMAGE_PATH, folder))
]

# Convert folder names to integers
class_ids = sorted(int(folder) for folder in class_folders)

print(f"Classes with image folders: {len(class_ids)}")

# Find missing classes
csv_class_ids = set(df["ClassId"])
image_class_ids = set(class_ids)

missing_classes = sorted(csv_class_ids - image_class_ids)

if missing_classes:
    print(f"Missing image folders: {missing_classes}")
else:
    print("Missing image folders: None")


# ============================================
# 4. Count images in each class
# ============================================

print("\n================================")
print("        IMAGES PER CLASS")
print("================================")

total_images = 0

for class_id in class_ids:

    class_path = os.path.join(IMAGE_PATH, str(class_id))

    images = [
        file
        for file in os.listdir(class_path)
        if file.lower().endswith((".png", ".jpg", ".jpeg"))
    ]

    count = len(images)
    total_images += count

    # Get class name from CSV
    class_info = df[df["ClassId"] == class_id]

    if not class_info.empty:
        class_name = class_info["Name"].iloc[0]
    else:
        class_name = "Unknown"

    print(
        f"Class {class_id:2d}: "
        f"{count:4d} images | "
        f"{class_name}"
    )


# ============================================
# 5. Test one image
# ============================================

print("\n================================")
print("          IMAGE TEST")
print("================================")

first_class = class_ids[0]
first_class_path = os.path.join(
    IMAGE_PATH,
    str(first_class)
)

first_image = next(
    file
    for file in os.listdir(first_class_path)
    if file.lower().endswith((".png", ".jpg", ".jpeg"))
)

image_path = os.path.join(
    first_class_path,
    first_image
)

image = cv2.imread(image_path)

if image is not None:

    print(f"Image: {first_image}")
    print(f"Shape: {image.shape}")
    print(f"Height: {image.shape[0]}")
    print(f"Width: {image.shape[1]}")
    print(f"Channels: {image.shape[2]}")

else:

    print("ERROR: Could not read image.")


# ============================================
# 6. Final summary
# ============================================

print("\n================================")
print("             SUMMARY")
print("================================")

print(f"Classes listed in CSV       : {len(df)}")
print(f"Classes with image folders  : {len(class_ids)}")
print(f"Total images                : {total_images}")

print("================================")
print("       DATASET EXPLORATION DONE")
print("================================")