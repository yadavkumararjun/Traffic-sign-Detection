import os
import cv2
import matplotlib.pyplot as plt
import pandas as pd


# ==========================================
# 1. Dataset paths
# ==========================================

dataset_path = "dataset/Indian-Traffic Sign-Dataset"
images_path = os.path.join(dataset_path, "Images")
csv_path = os.path.join(dataset_path, "traffic_sign.csv")


# ==========================================
# 2. Load class names
# ==========================================

df = pd.read_csv(csv_path)

class_names = dict(
    zip(df["ClassId"], df["Name"])
)


# ==========================================
# 3. Classes we want to inspect
# ==========================================

confused_pairs = [
    (47, 48),
    (23, 24),
    (2, 3),
    (42, 43),
    (25, 26),
    (36, 37),
    (49, 50)
]


# ==========================================
# 4. Display images
# ==========================================

for class1, class2 in confused_pairs:

    fig, axes = plt.subplots(
        2,
        5,
        figsize=(12, 5)
    )

    fig.suptitle(
        f"Class {class1}: {class_names.get(class1, 'Unknown')} "
        f"vs "
        f"Class {class2}: {class_names.get(class2, 'Unknown')}"
    )

    classes = [class1, class2]

    for row, class_id in enumerate(classes):

        folder = os.path.join(
            images_path,
            str(class_id)
        )

        if not os.path.exists(folder):

            print(
                f"Folder not found for class {class_id}"
            )

            continue

        image_files = os.listdir(folder)[:5]

        for col, image_file in enumerate(image_files):

            image_path = os.path.join(
                folder,
                image_file
            )

            image = cv2.imread(image_path)

            if image is None:
                continue

            image = cv2.cvtColor(
                image,
                cv2.COLOR_BGR2RGB
            )

            axes[row, col].imshow(image)

            axes[row, col].axis("off")

        axes[row, 0].set_ylabel(
            f"Class {class_id}",
            fontsize=12
        )

    plt.tight_layout()
    plt.show()