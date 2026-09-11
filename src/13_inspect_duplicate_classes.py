import os
import cv2
import matplotlib.pyplot as plt
import pandas as pd


# ==========================================
# 1. Paths
# ==========================================

dataset_path = "dataset/Indian-Traffic Sign-Dataset"
images_path = os.path.join(dataset_path, "Images")
csv_path = os.path.join(dataset_path, "traffic_sign.csv")


# ==========================================
# 2. Load CSV
# ==========================================

df = pd.read_csv(csv_path)

class_names = dict(
    zip(df["ClassId"], df["Name"])
)


# ==========================================
# 3. Duplicate class groups
# ==========================================

duplicate_groups = [
    [2, 3],
    [36, 37],
    [42, 43],
    [47, 48, 49, 50]
]


# ==========================================
# 4. Display images
# ==========================================

for group in duplicate_groups:

    print("\n===================================")
    print("GROUP:", group)

    for class_id in group:
        print(
            f"Class {class_id}: "
            f"{class_names.get(class_id, 'Unknown')}"
        )

    print("===================================")

    fig, axes = plt.subplots(
        len(group),
        5,
        figsize=(12, 3 * len(group))
    )

    # Handle groups with only one row
    if len(group) == 1:
        axes = [axes]

    fig.suptitle(
        "Duplicate Name Classes",
        fontsize=16
    )

    for row, class_id in enumerate(group):

        folder = os.path.join(
            images_path,
            str(class_id)
        )

        if not os.path.exists(folder):

            print(
                f"Class {class_id}: folder missing"
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

            axes[row][col].imshow(image)
            axes[row][col].axis("off")

        axes[row][0].set_ylabel(
            f"Class {class_id}\n"
            f"{class_names[class_id]}",
            fontsize=10
        )

    plt.tight_layout()
    plt.show()