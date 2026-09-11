import numpy as np
import pandas as pd


# ==========================================
# 1. Load original labels
# ==========================================

y_train = np.load("data/y_train.npy")
y_test = np.load("data/y_test.npy")


# ==========================================
# 2. Load class information
# ==========================================

df = pd.read_csv(
    "dataset/Indian-Traffic Sign-Dataset/traffic_sign.csv"
)


# ==========================================
# 3. Create ClassId → Name mapping
# ==========================================

class_to_name = dict(
    zip(
        df["ClassId"],
        df["Name"]
    )
)


# ==========================================
# 4. Get names actually present in dataset
# ==========================================

all_labels = np.concatenate(
    [y_train, y_test]
)

existing_names = sorted(
    set(
        class_to_name[label]
        for label in all_labels
    )
)


# ==========================================
# 5. Create semantic label mapping
# ==========================================

name_to_new_id = {
    name: new_id
    for new_id, name in enumerate(existing_names)
}


# ==========================================
# 6. Convert original labels
# ==========================================

y_train_semantic = np.array([
    name_to_new_id[class_to_name[label]]
    for label in y_train
])

y_test_semantic = np.array([
    name_to_new_id[class_to_name[label]]
    for label in y_test
])


# ==========================================
# 7. Save new labels
# ==========================================

np.save(
    "data/y_train_semantic.npy",
    y_train_semantic
)

np.save(
    "data/y_test_semantic.npy",
    y_test_semantic
)


# ==========================================
# 8. Save class names
# ==========================================

with open(
    "data/semantic_class_names.txt",
    "w"
) as file:

    for class_id, name in enumerate(existing_names):

        file.write(
            f"{class_id},{name}\n"
        )


# ==========================================
# 9. Display results
# ==========================================

print("\n==============================")
print("SEMANTIC LABEL MAPPING")
print("==============================")

print(
    "Original ClassIds : 58"
)

print(
    "Semantic Classes  :",
    len(existing_names)
)

print("\nClass mapping:\n")

for class_id, name in enumerate(existing_names):

    print(
        f"{class_id:2d} -> {name}"
    )


print("\n==============================")
print("LABEL SHAPES")
print("==============================")

print(
    "y_train:",
    y_train_semantic.shape
)

print(
    "y_test :",
    y_test_semantic.shape
)

print(
    "\nSemantic labels saved successfully!"
)