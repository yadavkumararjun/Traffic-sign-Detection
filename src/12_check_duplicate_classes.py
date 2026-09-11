import pandas as pd

# Load CSV
df = pd.read_csv(
    "dataset/Indian-Traffic Sign-Dataset/traffic_sign.csv"
)

# Find names that appear more than once
duplicate_names = df[
    df["Name"].duplicated(keep=False)
].sort_values("Name")

print("\n==============================")
print("DUPLICATE CLASS NAMES")
print("==============================\n")

print(
    duplicate_names.to_string(index=False)
)