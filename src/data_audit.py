import os
import pandas as pd
from PIL import Image

# =========================
# PATHS
# =========================

BASE_DIR = r"D:\diabetic"
DATASET_DIR = os.path.join(BASE_DIR, "dataset")

TRAIN_DIR = os.path.join(DATASET_DIR, "train_images")
TEST_DIR = os.path.join(DATASET_DIR, "test_images")

TRAIN_CSV = os.path.join(DATASET_DIR, "train.csv")
TEST_CSV = os.path.join(DATASET_DIR, "test.csv")

# =========================
# LOAD CSV FILES
# =========================

print("\n========== CSV AUDIT ==========\n")

train_df = pd.read_csv(TRAIN_CSV)
test_df = pd.read_csv(TEST_CSV)

print("Train CSV shape:", train_df.shape)
print("Test CSV shape :", test_df.shape)

print("\nTrain columns:")
print(train_df.columns.tolist())

print("\nTest columns:")
print(test_df.columns.tolist())

# =========================
# MISSING VALUES
# =========================

print("\n========== MISSING VALUES ==========\n")

print("Train:")
print(train_df.isnull().sum())

print("\nTest:")
print(test_df.isnull().sum())

# =========================
# DUPLICATE IDS
# =========================

print("\n========== DUPLICATE IDS ==========\n")

print(
    "Duplicate train IDs:",
    train_df["id_code"].duplicated().sum()
)

print(
    "Duplicate test IDs:",
    test_df["id_code"].duplicated().sum()
)

# =========================
# DIAGNOSIS CHECK
# =========================

print("\n========== DIAGNOSIS ==========\n")

print(train_df["diagnosis"].value_counts().sort_index())

print("\nUnique diagnosis values:")
print(sorted(train_df["diagnosis"].unique()))

# =========================
# IMAGE / CSV MATCHING
# =========================

print("\n========== IMAGE MATCHING ==========\n")

train_images = set(
    os.path.splitext(f)[0]
    for f in os.listdir(TRAIN_DIR)
    if f.lower().endswith((".png", ".jpg", ".jpeg"))
)

test_images = set(
    os.path.splitext(f)[0]
    for f in os.listdir(TEST_DIR)
    if f.lower().endswith((".png", ".jpg", ".jpeg"))
)

train_csv_ids = set(train_df["id_code"].astype(str))
test_csv_ids = set(test_df["id_code"].astype(str))

missing_train_images = train_csv_ids - train_images
extra_train_images = train_images - train_csv_ids

missing_test_images = test_csv_ids - test_images
extra_test_images = test_images - test_csv_ids

print("Train images found:", len(train_images))
print("Train CSV IDs     :", len(train_csv_ids))

print("Missing train images:", len(missing_train_images))
print("Extra train images  :", len(extra_train_images))

print("\nTest images found:", len(test_images))
print("Test CSV IDs     :", len(test_csv_ids))

print("Missing test images:", len(missing_test_images))
print("Extra test images  :", len(extra_test_images))

# =========================
# IMAGE INTEGRITY
# =========================

print("\n========== IMAGE INTEGRITY ==========\n")

corrupted = []
dimensions = {}
channels = {}

for filename in os.listdir(TRAIN_DIR):

    if not filename.lower().endswith((".png", ".jpg", ".jpeg")):
        continue

    path = os.path.join(TRAIN_DIR, filename)

    try:
        with Image.open(path) as img:

            img.verify()

        # Reopen after verify
        with Image.open(path) as img:

            size = img.size
            mode = img.mode

            dimensions[size] = dimensions.get(size, 0) + 1
            channels[mode] = channels.get(mode, 0) + 1

    except Exception as e:
        corrupted.append((filename, str(e)))

print("Corrupted images:", len(corrupted))

print("\nImage dimensions:")
for dimension, count in sorted(
    dimensions.items(),
    key=lambda x: x[1],
    reverse=True
)[:15]:
    print(dimension, "->", count)

print("\nImage modes:")
for mode, count in channels.items():
    print(mode, "->", count)

# =========================
# CLASS DISTRIBUTION
# =========================

print("\n========== CLASS DISTRIBUTION ==========\n")

class_counts = train_df["diagnosis"].value_counts().sort_index()

total = len(train_df)

for diagnosis, count in class_counts.items():

    percentage = (count / total) * 100

    print(
        f"Grade {diagnosis}: "
        f"{count} images "
        f"({percentage:.2f}%)"
    )

# =========================
# FINAL SUMMARY
# =========================

print("\n========== AUDIT SUMMARY ==========\n")

print("Total labelled training images:", len(train_df))
print("Total test images:", len(test_df))
print("Corrupted training images:", len(corrupted))
print("Missing training images:", len(missing_train_images))
print("Missing test images:", len(missing_test_images))
print("Duplicate training IDs:", train_df["id_code"].duplicated().sum())
print("Duplicate test IDs:", test_df["id_code"].duplicated().sum())

print("\nAudit completed.")