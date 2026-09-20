import os
from PIL import Image
from torch.utils.data import Dataset


class Stage1Dataset(Dataset):
    """
    Stage 1:
    Grade 0, 1 -> 0 (Non-referable)
    Grade 2, 3, 4 -> 1 (Referable)
    """

    def __init__(self, dataframe, image_dir, transform=None):
        self.dataframe = dataframe.reset_index(drop=True)
        self.image_dir = image_dir
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, index):
        row = self.dataframe.iloc[index]

        image_id = row["id_code"]
        original_label = int(row["diagnosis"])

        # Convert Grade 0-4 → Stage 1 label
        label = 0 if original_label in [0, 1] else 1

        image_path = os.path.join(
            self.image_dir,
            image_id + ".png"
        )

        image = Image.open(image_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, label


class Stage2Dataset(Dataset):
    """
    Stage 2:
    Grade 2 -> 0
    Grade 3 -> 1
    Grade 4 -> 2

    Only referable images should be included.
    """

    def __init__(self, dataframe, image_dir, transform=None):
        self.dataframe = dataframe.reset_index(drop=True)
        self.image_dir = image_dir
        self.transform = transform

        # Safety check
        invalid_labels = set(
            self.dataframe["diagnosis"].unique()
        ) - {2, 3, 4}

        if invalid_labels:
            raise ValueError(
                f"Stage 2 dataset contains invalid labels: "
                f"{invalid_labels}"
            )

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, index):
        row = self.dataframe.iloc[index]

        image_id = row["id_code"]
        original_label = int(row["diagnosis"])

        # Convert Grade 2-4 → Stage 2 label
        mapping = {
            2: 0,
            3: 1,
            4: 2
        }

        label = mapping[original_label]

        image_path = os.path.join(
            self.image_dir,
            image_id + ".png"
        )

        image = Image.open(image_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, label