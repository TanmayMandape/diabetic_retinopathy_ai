import torch
from torch.utils.data import Dataset

from src.preprocessing import load_image, resolve_image_path


class DRDataset(Dataset):
    def __init__(self, dataframe, image_dir, transform=None, image_size=224):
        self.dataframe = dataframe.reset_index(drop=True)
        self.image_dir = image_dir
        self.transform = transform
        self.image_size = image_size

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, index):
        row = self.dataframe.iloc[index]

        image_path = resolve_image_path(self.image_dir, row["id_code"])

        # load_image() applies the SAME crop + resize used at
        # inference time (src/predict.py), so train/serve skew
        # doesn't creep in.
        image = load_image(image_path, image_size=self.image_size)

        label = int(row["diagnosis"])

        if self.transform:
            image = self.transform(image)

        return image, label
