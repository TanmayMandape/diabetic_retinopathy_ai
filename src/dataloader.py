from torch.utils.data import DataLoader

from src.dataset import DRDataset
from src.transforms import (
    train_transform,
    val_transform
)


def create_dataloaders(
    train_df,
    val_df,
    test_df,
    image_dir,
    batch_size=32
):

    train_dataset = DRDataset(
        train_df,
        image_dir,
        train_transform
    )

    val_dataset = DRDataset(
        val_df,
        image_dir,
        val_transform
    )

    test_dataset = DRDataset(
        test_df,
        image_dir,
        val_transform
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0
    )

    return train_loader, val_loader, test_loader
