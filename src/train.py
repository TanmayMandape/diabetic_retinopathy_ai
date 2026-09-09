"""
Run with:  python -m src.train
(run from the project root, diabetic-retinopathy-ai/)

Expects:
  data/train.csv           columns: id_code, diagnosis
  data/train_images/       fundus images referenced by id_code
"""

import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, f1_score
from tqdm import tqdm

from src.config import TRAIN_CSV, IMG_DIR, MODEL_PATH, CLASS_NAMES
from src.dataset import DRDataset
from src.transforms import train_transform, val_transform
from src.model import create_model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss, correct, total = 0, 0, 0

    for images, labels in tqdm(loader, desc="train"):
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        predictions = outputs.argmax(dim=1)
        correct += (predictions == labels).sum().item()
        total += labels.size(0)

    return running_loss / total, correct / total


def validate(model, loader, criterion, device):
    model.eval()
    running_loss, correct, total = 0, 0, 0

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)
            predictions = outputs.argmax(dim=1)
            correct += (predictions == labels).sum().item()
            total += labels.size(0)

    return running_loss / total, correct / total


def main(num_epochs=10, batch_size=16, lr=1e-4):
    df = pd.read_csv(TRAIN_CSV)
    print("Number of images:", len(df))
    print(df["diagnosis"].value_counts().sort_index())

    train_df, temp_df = train_test_split(
        df, test_size=0.30, stratify=df["diagnosis"], random_state=42
    )
    val_df, test_df = train_test_split(
        temp_df, test_size=0.50, stratify=temp_df["diagnosis"], random_state=42
    )
    print("Train:", len(train_df), "Val:", len(val_df), "Test:", len(test_df))

    train_dataset = DRDataset(train_df, IMG_DIR, train_transform)
    val_dataset = DRDataset(val_df, IMG_DIR, val_transform)
    test_dataset = DRDataset(test_df, IMG_DIR, val_transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    classes = np.array([0, 1, 2, 3, 4])
    weights = compute_class_weight(
        class_weight="balanced", classes=classes, y=train_df["diagnosis"]
    )
    class_weights = torch.tensor(weights, dtype=torch.float32).to(device)

    model = create_model(num_classes=5).to(device)
    criterion = torch.nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    best_val_loss = float("inf")

    for epoch in range(num_epochs):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = validate(model, val_loader, criterion, device)

        print(f"Epoch {epoch + 1}/{num_epochs}")
        print(f"Train Loss: {train_loss:.4f}  Train Acc: {train_acc:.4f}")
        print(f"Val Loss:   {val_loss:.4f}  Val Acc:   {val_acc:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), MODEL_PATH)
            print(f"Model saved to {MODEL_PATH}")

    # Final evaluation on the held-out test set
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device, weights_only=True))
    model.eval()

    all_predictions, all_labels = [], []
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            outputs = model(images)
            predictions = outputs.argmax(dim=1)
            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(labels.numpy())

    print(classification_report(all_labels, all_predictions, target_names=CLASS_NAMES))
    print("Macro F1:", f1_score(all_labels, all_predictions, average="macro"))


if __name__ == "__main__":
    main()
