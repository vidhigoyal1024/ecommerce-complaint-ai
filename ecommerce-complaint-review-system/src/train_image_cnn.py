"""
Trains the image-authenticity CNN.

Reads authentic photos from data/images/real/, synthetically tampers copies
of them (see src/manipulate.py) to build the "fake" class, computes the ELA
representation of every image (see src/image_utils.py), and trains
ImageAuthenticityCNN to tell them apart.

Usage:
    python scripts/generate_sample_images.py --count 25    # if you have no real photos yet
    python src/train_image_cnn.py --epochs 15
"""
import os
import sys
import glob
import random
import argparse

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.image_utils import compute_ela, preprocess_for_model, load_image_any_format
from src.manipulate import generate_fake
from src.models.image_cnn import ImageAuthenticityCNN

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REAL_DIR = os.path.join(ROOT, "data", "images", "real")
CKPT_DIR = os.path.join(ROOT, "checkpoints")
IMAGE_SIZE = 128


class ELADataset(Dataset):
    """Pre-computes ELA arrays for a list of (PIL image, label) pairs held in memory."""

    def __init__(self, images, labels):
        self.arrays = [preprocess_for_model(img, IMAGE_SIZE) for img in images]
        self.labels = labels

    def __len__(self):
        return len(self.arrays)

    def __getitem__(self, idx):
        return (
            torch.tensor(self.arrays[idx], dtype=torch.float32),
            torch.tensor(self.labels[idx], dtype=torch.long),
        )


def build_dataset(fakes_per_real=1):
    paths = sorted(glob.glob(os.path.join(REAL_DIR, "*.jp*g")) + glob.glob(os.path.join(REAL_DIR, "*.png")))
    if len(paths) < 4:
        raise RuntimeError(
            f"Need at least 4 real images in {REAL_DIR}. "
            f"Run `python scripts/generate_sample_images.py` first, or add your own photos."
        )

    real_images = [load_image_any_format(p) for p in paths]

    images, labels = [], []
    for img in real_images:
        images.append(img)
        labels.append(0)  # 0 = Real
        for _ in range(fakes_per_real):
            fake_img = generate_fake(img, donor_pool=real_images)
            images.append(fake_img)
            labels.append(1)  # 1 = Fake

    return images, labels


def evaluate(model, loader, device):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            correct += (logits.argmax(1) == y).sum().item()
            total += x.size(0)
    return correct / total


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--fakes_per_real", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    os.makedirs(CKPT_DIR, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    images, labels = build_dataset(fakes_per_real=args.fakes_per_real)
    train_imgs, val_imgs, train_labels, val_labels = train_test_split(
        images, labels, test_size=0.2, random_state=args.seed, stratify=labels
    )

    train_ds = ELADataset(train_imgs, train_labels)
    val_ds = ELADataset(val_imgs, val_labels)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size)

    model = ImageAuthenticityCNN(input_size=IMAGE_SIZE).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = torch.nn.CrossEntropyLoss()

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        val_acc = evaluate(model, val_loader, device)
        print(f"[ImageCNN] epoch {epoch:02d}/{args.epochs}  loss={total_loss/len(train_loader):.4f}  val_acc={val_acc:.3f}")

    ckpt_path = os.path.join(CKPT_DIR, "image_cnn.pt")
    torch.save(model.state_dict(), ckpt_path)
    print(f"\nSaved image authenticity CNN to: {ckpt_path}")


if __name__ == "__main__":
    main()
