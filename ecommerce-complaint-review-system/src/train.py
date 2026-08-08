"""
Trains the CNN, RNN, and LSTM classifiers on data/sample_complaints.csv and
saves weights + vocabulary + label maps to checkpoints/.

Usage:
    python src/train.py --epochs 15
"""
import os
import sys
import argparse

import pandas as pd
import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.preprocessing import Vocabulary, LabelMap, CATEGORIES, SEVERITIES
from src.dataset import ComplaintDataset
from src.models.cnn_model import TextCNN
from src.models.rnn_model import TextRNN
from src.models.lstm_model import TextLSTM

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(ROOT, "data", "sample_complaints.csv")
CKPT_DIR = os.path.join(ROOT, "checkpoints")
MAX_LEN = 30


def train_one_model(model, train_loader, val_loader, epochs, lr, device, name):
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = torch.nn.CrossEntropyLoss()

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        for ids, cat, sev in train_loader:
            ids, cat, sev = ids.to(device), cat.to(device), sev.to(device)
            optimizer.zero_grad()
            cat_logits, sev_logits = model(ids)
            loss = criterion(cat_logits, cat) + criterion(sev_logits, sev)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        val_acc_cat, val_acc_sev = evaluate(model, val_loader, device)
        print(f"[{name}] epoch {epoch:02d}/{epochs}  loss={total_loss/len(train_loader):.4f}  "
              f"val_cat_acc={val_acc_cat:.3f}  val_sev_acc={val_acc_sev:.3f}")

    return model


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    correct_cat, correct_sev, total = 0, 0, 0
    for ids, cat, sev in loader:
        ids, cat, sev = ids.to(device), cat.to(device), sev.to(device)
        cat_logits, sev_logits = model(ids)
        correct_cat += (cat_logits.argmax(1) == cat).sum().item()
        correct_sev += (sev_logits.argmax(1) == sev).sum().item()
        total += ids.size(0)
    return correct_cat / total, correct_sev / total


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    os.makedirs(CKPT_DIR, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)

    df = pd.read_csv(DATA_PATH)
    train_df, val_df = train_test_split(
        df, test_size=0.2, random_state=args.seed, stratify=df["category"]
    )

    vocab = Vocabulary(min_freq=1).build(df["text"].tolist())
    category_map = LabelMap(CATEGORIES)
    severity_map = LabelMap(SEVERITIES)

    vocab.save(os.path.join(CKPT_DIR, "vocab.json"))
    category_map.save(os.path.join(CKPT_DIR, "category_map.json"))
    severity_map.save(os.path.join(CKPT_DIR, "severity_map.json"))

    train_ds = ComplaintDataset(
        train_df["text"].tolist(), train_df["category"].tolist(), train_df["severity"].tolist(),
        vocab, category_map, severity_map, max_len=MAX_LEN
    )
    val_ds = ComplaintDataset(
        val_df["text"].tolist(), val_df["category"].tolist(), val_df["severity"].tolist(),
        vocab, category_map, severity_map, max_len=MAX_LEN
    )
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size)

    vocab_size = len(vocab)
    num_categories = len(category_map)
    num_severities = len(severity_map)

    cnn = TextCNN(vocab_size, num_categories, num_severities)
    rnn = TextRNN(vocab_size, num_categories, num_severities)
    lstm = TextLSTM(vocab_size, num_categories, num_severities)

    cnn = train_one_model(cnn, train_loader, val_loader, args.epochs, args.lr, device, "CNN")
    rnn = train_one_model(rnn, train_loader, val_loader, args.epochs, args.lr, device, "RNN")
    lstm = train_one_model(lstm, train_loader, val_loader, args.epochs, args.lr, device, "LSTM")

    torch.save(cnn.state_dict(), os.path.join(CKPT_DIR, "cnn.pt"))
    torch.save(rnn.state_dict(), os.path.join(CKPT_DIR, "rnn.pt"))
    torch.save(lstm.state_dict(), os.path.join(CKPT_DIR, "lstm.pt"))

    print(f"\nSaved checkpoints and vocab/label maps to: {CKPT_DIR}")


if __name__ == "__main__":
    main()
