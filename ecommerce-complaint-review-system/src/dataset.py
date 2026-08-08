"""PyTorch Dataset for encoded complaint text with category + severity labels."""
import torch
from torch.utils.data import Dataset


class ComplaintDataset(Dataset):
    def __init__(self, texts, categories, severities, vocab, category_map, severity_map, max_len=30):
        self.texts = texts
        self.categories = categories
        self.severities = severities
        self.vocab = vocab
        self.category_map = category_map
        self.severity_map = severity_map
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        ids = self.vocab.encode(self.texts[idx], self.max_len)
        cat = self.category_map.encode(self.categories[idx])
        sev = self.severity_map.encode(self.severities[idx])
        return (
            torch.tensor(ids, dtype=torch.long),
            torch.tensor(cat, dtype=torch.long),
            torch.tensor(sev, dtype=torch.long),
        )
