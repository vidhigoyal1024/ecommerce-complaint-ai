"""
Preprocessing utilities: text cleaning, vocabulary building, and sequence encoding.
Shared by CNN, RNN, and LSTM models so all three see identical input representations.
"""
import re
import json
from collections import Counter

PAD_TOKEN = "<pad>"
UNK_TOKEN = "<unk>"


def clean_text(text: str) -> str:
    """Lowercase, strip punctuation/extra whitespace."""
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def tokenize(text: str):
    return clean_text(text).split()


class Vocabulary:
    """Simple word-level vocabulary with save/load support."""

    def __init__(self, min_freq: int = 1):
        self.min_freq = min_freq
        self.word2idx = {PAD_TOKEN: 0, UNK_TOKEN: 1}
        self.idx2word = {0: PAD_TOKEN, 1: UNK_TOKEN}

    def build(self, texts):
        counter = Counter()
        for t in texts:
            counter.update(tokenize(t))
        for word, freq in counter.items():
            if freq >= self.min_freq and word not in self.word2idx:
                idx = len(self.word2idx)
                self.word2idx[word] = idx
                self.idx2word[idx] = word
        return self

    def encode(self, text: str, max_len: int = 30):
        tokens = tokenize(text)
        ids = [self.word2idx.get(tok, self.word2idx[UNK_TOKEN]) for tok in tokens]
        if len(ids) < max_len:
            ids = ids + [self.word2idx[PAD_TOKEN]] * (max_len - len(ids))
        else:
            ids = ids[:max_len]
        return ids

    def __len__(self):
        return len(self.word2idx)

    def save(self, path: str):
        with open(path, "w") as f:
            json.dump(self.word2idx, f)

    @classmethod
    def load(cls, path: str):
        vocab = cls()
        with open(path, "r") as f:
            vocab.word2idx = json.load(f)
        vocab.idx2word = {int(v): k for k, v in vocab.word2idx.items()}
        return vocab


class LabelMap:
    """Bidirectional mapping between string labels and integer ids."""

    def __init__(self, labels):
        self.label2idx = {label: i for i, label in enumerate(labels)}
        self.idx2label = {i: label for label, i in self.label2idx.items()}

    def encode(self, label):
        return self.label2idx[label]

    def decode(self, idx):
        return self.idx2label[int(idx)]

    def save(self, path):
        with open(path, "w") as f:
            json.dump(self.label2idx, f)

    @classmethod
    def load(cls, path):
        with open(path, "r") as f:
            label2idx = json.load(f)
        obj = cls.__new__(cls)
        obj.label2idx = label2idx
        obj.idx2label = {int(v): k for k, v in label2idx.items()}
        return obj

    def __len__(self):
        return len(self.label2idx)


CATEGORIES = [
    "Delayed Delivery",
    "Defective Product",
    "Wrong Item Delivered",
    "Refund Not Processed",
    "Billing/Overcharge",
    "Poor Customer Service",
]

SEVERITIES = ["Low", "Medium", "High"]
