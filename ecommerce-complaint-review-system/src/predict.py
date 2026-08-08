"""
Loads trained CNN, RNN, and LSTM checkpoints and runs inference + fusion
on a single complaint text.
"""
import os
import torch

from src.preprocessing import Vocabulary, LabelMap, CATEGORIES, SEVERITIES
from src.models.cnn_model import TextCNN
from src.models.rnn_model import TextRNN
from src.models.lstm_model import TextLSTM
from src.models.fusion import fuse_predictions

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CKPT_DIR = os.path.join(ROOT, "checkpoints")
MAX_LEN = 30


class ComplaintClassifier:
    """Wraps the trained ensemble for easy reuse in the pipeline / API."""

    def __init__(self, ckpt_dir=CKPT_DIR, device=None):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        vocab_path = os.path.join(ckpt_dir, "vocab.json")
        if not os.path.exists(vocab_path):
            raise FileNotFoundError(
                f"No trained checkpoints found at {ckpt_dir}. Run `python src/train.py` first."
            )

        self.vocab = Vocabulary.load(vocab_path)
        self.category_map = LabelMap.load(os.path.join(ckpt_dir, "category_map.json"))
        self.severity_map = LabelMap.load(os.path.join(ckpt_dir, "severity_map.json"))

        vocab_size = len(self.vocab)
        num_categories = len(self.category_map)
        num_severities = len(self.severity_map)

        self.cnn = TextCNN(vocab_size, num_categories, num_severities)
        self.rnn = TextRNN(vocab_size, num_categories, num_severities)
        self.lstm = TextLSTM(vocab_size, num_categories, num_severities)

        self.cnn.load_state_dict(torch.load(os.path.join(ckpt_dir, "cnn.pt"), map_location=self.device))
        self.rnn.load_state_dict(torch.load(os.path.join(ckpt_dir, "rnn.pt"), map_location=self.device))
        self.lstm.load_state_dict(torch.load(os.path.join(ckpt_dir, "lstm.pt"), map_location=self.device))

        for m in (self.cnn, self.rnn, self.lstm):
            m.to(self.device)
            m.eval()

    @torch.no_grad()
    def predict(self, text: str, weights=(1 / 3, 1 / 3, 1 / 3)):
        ids = torch.tensor([self.vocab.encode(text, MAX_LEN)], dtype=torch.long).to(self.device)

        cnn_out = self.cnn(ids)
        rnn_out = self.rnn(ids)
        lstm_out = self.lstm(ids)

        fused = fuse_predictions(cnn_out, rnn_out, lstm_out, weights=weights)

        category = self.category_map.decode(fused["category_idx"].item())
        severity = self.severity_map.decode(fused["severity_idx"].item())
        confidence = round(fused["category_confidence"].item(), 4)
        severity_confidence = round(fused["severity_confidence"].item(), 4)

        return {
            "text": text,
            "category": category,
            "severity": severity,
            "category_confidence": confidence,
            "severity_confidence": severity_confidence,
        }


if __name__ == "__main__":
    clf = ComplaintClassifier()
    sample = "I ordered a laptop but three weeks later it still hasn't shipped and support won't respond."
    result = clf.predict(sample)
    print(result)
