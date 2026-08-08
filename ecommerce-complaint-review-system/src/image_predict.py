"""
Loads the trained image-authenticity CNN and classifies a single complaint
photo as Real or Fake (digitally manipulated), with a confidence score.
"""
import os
import torch
import torch.nn.functional as F
from PIL import Image

from src.image_utils import preprocess_for_model, IMAGE_SIZE
from src.models.image_cnn import ImageAuthenticityCNN

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CKPT_DIR = os.path.join(ROOT, "checkpoints")

LABELS = {0: "Real", 1: "Fake"}


class ImageAuthenticityChecker:
    def __init__(self, ckpt_dir=CKPT_DIR, device=None):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        ckpt_path = os.path.join(ckpt_dir, "image_cnn.pt")
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(
                f"No image CNN checkpoint found at {ckpt_path}. "
                f"Run `python scripts/generate_sample_images.py` then "
                f"`python src/train_image_cnn.py` first."
            )
        self.model = ImageAuthenticityCNN(input_size=IMAGE_SIZE)
        self.model.load_state_dict(torch.load(ckpt_path, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()

    @torch.no_grad()
    def predict(self, image: Image.Image) -> dict:
        arr = preprocess_for_model(image, IMAGE_SIZE)
        x = torch.tensor(arr, dtype=torch.float32).unsqueeze(0).to(self.device)
        logits = self.model(x)
        probs = F.softmax(logits, dim=-1).squeeze(0)
        pred_idx = int(torch.argmax(probs).item())

        return {
            "label": LABELS[pred_idx],
            "is_real": pred_idx == 0,
            "confidence": round(float(probs[pred_idx].item()), 4),
            "real_probability": round(float(probs[0].item()), 4),
            "fake_probability": round(float(probs[1].item()), 4),
        }
