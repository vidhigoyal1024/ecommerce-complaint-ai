"""
ImageAuthenticityCNN: a CNN that classifies an image's Error-Level-Analysis
map as "Real" (untampered) or "Fake" (digitally manipulated).

This model is used for exactly one purpose in this system: verifying
whether a photo a customer attaches to a complaint (e.g. "here's my
damaged product") shows signs of digital tampering. It is intentionally
kept separate from the CNN used for complaint *text* classification.
"""
import torch
import torch.nn as nn


class ImageAuthenticityCNN(nn.Module):
    def __init__(self, in_channels=3, num_classes=2, input_size=128):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # -> size/2

            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # -> size/4

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # -> size/8
        )
        reduced = input_size // 8
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * reduced * reduced, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)
