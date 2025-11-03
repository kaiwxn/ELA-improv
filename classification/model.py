import torch

import torch.nn as nn
import torch.nn.functional as F


class CNN(nn.Module):

    """
    Input images are (128x128) grayscale, 3 channels
    Input shape: [3, 128, 128].
    """
    def __init__(self):
        super().__init__()

        self.features = nn.Sequential(
            # Input: 3 x 128 x 128
            nn.Conv2d(3, 16, kernel_size=3, padding=1),  # → 16 x 128 x 128
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2),  # → 16 x 64 x 64

            nn.Conv2d(16, 32, kernel_size=3, padding=1),  # → 32 x 64 x 64
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),  # → 32 x 32 x 32

            nn.Conv2d(32, 64, kernel_size=3, padding=1),  # → 64 x 32 x 32
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),  # → 64 x 16 x 16

            nn.Conv2d(64, 128, kernel_size=3, padding=1),  # → 128 x 16 x 16
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),  # → 128 x 8 x 8
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 8 * 8, 256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, 2)  # binary classification
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x