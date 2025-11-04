import torch

import torch.nn as nn
import torch.nn.functional as F

    

class CNN(nn.Module):
    """
    Input images are (128x128) grayscale, 3 channels. 32 Batch size.
    Input shape: [32, 3, 128, 128].
    """
    
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1),  # 128 → 64
            nn.ReLU(),
            nn.MaxPool2d(2),  # 64 → 32

            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),  # 32 → 16
            nn.ReLU(),
            nn.MaxPool2d(2),  # 16 → 8
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 8 * 8, 512),
            nn.ReLU(),
            nn.Linear(512, 2),  # binary classification
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x