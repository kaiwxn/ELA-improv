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
    
class DeepCNN(nn.Module):
    """
    Deep CNN for 384x256 ELA images, 3 channels (landscape format).
    All strides = 1 to preserve fine-grained compression patterns.
    """
    def __init__(self):
        super().__init__()
        
        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 256x384 → 128x192

            # Block 2
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 128x192 → 64x96

            # Block 3
            nn.Conv2d(64, 512, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)   # 16x24 → 8x12
        )
        
        # Flattened Feature Map: 512 x 8 x 12 = 49,152
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512 * 8 * 12, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 2)  # binary classification
        )
    
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x