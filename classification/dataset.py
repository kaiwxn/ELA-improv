
from torch.utils.data import Dataset
from PIL import Image
import torch

import os



def load_dataset():
    import kagglehub

    # Download latest version
    path = kagglehub.dataset_download("divg07/casia-20-image-tampering-detection-dataset", path="data/")
    print("Path to dataset files:", path)


class CASIADataset(Dataset):
    def __init__(self, root_dir, transform=None, max_samples_per_class=None):
        """
        root_dir: path containing 'Au' (authentic) and 'Tp' (tampered)
        transform: torchvision transform pipeline
        max_samples_per_class: optional limit per class
        """
        self.samples = []
        self.transform = transform

        auth_dir = os.path.join(root_dir, "Au")
        tampered_dir = os.path.join(root_dir, "Tp")

        # Authentic = label 1
        for dirname, _, filenames in os.walk(auth_dir):
            for filename in filenames:
                if filename.lower().endswith(("jpg", "png")):
                    self.samples.append((os.path.join(dirname, filename), 1))
                    if max_samples_per_class and len(
                        [s for s in self.samples if s[1] == 1]
                    ) >= max_samples_per_class:
                        break

        # Tampered = label 0
        for dirname, _, filenames in os.walk(tampered_dir):
            for filename in filenames:
                if filename.lower().endswith(("jpg", "png")):
                    self.samples.append((os.path.join(dirname, filename), 0))
                    if max_samples_per_class and len(
                        [s for s in self.samples if s[1] == 0]
                    ) >= max_samples_per_class:
                        break

        print(f"Total samples loaded: {len(self.samples)}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")

        if self.transform:
            img = self.transform(img)

        return img, torch.tensor(label, dtype=torch.long), path