
from torch.utils.data import Dataset
from PIL import Image, ImageOps
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
                    if Image.open(os.path.join(dirname, filename)).size != (384, 256):
                        continue
                    self.samples.append((os.path.join(dirname, filename), 1))
                    # print(self.samples[-1])
                    if max_samples_per_class and len(
                        [s for s in self.samples if s[1] == 1]
                    ) >= max_samples_per_class:
                        break

        # Tampered = label 0
        for dirname, _, filenames in os.walk(tampered_dir):
            for filename in filenames:
                if filename.lower().endswith(("jpg", "png")):
                    if Image.open(os.path.join(dirname, filename)).size != (384, 256):
                        continue
                    self.samples.append((os.path.join(dirname, filename), 0))
                    # print(self.samples[-1])
                    if max_samples_per_class and len(
                        [s for s in self.samples if s[1] == 0]
                    ) >= max_samples_per_class:
                        break

        print(f"Total samples loaded: {len(self.samples)}")

       

    def __len__(self):
        return len(self.samples)


    def __getitem__(self, idx):
        """
        Wegen der unterschiedlichen Bildgrößen in CASIA wird hier das Bild
        quadratisch gepaddet, bevor die Transformationen angewendet werden.
        """
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert("RGB")

        # # --- Padding: Quadratisches Bild erzeugen ---
        # max_side = max(image.size)
        # image = ImageOps.pad(image, (max_side, max_side), color=(0,0,0))  # schwarze Ränder

        
        if self.transform:
            image = self.transform(image)

        return image, torch.tensor(label), img_path