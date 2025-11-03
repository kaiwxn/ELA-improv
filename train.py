
import os
import random
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from ela import convert_to_ela_image


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

class ELA:
    def __init__(self, quality=90):
        self.quality = quality

    def __call__(self, img):
        return convert_to_ela_image(img, quality=self.quality)

class Trainer:

    def __init__(self, model, device, loss_fn, optimizer, scheduler):
        self.model = model
        self.device = device
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.scheduler = scheduler

    def train(self, dataloader):
        size = len(dataloader.dataset)
        self.model.train()

        avg_loss = 0

        for batch, (X, y, path) in enumerate(dataloader):  # X: Image, y: Label
            X, y = X.to(self.device), y.to(self.device)

            # Compute prediction error
            pred = self.model(X)
            loss = self.loss_fn(pred, y)

            # Backpropagation
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()

            # Info
            loss, current = loss.item(), (batch + 1) * len(X)
            avg_loss += loss
            print(f"[{current:>5d}/{size:>5d}] loss: {loss:>7f}")

        avg_loss /= len(dataloader)
        print(f"Training error: \n Avg loss: {avg_loss:>8f} \n")
        return avg_loss

    def validate(self, dataloader):
        size = len(dataloader.dataset)
        num_batches = len(dataloader)
        self.model.eval()
        loss, correct = 0, 0

        with torch.no_grad():
            for X, y, path in dataloader:
                X, y = X.to(self.device), y.to(self.device)
                pred = self.model(X)
                loss += self.loss_fn(pred, y).item()
                correct += (pred.argmax(1) == y).type(torch.float).sum().item()

        avg_loss = loss / num_batches
        correct /= size

        print(f"Validation Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {avg_loss:>8f} \n")

        return avg_loss, correct

    def run_epochs(
        self,
        EPOCHS,
        train_dataloader,
        valid_dataloader,
    ):

        final_validation_loss = []
        final_accuracy = []

        train_losses = []
        validation_losses = []

        for epoch in range(EPOCHS):
            print(f"Epoch {epoch+1}\n-------------------------------")

            train_loss = self.train(train_dataloader)
            train_losses.append(train_loss)

            self.scheduler.step()
            validation_loss, accuracy = self.validate(valid_dataloader)

            validation_losses.append(validation_loss)
            final_validation_loss.append(validation_loss)
            final_accuracy.append(accuracy)

        # Return the average validation loss and average accuracy over all epochs
        return (
            train_losses,
            validation_losses,
            sum(final_validation_loss) / len(final_validation_loss),
            sum(final_accuracy) / len(final_accuracy),
        )

def main():
    """
    1. Lade den Datensatz mit auth und manip Bildern
    2. Wandle alle Bilder in ELA-Bilder um
    3. Mache ein train/test split (80/20)

    4. Trainiere ein CNN auf den ELA-Bildern
    5. speichere das Modell ab
    6. Evaluiere das Modell auf dem Testset

    Overview über den Datensatz (CASIA v1.0 und v2.0):
    https://www.researchgate.net/figure/Overview-of-CASIA-v10-and-CASIA-v20_tbl1_332949646 
    https://www.researchgate.net/figure/The-details-of-the-CASIA-10-CASIA-20-and-CUISDE-datasets_tbl2_367220128 

    DOWNLOAD 1.0:
    https://github.com/namtpham/casia1groundtruth

    DOWNLOAD 2.0:
    https://www.kaggle.com/datasets/divg07/casia-20-image-tampering-detection-dataset/data 
    """

    # enable CUDA if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.backends.cudnn.benchmark = True

    # deterministic seeds
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)

    preprocess = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.Resize((128, 128)),
        ELA(),
        transforms.ToTensor(),
    ])

    dataset = CASIADataset("./data/CASIA2", transform=preprocess, max_samples_per_class=1000)

    # Split train/test 80/20
    total_size = len(dataset)
    train_size = int(0.8 * total_size)
    test_size = total_size - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4)

    print(f"Train size: {len(train_dataset)}, Test size: {len(test_dataset)}")


    model = CNN().to(device)

    # Collect only the parameters that require gradient computation
    params = [p for p in model.parameters() if p.requires_grad]

    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(params, lr=1e-3)
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.9)
    
    trainer = Trainer(model, device, loss_fn, optimizer, scheduler=None)
    
    print(trainer.run_epochs(EPOCHS=10, train_dataloader=train_loader, valid_dataloader=test_loader))



if __name__ == "__main__":
    main()