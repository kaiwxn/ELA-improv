import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from dataset import CASIADataset
from model import DeepCNN
from PIL import Image, ImageChops, ImageEnhance
import io

class ELA:
    def __init__(self, quality=80):
        self.quality = quality

    def __call__(self, img):
        buffer = io.BytesIO()
        img.save(buffer, format='JPEG', quality=self.quality)
        buffer.seek(0)
        compressed = Image.open(buffer).convert('RGB')
        ela_image = ImageChops.difference(img, compressed)
        extrema = ela_image.getextrema()
        max_diff = max([ex[1] for ex in extrema]) or 1 # type: ignore
        scale = 255.0 / max_diff
        ela_image = ImageEnhance.Brightness(ela_image).enhance(scale)
        return ela_image

def test_model(model_path, test_dir, batch_size=16, device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Transformation wie beim Training
    preprocess = transforms.Compose([
        ELA(quality=80),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5])
    ])

    # Dataset und DataLoader
    test_dataset = CASIADataset(test_dir, transform=preprocess)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Modell laden
    model = DeepCNN().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    wrong_predictions = []

    with torch.no_grad():
        for images, labels, paths in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            preds = outputs.argmax(dim=1)
            for i in range(len(preds)):
                if preds[i] != labels[i]:
                    wrong_predictions.append(paths[i])

    print(f"Anzahl falscher Vorhersagen: {len(wrong_predictions)}")
    if wrong_predictions:
        print("Falsch klassifizierte Bilder:")
        for path in wrong_predictions:
            print(path)

if __name__ == "__main__":
    MODEL_PATH = "CNN_ELA_DEEPER_CNN_15_NORM_wdc5.pth"
    TEST_DIR = "./data/CASIA2/"  
    test_model(MODEL_PATH, TEST_DIR)
