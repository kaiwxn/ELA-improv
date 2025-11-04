import torch
from torchvision import transforms
from PIL import Image
from model import CNN
from ela import convert_to_ela_image

# --- Hilfsklasse für ELA-Vorverarbeitung ---
class ELA:
    def __init__(self, quality=90):
        self.quality = quality

    def __call__(self, img):
        return convert_to_ela_image(img, quality=self.quality)


def predict_image(model_path, image_path, quality=90):
    """
    Prüft ein einzelnes Bild mit einem trainierten CNN-Modell.
    Gibt die vorhergesagte Klasse und Konfidenz zurück.
    """

    # --- Gerät wählen ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- Modell laden ---
    model = CNN().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.Resize((128, 128)),
        ELA(quality=quality),
        transforms.ToTensor(),
    ])
    

    # --- Bild laden ---
    image = Image.open(image_path).convert("RGB")
    image_tensor = transform(image).unsqueeze(0).to(device) # type: ignore

    # --- Vorhersage ---
    with torch.no_grad():
        output = model(image_tensor)
        probs = torch.softmax(output, dim=1)
        pred_class = probs.argmax(dim=1).item()
        confidence = probs[0, pred_class].item() # type: ignore

    # --- Ergebnis ausgeben ---
    label_map = {0: "Authentisch", 1: "Manipuliert"}  # evtl. anpassen

    print("\n=== Bildanalyse ===")
    print(f"Bild: {image_path}")
    print(f"Vorhergesagte Klasse: {label_map[pred_class]}") # type: ignore
    print(f"Konfidenz: {confidence*100:.2f}%\n")

    return label_map[pred_class], confidence # type: ignore


# --- Beispielhafte Nutzung ---
if __name__ == "__main__":
    model_path = "CNN_ELA_20.pth"          # Pfad zum trainierten Modell
    image_path = "./images/zebra.jpg"  # Pfad zum Bild, das geprüft werden soll

    label, conf = predict_image(model_path, image_path)