

"""
# Add dataset (create binary labled images with and without artifacts)
#
# First test ELA on example images --> check, if it works as expected
# 
# Add a neural network to train ELA on the dataset
#
# Export the model
"""

import io
import numpy as np
from PIL import Image, ImageChops, ImageEnhance

def convert_to_ela_image(path, quality):

    image = Image.open(path).convert('RGB')

    # Save the image at a lower quality level
    # JPEG in-memory speichern (kein temporäres Datei)
    buffer = io.BytesIO()
    image.save(buffer, format='JPEG', quality=quality)
    buffer.seek(0)
    compressed = Image.open(buffer).convert('RGB')

    ela_image = ImageChops.difference(image, compressed)

    extrema = ela_image.getextrema()

    max_diff = max([ex[1] for ex in extrema]) or 1 # type: ignore
    scale = 255.0 / max_diff

    ela_image = ImageEnhance.Brightness(ela_image).enhance(scale)

    return ela_image



def detect_manipulation(ela_image, threshold=35.0):
    """
    Berechnet einfache Heuristik: misst mittlere Helligkeit im ELA-Bild.
    Wenn der Wert > threshold → manipuliert.
    """
    ela_array = np.array(ela_image.convert('L'))  # Grauwerte
    mean_val = ela_array.mean()
    std_val = ela_array.std()

    manipulated = mean_val > threshold

    return manipulated, mean_val, std_val


def main():
    QUALITY = 90
    THRESHOLD = 35.0
    test_image_path = './z_shear.jpg'
    saved_ela_img_path = f'./img/z_shear_ela{QUALITY}.png'

    # ELA berechnen
    ela = convert_to_ela_image(test_image_path, quality=QUALITY)
    ela.save(saved_ela_img_path)

    ela.show()
    # Manipulation erkennen
    manipulated, mean_val, std_val = detect_manipulation(ela, threshold=THRESHOLD)

    print(f"[INFO] ELA saved to {saved_ela_img_path}")
    print(f"[INFO] Mean intensity: {mean_val:.2f}, Std: {std_val:.2f}")
    if manipulated:
        print(f"[ALERT] Possible manipulation detected (>{THRESHOLD})!")
    else:
        print(f"[OK] Image appears authentic (≤{THRESHOLD}).")


if __name__ == "__main__":
    main()