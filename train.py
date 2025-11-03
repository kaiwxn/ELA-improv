


import torch
from PIL import Image
from torchvision import datasets, transforms

from ela import convert_to_ela_image


def load_dataset():
    import kagglehub

    # Download latest version
    path = kagglehub.dataset_download("divg07/casia-20-image-tampering-detection-dataset", path="data/")
    print("Path to dataset files:", path)


class ELA:
    def __init__(self, quality=90):
        self.quality = quality

    def __call__(self, img):
        return convert_to_ela_image(img, quality=self.quality)



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

    # load_dataset()

    preprocess = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.Resize((128, 128)),
        ELA(),
        transforms.ToTensor(),
    ])
    img = Image.open('./manual/z_lion.png').convert('RGB')

    img_tensor = preprocess(img)

    print(img_tensor.shape)

if __name__ == "__main__":
    main()