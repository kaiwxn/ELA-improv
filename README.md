# AI-assisted Error Level Analysis

## What is Error Level Analysis?
The Error Level Analysis (ELA) technique is a powerful method for detecting image manipulation by analyzing the compression artifacts in an image, e.g. for JPEG graphics. 

Given the original image, ELA works by re-saving the image at a known compression level and then comparing the original and re-saved images to identify areas that may have been altered.

![ELA Example](/images/ela_imgs/zebra_tree_ela85_enhanced.png)

## AI-assisted ELA
**Humans** are required to visually inspect the ELA output to identify potential manipulations, which can be time-consuming and subjective. Instead, **CNNs** can be trained to detect patterns in those images and classify them as manipulated or not.

## Dataset
The dataset used is the [CASIA v2.0](https://www.kaggle.com/datasets/divg07/casia-20-image-tampering-detection-dataset) dataset, which contains a large number of images with various types of manipulations, including splicing, copy-move, and retouching. Each image is labeled.

## Training the Model
![Training Process](/images/Training%20data%20pipeline.png)




