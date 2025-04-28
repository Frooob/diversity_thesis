# Yoho. I just want to have some plots huh


p_train = '/Users/matt/programming/recon_diffuser/data/train_data/deeprecon'
p_art = "/Users/matt/programming/recon_diffuser/data/annots/artificial_shapes/stimuli/source"

thesis_plots_path = "/Users/matt/ownCloud/gogo/MA/thesis/diversity_thesis/plots"

import os
import random
from PIL import Image

import matplotlib.pyplot as plt

def plot_random_images(p_train, p_art):
    train_images = [os.path.join(p_train, f) for f in os.listdir(p_train) if f.endswith('.JPEG')]
    art_images = [os.path.join(p_art, f) for f in os.listdir(p_art) if f.endswith('.tiff')]
    art_images.sort()

    selected_train_images = random.sample(train_images, 5)
    selected_art_images = art_images[::6][:5]

    fig, axes = plt.subplots(2, 6, figsize=(15, 6))

    axes[0, 0].text(0.5, 0.5, 'Train/Test images', horizontalalignment='center', verticalalignment='center', fontsize=18)
    axes[0, 0].axis('off')
    axes[1, 0].text(0.5, 0.5, 'Artificial shapes', horizontalalignment='center', verticalalignment='center', fontsize=18)
    axes[1, 0].axis('off')

    for i, img_path in enumerate(selected_train_images):
        img = Image.open(img_path)
        axes[0, i + 1].imshow(img)
        axes[0, i + 1].axis('off')

    for i, img_path in enumerate(selected_art_images):
        img = Image.open(img_path)
        axes[1, i + 1].imshow(img)
        axes[1, i + 1].axis('off')

    plt.tight_layout()
    # plt.show()
    im_path = os.path.join(thesis_plots_path, "datasets_train_art_images.jpeg")
    plt.savefig(im_path, dpi=200)


plot_random_images(p_train, p_art)


# Random noise

from PIL import Image
import numpy as np

# Generate random noise (100x100) with values between 0 and 255
noise = (np.random.rand(300, 300, 3) * 255).astype(np.uint8)

# Convert to a PIL image
image = Image.fromarray(noise)

# Save as PNG
image.save("random_noise.png")

print("Random noise image saved as random_noise.png")


## Get all the test captions

import pandas as pd

captions_art = pd.read_csv("/Users/matt/programming/recon_diffuser/data/annots/artificial_shapes/artificial_shapes_captions.csv")
captions_test = pd.read_csv("/Users/matt/programming/recon_diffuser/data/annots/ImagenetTest/captions/amt_20181204.csv")
captions_art.iloc[1:51:5]["caption"]
captions_test.iloc[1:51:5]["caption"]

captions_art = captions_art.iloc[:51:5]
captions_art = captions_art.iloc[:50]