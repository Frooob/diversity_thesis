import os

import numpy as np
from tqdm import tqdm
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from low_level_clustering import sort_im_idx, images_to_pixel_features
from dropout_random import get_all_input_imgs


def img_to_quantized_highest_quantized_count(img, bin_size = 10):
    pixels = np.array(img.getdata())
    quantized_pixels = []
    for (r, g, b) in pixels:
        qr = r // bin_size
        qg = g // bin_size
        qb = b // bin_size
        quantized_pixels.append((qr, qg, qb))
    quantized_series = pd.Series(quantized_pixels).value_counts()
    highest_count = quantized_series.iloc[0]
    return highest_count

def generate_quantized_counts(input_filepaths, im_idx):
    quantized_counts = []
    all_images = []
    for f, fidx in tqdm(zip(input_filepaths,im_idx), total=len(input_filepaths)):
        img = Image.open(f).convert("RGB").resize((100,100))
        quantized_count = img_to_quantized_highest_quantized_count(img)
        quantized_counts.append((f, quantized_count, fidx))
    sorted_quantized_counts = sorted(quantized_counts, key=lambda x: x[1], reverse=True)
    return sorted_quantized_counts

def generate_quantized_count_sample(out_path_base, dataset, sorted_quantized_counts, fraction, reverse = False):
    sample_size = int(fraction * len(sorted_quantized_counts))
    sorted_again_quantized_counts = sorted(sorted_quantized_counts, key=lambda x: x[1], reverse= not reverse)
    chosen_im_idx = np.array([s[2] for s in sorted_again_quantized_counts[:sample_size]])
    path_dropout_samples = os.path.join(out_path_base, f'data/dropout_samples/{dataset}/')
    sort_name = "Boring" if not reverse else "Party"
    algorithm_name = "quantizedCount"+sort_name
    sample_filename = f"dropout-{algorithm_name}_{fraction}_00"
    np.save(os.path.join(path_dropout_samples, sample_filename), chosen_im_idx)
    return chosen_im_idx


def plot_two_rows_of_images(image_paths, thesis_plot_path):
    """
    image_paths: list of file paths (strings) for images. 
                 Assumes at least 10 image paths in the list.
    """
    titlesize = 20
    # Create 2 rows and 5 columns of subplots
    fig, axes = plt.subplots(2, 5, figsize=(12, 5))
    
    # First row: boring images
    for i, img_path in enumerate(image_paths[:5]):
        img = mpimg.imread(img_path[0])
        axes[0, i].imshow(img)
        axes[0, i].axis('off')  # Hide axes/ticks
    # Place title on the middle image (column index 2)
    axes[0, 2].set_title('Monotone Images', fontsize=titlesize)
    
    # Second row: party images
    for j, img_path in enumerate(image_paths[-5:]):
        img = mpimg.imread(img_path[0])
        axes[1, j].imshow(img)
        axes[1, j].axis('off')  # Hide axes/ticks
    # Place title on the middle image (column index 2)
    axes[1, 2].set_title('Heterogeneous Images', fontsize=titlesize)
    
    plt.tight_layout()
    plt.savefig(os.path.join(thesis_plot_path, "dropout_discussion_monohetero_qual.jpeg"), dpi=200)
    plt.show()



if __name__ == "__main__":
    np.random.seed(42)
    out_path_base = "/Users/matt/programming/recon_diffuser/"
    dataset = "deeprecon"
    path_features = f"{dataset}_train_features.npz"
    im_suffix = "JPEG"

    # make sure the input_filepaths have the same ordering than the npz_images
    im_idx, all_input_filepaths = get_all_input_imgs(out_path_base, dataset, im_suffix)
    npz = np.load(path_features)
    npz_im_idx = npz["im_idx"]
    im_idx_sorted, all_input_filepaths = sort_im_idx(npz_im_idx, im_idx, all_input_filepaths)


    # Show the first 10 images
    sorted_quantized_counts = generate_quantized_counts(all_input_filepaths, im_idx_sorted)

    boring_idx = generate_quantized_count_sample(out_path_base, dataset, sorted_quantized_counts, 0.25, False)
    party_idx = generate_quantized_count_sample(out_path_base, dataset, sorted_quantized_counts, 0.25, True)

    assert sorted_quantized_counts[0][2] == boring_idx[0]
    assert sorted_quantized_counts[-1][2] == party_idx[0]


    # Make a plot of the top 5 "monotone" and "heterogeneous" images
    image_paths = sorted_quantized_counts
    thesis_plot_path = "/Users/matt/ownCloud/gogo/MA/thesis/diversity_thesis/plots"
    plot_two_rows_of_images(sorted_quantized_counts, thesis_plot_path)