from skimage.metrics import structural_similarity as ssim
import cv2
import numpy as np

from dropout_random import get_all_input_imgs
import numpy as np
import cv2
import numpy as np
from sklearn.preprocessing import StandardScaler
import umap
from tqdm import tqdm
import base64
import pandas as pd
import io
import os

from skimage.metrics import structural_similarity as ssim
from joblib import Parallel, delayed
import numpy as np

from PIL import Image
import matplotlib.pyplot as plt
import cv2

# Function to load and preprocess images
def load_image_small_grayscale(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    image = cv2.resize(image, (128, 128))  # Resize for uniformity
    return image

# Compute SSIM between one image and all others
def compute_ssim_for_image(index, images):
    n_images = len(images)
    ssim_values = np.zeros(n_images)
    print(index)
    for j in range(index + 1, n_images):
        ssim_values[j] = ssim(images[index], images[j])
    return ssim_values

def compute_ssim_matrix(images):
    # Parallel computation for all images
    n_images = len(images)
    results = Parallel(n_jobs=-1)(delayed(compute_ssim_for_image)(i, images) for i in range(n_images))
    # Combine results into a full SSIM matrix
    ssim_matrix = np.zeros((n_images, n_images))
    for i, row in tqdm(enumerate(results)):
        ssim_matrix[i, i+1:] = row[i+1:]
        ssim_matrix[i+1:, i] = row[i+1:]

    return ssim_matrix


def select_diverse_subset(ssim_matrix, subset_size, start_index=-1):
    dissimilarity_matrix = 1 - ssim_matrix
    n_images = dissimilarity_matrix.shape[0]
    selected_indices = []
    remaining_indices = list(range(n_images))

    if start_index == -1:
        start_index = np.random.randint(n_images)

    # Start with a random image
    selected_indices.append(remaining_indices.pop(start_index))

    for _ in tqdm(range(subset_size-1), 'finding optimal subset', total=subset_size):
        # Find the image that maximizes the minimum dissimilarity
        max_dissimilarity = -np.inf
        best_candidate = -1

        for candidate in remaining_indices:
            avg_dissimilarity = np.mean(
                [dissimilarity_matrix[candidate, selected] for selected in selected_indices]
            )
            if avg_dissimilarity > max_dissimilarity:
                max_dissimilarity = avg_dissimilarity
                best_candidate = candidate

        # Add the best candidate to the subset
        selected_indices.append(best_candidate)
        remaining_indices.remove(best_candidate)

    return selected_indices


def plot_image_subset(im_paths_subset):
    # List of image paths

    # Number of images
    n_images = 10

    # Set the number of rows and columns for the plot grid
    n_cols = int(n_images**0.5) + 1  # Calculate roughly square grid
    n_rows = (n_images // n_cols) + (n_images % n_cols > 0)

    # Create a figure
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 15))
    axes = axes.ravel()  # Flatten the 2D array of axes for easy indexing

    # Plot each image
    for (i, path),_ in zip(enumerate(im_paths_subset), range(n_images)):
        image = cv2.imread(path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert from BGR to RGB
        axes[i].imshow(image)
        axes[i].axis("off")
        axes[i].set_title(f"Image {i+1}")

    # Hide unused subplots
    for j in range(i + 1, len(axes)):
        axes[j].axis("off")

    # Adjust layout
    plt.tight_layout()
    plt.show()


def main():
    ...

if __name__ == "__main__":
    out_path_base = "/home/matt/programming/recon_diffuser/"
    dataset = "deeprecon"
    # path_features = f"{dataset}_train_features_clip_dreamsim.npz"
    path_features = "deeprecon_train_features_clip_dreamsim.npz"
    im_suffix = "JPEG"
    im_idx, all_input_filepaths = get_all_input_imgs(out_path_base, dataset, im_suffix)
    npz = np.load(path_features)
    captions = pd.read_csv("amt_20181204.csv")
    captions_id1 = captions[captions["counter"] == 1]
    captions_id1.index=captions_id1["content_id"]

    captions_for_annot = captions_id1.loc[im_idx]["caption"].values
    images = [load_image_small_grayscale(path) for path in all_input_filepaths]

    ssim_matrix = compute_ssim_matrix(images)

    dropout_ratios = [0.1, 0.25, 0.5]
    subset_sizes = []
    for dropout_ratio in dropout_ratios:
        subset_sizes.append(int(np.round(len(images)*dropout_ratio)))
    
    one_subset = select_diverse_subset(ssim_matrix, sample_size, start_index=-1)

    num_samples = 5

    for sample_size, dropout_ratio in zip(subset_sizes, dropout_ratios):
        for sample_n in range(num_samples):        
            # Select diverse subset
            selected_indices = select_diverse_subset(ssim_matrix, sample_size, start_index=-1)

            # Get the corresponding image paths
            selected_image_paths = [all_input_filepaths[i] for i in selected_indices]
            sample = np.array([im_idx[i] for i in selected_indices])

            path_dropout_samples = os.path.join(out_path_base, f'data/dropout_samples/{dataset}/')
            # os.makedirs(path_dropout_samples, exist_ok=True)
            # sample_filename = f"dropout-ssimgreedy_{dropout_ratio}_{sample_n:0>2}"
            # np.save(os.path.join(path_dropout_samples, sample_filename),sample)



