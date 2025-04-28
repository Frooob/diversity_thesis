"""
I want to validate, if the dropout actually works the way I think it does.

Meaning: 
compared to the random dropout condition, I want to make sure that in the clip space, the variance of the dropout condition is higher than the random dropout condition.

"""

import os
import numpy as np
import pandas as pd
from dropout_random import get_all_input_imgs
from low_level_clustering import sort_im_idx, images_to_pixel_features
import numba
import numpy as np
import seaborn as sns
from numba import prange
import matplotlib.pyplot as plt
sns.set_theme()


@numba.njit(parallel=True)
def average_min_distance_to_subsample(
    subsample_features,
    all_im_features
):
    n_all = all_im_features.shape[0]
    n_sub = subsample_features.shape[0]
    if n_sub == 0 or n_all == 0:
        return 0.0
    local_min_dists = np.zeros(n_all, dtype=np.float64)
    # Parallelize over 'i'
    for i in prange(n_all):
        min_dist_sq = 1e20
        # Find the closest image in the subsample
        for j in range(n_sub):
            dist_sq = 0.0
            for k in range(all_im_features.shape[1]):
                diff = all_im_features[i, k] - subsample_features[j, k]
                dist_sq += diff * diff
            if dist_sq < min_dist_sq:
                min_dist_sq = dist_sq
        # Store the minimum distance (take sqrt at the end)
        local_min_dists[i] = np.sqrt(min_dist_sq)
    # Now compute the mean of all minimum distances
    return local_min_dists.mean()


# open the deeprecon_train_features.npz
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
im_idx = im_idx_sorted

# Get the captions for fun
captions = pd.read_csv("amt_20181204.csv")
captions_id1 = captions[captions["counter"] == 1]
captions_id1.index=captions_id1["content_id"]
captions_for_annot = captions_id1.loc[im_idx_sorted]["caption"].values


# load all images into memory
from PIL import Image
from tqdm import tqdm

all_images = []
for f in tqdm(all_input_filepaths):
    all_images.append(Image.open(f).convert('RGB'))

dir_dropout_samples = "/Users/matt/programming/recon_diffuser/data/dropout_samples/deeprecon/"
file_dropout_sample = "dropout-dreamsim_0.1_00.npy"
path_dropout_sample = os.path.join(dir_dropout_samples, file_dropout_sample)

dropout_sample = np.load(path_dropout_sample)

if os.path.exists('variance_results.csv'):
    df_variance_results = pd.read_csv("variance_results.csv")
else:

    features_dreamsim = npz['features_dreamsim'].astype(np.float32)
    features_clipvision = npz['features_clipvision'].astype(np.float32)
    features_pixels = images_to_pixel_features(all_images).astype(np.float32)


    dreamsim_samples = []
    variance_results = []
    for f in tqdm(os.listdir(dir_dropout_samples)):
        if not f.endswith(".npy"):
            continue
        variance_result = {}
        parts_dropout = f.split("_")
        dropout_algorithm = parts_dropout[0]
        # if dropout_algorithm == 'dropout-pixels':
        #     continue
        dropout_sample_size = float(parts_dropout[1])
        if dropout_sample_size != 0.25:
            continue
        
        dropout_sample_n = int(parts_dropout[2].split(".")[0])
        # if dropout_algorithm != 'dropout-random' and dropout_sample_n != 0:
        #     continue
        if dropout_sample_n > 9:
            print("A chosen one speaks to us.")  # numbering above 10 are the chosen subsets for the all subject analysis
            continue

        dropout_sample = np.load(os.path.join(dir_dropout_samples, f))
        dreamsim_samples.append(set([str(s) for s in dropout_sample]))

        idx_sample = [np.argwhere(npz_im_idx == image_tag).item() for image_tag in dropout_sample]

        features_sample_dreamsim = features_dreamsim[idx_sample].astype(np.float32)
        features_sample_clipvision = features_clipvision[idx_sample].astype(np.float32)
        features_sample_pixels = features_pixels[idx_sample].astype(np.float32)

        avg_min_distance_dreamsim = average_min_distance_to_subsample(features_sample_dreamsim, features_dreamsim)
        variance_result["avg_min_distance_dreamsim"] = avg_min_distance_dreamsim
        print("Computed dreamsim")
        avg_min_distance_clipvision = average_min_distance_to_subsample(features_sample_clipvision, features_clipvision)
        variance_result["avg_min_distance_clipvision"] = avg_min_distance_clipvision
        print("Computed clipvision")
        avg_min_distance_pixels = average_min_distance_to_subsample(features_sample_pixels, features_pixels)
        variance_result["avg_min_distance_pixels"] = avg_min_distance_pixels
        print("Computed Pixels")

        variance_result["dropout_algorithm"] = dropout_algorithm
        variance_result["dropout_sample_size"] = dropout_sample_size
        variance_result["dropout_sample_n"] = dropout_sample_n
        
        variance_results.append(variance_result)

    df_variance_results = pd.DataFrame(variance_results)

    df_variance_results.to_csv("variance_results.csv")

metric = "avg_min_distance"

thesis_plots_path = "/Users/matt/ownCloud/gogo/MA/thesis/diversity_thesis/plots"

def create_avg_min_distance_plot(data, space, thesis_plots_path=None):
    xticks = ['dreamsim', 'pixels', 'clipvision', 'random']
    ax = sns.boxplot(data=data.query('dropout_sample_size == 0.25'), x="dropout_algorithm", y=f"avg_min_distance_{space}")
    ax.set_title(f"Avg Min Distance in {space}-Space")
    ax.set_xlabel('Dropout Algorithm')
    ax.set_ylabel('Average Min Distance')
    ax.set_xticklabels(xticks)
    if thesis_plots_path:
        plt.savefig(os.path.join(thesis_plots_path, f"dropout_avg_min_distance_{space}.png"))
    plt.show()
    
def set_ticks_and_axes(ax, xticklabels, title, ylabel, sethline=False):
    ax.set_xticks(range(len(xticklabels)))  # Ensure it aligns with categorical data
    ax.set_xticklabels(xticklabels, size=16)
    ax.yaxis.get_label().set_fontsize(17)
    ax.tick_params(axis='x', labelrotation=45)
    ax.xaxis.set_tick_params(pad=-5)
    ax.set_title(title, size=18)
    ax.set(ylabel=ylabel)
    if sethline:
        ax.axhline(0.5, color='rosybrown', linestyle='-', linewidth=1, alpha=0.8)

def create_avg_min_distance_plot_all_spaces(df_variance_results, thesis_plots_path):
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    spaces_titles = ['DreamSim', 'CLIP', 'Pixels']
    spaces = ["dreamsim", "clipvision", "pixels"] 

    df = df_variance_results.copy()
    dropout_algorithm_renamer = {'dropout-random': 'random','dropout-pixels': 'Pixels', 'dropout-clipvision': 'CLIP','dropout-dreamsim': 'DreamSim'}

    df['dropout_algorithm'] = df['dropout_algorithm'].replace(dropout_algorithm_renamer)

    for i, space in enumerate(spaces):
        ax = axs[i]
        plt.sca(ax)
        title = f"Distance in {spaces_titles[i]}-Space"
        set_ticks_and_axes(ax, dropout_algorithm_renamer.values(), title, "", sethline=False)
        sns.boxplot(data=df.query('dropout_sample_size == 0.25'), x="dropout_algorithm", y=f"avg_min_distance_{space}", ax=axs[i], order = dropout_algorithm_renamer.values())
        ax.set_xlabel('Dropout Strategy', fontweight='bold', size=16)
        if i == 0:
            ax.set_ylabel('Average Min Distance', fontweight = 'bold')
        else:
            ax.set_ylabel('')
        ax.set_xticklabels(dropout_algorithm_renamer.values())
    # fig.suptitle('Avg-min-distance of subsample to All Training Images',fontweight='bold')
    plt.savefig(os.path.join(thesis_plots_path, "dropout_avg_min_distance.png"), dpi=300, bbox_inches='tight')
    # plt.clf()
    plt.show()

create_avg_min_distance_plot_all_spaces(df_variance_results, thesis_plots_path)

best_dreamsim_subset = df_variance_results[df_variance_results['avg_min_distance_dreamsim'] == df_variance_results['avg_min_distance_dreamsim'].min()]
print(f"dreamsim best sample: {best_dreamsim_subset['dropout_sample_n'].item()}")

best_clipvision_subset = df_variance_results[df_variance_results['avg_min_distance_clipvision'] == df_variance_results['avg_min_distance_clipvision'].min()]
best_clipvision_subset
print(f"clipvision best sample: {best_clipvision_subset['dropout_sample_n'].item()}")

best_pixels_subset = df_variance_results[df_variance_results['avg_min_distance_pixels'] == df_variance_results['avg_min_distance_pixels'].min()]
print(f"pixels best sample: {best_pixels_subset['dropout_sample_n'].item()}")
