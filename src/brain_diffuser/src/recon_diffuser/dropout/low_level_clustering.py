from dropout_random import get_all_input_imgs
import numpy as np
import cv2
import numpy as np
from sklearn.preprocessing import StandardScaler
import umap
from tqdm import tqdm
import base64
import pandas as pd
import plotly.express as px
import io
import seaborn as sns
import os
from dash import Dash, dcc, html, Input, Output, no_update, callback
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

from PIL import Image


def make_okay_plot(umap_features, title=None, save_path=None):
    df = pd.DataFrame(umap_features, columns=["UMAP1", "UMAP2"])
    indices, clusters, centroids = get_clusters_and_indices(umap_features, 0.0166) # around 20 clusters
    palette = plt.get_cmap('tab20').colors
    colors = [palette[cin] for cin in clusters]

    sizes = [100 if n in indices else 0.001 for n in range(len(clusters))]

    x = df["UMAP1"].values
    y = df["UMAP2"].values

    x_chosen = x[indices]
    y_chosen = y[indices]
    colors_chosen = np.clip(np.array(colors)[indices] - 0.25, 0,1)
    plt.scatter(x, y, color=colors, s=5, alpha=0.5)
    plt.scatter(x_chosen, y_chosen, color=colors_chosen, s=200, marker="+")
    plt.scatter(centroids[:,0], centroids[:,1], color=colors_chosen)
    plt.xlabel("UMAP1")
    plt.ylabel("UMAP2")

    if title:
        plt.title(title)
    if save_path:
        plt.savefig(save_path)
    plt.clf()

def make_okay_plot_all_umaps(umap_pixels, umap_dreamsim, umap_clipvision, save_path):
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.8))

    for i,umap_features in enumerate([umap_pixels, umap_dreamsim, umap_clipvision]):
        ax = axes[i]
        # activate ax
        plt.sca(ax)
        df = pd.DataFrame(umap_features, columns=["UMAP1", "UMAP2"])
        indices, clusters, centroids = get_clusters_and_indices(umap_features, 0.0166) # around 20 clusters
        palette = plt.get_cmap('tab20').colors
        colors = [palette[cin] for cin in clusters]

        sizes = [100 if n in indices else 0.001 for n in range(len(clusters))]

        x = df["UMAP1"].values
        y = df["UMAP2"].values

        x_chosen = x[indices]
        y_chosen = y[indices]
        colors_chosen = np.clip(np.array(colors)[indices] - 0.25, 0,1)
        plt.scatter(x, y, color=colors, s=5, alpha=0.5)
        plt.scatter(x_chosen, y_chosen, color=colors_chosen, s=200, marker="+")
        plt.scatter(centroids[:,0], centroids[:,1], color=colors_chosen)
        ax.set_xlabel("UMAP1", fontsize=18)
        if i > 0:
            # plt.yticks([])
            plt.ylabel("")
        else:
            
            ax.set_ylabel("UMAP2", fontsize=18)
        ax.set_title(["Pixel-Space", "DreamSim-Space", "CLIP Vision-Space"][i], fontsize=20)
        # plt.title()

    # fig.suptitle("UMAP + k-means of Pixel-Space, Dreamsim and Clipvision Space", fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.show()
    plt.clf()


def make_nice_plot(umap_features, all_input_filepaths, captions_for_annot, name):
    df = pd.DataFrame(umap_features, columns=["UMAP1", "UMAP2"])
    df['global_index'] = range(len(df))
    indices, clusters, centroids = get_clusters_and_indices(umap_features, 0.0166) # around 20 clusters
    palette = plt.get_cmap('tab20').colors
    colors = [palette[cin] for cin in clusters]

    x = df["UMAP1"].values
    y = df["UMAP2"].values

    fig = px.scatter(data_frame= df, x="UMAP1", y="UMAP2", title=f"{name}", color=colors, custom_data="global_index" )
    # fig = px.scatter(x=x, y=y, title=f"{name}")

    fig.update_layout(showlegend=False)
    fig.update_traces(
        hoverinfo="none",
        hovertemplate=None,
        marker=dict(size=5))
    # Set up the app now
    app = Dash(__name__)
    app.layout = html.Div(
        className="container",
        children=[
            dcc.Graph(id="graph-2-dcc", figure=fig, clear_on_unhover=True),
            dcc.Tooltip(id="graph-tooltip-2", direction='bottom'),],)
    @callback(
        Output("graph-tooltip-2", "show"),
        Output("graph-tooltip-2", "bbox"),
        Output("graph-tooltip-2", "children"),
        Output("graph-tooltip-2", "direction"),
        Input("graph-2-dcc", "hoverData"),
    )
    def display_hover(hoverData):
        if hoverData is None:
            return False, no_update, no_update, no_update
        # Load image with pillow
        hover_data = hoverData["points"][0]
        num = hover_data["customdata"][0]  # Retrieve the global index
        # print(hover_data)
        # num = 1 
        # print(hover_data)
        # num = hover_data["pointNumber"]
        # print(num)
        image_path = all_input_filepaths[num]
        im = Image.open(image_path).resize((100,100))
        cap = str(captions_for_annot[num])
        # dump it to base64
        buffer = io.BytesIO()
        im.save(buffer, format="jpeg")
        encoded_image = base64.b64encode(buffer.getvalue()).decode()
        im_url = "data:image/jpeg;base64, " + encoded_image
        # demo only shows the first point, but other points may also be available
        hover_data = hoverData["points"][0]
        bbox = hover_data["bbox"]
        # control the position of the tooltip
        y = hover_data["y"]
        direction = "bottom" if y > 1.5 else "top"
        children = [html.Img(src=im_url,style={"width": "150px"},),
                    html.P(cap)]
        return True, bbox, children, direction

    app.run(debug=True)


def get_clusters_and_indices(ump, dropout_ratio):

    n_images = int(ump.shape[0] * dropout_ratio)
    kmeans = KMeans(n_clusters=n_images, n_init="auto").fit(ump)

    centroids = []
    picked_indices = []
    for n_cluster in range(n_images):
        cluster_n_indices = [n for n, label in enumerate(kmeans.labels_) if label == n_cluster]
        cluster_n_coordinates = ump[cluster_n_indices]
        cluster_n_centroid = np.mean(cluster_n_coordinates, axis=0)
        centroids.append(cluster_n_centroid)

        distances_to_centroid = np.linalg.norm(cluster_n_coordinates-cluster_n_centroid, axis=1)
        idx_smallest_distance = distances_to_centroid.argmin()
        picked_indices.append(int(cluster_n_indices[idx_smallest_distance]))
    return (np.array(picked_indices), kmeans.labels_, np.array(centroids))
        
def compute_dropout_sample(out_path_base, dataset, sample_size, sample_n, im_suffix, umap_name, indexes, npz_im_idx):
    # take a random sample from the image indexes
    sample = np.array(npz_im_idx)[indexes]
    print(len(sample))
    print(sample)
    path_dropout_samples = os.path.join(out_path_base, f'data/dropout_samples/{dataset}/')
    os.makedirs(path_dropout_samples, exist_ok=True)
    sample_filename = f"dropout-{umap_name}_{sample_size}_{sample_n:0>2}"
    sample_path = os.path.join(path_dropout_samples, sample_filename)
    np.save(sample_path,sample)
    print(f'Saved to {sample_path}')

def main_deeprecon(reducer, umap_name, im_idx, ump_data):
    print(f"Main for {umap_name}")
    out_path_base = "/Users/matt/programming/recon_diffuser/"
    dataset = "deeprecon"
    im_suffix = "JPEG"

    sample_sizes = [0.95, 0.9, 0.75, 0.5, 0.25, 0.1]
    sample_sizes = [0.25]
    n_samples = 9
    for sample_size in sample_sizes:
        print(sample_size)
        for sample_n in range(n_samples):
            print(sample_n)
            ump = reducer.fit_transform(ump_data)
            indexes, clusters, centroids = get_clusters_and_indices(ump, sample_size)
            compute_dropout_sample(out_path_base, dataset, sample_size, sample_n, im_suffix, umap_name, indexes, im_idx)


def sort_im_idx(im_idx_correct_order, im_idx_wrong_order, im_paths_wrong_order):
    # make sure the input_filepaths have the same ordering than the npz_images
    indices = [im_idx_wrong_order.index(i) for i in im_idx_correct_order]
    im_idx_sorted = [im_idx_wrong_order[i] for i in indices]

    if not np.all(im_idx_sorted == im_idx_correct_order):
        raise ValueError("Warning: The image indexes are not the same after sorting")

    all_input_filepaths = [im_paths_wrong_order[i] for i in indices]
    return im_idx_sorted, all_input_filepaths

def images_to_pixel_features(all_images):
    all_images_resized = np.array([np.array(im.resize((256,256))) for im in all_images])
    all_images_flat = all_images_resized.reshape(all_images_resized.shape[0], np.prod(all_images_resized.shape[1:]))
    return all_images_flat

def similarity_plot(all_input_filepaths, umap_dreamsim, umap_pixels, umap_clipvision, save_name=None, test_indices=None):

    # 1. Pick 10 random images from the test data
    if test_indices is None:
        test_indices = np.random.choice(len(all_input_filepaths), 10, replace=False)
    print(test_indices)
    test_images = [all_input_filepaths[i] for i in test_indices]

    # 2. find the image that is closest to the image in each feature space
    closest_images = {"dreamsim": [], "pixels": [], "clipvision": []}
    feature_spaces = {
        "dreamsim": umap_dreamsim,
        "pixels": umap_pixels,
        "clipvision": umap_clipvision,
    }

    for idx in test_indices:
        # original_image = umap_pixels[idx]  # Using pixel space as reference
        for name, features in feature_spaces.items():
            original_image_features = features[idx]
            # raise Exception
            distances = np.linalg.norm(features - original_image_features, axis=1)
            closest_idx = np.argsort(distances)
            second_closest_idx = closest_idx[1]
            closest_images[name].append(all_input_filepaths[second_closest_idx])

    # 3. Plot the images next to each other. 10 rows, 4 columns (in the rows are the 10 images, in the columns are the real images and the 3 feature spaces)

    # Create the subplots: 10 rows, 4 columns
    fig, axes = plt.subplots(10, 4, figsize=(20, 40))

    # Define the column titles
    column_titles = ["Original"] + list(closest_images.keys())

    # Set the titles on the first row (and increase font size)
    for col, title in enumerate(column_titles):
        axes[0, col].set_title(title, fontsize=50)

    # Plot the images without setting a title for every subplot
    for i, idx in enumerate(test_indices):
        # Original image (first column)
        original_image = Image.open(all_input_filepaths[idx]).resize((250, 250))
        axes[i, 0].imshow(original_image)
        axes[i, 0].axis("off")
        
        # Closest images in each feature space (remaining columns)
        for j, (name, images) in enumerate(closest_images.items()):
            closest_image = Image.open(images[i]).resize((250, 250))
            axes[i, j + 1].imshow(closest_image)
            axes[i, j + 1].axis("off")

    plt.tight_layout()
    if save_name:
        plt.savefig(save_name,dpi=50)
    # plt.show()


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

    captions = pd.read_csv("amt_20181204.csv")
    captions_id1 = captions[captions["counter"] == 1]
    captions_id1.index=captions_id1["content_id"]

    captions_for_annot = captions_id1.loc[im_idx_sorted]["caption"].values


    # # # dreamsim
    reducer_dreamsim = umap.UMAP(n_neighbors=15, min_dist=0.1, n_components=2)
    umap_dreamsim = reducer_dreamsim.fit_transform(npz["features_dreamsim"])
    # # make_nice_plot(umap_dreamsim, all_input_filepaths, captions_for_annot, "dreamsim")
    # # make_okay_plot(umap_dreamsim, "dreamsim")
    main_deeprecon(reducer_dreamsim, "dreamsim", npz_im_idx, npz["features_dreamsim"])

    # # # # Pixel space
    from tqdm import tqdm
    all_images = []
    for f in tqdm(all_input_filepaths):
        all_images.append(Image.open(f).convert('RGB'))

    all_images_flat = images_to_pixel_features(all_images)

    reducer_pixels = umap.UMAP(n_neighbors=15, min_dist=0.1, n_components=2)
    umap_pixels = reducer_pixels.fit_transform(all_images_flat)
    # # make_nice_plot(umap_pixels, all_input_filepaths, captions_for_annot, "pixels")
    # # make_okay_plot(umap_pixels, 'pixels')
    main_deeprecon(reducer_pixels, "pixels", npz_im_idx, all_images_flat)

    reducer_clipvision = umap.UMAP(n_neighbors=15, min_dist=0.1, n_components=2)
    umap_clipvision = reducer_clipvision.fit_transform(npz["features_clipvision"])
    # # make_nice_plot(umap_clipvision, all_input_filepaths, captions_for_annot, "clipvision")
    # # make_okay_plot(umap_clipvision, 'clipvision')
    main_deeprecon(reducer_clipvision, "clipvision", npz_im_idx, npz["features_clipvision"])


    # ### Thesis plots
    thesis_plots_path = "/Users/matt/ownCloud/gogo/MA/thesis/diversity_thesis/plots"
    title_dreamsim = "UMAP of dreamsim features"
    title_pixels = "UMAP of Pixel features"
    title_clipvision = "UMAP of CLIP Vision features"
    # make_okay_plot(umap_dreamsim, title_dreamsim, os.path.join(thesis_plots_path, "dropout_umap_dreamsim.png"))
    # make_okay_plot(umap_pixels, title_pixels, os.path.join(thesis_plots_path, "dropout_umap_pixels.png"))
    # make_okay_plot(umap_clipvision, title_clipvision, os.path.join(thesis_plots_path, "dropout_umap_clipvision.png"))
    make_okay_plot_all_umaps(umap_pixels, umap_dreamsim, umap_clipvision, os.path.join(thesis_plots_path, "dropout_umap.png"))

    # Similarity plot
    save_name = os.path.join(thesis_plots_path, "dropout_similarity_plot_OLD.JPEG")
    similarity_plot(all_input_filepaths, umap_dreamsim, umap_pixels, umap_clipvision, save_name=save_name)