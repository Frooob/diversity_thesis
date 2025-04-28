import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import logging
from datetime import datetime

def simple_log(l):
    print(f"{datetime.now()}: {l}")


def visualize_results(out_path_base, dataset, sub, name, result_names="", include_vdvae=True, num_images=10, start_img_idx=0, plot_folder='plots/', show_img=False, caption_names = None, title = None, true_features=False, only_vdvae=False, set_save_name="", set_save_folder="", row_label_size = 25, save_dpi=200, save_format="png", sample_indices= None ):

    if not result_names:
        result_names = [name]
    elif type(result_names) == str:
        result_names = [result_names]
    
    result_names_str = "-".join(result_names)

    simple_log(f'Starting visualize results for  {dataset, sub, result_names}')
    path_processed = os.path.join(out_path_base, f'data/processed_data/{dataset}/subj{sub}/')
    stim_path = os.path.join(path_processed, f'{name}_stim_sub{sub}.npy')

    if true_features:
        vdvae_result_folder = os.path.join(out_path_base, f"results/vdvae/{dataset}/subj{sub}/{name}_true_bd")
    else:
        if only_vdvae:
            vdvae_result_folders = [os.path.join(out_path_base, f"results/vdvae/{dataset}/subj{sub}/{result_name}/") for result_name in result_names]
            
        else:
            vdvae_result_folder = os.path.join(out_path_base, f"results/vdvae/{dataset}/subj{sub}/{name}/")
    

    stim = np.load(stim_path)
    indices = list(range(len(stim)))

    # sample_indices = np.random.choice(indices, num_images, replace=False)
    if sample_indices is None:
        sample_indices = np.arange(num_images)
        sample_indices += start_img_idx
    

    if include_vdvae:

        if only_vdvae:
            vdvae_images_pil = []
            for vdvae_result_folder in vdvae_result_folders:
                vdvae_images = np.array(sorted(os.listdir(vdvae_result_folder), key=lambda x: int(x.split('.')[0])))[sample_indices]
                vdvae_images_pil.append([Image.open(os.path.join(vdvae_result_folder, vdvae_image)) for vdvae_image in vdvae_images])

        else:
            vdvae_images = np.array(sorted(os.listdir(vdvae_result_folder), key=lambda x: int(x.split('.')[0])))[sample_indices]
            vdvae_images_pil = [[Image.open(os.path.join(vdvae_result_folder, vdvae_image)) for vdvae_image in vdvae_images]]


    reconstructed_result_folders = []
    for result_name in result_names:
        if result_name.startswith("icnn"):
            result_name = result_name[5:]
            reconstructed_result_folders.append(os.path.join(out_path_base, f"results/icnn/{dataset}/subj{sub}/{result_name}"))
        else:
            reconstructed_result_folders.append(os.path.join(out_path_base, f"results/versatile_diffusion/{dataset}/subj{sub}/{result_name}"))

    # reconstructed_result_folders = [os.path.join(out_path_base, f"results/versatile_diffusion/{dataset}/subj{sub}/{result_name}") for result_name in result_names]

    all_fnames = []
    all_diffusion_images = []
    for vd_res_folder in reconstructed_result_folders:
        all_image_files = [f for f in os.listdir(vd_res_folder) if f.endswith('png')]
        sorted_files = sorted(all_image_files, key=lambda x: int(x.split('.')[0]))
        diffusion_images = np.array(sorted_files)[sample_indices]
        all_fnames.append(diffusion_images)
        diffusion_images_pil = [Image.open(os.path.join(vd_res_folder, diffusion_image)) for diffusion_image in diffusion_images]
        all_diffusion_images.append(diffusion_images_pil)
    assert len(set([tuple(f) for f in all_fnames])) == 1  # make sure the same images are in each of the images...

    gr_images = stim[sample_indices]
    gr_images_pil = [Image.fromarray(np.uint8(img_data)) for img_data in gr_images]

    images = gr_images_pil 
    n_rows = 1

    if include_vdvae:
        for opened_vdvae_images in vdvae_images_pil:
            images = images + opened_vdvae_images
            n_rows += 1
    
    if not only_vdvae:
        for diffusion_image_set_pil in all_diffusion_images:
            images = images + diffusion_image_set_pil
            n_rows += 1

    # plotting 
    fig, axes = plt.subplots(n_rows, num_images, figsize=(num_images*3, n_rows*3.5))

    if not title:
        fig.suptitle(f"Image Comparison {dataset} {result_names_str} {sub} ", fontsize=30)
    else:
        fig.suptitle(title, fontsize=30)

    row_labels = ["Ground truth"]
    if include_vdvae and not caption_names:
      row_labels.append("VDVAE")
    
    if caption_names:
        row_labels += caption_names
    else:
        for num,result_name in enumerate(result_names):

            if result_name == name:
                row_labels.append("diffusion")
            else:
                row_labels.append(result_name)

    for i in range(n_rows):
        for j in range(num_images):
            img_index = i * num_images + j
            axes[i, j].imshow(images[img_index])
            axes[i, j].xaxis.set_visible(False)
            plt.setp(axes[i, j].spines.values(), visible=False)
            axes[i, j].tick_params(left=False, labelleft=False)
            axes[i, j].patch.set_visible(False)
            if j == 0:
                label = row_labels[i].replace("_size224_iter500_scaled", "")
                axes[i, j].set_ylabel(label,fontsize=row_label_size)

    plt.tight_layout()

    if set_save_name:
        save_name = f"{set_save_name}.{save_format}"
    elif not title:
        save_name = f"{dataset}_sub{sub}_img{start_img_idx}-{num_images+start_img_idx-1}_{result_names_str}__qual_eval_.{save_format}"
    else:
        save_name = f"{dataset}_sub{sub}_img{start_img_idx}-{num_images+start_img_idx-1}_{title}_qual_eval_.{save_format}"

    if set_save_folder:
        save_path = os.path.join(set_save_folder, save_name)
    else:
        save_path = os.path.join(plot_folder, save_name)

    if save_format == "png":
        plt.savefig(save_path, dpi=save_dpi)
    else:
        plt.savefig(save_path, dpi=save_dpi)

    simple_log(f'Saved qualitative images to {save_name}')
    if show_img:
        plt.show()
    else:
        plt.clf()



if __name__ == "__main__":
  # Example: sub: "AM", name: "art", dataset: "deeprecon"

  # sub = "1"
  # name = "test"
  # dataset = "nsd"

  sub = "AM"
  name = "test"
  dataset = "deeprecon"

  out_path_base = "/home/matt/programming/recon_diffuser/"
#   visualize_results(out_path_base, dataset, sub, name)
