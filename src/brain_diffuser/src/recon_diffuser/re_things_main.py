"""
Serves as the main entrypoint for the whole pipeline computation for the things dataset.

Attributes:
    sub (str): The subject ID from the things dataset.
    out_path_base (str): The base for the output of the scripts. Intermediate results and models will be saved here.
    dataset (str): Should be 'things'. This is used by the other scripts to create the files at the correct folders.
    things_data_root (str): The base folder where the things fmri input data is stored.
    image_data_root (str): The base folder where the image data for the things stimuli is stored.
    captions_data_root (str): The base folder where the captions for the things images is stored.
"""
print("hi")

import os

from re_prepare_things_data import prepare_things_data_main
from re_vdvae import vdvae_main
from re_clip import clip_main
from re_vd_reconstruct_images import vd_reconstruct_main
from re_utils import simple_log


def main(
        out_path_base, things_fmri_data_root, things_images_root, things_captions_path, dataset, sub
        ):
    simple_log(f"Working on {dataset} subject {sub}")
    simple_log(f"Preparing things data")
    prepare_things_data_main(out_path_base, things_fmri_data_root, things_images_root, things_captions_path, dataset, sub)
    simple_log(f"Doing VDVAE")
    vdvae_main(out_path_base, dataset, sub, include_art_dataset=False)
    simple_log(f"Clipping")
    clip_main(out_path_base, dataset, sub, include_art_dataset=False)
    simple_log(f"Reconstructing")
    vd_reconstruct_main(out_path_base, dataset, sub, include_art_dataset=False)

if __name__ == "__main__":
    subs = ['01', '02', '03']
    sub = subs[1]
    out_path_base = "/home/matt/programming/recon_diffuser/"
    dataset="things"

    things_fmri_data_root = "/home/share/data/fmri_shared/datasets/THINGS"
    things_images_root = "/home/kiss/data/contents_shared/THINGS-fMRI1/source"
    things_captions_path = "/home/share/data/contents_shared/THINGS-fMRI1/derivatives/captions/BLIP_ViT_large/image_captions_KS231012.csv"

    main(out_path_base, things_fmri_data_root, things_images_root, things_captions_path, dataset, sub)



