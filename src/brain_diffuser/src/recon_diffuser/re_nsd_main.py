"""
Serves as the main entrypoint for the whole pipeline computation for the nsd dataset.

Attributes:
    sub (str): The subject number, should be one of [1,2,5,7] since these are the participants in the NSD 
        dataset that have completed all test images.
    out_path_base (str): The base for the output of the scripts. Intermediate results and models will be saved here.
    dataset (str): Should be 'nsd'. This is used by the other scripts to create the files at the correct folders.
    nsd_data_root (str): The base folder where the nsd input data is stored.
"""


from re_utils import simple_log
simple_log("Importing libraries.")
from re_prepare_nsd_data import prepare_nsd_data
from re_vdvae import vdvae_main
from re_clip import clip_main
from re_vd_reconstruct_images import vd_reconstruct_main

def main(out_path_base, nsd_data_root, dataset, sub):
    simple_log(f"Working on {dataset} subject {sub}")
    simple_log(f"Preparing nsd data")
    prepare_nsd_data(nsd_data_root, out_path_base, sub)
    simple_log(f"Doing VDVAE")
    vdvae_main(out_path_base, dataset, sub, include_art_dataset=False)
    simple_log(f"Clipping")
    clip_main(out_path_base, dataset, sub, include_art_dataset=False)
    simple_log(f"Reconstructing")
    vd_reconstruct_main(out_path_base, dataset, sub, include_art_dataset=False)

if __name__ == "__main__":
    subs = ['1', '2', '5', '7']
    sub = "7"
    out_path_base = "/home/matt/programming/recon_diffuser/"
    dataset="nsd"

    nsd_data_root = "/home/matt/diffusing/brain-diffuser/data"

    main(out_path_base, nsd_data_root, dataset, sub)




