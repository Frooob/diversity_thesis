"""
Serves as the main entrypoint for the whole pipeline computation for the deeprecon dataset.

Attributes:
    sub (str): The subject ID from the deeprecon dataset.
    out_path_base (str): The base for the output of the scripts. Intermediate results and models will be saved here.
    dataset (str): Should be 'deeprecon'. This is used by the other scripts to create the files at the correct folders.
    deeprecon_data_root (str): The base folder where the deeprecon fmri input data is stored.
    image_data_root (str): The base folder where the image data for the deeprecon stimuli is stored.
    captions_data_root (str): The base folder where the captions for the deeprecon images is stored.
"""
import os
from datetime import datetime
import logging

from re_prepare_deeprecon_data import prepare_deeprecon_data_main
from re_vdvae import vdvae_main
from re_clip import clip_main
from re_vd_reconstruct_images import vd_reconstruct_main
from re_utils import simple_log

logger = logging.getLogger("recdi_deeprecon_main")

def main(
        out_path_base, deeprecon_data_root, image_data_root, captions_data_root, dataset, sub, compute_train_mean=True
        ):

    if not compute_train_mean:
        output_name = "_noinputavg"
        logger.info(f"Setting output name to {output_name}")
    else:
        output_name = ""

    simple_log(f"Working on {dataset} subject {sub}")
    simple_log(f"Preparing deeprecon data")
    prepare_deeprecon_data_main(out_path_base, deeprecon_data_root, image_data_root, captions_data_root, dataset, sub, compute_train_mean=compute_train_mean)
    simple_log(f"Doing VDVAE")
    vdvae_main(out_path_base, dataset, sub, include_art_dataset=True)
    simple_log(f"Clipping")
    clip_main(out_path_base, dataset, sub, include_art_dataset=True, output_name=output_name)
    simple_log(f"Reconstructing")
    vd_reconstruct_main(out_path_base, dataset, sub, include_art_dataset=True, output_name=output_name)

if __name__ == "__main__":
    subs = ['KS', 'AM', 'ES', 'JK', 'TH']
    sub = "AM"
    out_path_base = "/home/matt/programming/recon_diffuser/"
    dataset="deeprecon"

    fname_log = f"{datetime.now()}_recon_deeprecon_{sub}_{dataset}.log"

    logging.basicConfig(
        format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=[
            logging.StreamHandler()
        ],
        level=logging.INFO)
    
    logger.info(f"Starting deeprecon main with {dataset=} {sub=}")
    
    deeprecon_data_root = "/home/kiss/data/fmri_shared/datasets/Deeprecon/fmriprep"
    image_data_root = "/home/kiss/data/contents_shared"
    captions_data_root = "/home/matt/programming/recon_diffuser/data/annots"

    main(out_path_base, deeprecon_data_root, image_data_root, captions_data_root, dataset, sub)



