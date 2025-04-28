import pandas as pd
import os
import logging
from datetime import datetime
from re_clip import clip_main
from re_vd_reconstruct_images import vd_reconstruct_main
from re_vdvae import vdvae_main
from re_utils import load_clip_model
from sklearn.metrics import r2_score

logger = logging.getLogger("recdi_dropmain")


def im_idx_is_prepared(out_path_base, dataset, sub):
    """ Checks if all neccesary data is prepared for the perturbed data reconstruction. """
    data_is_prepared = True
    
    names_to_check = ['train']
    # make sure fmri data and captions are prepared
    for name in names_to_check:
        im_idx_path = os.path.join(out_path_base, 'data', 'processed_data', dataset, f"subj{sub}", f'{name}_im_idx_sub{sub}.npy')
        if not os.path.exists(im_idx_path):
            logger.error(f"Im idx is not prepared for subject {sub}")
            data_is_prepared = False

    return data_is_prepared


def main(out_path_base, dataset, sub, output_name):

    if dataset == 'deeprecon':
        include_art_dataset = True

    logger.info(f"Working on {dataset} subject {sub}")
    logger.info(f"Checking if perturbed data is prepared...")
    if not im_idx_is_prepared(out_path_base, dataset, sub):
        raise AssertionError(f"Im idx is not prepared for subject. Please run the base main script first for dataset {dataset} (e.g. deeprecon_main or nsd_main or things_main).")

    logger.info(f"VDVAE")
    vdvae_main(out_path_base, dataset, sub, include_art_dataset=True, output_name=output_name)

    clip_model = load_clip_model()

    logger.info(f"Clipping")
    clip_main(out_path_base, dataset, sub, include_art_dataset, output_name, clip_model)

    logger.info(f"Reconstructing")
    vd_reconstruct_main(out_path_base, dataset, sub, include_art_dataset=include_art_dataset, output_name=output_name, clip_model=clip_model)



if __name__ == "__main__":
    sub = "AM"
    out_path_base = "/home/matt/programming/recon_diffuser/"
    dataset="deeprecon"
    output_name = "dropout-random_0.1_00"
    
    logging.basicConfig(
        format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=[
            logging.StreamHandler()
        ],
        level=logging.INFO)
    
    main(out_path_base, dataset, sub, output_name)

    