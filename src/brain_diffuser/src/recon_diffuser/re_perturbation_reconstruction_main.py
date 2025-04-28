import os
import logging
from datetime import datetime
from re_clip import clip_main_perturbated
from re_vd_reconstruct_images import vd_reconstruct_main
import sys
sys.path.append('/home/matt/programming/recon_diffuser/src/recon_diffuser/adv_pert')
from clip_module_testing import get_im_idx_sub

logger = logging.getLogger("recdi_pertmain")

"""
okay, alles ist machbar, ich muss es nur Stück für Stück angehen. 
1. Alte Bilder entsprechend umbenennen - CHECK
2. In perturbations all images den output path verändern und dafür sorgen, dass die Bilder gemäß im_idx korrekt benannt werden - CHECK 
3. In perturbation_reconstruction_main.py eine prepare perturbation input data erstellen (dass die Bilder für die VPN etsprechend gematched werden)
4. Sollte eigentlich dann schon reichen oder?

"""

def prepare_perturbed_input_data(out_path_base, dataset, sub, output_name):
    if dataset != "deeprecon":
        raise NotImplementedError(f"Only deeprecon dataset can be used for perturbed input data")

    # First: need to load the fmri data for that subject

    deeprecon_data_root = "/home/kiss/data/fmri_shared/datasets/Deeprecon/fmriprep"
    fmri_data_train_path = f'{deeprecon_data_root}/{sub}_ImageNetTraining_volume_native.h5'
    from re_prepare_deeprecon_data import get_im_idx_sub
    im_idx = get_im_idx_sub(out_path_base, sub)



def perturbed_input_data_is_prepared(out_path_base, dataset, sub, output_name, include_art_dataset):
    """ Checks if all neccesary data is prepared for the perturbed data reconstruction. """
    data_is_prepared = True
    
    names_to_check = ['train', 'test', 'art'] if include_art_dataset else ['train', 'test']
    
    paths_to_check = []

    # make sure fmri data and captions are prepared
    for name in names_to_check:
        if name == 'train' and output_name.endswith("_noinputavg"):
            logger.info("Checking if no input average data has been prepared...")
            fmri_path = os.path.join(out_path_base, 'data', 'processed_data', dataset, f"subj{sub}", f"{name}_fmri_noavg_general_sub{sub}.npy")
        else:
            fmri_path = os.path.join(out_path_base, 'data', 'processed_data', dataset, f"subj{sub}", f"{name}_fmriavg_general_sub{sub}.npy")
        
        cap_path = os.path.join(out_path_base, 'data', 'processed_data', dataset, f"subj{sub}", f"{name}_cap_sub{sub}.npy")
        paths_to_check.append(fmri_path)
        paths_to_check.append(cap_path)
        if name != 'train':
            vdvae_path = os.path.join(out_path_base, 'results', 'vdvae', dataset, f"subj{sub}", name)
            paths_to_check.append(vdvae_path)

    ## make sure perturbed stimuli are prepared
    output_name = output_name.replace("_noinputavg", "")
    perturbed_folder = os.path.join(out_path_base, 'data', 'train_data', 'deeprecon_perturbed', output_name)
    paths_to_check.append(perturbed_folder)

    for path_to_check in paths_to_check:
        if not os.path.exists(path_to_check):
            logger.error(f"Perturbed input data isn't prepared: {path_to_check} is missing.")
            data_is_prepared = False

    return data_is_prepared


def main(out_path_base, dataset, sub, output_name):

    if dataset == 'deeprecon':
        include_art_dataset = True

    logger.info(f"Working on {dataset} subject {sub}")
    logger.info(f"Checking if perturbed data is prepared...")
    if not perturbed_input_data_is_prepared(out_path_base, dataset, sub, output_name, include_art_dataset):
        raise AssertionError(f"Perturbed data isn't prepared yet. Please run the base main script first for dataset {dataset}")
    
    logger.info(f"Clipping")
    clip_main_perturbated(out_path_base, dataset, sub, include_art_dataset, output_name)

    logger.info(f"Reconstructing")
    vd_reconstruct_main(out_path_base, dataset, sub, include_art_dataset=include_art_dataset, output_name=output_name)



if __name__ == "__main__":
    sub = "AM"
    out_path_base = "/home/matt/programming/recon_diffuser/"
    dataset="deeprecon"
    output_name = "ic_friendly_90-10_500_5"

    fname_log = f"{datetime.now()}_recon_pert_{sub}_{dataset}_{output_name}.log"
    
    logging.basicConfig(
        format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=[
            logging.FileHandler(fname_log),
            logging.StreamHandler()
        ],
        level=logging.INFO)

    logger.info(f"Starting re perturbation recon main with {fname_log}")

    main(out_path_base, dataset, sub, output_name)