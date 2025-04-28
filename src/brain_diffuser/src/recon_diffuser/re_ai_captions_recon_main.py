import os
import logging
from datetime import datetime
from re_clip import clip_main
from re_vd_reconstruct_images import vd_reconstruct_main
from re_prepare_deeprecon_data import prepare_data
from re_ai_captions_utils import get_prompt_name_and_params_from_output_name
from re_utils import load_clip_model

logger = logging.getLogger("recdi_aicapmain")

def prepare_captions(out_path_base, dataset, sub, prompt_name):
    if not dataset == "deeprecon":
        raise NotImplementedError("Can only prepare ai captions for deeprecon data.")
    
    deeprecon_data_root = "/home/kiss/data/fmri_shared/datasets/Deeprecon/fmriprep"
    fmri_data_train_path = f'{deeprecon_data_root}/{sub}_ImageNetTraining_volume_native.h5'
    captions_path = os.path.join(out_path_base, "src/recon_diffuser/ai_captions", prompt_name+".csv")

    image_path = None # not needed for caption extraction
    im_dim = None # not needed for caption extraction
    im_suffix = None # not needed for caption extraction
    ROI = None # not needed for caption extraction
    fmri_path = fmri_data_train_path

    prepare_data(dataset, "train", sub, captions_path, fmri_data_train_path, image_path, im_dim, im_suffix, ROI, out_path_base, captions_only=True, captions_prompt_name=prompt_name)


def captions_are_prepared(out_path_base, dataset, sub, prompt_name):
    """ Checks if all neccesary data is prepared for the perturbed data reconstruction. """
    data_is_prepared = True

    if dataset != "deeprecon":
        raise NotImplementedError("The training-data AI-captions have been generated only for the deeprecon dataset.")

    # captions_path = os.path.join(out_path_base, "src/recon_diffuser/ai_captions", prompt_name+".csv")

    subject_captions_path = os.path.join(out_path_base, f'data/processed_data/deeprecon/subj{sub}/train_{prompt_name}_cap_sub{sub}.npy')

    if not os.path.exists(subject_captions_path):
        data_is_prepared = False

    return data_is_prepared


def main(out_path_base, dataset, sub, output_name):
    if dataset == 'deeprecon':
        include_art_dataset = True
    logger.info(f"Working on {dataset} subject {sub}")
    logger.info(f"Checking if AI captions have been generated already...")

    prompt_name, params = get_prompt_name_and_params_from_output_name(output_name)
    
    if not captions_are_prepared(out_path_base, dataset, sub, prompt_name):
        logger.info("Captions do not exist yet, trying to generate new ones first...")
        prepare_captions(out_path_base, dataset, sub, prompt_name)

    if not captions_are_prepared(out_path_base, dataset, sub, prompt_name): # try again
        raise ValueError(f"Caption data for {prompt_name} isn't prepared yet.")
    
    clip_model = load_clip_model()

    logger.info(f"Clipping")
    clip_main(out_path_base, dataset, sub, include_art_dataset, output_name, prompt_name=prompt_name, clip_model=clip_model)

    logger.info(f"Reconstructing")
    vd_reconstruct_main(out_path_base, dataset, sub, include_art_dataset=include_art_dataset, output_name=output_name, clip_model=clip_model)



if __name__ == "__main__":
    sub = "AM"
    out_path_base = "/home/matt/programming/recon_diffuser/"
    dataset="deeprecon"
    output_name = "aicap_low_level_short"
    prompt_name = "low_level_short"

    fname_log = f"{datetime.now()}_recon_aicap_{sub}_{dataset}_{output_name}.log"
    
    logging.basicConfig(
        format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=[
            logging.StreamHandler()
        ],
        level=logging.INFO)

    logger.info(f"Starting re perturbation recon main with {fname_log}")

    main(out_path_base, dataset, sub, output_name)