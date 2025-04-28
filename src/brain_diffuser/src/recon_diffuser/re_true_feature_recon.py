from re_icnn_main import icnn_main
from re_vdvae import vdvae_main
from re_vd_reconstruct_images import vd_reconstruct_main
from re_clip import clip_main
from datetime import datetime
import logging
import os
from re_utils import load_clip_model
from re_ai_captions_utils import parse_true_feat_output_name

logger = logging.getLogger("recdi_truefeatmain")


def bd_features_computed_already(out_path_base, dataset, sub):
    # probably not necessary. Because the features will be computed nevertheless. 

    # check for brain-diffuser

    # Check VDVAE

    # Check cliptext

    # check clipvision
    ...

def icnn_features_computed_already(out_path_base, dataset, sub):
    # Check for ICNN
    features_missing = False
    true_feature_dir = os.path.join(out_path_base, 'data/extracted_features', dataset, f"subj{sub}")

    test_dataset_features_exist = os.path.exists(os.path.join(true_feature_dir, 'test_icnn_extracted_features.npy'))
    if not test_dataset_features_exist:
        logger.warning(f"True Features {dataset, sub} test dataset don't exist yet. They will be created")
        features_missing = True
    if dataset == 'deeprecon':
        art_dataset_features_exist = os.path.join(true_feature_dir, 'art_icnn_extracted_features')
        if not art_dataset_features_exist:
            logger.warning(f"True Features {dataset, sub} art dataset don't exist yet. They will be created")
            features_missing = True
    if features_missing:
        logger.info("Not all true features have been computed already, they will be computed.")
    return features_missing


def reconstruct_true_features_icnn(out_path_base, dataset, sub, output_name):
    # output_name will most likely not be used
    icnn_features_computed_already(out_path_base, dataset, sub)

    # do icnn main
    icnn_main(out_path_base, dataset, sub, output_name=output_name, device="cuda:0", true_features_reconstruction=True)
    ...

def reconstruct_true_features_bd(out_path_base, dataset, sub, output_name):
    # output_name might be used if only clipvision or only cliptext true features should be used
    # But I don't really want to make that distinction
    bd_features_computed_already(out_path_base, dataset, sub)

    # if not true features computed already, compute them I guess?
    include_art_dataset = dataset == 'deeprecon'
    # do vdvae main
    # vdvae_main(out_path_base, dataset, sub, include_art_dataset=include_art_dataset, output_name=output_name, use_true_features=True)

    # extract clip true features
    clip_model = load_clip_model()
    # clip_main(out_path_base, dataset, sub, include_art_dataset, output_name, clip_model, use_true_features=True)
    # do vd reconstruct main
    vd_reconstruct_main(out_path_base, dataset, sub, include_art_dataset, output_name, clip_model, use_true_features=True)

def true_features_main(out_path_base, dataset, sub, output_name):
    algorithm, params = parse_true_feat_output_name(output_name)

    if algorithm == 'icnn':
        reconstruct_true_features_icnn(out_path_base, dataset, sub, output_name)
    elif algorithm == 'bd':
        reconstruct_true_features_bd(out_path_base, dataset, sub, output_name)
    else:
        raise ValueError("Algorithm must either be icnn or bd")


if __name__ == "__main__":
    sub = "AM"
    out_path_base = "/home/matt/programming/recon_diffuser/"
    dataset="deeprecon"
    # output_name = "true_icnn"
    output_name = "true_bd"

    fname_log = f"{datetime.now()}_recon_aicap_{sub}_{dataset}_{output_name}.log"
    
    logging.basicConfig(
        format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=[
            logging.StreamHandler()
        ],
        level=logging.INFO)

    logger.info(f"Starting re perturbation recon main with {fname_log}")

    true_features_main(out_path_base, dataset, sub, output_name)