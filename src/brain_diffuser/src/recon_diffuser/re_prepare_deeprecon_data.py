"""
Equivalent to the prepare_nsd_data script. Makes sure that all the input data follows the same structure such that all subsequent scripts may work exactly the same. 

Within this preprocessing all the neccessary information will be bundled together in an input file. This contains fmri data, stimulus data and captions for the stimuli. The data format is specified in the README.md.

Entrypoint is the prepare_deeprecon_data_main function.

Attributes:
    Same attributes as in re_deeprecon_main.py
"""

import os
import numpy as np
from tqdm import tqdm
from PIL import Image
from typing import Tuple
import pandas as pd
import logging

import bdpy
import numpy as np
from collections import defaultdict

logger = logging.getLogger("recdi_prepare_deeprecon")

def fmri_indexes_for_image_labels(image_labels: list) -> dict:
    mapper = defaultdict(list)
    for i, image_label in enumerate(image_labels):
        mapper[image_label].append(i)
    return mapper

def load_picture(label: str, base_path: str, suffix: str, resize: int = 0) -> np.array:
    img_path = os.path.join(base_path, label+"."+suffix)
    img = Image.open(img_path)
    if resize > 0:
        width, height = img.size
        if not width == height:
            raise ValueError(f"Image {base_path} {label} isn't square. Cannot resize it to square.")
        img = img.resize((resize, resize))
    # Then resize it. 
    img_data = np.asarray(img)
    if len(img_data.shape) == 2:
        img_data = np.array(img.convert("RGB"))
    return img_data

def build_stim_array_and_fmri_mean_array(fmri: np.array, sig: dict, im_idx: list, base_path: str, im_dim: int, im_suffix: str="JPEG", resize: str=0, compute_train_mean = True) -> Tuple[np.array, np.array]:
    num_images = len(im_idx)
    if compute_train_mean:
        fmri_array = np.zeros((num_images, fmri.shape[1])) # 1200 x num_voxel
    else:
        if not len(fmri) / num_images == 5:
            raise ValueError("We don't have exactly 5 recordings per image in the fmri data. I don't like this.")
        fmri_array = np.zeros((num_images, 5, fmri.shape[1]))
    stim_array = np.zeros((num_images, im_dim, im_dim, 3), dtype=np.uint8) # 3 color channels
    for i,label in tqdm(enumerate(im_idx), desc="Loading image data and setting data means...", total=len(im_idx)):
        img_data = load_picture(label, base_path, im_suffix, resize=resize)
        if compute_train_mean:
            fmri_array[i] = fmri[sorted(sig[label])].mean(0)
        else:
            fmri_array[i] = fmri[sorted(sig[label])]
        stim_array[i] = img_data
    return fmri_array, stim_array

def save_captions(annots, image_ids, name, subject_captions_path):
    # Annots has column content_id and caption. 
    captions_list = []
    for i,idx in tqdm(enumerate(image_ids), desc=f"Setting captions for {name}"):
        captions = annots[annots["content_id"]==idx]["caption"]
        captions_list.append(list(captions.values))
    captions_array = np.array(captions_list)
    np.save(subject_captions_path,captions_array)

DEEPRECON_TEST_MASTER_ORDERING = [
    'n02139199_10398','n03237416_58334','n02437971_5013','n02882301_14188','n01443537_22563','n03941684_21672',
    'n04252077_10859','n02690373_7713','n02071294_46212','n04507155_21299','n02416519_12793','n02128385_20264',
    'n01846331_17038','n02951358_23759','n03482252_22530','n03954393_10038','n03710193_22225','n04554684_53399',
    'n03761084_43533','n03452741_24622','n03272010_11001','n01858441_11077','n02190790_15121','n04533802_19479',
    'n04572121_3262','n02274259_24319','n02916179_24850','n04254777_16338','n01976957_13223','n03495258_9895',
    'n01677366_18182','n03124170_13920','n01943899_24131','n03455488_28622','n02437136_12836','n03379051_8496',
    'n03716966_28524','n01621127_19020','n04387400_16693','n04297750_25624','n04210120_9062','n03122295_31279',
    'n03064758_38750','n03626115_19498','n02950256_22949','n03345837_12501','n02797295_15411','n02824058_18729',
    'n03584254_5040','n03767745_109']

DEEPRECON_ART_MASTER_ORDERING = [
    'colorExpStim29_cyan_+','colorExpStim08_green_largering','colorExpStim24_magenta_+',
    'colorExpStim17_yellow_smallring','colorExpStim15_blue_X','colorExpStim03_red_largering',
    'colorExpStim33_white_largering','colorExpStim09_green_+','colorExpStim40_black_X','colorExpStim26_cyan_square',
    'colorExpStim22_magenta_smallring','colorExpStim21_magenta_square','colorExpStim05_red_X',
    'colorExpStim32_white_smallring','colorExpStim38_black_largering','colorExpStim36_black_square',
    'colorExpStim35_white_X','colorExpStim13_blue_largering','colorExpStim34_white_+','colorExpStim16_yellow_square',
    'colorExpStim06_green_square','colorExpStim19_yellow_+','colorExpStim37_black_smallring',
    'colorExpStim31_white_square','colorExpStim01_red_square','colorExpStim39_black_+',
    'colorExpStim07_green_smallring','colorExpStim10_green_X','colorExpStim30_cyan_X','colorExpStim14_blue_+',
    'colorExpStim12_blue_smallring','colorExpStim04_red_+','colorExpStim02_red_smallring',
    'colorExpStim27_cyan_smallring','colorExpStim20_yellow_X','colorExpStim18_yellow_largering',
    'colorExpStim11_blue_square','colorExpStim25_magenta_X','colorExpStim28_cyan_largering',
    'colorExpStim23_magenta_largering'
 ]


def prepare_data(dataset, name, sub, captions_path, fmri_path, image_path, im_dim, im_suffix, ROI, out_path_base, compute_train_mean = True, captions_only = False, captions_prompt_name = None):
    logger.info(f"{dataset}: Preparing {name} data for subject {sub}...")
    path_processed = os.path.join(out_path_base, f'data/processed_data/{dataset}/subj{sub}/')
    os.makedirs(path_processed, exist_ok=True)

    bdata = bdpy.BData(fmri_path)
    logger.info(f"{name} Data loaded")
    image_labels = bdata.get_labels('stimulus_name')

    #### First: For each image, add indexes of fmri data presentations to dict.
    # Mapping from image label to list of indexes in the dataset
    sig = fmri_indexes_for_image_labels(image_labels)
    im_idx = list(sig.keys())

    if name == 'test':
        logger.info("Ordering the test images...")
        if set(im_idx) != set(DEEPRECON_TEST_MASTER_ORDERING):
            raise ValueError(f'Apparently in your {name} dataset, the images are not the same as in the master ordering...')
        im_idx = DEEPRECON_TEST_MASTER_ORDERING
    elif name =='art':
        logger.info("Ordering the art images...")
        if set(im_idx) != set(DEEPRECON_ART_MASTER_ORDERING):
            raise ValueError(f'Apparently in your {name} dataset, the images are not the same as in the master ordering...')
        im_idx = DEEPRECON_ART_MASTER_ORDERING

    if captions_prompt_name is None:
        subject_captions_path = os.path.join(path_processed, f'{name}_cap_sub{sub}.npy')
    else:
        subject_captions_path = os.path.join(path_processed, f'{name}_{captions_prompt_name}_cap_sub{sub}.npy')

    logger.info(f"Saving Caption data under {subject_captions_path}")
    annots = pd.read_csv(captions_path)
    save_captions(annots, im_idx, name, subject_captions_path)

    if captions_only:
        logger.info("Preparing only captions. Returning...")
        return
    
    fmri = bdata.select(ROI)

    fmri_mean, stim = build_stim_array_and_fmri_mean_array(fmri, sig, im_idx, image_path, im_dim, im_suffix, compute_train_mean=compute_train_mean)

    if not compute_train_mean:
        if not name == "train":
            raise ValueError("compute train mean is False, but the name is not train. It wouldn't make sense to not compute the mean for anything else but the training data.")
        fmri_mean_path = os.path.join(path_processed, f'{name}_fmri_noavg_general_sub{sub}.npy')
    else:
        fmri_mean_path = os.path.join(path_processed, f'{name}_fmriavg_general_sub{sub}.npy')

    logger.info(f"Saving fmri mean under {fmri_mean_path}")
    np.save(fmri_mean_path,fmri_mean)

    stim_path = os.path.join(path_processed, f'{name}_stim_sub{sub}.npy')
    logger.info(f"Saving stim under {stim_path}")
    np.save(stim_path,stim)

    im_idx_path = os.path.join(path_processed, f'{name}_im_idx_sub{sub}.npy')
    logger.info(f"Saving im_idx under {im_idx_path}")
    np.save(im_idx_path,im_idx)

    logger.info(f"Done preparing {name} Data for subject {sub}.")

def prepare_deeprecon_data_main(out_path_base, deeprecon_data_root, image_data_root, captions_data_root, dataset, sub, compute_train_mean = True):
    logger.info(f"prepare deeprecon data main: compute_train_mean {compute_train_mean}")
    # ROI_VC = [LVC, hV4, HVC]
    ROI = "ROI_VC"
    im_dim = 500
    im_dim_art = 240

    fmri_data_train_path = f'{deeprecon_data_root}/{sub}_ImageNetTraining_volume_native.h5'
    fmri_data_test_path = f'{deeprecon_data_root}/{sub}_ImageNetTest_volume_native.h5'
    fmri_data_art_path = f'{deeprecon_data_root}/{sub}_ArtificialShapes_volume_native.h5'

    if not os.path.exists(fmri_data_train_path):
        fmri_data_train_path = f'{deeprecon_data_root}/{sub}_ImageNetTraining_volume_native_rep3.h5'
        if not os.path.exists(fmri_data_train_path):
            raise ValueError(f"Not even {fmri_data_train_path} doesn't exists. All hope is lost.")
    
    assert os.path.exists(fmri_data_train_path)
    assert os.path.exists(fmri_data_test_path)
    assert os.path.exists(fmri_data_art_path)
    
    
    captions_train_path = f"{captions_data_root}/ImagenetTrain/captions/amt_20181204/amt_20181204.csv"
    captions_test_path = f"{captions_data_root}/ImagenetTest/captions/amt_20181204/amt_20181204.csv"
    captions_art_path = f"{captions_data_root}/artificial_shapes/artificial_shapes_captions.csv"

    train_image_path = f"{image_data_root}/ImageNetTraining/source"
    test_image_path = f"{image_data_root}/ImageNetTest/source"
    art_image_path = f"{image_data_root}/ArtificialShapes/source"

    prepare_data(dataset, "test", sub, captions_test_path, fmri_data_test_path, test_image_path, im_dim, "JPEG", ROI, out_path_base)
    prepare_data(dataset, "art", sub, captions_art_path, fmri_data_art_path, art_image_path, im_dim_art, "tiff", ROI, out_path_base)
    prepare_data(dataset, "train", sub, captions_train_path, fmri_data_train_path, train_image_path, im_dim, "JPEG", ROI, out_path_base, compute_train_mean)

def prepare_deeprecon_train_data_no_avg(out_path_base, deeprecon_data_root, image_data_root, captions_data_root, dataset, sub):
    # ROI_VC = [LVC, hV4, HVC]
    ROI = "ROI_VC"
    im_dim = 500

    fmri_data_train_path = f'{deeprecon_data_root}/{sub}_ImageNetTraining_volume_native.h5'

    if not os.path.exists(fmri_data_train_path):
        fmri_data_train_path = f'{deeprecon_data_root}/{sub}_ImageNetTraining_volume_native_rep3.h5'
        if not os.path.exists(fmri_data_train_path):
            raise ValueError(f"Not even {fmri_data_train_path} doesn't exists. All hope is lost.")
    
    assert os.path.exists(fmri_data_train_path)
    
    captions_train_path = f"{captions_data_root}/ImagenetTrain/captions/amt_20181204/amt_20181204.csv"

    train_image_path = f"{image_data_root}/ImageNetTraining/source"

    prepare_data(dataset, "train", sub, captions_train_path, fmri_data_train_path, train_image_path, im_dim, "JPEG", ROI, out_path_base, compute_train_mean)


if __name__ == "__main__":

    sub = "AM" 
    dataset = "deeprecon"
    logger.info(f"Starting data preparation for sub {sub}")

    out_path_base = "/home/matt/programming/recon_diffuser/"

    deeprecon_data_root = "/home/kiss/data/fmri_shared/datasets/Deeprecon/fmriprep"
    image_data_root = "/home/kiss/data/contents_shared"
    captions_data_root = "/home/matt/programming/recon_diffuser/data/annots"
    compute_train_mean = False
        
    logging.basicConfig(
        format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=[
            logging.StreamHandler()
        ],
        level=logging.INFO)
    
    
    prepare_deeprecon_data_main(out_path_base, deeprecon_data_root, image_data_root, captions_data_root, dataset, sub, compute_train_mean)






