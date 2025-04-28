import os
import numpy as np
from tqdm import tqdm
from PIL import Image
from typing import Tuple
import pandas as pd

import bdpy
import numpy as np
from collections import defaultdict
from re_utils import simple_log

from re_prepare_deeprecon_data import fmri_indexes_for_image_labels, load_picture, build_stim_array_and_fmri_mean_array, save_captions

THINGS_TEST_MASTER_ORDERING = [
    'candelabra_14s','pacifier_14s','mango_13s','cookie_15s','blind_19s','pear_13s','butterfly_16n','cow_16n',
    'kimono_14s','beer_14s','cufflink_16s','crayon_15s','coat_rack_13s','bean_13s','footprint_15s','wasp_15n',
    'spoon_13n','headlamp_14s','television_14n','umbrella_13s','axe_14n','piano_13n','microscope_14n','seesaw_13s',
    'hula_hoop_16s','helicopter_25s','graffiti_17s','watch_13s','sim_card_13s','dragonfly_13s','wig_13s','donut_15s',
    'iguana_20s','horse_17s','drawer_13s','speaker_16s','bench_16n','hippopotamus_16s','pan_14s','bulldozer_22n',
    'chipmunk_16n','bobsled_14s','t-shirt_13s','monkey_18n','earring_16n','urinal_13s','starfish_15n','cheese_14n',
    'bike_14s','whip_14s','beaver_13n','shredder_13s','banana_13s','alligator_14n','ribbon_13s','clipboard_14s',
    'altar_13s','crank_16s','grate_13s','jam_15s','pumpkin_21n','horseshoe_13n','typewriter_13s','streetlight_13s',
    'peach_14n','joystick_14s','quill_15s','kazoo_14s','boa_13s','jar_14s','dress_13s','mousetrap_14s','ashtray_14n',
    'mosquito_net_13s','tamale_16s','marshmallow_13s','brace_14s','boat_14n','tent_14n','bed_20n','rabbit_13n',
    'key_17s','bamboo_13s','ferris_wheel_20s','lemonade_14s','drain_16s','fudge_14s','lasagna_13s','grape_20s',
    'stalagmite_14s','guacamole_18s','wallpaper_13s','chest1_14s','dough_19s','uniform_14s','easel_15s',
    'hovercraft_13n','beachball_16s','brownie_14s','nest_13s'
]


def prepare_things_data(dataset, name, sub, things_captions_path, fmri_path, things_images_root, resize_image, im_suffix, ROI_THINGS, out_path_base):
    simple_log(f"{dataset}: Preparing {dataset} {name} data for subject {sub}...")
    path_processed = os.path.join(out_path_base, f'data/processed_data/{dataset}/subj{sub}/')
    os.makedirs(path_processed, exist_ok=True)

    fmri_mean_path = os.path.join(path_processed, f'{name}_fmriavg_general_sub{sub}.npy')
    stim_path = os.path.join(path_processed, f'{name}_stim_sub{sub}.npy')
    subject_captions_path = os.path.join(path_processed, f'{name}_cap_sub{sub}.npy')
    simple_log(f"DEBUG: Loading fmri data from {fmri_path}")
    bdata = bdpy.BData(fmri_path)
    simple_log(f"{name} Data loaded")
    image_labels = bdata.get_labels('stimulus_name')
    sig = fmri_indexes_for_image_labels(image_labels)
    im_idx = list(sig.keys())

    """
    How many test images?
    - 100, same images for all participants (different order though)
    Where test images?
    - Among the other images. Well hidden, but the test image labels are specified in the bdata. 
    Captions exist? 
    - Shirakawa-san generated some. Though I guess some of them are not of highest quality. 
    Are the test images the same across all 3 participants?
    - Yes they are
    Which image size is it?
    - Not the same for all images sadly. 
    """

    if name == 'test':
        simple_log("Ordering the test images...")
        if set(im_idx) != set(THINGS_TEST_MASTER_ORDERING):
            raise ValueError(f'Apparently in your {name} dataset, the images are not the same as in the master ordering...')
        im_idx = THINGS_TEST_MASTER_ORDERING

    fmri = bdata.select(ROI_THINGS)

    
    fmri_mean, stim = build_stim_array_and_fmri_mean_array(fmri, sig, im_idx, things_images_root, resize_image, im_suffix, resize_image )

    simple_log(f"Saving fmri mean under {fmri_mean_path}")
    np.save(fmri_mean_path,fmri_mean)
    simple_log(f"Saving stim under {stim_path}")
    np.save(stim_path,stim)

    annots_raw = pd.read_csv(things_captions_path)
    annots = pd.wide_to_long(annots_raw, ["caption"], "image_name", "counter").reset_index().rename(columns={"image_name": "content_id"})
    annots["content_id"] = annots["content_id"].str[:-4] # no file endings
    save_captions(annots, im_idx, name, subject_captions_path)

def prepare_things_data_main(out_path_base, things_fmri_data_root, things_images_root, things_captions_path, dataset, sub):
    # ROI_THINGS = "V1 + V2 + V3 + hV4 + VO1 + VO2 + LO1 (prf) + LO2 (prf) + TO1 + TO1 + TO2 + V3b + V3a + lEBA + rEBA + lFFA + rFFA + lOFA + rOFA + lSTS + rSTS + lPPA + rPPA + lRSC + rRSC + lTOS + rTOS + lLOC + rLOC"

    # Apparently I cannot select the rois with a space in between. Sad.
    ROI_THINGS = "V1 + V2 + V3 + hV4 + VO1 + VO2 + TO1 + TO2 + V3b + V3a + lEBA + rEBA + lFFA + rFFA + lOFA + rOFA + lPPA + rPPA + lRSC + rRSC + lTOS + rTOS + lLOC + rLOC + lSTS + rSTS"

    if sub == "03": # STS couldn't be measured for that participant
        ROI_THINGS = "V1 + V2 + V3 + hV4 + VO1 + VO2 + TO1 + TO2 + V3b + V3a + lEBA + rEBA + lFFA + rFFA + lOFA + rOFA + lPPA + rPPA + lRSC + rRSC + lTOS + rTOS + lLOC + rLOC"

    resize_image = 500  # If you change that here you need to also change it in "things_save_test_images.py"!
    im_suffix = "jpg"
    

    # Do preparation for test
    test_fmri_path = os.path.join(things_fmri_data_root, f"sub-{sub}_test.h5")
    prepare_things_data(dataset, "test", sub, things_captions_path, test_fmri_path, things_images_root, resize_image, im_suffix, ROI_THINGS, out_path_base)

    # Do preparation for train
    train_fmri_path = os.path.join(things_fmri_data_root, f"sub-{sub}_training.h5")
    prepare_things_data(dataset, "train", sub, things_captions_path, train_fmri_path, things_images_root, resize_image, im_suffix, ROI_THINGS, out_path_base)

if __name__ == "__main__":
    out_path_base = "/home/matt/programming/recon_diffuser/"
    things_fmri_data_root = "/home/share/data/fmri_shared/datasets/THINGS"
    things_captions_path = "/home/share/data/contents_shared/THINGS-fMRI1/derivatives/captions/BLIP_ViT_large/image_captions_KS231012.csv"
    things_images_root = "/home/kiss/data/contents_shared/THINGS-fMRI1/source"

    sub = "01"
    dataset = "things"
    prepare_things_data_main(out_path_base, things_fmri_data_root, things_images_root, things_captions_path, dataset, sub)
    ...