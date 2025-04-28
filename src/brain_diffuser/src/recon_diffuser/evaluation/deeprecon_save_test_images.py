import os
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import torchvision.transforms as T

import numpy as np

from eval_utils import simple_log

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


def move_and_rename_imgs(out_path_base, img_root, name):
    if name == 'test':
        correct_stim_order = DEEPRECON_TEST_MASTER_ORDERING
    elif name == 'art':
        correct_stim_order = DEEPRECON_ART_MASTER_ORDERING
    else:
        raise NotImplementedError(f"{name} not supported.")
    
    file_renamer = {im_id:f"{n}.png" for n,im_id in enumerate(correct_stim_order)}

    images_in_img_root = os.listdir(img_root)

    for im_idx in correct_stim_order:
        suffix = '.JPEG' if name == 'test' else '.tiff'
        fname = f"{im_idx}{suffix}"
        if fname not in images_in_img_root:
            raise ValueError(f"image {fname} doesn't exist in img root. CRITICAL!")
        
        fpath = os.path.join(img_root, fname)
        # open the image
        img = Image.open(fpath)

        # out path 
        out_dir = os.path.join(out_path_base, f'data/stimuli/deeprecon/{name}')
        os.makedirs(out_dir, exist_ok=True)

        out_path = os.path.join(out_dir, file_renamer[im_idx])
        # save as png
        img.save(out_path)
        simple_log(f'saved {fname} to {out_path}')

if __name__ == "__main__":
    image_data_root = "/home/kiss/data/contents_shared"

    test_image_path = f"{image_data_root}/ImageNetTest/source"
    art_image_path = f"{image_data_root}/ArtificialShapes/source"

    out_path_base = "/home/matt/programming/recon_diffuser/"
    move_and_rename_imgs(out_path_base, test_image_path, 'test')
    move_and_rename_imgs(out_path_base, art_image_path, 'art')
