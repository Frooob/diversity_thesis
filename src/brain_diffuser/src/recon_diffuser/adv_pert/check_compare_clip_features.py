"""
All clip features throughout the different patches should be compared between the original images and the perturbed images. 
"""
import os
import pathlib
ppath = str(pathlib.Path(__file__).parent.resolve().parent.resolve())
os.chdir(ppath)

import clip
import torch
from re_clip import load_clip_model
from re_utils import simple_log
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# specify dataset, participant, name, out_addition
dataset = "deeprecon"
subj = "AM"
input_name = "train"
out_addition = "perturb_5050"

out_path_base = "/home/matt/programming/recon_diffuser/"

simple_log("Loading libs")
clip_model_vd = load_clip_model() # versatile diffusion model

simple_log('loading gt images into memory...')

input_images_path = os.path.join(out_path_base, "data", "processed_data", dataset, f"subj{subj}", f"{input_name}_stim_sub{subj}.npy")
images_gt = np.load(input_images_path).astype(np.uint8) # loads all ground truth images. might take some time.
simple_log('Done loading gt images.')

# specify image number (index)
im_idx = 0

# load images 
img_gt = Image.fromarray(images_gt[im_idx])

    # perturbed from png
pert_no = 0 # 5 possible perturbations usually
perturb_kind = 'friendly' # adversarial or friendly

pert_img_path = os.path.join(out_path_base, "data", "processed_data", dataset, f"subj{subj}", out_addition, perturb_kind, f"im{im_idx}_{perturb_kind}_pert{pert_no}.png")
img_pt = Image.open(pert_img_path)


# vd library gt
gt_features = clip_model_vd.clip_encode_vision(img_gt).squeeze(0) # returns Tensor([257,768])

# vd library perturbed
pt_features = clip_model_vd.clip_encode_vision(img_pt).squeeze(0) # returns Tensor([257,768])

