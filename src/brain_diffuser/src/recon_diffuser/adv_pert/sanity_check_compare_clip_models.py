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
# Load clip models
    # clip library
clip_model_lib, preprocess_lib = clip.load("ViT-L/14", device=device)

    # versatile diffusion
clip_model_vd = load_clip_model()

simple_log('loading gt images into memory...')

input_images_path = os.path.join(out_path_base, "data", "processed_data", dataset, f"subj{subj}", f"{input_name}_stim_sub{subj}.npy")
images_gt = np.load(input_images_path).astype(np.uint8) # loads all ground truth images. might take some time.
simple_log('Done loading gt images.')

# specify image number (index)
im_idx = 2

# load images 
    # ground truth
img_gt = Image.fromarray(images_gt[im_idx])

    # perturbed from png
pert_no = 0 # 5 possible perturbations usually
perturb_kind = 'adversarial' # adversarial or friendly

pert_img_path = os.path.join(out_path_base, "data", "processed_data", dataset, f"subj{subj}", out_addition, perturb_kind, f"im{im_idx}_{perturb_kind}_pert{pert_no}.png")
img_pt = Image.open(pert_img_path)


# OPTIONAL: Show both of the images
plt.imshow(img_gt)
plt.imshow(img_pt)

# do the feature extraction 2x2
    # clip library gt
processed_img_gt = preprocess_lib(img_gt).unsqueeze(0).to(device)
gt_features_lib = clip_model_lib.encode_image(processed_img_gt).squeeze(0) # returns Tensor([768])

    # clip library perturbed
processed_img_pert = preprocess_lib(img_pt).unsqueeze(0).to(device)
pt_features_lib = clip_model_lib.encode_image(processed_img_pert).squeeze(0) # returns Tensor([768])

    # vd library gt
gt_features_vd = clip_model_vd.clip_encode_vision(img_gt).squeeze(0) # returns Tensor([257,768])

gt_features_vd.shape
gt_features_lib.shape
    # vd library perturbed
pt_features_vd = clip_model_vd.clip_encode_vision(img_pt).squeeze(0) # returns Tensor([257,768])

# Compare features
# compare gt_features_vd with gt_features_lib
"""
Sanity check do both the clip lib and the vd lib actually use the same clip model/parameters.

There are 4 different comparisons to make:
gt_clip_lib vs gt_vd_lib should be 0
pt_clip_lib vs pt_vd_lib should be 0

gt_clip_lib vs pt_vd_lib should be ~ .5
pt_clip_lib vs gt_vd_lib should be ~ .5 (same as above)
"""
gt_clip_lib_vs_gt_vd_lib = (1 - torch.nn.functional.cosine_similarity(gt_features_lib, gt_features_vd[0].unsqueeze(0))).item() # should be 0
pt_clip_lib_vs_pt_vd_lib = (1 - torch.nn.functional.cosine_similarity(pt_features_lib, pt_features_vd[0].unsqueeze(0))).item() # should be 0

gt_clip_lib_vs_pt_vd_lib = (1 - torch.nn.functional.cosine_similarity(gt_features_lib, pt_features_vd[0].unsqueeze(0))).item() # should be ~ .5
pt_clip_lib_vs_gt_vd_lib = (1 - torch.nn.functional.cosine_similarity(pt_features_lib, gt_features_vd[0].unsqueeze(0))).item() # should be ~ .5

print(
    f"gt_clip_lib vs gt_vd_lib should be 0: {gt_clip_lib_vs_gt_vd_lib:.4f}\n\
pt_clip_lib vs pt_vd_lib should be 0: {pt_clip_lib_vs_pt_vd_lib:.4f}\n\
gt_clip_lib vs pt_vd_lib should be ~ .5: {gt_clip_lib_vs_pt_vd_lib:.4f}\n\
pt_clip_lib vs gt_vd_lib should be ~ .5 (same as above): {pt_clip_lib_vs_gt_vd_lib:.4f}")


"""
Second sanity Check:
Are the tokens after the perturbation process actually different or only the final result?
"""

gt_vd_vs_pt_vd = (1 - torch.nn.functional.cosine_similarity(gt_features_vd, pt_features_vd)) # should be ~ .5
print(f"Mean difference between ground truth tokens and perturbed tokens: {gt_vd_vs_pt_vd.mean().item():.4f}")
import seaborn as sns
gt_vd_vs_pt_vd
sns.histplot(gt_vd_vs_pt_vd.cpu().numpy()).set_title("Cosine dists between clip patches perturbed vs gt")


"""
Third Check:
Is there a difference between friendly and adverserial features?
"""
from tqdm import tqdm
from collections import defaultdict
from skimage.metrics import structural_similarity as ssim

metrics_fourth_check = []
for im_idx in tqdm(range(1000)):
    # load images 
        # ground truth
    img_gt = Image.fromarray(images_gt[im_idx])
    img_gt_resized = img_gt.resize((224,224))

        # perturbed from png
    pert_no = 0 # 5 possible perturbations usually
    
    for perturb_kind in ['friendly', 'adversarial']:
        d = {}
        pert_img_path = os.path.join(out_path_base, "data", "processed_data", dataset, f"subj{subj}", out_addition, perturb_kind, f"im{im_idx}_{perturb_kind}_pert{pert_no}.png")
        img_pt = Image.open(pert_img_path)

        gt_features_vd = clip_model_vd.clip_encode_vision(img_gt).squeeze(0)
        pt_features_vd = clip_model_vd.clip_encode_vision(img_pt).squeeze(0)
        # gt_vs_pt = (1 - torch.nn.functional.cosine_similarity(gt_features_vd, pt_features_vd)).mean().item() # all 257 features

        gt_vs_pt = (1 - torch.nn.functional.cosine_similarity(gt_features_vd, pt_features_vd))[0].mean().item() # only main feature
        d['perturb_kind'] = perturb_kind
        d['cosine_dist'] = gt_vs_pt

        img_pt_np = np.array(img_pt)
        img_gt_resized_np = np.array(img_gt_resized)
        mse = np.linalg.norm(img_pt_np - img_gt_resized_np)
        ssim_measured = ssim(img_pt_np, img_gt_resized_np, multichannel=True, data_range=img_pt_np.max() - img_gt_resized_np.min())
        d['mse'] = mse
        d['ssim'] = ssim_measured

        metrics_fourth_check.append(d)

import pandas as pd

df = pd.DataFrame(metrics_fourth_check)
sns.boxplot(data=df, x='perturb_kind', y='cosine_dist').set_title('adv vs friendly main feature')

"""
Fourth Check:
Is there a difference between SSIM/MSE between friendly and adverserial features?

"""
from scipy import stats


sns.boxplot(data=df, x='perturb_kind', y='mse').set_title('adv vs friendly main feature')
df_friendly = df[df['perturb_kind'] == 'friendly']
df_adversarial = df[df['perturb_kind'] == 'adversarial']
sns.boxplot(data=df, x='perturb_kind', y='ssim').set_title('adv vs friendly main feature')

stats.ttest_ind(df_friendly['mse'], df_adversarial['mse'])
stats.ttest_ind(df_friendly['ssim'], df_adversarial['ssim'])
stats.ttest_ind(df_friendly['cosine_dist'], df_adversarial['cosine_dist'])


"""
Fifth check:
Does it also work with the dataloader that I've created?
"""


"""
X sanity Check:
Do we comply with the losses that are computed from the data? 
Can first explore this after KS is completed.
"""

# read the log
# extract 
