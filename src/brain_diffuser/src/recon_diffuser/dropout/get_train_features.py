"""
We need to get the features for all the train images. 

This would mean clip features and dreamsim features
"""
import torch
from tqdm import tqdm
from dreamsim import dreamsim
import clip
from dropout_random import get_all_input_imgs
from PIL import Image
import numpy as np
import pandas as pd
from collections import defaultdict

out_path_base = "/home/matt/programming/recon_diffuser/"
dataset = "deeprecon"
im_suffix = "JPEG"
im_idx, all_input_filepaths = get_all_input_imgs(out_path_base, dataset, im_suffix)

device = "cuda"
model_clip, preprocess_clip = clip.load("ViT-L/14", device=device)
model_dreamsim, preprocess_dreamsim = dreamsim(pretrained=True, device=device)


if dataset != "deeprecon":
    raise NotImplementedError(f"Dataset {dataset} captions are not here. Please provide path")

captions_path = "/home/matt/programming/recon_diffuser/data/annots/ImagenetTrain/captions/amt_20181204/amt_20181204.csv"
df_captions = pd.read_csv(captions_path)

captions_collector = defaultdict(list)
for index, row in tqdm(df_captions.iterrows()):
    caption = row["caption"]
    tokens = clip.tokenize(caption).to(device) # tokenize the caption
    feature_vector = model_clip.encode_text(tokens).detach()
    captions_collector[row["content_id"]].append(feature_vector)

# average the captions vector for each caption
# in the correct order
averaged_clip_texts = [np.mean([capvec.cpu().numpy() for capvec in captions_collector[idx]], axis=0).squeeze() for idx in im_idx]
single_clip_texts = [[capvec.cpu().numpy() for capvec in captions_collector[idx]][0].squeeze() for idx in im_idx]
clip_text_raw = [[capvec.cpu().numpy().squeeze() for capvec in captions_collector[idx]] for idx in im_idx]

features_clipvision = []
for im_path in tqdm(all_input_filepaths, desc="Encoding clip features..."):
    with torch.no_grad():    
        im_clip_preprocessed = preprocess_clip(Image.open(im_path)).unsqueeze(0).to(device)
        vector_clip = model_clip.encode_image(im_clip_preprocessed).squeeze().cpu().numpy()
    features_clipvision.append(vector_clip)
features_clipvision = np.array(features_clipvision)


features_dreamsim = []
for im_path in tqdm(all_input_filepaths, desc="Encoding dreamsim features..."):
    with torch.no_grad():    
        im_dreamsim_preprocessed = preprocess_dreamsim(Image.open(im_path)).to(device)
        vector_dreamsim = model_dreamsim.embed(im_dreamsim_preprocessed).squeeze().cpu().numpy()
    features_dreamsim.append(vector_dreamsim)
features_dreamsim = np.array(features_dreamsim)

path_features = f"{dataset}_train_features.npz"
np.savez(path_features, im_idx=im_idx, features_clipvision=features_clipvision, features_dreamsim=features_dreamsim, features_cliptext = averaged_clip_texts, features_cliptext_single = single_clip_texts, cliptext_raw = clip_text_raw)


npz = np.load(path_features)
