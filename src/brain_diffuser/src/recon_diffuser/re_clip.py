"""
This module provides the functionality train the clip decoder used in the seconds step of the brain-diffuser algorithm.

Entrypoint is the clip_main function. The functionality is described there.
"""


import os
import numpy as np
import pandas as pd
import logging
import sys 
import torch
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as T
from tqdm import tqdm
from PIL import Image
import sklearn.linear_model as skl
from re_utils import batch_generator_external_images, load_normalized_train_fmri, get_dropout_indices, load_clip_model
sys.path.append('/home/matt/programming/recon_diffuser/src/recon_diffuser/adv_pert')
from clip_module_testing import get_im_idx_sub

logger = logging.getLogger("recdi_clip")


def cliptext_extract_features(out_path_base, dataset, sub, name, clip_model, output_name="", save_features = True, prompt_name = None):
    logger.info(f"Extracting {name} clip text features for subject {sub}")
    
    if prompt_name is None:
        caps = np.load(os.path.join(out_path_base, f'data/processed_data/{dataset}/subj{sub}/{name}_cap_sub{sub}.npy'))
    else:
        logger.info("Reading prompt name captions")
        caps = np.load(os.path.join(out_path_base, f'data/processed_data/{dataset}/subj{sub}/{name}_{prompt_name}_cap_sub{sub}.npy'))

    if name == "train" and output_name.startswith("dropout"):
        indexer = get_dropout_indices(out_path_base, dataset, name, sub, output_name)
        caps = caps[indexer]
        name = name+"_"+output_name
    elif output_name != "": # to be able to save the true text features for the aicap experiment
        name = name+"_"+output_name

    num_embed, num_features, num = 77, 768, len(caps)
    cliptext_features = np.zeros((num,num_embed, num_features))

    with torch.no_grad():            
        for i,annots in tqdm(enumerate(caps), desc=f"Annot {name} Caps", total=len(caps)):
            cin = list(annots[annots!=''])
            c = clip_model.clip_encode_text(cin)
            cliptext_features[i] = c.to('cpu').numpy().mean(0) # compute the mean of the features across all 5 annotations per image
        if save_features:
            save_path = os.path.join(out_path_base, f'data/extracted_features/{dataset}/subj{sub}/cliptext_{name}.npy')
            logger.info(f"Saving cliptext features {name} to {save_path}")
            np.save(save_path,cliptext_features)
            logger.info("Done.")
    return cliptext_features


class batch_generator_perturbed_images(Dataset):
    def __init__(self, data_path, im_idx):
        self.data_path = data_path
        self.image_names = [im for im in os.listdir(data_path) if im.endswith('png')]
        self.im_idx = im_idx
        image_names = self.image_names
        data_files_sub_order = []
        for sub_im_id in im_idx:
            for file in image_names:
                if file.startswith(sub_im_id + "_"):
                    data_files_sub_order.append(file)
        self.translated_files = data_files_sub_order

        if not all(pd.Series([f[:-6] for f in self.translated_files]).value_counts() == 5):
            raise Exception("ERROR! If you have pert_n (number of perturbated images) set to some other value than 5, you can simply delete this error. Otherwise, you might have a big problem.")
        else:
            logger.info("All translated files are 5 of each kind. That's nice honey. ")

    def __getitem__(self,idx):
        img = Image.open(os.path.join(self.data_path, self.translated_files[idx]))
        img = T.functional.to_tensor(img).float()
        img = img*2 - 1 # because the clip model does the opposite calculation. Probably doesn't change much in the result anyways.
        return img

    def __len__(self):
        return len(self.image_names)

# does the output format look the same as batch generator from before?

def clipvision_extract_features(out_path_base, dataset, sub, name, clip_model, output_name="", save_features=True):
    # if output name is empty: perform the baseline feature extraction
    # Else: do some stuff according to the output name (might be perturbation or extended captions or such)
    logger.info(f"Extracting {name} clip vision features for subject {sub}")

    batch_size=1
    if name in ['test', 'art'] or output_name == "" or output_name.startswith("aicap"):
        logger.info("Creating batch generator for baseline clipvision feature extraction.")
        image_path = os.path.join(out_path_base, f'data/processed_data/{dataset}/subj{sub}/{name}_stim_sub{sub}.npy')
        images = batch_generator_external_images(data_path = image_path, clipvision=True, resize_size=512)
        if output_name != "":
            name = name+"_"+output_name
    elif output_name.startswith("dropout"):
        indexer = get_dropout_indices(out_path_base, dataset, name, sub, output_name)
        logger.info(f"Creating batch generator for {output_name} clipvision feature extraction.")
        image_path = os.path.join(out_path_base, f'data/processed_data/{dataset}/subj{sub}/{name}_stim_sub{sub}.npy') # same as above
        images = batch_generator_external_images(data_path = image_path, clipvision=True, resize_size=512, select_indices=indexer)
        name = name+"_"+output_name
    elif output_name.startswith("true"):
        logger.info("Clipvision feature extraction for true features, saving under normal name.")
        image_path = os.path.join(out_path_base, f'data/processed_data/{dataset}/subj{sub}/{name}_stim_sub{sub}.npy') # same as above
        images = batch_generator_external_images(data_path = image_path, clipvision=True, resize_size=512)
        name = name
    else: # perturbed data
        if dataset != 'deeprecon':
            raise NotImplementedError("Can do perturbation only for deeprecon dataset")
        image_folder = os.path.join(out_path_base, f'data/train_data/deeprecon_perturbed/', output_name)
        logger.info(f"Creating batch generator from {image_folder} feature_extraction.")
        im_idx = get_im_idx_sub(out_path_base, sub)
        images = batch_generator_perturbed_images(image_folder, im_idx)
        name = name+"_"+output_name

    logger.info(f"Created batch generator of size {len(images)} images")
    loader = DataLoader(images,batch_size,shuffle=False)
    logger.info(f"Created dataloader")
    num_embed, num_features, num = 257, 768, len(images)
    clipvision_features = np.zeros((num,num_embed,num_features))
    logger.info(f"Created zeroes")

    with torch.no_grad():
        for i,cin in tqdm(enumerate(loader), desc=f"Encoding {output_name} vision", total=len(loader)):
            c = clip_model.clip_encode_vision(cin)
            clipvision_features[i] = c[0].cpu().numpy()
    if save_features:
        feature_path = os.path.join(out_path_base, f'data/extracted_features/{dataset}/subj{sub}/clipvision_{name}.npy')
        logger.info(f"Saving extracted vision features {name} to {feature_path}")
        np.save(feature_path,clipvision_features)
        logger.info("Done.")
        
    return clipvision_features

def get_test_data_collector(out_path_base, dataset, sub, names, cliptype, norm_mean_train, norm_scale_train, clip_features=None):
    test_data_collector = []
    if "train" in names:
        raise ValueError("Train shouldn't be here. There must be an error in the code somewhere.")
    for name in names: # ideally names are "test" and "art"
        name_fmri_path = os.path.join(out_path_base, f'data/processed_data/{dataset}/subj{sub}/{name}_fmriavg_general_sub{sub}.npy')
        name_fmri = np.load(name_fmri_path)
        name_fmri = (name_fmri - norm_mean_train) / norm_scale_train
        if clip_features:
            name_clip = clip_features[f"{cliptype}_{name}"]
        else:
            name_clip = np.load(os.path.join(out_path_base, f'data/extracted_features/{dataset}/subj{sub}/{cliptype}_{name}.npy'))
        name_pred_clip = np.zeros_like(name_clip)
        test_data_collector.append((name, name_fmri, name_clip, name_pred_clip))
    return test_data_collector


def clip_regression(out_path_base, dataset, sub, names, cliptype, ridge_alpha, ridge_max_iter, output_name = "", clip_features=None,):
    # names is either ['test'] or ['art', 'test']
    # cliptype is either 'clipvision' or 'cliptext'. Only used for naming/paths.

    logger.info(f"Doing {cliptype} regression")
    if output_name.endswith("_noinputavg"):
        input_avg = False
    else:
        input_avg = True
    
    if output_name.startswith("dropout"):
        indexer = get_dropout_indices(out_path_base, dataset, "train", sub, output_name)
        train_fmri, norm_mean_train, norm_scale_train = load_normalized_train_fmri(out_path_base, dataset, sub, input_avg, select_indices=indexer)
    else:
        train_fmri, norm_mean_train, norm_scale_train = load_normalized_train_fmri(out_path_base, dataset, sub, input_avg)
    
    logger.info(f"Opened train fmri with shape {train_fmri.shape}")

    test_data_collector = get_test_data_collector(out_path_base, dataset, sub, names, cliptype, norm_mean_train, norm_scale_train, clip_features)

    num_voxels = train_fmri.shape[-1]
    
    if output_name and output_name != "_noinputavg":  # latter case would be baseline without inputavg, we need the normal train features here.
        train_name = f"train_{output_name}" # for example train_perturb_5050_friendly 
    else:
        train_name = "train"

    if clip_features: # skip IO when reading clip features from argument directly. 
        train_clip = clip_features[f"{cliptype}_train"]
    else:
        train_clip = np.load(os.path.join(out_path_base, f'data/extracted_features/{dataset}/subj{sub}/{cliptype}_{train_name}.npy'))
    
    if output_name ==  "_noinputavg": # we have the case that we have baseline data without input avg
        # need to duplicate the input n times 
        num_fmri_recordings = train_fmri.shape[0] * train_fmri.shape[1]
        n_duplicate_train_features = num_fmri_recordings / train_clip.shape[0]
        logger.info(f"Duplicating train features {n_duplicate_train_features} times")
        train_clip = np.repeat(train_clip, n_duplicate_train_features, axis=0)

    
    num_samples,num_embed,num_dim = train_clip.shape
    
    if output_name:
        if output_name.endswith("_noinputavg"):
            num_fmri_recordings = train_fmri.shape[0] * train_fmri.shape[1]
            logger.info("Reshaping train_fmri")
            train_fmri = train_fmri.reshape(num_fmri_recordings, num_voxels)
        else:
            num_fmri_recordings = train_fmri.shape[0]
            num_samples_per_fmri_image = num_samples / num_fmri_recordings
            if not int(num_samples_per_fmri_image) == num_samples_per_fmri_image:
                raise ValueError("You don't seem to have an even number of perturbed images per mri recording.")
            num_samples_per_fmri_image = int(num_samples_per_fmri_image)
            # expand the train fmri n times to have the same shape as the perturbed images
            train_fmri = np.repeat(train_fmri, num_samples_per_fmri_image, axis=0)

    logger.info("Training Regression")
    reg_w = np.zeros((num_embed,num_dim,num_voxels)).astype(np.float32)
    reg_b = np.zeros((num_embed,num_dim)).astype(np.float32)
    for i in tqdm(range(num_embed), "Fitting ridge Regressions"):
        # cliptext we have 77 'layers' (tokens) with 768 each
        # clipvision we have 257 'layers' (patches + final clip embedding)
        reg = skl.Ridge(alpha=ridge_alpha, max_iter=ridge_max_iter, fit_intercept=True)
        reg.fit(train_fmri, train_clip[:,i])

        reg_w[i] = reg.coef_
        reg_b[i] = reg.intercept_
        clip_mean_train = np.mean(train_clip[:,i],axis=0)
        clip_std_train =  np.std(train_clip[:,i],axis=0)
        for name, name_fmri, name_clip, name_pred_clip in test_data_collector: # Only two (art, test)
            features_pred = predict_features_for_one_test_dataset(reg, name_fmri, clip_mean_train, clip_std_train)
            name_pred_clip[:, i] = features_pred
            # logger.info(f"Embedding {i}: {name}")
    
    for name, name_fmri, name_clip, name_pred_clip in test_data_collector:
        if output_name:
            name = f"{name}_{output_name}"
        pred_out_path = os.path.join(out_path_base, f'data/predicted_features/{dataset}/subj{sub}/{cliptype}_pred{name}_general.npy')
        logger.info(f"Saving {name} pred {cliptype} to {pred_out_path}")
        np.save(pred_out_path,name_pred_clip)

    logger.info("All done...")

def predict_features_for_one_test_dataset(reg, fmri,clip_mean_train, clip_std_train):
        """
        Denormalizes the predicted features using the std and mean of the train dataset.
        """
        pred_latent = reg.predict(fmri)
        ## Slight Info leakage, but ok. We could do this for each test img separately too. This is just faster to do it for all of them at once.
        std_norm_latent = (pred_latent - np.mean(pred_latent,axis=0)) / np.std(pred_latent,axis=0) 
        train_normalized_pred = std_norm_latent * clip_std_train + clip_mean_train
        return train_normalized_pred


def clip_main(out_path_base, dataset, sub, include_art_dataset=False, output_name="", clip_model = None, prompt_name = None, use_true_features=False):
    """ Main entrypoint for CLIP computation. 
    First extracts the clip features for all datasets (train, test and deeprecon artificial images). Done for both the cliptext and the clipvision features.

    Then the regression is trained on the training data for both the cliptext and the clipvision features. 
    
    The trained regression can be used to predict the latents of the test fmri. This is done in the reconstruction step.

    See the original paper from Ozcelik for a visual explanation of the clip step.
    """
    logger.info(f"Doing clip_main for {output_name}")

    if clip_model is None:
        clip_model = load_clip_model()
    clip_features = {}

    ct_test = cliptext_extract_features(out_path_base, dataset, sub, "test", clip_model, output_name, save_features=True, prompt_name=None) # prompt name only matters for train data
    clip_features["cliptext_test"] = ct_test
    cv_test = clipvision_extract_features(out_path_base, dataset, sub, "test", clip_model, output_name, save_features=True)
    clip_features["clipvision_test"] = cv_test

    if not use_true_features: # don't need train data for true feature reconstruction.
        ct_train = cliptext_extract_features(out_path_base, dataset, sub, "train", clip_model, output_name, save_features=False, prompt_name=prompt_name)
        clip_features["cliptext_train"] = ct_train
        cv_train = clipvision_extract_features(out_path_base, dataset, sub, "train", clip_model, output_name, save_features=False)
        clip_features["clipvision_train"] = cv_train

    if include_art_dataset:
        ct_art = cliptext_extract_features(out_path_base, dataset, sub, "art", clip_model, output_name, save_features=True, prompt_name=None)
        clip_features["cliptext_art"] = ct_art
        cv_art = clipvision_extract_features(out_path_base, dataset, sub, "art", clip_model, output_name, save_features=True)
        clip_features["clipvision_art"] = cv_art

    if not use_true_features:
        names = ["test", "art"] if include_art_dataset else ["test"]

        clip_regression(out_path_base, dataset, sub, names, "cliptext" , ridge_alpha=100000, ridge_max_iter=50000, output_name=output_name, clip_features=clip_features)
        clip_regression(out_path_base, dataset, sub, names, "clipvision" , ridge_alpha=60000, ridge_max_iter=50000, output_name=output_name, clip_features=clip_features)

def clip_main_perturbated(out_path_base, dataset, sub, include_art_dataset, output_name):
    clip_model = load_clip_model()
    
    names_to_train = ['train', 'test', 'art'] if include_art_dataset else ['train', 'test']
    names_to_test = [name for name in names_to_train if name != "train"]
    for name_to_test in names_to_test:
        if not os.path.exists(
        os.path.join(out_path_base, f'data/predicted_features/{dataset}/subj{sub}/cliptext_pred{name_to_test}_general.npy')):
            raise ValueError(f"Cliptext regression hasn't been done yet.")
    
    vision_features = {}
    for name in names_to_train:
        if not os.path.exists(
            os.path.join(out_path_base, f'data/extracted_features/{dataset}/subj{sub}/cliptext_{name}.npy')):
            raise ValueError(f"The cliptext features haven't been generated yet.")
        if name == "train":
            save_features = False
        else: 
            save_features = True
        vision_features[f"clipvision_{name}"] = clipvision_extract_features(
            out_path_base, dataset, sub, name, clip_model, output_name.replace("_noinputavg", ""), save_features=save_features)
        # will generate the perturbed features for train
        # generates regular test/art features for test/art
        
    clip_regression(out_path_base, dataset, sub, names_to_test, "clipvision" , ridge_alpha=60000, ridge_max_iter=50000, output_name=output_name, clip_features=vision_features)


if __name__ == "__main__":
    logger.info("Executing re_vdvae")
    sub = "AM"
    batch_size=25
    out_path_base = "/home/matt/programming/recon_diffuser/"
    dataset="deeprecon"
    output_name = "dropout-random_0.1_00"
    # output_name = ""
    include_art_dataset = (dataset=='deeprecon')
    clip_model = None

    cliptype = "cliptext"
    ridge_alpha = 100000
    ridge_max_iter = 50000
    names = ['art', 'test']

    logging.basicConfig(
        format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=[
            logging.StreamHandler()
        ],
        level=logging.INFO)

    clip_main(out_path_base, dataset, sub, include_art_dataset, output_name)

