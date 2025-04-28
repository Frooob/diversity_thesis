"""
This module provides the functionality to compute the vdvae for the first step of the brain-diffuser pipeline. 

Entrypoint is the vdvae_main function. The functionality is described there.
"""

import pandas as pd
import torch
import numpy as np
import os
from tqdm import tqdm
from vdvae.model_utils import set_up_data, load_vaes
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import torchvision.transforms as T
import os
import numpy as np
import sklearn.linear_model as skl
import pickle
from re_utils import get_ema_vae_and_preprocess_fn, batch_generator_external_images, load_normalized_train_fmri, get_dropout_indices
from scipy import spatial

import logging

logger = logging.getLogger("recdi_VDVAE")


def stats_to_cpu(stats):
    cpu_stats = []
    for layer in stats:
        d = {}
        for key, value in layer.items():
            d[key] = value.cpu()
        cpu_stats.append(d)
    return cpu_stats

def vdvae_extract_features(out_path_base, dataset, sub, name, save=True, num_latents=31, batch_size=25, output_name=""):
    if output_name:
        if "dropout" in output_name:
            select_indices = get_dropout_indices(out_path_base, dataset, name, sub, output_name)
            output_name = name+"_"+output_name
        elif 'true' in output_name: # for true feature reconstruction
            output_name = name
            select_indices = None
        else:
            raise ValueError(f"Specified output name {output_name} isn't supported.")
    else:
        output_name = name
        select_indices = None
        
    logger.info(f"Extracting {name} vdvae features for sub {sub}")
    path_processed = os.path.join(out_path_base, f'data/processed_data/{dataset}/subj{sub}/')
    ema_vae, preprocess_fn = get_ema_vae_and_preprocess_fn()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    image_path = os.path.join(path_processed,f'{name}_stim_sub{sub}.npy')
    images = batch_generator_external_images(data_path = image_path, resize_size=64, select_indices=select_indices)

    dataloader = DataLoader(images,batch_size,shuffle=False)

    logger.info("Dataloaders created")

    ema_vae.decoder = ema_vae.decoder.to(device)

    def get_latents(dataloader, name):
        latents = []
        for i,x in tqdm(enumerate(dataloader), desc=f"{name} latents", total=len(dataloader)):
            data_input, target = preprocess_fn(x)
            with torch.no_grad():
                    data_input = data_input.to(device)
                    activations = ema_vae.encoder.forward(data_input)
                    px_z, stats = ema_vae.decoder.forward(activations, get_latents=True)
                    batch_latent = []
                    for i in range(num_latents):
                        batch_latent.append(stats[i]['z'].cpu().numpy().reshape(len(data_input),-1))
                    latents.append(np.hstack(batch_latent))
        latents = np.concatenate(latents) 
        return latents, stats

    latents, stats = get_latents(dataloader, output_name)
    logger.info("Latents computed")

    if save:
        base_path = os.path.join(out_path_base, f"data/extracted_features/{dataset}/subj{sub}/")
        os.makedirs(base_path, exist_ok=True)
        latents_path = os.path.join(base_path, f"vdvae_features_{output_name}_{num_latents}l.npz")
        logger.info(f"Saving features at {latents_path}")
        cpu_stats = stats_to_cpu(stats)
        np.savez(latents_path,latents=latents, stats=cpu_stats)
    logger.info(f"Done extracting {output_name} features for subject {sub}")

    return latents, stats


def vdvae_regression(out_path_base, dataset, sub, num_latents=31, save_regression_weights=True, output_name=""):
    logger.info(f"Doing vdvae regression sub {sub}")
    if output_name:
        if "dropout" in output_name:
            logger.info(f"CALLING FROM REGRESSION")
            select_indices = get_dropout_indices(out_path_base, dataset, 'train', sub, output_name)
            output_name = "train_"+output_name
        else:
            raise ValueError(f"Specified output name {output_name} isn't supported.")
    else:
        output_name = "train"
        select_indices = None
    
    path_vdvae_features_train = os.path.join(out_path_base, f'data/extracted_features/{dataset}/subj{sub}/vdvae_features_{output_name}_{num_latents}l.npz')
    logger.info(f"Loading train features from {path_vdvae_features_train} for regression.")
    vdvae_features_train = np.load(path_vdvae_features_train)

    train_latents = vdvae_features_train['latents']

    train_fmri, norm_mean_train, norm_scale_train = load_normalized_train_fmri(out_path_base, dataset, sub, input_avg=True, select_indices=select_indices)

    train_latents_mean = np.mean(train_latents,axis=0) 
    train_latents_std = np.std(train_latents, axis=0)

    logger.info('Training latents Feature Regression')

    reg = skl.Ridge(alpha=50000, max_iter=10000, fit_intercept=True)
    reg.fit(train_fmri, train_latents)

    if save_regression_weights:
        logger.info("Saving regression weights.")
        datadict = {
            'weight' : reg.coef_,
            'bias' : reg.intercept_,

        }
        base_path_regression = os.path.join(out_path_base,f'data/regression_weights/{dataset}/subj{sub}/')
        os.makedirs(base_path_regression, exist_ok=True)

        with open(os.path.join(base_path_regression,f"vdvae_regression_weights_{output_name}.pkl"),"wb") as f:
            pickle.dump(datadict,f)

    return reg, norm_mean_train, norm_scale_train, train_latents_mean, train_latents_std

def vdvae_predict(out_path_base, dataset, reg, norm_mean_train, norm_scale_train, train_latents_mean, train_latents_std, sub, name, num_latents=31, save_latents=True, output_name=""):
    # name would be one of 'test' and 'art'
    if output_name:
        if "dropout" in output_name:
            output_name = name+"_"+output_name
        else:
            raise ValueError(f"Specified output name {output_name} isn't supported.")
    else:
        output_name = name
    

    logger.info(f"Predicting {output_name} vdvae features from fmri for sub {sub}")

    true_vdvae_features = np.load(os.path.join(out_path_base, f'data/extracted_features/{dataset}/subj{sub}/vdvae_features_{name}_{num_latents}l.npz'))
    true_latents = true_vdvae_features['latents']

    fmri_path = os.path.join(out_path_base,f'data/processed_data/{dataset}/subj{sub}/{name}_fmriavg_general_sub{sub}.npy')
    fmri_data = np.load(fmri_path)

    fmri_data_normalized = (fmri_data - norm_mean_train) / norm_scale_train 

    pred_latent = reg.predict(fmri_data_normalized)
    std_norm_test_latent = (pred_latent - np.mean(pred_latent,axis=0)) / np.std(pred_latent,axis=0) # INFO LEAKAGE!
    pred_latents = std_norm_test_latent * train_latents_std + train_latents_mean
    
    logger.info(f"VDAVE Reg {output_name}")
    base_path = os.path.join(out_path_base,f"data/predicted_features/{dataset}/subj{sub}/")
    os.makedirs(base_path, exist_ok=True)
    if save_latents:
        np.savez(os.path.join(base_path,f'vdvae_general_pred_{output_name}_sub{sub}_{num_latents}l_alpha50k.npz'),pred_latents = pred_latents)
    return pred_latents

def vdvae_reconstruct_images(out_path_base, dataset, sub, name, num_latents=31, batch_size=25, output_name="", use_true_features=False):
    if output_name:
        if "dropout" in output_name:
            output_name = name+"_"+output_name
        elif 'true' in output_name:
            output_name = name+"_"+output_name
        else:
            raise ValueError(f"Specified output name {output_name} isn't supported.")
    else:
        output_name = name

    logger.info(f"Reconstructing vdvae {output_name} images for sub {sub}")
    ema_vae, preprocess_fn = get_ema_vae_and_preprocess_fn()
    
    image_path = os.path.join(out_path_base,f"data/processed_data/{dataset}/subj{sub}/{name}_stim_sub{sub}.npy") # test/art images are only loaded to get the number of images.
    images = batch_generator_external_images(data_path = image_path)

    vdvae_features = np.load(os.path.join(out_path_base, f'data/extracted_features/{dataset}/subj{sub}/vdvae_features_{name}_{num_latents}l.npz'), allow_pickle=True) # test/art latents need to be loaded only for the shape. Not for data.

    stats = vdvae_features["stats"]

    if use_true_features:
        logger.info("Using true Features for vdvae Reconstruction")
        pred_latents = vdvae_features['latents']
    else:
        pred_latents_all = np.load(os.path.join(out_path_base, f'data/predicted_features/{dataset}/subj{sub}/vdvae_general_pred_{output_name}_sub{sub}_{num_latents}l_alpha50k.npz'))
        pred_latents = pred_latents_all["pred_latents"]

    ref_latent = stats

    # Transform latents from flattened representation to hierarchical
    def latent_transformation(latents, ref):
        layer_dims = np.array([2**4,2**4,2**8,2**8,2**8,2**8,2**10,2**10,2**10,2**10,2**10,2**10,2**10,2**10,2**12,2**12,2**12,2**12,2**12,2**12,2**12,2**12,2**12,2**12,2**12,2**12,2**12,2**12,2**12,2**12,2**14])
        transformed_latents = []
        for i in range(31):
            t_lat = latents[:,layer_dims[:i].sum():layer_dims[:i+1].sum()]
            c,h,w=ref[i]['z'].shape[1:]
            transformed_latents.append(t_lat.reshape(len(latents),c,h,w))
        return transformed_latents

    n_images = len(images)
    idx = range(n_images)
    input_latent = latent_transformation(pred_latents[idx],ref_latent)
    
    def sample_from_hier_latents(latents,sample_ids):
        sample_ids = [id for id in sample_ids if id<len(latents[0])]
        layers_num=len(latents)
        sample_latents = []
        for i in range(layers_num):
            sample_latents.append(torch.tensor(latents[i][sample_ids]).float().cuda())
        return sample_latents

    def run_reco(n_images, input_latent, output_name):
        num_batches = int(np.ceil(n_images/batch_size))
        for i in tqdm(range(num_batches), desc=f"{output_name}: sample from hier latents"):
            samp = sample_from_hier_latents(input_latent,range(i*batch_size,(i+1)*batch_size))
            px_z = ema_vae.decoder.forward_manual_latents(len(samp[0]), samp, t=None)
            sample_from_latent = ema_vae.decoder.out_net.sample(px_z)
            base_path = os.path.join(out_path_base, f'results/vdvae/{dataset}/subj{sub}/{output_name}')
            os.makedirs(base_path, exist_ok=True)
            for j in range(len(sample_from_latent)):
                im = sample_from_latent[j]
                im = Image.fromarray(im)
                im = im.resize((512,512),resample=3)

                im.save(os.path.join(base_path, f'{i*batch_size+j}.png'))
    
    run_reco(n_images, input_latent, output_name)

def vdvae_main(out_path_base, dataset, sub, batch_size=25, include_art_dataset=False, output_name="", use_true_features=False):
    """ Main entrypoint for VDVAE computation. 
    First extracts the vdvae features for all datasets (train, test and deeprecon artificial images).

    Then the regression is trained on the training data. The trained regression is used to predict the latents of the test fmri and these latents are used to reconstruct the png images that serve as the foundation for the brain-diffuser shape of the images.

    See the original paper from Ozcelik for a visual explanation of the vdvae step.

    extract_features(train)
        - Extracts latents from the images
    train_regression(train)
        - Learns regression to map brain data to latents
        - reg.fit(train_fmri, train_latents)
        - Saves the regression weights if wanted

    extract_features(test)
        - Extracted to measure the prediction accuracy of the regression
    predict_latents(test_features)
        - Uses the trained regression to predict the latents from the test fmri data
    reconstruct_images(test_latents)
        - Reconstructs the images from the predicted latents
        - Reconstructed images are stored as png images under results/vdvae_images

    """
    if not use_true_features:
        train_latents, train_stats = vdvae_extract_features(out_path_base, dataset, sub, "train", output_name=output_name)
        reg, norm_mean_train, norm_scale_train, train_latents_mean, train_latents_std = vdvae_regression(out_path_base, dataset, sub, save_regression_weights=False, output_name=output_name)

    test_latents, test_stats = vdvae_extract_features(out_path_base, dataset, sub, "test") # output name isn't supposed to change test/art images
    if not use_true_features:
        test_pred_latents = vdvae_predict(out_path_base, dataset, reg, norm_mean_train, norm_scale_train, train_latents_mean, train_latents_std, sub, "test", output_name=output_name)
    
    vdvae_reconstruct_images(out_path_base, dataset, sub, "test", output_name=output_name, use_true_features=use_true_features)

    if include_art_dataset:
        art_latents, art_stats = vdvae_extract_features(out_path_base, dataset, sub, "art") # output name isn't supposed to change test/art images
        if not use_true_features:
            art_pred_latents = vdvae_predict(out_path_base, dataset, reg, norm_mean_train, norm_scale_train, train_latents_mean, train_latents_std, sub, "art", output_name=output_name)
        vdvae_reconstruct_images(out_path_base, dataset, sub, "art", output_name=output_name, use_true_features=use_true_features)


if __name__ == "__main__":
    ...
    logger.info("Executing re_vdvae")
    sub = "AM"
    batch_size=25
    out_path_base = "/home/matt/programming/recon_diffuser/"
    dataset="deeprecon"
    name = "test"
    # output_name = "dropout-random_0.1_00"
    output_name = ""

    logging.basicConfig(
        format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=[
            logging.StreamHandler()
        ],
        level=logging.INFO)

    # vdvae_main(out_path_base, dataset, sub, batch_size, include_art_dataset=True, output_name=output_name)
    vdvae_main(out_path_base, dataset, sub, batch_size, include_art_dataset=True)

