"""
Serves as the main entrypoint for the whole pipeline computation for the deeprecon dataset.

Attributes:
    sub (str): The subject ID from the deeprecon dataset.
    out_path_base (str): The base for the output of the scripts. Intermediate results and models will be saved here.
    dataset (str): Should be 'deeprecon'. This is used by the other scripts to create the files at the correct folders.
    deeprecon_data_root (str): The base folder where the deeprecon fmri input data is stored.
    image_data_root (str): The base folder where the image data for the deeprecon stimuli is stored.
    captions_data_root (str): The base folder where the captions for the deeprecon images is stored.
"""
import os
import logging
from collections import defaultdict
import pickle
from functools import partial

import PIL
import pandas as pd
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
import numpy as np
from re_utils import  get_dropout_indices, batch_generator_external_images, load_normalized_train_fmri
import sklearn.linear_model as skl
import torch.optim as optim
from scipy import spatial

from bdpy.dl.torch.models import  layer_map, model_factory
from bdpy.dl.torch import FeatureExtractor
from bdpy.recon.utils import normalize_image, clip_extreme
from bdpy.recon.torch.icnn import reconstruct

logger = logging.getLogger("recdi_icnn_main")

def feature_scaling(feat, feat_mean_train, feat_std_train, layer_mapping):
    std_ddof = 1
    channel_axis = 0
    for layer, ft in feat.items():
        layer_name_in_train_mean = layer_mapping[layer]
        a_feat = ft[0]
        # Get proper axis_along for fc/conv layers
        if a_feat.ndim == 1:
            axes_along = None
        else:
            axes = list(range(a_feat.ndim))
            axes.remove(channel_axis)
            axes_along = tuple(axes)
        a_feat = a_feat - feat_mean_train[layer_name_in_train_mean][0]
        feat_std = np.mean(np.std(a_feat, axis=axes_along, ddof=std_ddof, keepdims=True), keepdims=True)
        a_feat = (a_feat / feat_std) * np.mean(feat_std_train[layer_name_in_train_mean])
        a_feat = a_feat + feat_mean_train[layer_name_in_train_mean][0]
        # Adjust dimension
        ft = a_feat[np.newaxis]
        feat.update({layer: ft})


def load_generator(device):
    generator_params_path = '/home/kiss/data/models_shared/pytorch/bvlc_reference_caffenet/generators/ILSVRC2012_Training/relu7/generator.pt'
    # Generator model
    generator = model_factory('relu7generator')
    generator.to(device)
    generator.load_state_dict(torch.load(generator_params_path))
    generator.eval()
    return generator

def load_encoder(device):
    encoder = model_factory('vgg19')
    encoder.to(device)
    encoder.load_state_dict(torch.load('/home/kiss/data/models_shared/pytorch/VGG_ILSVRC_19_layers/VGG_ILSVRC_19_layers.pt'))
    encoder.eval()
    return encoder

def dist_loss(
        x: torch.Tensor, y: torch.Tensor,
        layer: str,
        weight=1,
        alpha={}, beta={},
        c1=1e-6, c2=1e-6
) -> torch.Tensor:
    if 'fc' in layer:
        x_mean = x.mean()
        y_mean = y.mean()
        x_var = ((x - x_mean) ** 2).mean()
        y_var = ((y - y_mean) ** 2).mean()
        xy_cov = (x * y).mean() - x_mean * y_mean
    else:
        x_mean = x.mean([2, 3], keepdim=True)
        y_mean = y.mean([2, 3], keepdim=True)
        x_var = ((x - x_mean) ** 2).mean([2, 3], keepdim=True)
        y_var = ((y - y_mean) ** 2).mean([2, 3], keepdim=True)
        xy_cov = (x * y).mean([2, 3], keepdim=True) - x_mean * y_mean
    s1 = (2 * x_mean * y_mean + c1) / (x_mean**2 + y_mean**2 + c1)
    dist1 = s1.mean() * alpha[layer]
    s2 = (2 * xy_cov + c2) / (x_var + y_var + c2)
    dist2 = s2.mean() * beta[layer]
    dist = (dist1 + dist2) * weight
    return -dist

def get_opts(n_iter=200):

    # Average image of ImageNet
    image_mean = np.load('/home/kiss/data/models_shared/caffe/VGG_ILSVRC_19_layers/ilsvrc_2012_mean.npy')
    image_mean = np.float32([image_mean[0].mean(), image_mean[1].mean(), image_mean[2].mean()])
    # DIST loss settings -----------------------------------------------------
    dists_weight = 343639
    dists_alpha = {
        'fc8':     0.06661523244761258,
        'fc7':     0.03871265134313895,
        'fc6':     0.000629031742843134,
        'conv5_4': 0.5310432634795051,
        'conv5_3': 0.01975314483213067,
        'conv5_2': 0.714010791024532,
        'conv5_1': 0.08536182104218124,
        'conv4_4': 0.030798346318926202,
        'conv4_3': 0.004025735147052829,
        'conv4_2': 0.0021716504774059618,
        'conv4_1': 0.02880295296139471,
        'conv3_4': 0.014169279688225732,
        'conv3_3': 0.00019573287900056505,
        'conv3_2': 0.0004887923569668929,
        'conv3_1': 0.006857140440209977,
        'conv2_2': 0.08084213863904581,
        'conv2_1': 0.00024056214287883663,
        'conv1_2': 0.003886371646003732,
        'conv1_1': 0.009952859626673973,
    }
    dists_beta = {
        'fc8': 0.008304840607742099,
        'fc7': 0.044481711593671994,
        'fc6': 0.038457933646483915,
        'conv5_4': 0.0012780195483159135,
        'conv5_3': 0.0018775814111698145,
        'conv5_2': 0.5074163077203029,
        'conv5_1': 0.002337825161420017,
        'conv4_4': 0.7100372437615771,
        'conv4_3': 0.5166895849277143,
        'conv4_2': 0.03998274022264576,
        'conv4_1': 0.04328555659354602,
        'conv3_4': 0.024733951474856346,
        'conv3_3': 0.0004859871528150426,
        'conv3_2': 0.039778524165843814,
        'conv3_1': 0.0002639605292406699,
        'conv2_2': 0.02472305546171304,
        'conv2_1': 0.12888847991806807,
        'conv1_2': 0.008627502425502372,
        'conv1_1': 0.000865427897168344
    }

    # Reconstruction options -------------------------------------------------

    opts = {
        "loss_func": torch.nn.MSELoss(reduction="sum"),
        "custom_layer_loss_func": partial(dist_loss, weight=dists_weight, alpha=dists_alpha, beta=dists_beta),
        "n_iter": n_iter,
        "lr": (2., 1e-10),
        "momentum": (0.9, 0.9),
        "decay": (0.01, 0.01),
        "blurring": False,
        "channels": None,
        "masks": None,
        "disp_interval": 1,
    }
    initial_gen_feat = np.random.normal(0, 1, [4096]) # Taken from recon icnn
    upper_bound = np.loadtxt('/home/kiss/data/models_shared/pytorch/bvlc_reference_caffenet/generators/ILSVRC2012_Training/relu7/act_range_3x.txt', delimiter=" ")
    upper_bound = upper_bound.reshape([4096])

    opts.update({
        # The initial image for the optimization (setting to None will use random noise as initial image)
        "initial_feature": initial_gen_feat,
        "feature_upper_bound": upper_bound,
        "feature_lower_bound": 0.,
    })
    return opts


def image_preprocess(img, image_mean=np.float32([104, 117, 123])):
    """Convert to Caffe's input image layout."""
    return np.float32(np.transpose(img, (2, 0, 1))[::-1]) - np.reshape(image_mean, (3, 1, 1))


def image_deprocess(img, image_mean=np.float32([104, 117, 123])):
    """Donvert from Caffe's input image layout."""
    return np.dstack((img + np.reshape(image_mean, (3, 1, 1)))[::-1])
##

def icnn_main(out_path_base, dataset, sub, device, save_features=True, output_name="", resize_size=224, scale_features=True, use_cached_features=True, n_iter=500, true_features_reconstruction=False):
    print(f"Save features is {save_features} setting. Setting it to True")
    save_features = True

    include_art_dataset= dataset=="deeprecon"

    encoder = load_encoder(device)
    to_layer_name = layer_map("vgg19")
    layer_names = list(to_layer_name.values())
    layer_mapping = to_layer_name
    inv_layer_mapping = {v: k for k, v in layer_mapping.items()}
    feature_extractor = FeatureExtractor(encoder, layer_names, device=device, detach=True)

    if output_name:
        if "dropout" in output_name:
            select_indices = get_dropout_indices(out_path_base, dataset, 'train', sub, output_name)
        elif 'true' in output_name:
            select_indices = None
        else:
            raise ValueError(f"Specified output name {output_name} isn't supported.")
    else:
        select_indices = None

    output_name = "_".join((output_name, f"size{resize_size}", f"iter{n_iter}"))
    if scale_features:
        output_name += "_scaled"

    all_names = ['test', 'art', 'train'] if include_art_dataset else ['test', 'train']
    test_names = [name for name in all_names if name != "train"]

    all_icnn_features = {}
    for name in all_names: 
        logger.info(f"Extracting {name} icnn features for sub {sub}")
        select_indices_for_name = select_indices if name == "train" else None
        path_processed = os.path.join(out_path_base, f'data/processed_data/{dataset}/subj{sub}/')
        image_path = os.path.join(path_processed,f'{name}_stim_sub{sub}.npy')
        images = batch_generator_external_images(data_path = image_path, resize_size=resize_size, select_indices=select_indices_for_name, icnn_preproc=True)
        logger.info(f"Created batch generator for {name} data with {len(images)} images.")

        batch_size = 50
        dataloader = DataLoader(images, batch_size=batch_size, shuffle=False)
        logger.info("Dataloaders created")

        icnn_features = defaultdict(list)
        for i,x in tqdm(enumerate(dataloader), desc=f"Computing Features for {name} {dataset}.", total=len(dataloader)):
            with torch.no_grad():
                    x = x.to(device)
                    fts = feature_extractor(x)
                    for ft_name, ft_value in fts.items():
                        icnn_features[ft_name].append(ft_value)
        
        # make a numpy array out of the icnn_features
        all_shapes = []
        for ft_name, ft_value_list in tqdm(icnn_features.items(), desc="Concatenating the latent features..."):
            ft_value_arr = np.concatenate(ft_value_list)
            icnn_features[ft_name] = ft_value_arr
            all_shapes.append(ft_value_arr.shape)

        all_icnn_features[name] = icnn_features
    
    # Save the extracted Features for test and art dataset
    for test_name in test_names:
        extracted_feature_dir = os.path.join(out_path_base,f'data/extracted_features/{dataset}/subj{sub}/')
        os.makedirs(extracted_feature_dir, exist_ok = True)
        extracted_feature_fname = f'{test_name}_icnn_extracted_features.npy'
        extracted_feature_path = os.path.join(extracted_feature_dir, extracted_feature_fname)
        if os.path.exists(extracted_feature_path):
            logger.info(f"Extracted (True) Features at {extracted_feature_path} exist already. Won't save them again.")
        else:
            logger.info(f"Saving extracted (True) Features to {extracted_feature_path}.")
            np.save(extracted_feature_path, all_icnn_features[test_name])    
    del encoder 
    del feature_extractor
    torch.cuda.empty_cache()
    
    if not true_features_reconstruction:
        stds_train = {}
        means_train = {}
        # Compute norms for training features
        for ft_name, ft_value_arr in tqdm(all_icnn_features['train'].items(), desc="Computing norms"):
            y_mean_train = np.mean(ft_value_arr, axis=0)[np.newaxis, :]
            y_std_train = np.std(ft_value_arr, axis=0, ddof=1)[np.newaxis, :]
            means_train[ft_name] = y_mean_train
            stds_train[ft_name] = y_std_train

        
        # normalize the train data only (the other features are computed as true features)
        for ft_name, ft_value_arr in tqdm(all_icnn_features['train'].items(), desc="Normalizing data"):
            y_mean_train = means_train[ft_name]
            y_std_train = stds_train[ft_name]
            all_icnn_features['train'][ft_name] = (all_icnn_features['train'][ft_name] - y_mean_train) / y_std_train

        train_fmri, fmri_mean_train, fmri_norm_train = load_normalized_train_fmri(out_path_base, dataset, sub, input_avg=True, select_indices=select_indices)
        train_fmri = train_fmri.astype(np.float32)
        fmri_mean_train = fmri_mean_train.astype(np.float32)
        fmri_norm_train = fmri_norm_train.astype(np.float32)

        test_fmris = {}
        for test_name in test_names:
            fmri_path = os.path.join(out_path_base,f'data/processed_data/{dataset}/subj{sub}/{test_name}_fmriavg_general_sub{sub}.npy')
            fmri_data = np.load(fmri_path)
            fmri_data_normalized = (fmri_data - fmri_mean_train) / fmri_norm_train 
            test_fmris[test_name] = fmri_data_normalized

        if resize_size == 50:
            output_file_name = f"pred_icnn-features_{output_name}_SMALL.pkl"
        else:
            output_file_name = f"pred_icnn-features_{output_name}.pkl"
        
        pred_feature_path = os.path.join(out_path_base, 'data', 'predicted_features', dataset, f"subj{sub}", output_file_name)

        if os.path.exists(pred_feature_path) and use_cached_features:
            logger.info(f"The features exist already. Loading them from {pred_feature_path}")

            predicted_features_pd = pd.read_pickle(pred_feature_path)
            if len(predicted_features_pd['test']['features[0]']) != 64:
                raise ValueError("Wtf man. Not cool")
            predicted_features = predicted_features_pd
            logger.info("Done loading features")
        else:
            #### Predict all the features
            if use_cached_features:
                logger.info(f"Features for the given configuration don't exist at {pred_feature_path} yet. Now I have to predict them.")
            logger.info("Predicting all the features...")
            predicted_features = defaultdict(lambda: defaultdict(list))
            for ft_name, ft_value_arr in all_icnn_features['train'].items():
                X = train_fmri
                logger.info(f"Doing regression and prediction for feature: {ft_name}")
                for chunk_n in tqdm(range(ft_value_arr.shape[1]), "Doing all chunks...", total=ft_value_arr.shape[1]):
                    if len(ft_value_arr.shape) > 2:
                        ft_value_chunk = ft_value_arr[:,chunk_n]
                        ft_value_chunk = ft_value_chunk.reshape(*ft_value_chunk.shape[:-2], -1)
                    else: # last layers are 1d anyways, no need to chunk them.
                        ft_value_chunk = ft_value_arr
                        ft_value_chunk[np.isnan(ft_value_chunk)] = 0 # There were some nans. 
                    Y = ft_value_chunk
                    reg = skl.Ridge(alpha=50000, max_iter=10000, fit_intercept=True)
                    reg.fit(X, Y)

                    for test_name in test_names:
                        test_name_fmri = test_fmris[test_name]
                        test_name_pred = reg.predict(test_name_fmri)
                        predicted_features[test_name][ft_name].append(test_name_pred)
                        
                    if len(ft_value_arr.shape) == 2:
                        logger.info(f"breaking layer with shape {ft_value_arr.shape}")
                        break
            predicted_features_dict = {}
            # Turn the defaultdict to a dict again lol
            for test_name in test_names:
                predicted_features_dict[test_name] = dict(predicted_features[test_name])

            # SAVING FEATURES TO DISK
            if save_features:
                logger.info(f"Saving the features to disk in file {pred_feature_path}")
                with open(pred_feature_path, "wb") as f:
                    pickle.dump(predicted_features_dict, f)
                logger.info(f"Done")
            else:
                logger.info(f"Not saving features.")

        ### End of the feature prediction
        
        # feature mapping (shapes)
        layer_shape_mapping = {
            'features[0]': (1200, 64, resize_size, resize_size),
            'features[2]': (1200, 64, resize_size, resize_size),
            'features[5]': (1200, 128, resize_size//2, resize_size//2),
            'features[7]': (1200, 128, resize_size//2, resize_size//2),
            'features[10]': (1200, 256, resize_size//4, resize_size//4),
            'features[12]': (1200, 256, resize_size//4, resize_size//4),
            'features[14]': (1200, 256, resize_size//4, resize_size//4),
            'features[16]': (1200, 256, resize_size//4, resize_size//4),
            'features[19]': (1200, 512, resize_size//8, resize_size//8),
            'features[21]': (1200, 512, resize_size//8, resize_size//8),
            'features[23]': (1200, 512, resize_size//8, resize_size//8),
            'features[25]': (1200, 512, resize_size//8, resize_size//8),
            'features[28]': (1200, 512, resize_size//16, resize_size//16),
            'features[30]': (1200, 512, resize_size//16, resize_size//16),
            'features[32]': (1200, 512, resize_size//16, resize_size//16),
            'features[34]': (1200, 512, resize_size//16, resize_size//16),
            'classifier[0]': (1200, 4096),
            'classifier[1]': (1200, 4096),
            'classifier[3]': (1200, 4096),
            'classifier[4]': (1200, 4096),
            'classifier[6]': (1200, 1000)}

        for test_name in test_names:
            evaluation_metrics = []
            for ft_name in list(predicted_features[test_name].keys()):
                ft_array_list = predicted_features[test_name][ft_name]
                if len(ft_array_list) == 2:
                    logger.info("WARNING: There was an error in the classifier layers. I'm sorry. Need to cut down the size of the list to one")
                    ft_array_list = [ft_array_list[0]]
                layer_shape = list(layer_shape_mapping[ft_name])
                ft_name_nice = inv_layer_mapping[ft_name]
                num_test_images = ft_array_list[0].shape[0]
                layer_shape[0] = num_test_images
                stacked_array = np.stack(ft_array_list, axis=1)  # Shape: (50, 64, 2500)
                reshaped_array = stacked_array.reshape(layer_shape).astype(np.float32)
                # denormalize the values
                train_mean = means_train[ft_name]
                train_std = stds_train[ft_name]

                feat_flat_true = all_icnn_features[test_name][ft_name]
                feat_flat_true = feat_flat_true.reshape(feat_flat_true.shape[0], np.prod(feat_flat_true.shape[1:]))
                reshaped_array = (reshaped_array * train_std) + train_mean
                ft_name_nice = inv_layer_mapping[ft_name]

                assert reshaped_array.shape == tuple(layer_shape)
                predicted_features[test_name][ft_name] = reshaped_array
                # rename the key to the human readable one again
                predicted_features[test_name][ft_name_nice] = predicted_features[test_name].pop(ft_name)
            # Save the dataframe of the evaluation metrics for the specific test_dataset in the corresponding results folder
            save_dir = os.path.join(out_path_base, 'results', 'icnn', dataset, f'subj{sub}', "_".join((test_name, output_name)))
    else:
        logger.info("I'm just doing true feature reconstruction, so no prediction here nice. ")
    # now the predicted features have been computed

    #### True features, recon loop begins.
    for test_name in test_names:
            
        save_dir = os.path.join(out_path_base, 'results', 'icnn', dataset, f'subj{sub}', "_".join((test_name, output_name)))
        from copy import deepcopy
        true_features = deepcopy(all_icnn_features[test_name])

        # Rename the layers of the true features to the human friendly ones
        for ft_name in list(true_features.keys()):
            ft_name_nice = inv_layer_mapping[ft_name]
            true_features[ft_name_nice] = true_features.pop(ft_name)

        n_images = next(iter(true_features.values())).shape[0]

        encoder_layers = ['conv1_1', 'conv1_2', 'conv2_1', 'conv2_2', 'conv3_1', 'conv3_2', 'conv3_3', 'conv3_4', 'conv4_1', 'conv4_2', 'conv4_3', 'conv4_4', 'conv5_1', 'conv5_2', 'conv5_3', 'conv5_4', 'fc6', 'fc7', 'fc8']

        image_size = [resize_size, resize_size, 3]

        logger.info(f"Starting reconstruction with {n_images} images. and ")

        test_results = {}
        image_n = 0
        for image_n in range(n_images):
            snapshot_dir = os.path.join(save_dir, 'snapshots', f"{image_n}")
            os.makedirs(snapshot_dir, exist_ok=True)

            if true_features_reconstruction:
                recon_feat = {k:np.expand_dims(v[image_n],0) for k,v in true_features.items() if k in encoder_layers}
            else:
                recon_feat = {k:np.expand_dims(v[image_n],0) for k,v in predicted_features[test_name].items() if k in encoder_layers}
                if scale_features:
                    feature_scaling(recon_feat, means_train, stds_train, layer_mapping)
            
            opts = get_opts(n_iter=n_iter)
            # only pick the 19 layers that are expected in the former script

            ### Add things to ops
            # Norm of the DNN features for each layer
            feat_norm = np.array(
                [np.linalg.norm(recon_feat[layer]) for layer in encoder_layers],
                dtype="float32"
            )

            # Weight of each layer in the total loss function
            # Use the inverse of the squared norm of the DNN features as the
            # weight for each layer
            weights = 1. / (feat_norm ** 2)

            # Normalise the weights such that the sum of the weights = 1
            weights = weights / weights.sum()
            layer_weights = dict(zip(encoder_layers, weights))

            opts.update({"layer_weights": layer_weights})

            encoder = load_encoder(device)
            generator = load_generator(device)

            # Do reconstruction
            recon_image, loss_hist = reconstruct(
                recon_feat,
                encoder,
                generator=generator,
                layer_mapping=layer_mapping,
                optimizer=optim.SGD,
                image_size=image_size,
                crop_generator_output=True,
                preproc=image_preprocess,
                postproc=image_deprocess,
                output_dir=save_dir,
                save_snapshot=False,
                snapshot_dir=snapshot_dir,
                snapshot_interval=10,
                snapshot_ext="jpg",
                snapshot_postprocess=normalize_image,
                return_loss=True,
                device=device,
                **opts
            )

            out_file_name = str(image_n) + "." + "png"
            recon_image_normalized_file = os.path.join(save_dir, out_file_name)
            reconstructed_image = PIL.Image.fromarray(normalize_image(clip_extreme(recon_image, pct=4)))
            reconstructed_image.save(recon_image_normalized_file)
            logger.info(f"Saved reconstructed image to {recon_image_normalized_file}")
        
    return icnn_features


if __name__ == "__main__":
    sub = "AM"
    sub = "KS"
    subs = ["KS"]
    # subs = ['AM']

    fname_log = f'loglog{subs}.log'
    for sub in subs:
        out_path_base = "/home/matt/programming/recon_diffuser/"
        dataset="deeprecon"
        output_name = ""
        device = 'cuda:0'
        resize_size = 50
        use_cached_features = True
        n_iter = 200 
        scale_features = True

        logging.basicConfig(
            format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S',
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler(fname_log)
            ],
            level=logging.INFO)
        
        logger.info(f"Starting deeprecon main with {dataset=} {sub=}")
        
        deeprecon_data_root = "/home/kiss/data/fmri_shared/datasets/Deeprecon/fmriprep"
        image_data_root = "/home/kiss/data/contents_shared"
        captions_data_root = "/home/matt/programming/recon_diffuser/data/annots"

        icnn_main(out_path_base, dataset, sub, device, save_features=True, output_name=output_name, resize_size=resize_size, scale_features=True, use_cached_features=use_cached_features, n_iter=n_iter)

