import os
from datetime import datetime
from torch.utils.data import DataLoader, Dataset
import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image
from vdvae.model_utils import set_up_data, load_vaes
from versatile_diffusion.lib.cfg_helper import model_cfg_bank
from versatile_diffusion.lib.model_zoo import get_model

import logging 

logger = logging.getLogger("recdi_utils")

def simple_log(l):
    print(f"{datetime.now()}: {l}")


def image_preprocess(img, image_mean=np.float32([104, 117, 123])):
    """Convert to Caffe's input image layout."""
    return np.float32(np.transpose(img, (2, 0, 1))[::-1]) - np.reshape(image_mean, (3, 1, 1))

def get_ema_vae_and_preprocess_fn():
    simple_log("Loading ema vae...")
    H = {'image_size': 64, 'image_channels': 3,'seed': 0, 'port': 29500, 'save_dir': './saved_models/test', 'data_root': './', 'desc': 'test', 'hparam_sets': 'imagenet64', 'restore_path': 'imagenet64-iter-1600000-model.th', 'restore_ema_path': 'vdvae/model/imagenet64-iter-1600000-model-ema.th', 'restore_log_path': 'imagenet64-iter-1600000-log.jsonl', 'restore_optimizer_path': 'imagenet64-iter-1600000-opt.th', 'dataset': 'imagenet64', 'ema_rate': 0.999, 'enc_blocks': '64x11,64d2,32x20,32d2,16x9,16d2,8x8,8d2,4x7,4d4,1x5', 'dec_blocks': '1x2,4m1,4x3,8m4,8x7,16m8,16x15,32m16,32x31,64m32,64x12', 'zdim': 16, 'width': 512, 'custom_width_str': '', 'bottleneck_multiple': 0.25, 'no_bias_above': 64, 'scale_encblock': False, 'test_eval': True, 'warmup_iters': 100, 'num_mixtures': 10, 'grad_clip': 220.0, 'skip_threshold': 380.0, 'lr': 0.00015, 'lr_prior': 0.00015, 'wd': 0.01, 'wd_prior': 0.0, 'num_epochs': 10000, 'n_batch': 4, 'adam_beta1': 0.9, 'adam_beta2': 0.9, 'temperature': 1.0, 'iters_per_ckpt': 25000, 'iters_per_print': 1000, 'iters_per_save': 10000, 'iters_per_images': 10000, 'epochs_per_eval': 1, 'epochs_per_probe': None, 'epochs_per_eval_save': 1, 'num_images_visualize': 8, 'num_variables_visualize': 6, 'num_temperatures_visualize': 3, 'mpi_size': 1, 'local_rank': 0, 'rank': 0, 'logdir': './saved_models/test/log'}

    class dotdict(dict):
        """dot.notation access to dictionary attributes"""
        __getattr__ = dict.get
        __setattr__ = dict.__setitem__
        __delattr__ = dict.__delitem__
    H = dotdict(H)

    H, preprocess_fn = set_up_data(H)

    simple_log('ema vae model is Loading')
    ema_vae = load_vaes(H)
    simple_log('Loaded ema vae')
    return ema_vae, preprocess_fn

class batch_generator_external_images(Dataset):
    def __init__(self, data_path, clipvision=False, resize_size=64, select_indices=None, icnn_preproc = False):
        # if select indices (for dropout), the given indices will be selected first to remove them from the dataset.
        self.data_path = data_path
        self.im = np.load(data_path).astype(np.uint8)
        if select_indices is not None:
            self.im = self.im[select_indices]
        self.clipvision = clipvision
        self.resize_size = resize_size
        self.icnn_preproc = icnn_preproc

    def __getitem__(self,idx):
        img = Image.fromarray(self.im[idx])
        if self.icnn_preproc:
            img = img.resize((self.resize_size,self.resize_size))
        else:
            img = T.functional.resize(img,(self.resize_size,self.resize_size))
        if self.clipvision:
            img = T.functional.to_tensor(img).float()
        elif self.icnn_preproc:
            img = image_preprocess(img)
        else:
            img = torch.tensor(np.array(img)).float()
        #img = img/255
        if self.clipvision:
            img = img*2 - 1
        return img
    def __len__(self):
        return len(self.im)

def load_normalized_train_fmri(out_path_base, dataset, sub, input_avg, select_indices=None):
    if input_avg:
        train_path = os.path.join(out_path_base, f'data/processed_data/{dataset}/subj{sub}/train_fmriavg_general_sub{sub}.npy')
    else:
        train_path = os.path.join(out_path_base, f'data/processed_data/{dataset}/subj{sub}/train_fmri_noavg_general_sub{sub}.npy')
        
    logging.info(f"Opening train fmri from path {train_path}")
    train_fmri = np.load(train_path)

    if select_indices is not None:
        logging.info(f"Selecting dropout indices from train fmri...")
        train_fmri = train_fmri[select_indices]

    if len(train_fmri.shape) == 3:
        # we have multiple fmri recordings per image
        train_fmri_avg = train_fmri.mean(axis=1).shape
        norm_mean_train = np.mean(train_fmri_avg, axis=0)
        norm_scale_train = np.std(train_fmri_avg, axis=0, ddof=1)
        train_fmri_normalized = (train_fmri - norm_mean_train) / norm_scale_train
    elif len(train_fmri.shape) == 2:
        norm_mean_train = np.mean(train_fmri, axis=0)
        norm_scale_train = np.std(train_fmri, axis=0, ddof=1)
        train_fmri_normalized = (train_fmri - norm_mean_train) / norm_scale_train
    else:
        raise ValueError("Fmri shape needs to have either 2 or 3 dimensions.")
    ## Preprocessing fMRI

    return train_fmri_normalized, norm_mean_train, norm_scale_train

def get_dropout_indices(out_path_base, dataset, name, sub, output_name):
    if name != 'train':
        logger.warning("You're doing a dropout, even though it's not the train dataset. You sure it's correct?")

    dropout_sample_path = os.path.join(out_path_base, 'data/dropout_samples', dataset)
    selected_img_names_path = os.path.join(dropout_sample_path, output_name+'.npy')
    select_img_names = np.load(selected_img_names_path)
    im_idx_path = os.path.join(out_path_base, f'data/processed_data/{dataset}/subj{sub}/', f'{name}_im_idx_sub{sub}.npy')
    sub_im_idx = np.load(im_idx_path)
    select_indices = np.array([np.where(sub_im_idx == value)[0][0] for value in select_img_names])
    return select_indices

def load_clip_model():
    logger.info("Loading Clip model")
    cfgm_name = 'vd_noema'
    pth = 'versatile_diffusion/pretrained/vd-four-flow-v1-0-fp16-deprecated.pth'
    cfgm = model_cfg_bank()(cfgm_name)
    net = get_model()(cfgm)
    sd = torch.load(pth, map_location='cpu')
    net.load_state_dict(sd, strict=False)    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device {device}")
    net.clip = net.clip.to(device)
    return net