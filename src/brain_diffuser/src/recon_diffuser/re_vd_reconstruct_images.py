"""
This module provides the functionality for the image reconstruction step of the brain-diffuser algorithm.
It combines the information from the vdvae, cliptext and clipvision models to produce the final output images. 

Entrypoint is the vd_reconstruct_main function. The functionality is described there.
"""


import os
import PIL
from PIL import Image
import numpy as np
from tqdm import tqdm
import logging 

import torch
import torchvision.transforms as tvtrans
from versatile_diffusion.lib.model_zoo.ddim_vd import DDIMSampler_VD
from versatile_diffusion.lib.experiments.sd_default import color_adjust

from re_utils import load_clip_model
from re_ai_captions_utils import get_prompt_name_and_params_from_output_name, parse_true_feat_output_name

logger = logging.getLogger("recdi_vd_recon")

def regularize_image(x, im_size):
        BICUBIC = PIL.Image.Resampling.BICUBIC
        if isinstance(x, str):
            x = Image.open(x).resize([im_size, im_size], resample=BICUBIC)
            x = tvtrans.ToTensor()(x)
        elif isinstance(x, PIL.Image.Image):
            x = x.resize([im_size, im_size], resample=BICUBIC)
            x = tvtrans.ToTensor()(x)
        elif isinstance(x, np.ndarray):
            x = PIL.Image.fromarray(x).resize([im_size, im_size], resample=BICUBIC)
            x = tvtrans.ToTensor()(x)
        elif isinstance(x, torch.Tensor):
            pass
        else:
            assert False, 'Unknown image type'

        assert (x.shape[1]==im_size) & (x.shape[2]==im_size), \
            'Wrong image size'
        return x

def bad_shuffle_array(arr):
    original_shape = arr.shape
    flattened = arr.ravel()
    np.random.shuffle(flattened)
    shuffled = flattened.reshape(original_shape)
    return shuffled

def reconstruct_images(out_path_base, dataset, sub, name, clip_model, strength, mixing, scale, ddim_steps, ddim_eta, im_size, n_samples, output_name="", use_true_features=False):

    if output_name: 
        model_name = f"{name}_{output_name}" # for example train_perturb_5050_friendly 
    else:
        model_name = name

    badshuffle = False
    use_true_text_feat = False
    use_baseline_vdvae = False
    use_baseline_cliptext = False
    if output_name.startswith('aicap'):
        logger.info("AiCap experiment. Using baseline vdvae")
        use_baseline_vdvae = True
        prompt_name, params = get_prompt_name_and_params_from_output_name(output_name)
        if 'mix' in params:
            mixing = params['mix']
            logger.info(f"Setting mix to {mixing}.")
        if 'badshuffle' in params:
            logger.info(f"Bad shuffle is on. {params['badshuffle']}")
            badshuffle = params['badshuffle']
        if 'usetruecliptext' in params:
            logger.info(f"Usetruecliptext. {params['usetruecliptext']}")
            use_true_text_feat = params['usetruecliptext']
    elif output_name.startswith('true'):
        algorithm, params = parse_true_feat_output_name(output_name)
        if 'mix' in params:
            mixing = params['mix']
            logger.info(f"Setting mix to {mixing}.")
    elif output_name.startswith('fgsm') or output_name.startswith('ic'):
        logger.info("AdvPert experiment. Using baseline vdvae")
        use_baseline_vdvae = True
        use_baseline_cliptext = True

    if use_true_features:
        logger.info("Doing true feature reconstruction VD.")
        # variables are called pred, even though it's actually extracted not predicted here.
        pred_text_path = os.path.join(out_path_base, f'data/extracted_features/{dataset}/subj{sub}/cliptext_{name}.npy')
        pred_text = np.load(pred_text_path)
    elif use_true_text_feat:
        logger.info("Doing true feature reconstruction VD.")
        # variables are called pred, even though it's actually extracted not predicted here.
        pred_text_path = os.path.join(out_path_base, f'data/extracted_features/{dataset}/subj{sub}/cliptext_{model_name}.npy')
        pred_text = np.load(pred_text_path)
    elif use_baseline_cliptext:
        logger.info("Using baseline cliptext features")
        pred_text_path = os.path.join(out_path_base, f'data/predicted_features/{dataset}/subj{sub}/cliptext_pred{name}_general.npy')
        pred_text = np.load(pred_text_path)
    else:
        pred_text_path = os.path.join(out_path_base, f'data/predicted_features/{dataset}/subj{sub}/cliptext_pred{model_name}_general.npy')
        pred_text = np.load(pred_text_path)
        if badshuffle:
            pred_text = bad_shuffle_array(pred_text)

    logger.info(f"Loaded clip-text preds from {pred_text_path}.")
    pred_text = torch.tensor(pred_text).half().cuda(0)

    if use_true_features:
        vision_pred_path = os.path.join(out_path_base, f'data/extracted_features/{dataset}/subj{sub}/clipvision_{name}.npy')
        pred_vision = np.load(vision_pred_path)
    else:
        vision_pred_path = os.path.join(out_path_base, f'data/predicted_features/{dataset}/subj{sub}/clipvision_pred{model_name}_general.npy')
        pred_vision = np.load(vision_pred_path)
    logger.info(f"Loaded clip-vision preds from {vision_pred_path}.")

    pred_vision = torch.tensor(pred_vision).half().cuda(0)

    sampler = DDIMSampler_VD
    sampler = sampler(clip_model)
    batch_size = 1

    xtype = 'image'
    ctype = 'prompt'

    for im_id in tqdm(range(len(pred_vision)), desc="Reconstructing images"):

        # since true_bd is also an output name, the true features are taken automatically.       
        if use_baseline_vdvae:
            vdvae_image_path =  os.path.join(out_path_base, f'results/vdvae/{dataset}/subj{sub}/{name}/{im_id}.png')
        else:
            vdvae_image_path =  os.path.join(out_path_base, f'results/vdvae/{dataset}/subj{sub}/{model_name}/{im_id}.png')
        zim = Image.open(vdvae_image_path)
        logger.info(f"Opened vdvae image from {vdvae_image_path}")
    
        zim = regularize_image(zim, im_size)
        zin = zim*2 - 1
        zin = zin.unsqueeze(0).cuda(0).half()

        init_latent = clip_model.autokl_encode(zin)
        
        sampler.make_schedule(ddim_num_steps=ddim_steps, ddim_eta=ddim_eta, verbose=False)
        #strength=0.75
        assert 0. <= strength <= 1., 'can only work with strength in [0.0, 1.0]'
        t_enc = int(strength * ddim_steps)
        device = 'cuda:0'
        z_enc = sampler.stochastic_encode(init_latent, torch.tensor([t_enc]).to(device))

        dummy = ''
        utx = clip_model.clip_encode_text(dummy)
        utx = utx.cuda(0).half()
        
        dummy = torch.zeros((1,3,224,224)).cuda(0)
        uim = clip_model.clip_encode_vision(dummy)
        uim = uim.cuda(0).half()
        
        z_enc = z_enc.cuda(0)

        h, w = im_size,im_size
        shape = [n_samples, 4, h//8, w//8]

        cim = pred_vision[im_id].unsqueeze(0)
        ctx = pred_text[im_id].unsqueeze(0)
                
        sampler.model.model.diffusion_model.device='cuda:0'
        sampler.model.model.diffusion_model.half().cuda(0)
        
        z = sampler.decode_dc(
            x_latent=z_enc,
            first_conditioning=[uim, cim],
            second_conditioning=[utx, ctx],
            t_start=t_enc,
            unconditional_guidance_scale=scale,
            xtype='image', 
            first_ctype='vision',
            second_ctype='prompt',
            mixed_ratio=(1-mixing), )
        
        z = z.cuda(0).half()
        x = clip_model.autokl_decode(z)
        color_adj='None'
        #color_adj_to = cin[0]
        color_adj_flag = (color_adj!='none') and (color_adj!='None') and (color_adj is not None)
        color_adj_simple = (color_adj=='Simple') or color_adj=='simple'
        color_adj_keep_ratio = 0.5
        
        if color_adj_flag and (ctype=='vision'):
            x_adj = []
            for xi in x:
                color_adj_f = color_adjust(ref_from=(xi+1)/2, ref_to=color_adj_to)
                xi_adj = color_adj_f((xi+1)/2, keep=color_adj_keep_ratio, simple=color_adj_simple)
                x_adj.append(xi_adj)
            x = x_adj
        else:
            x = torch.clamp((x+1.0)/2.0, min=0.0, max=1.0)
            x = [tvtrans.ToPILImage()(xi) for xi in x]
        
        out_img_base = os.path.join(out_path_base, f"results/versatile_diffusion/{dataset}/subj{sub}/{model_name}")
        os.makedirs(out_img_base, exist_ok=True)
        out_path = os.path.join(out_path_base, out_img_base, f"{im_id}.png")
        x[0].save(out_path)
        logger.info(f"Saved image to {out_path}")

def vd_reconstruct_main(out_path_base, dataset, sub, include_art_dataset=False, output_name="", clip_model=None, use_true_features=False):
    """ Reconstructs all the images for the given subject. 

    Loads the predicted features from cliptext and clipvision aswell as the corresponding base image that was generated using the vdvae. 
    Then combines the information as described in the paper by Ozcelik to produce the final output images in results/versatile_diffusion
    """
    logger.info("Starting vd_reconstruct main.")
    if clip_model is None:
        clip_model = load_clip_model()

    clip_model.clip.cuda(0)
    clip_model.autokl.cuda(0)
    clip_model.autokl.half()

    strength = 0.75
    mixing = 0.4
    im_size = 512

    n_samples = 1
    ddim_steps = 50
    ddim_eta = 0
    scale = 7.5
    reconstruct_images(out_path_base, dataset, sub, "test", clip_model, strength, mixing, scale, ddim_steps, ddim_eta, im_size, n_samples, output_name, use_true_features=use_true_features)
    
    if include_art_dataset:
        reconstruct_images(out_path_base, dataset, sub, "art", clip_model, strength, mixing, scale, ddim_steps, ddim_eta, im_size, n_samples, output_name, use_true_features=use_true_features)

    ...

if __name__ == "__main__":
    ...
    # sub="AM"
    # out_path_base = "/home/matt/diffusing/recon_diffuser/"
    # dataset="deeprecon"
    
    # torch.manual_seed(0)
    # vd_reconstruct_main(out_path_base, dataset, sub)

