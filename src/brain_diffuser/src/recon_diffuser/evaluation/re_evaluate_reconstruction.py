import os
import numpy as np
import scipy as sp
from PIL import Image
from tqdm import tqdm

from scipy.stats import binom
import numpy as np
from skimage.color import rgb2gray
from skimage.metrics import structural_similarity as ssim
from eval_utils import simple_log
import pandas as pd
### Dreamsim
import torch
from dreamsim import dreamsim
import clip
from scipy import spatial

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model_dreamsim, preprocess_dreamsim = dreamsim(pretrained=True, device=device)
model_clip, preprocess_clip = clip.load("ViT-L/14", device=device)
net_clip = model_clip.visual
net_clip = net_clip.to(torch.float32)
net_clip = net_clip.eval()

def pairwise_corr_all(ground_truth, predictions, aggregate_results=True):
    """
    Pairwise correlation: Computes the correlation of the Features of all 50 gt and 50 pred pictures. 

    The correlation is stored in r, where the rows is gt and the columns is pred.

    The congruents is the diagonal of r (the corresponding pictures of gt and pred are the same).

    The success rate for each column
    
    """
    r = np.corrcoef(ground_truth, predictions)#cosine_similarity(ground_truth, predictions)#
    r = r[:len(ground_truth), len(ground_truth):]  # rows: groundtruth, columns: predicitons
    # congruent pairs are on diagonal
    congruents = np.diag(r)
    
    # for each column (predicition) we should count the number of rows (groundtruth) that the value is lower than the congruent (e.g. success).
    success = r < congruents
    success_cnt = np.sum(success, 0)

    # note: diagonal of 'success' is always zero so we can discard it. That's why we divide by len-1
    perf = success_cnt / (len(ground_truth)-1)    
    if aggregate_results:
        perf = perf.mean()

    p = 1 - binom.cdf(perf*len(ground_truth)*(len(ground_truth)-1), len(ground_truth)*(len(ground_truth)-1), 0.5)
    return perf, p

def eval_subject(out_path_base, dataset_name, name, sub, result_name="", aggregate_results=True, algorithm='bd'):
    if not result_name:
        result_name = name

    if algorithm == 'icnn':
        result_name = 'icnn_' + result_name
    simple_log(f'Evaluating Subject {dataset_name}, {result_name}, {sub}')
    metrics_base = {"dataset_name": dataset_name, "name" : result_name, "sub": sub}
    metrics = {}

    net_list = [
        ('inceptionv3','avgpool'),
        ('clip','final'),
        ('alexnet',2),
        ('alexnet',5),
        ('efficientnet','avgpool'),
        ('swav','avgpool')
        ]

    sub_feats_dir = os.path.join(out_path_base, f'data/eval_features/{dataset_name}/{result_name}/subj{sub}')
    gt_feats_dir = os.path.join(out_path_base, f'data/eval_features/{dataset_name}/{name}/ground_truth')

    if dataset_name == 'nsd':    
        num_test = 982
        gt_image_size = 425
    elif dataset_name == "deeprecon" and "art" in name:
        num_test = 40
        gt_image_size = 240
    elif dataset_name == "deeprecon" and "test" in name:
        num_test = 50
        gt_image_size = 500
    elif dataset_name == "things":
        num_test = 100
        gt_image_size = 500
    else:
        raise NotImplementedError(f"Dataset {dataset_name} not supported.")

    distance_fn = sp.spatial.distance.correlation
    for (net_name,layer) in tqdm(net_list, desc='Doing high level evaluations...'):
        gt_file_name = f'{gt_feats_dir}/{net_name}_{layer}.npy'
        gt_feat = np.load(gt_file_name)
        sub_file_name = f'{sub_feats_dir}/{net_name}_{layer}.npy'
        eval_feat = np.load(sub_file_name)
        gt_feat = gt_feat.reshape((len(gt_feat),-1))
        eval_feat = eval_feat.reshape((len(eval_feat),-1))
        if net_name in ['efficientnet','swav']:
            metric = np.array([distance_fn(gt_feat[i],eval_feat[i]) for i in range(num_test)]) # compute distance
            if aggregate_results:
                metric = metric.mean()
        else:
            metric = pairwise_corr_all(gt_feat[:num_test],eval_feat[:num_test], aggregate_results)[0] # compute pairwise corr

        metrics[f"{net_name}_{layer}"] = metric     
            
    ssim_list = []
    pixcorr_list = []
    for i in tqdm(range(num_test), desc='Computing low-level correlations for all images.'):
        
        if algorithm == 'bd':
            gen_image_path = os.path.join(out_path_base, f'results/versatile_diffusion/{dataset_name}/subj{sub}/{result_name}/{i}.png')
        elif algorithm == "icnn":
            gen_image_path = os.path.join(out_path_base, f'results/icnn/{dataset_name}/subj{sub}/{result_name[5:]}/{i}.png')
        else:
            raise NotImplementedError(f"Algorithm {algorithm} not known.")

        gen_image = Image.open(
             gen_image_path
            ).resize((gt_image_size,gt_image_size))
        gt_image = Image.open(os.path.join(out_path_base, f'data/stimuli/{dataset_name}/{name}/{i}.png'))
        gen_image = np.array(gen_image)/255.0
        gt_image = np.array(gt_image)/255.0
        pixcorr_res = np.corrcoef(gt_image.reshape(1,-1), gen_image.reshape(1,-1))[0,1]
        pixcorr_list.append(pixcorr_res)
        gen_image = rgb2gray(gen_image)
        gt_image = rgb2gray(gt_image)
        ssim_res = ssim(
            gen_image, gt_image, multichannel=True, gaussian_weights=True, sigma=1.5, use_sample_covariance=False, data_range=1.0)
        ssim_list.append(ssim_res)
        
    ssims = np.array(ssim_list)
    pixcorrs = np.array(pixcorr_list)
    if aggregate_results:
        ssims = ssims.mean()
        pixcorrs = pixcorrs.mean()

    metrics["PixCorr"] = pixcorrs
    metrics["SSIM"] = ssims


    dreamsim_dists = []
    for i in tqdm(range(num_test), desc='Computing dreamsim distances'):
        if algorithm == 'bd':
            gen_image_path = os.path.join(out_path_base, f'results/versatile_diffusion/{dataset_name}/subj{sub}/{result_name}/{i}.png')
        elif algorithm == "icnn":
            gen_image_path = os.path.join(out_path_base, f'results/icnn/{dataset_name}/subj{sub}/{result_name[5:]}/{i}.png')
        else:
            raise NotImplementedError(f"Algorithm {algorithm} not known.")

        gen_image = preprocess_dreamsim(Image.open(
             gen_image_path
            ).resize((gt_image_size,gt_image_size))).to(device)
        gt_image = preprocess_dreamsim(Image.open(os.path.join(out_path_base, f'data/stimuli/{dataset_name}/{name}/{i}.png'))).to(device)
        dist_dreamsim = model_dreamsim(gen_image, gt_image) # The model takes an RGB image from [0, 1], size batch_sizex3x224x224
        dreamsim_dists.append(dist_dreamsim.item())
    
    if aggregate_results:
        dreamsim_dists = np.array(dreamsim_dists).mean()
    metrics['dreamsim'] = dreamsim_dists


    clip_dists = []
    for i in tqdm(range(num_test), desc='Computing clip distances'):
        if algorithm == 'bd':
            gen_image_path = os.path.join(out_path_base, f'results/versatile_diffusion/{dataset_name}/subj{sub}/{result_name}/{i}.png')
        elif algorithm == "icnn":
            gen_image_path = os.path.join(out_path_base, f'results/icnn/{dataset_name}/subj{sub}/{result_name[5:]}/{i}.png')
        else:
            raise NotImplementedError(f"Algorithm {algorithm} not known.")
        
        clip_gen_image = net_clip(preprocess_clip(Image.open( gen_image_path).resize((gt_image_size,gt_image_size))).unsqueeze(0).to(device)).detach().cpu()
        clip_gt_image = net_clip(preprocess_clip(Image.open(os.path.join(out_path_base, f'data/stimuli/{dataset_name}/{name}/{i}.png'))).unsqueeze(0).to(device)).detach().cpu()
        dist_clip = spatial.distance.cosine(clip_gen_image.squeeze(), clip_gt_image.squeeze()).item()
        clip_dists.append(dist_clip)

    if aggregate_results:
        clip_dists = np.array(clip_dists).mean()
    metrics['clip_dist'] = clip_dists

    if not aggregate_results:
        metrics = [{'im': i + 1, **{key: values[i] for key, values in metrics.items()}} for i in range(len(next(iter(metrics.values()))))]

    # add the base information
    if aggregate_results:
        metrics = {**metrics_base, **metrics}
    else:
        metrics = [{**metrics_base, **m} for m in metrics]

    return metrics


if __name__ == "__main__":
    out_path_base = "/home/matt/programming/recon_diffuser/"
    dataset_name="deeprecon"
    name='test'
    sub='AM'
    eval_subject(out_path_base, dataset_name, name, sub)
