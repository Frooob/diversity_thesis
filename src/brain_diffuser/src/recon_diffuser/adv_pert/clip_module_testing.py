import os
import numpy as np
import sys
import re
import requests
import pandas as pd
import time
from tqdm import tqdm
from collections import defaultdict
import torch
import torchvision.transforms as T
import PIL
from PIL import Image
import matplotlib.pyplot as plt
import clip
import logging
from skimage.metrics import structural_similarity as ssim
import seaborn as sns
import pickle
import socket
import bdpy
import matplotlib.ticker as ticker

sns.set_theme()
logger = logging.getLogger("recdi_adp")

def get_caption_vector(clip_model, caps, im_idx, caption_num="rand", device=None):
    if not device:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    if caption_num == "rand": # Choose a random caption of the n-caption per image
        caption_num = np.random.randint(0,caps.shape[1])
    caption = caps[im_idx, caption_num]
    logger.info(f"The selected caption is: {caption}")
    tokens = clip.tokenize(caption).to(device) # tokenize the caption
    # create feature vector of caption
    feature_vector = clip_model.encode_text(tokens).detach()
    return feature_vector

# used for thesis plot
def sanity_check_friendly_cap_closer_than_adversarial_cap(images, captions, save_name=None, device=None):
    """Is the encoded caption of an image closer to the encoded image than a random other encoded caption?"""
    if not device:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    im_idxs = list(range(len(images)))
    metrics_collector = []
    for im_idx in tqdm(im_idxs, desc="Running friendly vs adversarial dist to gt image.."):
        deeprecon_input_image = preprocess(Image.fromarray(images[im_idx])).unsqueeze(0).to(device)
        deeprecon_im_gt_feature = clip_model.encode_image(deeprecon_input_image)
        deeprecon_im_friendly_caption_feature = get_caption_vector(clip_model, captions, im_idx)
        adversarial_idx = im_idx
        while adversarial_idx == im_idx:
            adversarial_idx = np.random.randint(0, len(images))
        deeprecon_im_adversarial_caption_feature = get_caption_vector(clip_model, captions, adversarial_idx)
        friendly_dist = 1 - torch.nn.functional.cosine_similarity(deeprecon_im_gt_feature, deeprecon_im_friendly_caption_feature).mean().item()
        adversarial_dist = 1 - torch.nn.functional.cosine_similarity(deeprecon_im_gt_feature, deeprecon_im_adversarial_caption_feature).mean().item()
        
        metrics_collector.append({"caption type": "friendly", 'dist': friendly_dist})
        metrics_collector.append({"caption type": "adversarial", 'dist': adversarial_dist})

    df_loss = pd.DataFrame(metrics_collector)
    ax = sns.boxplot(data=df_loss, x='caption type', y='dist')
    ax.set_ylabel('cosine dist')
    plt.savefig(save_name, dpi=300)
    

def do_perturbation(clip_model, preprocess, input_image, perturb_params, collect_imgs, collect_metrics_only=False, device=None, pert_n=None):
    if collect_metrics_only and pert_n is None:
        raise ValueError(f"If you want to collect the metrics, you also have to give me the pert n")
    if not device:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    t1 = time.time()
    perturb_kind = perturb_params['perturb_kind']

    if perturb_kind in ["iterativeCriterion", 'fgsm']:
        captions = perturb_params["captions"]
        im_idx = perturb_params["im_idx"]
        caption_num = perturb_params["caption_num"]
        feature_caption = get_caption_vector(clip_model, captions, im_idx, caption_num, device=device)
    else:
        raise NotImplementedError(f"{perturb_kind=} not implemented.")
    
    # normalize the perturbed feature
    feature_caption = feature_caption / feature_caption.norm(dim=-1, keepdim=True)

    # Set hyperparameters
    steps = perturb_params.get('max_pert_steps', 500)
    LR = 3  # Learning rate only used for iterativeCriterion
    if perturb_kind == 'iterativeCriterion':
        loss_ratio_pattern = r"(\d+)-(\d+)"
        loss_ratio1, loss_ratio2 = re.match(loss_ratio_pattern, perturb_params['criterion']).groups()
        wanted_loss_ratio = int(loss_ratio2) / int(loss_ratio1)

    img = input_image.clone().detach().to(device).requires_grad_(True)
    input_image_np = input_image.cpu().numpy()
    feature_image_initial = None
    metrics_collector = defaultdict(list)
    if collect_imgs or collect_metrics_only:
        metrics_collector["img"].append(img.clone().detach().cpu().numpy())
        pixcorr = np.corrcoef(input_image_np.reshape(1,-1), input_image_np.reshape(1,-1))[0,1]
        metrics_collector["mse"].append(0)
        metrics_collector["ssim"].append(1)
        metrics_collector["pixcorr"].append(pixcorr)
    with tqdm(total=steps, desc="perturbing", unit="step") as pbar:

        for step in tqdm(range(steps), desc="Doing perturbation iteration", total=steps):
            if step > 50:
                LR = 2
            if step > 100:
                LR = 1.5
            if step > 300:
                LR = 1

            if img.grad is not None:
                img.grad.zero_()
            
            feature_image = clip_model.encode_image(img)
            feature_image = feature_image / feature_image.norm(dim=-1, keepdim=True)
            if step == 0:
                # get initial cosine_similarity from feature to perturbed feature
                dist_to_caption_first = 1 - torch.nn.functional.cosine_similarity(feature_image, feature_caption).mean()
                feature_image_initial = feature_image
                dist_to_initial_features = 1 - torch.nn.functional.cosine_similarity(feature_image, feature_image_initial).mean()
                metrics_collector["dist_to_caption"].append(dist_to_caption_first.item())
                metrics_collector["dist_to_initial"].append(dist_to_initial_features.item())
                metrics_collector["loss_ratio"].append(0)
            
            loss = 1 - torch.nn.functional.cosine_similarity(feature_image, feature_caption).mean()
            loss.backward()
            with torch.no_grad():
                img_denormalized = denormalize_clip_img(img)
                if perturb_kind == 'iterativeCriterion':
                    img_denormalized -= LR * img.grad
                elif perturb_kind == 'fgsm':
                    epsilon = perturb_params['epsilon']
                    img_denormalized -= epsilon * img.grad.sign()
                else:
                    raise ValueError(f"Perturb kind {perturb_kind} not supported.")

                img_denormalized = img_denormalized.clamp(0,1)
                img = preprocess(clip_processed_to_PIL(img_denormalized, denormalize=False)).to(device).unsqueeze(0)
                feature_image = clip_model.encode_image(img)
                feature_image = feature_image / feature_image.norm(dim=-1, keepdim=True)
                dist_to_caption = 1 - torch.nn.functional.cosine_similarity(feature_image, feature_caption).mean()
                dist_to_initial = 1 - torch.nn.functional.cosine_similarity(feature_image, feature_image_initial).mean()

            img.requires_grad_(True)
            current_loss_ratio = dist_to_initial.item() / dist_to_caption.item()


            if collect_imgs or collect_metrics_only:
                img_np = img.clone().detach().cpu().numpy()
                metrics_collector["img"].append(img_np)
                pixcorr = np.corrcoef(input_image_np.reshape(1,-1), img_np.reshape(1,-1))[0,1]
                mse = np.linalg.norm(input_image_np - img_np)
                ssim_measured = ssim(input_image_np.squeeze().transpose(1,2,0), img_np.squeeze().transpose(1,2,0), multichannel=True, data_range=img_np.max() - img_np.min())
                metrics_collector["mse"].append(mse)
                metrics_collector["ssim"].append(ssim_measured)
                metrics_collector["pixcorr"].append(pixcorr)

            metrics_collector["dist_to_caption"].append(dist_to_caption.item())
            metrics_collector["dist_to_initial"].append(dist_to_initial.item())
            metrics_collector["loss_ratio"].append(current_loss_ratio)

            pbar.set_postfix(loss=f"{current_loss_ratio:.4f}")  # Update loss display
            pbar.update(1)  # Increment progress

            if perturb_kind == "fgsm":
                break

            if current_loss_ratio > wanted_loss_ratio:
                break


    final_img = img.detach().cpu().numpy()

    if perturb_kind == 'iterativeCriterion':
        if current_loss_ratio / wanted_loss_ratio >= wanted_loss_ratio:
            log_msg = f"At step {step} the wanted loss ratio initial/perturbed {steps} is reached."
        else:
            log_msg = f"After {step} steps the wanted loss ratio {steps} is still not reached."
    elif perturb_kind == 'fgsm':
        log_msg = f"Finished fgsm step. "
    
    log_msg += f"SSIM: {ssim_measured}; loss_ratio: {current_loss_ratio}, dist_to_caption_first_initial: {dist_to_caption_first}; dist_to_initial_image: {dist_to_initial.item()}; dist_to_caption: {dist_to_caption.item()}). It took {time.time() -t1}s. Returning."

    logger.info(log_msg)
    if collect_metrics_only:
        metrics_collector.pop('img')
        n_steps = len(next(iter(metrics_collector.values())))
        metrics_collector['step'] = list(range(n_steps))
        metrics_collector['im_idx'] = [im_idx for _ in range(n_steps)]
        metrics_collector['pert_type'] = [perturb_kind for _ in range(n_steps)]
        metrics_collector['pert_n'] = [pert_n for _ in range(n_steps)]

    return final_img, metrics_collector


def denormalize_clip_img(clip_img, norm_values = [(0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)]):
    clip_mean, clip_std = torch.tensor(norm_values[0]), torch.tensor(norm_values[1])
    # Ensure the mean and std are broadcastable to the image tensor shape
    clip_mean = clip_mean.view(3, 1, 1)
    clip_std = clip_std.view(3, 1, 1)
    if type(clip_img) == torch.Tensor:
        clip_img_device = clip_img.get_device()
        clip_mean = clip_mean.to(clip_img_device)
        clip_std = clip_std.to(clip_img_device)
    else:
        clip_img = torch.Tensor(clip_img)
    # De-normalize the image
    clip_img = clip_img * clip_std + clip_mean
    clip_img = clip_img.clamp(0, 1)
    return clip_img

def clip_processed_to_PIL(clip_img, denormalize=True):
    if denormalize:
        clip_img = denormalize_clip_img(clip_img)
    try:
        clip_img = clip_img.detach().cpu()
    except:
        pass
    clip_img = clip_img.squeeze(0)
    img_pil = T.ToPILImage()(clip_img)
    return img_pil


## Helper functions
def get_adversarial_caption_index(im_idx, images):
    adversarial_idx = im_idx # getting an adversarial image index
    while adversarial_idx == im_idx:
        adversarial_idx = np.random.randint(0, len(images))
        adversarial_cap_num = np.random.randint(0,5)
    return adversarial_idx, adversarial_cap_num

def out_name_from_perturb_params(perturb_params):
    pp = perturb_params # better readability
    perturb_kind = pp['perturb_kind']
    if perturb_kind == 'fgsm':
        out_name = f"fgsm_{pp['caption_type']}_{pp['epsilon']}_{pp['n_perts']}"
    elif perturb_kind == 'iterativeCriterion':
        out_name = f"ic_{pp['caption_type']}_{pp['criterion']}_{pp['max_pert_steps']}_{pp['n_perts']}"
    else:
        raise ValueError(f"Unknown perturb kind {perturb_kind}")
    return out_name

def fmri_indexes_for_image_labels(image_labels: list) -> dict:
    mapper = defaultdict(list)
    for i, image_label in enumerate(image_labels):
        mapper[image_label].append(i)
    return mapper

def get_im_idx_sub(out_path_base, sub):
    im_idx_path = os.path.join(out_path_base, 'data', 'train_data', 'deeprecon_perturbed', f'im_idx_sub{sub}.npy')

    if os.path.exists(im_idx_path):
        return np.load(im_idx_path)
    else:
        # Needed to translate the images/captions to their key
        deeprecon_data_root = "/home/kiss/data/fmri_shared/datasets/Deeprecon/fmriprep"
        fmri_path = f'{deeprecon_data_root}/{sub}_ImageNetTraining_volume_native.h5'
        print(f"opening fmri train data of subject {sub} to get the im_idx...")
        bdata = bdpy.BData(fmri_path)
        image_labels = bdata.get_labels('stimulus_name')
        sig = fmri_indexes_for_image_labels(image_labels)
        im_idx = list(sig.keys())
        np.save(im_idx_path, im_idx)
        return im_idx

def rename_old_ims(ims_path, im_idx):
    all_old_ims = [p for p in os.listdir(ims_path) if p.endswith('png')]
    # if len(all_old_ims) != 6000:
    #     raise Exception("Should be 6000")
    
    for fname_im in tqdm(all_old_ims):
        if fname_im.startswith('n'):
            continue
        if fname_im.endswith('.csv'):
            continue
        path_im = os.path.join(ims_path, fname_im)
        im_num = int(fname_im[2:fname_im.index("_")])
        im_id = im_idx[im_num]
        pert_n = int(fname_im[fname_im.rfind("_")+1:fname_im.rfind(".")])
        out_name = f"{im_id}_{pert_n}.png"
        out_path = os.path.join(ims_path, out_name)
        os.rename(path_im, out_path)



## Main functions
def perturbations_all_images(
        out_path_base, dataset, sub, input_name, perturb_params, device=None):
    if not device:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    if sub != "AM":
        raise Exception("Only for subAM the perturbations are made. They will be used universally though.")
    logger.info(f"Doing all perturbations for {out_path_base}, {dataset}, {sub}, {input_name}, {perturb_params} on host {socket.gethostname()}.")
    logger.info(f"Loading clip model")
    clip_model, preprocess = clip.load("ViT-L/14", device=device)
    input_images_path = os.path.join(out_path_base, "data", "processed_data", dataset, f"subj{sub}", f"{input_name}_stim_sub{sub}.npy")
    input_captions_path = os.path.join(out_path_base, "data", "processed_data", dataset, f"subj{sub}", f"{input_name}_cap_sub{sub}.npy")
    im_idx = get_im_idx_sub(out_path_base, 'AM')
    
    logger.info(f"Loading images from {input_images_path}.")
    images = np.load(input_images_path).astype(np.uint8)
    captions = np.load(input_captions_path)

    out_name = out_name_from_perturb_params(perturb_params)
    out_dir = os.path.join(out_path_base, 'data', 'train_data','deeprecon_perturbed', out_name)
    os.makedirs(out_dir, exist_ok=True)
    logger.info(f"Created out_folder at {out_dir}.")
    
    n_perts = perturb_params['n_perts']
    perturb_kind = perturb_params['perturb_kind']

    preprocessed_images = [preprocess(Image.fromarray(images[im_idx])) for im_idx in range(images.shape[0])]
    logger.info(f"Starting {out_name} perturbations. ")
    metrics_collector = defaultdict(list)
    for im_id in tqdm(range(len(preprocessed_images)), desc=f'Perturbing all {dataset} {input_name} images...', total=int(len(images))):
        input_image = preprocessed_images[im_id].unsqueeze(0).to(device)
        for n in range(n_perts):
            out_img_path = os.path.join(out_dir, f"im{im_id}_{out_name}_{n}.png")
            if perturb_kind in ['fgsm', 'iterativeCriterion']:
                caption_type = perturb_params['caption_type']
                if caption_type == 'friendly':
                    perturb_params = {**perturb_params, **{"im_idx": im_id, "captions": captions, "caption_num": n}}
                    logger.info(f"Doing perturbation for {dataset}-{input_name}-{sub}-{out_name}. Image {im_id} caption {n}.")
                elif caption_type == 'adversarial':
                    adversarial_idx, adversarial_cap_num = get_adversarial_caption_index(im_id, images)
                    perturb_params = {**perturb_params, **{"im_idx": adversarial_idx, "captions": captions, "caption_num": adversarial_cap_num}}
                    logger.info(f"Doing perturbation for {dataset}-{input_name}-{sub}-{out_name}. Image {im_id}(adv index {adversarial_idx}; caption {adversarial_cap_num}).")
                else:
                    raise ValueError(f"Unknown caption type {caption_type}")
            img, metrics = do_perturbation(clip_model, preprocess, input_image, perturb_params, False, collect_metrics_only=True, device=device, pert_n=n)
            for key,value in metrics.items():
                metrics_collector[key] += value
            pil_img = clip_processed_to_PIL(img)
            pil_img.save(out_img_path)
            logger.info(f"Saved perturbed image to {out_img_path}")

    df_metrics = pd.DataFrame(metrics_collector)
    df_metrics.to_csv(os.path.join(out_dir, 'df_metrics.csv'))
    rename_old_ims(out_dir, im_idx)


def single_image_perturbations(clip_model, images, captions, im_id, perturb_algorithm, perturb_type, ic_criterion=None, fgsm_epsilon=None):
    # perturb_algorithm should be one of ["iterativeCriterion", 'fgsm']
    input_image = images[im_id]
    caption = captions[im_id]
    perturb_params = {'perturb_kind': perturb_algorithm, 'captions': captions, 'caption_num':"rand"}
    
    input_image = preprocess(Image.fromarray(input_image)).unsqueeze(0).to(device)

    if perturb_type == 'adversarial':
        caption_id = im_id
        while caption_id == im_id:
            caption_id = np.random.randint(0, len(images))
    elif perturb_type == 'friendly':
        caption_id = im_id
    
    if perturb_algorithm == 'fgsm':
        perturb_params['epsilon'] = fgsm_epsilon
    elif perturb_algorithm == 'iterativeCriterion':
        perturb_params['criterion'] = ic_criterion

    perturb_params['im_idx'] = caption_id

    collect_imgs = True
    perturb_params['max_pert_steps'] = 200
    
    final_img, metrics_collector = do_perturbation(clip_model, preprocess, input_image, perturb_params, collect_imgs, device=None)
    return final_img, metrics_collector


###### Thesis plots

## 1. Sanity Check if friendly captions are closer to the images than adversarial captions

def sanity_check_plot():
    sanity_check_savename = os.path.join(thesis_plots_folder, 'advpert_sanity_check_friendly_vs_adversarial_cap.png')
    sanity_check_friendly_cap_closer_than_adversarial_cap(images, captions, sanity_check_savename)

## 2. IC 50-50 validaion quantitative

def add_im_id(df):
    df = df.copy()
    id_col = ((df["step"] == 0).cumsum()/(df["pert_n"]+1)).astype(int)
    df["im_id"] = id_col
    return df

def ic_plot_loss_curves():

    def custom_formatter(x, pos):
        if x == 1:
            return "1"  # Show 1 as "1"
        elif 0 < x < 1:
            return f"{x:.4f}".lstrip("0")  # Remove leading zero and limit to 3 decimal places
        return f"{x:.4f}"  # Limit to 3 decimal places for all other numbers

    df_50_50_adversarial = pd.read_csv("/home/matt/programming/recon_diffuser/data/train_data/deeprecon_perturbed/ic_adversarial_50-50_500_5/df_metrics.csv")
    df_50_50_friendly = pd.read_csv("/home/matt/programming/recon_diffuser/data/train_data/deeprecon_perturbed/ic_friendly_50-50_500_5/df_metrics.csv")


    df_50_50_friendly["caption_type"] = "friendly"
    df_50_50_adversarial["caption_type"] = "adversarial"

    df_50_50_friendly = add_im_id(df_50_50_friendly)
    df_50_50_adversarial = add_im_id(df_50_50_adversarial)
    df_50_50 = pd.concat((df_50_50_friendly, df_50_50_adversarial))

    df_50_50["loss_ratio"] = df_50_50["dist_to_initial"] / df_50_50["dist_to_caption"]

    # Get the maximum step for each combination
    max_steps = df_50_50.groupby(["caption_type", "pert_n", "im_id"])[["step"]].max().reset_index()
    df_max_steps = df_50_50.merge(max_steps, on=["caption_type", "pert_n", "im_id", "step"])
    # Create a DataFrame with all missing steps
    new_rows = []
    for _, row in tqdm(df_max_steps.iterrows(), total=len(max_steps)):
        # caption_type, pert_n, im_id, max_step, loss_ratio = row
        row_d = row.to_dict()
        row_d["loss_ratio"] = 1
        max_step = row_d["step"]
        
        # Generate new rows for missing steps
        for step in range(max_step + 1, 501):
            row_d["step"] = step  # Update step number
            new_rows.append(row_d.copy())
    df_new = pd.DataFrame(new_rows)
    df_big = pd.concat([df_50_50, df_new])

    df = df_big
    # Takes quite some time to execute because a lot of number crunching is happening here
    criterion_titles = ["PixCorr perturbed / initial", "cos_dist to caption", "cos_dist to initial image", "cos_dist ratio", ]
    hue_order = ["adversarial", "friendly"]
    criterions = [ "pixcorr", "dist_to_caption", "dist_to_initial", "loss_ratio"]

    df = df.groupby(["caption_type", "step"])[criterions].mean().reset_index()

    fig, axs = plt.subplots(nrows=1, ncols=4, figsize=(13, 2.7))
    for num,criterion in enumerate(criterions):
        ax = axs[num]
        sns.lineplot(data=df, x="step", y=criterion, hue="caption_type", ax=ax, hue_order=hue_order)
        ax.set_ylabel("")
        ax.set_title(criterion_titles[num])
        if num > 0:
            ax.get_legend().remove()
        if num == 0:
            ax.yaxis.set_major_formatter(ticker.FuncFormatter(custom_formatter))
    save_name = os.path.join(thesis_plots_folder, "advpert_validation_ic_loss_curves.png")
    fig.savefig(save_name, dpi=300,bbox_inches='tight')

def ic_image_evolution_plot():
    im_num = 13
    original_image = Image.fromarray(images[im_num])
    original_image

    # make sure the adversarial and friendly image have roughly the same number of steps
    # Not needed anymore, I'm only doing friendly anyways (It's not about the caption type comparison here)
    final_img_ic_friendly, metrics_collector_ic_friendly = single_image_perturbations(clip_model, images, captions, im_num, "iterativeCriterion", "friendly", "50-50")

    desired_loss_ratios = [0, 0.1, 0.3, 0.6, 1]

    indices_friendly = []
    for desired_loss_ratio in desired_loss_ratios:
        loss_ratios_friendly = metrics_collector_ic_friendly["loss_ratio"]
        closest_index_friendly = np.abs(np.array(loss_ratios_friendly) - desired_loss_ratio).argmin()
        indices_friendly.append(closest_index_friendly)


    images_friendly = [metrics_collector_ic_friendly["img"][i] for i in indices_friendly]

    n_columns = len(desired_loss_ratios)
    fig, axes = plt.subplots(1, n_columns, figsize=(n_columns*3, 3))

    # for row_n, caption_type in enumerate(["friendly", "adversarial"]):
    row_n = 0
    indices = indices_friendly
    collector = metrics_collector_ic_friendly
    for col_n in range(n_columns):
        index = indices[col_n]
        image = collector["img"][index]
        img_pil = clip_processed_to_PIL(image)
        loss_ratio = np.round(collector["loss_ratio"][index],2)
        pixcorr = np.round(collector["pixcorr"][index],3)
        step = index
        # ax = axes[row_n, col_n]
        ax = axes[col_n]
        ax.imshow(img_pil)
        ax.axis('off')
        # ax.set_title(f"step: {step}, loss_ratio: {loss_ratio}")
        ax.set_title(f"PixCorr: {pixcorr},\n cos_dist ratio: {loss_ratio}", fontsize=16)
    fig.savefig(os.path.join(thesis_plots_folder, "advpert_ic_qual_validation_evolution.png"),bbox_inches='tight')


def fgsm_plot_loss_curves():
    im_num = 13
    fgsm_epsilons = list(np.linspace(0, 0.1, 50))


    ### We should do both plots right here, since computing all values with fgsm is very fast

    ## We want to go with multiple epsilons (kind of correspond to number of steps in the previous plots)
    def custom_formatter(x, pos):
        if x == 1:
            return "1"  # Show 1 as "1"
        elif 0 < x < 1:
            return f"{x:.2f}".lstrip("0")  # Remove leading zero and limit to 3 decimal places
        return f"{x:.2f}"  # Limit to 3 decimal places for all other numbers

    x = fgsm_epsilons

    df_value_collector = []

    ims = []

    for fgsm_epsilon in fgsm_epsilons:
        final_img_fgsm_friendly, metrics_collector_fgsm_friendly = single_image_perturbations(clip_model, images, captions, im_num, "fgsm", "friendly", fgsm_epsilon=fgsm_epsilon)
        mc = metrics_collector_fgsm_friendly
        d1 = {"pixcorr": mc["pixcorr"][1], "dist_to_caption": mc["dist_to_caption"][1], "dist_to_initial": mc["dist_to_initial"][1], "loss_ratio": mc["loss_ratio"][1], "caption_type": "friendly", "epsilon": fgsm_epsilon}
        final_img_fgsm_adversarial, metrics_collector_fgsm_adversarial = single_image_perturbations(clip_model, images, captions, im_num, "fgsm", "adversarial", fgsm_epsilon=fgsm_epsilon)
        limg = metrics_collector_fgsm_friendly["img"][-1]
        mc = metrics_collector_fgsm_adversarial
        d2 = {"pixcorr": mc["pixcorr"][1], "dist_to_caption": mc["dist_to_caption"][1], "dist_to_initial": mc["dist_to_initial"][1], "loss_ratio": mc["loss_ratio"][1], "caption_type": "adversarial", "epsilon": fgsm_epsilon}
        df_value_collector.append(d1)
        df_value_collector.append(d2)
        ims.append(mc["img"][1])
    

    df = pd.DataFrame(df_value_collector)

    criterion_titles = ["PixCorr perturbed / initial", "cos_dist to caption", "cos_dist to initial image", "cos_dist ratio", ]
    hue_order = ["adversarial", "friendly"]
    criterions = [ "pixcorr", "dist_to_caption", "dist_to_initial", "loss_ratio"]

    fig, axs = plt.subplots(nrows=1, ncols=4, figsize=(13, 2.7))
    for num,criterion in enumerate(criterions):
        ax = axs[num]
        sns.lineplot(data=df, x="epsilon", y=criterion, hue="caption_type", ax=ax, hue_order=hue_order)
        ax.set_ylabel("")
        ax.set_title(criterion_titles[num])
        if num > 0:
            ax.get_legend().remove()
        # if num == 0:
        ax.yaxis.set_major_formatter(ticker.FuncFormatter(custom_formatter))
    save_name = os.path.join(thesis_plots_folder, "advpert_validation_fgsm_loss_curves.png")
    fig.savefig(save_name, dpi=300,bbox_inches='tight')


def fgsm_image_evolution_plot():
    im_num = 13
    original_image = Image.fromarray(images[im_num])
    original_image

    fgsm_epsilons = [0, 0.01, 0.03, 0.05, 0.1]


    fgsm_images = []
    loss_ratios = []
    pixcorrs = []
    for fgsm_epsilon in fgsm_epsilons:
        final_img_fgsm_friendly, metrics_collector_fgsm_friendly = single_image_perturbations(clip_model, images, captions, im_num, "fgsm", "friendly", fgsm_epsilon=fgsm_epsilon)
        loss_ratios_friendly = metrics_collector_fgsm_friendly["loss_ratio"]
        fgsm_images.append(metrics_collector_fgsm_friendly["img"][1])
        loss_ratios.append(metrics_collector_fgsm_friendly["loss_ratio"][1])   
        pixcorrs.append(metrics_collector_fgsm_friendly["pixcorr"][1])


    n_columns = len(fgsm_epsilons)
    fig, axes = plt.subplots(1, n_columns, figsize=(n_columns*3, 3))

    for col_n in range(n_columns):
        image = fgsm_images[col_n]
        epsilon = fgsm_epsilons[col_n]
        loss_ratio = np.round(loss_ratios[col_n],2)
        pixcorr = np.round(pixcorrs[col_n],2)
        img_pil = clip_processed_to_PIL(image)
        # ax = axes[row_n, col_n]
        ax = axes[col_n]
        ax.imshow(img_pil)
        ax.axis('off')
        # ax.set_title(f"step: {step}, loss_ratio: {loss_ratio}")
        ax.set_title(f"PixCorr: {pixcorr},\ncos_dist ratio: {loss_ratio}\nepsilon: {epsilon}", fontsize=16)
    fig.savefig(os.path.join(thesis_plots_folder, "advpert_fgsm_qual_validation_evolution.png"),bbox_inches='tight')


def final_chosen_dataset_analysis():

    def custom_formatter(x, pos):
        if x == 1:
            return "1"  # Show 1 as "1"
        elif 0 < x < 1:
            return f"{x:.2f}"  # Remove leading zero and limit to 3 decimal places
        return f"{x:.2f}"  # Limit to 3 decimal places for all other numbers

    chosen_criterion = "80-20"
    chosen_epsilon = "0.03"

    df_ic_chosen_adversarial = pd.read_csv(f"/home/matt/programming/recon_diffuser/data/train_data/deeprecon_perturbed/ic_adversarial_{chosen_criterion}_500_5/df_metrics.csv")
    df_ic_chosen_friendly = pd.read_csv(f"/home/matt/programming/recon_diffuser/data/train_data/deeprecon_perturbed/ic_friendly_{chosen_criterion}_500_5/df_metrics.csv")
    df_ic_chosen_friendly["caption_type"] = "friendly"
    df_ic_chosen_adversarial["caption_type"] = "adversarial"
    df_ic_chosen_friendly = add_im_id(df_ic_chosen_friendly)
    df_ic_chosen_adversarial = add_im_id(df_ic_chosen_adversarial)
    df_9010 = pd.concat((df_ic_chosen_friendly, df_ic_chosen_adversarial))
    max_steps = df_9010.groupby(["caption_type", "pert_n", "im_id"])[["step"]].max().reset_index()
    df_ic_chosen = df_9010.merge(max_steps, on=["caption_type", "pert_n", "im_id", "step"])


    df_fgsm_chosen_adversarial = pd.read_csv(f"/home/matt/programming/recon_diffuser/data/train_data/deeprecon_perturbed/fgsm_adversarial_{chosen_epsilon}_5/df_metrics.csv")
    df_fgsm_chosen_friendly = pd.read_csv(f"/home/matt/programming/recon_diffuser/data/train_data/deeprecon_perturbed/fgsm_friendly_{chosen_epsilon}_5/df_metrics.csv")
    df_fgsm_chosen_adversarial = df_fgsm_chosen_adversarial.query("step == 1")
    df_fgsm_chosen_friendly = df_fgsm_chosen_friendly.query("step == 1")
    df_fgsm_chosen_friendly["caption_type"] = "friendly"
    df_fgsm_chosen_adversarial["caption_type"] = "adversarial"
    df_ep03 = pd.concat((df_fgsm_chosen_friendly, df_fgsm_chosen_adversarial))

    df_advpert = pd.concat((df_ic_chosen, df_ep03))

    ic_value_name =  f"IC {chosen_criterion}"
    fgsm_value_name = f"fgsm {chosen_epsilon}"

    value_renamer = {"pert_type": {"iterativeCriterion":ic_value_name, "fgsm": fgsm_value_name}}
    for col, renamer in value_renamer.items():
        df_advpert[col] = df_advpert[col].replace(renamer)

    col_renamer = {"pert_type": "pert algorithm", "caption_type": "caption type"}
    # for col, new_col in col_renamer.items():
    df_advpert = df_advpert.rename(columns = col_renamer)


    criterion_titles = ["PixCorr perturbed / initial", "cos_dist to caption", "cos_dist to initial image", "cos_dist ratio", ]
    hue_order = [ic_value_name, fgsm_value_name]
    criterions = [ "pixcorr", "dist_to_caption", "dist_to_initial", "loss_ratio"]

    fig, axs = plt.subplots(nrows=1, ncols=4, figsize=(13, 2.7))
    for num,criterion in enumerate(criterions):
        ax = axs[num]
        sns.pointplot(data=df_advpert, x="caption type", y=criterion, hue="pert algorithm", ax=ax,  linestyle="None", errorbar="sd", hue_order=hue_order, dodge=0.1)
        ax.set_ylabel("")
        ax.set_title(criterion_titles[num])
        if num != 1:
            ax.get_legend().remove()
        # if num == 0:
        ax.tick_params(axis='y', which='major', pad=0)
        ax.yaxis.set_major_formatter(ticker.FuncFormatter(custom_formatter))
    save_name = os.path.join(thesis_plots_folder, "advpert_validation_chosen_perts.png")
    fig.savefig(save_name, dpi=300,bbox_inches='tight')

def plot_some_chosen_pert_images():
    n_images = 5

    chosen_criterion = "80-20"
    chosen_epsilon = "0.03"

    # some_indexes = np.random.randint(0,1200, n_images)
    some_indexes = np.array([317, 421, 467, 138, 396]) # make it reproducible
    print(f"Some indexes are: {some_indexes}")

    folder_gt = "/home/matt/programming/recon_diffuser/data/train_data/deeprecon"
    folder_fgsm = f"/home/matt/programming/recon_diffuser/data/train_data/deeprecon_perturbed/fgsm_adversarial_{chosen_epsilon}_5/"
    folder_ic = f"/home/matt/programming/recon_diffuser/data/train_data/deeprecon_perturbed/ic_adversarial_{chosen_criterion}_500_5/"

    images_fgsm = []
    images_ic = []
    images_gt = []
    for index in some_indexes:
        image_name = sorted(os.listdir(folder_gt))[index][:-5]
        p_image_gt = os.path.join(folder_gt, image_name+".JPEG")
        image_gt = Image.open(p_image_gt).resize((224,224))
        images_gt.append(image_gt)

        p_image_fgsm = os.path.join(folder_fgsm, image_name+"_1.png")
        image_fgsm = Image.open(p_image_fgsm)
        images_fgsm.append(image_fgsm) 

        p_image_ic = os.path.join(folder_ic, image_name+"_1.png")
        image_ic = Image.open(p_image_ic)
        images_ic.append(image_ic)


    fig, axs = plt.subplots(nrows=3, ncols=n_images, figsize=(n_images*3, 9))
    for row_n in range(3):
        for col_n in range(n_images):
            ax = axs[row_n, col_n]

            if row_n == 0:
                ims = images_gt 
                row_desc = "Original"
            elif row_n == 1:
                ims = images_ic
                row_desc = f"IC {chosen_criterion}"
            elif row_n == 2:
                ims = images_fgsm
                row_desc = f"fgsm {chosen_epsilon}"
            im = ims[col_n]
            ax.imshow(im)
            ax.axis('off')

            if col_n == 0:
                ax.text(-1, np.array(im).shape[0] // 2, row_desc, 
                fontsize=16, va="center", ha="right", rotation=90)

    plt.subplots_adjust(hspace=0, wspace=0.05)
    fig.savefig(os.path.join(thesis_plots_folder, "advpert_validation_chosen_qual.png"),bbox_inches='tight')

if __name__ == "__main__":
    out_path_base = "/home/matt/programming/recon_diffuser/"
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    logging.basicConfig(
        format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=[
            logging.StreamHandler(),
        ],
        level=logging.INFO)

    thesis_plots_folder = "/home/matt/programming/recon_diffuser/src/recon_diffuser/adv_pert/thesis_plots"

    clip_model, preprocess = clip.load("ViT-L/14", device=device)
    images = np.load("/home/matt/programming/recon_diffuser/data/processed_data/deeprecon/subjAM/train_stim_subAM.npy").astype(np.uint8)

    captions = np.load("/home/matt/programming/recon_diffuser/data/processed_data/deeprecon/subjAM/train_cap_subAM.npy")

    ic_plot_loss_curves()
    sanity_check_plot()
    ic_image_evolution_plot()
    fgsm_plot_loss_curves()
    fgsm_image_evolution_plot()
    final_chosen_dataset_analysis()
    plot_some_chosen_pert_images()
