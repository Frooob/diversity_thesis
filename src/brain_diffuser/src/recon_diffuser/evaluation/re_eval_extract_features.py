import os
# import sys
import numpy as np
# import h5py
# import scipy.io as spio
# import nibabel as nib

import torch
# import torchvision
import torchvision.models as tvmodels
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as T
from PIL import Image
import clip
from tqdm import tqdm

def eval_extract_features(out_path_base, dataset_name, name, sub, algorithm="bd"):
    # print(f"Doing eval extract features for sub {sub} on {dataset_name}/{name} in {out_path_base}")

    if sub == 0:  # Ground truth subject, only works for non-perturbed data where all images are present.
        images_dir = os.path.join(out_path_base, f'data/stimuli/{dataset_name}/{name}')
        feats_dir = os.path.join(out_path_base, f'data/eval_features/{dataset_name}/{name}/ground_truth')
    else:
        if algorithm == 'bd':
            images_dir = os.path.join(out_path_base, f'results/versatile_diffusion/{dataset_name}/subj{sub}/{name}')
            feats_dir = os.path.join(out_path_base, f'data/eval_features/{dataset_name}/{name}/subj{sub}')
        elif algorithm == 'icnn':
            images_dir = os.path.join(out_path_base, f'results/icnn/{dataset_name}/subj{sub}/{name}')
            feats_dir = os.path.join(out_path_base, f'data/eval_features/{dataset_name}/icnn_{name}/subj{sub}')
        else:
            raise NotImplementedError(f"Unknown algorithm {algorithm}")
        

    # print(f"DEBUG: {images_dir=}, {feats_dir=}")

    if not os.path.exists(feats_dir):
        os.makedirs(feats_dir)
    # print(f"Created feats dir {feats_dir}")

    class batch_generator_external_images(Dataset):

        def __init__(self, data_path ='', prefix='', net_name='clip'):
            self.data_path = data_path
            self.prefix = prefix
            self.net_name = net_name

            if dataset_name == 'nsd':
                n_test = 982
            elif dataset_name == 'deeprecon' and 'test' in name:
                n_test = 50
            elif dataset_name == 'deeprecon' and 'art' in name:
                n_test = 40
            elif dataset_name == "things":
                n_test = 100
            else:
                raise NotImplementedError(f'Dataset {dataset_name} not implemented yet.')

            self.num_test = n_test
            
            if self.net_name == 'clip':
                # TODO: Are these the values for NSD or what are they supposed to be?
                # If I calculate the mean across all test images for all color channels (see function below):
                # resized to 224:
                    # means= [0.4837062, 0.46796483, 0.4244571]
                    # stds= [0.2612872, 0.2546313, 0.27801764]
                # Full size:
                    # means= [0.48369646, 0.4679516, 0.42444417]
                    # stds= [0.2691188, 0.2624733, 0.28499538]

                self.normalize = transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711])
            else:
                self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            
        def __getitem__(self,idx):
            img = Image.open(f'{self.data_path}/{self.prefix}{idx}.png')
            img = T.functional.resize(img,(224,224))
            img = T.functional.to_tensor(img).float()
            img = self.normalize(img)
            return img

        def __len__(self):
            return  self.num_test


    global feat_list
    feat_list = []
    def fn(module, inputs, outputs):
        feat_list.append(outputs.cpu().numpy())

    net_list = [
        ('inceptionv3','avgpool'),
        ('clip','final'),
        ('alexnet',2),
        ('alexnet',5),
        ('efficientnet','avgpool'),
        ('swav','avgpool')
        ]

    device = 0
    net = None
    batchsize=64

    for (net_name,layer) in tqdm(net_list, desc='Computing eval features...'):
        feat_list = []
        # print(net_name,layer)
        dataset = batch_generator_external_images(data_path=images_dir,net_name=net_name,prefix='')
        loader = DataLoader(dataset,batchsize,shuffle=False)
        
        if net_name == 'inceptionv3': # SD Brain uses this
            net = tvmodels.inception_v3(pretrained=True)
            if layer== 'avgpool':
                net.avgpool.register_forward_hook(fn) 
            elif layer == 'lastconv':
                net.Mixed_7c.register_forward_hook(fn)
        elif net_name == 'alexnet':
            net = tvmodels.alexnet(pretrained=True)
            if layer==2:
                net.features[4].register_forward_hook(fn)
            elif layer==5:
                net.features[11].register_forward_hook(fn)
            elif layer==7:
                net.classifier[5].register_forward_hook(fn)
        elif net_name == 'clip':
            model, _ = clip.load("ViT-L/14", device=f'cuda:{device}')
            net = model.visual
            net = net.to(torch.float32)
            if layer==7:
                net.transformer.resblocks[7].register_forward_hook(fn)
            elif layer==12:
                net.transformer.resblocks[12].register_forward_hook(fn)
            elif layer=='final':
                net.register_forward_hook(fn)
        elif net_name == 'efficientnet':
            net = tvmodels.efficientnet_b1(weights=True)
            net.avgpool.register_forward_hook(fn) 
        elif net_name == 'swav':
            net = torch.hub.load('facebookresearch/swav:main', 'resnet50')
            net.avgpool.register_forward_hook(fn) 
        else:
            raise ValueError(f"Net {net_name} isn't supported.")
        net.eval()
        net.cuda(device)
        
        with torch.no_grad():
            for i,x in tqdm(enumerate(loader), desc=f'Computing features for net {net_name}'):
                # print(i*batchsize)
                x = x.cuda(device)
                _ = net(x)
        if net_name == 'clip':
            if layer == 7 or layer == 12:
                feat_list = np.concatenate(feat_list,axis=1).transpose((1,0,2))
            else:
                feat_list = np.concatenate(feat_list)
        else:   
            feat_list = np.concatenate(feat_list)
        
        file_name = f'{feats_dir}/{net_name}_{layer}.npy'
        np.save(file_name,feat_list)

def compute_mean_stds_for_dataset(imgs_path, resize=None):
    class PureDataset(Dataset):
        def __init__(self, data_path =''):
            self.data_path = data_path
            self.num_imgs = len(os.listdir(data_path))
        def __getitem__(self,idx):
            img = Image.open(f'{self.data_path}/{idx}.png')
            if resize:
                img = T.functional.resize(img,(resize,resize))
            img = T.functional.to_tensor(img).float().numpy()
            return img
        def __len__(self):
            return self.num_imgs
    images = PureDataset(imgs_path)
    # print("Loading images...")
    images_np = np.array([images[img] for img in range(len(images))])
    means = np.mean(images_np, axis=(0,2,3))
    stds = np.std(images_np, axis=(0,2,3))
    return means, stds
        
if __name__ == "__main__":
    # # For NSD takes about 17 min for all participants:
    # sub = [0, "1", "2", "5", "7"]
    # # sub = "5"
    # out_path_base = "/home/matt/programming/recon_diffuser/"
    # dataset_name="nsd"
    # name='test'

    # images_dir = os.path.join(out_path_base, f'data/stimuli/{dataset_name}/{name}')

    # # Computing mean and std for the images because they might be needed for normalization...
    # # mean_std_224 = compute_mean_stds_for_dataset(images_dir, resize=224)
    # # mean_std_full_size = compute_mean_stds_for_dataset(images_dir)
    
    # if len(sub) > 1:
    #     for subn in sub:
    #         eval_extract_features(out_path_base, dataset_name, name, subn)
    # else:
    #     eval_extract_features(out_path_base, dataset_name, name, sub)


    # Deeprecon

    sub = 0
    out_path_base = "/home/matt/programming/recon_diffuser/"
    dataset_name="deeprecon"
    name='test'
    eval_extract_features(out_path_base, dataset_name, name, sub)
