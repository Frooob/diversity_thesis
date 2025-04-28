"""
Equivalent to the prepare_deeprecon_data script. Makes sure that all the input data follows the same structure such that all subsequent scripts may work exactly the same. 

Within this preprocessing all the neccessary information will be bundled together in an input file. This contains fmri data, stimulus data and captions for the stimuli. The data format is specified in the README.md.

Entrypoint is the prepare_deeprecon_data_main function.

Attributes:
    Same attributes as in re_nsd_main.py
"""

import os
import sys
import numpy as np
import h5py
import scipy.io as spio
import nibabel as nib
from tqdm import tqdm
from datetime import datetime
import pickle
#os.system('ls -l')
def simple_log(l):
    print(f"{datetime.now()}: {l}")



def loadmat(filename):
    '''
    this function should be called instead of direct spio.loadmat
    as it cures the problem of not properly recovering python dictionaries
    from mat files. It calls the function check keys to cure all entries
    which are still mat-objects
    '''
    def _check_keys(d):
        '''
        checks if entries in dictionary are mat-objects. If yes
        todict is called to change them to nested dictionaries
        '''
        for key in d:
            if isinstance(d[key], spio.matlab.mio5_params.mat_struct):
                d[key] = _todict(d[key])
        return d

    def _todict(matobj):
        '''
        A recursive function which constructs from matobjects nested dictionaries
        '''
        d = {}
        for strg in matobj._fieldnames:
            elem = matobj.__dict__[strg]
            if isinstance(elem, spio.matlab.mio5_params.mat_struct):
                d[strg] = _todict(elem)
            elif isinstance(elem, np.ndarray):
                d[strg] = _tolist(elem)
            else:
                d[strg] = elem
        return d

    def _tolist(ndarray):
        '''
        A recursive function which constructs lists from cellarrays
        (which are loaded as numpy ndarrays), recursing into the elements
        if they contain matobjects.
        '''
        elem_list = []
        for sub_elem in ndarray:
            if isinstance(sub_elem, spio.matlab.mio5_params.mat_struct):
                elem_list.append(_todict(sub_elem))
            elif isinstance(sub_elem, np.ndarray):
                elem_list.append(_tolist(sub_elem))
            else:
                elem_list.append(sub_elem)
        return elem_list
    data = spio.loadmat(filename, struct_as_record=False, squeeze_me=True)
    return _check_keys(data)


def prepare_nsd_data(nsd_data_root, out_path_base, sub):
    dataset = "nsd"
    sub = int(sub)
    stim_order_f = os.path.join(nsd_data_root, 'nsddata/experiments/nsd/nsd_expdesign.mat')
    stim_order = loadmat(stim_order_f)

    sig_train = {}
    sig_test = {}
    num_trials = 37*750
    for idx in tqdm(range(num_trials), desc="Loading nsid"):
        ''' nsdId as in design csv files'''

        nsdId = stim_order['subjectim'][sub-1, stim_order['masterordering'][idx] - 1] - 1

        if stim_order['masterordering'][idx]>1000:  # The first 1000 images are Test (Shared among participants)
            if nsdId not in sig_train:
                sig_train[nsdId] = []  # If nsdID wasn't shown yet
            sig_train[nsdId].append(idx) # add Trial number where this specific image was shown. 
        else:
            if nsdId not in sig_test:
                sig_test[nsdId] = []
            sig_test[nsdId].append(idx)

    # Apparently some images were shown only twice. huh. 

    train_im_idx = list(sig_train.keys())
    test_im_idx = list(sig_test.keys())


    roi_dir = os.path.join(nsd_data_root, 'nsddata/ppdata/subj{:02d}/func1pt8mm/roi/'.format(sub))
    betas_dir = os.path.join(nsd_data_root, 'nsddata_betas/ppdata/subj{:02d}/func1pt8mm/betas_fithrf_GLMdenoise_RR/'.format(sub))
    simple_log(f"Loading roi from {roi_dir} and betas from {betas_dir}")

    # nsdgeneral is a general ROI that was manually drawn on fsaverage covering voxels responsive to the NSD experiment in the posterior aspect of cortex.
    # https://cvnlab.slite.page/p/X_7BBMgghj/Untitled
    mask_filename = 'nsdgeneral.nii.gz'
    mask = nib.load(roi_dir+mask_filename).get_fdata() # shape 82,106,84

    num_voxel = mask[mask>0].shape[0]

    fmri = np.zeros((num_trials, num_voxel)).astype(np.float32)

    simple_log("Loading fmri Data...")
    if os.path.exists(f"sub{sub}_fmri.pkl"):
        simple_log("Loading fmri from pickle")
        with open(f"sub{sub}_fmri.pkl", "rb") as f:
            fmri = pickle.load(f)

    else:
        for i in tqdm(range(37), desc="Loading fmri data"):
            # Each of the 37 sessions is opened after one another.
            beta_filename = "betas_session{0:02d}.nii.gz".format(i+1)

            beta_f = nib.load(betas_dir+beta_filename).get_fdata().astype(np.float32) # Shape 82,106,84,750

            # only add the Voxels. that are within the mask 
            # This is 14278 Voxels for participant 1
            fmri[i*750:(i+1)*750] = beta_f[mask>0].transpose()
            del beta_f

    simple_log("fMRI Data are loaded.")

    stimuli_file = os.path.join(nsd_data_root, 'nsddata_stimuli/stimuli/nsd/nsd_stimuli.hdf5')
    simple_log(f"Loading stimuli from {stimuli_file}....")
    # Number of images that were shown in train and test
    num_train, num_test = len(train_im_idx), len(test_im_idx)

    f_stim = h5py.File(stimuli_file, 'r')
    stim = f_stim['imgBrick'][:]
    # Image.fromarray(stim[0]).save("image.png")
    # If you want to save a nice image. 

    simple_log("Stimuli are loaded.")

    # NUmber of voxels, image dimensions and color channels
    vox_dim, im_dim, im_c = num_voxel, 425, 3

    fmri_array = np.zeros((num_train,vox_dim)) # shape 8859, 15724
    stim_array = np.zeros((num_train,im_dim,im_dim,im_c))  # shape 8859, 425, 425, 3

    # Iterate all of the 8859 Training images. 
    for i,idx in tqdm(enumerate(train_im_idx), desc="setting train data means"): # 
        
        # for each of the 8859 training images fill them with the image data.
        stim_array[i] = stim[idx]

        # For all the 2/3 trials of a given image, take the mri data and get the mean for it. 
        fmri_array[i] = fmri[sorted(sig_train[idx])].mean(0)
    simple_log("Dividing Train fmri data by 300")
    fmri_array = fmri_array / 300
    # Save that average of fmri activation for the training images. 

    processed_data_folder = os.path.join(out_path_base,f'data/processed_data/{dataset}/subj{sub}')
    os.makedirs(processed_data_folder, exist_ok=True)

    fmri_train_path = os.path.join(out_path_base,f'data/processed_data/{dataset}/subj{sub}/train_fmriavg_general_sub{sub}.npy')
    np.save(fmri_train_path,fmri_array )
    simple_log(f"Saved fmri train to {fmri_train_path}")

    # Save the image data for all the 8859 training images. 
    train_img_path = os.path.join(out_path_base,f'data/processed_data/{dataset}/subj{sub}/train_stim_sub{sub}.npy')
    np.save(train_img_path,stim_array )
    simple_log(f"Saved img train to {train_img_path}")

    simple_log("Training data is saved.")


    # Do the same for the test images. 
    fmri_array = np.zeros((num_test,vox_dim))
    stim_array = np.zeros((num_test,im_dim,im_dim,im_c))
    for i,idx in tqdm(enumerate(test_im_idx), desc="Setting test data means"): # Iterate all test images

        stim_array[i] = stim[idx] # Save test image data for all test images.
        fmri_array[i] = fmri[sorted(sig_test[idx])].mean(0) # Save the mean of the fmri activation for each of the test images.
    simple_log("Dividing test fmri data by 300")
    fmri_array = fmri_array / 300

    fmri_test_path = os.path.join(out_path_base,f'data/processed_data/{dataset}/subj{sub}/test_fmriavg_general_sub{sub}.npy')
    test_img_path = os.path.join(out_path_base,f'data/processed_data/{dataset}/subj{sub}/test_stim_sub{sub}.npy')
    np.save(fmri_test_path,fmri_array) 
    simple_log(f"Saved fmri test to {fmri_test_path}")
    np.save(test_img_path,stim_array)
    simple_log(f"Saved img test to {fmri_test_path}")
    
    simple_log("Test data is saved.")

    ## Annotations. 

    ## Save also the annotations. 
    # I assume they are just the Strings that the images have been annotated with. 

    annots_cur = np.load(os.path.join(nsd_data_root,'annots/COCO_73k_annots_curated.npy'))

    captions_array = np.empty((num_train,5),dtype=annots_cur.dtype)
    for i,idx in tqdm(enumerate(train_im_idx), desc="Setting captions train"):
        captions_array[i,:] = annots_cur[idx,:]
    
    train_annot_path = os.path.join(out_path_base,f'data/processed_data/{dataset}/subj{sub}/train_cap_sub{sub}.npy')
    np.save(train_annot_path,captions_array)
    simple_log(f"Saved train annot to {train_annot_path}")
        
    captions_array = np.empty((num_test,5),dtype=annots_cur.dtype)
    for i,idx in tqdm(enumerate(test_im_idx), desc="Setting captions test"):
        captions_array[i,:] = annots_cur[idx,:]

    test_annot_path = os.path.join(out_path_base,f'data/processed_data/{dataset}/subj{sub}/test_cap_sub{sub}.npy')
    np.save(test_annot_path,captions_array)
    simple_log(f"Saved test annot to {test_annot_path}")

    simple_log("Caption data are saved.")


if __name__ == "__main__":
    sub=1
    assert sub in [1,2,5,7]
    nsd_data_root = "/home/matt/programming/brain-diffuser/data"
    out_path_base = "/home/matt/programming/recon_diffuser/data"
    prepare_nsd_data(nsd_data_root, out_path_base, sub)
