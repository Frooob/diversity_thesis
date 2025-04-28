# recon-diffuser

This repo facilitates the usage of the "Brain-Diffuser" algorithm described in Ozcelik et. al 2023 "Natural scene reconstruction from fMRI signals using generative latent diffusion". Most of the code in the repo here is based on Ozceliks [github repo](https://github.com/ozcelikfu/brain-diffuser). The code has been thoroughly refactored in order to support multiple datasets. Curently both NSD and deeprecon datasets are supported. This repo also contains convenient scripts that facilitate the usage of the whole workflow or only parts of it. 

## Important note on reproducing the results

All human generated captions from the Deeprecon dataset (both for artificial shapes and training data) has been removed from this repository due to copyright issues. The captions that were generated using an AI can still be used (the high-level captions are very similar to human generated captions). 

Also, the reproduction without access to the Kyoto University servers may be bothersome, because the data formats are not laid out perfectly. Reproducing with the public NSD dataset may be easier than with the Deeprecon dataset.

## Usage

First you need to get your data in order. For the NSD dataset, execute the re_download_nsddata.py script. For deeprecon, make sure you know where to find the files on your harddrive. 
Second, you need to download the ML-models that are described in Ozceliks github repo. 
Execute either the nsd_main.py or deeprecon_main.py scripts to run the whole pipeline (first change the paths in the scripts to fit to your needs and input data locations). These two scripts will in turn execute multiple steps for the whole algorithm. Below is a description of each of the involved scripts. 

### Installing Requirements
It's crucial to have the correct requirements, since the image reconstruction will not work if the requirements do not match the previously downloaded models. 

First install the conda environment using ```conda env create -f environment_brain-diffuser2.yml```. For the evaluation scripts, you need to install the environment ```conda env create -f environment_brain-diffuser2.yml```. 

The requirement ```clip``` needs to be separately installed by using: ```pip install git+https://github.com/openai/CLIP.git```. 

Also you need to make sure that ```bdpy``` is installed. Usually you might want to install it directly from [source](https://github.com/KamitaniLab/bdpy). Using ```pip install git+https://github.com/KamitaniLab/bdpy.git```.

### Data Formats
For the description of the input data format for NSD and deeprecon have a look at the respective documentation of the datasets. 

During the prepare_data step for each of the datasets a common data format is defined to streamline the subsequent analysis for all datasets.

For each dataset of each participants the following files will be created:
- ```{name}_fmriavg_general_sub{sub}.npy```: n_img x n_voxel array of mri activation for each of the presented images. For each image, the mean activation across all presentations is saved.
- ```{name}_stim_sub{sub}.npy```: n_img x n_width x n_height x n_color_channel array of image data for each of the 50 images. Same order of images as the fmri data array.
- ```{name}_cap_sub{sub}.npy```: n_img x n_captions_per_img array with the string captions for each image. Again the same ordering as in the fmri activation data. 

### Datasets
You can use the codebase to work on different datasets. This is NSD, THINGS-fmri1 and Deeprecon. For NSD and THINGS, only the main analysis (baseline) can be executed. For the deeprecon dataset, three experiments are available (dataset dropout, aicaptions and adversarial perturbations.)

#### NSD
For NSD, as the original author described for the participants [1,2,5,7] all test data exist. That's why they are used for the analysis.

#### Deeprecon
Deeprecon is the dataset recorded by kamitani lab. The amount of recorded sessions varies by participant. The data is stored in ```"/home/kiss/data/fmri_shared/datasets/Deeprecon/fmriprep"```. The data is stored in ```{SUB}_ImageNetTest_volume_native.h5``` for test data, ```{SUB}_ArtificialShapes_volume_native.h5``` for the artificial shapes test dataset and ```{SUB}_ImageNetTraining_volume_native.h5``` for the test dataset. All images were presented 5 times. The BOLD signal was averaged across all trials. For some participants only 3 or even 1 repetition of each image exist. 

To find out which participants are eligible you can use this snippet:
```python
import os
files =  os.listdir('/home/kiss/data/fmri_shared/datasets/Deeprecon/fmriprep')
p5 = [f[:2] for f in files if f.endswith('ImageNetTraining_volume_native.h5')]
full_participants = []
for SUB in p5:
    test_exists = f'{SUB}_ImageNetTest_volume_native.h5' in files
    art_exists = f'{SUB}_ArtificialShapes_volume_native.h5' in files
    train_exists = f'{SUB}_ImageNetTraining_volume_native.h5' in files
    if test_exists and art_exists and train_exists:
        full_participants.append(SUB)
print(f'{full_participants} can be used.')
```
Participants that have completed all 5 iterations and provide all test sets are:
```['KS', 'AM', 'ES', 'JK', 'JP', 'TH']```

However Subject JP will be discarded due to problems in the recording procedure.

#### THINGS-fmri1
[THINGS](https://elifesciences.org/articles/82580) is another dataset for fmri object recognition. 

The data root on the KU server is in ```/mnt/smith-bk02-a/share/storage/sync/sync-brain/fmri/datasets/THINGS/```. In the ```README.md``` there is a description of the available ROIs. 

Same as for deeprecon you have to evaluation/things_save_test_images.py

### Execution

You might need to change a few paths in order to properly execute the whole pipeline, make sure that the out_path_base in all the scripts points to the root of this project. 

#### Baseline
For the baseline experiment, you need to execute either of the main scripts for the datsets, ```nsd_main.py```, ```deeprecon_main.py``` or ```things_main.py```.


#### Experiment 1 Dropout

For the dropout experiment, you'll first need to create the training data subsamples. 
First, execute dropout/dropout_random.py to create the random subsamples. The diversity based subsamples are created by executing dropout/low_level_clustering.py

The monotone/heterogeneous data subset can be created by executing dropout/plain_background-subset.py.

Now, when all the subsets are in place, the reconstruction pipeline can be executed. In order to do so, uncomment the appropriate settings you want to use in re_recon_main.py::generate_dropout_tasks and re_recon_main.py::generate_dropout_icnn_tasks. Then you can execute the script and let the reconstruction run.


#### Experiment 2 AI-Cap

For the experiment with the AI generated captions, you can create new captions using the script in ai_captions/parallel_queries.py and create new shuffled captions using shuffle_human_captions.py. Or you can simply use the already created captions. Note, that the captions created by humans are not available publicly. 

After all captions are in place, you can uncomment the configurations that you want to run in re_recon_main.py::generate_aicap_tasks and run the script to start the reconstructions.


#### Experiment 3 adversarial perturbations

First you need to generate the perturbations, in order to do so, uncommend to corresponding configurations in adv_pert/multi_process_pert_generation.py and execute the script to create the perturbations. 

Once all the perturbed training images have been created, you can uncommt all configurations that you want to run in re_recon_main.py::generate_pert_tasks and run the script to start the reconstructions.


### Evaluation

Two main ways of evaluation exist: qualitative and quantitative. Qualitative evaluation includes comparing the reconstructed and actual image data by looking at them side by side. 
Quantitative evaluation can be done with different algorithms like pixel-correlation. In the original paper by Ozcelik et. al. several quantiative measures were used: PixCorr, SSIM, AlexNet(2), AlexNet(5), Inception, CLIP, EffNet-B and SwAV.

#### Qualitative Evaluation
You can use the script ```evaluation/qualitative_eval.py``` to create some images and see how ground truth / autoencoder / versatile diffusion of an image looks like. The plots will be saved to the specified path at the top. 

#### Quantitative Evaluation
You need to uncomment the configurations that you want to evaluate in the script evaluation/quantitative_eval.py and then run the script. 

##### Low-level metrics
- 'PixelCorrelation'
- 'SSIM'
- 'alexnet', layer '2', 'pairwise_corr'
- 'alexnet', layer '5', 'pairwise_corr'

##### High-level metrics
- 'inceptionv3',layer 'avgpool', 'pairwise_corr'
- 'clip',layer 'final', 'pairwise_corr'
- 'efficientnet',layer 'avgpool', 'distance'
- 'swav',layer 'avgpool', 'distance'
