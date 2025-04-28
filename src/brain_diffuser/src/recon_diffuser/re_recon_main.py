import multiprocessing
import time
import itertools
from datetime import datetime
import logging
from concurrent.futures import ProcessPoolExecutor, as_completed
from copy import deepcopy
import traceback
import os

logger = logging.getLogger("recdi_genpert_multi")


def setup_logger(fname):
    """Manually set up a logger for each worker process."""
    logger = logging.getLogger() 
    logger.setLevel(logging.INFO)
    if logger.hasHandlers():
        logger.handlers.clear()
    file_handler = logging.FileHandler(os.path.join("logs", fname))
    stream_handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    logger.info(f"Setup logger at {fname}")
    return logger


def do_nsd_main():
    ...

def do_deeprecon_main(out_path_base, deeprecon_data_root, image_data_root, captions_data_root, dataset, sub, compute_train_mean=True):
    fname_log = f"{datetime.now()}_recon_deeprecon_{sub}_{dataset}.log"
    setup_logger(fname_log)
    from re_deeprecon_main import main as deeprecon_main
    deeprecon_main(out_path_base, deeprecon_data_root, image_data_root, captions_data_root, dataset, sub, compute_train_mean=compute_train_mean)

def do_things_main():
    ...

def do_pert_main(out_path_base, dataset, sub, output_name):
    fname_log = f"{datetime.now()}_recon_pert_{sub}_{dataset}_{output_name}.log"
    setup_logger(fname_log)
    from re_perturbation_reconstruction_main import main as pert_main
    pert_main(out_path_base, dataset, sub, output_name)

def do_dropout_recon_main(out_path_base, dataset, sub, output_name):
    fname_log = f"{datetime.now()}_recon_dropout_{sub}_{dataset}_{output_name}.log"
    setup_logger(fname_log)
    from re_dropout_recon_main import main as dropout_recon_main
    dropout_recon_main(out_path_base, dataset, sub, output_name)

def do_dropout_icnn_main(out_path_base, dataset, sub, output_name, scale_features, device, resize_size, n_iter):
    output_name_for_log = "_".join((output_name, f"size{resize_size}", f"iter{n_iter}"))
    if scale_features:
        output_name_for_log += "_scaled"

    fname_log = f"{datetime.now()}_recon_icnn_dropout_{sub}_{dataset}_{output_name_for_log}.log"
    setup_logger(fname_log)
    from re_icnn_main import icnn_main
    icnn_main(out_path_base, dataset, sub, device, save_features=True, output_name=output_name, resize_size=resize_size, scale_features=scale_features, n_iter=n_iter)

def do_aicap_main(out_path_base, dataset, sub, output_name):
    fname_log = f"{datetime.now()}_aicap_{sub}_{dataset}_{output_name}.log"
    setup_logger(fname_log)
    from re_ai_captions_recon_main import main as aicap_main
    aicap_main(out_path_base, dataset, sub, output_name)

def do_true_recon_main(out_path_base, dataset, sub, output_name):
    fname_log = f"{datetime.now()}true_feat_recon_{sub}_{dataset}_{output_name}.log"
    setup_logger(fname_log)
    from re_true_feature_recon import true_features_main 
    true_features_main(out_path_base, dataset, sub, output_name)

def main_dispatcher(**params):
    params = deepcopy(params)
    main_func = params.pop('main_func')
    
    if main_func == "do_pert_main":
        do_pert_main(**params)
    elif main_func == "do_deeprecon_main":
        do_deeprecon_main(**params)
    elif main_func == "do_dropout_recon_main":
        do_dropout_recon_main(**params)
    elif main_func == "do_dropout_icnn_main":
        do_dropout_icnn_main(**params)
    elif main_func == "do_aicap_main":
        do_aicap_main(**params)
    elif main_func == "do_true_recon_main":
        do_true_recon_main(**params)
    else:
        raise NotImplementedError(f"{main_func=} not supported yet.")

def generate_dropout_icnn_tasks():
    out_path_base = "/home/matt/programming/recon_diffuser/"
    dataset = 'deeprecon'
    subs = ['AM', 'KS', 'ES', 'JK', 'TH', ]
    # subs = ['AM']
    base_params_dropout_recon = {
        "main_func": "do_dropout_icnn_main",
        "out_path_base": out_path_base,
        "dataset": dataset,
        "n_iter": 500,
        "resize_size": 224,
        "device": "cuda:0",
        "scale_features": True}

    configuration_params = [
        ## Baseline should've been executed before already.
        # {"output_name": ""},
        ## All the random ones only need to be exectuted for subject AM, they're part of the evaluation. 
        # {"output_name": "dropout-random_0.1_00"},
        # {"output_name": "dropout-random_0.1_01"},
        # {"output_name": "dropout-random_0.1_02"},
        # {"output_name": "dropout-random_0.1_03"},
        # {"output_name": "dropout-random_0.1_04"},

        # {"output_name": "dropout-random_0.25_00"},
        # {"output_name": "dropout-random_0.25_01"},
        # {"output_name": "dropout-random_0.25_02"},
        # {"output_name": "dropout-random_0.25_03"},
        # {"output_name": "dropout-random_0.25_04"},

        # {"output_name": "dropout-random_0.5_00"},
        # {"output_name": "dropout-random_0.5_01"},
        # {"output_name": "dropout-random_0.5_02"},
        # {"output_name": "dropout-random_0.5_03"},
        # {"output_name": "dropout-random_0.5_04"},

        # {"output_name": "dropout-random_0.75_00"},
        # {"output_name": "dropout-random_0.75_01"},
        # {"output_name": "dropout-random_0.75_02"},
        # {"output_name": "dropout-random_0.75_03"},
        # {"output_name": "dropout-random_0.75_04"},

        # Best ones in their representative domain
        # {"output_name": "dropout-dreamsim_0.1_11"},
        # {"output_name": "dropout-clipvision_0.1_33"},
        # random and pixels are a random selection
        # {"output_name": "dropout-random_0.1_22"},
        # {"output_name": "dropout-pixels_0.1_44"},

        # Best ones in their representative domain
        # {"output_name": "dropout-dreamsim_0.25_55"},
        # {"output_name": "dropout-clipvision_0.25_88"},
        # random and pixels are a random selection
        # {"output_name": "dropout-random_0.25_33"},
        # {"output_name": "dropout-pixels_0.25_44"},

        ## Discussion plots for the most boring and interesting images
        # {"output_name": "dropout-quantizedCountBoring_0.25_00"},
        # {"output_name": "dropout-quantizedCountParty_0.25_00"},
    ]
    all_params = []
    for sub in subs:
        for configuration in configuration_params:
            all_param_dict = {"sub": sub, **base_params_dropout_recon, **configuration}
            all_params.append(all_param_dict)
    return all_params

def generate_dropout_tasks():
    out_path_base = "/home/matt/programming/recon_diffuser/"
    dataset = 'deeprecon'
    subs = ['AM', 'KS', 'ES', 'JK', 'TH', ]
    # subs = ['AM']

    base_params_dropout_recon = {
        "main_func": "do_dropout_recon_main",
        "out_path_base": out_path_base,
        "dataset": dataset}
    
    configuration_params = [
        ## Baseline should've been executed before already.
        # {"output_name": ""},
        ## All the random ones only need to be exectuted for subject AM, they're part of the evaluation. 
        # {"output_name": "dropout-random_0.1_00"}, 
        # {"output_name": "dropout-random_0.1_01"},
        # {"output_name": "dropout-random_0.1_02"},
        # {"output_name": "dropout-random_0.1_03"},
        # {"output_name": "dropout-random_0.1_04"},

        # {"output_name": "dropout-random_0.25_00"},
        # {"output_name": "dropout-random_0.25_01"},
        # {"output_name": "dropout-random_0.25_02"},
        # {"output_name": "dropout-random_0.25_03"},
        # {"output_name": "dropout-random_0.25_04"},

        # {"output_name": "dropout-random_0.5_00"},
        # {"output_name": "dropout-random_0.5_01"},
        # {"output_name": "dropout-random_0.5_02"},
        # {"output_name": "dropout-random_0.5_03"},
        # {"output_name": "dropout-random_0.5_04"},

        # {"output_name": "dropout-random_0.75_00"},
        # {"output_name": "dropout-random_0.75_01"},
        # {"output_name": "dropout-random_0.75_02"},
        # {"output_name": "dropout-random_0.75_03"},
        # {"output_name": "dropout-random_0.75_04"},

        # # Best ones in their representative domain
        # {"output_name": "dropout-dreamsim_0.1_11"},
        # {"output_name": "dropout-clipvision_0.1_33"},
        # # random and pixels are a random selection
        # {"output_name": "dropout-random_0.1_22"},
        # {"output_name": "dropout-pixels_0.1_44"},

        # Best ones in their representative domain
        # {"output_name": "dropout-dreamsim_0.25_55"},
        # {"output_name": "dropout-clipvision_0.25_88"},
        # random and pixels are a random selection
        # {"output_name": "dropout-random_0.25_33"},
        # {"output_name": "dropout-pixels_0.25_44"},

        ## Discussion plots for the most boring and interesting images
        # {"output_name": "dropout-quantizedCountBoring_0.25_00"},
        # {"output_name": "dropout-quantizedCountParty_0.25_00"},
    ]

    all_params = []
    for sub in subs:
        for configuration in configuration_params:
            all_param_dict = {"sub": sub, **base_params_dropout_recon, **configuration}
            all_params.append(all_param_dict)
    return all_params

def generate_pert_tasks():
    all_params = []
    out_path_base = "/home/matt/programming/recon_diffuser/"
    dataset = 'deeprecon'
    subs = ['AM','KS', 'ES', 'JK', 'TH']
    # subs = ['TH']
    # smith7 macht gerade fgsm 0.03 friendly jp und th

    base_params_pert_recon = {'main_func': 'do_pert_main',
         'out_path_base': out_path_base,
         'dataset': dataset}

    configuration_params = [
        # {"output_name": "fgsm_friendly_0_5"}, # defacto baseline

        # {"output_name": "fgsm_adversarial_0.1_5"}, 
        # {"output_name": "fgsm_friendly_0.1_5"}, 
        # {"output_name": "fgsm_adversarial_0.03_5"}, 
        # {"output_name": "fgsm_friendly_0.03_5"}, 
        # {"output_name": "ic_adversarial_90-10_500_5"}, 
        # {"output_name": "ic_friendly_90-10_500_5"}, 

        # {"output_name": "ic_adversarial_50-50_500_5"}, 
        # {"output_name": "ic_friendly_50-50_500_5"}, 
        # {"output_name": "ic_adversarial_70-30_500_5"}, 
        # {"output_name": "ic_friendly_70-30_500_5"}, 
        # {"output_name": "ic_adversarial_80-20_500_5"}, 
        # {"output_name": "ic_friendly_80-20_500_5"}, 

    ]

    all_params = []
    for sub in subs:
        for configuration in configuration_params:
            all_param_dict = {"sub": sub, **base_params_pert_recon, **configuration}
            all_params.append(all_param_dict)
    return all_params

def generate_aicap_tasks():
    all_params = []
    out_path_base = "/home/matt/programming/recon_diffuser/"
    dataset = 'deeprecon'
    subs = ['AM', 'KS', 'ES', 'JK', 'TH', ]
    # subs = ['AM']


    base_params_pert_recon = {'main_func': 'do_aicap_main',
         'out_path_base': out_path_base,
         'dataset': dataset}

    configuration_params = [
        # {"output_name": "aicap_human_captions"},
        # {"output_name": "aicap_low_level_short"},
        # {"output_name": "aicap_high_level_short"},
        # {"output_name": "aicap_human_captions_shuffled_single_caption"},
        

        # {"output_name": "aicap_human_captions-usetruecliptext_True"},

        # Only for subject AM
        # {"output_name": "aicap_human_captions-mix_0.0"},
        # {"output_name": "aicap_human_captions-mix_0.25"},
        # {"output_name": "aicap_human_captions-mix_0.5"},
        # {"output_name": "aicap_human_captions-mix_0.75"},
        # {"output_name": "aicap_human_captions-mix_0.99"},

        # {"output_name": "aicap_human_captions-usetruecliptext_True-mix_0.0"},
        # {"output_name": "aicap_human_captions-usetruecliptext_True-mix_0.25"},
        # {"output_name": "aicap_human_captions-usetruecliptext_True-mix_0.5"},
        # {"output_name": "aicap_human_captions-usetruecliptext_True-mix_0.75"},
        # {"output_name": "aicap_human_captions-usetruecliptext_True-mix_0.99"},

        # {"output_name": "aicap_human_captions_shuffled_single_caption-mix_0.0"},
        # {"output_name": "aicap_human_captions_shuffled_single_caption-mix_0.25"},
        # {"output_name": "aicap_human_captions_shuffled_single_caption-mix_0.5"},
        # {"output_name": "aicap_human_captions_shuffled_single_caption-mix_0.75"},
        # {"output_name": "aicap_human_captions_shuffled_single_caption-mix_0.99"},

        # {"output_name": "aicap_human_captions-mix_0.8"}, # Baseline mix 80
        # {"output_name": "aicap_human_captions_shuffled_single_caption-mix_0.8"}, 
        # {"output_name": "aicap_low_level_short-mix_0.8"},
        # {"output_name": "aicap_high_level_short-mix_0.8"},
        ]

    all_params = []
    for sub in subs:
        for configuration in configuration_params:
            all_param_dict = {"sub": sub, **base_params_pert_recon, **configuration}
            all_params.append(all_param_dict)
    return all_params

def generate_true_feature_recon_tasks():
    all_params = []
    out_path_base = "/home/matt/programming/recon_diffuser/"
    dataset = 'deeprecon'
    subs = ['AM']

    base_params_true_recon = {'main_func': 'do_true_recon_main',
         'out_path_base': out_path_base,
         'dataset': dataset}

    configuration_params = [
        # {"output_name": "true_icnn"}
        # {"output_name": "true_bd"}
        # {"output_name": "true_bd-mix_0.0"}
    ]

    all_params = []
    for sub in subs:
        for configuration in configuration_params:
            all_param_dict = {"sub": sub, **base_params_true_recon, **configuration}
            all_params.append(all_param_dict)
    return all_params

def generate_deeprecon_tasks():
    deeprecon_data_root = "/home/kiss/data/fmri_shared/datasets/Deeprecon/fmriprep"
    image_data_root = "/home/kiss/data/contents_shared"
    captions_data_root = "/home/matt/programming/recon_diffuser/data/annots"

    all_params = []
    out_path_base = "/home/matt/programming/recon_diffuser/"
    dataset = 'deeprecon'
    subs = ['AM', 'KS', 'ES', 'JK', 'TH']

    base_params_deeprecon_recon = {
        'main_func': 'do_deeprecon_main',
        'out_path_base': out_path_base,
        'dataset': dataset,
        'deeprecon_data_root': deeprecon_data_root,
        'image_data_root': image_data_root,
        'captions_data_root': captions_data_root,
        # 'compute_train_mean': False
        }

    all_params = []
    for sub in subs:
        all_param_dict = {"sub": sub, **base_params_deeprecon_recon}
        all_params.append(all_param_dict)
    return all_params


# Wrapper to start processes with different parameters
def main():

    all_params = []

    all_params += generate_pert_tasks()
    # all_params += generate_deeprecon_tasks()
    all_params += generate_dropout_tasks()
    all_params += generate_dropout_icnn_tasks()
    all_params += generate_aicap_tasks()
    all_params += generate_true_feature_recon_tasks()

    print(f"Doing recon for {len(all_params)} configurations...")

    # Limit the number of processes (e.g., max 2 processes at a time)
    max_processes = 1

    # Run the processes with limited concurrency
    with ProcessPoolExecutor(max_workers=max_processes) as executor:
        # Submit all tasks to the pool
        futures = {executor.submit(main_dispatcher, **params): params for params in all_params}
        
        # Retrieve results as each task completes
        for future in as_completed(futures):
            params = futures[future]
            try:
                result = future.result()
                print(f"Completed with params {params}: {result}")
            except Exception as exc:
                print(f"Generated an exception: {exc} with params {params} Full traceback: {traceback.format_exc()}")


if __name__ == "__main__":
    main()
    ...
