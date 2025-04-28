import multiprocessing
import time
import itertools
from datetime import datetime
from copy import deepcopy
import logging
from concurrent.futures import ProcessPoolExecutor, as_completed
import socket
import torch
import traceback
import torch.multiprocessing as mp

multiprocessing.set_start_method("spawn", force=True)
from multiprocessing import Lock

# Define locks for each device
device_locks = {
    "cuda:0": Lock(),
    "cuda:1": Lock()
}

logger = logging.getLogger("recdi_genpert_multi")


def setup_logger(fname):
    """Manually set up a logger for each worker process."""
    logger = logging.getLogger()  # This is the root logger
    logger.setLevel(logging.INFO)
    
    # Remove any previously attached handlers
    if logger.hasHandlers():
        logger.handlers.clear()
    
    # Create a new file handler for this process
    file_handler = logging.FileHandler(fname)
    stream_handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s')
    file_handler.setFormatter(formatter)
    
    # Attach the file handler to the logger
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    return logger

def gogo_generate(out_path_base, dataset, sub, input_name, perturb_params, device):
    collect_imgs = False
    from clip_module_testing import perturbations_all_images, out_name_from_perturb_params
    out_name = out_name_from_perturb_params(perturb_params)
    
    fname_log = f"{datetime.now()}_train_pert_{sub}_{dataset}_{input_name}_{out_name}.log"
    setup_logger(fname_log)
    
    logger.info(f"Let's go {fname_log}")

    perturbations_all_images(out_path_base, dataset, sub, input_name, perturb_params, device=device)

# Wrapper to start processes with different parameters
def main():
    out_path_base = "/home/matt/programming/recon_diffuser/"
    dataset = 'deeprecon'

    # Only needs to be done for subject AM! The perturbed files will be used for all subjects (they are sorted accordingly in re_clip::batch_generator_perturbed_images)
    subs = ['AM']
    input_name = 'train'

    params_base = {"out_path_base": out_path_base,
                   "dataset": dataset,
                   "input_name": input_name}

    all_perturb_params = [
        # {
        #     'perturb_kind': 'iterativeCriterion',
        #     'caption_type': 'friendly',
        #     'criterion': '90-10',
        #     'n_perts': 5,
        #     'max_pert_steps': 500
        # },
        # {
        #     'perturb_kind': 'iterativeCriterion',
        #     'caption_type': 'adversarial',
        #     'criterion': '90-10',
        #     'n_perts': 5,
        #     'max_pert_steps': 500
        # },
        # {
        #     'perturb_kind' : 'fgsm',
        #     'caption_type' : 'friendly',
        #     'epsilon' : 0.03,
        #     'n_perts': 5,
        # },
        # {
        #     'perturb_kind' : 'fgsm',
        #     'caption_type' : 'adversarial',
        #     'epsilon' : 0.03,
        #     'n_perts': 5,
        # },
        # {
        #     'perturb_kind' : 'fgsm',
        #     'caption_type' : 'friendly',
        #     'epsilon' : 0.1,
        #     'n_perts': 5,
        # },
        # {
        #     'perturb_kind' : 'fgsm',
        #     'caption_type' : 'adversarial',
        #     'epsilon' : 0.1,
        #     'n_perts': 5,
        # },
        # {
        #     'perturb_kind' : 'fgsm',
        #     'caption_type' : 'friendly',
        #     'epsilon' : 0.05,
        #     'n_perts': 5,
        # },
        # {
        #     'perturb_kind' : 'fgsm',
        #     'caption_type' : 'adversarial',
        #     'epsilon' : 0.05,
        #     'n_perts': 5,
        # },
        # {
        #     'perturb_kind' : 'fgsm',
        #     'caption_type' : 'friendly',
        #     'epsilon' : 0.2,
        #     'n_perts': 5,
        # },
        # {
        #     'perturb_kind' : 'fgsm',
        #     'caption_type' : 'adversarial',
        #     'epsilon' : 0.2,
        #     'n_perts': 5,
        # },
        # {
        #     'perturb_kind' : 'fgsm',
        #     'caption_type' : 'friendly',
        #     'epsilon' : 0,
        #     'n_perts': 5,
        # },
        # {
        #     'perturb_kind' : 'fgsm',
        #     'caption_type' : 'adversarial',
        #     'epsilon' : 0,
        #     'n_perts': 5,
        # },
        # {
        #     'perturb_kind': 'iterativeCriterion',
        #     'caption_type': 'friendly',
        #     'criterion': '70-30',
        #     'n_perts': 5,
        #     'max_pert_steps': 500
        # },
        # {
        #     'perturb_kind': 'iterativeCriterion',
        #     'caption_type': 'adversarial',
        #     'criterion': '70-30',
        #     'n_perts': 5,
        #     'max_pert_steps': 500
        # },
        # {
        #     'perturb_kind': 'iterativeCriterion',
        #     'caption_type': 'friendly',
        #     'criterion': '80-20',
        #     'n_perts': 5,
        #     'max_pert_steps': 500
        # },
        # {
        #     'perturb_kind': 'iterativeCriterion',
        #     'caption_type': 'adversarial',
        #     'criterion': '80-20',
        #     'n_perts': 5,
        #     'max_pert_steps': 500
        # },
        # {
        #     'perturb_kind': 'iterativeCriterion',
        #     'caption_type': 'friendly',
        #     'criterion': '50-50',
        #     'n_perts': 5,
        #     'max_pert_steps': 500
        # },
        # {
        #     'perturb_kind': 'iterativeCriterion',
        #     'caption_type': 'adversarial',
        #     'criterion': '50-50',
        #     'n_perts': 5,
        #     'max_pert_steps': 500
        # },
        # {
        #     'perturb_kind': 'iterativeCriterion',
        #     'caption_type': 'friendly',
        #     'criterion': '10-90',
        #     'n_perts': 5,
        #     'max_pert_steps': 500
        # },
        # {
        #     'perturb_kind': 'iterativeCriterion',
        #     'caption_type': 'adversarial',
        #     'criterion': '10-90',
        #     'n_perts': 5,
        #     'max_pert_steps': 500
        # },
        # {
        #     'perturb_kind': 'iterativeCriterion',
        #     'caption_type': 'friendly',
        #     'criterion': '51-49',
        #     'n_perts': 5,
        #     'max_pert_steps': 500
        # },
        {
            'perturb_kind': 'iterativeCriterion',
            'caption_type': 'friendly',
            'criterion': '52-48',
            'n_perts': 5,
            'max_pert_steps': 500
        },
        # {
        #     'perturb_kind': 'iterativeCriterion',
        #     'caption_type': 'adversarial',
        #     'criterion': '30-70',
        #     'n_perts': 5,
        #     'max_pert_steps': 500
        # },
    ]
    all_perturb_params = [{**params_base, "perturb_params":params} for params in all_perturb_params]

    all_perturb_params_all_subjects = []
    for sub in subs:
        for num,perturb_params in enumerate(all_perturb_params):
            all_perturb_params_all_subjects.append({**perturb_params, "sub":sub})


    # Limit the number of processes (e.g., max 2 processes at a time)
    max_processes = 2

    # Run the processes with limited concurrency
    with ProcessPoolExecutor(max_workers=max_processes) as executor:
        futures = {}
        # Submit all tasks to the pool
        for i, params in enumerate(all_perturb_params_all_subjects):
            hostname = socket.gethostname()
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            if hostname == 'smith-g10':
                device = "cuda:0" if i % 2 == 0 else "cuda:1"
            else:
                device = 'cuda'
            
            if hostname == 'smith-g06':
                modulo = i % 4
                if modulo == 0:
                    device = 'cuda:0'
                elif modulo == 1:
                    device = 'cuda:1'
                elif modulo == 2:
                    device = 'cuda:2'
                elif modulo == 3:
                    device = 'cuda:3'

            print(f"Setting device to {device} for the task")

            params['device'] = device
            futures[executor.submit(gogo_generate, **params)] = (params)

        # futures = {executor.submit(gogo_generate, **params): params for params in all_perturb_params_all_subjects}
     
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
