import pandas as pd
import re
from collections import defaultdict
import numpy as np
import os
from tqdm import tqdm

def extract_logs(log_file_path):
    ## start patterns
    pattern_start = re.compile(r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d+,\d+ root INFO Setup logger at \d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}.\d+_(?P<configname>.*?).log")
    
    pattern_config_baseline = re.compile(r"recon_deeprecon_(?P<sub>.*?)_(?P<dataset>.*)")
    pattern_config_pert = re.compile(r"recon_pert_(?P<sub>.*?)_(?P<dataset>.*?)_(?P<pert_algorithm>.*?)_(?P<pert_type>.*?)_(?P<pert_params>.*)")
    pattern_config_dropout = re.compile(r"recon_dropout_(?P<sub>.*?)_(?P<dataset>.*?)_(?P<dropout_algorithm>.*?)_(?P<dropout_ratio>.*?)_(?P<dropout_trial>.*)")

    pattern_regression_type = re.compile(r"Doing (?P<regression_type>\w+) regression")

    pattern_ic_params = re.compile(r"(?P<pert_ratio>.*?)_(?P<max_steps>\d+?)_(?P<pert_n>\d+)(?:_(?P<inputavg>noinputavg))?")
    pattern_fgsm_params = re.compile(r"(?P<epsilon>\d+\.\d+)_(?P<pert_n>\d+)(?:_(?P<inputavg>noinputavg))?")

    pattern_regscore = re.compile(r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d+,\d+ recdi_clip INFO Embedding (?P<embedding_n>\d+): (?P<test_dataset_name>.*?) reg score: (?P<reg_score>[-.\d]+)")
    pattern_vdvae_regscore = re.compile(r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d+,\d+ recdi_VDVAE INFO VDAVE Reg score on (?P<name>\w+)?[_ ].*?: (?P<reg_score>.*)")
    
    pattern_noinputavg_in_baseline = re.compile(r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d+,\d+ recdi_deeprecon_main INFO Setting output name to _noinputavg")

    records = []
    with open(log_file_path, 'r') as log_file:
        current_record = {}
        
        for line in log_file:
            match_start = pattern_start.search(line)
            if match_start:
                start_str = match_start.groups()[0]

                if match_config := pattern_config_pert.search(start_str):
                    ...
                elif match_config := pattern_config_baseline.search(start_str):
                    ...
                elif match_config := pattern_config_dropout.search(start_str):
                    ...
                else:
                    raise Exception(f"Start str {start_str} doesn't match the start matcher. Did something in the log format change?")
                config = match_config.groupdict()

                if not config.get('pert_algorithm'):
                    params_pattern = None
                elif config['pert_algorithm'] == 'fgsm':
                    params_pattern = pattern_fgsm_params
                elif config['pert_algorithm'] == 'ic':
                    params_pattern = pattern_ic_params
                else:
                    raise Exception("Algorithm not known.")
                
                if params_pattern:
                    pert_params_str = config.pop('pert_params')
                    pert_params = params_pattern.search(pert_params_str).groupdict()
                    config = {**config, **pert_params}

                current_record.update(config)
            match_noinputavg_baseline = pattern_noinputavg_in_baseline.search(line)
            if match_noinputavg_baseline:
                current_record["inputavg"] = "noinputavg"
            match_regression_type = pattern_regression_type.search(line)
            if match_regression_type:
                current_record.update(match_regression_type.groupdict())
                current_regtype = match_regression_type['regression_type']
                current_record[current_regtype + "_scores"] = []

            # Match the metrics line
            match_regscore = pattern_regscore.search(line)
            if match_regscore:
                current_record[current_regtype + "_scores"].append(match_regscore.groupdict())

            match_vdvae_regscore = pattern_vdvae_regscore.search(line)
            if match_vdvae_regscore:
                current_record[current_regtype + "_scores"].append(match_vdvae_regscore.groupdict())
                current_record[f"vdvae_{match_vdvae_regscore['name']}"] = float(match_vdvae_regscore["reg_score"])
    return current_record

def extract_all_logs(log_file_paths):
    all_records = []
    for log_file_path in tqdm(log_file_paths, desc='Parsing regression logs...'):
        record = extract_logs(log_file_path)
        all_records.append(record)
    records_with_mean_scores = []
    for record in tqdm(all_records, desc="Computing means of regression scores..."):
        mean_scores = {}
        for regtype in ['clipvision', 'cliptext']:
            if f"{regtype}_scores" in record.keys():
                scores_per_test_dataset = defaultdict(list)
                scores = record[f"{regtype}_scores"]
                for score in scores:
                    scores_per_test_dataset[score['test_dataset_name']].append(float(score['reg_score']))

                for test_dataset_name, scores in scores_per_test_dataset.items():
                    mean_scores[f"{regtype}_{test_dataset_name}_mean_reg_score"] = np.mean(scores)
        records_with_mean_scores.append({**record, **mean_scores})

    df = pd.DataFrame(records_with_mean_scores)
    return df

def extract_all_logs_from_directory(logs_dir):    
    all_log_files = [os.path.join(logs_dir, f) for f in os.listdir(logs_dir)]
    df = extract_all_logs(all_log_files)
    return df


if __name__ == "__main__":
    ...
    log_file_path = '/home/matt/programming/recon_diffuser/analysis/progress_presentation_1/regression_log/2024-11-15 02:27:18.505598_recon_deeprecon_AM_deeprecon_noinputavg.log'
    log_file_path = "/home/matt/programming/recon_diffuser/analysis/progress_presentation_1/regression_log/2024-11-15 02:24:44.579659_recon_pert_AM_deeprecon_ic_adversarial_90-10_500_5_noinputavg.log"
    log_file_path = "/home/matt/programming/recon_diffuser/analysis/progress-presentation_2/regression_log/2024-11-22 22:59:04.820245_recon_dropout_AM_deeprecon_dropout-random_0.1_00.log"
    log_file_path = "/home/matt/programming/recon_diffuser/analysis/progress_presentation_2/regression_log/2024-11-23 05:23:19.021350_recon_deeprecon_AM_deeprecon.log"
    # log_file_path = "/home/matt/programming/recon_diffuser/analysis/progress_presentation_1/regression_log/2024-11-14 22:01:54.665110_recon_pert_AM_deeprecon_fgsm_adversarial_0.03_5_noinputavg.log"
    # log_file_path = '/home/matt/programming/recon_diffuser/src/recon_diffuser/2024-11-06 03:37:16.956715_recon_pert_AM_deeprecon_fgsm_adversarial_0.05_5.log'
    # log_file_path = '/home/matt/programming/recon_diffuser/src/recon_diffuser/2024-11-06 18:18:00.948096_recon_pert_AM_deeprecon_ic_friendly_80-20_500_5.log'

    # Define all log_file_paths that should be evaluated
    # log_file_paths = ['/home/matt/programming/recon_diffuser/src/recon_diffuser/2024-11-06 03:37:16.956715_recon_pert_AM_deeprecon_fgsm_adversarial_0.05_5.log', '/home/matt/programming/recon_diffuser/src/recon_diffuser/2024-11-06 18:18:00.948096_recon_pert_AM_deeprecon_ic_friendly_80-20_500_5.log']

