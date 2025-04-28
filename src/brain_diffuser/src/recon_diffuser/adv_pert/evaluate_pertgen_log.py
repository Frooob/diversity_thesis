import pandas as pd
import re
import os
from tqdm import tqdm

def extract_pert_gen_info(log_file_path):
    ## start patterns
    perturb_pattern_adversarial = re.compile(r"(?P<datetime>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d+).*Doing perturbation for (?P<config>.*)? Image (?P<im_idx>\d+)\(adv index (?P<adv_index>\d+); caption (?P<caption_n>\d+)\)")
    perturb_pattern_friendly = re.compile(r"(?P<datetime>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d+).*Doing perturbation for (?P<config>.*)? Image (?P<im_idx>\d+) caption (?P<caption_n>\d+)")
    # perturb_pattern_ic_friendly = re.compile(r"(?P<datetime>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d+).*Doing perturbation for (?P<config>.*)? Image (?P<im_idx>\d+) caption (?P<caption_n>\d+)")


    ## config patterns
    # 1. dataset 2. name 3. sub 4. pert_type 5. caption_type 6. epsilon 7. pert_n
    fgsm_config_pattern = re.compile(r"(?P<dataset>.*)?-(?P<name>.*)?-(?P<sub>.*)?-(?P<pert_type>.*)?_(?P<caption_type>.*)?_(?P<epsilon>.*)?_(?P<pert_n>.*)?.")
    # deeprecon-train-AM-ic_friendly_80-20_500_5
    # 1. datset 2. name 3. sub 4. pert_type 5. caption_type 6. pert_ratio 7. max_steps 8. pert_n
    ic_config_pattern = re.compile(r"(?P<dataset>.*)?-(?P<name>.*)?-(?P<sub>.*)?-(?P<pert_type>.*)?_(?P<caption_type>.*)?_(?P<pert_ratio>.*)?_(?P<max_steps>.*)?_(P<pert_n>.*)?.")

    ## metrics patterns

    final_metrics_pattern = r"\(MSE: (?P<mse>[\d\.]+); SSIM: (?P<ssim>[\d\.]+); initial_gt_loss: (?P<initial_gt_loss>[\d\.]+); initial_feature_loss: (?P<initial_feature_loss>[\d\.]+); gt_loss: (?P<gt_loss>[\d\.]+); feature_loss: (?P<feature_loss>[\d\.]+)\)\. It took (?P<duration>[\d\.]+)s"

    fgsm_metrics_pattern = re.compile(
        r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d+).*Finished fgsm step\. " + final_metrics_pattern
    )

    ic_metrics_pattern = re.compile(
        r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d+).*?recdi_adp.*?(?P<steps>\d+).*?"+final_metrics_pattern
    )


    # Function to parse log file and extract data

    if "fgsm_adversarial" in log_file_path:
        perturb_pattern = perturb_pattern_adversarial
        config_pattern = fgsm_config_pattern
        metrics_pattern = fgsm_metrics_pattern
    elif "fgsm_friendly" in log_file_path:
        perturb_pattern = perturb_pattern_friendly
        config_pattern = fgsm_config_pattern
        metrics_pattern = fgsm_metrics_pattern
    elif "ic_friendly" in log_file_path:
        perturb_pattern = perturb_pattern_friendly
        config_pattern = ic_config_pattern
        metrics_pattern = ic_metrics_pattern
    elif "ic_adversarial" in log_file_path:
        perturb_pattern = perturb_pattern_adversarial
        config_pattern = ic_config_pattern
        metrics_pattern = ic_metrics_pattern
    else:
        raise ValueError(f"Cannot find correct patterns for logfile at {log_file_path}")


    records = []
    with open(log_file_path, 'r') as log_file:
        current_record = {}
        
        for line in log_file:
            perturb_match = perturb_pattern.search(line)
            if perturb_match:
                perturb_match_dict = perturb_match.groupdict()
                config = perturb_match_dict.pop('config')
                config_match = config_pattern.search(config)
                if not config_match:
                    raise ValueError("Couldn't match config. That's bad.")

                current_record.update(config_match.groupdict())
                current_record.update(perturb_match_dict)
                continue
            
            # Match the metrics line
            metrics_match = metrics_pattern.search(line)
            if metrics_match:
                current_record.update(metrics_match.groupdict())
                records.append(current_record)
                current_record = {}

    df = pd.DataFrame(records)
    return df

def extract_pertgen_all_from_folder(log_folder):
    all_log_files = [os.path.join(log_folder, f) for f in os.listdir(log_folder) if f.endswith(".log")]
    all_log_dfs = [extract_pert_gen_info(f) for f in tqdm(all_log_files, desc="Parsing pertgen log file...")]
    df_all = pd.concat(all_log_dfs)
    df_all = df_all.apply(pd.to_numeric, errors='ignore')
    return df_all


if __name__ == "__main__":
    p_fgsm_adverserial = '/home/matt/programming/recon_diffuser/src/recon_diffuser/adv_pert/2024-11-05 23:45:47.085121_train_pert_AM_deeprecon_train_fgsm_adversarial_0.1_5.log'
    p_fgsm_friendly = '/home/matt/programming/recon_diffuser/src/recon_diffuser/adv_pert/2024-11-05 23:16:43.176698_train_pert_AM_deeprecon_train_fgsm_friendly_0.1_5.log'

    p_ic_adversarial = '/home/matt/programming/recon_diffuser/src/recon_diffuser/adv_pert/2024-11-06 01:36:13.555165_train_pert_AM_deeprecon_train_ic_adversarial_90-10_500_5.log'
    p_ic_friendly = '/home/matt/programming/recon_diffuser/src/recon_diffuser/adv_pert/2024-11-05 23:03:25.478247_train_pert_AM_deeprecon_train_ic_friendly_80-20_500_5.log'

    log_file_path = p_ic_adversarial
    df_fgsm_adversarial = extract_pert_gen_info(p_fgsm_adverserial)
    df_fgsm_friendly = extract_pert_gen_info(p_fgsm_friendly)

    df_ic_adversarial = extract_pert_gen_info(p_ic_adversarial)
    df_ic_friendly = extract_pert_gen_info(p_ic_friendly)

    criteria = ['ssim', 'mse']

    df_all = pd.concat([df_fgsm_adversarial, df_fgsm_friendly, df_ic_adversarial, df_ic_friendly])

    df_all = df_all.apply(pd.to_numeric, errors='ignore')