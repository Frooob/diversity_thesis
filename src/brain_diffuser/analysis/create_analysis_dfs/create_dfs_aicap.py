# Alle dfs sind schon da. 

# Hier müssen sowohl die perturbation logs als auch die reconstruction, regression results drin sein.

# Bei den profile/patterns gibt es errors.
# Wahrscheinlich weil vdvae und cliptext hier gar nicht berechnet wurden

import pandas as pd
import os
import seaborn as sns
import sys
from collections import defaultdict
sys.path.append('/home/matt/programming/recon_diffuser/src/recon_diffuser')
from re_ai_captions_utils import get_prompt_name_and_params_from_output_name

out_path_base = '/home/matt/programming/recon_diffuser/'

subjects = ['AM', 'KS', 'ES', 'JK', 'TH', ]
dataset = 'deeprecon'

bd_conditions = [
    # Discussion Experiments only for sub AM
    'aicap_human_captions-mix_0.0',
    'aicap_human_captions-mix_0.25',
    'aicap_human_captions-mix_0.5',
    'aicap_human_captions-mix_0.75',
    'aicap_human_captions-mix_0.99',
    'aicap_human_captions-usetruecliptext_True-mix_0.0',
    'aicap_human_captions-usetruecliptext_True-mix_0.25',
    'aicap_human_captions-usetruecliptext_True-mix_0.5',
    'aicap_human_captions-usetruecliptext_True-mix_0.75',
    'aicap_human_captions-usetruecliptext_True-mix_0.99',
    'aicap_human_captions_shuffled_single_caption-mix_0.0',
    'aicap_human_captions_shuffled_single_caption-mix_0.25',
    'aicap_human_captions_shuffled_single_caption-mix_0.5',
    'aicap_human_captions_shuffled_single_caption-mix_0.75',
    'aicap_human_captions_shuffled_single_caption-mix_0.99',

    # All subjects experiment
    "aicap_human_captions", # Baseline mix 40
    "aicap_human_captions_shuffled_single_caption", 
    "aicap_low_level_short",
    "aicap_low_level_long",    
    "aicap_high_level_short",
    "aicap_high_level_long",

    "aicap_human_captions-mix_0.8", # Baseline mix 80
    "aicap_human_captions_shuffled_single_caption-mix_0.8", 
    "aicap_low_level_short-mix_0.8",
    "aicap_low_level_long-mix_0.8",    
    "aicap_high_level_short-mix_0.8",
    "aicap_high_level_long-mix_0.8",
]

sub = subjects[0]

dfs_bd_pattern_correlation = []
dfs_bd_profile_correlation = []
dfs_bd_reconstrucion = []
for sub in subjects:
    print(sub)
    # Fetch both the regression results and the reconstruction results.
    for condition in bd_conditions:
        for name in ['test', 'art']:
            oname = name if condition == '' else "_".join((name, condition))
            results_path = os.path.join(
                out_path_base, 'results', 'versatile_diffusion', dataset, f"subj{sub}", oname)
            
            # Get metadata
            if "aicap" in condition:
                prompt_name, params = get_prompt_name_and_params_from_output_name(condition)
            else:
                prompt_name, params = "NO", {}


            meta_information = {
                "prompt_name": prompt_name,
                "mixing": params.get('mix', 0.4),
                "badshuffle": params.get('badshuffle', False),
                "usetruecliptext": params.get("usetruecliptext", False),
                "dataset": dataset,
                "sub": sub,
                "test_dataset": name,
                "algorithm":"brain_diffuser"}

            if meta_information["usetruecliptext"] and sub != "AM":
                continue
            
            if meta_information["mixing"] not in [.4, .8] and sub != "AM":
                continue
            
            if meta_information["badshuffle"] and sub != "AM":
                continue

            # if meta_information["prompt_name"] == "human_captions" and meta_information["mixing"] != .4:
            #     continue


            # Pattern Correlation
            df_pattern_correlation = pd.concat([pd.read_csv(os.path.join(results_path, f"{n}_pattern_corr.csv"), index_col=0) for n in ["vdvae", "cliptext", "clipvision"] ], axis=1)

            df_pair_id_acc = pd.concat([pd.read_csv(os.path.join(results_path, f"{n}_pair_id_acc.csv"), index_col=0).rename(columns={"id_acc":f"id_acc_{n}"}) for n in ["vdvae", "cliptext", "clipvision"] ], axis=1)
            df_pair_id_acc = df_pair_id_acc / (49 if name == "test" else 39)
            df_pattern_correlation = pd.concat([df_pattern_correlation, df_pair_id_acc], axis=1)

            df_pattern_correlation['im'] = df_pattern_correlation.index + 1 
                    
            df_pattern_correlation = df_pattern_correlation.assign(**meta_information)
            dfs_bd_pattern_correlation.append(df_pattern_correlation)

            # Profile Correlation
            df_profile_correlation = pd.concat([pd.read_csv(os.path.join(results_path, f"{n}_profile_corr.csv"), index_col=0) for n in ["vdvae", "cliptext", "clipvision"] ])
            df_profile_correlation = df_profile_correlation.reset_index(names='ft_name')

            df_profile_correlation = df_profile_correlation.assign(**meta_information)
            dfs_bd_profile_correlation.append(df_profile_correlation)

            # Reconstruction Results
            path_folder_csv_recon_csv = os.path.join(out_path_base, 'results', 'quantitative')
            fname_csv_recon = f'res_quantitative_{dataset}_subj{sub}_{oname}.csv'
            path_csv_recon = os.path.join(path_folder_csv_recon_csv, fname_csv_recon)

            df_recon = pd.read_csv(path_csv_recon)
            df_recon = df_recon[df_recon["sub"] == sub]
            df_recon['im'] = pd.to_numeric(df_recon['im'])

            df_recon = df_recon.assign(**meta_information)
            dfs_bd_reconstrucion.append(df_recon)

data_out_path = os.path.join(out_path_base, 'analysis/thesis_analysis/data')


df_bd_pattern_correlation = pd.concat(dfs_bd_pattern_correlation)
df_bd_profile_correlation = pd.concat(dfs_bd_profile_correlation)
df_bd_reconstruction = pd.concat(dfs_bd_reconstrucion)

df_bd_pattern_correlation.to_csv(os.path.join(data_out_path, "df_pattern_correlation_aicap.csv"))
df_bd_profile_correlation.to_csv(os.path.join(data_out_path, "df_profile_correlation_aicap.csv"))
df_bd_reconstruction.to_csv(os.path.join(data_out_path, "df_reconstruction_aicap.csv"))
