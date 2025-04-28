# Alle dfs sind schon da. 

# Hier müssen sowohl die perturbation logs als auch die reconstruction, regression results drin sein.

# Bei den profile/patterns gibt es errors.
# Wahrscheinlich weil vdvae und cliptext hier gar nicht berechnet wurden

import pandas as pd
import os
import seaborn as sns
import sys
from collections import defaultdict
sys.path.append('/home/matt/programming/recon_diffuser/src/recon_diffuser/adv_pert')
sys.path.append('/home/matt/programming/recon_diffuser/src/recon_diffuser/evaluation')

from evaluate_pertgen_log import extract_pertgen_all_from_folder


out_path_base = '/home/matt/programming/recon_diffuser/'

subjects = ['AM', 'KS', 'ES', 'JK', 'TH', ]
dataset = 'deeprecon'

# TODO: Add identification accuracy!
bd_conditions = [
    '', # baseline
    'fgsm_friendly_0_5', # defacto baseline
    'ic_adversarial_90-10_500_5',
    'ic_friendly_90-10_500_5',

    'ic_adversarial_80-20_500_5',
    'ic_friendly_80-20_500_5',

    'ic_adversarial_70-30_500_5',
    'ic_friendly_70-30_500_5',

    'ic_adversarial_50-50_500_5',
    'ic_friendly_50-50_500_5',

    'fgsm_adversarial_0.03_5',
    'fgsm_friendly_0.03_5',
    # 'fgsm_adversarial_0.1_5',
    # 'fgsm_friendly_0.1_5',
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
            
            """
            Extract the following meta information:
            algorithm (ic/fgsm)
                - epsilon (for fgsm)
                - criterion
            n_perts (5 usually)

            """
            if condition == "":
                algorithm_meta_information = {
                    "pert_algorithm": "NO_PERT"
                }
            else:
                algorithm_meta_information = {}
                adv_pert_info = condition.split("_")
                pert_algorithm = adv_pert_info[0]
                algorithm_meta_information['pert_algorithm'] = pert_algorithm
                algorithm_meta_information['pert_type'] = adv_pert_info[1]
                
                if pert_algorithm == "fgsm":
                    algorithm_meta_information['pert_epsilon'] = adv_pert_info[2]
                    algorithm_meta_information['pert_n'] = adv_pert_info[3]
                elif pert_algorithm == "ic":
                    algorithm_meta_information['pert_ratio'] = adv_pert_info[2]
                    algorithm_meta_information['pert_max_steps'] = adv_pert_info[3]
                    algorithm_meta_information['pert_n'] = adv_pert_info[4]
                else:
                    raise ValueError("Unknown pert algorithm {pert_algorithm}")

            
            meta_information = {
                "dataset": dataset,
                "sub": sub,
                "test_dataset": name,
                "algorithm":"brain_diffuser"}

            meta_information = {**meta_information, **algorithm_meta_information}

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

df_bd_pattern_correlation = pd.concat(dfs_bd_pattern_correlation)
df_bd_profile_correlation = pd.concat(dfs_bd_profile_correlation)
df_bd_reconstruction = pd.concat(dfs_bd_reconstrucion)
data_out_path = os.path.join(out_path_base, 'analysis/thesis_analysis/data')

df_bd_pattern_correlation.to_csv(os.path.join(data_out_path, "df_pattern_correlation_advpert.csv"))
df_bd_profile_correlation.to_csv(os.path.join(data_out_path, "df_profile_correlation_advpert.csv"))
df_bd_reconstruction.to_csv(os.path.join(data_out_path, "df_reconstruction_advpert.csv"))
