import pandas as pd
import os
import seaborn as sns
import sys
from collections import defaultdict
sys.path.append('/home/matt/programming/recon_diffuser/src/recon_diffuser/adv_pert')
sys.path.append('/home/matt/programming/recon_diffuser/src/recon_diffuser/evaluation')
from visualize_results import visualize_results

from evaluate_pertgen_log import extract_pert_gen_info, extract_pertgen_all_from_folder

from extract_regression_scores_from_logs import extract_all_logs_from_directory


out_path_base = '/home/matt/programming/recon_diffuser/'

subjects = ['AM']
dataset = 'deeprecon'

bd_conditions = [
    '', # baseline
    'dropout-random_0.1_00',
    'dropout-random_0.25_00',
    'dropout-random_0.5_00',
    'dropout-random_0.75_00',
    'dropout-random_0.1_01',
    'dropout-random_0.25_01',
    'dropout-random_0.5_01',
    'dropout-random_0.75_01',
    'dropout-random_0.1_02',
    'dropout-random_0.25_02',
    'dropout-random_0.5_02',
    'dropout-random_0.75_02',
    'dropout-random_0.1_03',
    'dropout-random_0.25_03',
    'dropout-random_0.5_03',
    'dropout-random_0.75_03',
    'dropout-random_0.1_04',
    'dropout-random_0.25_04',
    'dropout-random_0.5_04',
    'dropout-random_0.75_04',
    # dreamsim dropout
    'dropout-dreamsim_0.1_00',
    'dropout-dreamsim_0.25_00',
    'dropout-dreamsim_0.5_00',
    'dropout-dreamsim_0.1_01',
    'dropout-dreamsim_0.25_01',
    'dropout-dreamsim_0.5_01',
    'dropout-dreamsim_0.1_02',
    'dropout-dreamsim_0.25_02',
    'dropout-dreamsim_0.5_02',
    # pixels dropout
    'dropout-pixels_0.1_00',
    'dropout-pixels_0.25_00',
    'dropout-pixels_0.5_00',
    'dropout-pixels_0.1_01',
    'dropout-pixels_0.25_01',
    'dropout-pixels_0.5_01',
    'dropout-pixels_0.1_02',
    'dropout-pixels_0.25_02',
    'dropout-pixels_0.5_02',
    # clipvision dropout
    'dropout-clipvision_0.1_00',
    'dropout-clipvision_0.25_00',
    'dropout-clipvision_0.5_00',
    'dropout-clipvision_0.1_01',
    'dropout-clipvision_0.25_01',
    'dropout-clipvision_0.5_01',
    'dropout-clipvision_0.1_02',
    'dropout-clipvision_0.25_02',
    'dropout-clipvision_0.5_02',
]

icnn_conditions = [
    'icnn:_size224_iter500_scaled',
    'icnn:dropout-random_0.1_00_size224_iter500_scaled',
    'icnn:dropout-random_0.25_00_size224_iter500_scaled',
    'icnn:dropout-random_0.5_00_size224_iter500_scaled',
    'icnn:dropout-random_0.75_00_size224_iter500_scaled',
    'icnn:dropout-random_0.1_01_size224_iter500_scaled',
    'icnn:dropout-random_0.25_01_size224_iter500_scaled',
    'icnn:dropout-random_0.5_01_size224_iter500_scaled',
    'icnn:dropout-random_0.75_01_size224_iter500_scaled',
    'icnn:dropout-random_0.1_02_size224_iter500_scaled',
    'icnn:dropout-random_0.25_02_size224_iter500_scaled',
    'icnn:dropout-random_0.5_02_size224_iter500_scaled',
    'icnn:dropout-random_0.75_02_size224_iter500_scaled',
    'icnn:dropout-random_0.1_03_size224_iter500_scaled',
    'icnn:dropout-random_0.25_03_size224_iter500_scaled',
    'icnn:dropout-random_0.5_03_size224_iter500_scaled',
    'icnn:dropout-random_0.75_03_size224_iter500_scaled',
    'icnn:dropout-random_0.1_04_size224_iter500_scaled',
    'icnn:dropout-random_0.25_04_size224_iter500_scaled',
    'icnn:dropout-random_0.5_04_size224_iter500_scaled',
    'icnn:dropout-random_0.75_04_size224_iter500_scaled',
    # dreamsim dropout
    'icnn:dropout-dreamsim_0.1_00_size224_iter500_scaled',
    'icnn:dropout-dreamsim_0.25_00_size224_iter500_scaled',
    'icnn:dropout-dreamsim_0.5_00_size224_iter500_scaled',
    'icnn:dropout-dreamsim_0.1_01_size224_iter500_scaled',
    'icnn:dropout-dreamsim_0.25_01_size224_iter500_scaled',
    'icnn:dropout-dreamsim_0.5_01_size224_iter500_scaled',
    'icnn:dropout-dreamsim_0.1_02_size224_iter500_scaled',
    'icnn:dropout-dreamsim_0.25_02_size224_iter500_scaled',
    'icnn:dropout-dreamsim_0.5_02_size224_iter500_scaled',
    # pixels dropout
    'icnn:dropout-pixels_0.1_00_size224_iter500_scaled',
    'icnn:dropout-pixels_0.25_00_size224_iter500_scaled',
    'icnn:dropout-pixels_0.5_00_size224_iter500_scaled',
    'icnn:dropout-pixels_0.1_01_size224_iter500_scaled',
    'icnn:dropout-pixels_0.25_01_size224_iter500_scaled',
    'icnn:dropout-pixels_0.5_01_size224_iter500_scaled',
    'icnn:dropout-pixels_0.1_02_size224_iter500_scaled',
    'icnn:dropout-pixels_0.25_02_size224_iter500_scaled',
    'icnn:dropout-pixels_0.5_02_size224_iter500_scaled',
    # clipvision dropout
    'icnn:dropout-clipvision_0.1_00_size224_iter500_scaled',
    'icnn:dropout-clipvision_0.25_00_size224_iter500_scaled',
    'icnn:dropout-clipvision_0.5_00_size224_iter500_scaled',
    'icnn:dropout-clipvision_0.1_01_size224_iter500_scaled',
    'icnn:dropout-clipvision_0.25_01_size224_iter500_scaled',
    'icnn:dropout-clipvision_0.5_01_size224_iter500_scaled',
    'icnn:dropout-clipvision_0.1_02_size224_iter500_scaled',
    'icnn:dropout-clipvision_0.25_02_size224_iter500_scaled',
    'icnn:dropout-clipvision_0.5_02_size224_iter500_scaled',
]

sub = subjects[0]

dfs_bd_pattern_correlation = []
dfs_bd_profile_correlation = []
dfs_bd_reconstrucion = []
# Fetch both the regression results and the reconstruction results.
for condition in bd_conditions:
    for name in ['test', 'art']:
        oname = name if condition == '' else "_".join((name, condition))
        results_path = os.path.join(
            out_path_base, 'results', 'versatile_diffusion', dataset, f"subj{sub}", oname)
        
        # Get metadata
        try:
            dropout_algorithm, dropout_ratio, dropout_trial = condition.split("_")
        except:
            dropout_algorithm, dropout_ratio, dropout_trial = 'no-dropout', 1, 0

        meta_information = {
            "dropout_algorithm": dropout_algorithm,
            "dropout_ratio": float(dropout_ratio),
            "dropout_trial": int(dropout_trial),
            "dataset": dataset,
            "sub": sub,
            "test_dataset": name,
            "algorithm":"brain_diffuser"}

        # Pattern Correlation
        df_pattern_correlation = pd.concat([pd.read_csv(os.path.join(results_path, f"{n}_pattern_corr.csv"), index_col=0) for n in ["vdvae", "cliptext", "clipvision"] ], axis=1)
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
        fname_csv_recon = f'res_quantitative_{dataset}_{oname}.csv'
        path_csv_recon = os.path.join(path_folder_csv_recon_csv, fname_csv_recon)

        df_recon = pd.read_csv(path_csv_recon)
        df_recon = df_recon[df_recon["sub"] == sub]
        df_recon['im'] = pd.to_numeric(df_recon['im'])

        df_recon = df_recon.assign(**meta_information)
        dfs_bd_reconstrucion.append(df_recon)

df_bd_pattern_correlation = pd.concat(dfs_bd_pattern_correlation)
df_bd_profile_correlation = pd.concat(dfs_bd_profile_correlation)
df_bd_reconstruction = pd.concat(dfs_bd_reconstrucion)

dfs_icnn_pattern_correlation = []
dfs_icnn_profile_correlation = []
dfs_icnn_reconstrucion = []
for condition in icnn_conditions:
    for name in ['test', 'art']:
        condition_name = condition[5:]
        oname = "_".join((name, condition_name))
        results_path = os.path.join(
            out_path_base, 'results', 'icnn', dataset, f"subj{sub}", oname)

        condition_list = condition_name.split("_")
        if condition_list[0] == 'dropout-random' or condition_list[0] == 'dropout-dreamsim' or condition_list[0] == 'dropout-pixels' or condition_list[0] == 'dropout-clipvision':
            dropout_algorithm = condition_list[0]
            dropout_ratio = float(condition_list[1])
            dropout_trial = int(condition_list[2])
        elif condition_list[0] == "":
            dropout_algorithm = "no-dropout"
            dropout_ratio = 1
            dropout_trial = 0
        else:
            raise NotImplementedError(f"Unknown dropout algorithm {condition_list[0]}")

        if 'scaled' in condition_list:
            scaled = True
        else:
            scaled = False

        meta_information = {
            "dropout_algorithm": dropout_algorithm,
            "dropout_ratio": dropout_ratio,
            "dropout_trial": dropout_trial,
            "dataset": dataset,
            "sub": sub,
            "test_dataset": name,
            "algorithm": "icnn",
            "scaled": scaled}


        # Pattern Correlation
        df_pattern_correlation = pd.read_csv(os.path.join(results_path, 'icnn_pattern_corr.csv'), index_col=0)

        df_pattern_correlation['im'] = df_pattern_correlation.index + 1 
                
        df_pattern_correlation = df_pattern_correlation.assign(**meta_information)
        dfs_icnn_pattern_correlation.append(df_pattern_correlation)

        # Profile Correlation
        df_profile_correlation = pd.read_csv(os.path.join(results_path, f"icnn_profile_corr.csv"), index_col=0) 
        df_profile_correlation = df_profile_correlation.reset_index(names='ft_name')

        df_profile_correlation = df_profile_correlation.assign(**meta_information)
        dfs_icnn_profile_correlation.append(df_profile_correlation)

        # Reconstruction Results
        path_folder_csv_recon_csv = os.path.join(out_path_base, 'results', 'quantitative')
        fname_csv_recon = f'res_quantitative_icnn_{dataset}_{oname}.csv'
        path_csv_recon = os.path.join(path_folder_csv_recon_csv, fname_csv_recon)

        df_recon = pd.read_csv(path_csv_recon)
        df_recon = df_recon[df_recon["sub"] == sub]
        df_recon['im'] = pd.to_numeric(df_recon['im'])

        df_recon = df_recon.assign(**meta_information)
        dfs_icnn_reconstrucion.append(df_recon)


df_icnn_pattern_correlation = pd.concat(dfs_icnn_pattern_correlation)
df_icnn_profile_correlation = pd.concat(dfs_icnn_profile_correlation)
df_icnn_reconstruction = pd.concat(dfs_icnn_reconstrucion)


# Merge bd and icnn dataframes

# pattern Correlation
merge_cols = ['dataset', 'sub', 'test_dataset', 'im', 'dropout_algorithm', 'dropout_ratio', 'dropout_trial']
df_pattern_correlation = pd.merge(df_bd_pattern_correlation, df_icnn_pattern_correlation, 'outer', merge_cols, suffixes=("_bd", "_icnn"))
df_pattern_correlation.to_csv('df_pattern_correlation.csv', index=False)

# Profile Correlation
df_profile_correlation = pd.concat([df_bd_profile_correlation, df_icnn_profile_correlation])
df_profile_correlation.to_csv("df_profile_correlation.csv")

# Reconstruction
df_reconstruction = pd.merge(df_bd_reconstruction, df_icnn_reconstruction,'outer', merge_cols, suffixes=("_bd", "_icnn"))
df_reconstruction.to_csv("df_reconstruction.csv")


# sns.lineplot(data = df[['dropout_ratio','dreamsim_icnn']].dropna(), x='dropout_ratio', y='dreamsim_icnn')
# sns.lineplot(data = df[['dropout_ratio','dreamsim_bd']].dropna(), x='dropout_ratio', y='dreamsim_bd')
# sns.lineplot(data = df_icnn, x='dropout_ratio', y='dreamsim')
