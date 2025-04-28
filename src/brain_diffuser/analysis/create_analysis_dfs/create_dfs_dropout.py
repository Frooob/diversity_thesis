import pandas as pd
import os
import seaborn as sns
import sys
from collections import defaultdict

out_path_base = '/home/matt/programming/recon_diffuser/'
subjects = ['AM', 'KS', 'ES', 'JK', 'TH']
dataset = 'deeprecon'

bd_conditions = [
    '', # baseline
    'dropout-random_0.1_00', # randoms only executed for sub AM
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

    'dropout-dreamsim_0.1_11', # These are executed for all subjects
    'dropout-clipvision_0.1_33',
    'dropout-random_0.1_22',
    'dropout-pixels_0.1_44',

    'dropout-dreamsim_0.25_55',
    'dropout-clipvision_0.25_88',
    'dropout-random_0.25_33',
    'dropout-pixels_0.25_44',

    "dropout-quantizedCountBoring_0.25_00",
    "dropout-quantizedCountParty_0.25_00",

]

icnn_conditions = [
    'icnn:_size224_iter500_scaled', # baseline
    'icnn:dropout-random_0.1_00_size224_iter500_scaled', # randoms only executed for sub AM
    'icnn:dropout-random_0.1_01_size224_iter500_scaled',
    'icnn:dropout-random_0.1_02_size224_iter500_scaled',
    'icnn:dropout-random_0.1_03_size224_iter500_scaled',
    'icnn:dropout-random_0.1_04_size224_iter500_scaled',
    'icnn:dropout-random_0.25_00_size224_iter500_scaled',
    'icnn:dropout-random_0.25_01_size224_iter500_scaled',
    'icnn:dropout-random_0.25_02_size224_iter500_scaled',
    'icnn:dropout-random_0.25_03_size224_iter500_scaled',
    'icnn:dropout-random_0.25_04_size224_iter500_scaled',
    'icnn:dropout-random_0.5_00_size224_iter500_scaled',
    'icnn:dropout-random_0.5_01_size224_iter500_scaled',
    'icnn:dropout-random_0.5_02_size224_iter500_scaled',
    'icnn:dropout-random_0.5_03_size224_iter500_scaled',
    'icnn:dropout-random_0.5_04_size224_iter500_scaled',
    'icnn:dropout-random_0.75_00_size224_iter500_scaled',
    'icnn:dropout-random_0.75_01_size224_iter500_scaled',
    'icnn:dropout-random_0.75_02_size224_iter500_scaled',
    'icnn:dropout-random_0.75_03_size224_iter500_scaled',
    'icnn:dropout-random_0.75_04_size224_iter500_scaled',

    'icnn:dropout-dreamsim_0.1_11_size224_iter500_scaled',  # These are executed for all subjects
    'icnn:dropout-clipvision_0.1_33_size224_iter500_scaled',
    'icnn:dropout-random_0.1_22_size224_iter500_scaled',
    'icnn:dropout-pixels_0.1_44_size224_iter500_scaled',

    'icnn:dropout-dreamsim_0.25_55_size224_iter500_scaled',
    'icnn:dropout-clipvision_0.25_88_size224_iter500_scaled',
    'icnn:dropout-random_0.25_33_size224_iter500_scaled',
    'icnn:dropout-pixels_0.25_44_size224_iter500_scaled',

    "icnn:dropout-quantizedCountBoring_0.25_00_size224_iter500_scaled",
    "icnn:dropout-quantizedCountParty_0.25_00_size224_iter500_scaled",

]


dfs_bd_pattern_correlation = []
dfs_bd_profile_correlation = []
dfs_bd_reconstrucion = []
# Fetch both the regression results and the reconstruction results.
for sub in subjects:
    print(f"sub{sub} bd ")
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
            
            if dropout_algorithm == 'dropout-random' and int(str(dropout_trial)) not in [22,33]  and sub != "AM":
                # print("onwards")
                continue
                

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

dfs_icnn_pattern_correlation = []
dfs_icnn_profile_correlation = []
dfs_icnn_reconstrucion = []
for sub in subjects:
    print(f"icnn sub {sub}")
    for condition in icnn_conditions:
        for name in ['test', 'art']:
            condition_name = condition[5:]
            oname = "_".join((name, condition_name))
            results_path = os.path.join(
                out_path_base, 'results', 'icnn', dataset, f"subj{sub}", oname)
            condition_list = condition_name.split("_")
            if condition_list[0] == 'dropout-random' or condition_list[0] == 'dropout-dreamsim' or condition_list[0] == 'dropout-pixels' or condition_list[0] == 'dropout-clipvision' or  condition_list[0] == 'dropout-quantizedCountBoring' or  condition_list[0] == 'dropout-quantizedCountParty':
                dropout_algorithm = condition_list[0]
                dropout_ratio = float(condition_list[1])
                dropout_trial = int(condition_list[2])
            elif condition_list[0] == "":
                dropout_algorithm = "no-dropout"
                dropout_ratio = 1
                dropout_trial = 0
            else:
                raise NotImplementedError(f"Unknown dropout algorithm {condition_list[0]}")

            if dropout_algorithm == 'dropout-random' and int(str(dropout_trial)) not in [22,33]  and sub != "AM":
                # print("onwards")
                continue
            
            meta_information = {
                "dropout_algorithm": dropout_algorithm,
                "dropout_ratio": dropout_ratio,
                "dropout_trial": dropout_trial,
                "dataset": dataset,
                "sub": sub,
                "test_dataset": name,
                "algorithm": "icnn"}


            # Pattern Correlation
            df_pattern_correlation = pd.read_csv(os.path.join(results_path, 'icnn_pattern_corr.csv'), index_col=0)
            
            df_pair_id_acc =pd.read_csv(os.path.join(results_path, f"icnn_pair_id_acc.csv"), index_col=0) 
            df_pair_id_acc = df_pair_id_acc / (50 if name == "test" else 40)
            df_pattern_correlation = pd.concat([df_pattern_correlation, df_pair_id_acc], axis=1)

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
            fname_csv_recon = f'res_quantitative_icnn_{dataset}_subj{sub}_{oname}.csv'
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
data_out_path = os.path.join(out_path_base, 'analysis/thesis_analysis/data')

# pattern Correlation
merge_cols = ['dataset', 'sub', 'test_dataset', 'im', 'dropout_algorithm', 'dropout_ratio', 'dropout_trial']
df_pattern_correlation = pd.merge(df_bd_pattern_correlation, df_icnn_pattern_correlation, 'outer', merge_cols, suffixes=("_bd", "_icnn"))
df_pattern_correlation.to_csv(os.path.join(data_out_path,'df_pattern_correlation_dropout.csv'), index=False)

# Profile Correlation
df_profile_correlation = pd.concat([df_bd_profile_correlation, df_icnn_profile_correlation])
df_profile_correlation.to_csv(os.path.join(data_out_path,"df_profile_correlation_dropout.csv"), index=False)

# Reconstruction
df_reconstruction = pd.merge(df_bd_reconstruction, df_icnn_reconstruction,'outer', merge_cols, suffixes=("_bd", "_icnn"))
df_reconstruction.to_csv(os.path.join(data_out_path,"df_reconstruction_dropout.csv"), index=False)
