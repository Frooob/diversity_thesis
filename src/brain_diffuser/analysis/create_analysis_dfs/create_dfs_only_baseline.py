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
]

icnn_conditions = [
    'icnn:_size224_iter500_scaled',
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
            meta_information = {
                "dataset": dataset,
                "sub": sub,
                "test_dataset": name,
                "algorithm":"brain_diffuser"}

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
    print("icnn", sub)
    for condition in icnn_conditions:
        for name in ['test', 'art']:
            condition_name = condition[5:]
            oname = "_".join((name, condition_name))
            results_path = os.path.join(
                out_path_base, 'results', 'icnn', dataset, f"subj{sub}", oname)

            condition_list = condition_name.split("_")

            meta_information = {
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
merge_cols = ['dataset', 'sub', 'test_dataset', 'im', ]
df_pattern_correlation = pd.merge(df_bd_pattern_correlation, df_icnn_pattern_correlation, 'outer', merge_cols, suffixes=("_bd", "_icnn"))
df_pattern_correlation.to_csv(os.path.join(data_out_path,'df_pattern_correlation_baseline.csv'), index=False)

# Profile Correlation
df_profile_correlation = pd.concat([df_bd_profile_correlation, df_icnn_profile_correlation])
df_profile_correlation.to_csv(os.path.join(data_out_path,"df_profile_correlation_baseline.csv"), index=False)

# Reconstruction
df_reconstruction = pd.merge(df_bd_reconstruction, df_icnn_reconstruction,'outer', merge_cols, suffixes=("_bd", "_icnn"))
df_reconstruction.to_csv(os.path.join(data_out_path,"df_reconstruction_baseline.csv"), index=False)


