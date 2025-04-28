import pandas as pd
import os
import seaborn as sns
import sys
from collections import defaultdict
sys.path.append('/home/matt/programming/recon_diffuser/src/recon_diffuser/adv_pert')
sys.path.append('/home/matt/programming/recon_diffuser/src/recon_diffuser/evaluation')
sys.path.append('/home/matt/programming/recon_diffuser/src/recon_diffuser')
from re_ai_captions_utils import get_prompt_name_and_params_from_output_name
from visualize_results import visualize_results


out_path_base = '/home/matt/programming/recon_diffuser/'

subjects = ['AM']
dataset = 'deeprecon'

bd_conditions = [
    "aicap_human_captions", # baseline, within the aicap experiment (but equivalent to 'real' baseline), makes it easier to vary this. 
    "aicap_low_level_short",
    "aicap_low_level_long",    
    "aicap_high_level_short",
    "aicap_high_level_long",
    "aicap_human_captions_shuffled",
    "aicap_human_captions_shuffled_single_caption",

    'aicap_human_captions-mix_0.0',
    
    'aicap_human_captions-mix_0.7',
    'aicap_low_level_short-mix_0.7',
    'aicap_low_level_long-mix_0.7',
    'aicap_high_level_short-mix_0.7',
    'aicap_high_level_long-mix_0.7',

    'aicap_human_captions-mix_0.95',
    'aicap_low_level_short-mix_0.95',
    'aicap_low_level_long-mix_0.95',
    'aicap_high_level_short-mix_0.95',
    'aicap_high_level_long-mix_0.95',

    "aicap_human_captions_shuffled_single_caption-mix_0.0",
    "aicap_human_captions_shuffled_single_caption-mix_0.7",
    "aicap_human_captions_shuffled_single_caption-mix_0.95",

    "aicap_human_captions-mix_0.0-badshuffle_True",
    "aicap_human_captions-mix_0.4-badshuffle_True",
    "aicap_human_captions-mix_0.7-badshuffle_True",
    "aicap_human_captions-mix_0.95-badshuffle_True",

    "aicap_low_level_short-mix_0.95",
    "aicap_low_level_long-mix_0.95",    
    "aicap_high_level_short-mix_0.95",
    "aicap_high_level_long-mix_0.95",

    "aicap_low_level_short-mix_0.8",
    "aicap_low_level_long-mix_0.8",    
    "aicap_high_level_short-mix_0.8",
    "aicap_high_level_long-mix_0.8",

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
        if "aicap" in condition:
            prompt_name, params = get_prompt_name_and_params_from_output_name(condition)
        else:
            prompt_name, params = "NO", {}


        meta_information = {
            "prompt_name": prompt_name,
            "mixing": params.get('mix', 0.4),
            "badshuffle": params.get('badshuffle', False),
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

# pattern Correlation
df_bd_pattern_correlation.to_csv('/home/matt/programming/recon_diffuser/analysis/progress_presentation_3/df_pattern_correlation.csv', index=False)

# Profile Correlation
df_bd_profile_correlation.to_csv("/home/matt/programming/recon_diffuser/analysis/progress_presentation_3/df_profile_correlation.csv")

# Reconstruction
df_bd_reconstruction.to_csv("/home/matt/programming/recon_diffuser/analysis/progress_presentation_3/df_reconstruction.csv")

