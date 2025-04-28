print("let's go")
import os

import pandas as pd
import numpy as np

from re_evaluate_reconstruction import eval_subject
from re_eval_extract_features import eval_extract_features
from eval_utils import simple_log
from feature_pred_scores import compute_profile_pattern_main
print("All libs loaded")

def eval_single_subject(out_path_base, dataset_name, name, sub, result_name="", aggregate_results=True, algorithm = 'bd'):
    if not result_name:
        result_name = name
    supported_datasets = ['nsd', 'deeprecon', 'things']

    if dataset_name not in supported_datasets:
        raise NotImplementedError(f"Datset {dataset_name} can't be evaluated yet")
    
    eval_data = []
    if sub == 0:
        return # I guess? I'm sorry I don't remember the code from ages ago. 
    
    # for sub in subs:
    #     if sub == 0:
    #         continue
    #     sub_metrics = eval_subject(out_path_base, dataset_name, name, sub, result_name, aggregate_results, algorithm=algorithm)
    #     if aggregate_results:
    #         eval_data.append(sub_metrics)
    #     else:
    #         eval_data+=sub_metrics

    sub_metrics = eval_subject(out_path_base, dataset_name, name, sub, result_name, aggregate_results, algorithm=algorithm)
    eval_data += sub_metrics

    df_eval = pd.DataFrame(eval_data)
    result_dir = os.path.join(out_path_base, f'results/quantitative/')
    os.makedirs(result_dir, exist_ok=True)
    if algorithm == 'bd':
        csv_file_name = f'res_quantitative_{dataset_name}_subj{sub}_{result_name}'
    elif algorithm == 'icnn':
        csv_file_name = f'res_quantitative_icnn_{dataset_name}_subj{sub}_{result_name}'
    else:
        raise NotImplementedError(f"Unknown algorithm {algorithm}.")
    if aggregate_results:
        csv_file_name+='_aggregated'
    csv_file = os.path.join(result_dir, csv_file_name+'.csv')
    df_eval.to_csv(csv_file, index=False)
    simple_log(f'Saved final quantitative metrics output csv to {csv_file}')
    return df_eval


def formatted_results(df, dataset, name):
    order = ['PixCorr', 'SSIM', 'alexnet_2', 'alexnet_5', 'inceptionv3_avgpool', 'clip_final','efficientnet_avgpool', 'swav_avgpool']
    m = df.mean(numeric_only=True)[order]
    col_string = "\t".join(order)
    num_string = "\t".join([str(np.round(x,3)) for x in m])
    print(f"Quant results for {dataset} {name}")
    print(col_string)
    print(num_string)



if __name__ == "__main__":
    opb = "/home/matt/programming/recon_diffuser/"
    out_path_base = opb

    # #### BASE EXPERIMENT
    # ## NSD
    # # First: Extract all features from the reconstructed train images (and the ground truth)
    # nsd_subs = [0, "1", "2", "5", "7"]
    # for nsd_sub in nsd_subs:
    #     eval_extract_features(opb, 'nsd', 'test', nsd_sub)
    # # Second: Compute the quantitative metrics that are computed in eval_subject
    # df_nsd = eval_all_subjects(opb, 'nsd', 'test', nsd_subs)

    # ## deeprecon
    # deeprecon_subs = [0, 'KS', 'AM', 'ES', 'JK', 'TH']

    # for deeprecon_sub in deeprecon_subs:
    #     print(deeprecon_sub)
    #     # eval_extract_features(opb, 'deeprecon', 'test', deeprecon_sub)
    #     # eval_extract_features(opb, 'deeprecon', 'art', deeprecon_sub)

    # df_deeprecon_test = eval_all_subjects(opb, 'deeprecon', 'test', deeprecon_subs, aggregate_results=False)
    # df_deeprecon_art = eval_all_subjects(opb, 'deeprecon', 'art', deeprecon_subs, aggregate_results=False)
    # formatted_results(df_deeprecon_art, "deeprecon", "art")

    # ## things
    # things_subs = [0, "01", "02", "03"]

    # for things_sub in things_subs:
    #     print(things_sub)
    #     eval_extract_features(opb, 'things', 'test', things_sub)
    # things_subs = [0, "01", "02", "03"]
    # df_things = eval_all_subjects(opb, "things", "test", things_subs)

    # formatted_results(df_things, "things", "test")
    #### PERTURBATION EXPERIMENT
    # The ground truth image features need to be extracted beforehand
    deeprecon_pert_subs = ['AM','KS', 'ES', 'JK', 'TH']
    # deeprecon_pert_subs = ['AM']

    output_names = [
        # ## Baseline
        # "",
        # 'icnn:_size224_iter500_scaled',

        # ## AdvPert
        # 'fgsm_friendly_0_5', # de facto baseline
        # 'ic_adversarial_90-10_500_5',
        # 'ic_friendly_90-10_500_5',
        # 'ic_adversarial_80-20_500_5',
        # 'ic_friendly_80-20_500_5',
        # "ic_adversarial_70-30_500_5",
        # "ic_friendly_70-30_500_5",
        # 'ic_adversarial_50-50_500_5',
        # 'ic_friendly_50-50_500_5',

        # 'fgsm_adversarial_0.03_5',
        # 'fgsm_friendly_0.03_5',
        # 'fgsm_adversarial_0.1_5',
        # 'fgsm_friendly_0.1_5',

        # ## dropout
        # 'dropout-random_0.1_00',
        # 'dropout-random_0.1_01',
        # 'dropout-random_0.1_02',
        # 'dropout-random_0.1_03',
        # 'dropout-random_0.1_04',
        # 'dropout-random_0.25_00',
        # 'dropout-random_0.25_01',
        # 'dropout-random_0.25_02',
        # 'dropout-random_0.25_03',
        # 'dropout-random_0.25_04',
        # 'dropout-random_0.5_00',
        # 'dropout-random_0.5_01',
        # 'dropout-random_0.5_02',
        # 'dropout-random_0.5_03',
        # 'dropout-random_0.5_04',
        # 'dropout-random_0.75_00',
        # 'dropout-random_0.75_01',
        # 'dropout-random_0.75_02',
        # 'dropout-random_0.75_03',
        # 'dropout-random_0.75_04',

        # 'dropout-dreamsim_0.1_11',
        # 'dropout-clipvision_0.1_33',
        # 'dropout-random_0.1_22',
        # 'dropout-pixels_0.1_44',

        # 'dropout-dreamsim_0.25_55',
        # 'dropout-clipvision_0.25_88',
        # 'dropout-random_0.25_33',
        # 'dropout-pixels_0.25_44',

        # "dropout-quantizedCountBoring_0.25_00",
        # "dropout-quantizedCountParty_0.25_00",

        # 'icnn:dropout-random_0.1_00_size224_iter500_scaled',
        # 'icnn:dropout-random_0.1_01_size224_iter500_scaled',
        # 'icnn:dropout-random_0.1_02_size224_iter500_scaled',
        # 'icnn:dropout-random_0.1_03_size224_iter500_scaled',
        # 'icnn:dropout-random_0.1_04_size224_iter500_scaled',
        # 'icnn:dropout-random_0.25_00_size224_iter500_scaled',
        # 'icnn:dropout-random_0.25_01_size224_iter500_scaled',
        # 'icnn:dropout-random_0.25_02_size224_iter500_scaled',
        # 'icnn:dropout-random_0.25_03_size224_iter500_scaled',
        # 'icnn:dropout-random_0.25_04_size224_iter500_scaled',
        # 'icnn:dropout-random_0.5_00_size224_iter500_scaled',
        # 'icnn:dropout-random_0.5_01_size224_iter500_scaled',
        # 'icnn:dropout-random_0.5_02_size224_iter500_scaled',
        # 'icnn:dropout-random_0.5_03_size224_iter500_scaled',
        # 'icnn:dropout-random_0.5_04_size224_iter500_scaled',
        # 'icnn:dropout-random_0.75_00_size224_iter500_scaled',
        # 'icnn:dropout-random_0.75_01_size224_iter500_scaled',
        # 'icnn:dropout-random_0.75_02_size224_iter500_scaled',
        # 'icnn:dropout-random_0.75_03_size224_iter500_scaled',
        # 'icnn:dropout-random_0.75_04_size224_iter500_scaled',

        # 'icnn:dropout-dreamsim_0.1_11_size224_iter500_scaled',
        # 'icnn:dropout-clipvision_0.1_33_size224_iter500_scaled',
        # 'icnn:dropout-random_0.1_22_size224_iter500_scaled',
        # 'icnn:dropout-pixels_0.1_44_size224_iter500_scaled',

        # 'icnn:dropout-dreamsim_0.25_55_size224_iter500_scaled',
        # 'icnn:dropout-clipvision_0.25_88_size224_iter500_scaled',
        # 'icnn:dropout-random_0.25_33_size224_iter500_scaled',
        # 'icnn:dropout-pixels_0.25_44_size224_iter500_scaled',

        "icnn:dropout-quantizedCountBoring_0.25_00_size224_iter500_scaled",
        # "icnn:dropout-quantizedCountParty_0.25_00_size224_iter500_scaled",

        # # Discussion Experiments only for sub AM
        # 'aicap_human_captions-mix_0.0',
        # 'aicap_human_captions-mix_0.25',
        # 'aicap_human_captions-mix_0.5',
        # 'aicap_human_captions-mix_0.75',
        # 'aicap_human_captions-mix_0.99',
        # 'aicap_human_captions-usetruecliptext_True-mix_0.0',
        # 'aicap_human_captions-usetruecliptext_True-mix_0.25',
        # 'aicap_human_captions-usetruecliptext_True-mix_0.5',
        # 'aicap_human_captions-usetruecliptext_True-mix_0.75',
        # 'aicap_human_captions-usetruecliptext_True-mix_0.99',
        # 'aicap_human_captions_shuffled_single_caption-mix_0.0',
        # 'aicap_human_captions_shuffled_single_caption-mix_0.25',
        # 'aicap_human_captions_shuffled_single_caption-mix_0.5',
        # 'aicap_human_captions_shuffled_single_caption-mix_0.75',
        # 'aicap_human_captions_shuffled_single_caption-mix_0.99',

        # # # All subjects experiment
        # "aicap_human_captions", # Baseline mix 40
        # "aicap_human_captions_shuffled_single_caption", 
        # "aicap_low_level_short",
        # "aicap_low_level_long",    
        # "aicap_high_level_short",
        # "aicap_high_level_long",

        # "aicap_human_captions-mix_0.8", # Baseline mix 80
        # "aicap_human_captions_shuffled_single_caption-mix_0.8", 
        # "aicap_low_level_short-mix_0.8",
        # "aicap_low_level_long-mix_0.8",    
        # "aicap_high_level_short-mix_0.8",
        # "aicap_high_level_long-mix_0.8",
 ]
        

    all_dfs = []
    
    for result_name in output_names:
        print(f"Doing quantitative results for confi {result_name}")
        for sub in deeprecon_pert_subs:
            if result_name.startswith("icnn:"):
                output_name = result_name[5:]
                algorithm = "icnn"
            else:
                output_name = result_name
                algorithm = 'bd'
            print(sub)
            sub = sub
            test_name = 'test_'+output_name if output_name else "test"
            art_name = 'art_'+output_name if output_name else "art"
            try:
                eval_extract_features(opb, 'deeprecon', test_name, sub, algorithm)
                eval_extract_features(opb, 'deeprecon', art_name, sub, algorithm)
                compute_profile_pattern_main(opb, "deeprecon", "test", sub, output_name, algorithm)
                compute_profile_pattern_main(opb, "deeprecon", "art", sub, output_name, algorithm)
                
                df_deeprecon_pert_test = eval_single_subject(opb, 'deeprecon', "test", sub, test_name, aggregate_results=False, algorithm=algorithm)
                df_deeprecon_pert_art = eval_single_subject(opb, 'deeprecon', "art", sub, art_name, aggregate_results=False, algorithm=algorithm)
            except Exception as e:
                raise Exception
                # print(f"Sub {sub} {result_name} didn't work because of {e}")
