import numpy as np
import pandas as pd
import seaborn as sns


def compute_output_name_for_aicap(df):
    output_name_cols = ['test_dataset', ]

    

def compute_output_name_for_dropout(df):
    output_name_cols = ['test_dataset', 'dropout_algorithm', 'data_fraction', 'dropout_trial']
    return df[output_name_cols].astype(str).agg('_'.join, axis=1)

def competition_criterion(df, baseline_name, compare_names, criteria, additional_cols=()):
    collector = []

    for compare_name in compare_names:
        for criterion in criteria:
            for sub in df['sub'].unique():
                for test_dataset in ['test', 'art']:
                    baseline_name_ds = f"{test_dataset}_{baseline_name}" 
                    compare_name_ds = f"{test_dataset}_{compare_name}"
                    df_setting = df.query(f'sub == "{sub}"').query(f'test_dataset == "{test_dataset}"')
                    
                    df_baseline = df_setting[df_setting['name'] == baseline_name_ds]
                    df_compare = df_setting[df_setting['name'] == compare_name_ds]

                    competitions_won = (df_compare[criterion].values >= df_baseline[criterion].values).sum() 
                    competitions_draw = (df_compare[criterion].values == df_baseline[criterion].values).sum() 
                    if competitions_draw == len(df_baseline):
                        print("Whee, it's a draw (or is it possible that you're comparing the condition with itself??)")
                        competitions_ratio = 0.5
                    else:
                        competitions_ratio = competitions_won / (len(df_baseline) - competitions_draw)

                    setting_dict = {'condition':compare_name, 'criterion': criterion, 'sub': sub, 'test_dataset': test_dataset, 'won': competitions_ratio}
                    for col in additional_cols:
                        unique_values_in_col_for_name = df_compare[col].unique()
                        if len(unique_values_in_col_for_name) > 1:
                            raise Exception("This shouldn't be. They must have only one entry here.")
                        setting_dict[col] = unique_values_in_col_for_name[0]
                    collector.append(setting_dict)
    df_cc = pd.DataFrame(collector)
    return df_cc


# df_reconstruction = ...
# df_pattern_corr = ...

# df = df_reconstruction
# df = df_pattern_corr

# baseline_col = 'vdvae'

# test_dataset = 'test'

# baseline_name = f'dropout-random_0.25_33'
# compare_name = f'dropout-dreamsim_0.25_55'

# criteria = ['id_acc_vdvae', 'id_acc_cliptext', 'id_acc_clipvision', 'id_acc_icnn', ]
# criteria = ['vdvae', 'cliptext_final', 'clipvision_final','all_icnn'] 


# sns.pointplot(data=df_cc, y='won', x = 'criterion', hue='test_dataset')