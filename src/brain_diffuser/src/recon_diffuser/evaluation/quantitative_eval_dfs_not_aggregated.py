import os
import pandas as pd
from quantitative_eval import formatted_results
import numpy as np
import seaborn as sns
import re

quantitative_results_folder = "/home/matt/programming/recon_diffuser/results/quantitative"

all_csvs = [f for f in os.listdir(quantitative_results_folder) if f.endswith(".csv") and "aggregated" not in f]

all_dfs = [pd.read_csv(os.path.join(quantitative_results_folder, p)) for p in all_csvs]

df = pd.concat(all_dfs)

def extract_number(tensor_str):
    match = re.search(r'\d+\.\d+', tensor_str)
    return float(match.group()) if match else None

df['dreamsim'] = df['dreamsim'].apply(extract_number)

df_sub = df[df["sub"] == "AM"].copy()

dataset = "deeprecon"
name = ""

metrics = ['PixCorr', 'SSIM', 'alexnet_2', 'alexnet_5', 'inceptionv3_avgpool', 'clip_final','efficientnet_avgpool', 'swav_avgpool']
order = ["name"] + metrics


print(f"Quant results for {dataset}")
col_string = "\t".join(order)
print(col_string)

df_sub['test_dataset'] = df_sub['name'].apply(lambda x: "art" if "art" in x else "test")
df_sub['pert_type'] = df_sub['name'].apply(lambda x: "friendly" if "friendly" in x else "adversarial" if 'adversarial' in x else 'baseline')

# plots for art
df_art = df_sub[df_sub['test_dataset'] == 'art']
df_test = df_sub[df_sub['test_dataset'] == 'test']

# need to filter this more lol
fgsm_adversarial_names = [
    'fgsm_adversarial_0.01_5',
    'fgsm_adversarial_0.03_5',
    'fgsm_adversarial_0.05_5',
    'fgsm_adversarial_0.1_5'
]
fgsm_friendly_names = [
    'fgsm_friendly_0.01_5',
    'fgsm_friendly_0.03_5',
    'fgsm_friendly_0.05_5',
    'fgsm_friendly_0.1_5'
]
ic_adversarial_names = [
    'ic_adversarial_90-10_500_5',
    'ic_adversarial_80-20_500_5',
    'ic_adversarial_70-30_500_5',
]
ic_friendly_names = [
    'ic_friendly_90-10_500_5',
    'ic_friendly_80-20_500_5',
    'ic_friendly_70-30_500_5'
] 


test_dataset_name = 'art'
names = fgsm_adversarial_names
criterion = 'inceptionv3_avgpool'

def do_boxplot(df_sub, test_dataset_name, names, criterion):
    names = [test_dataset_name] + ["_".join((test_dataset_name, name)) for name in names]
    df_plot = df_sub[df_sub['test_dataset'] == test_dataset_name]
    df_plot = df_plot.loc[df_plot['name'].isin(names)]

    sns.boxplot(data=df_plot, x = criterion, y = 'name', order = names)

do_boxplot(df_sub, 'art', fgsm_adversarial_names, 'clip_final')


# 'names could be fgsm or ic'
def do_boxplot_with_ad_type_hue(df_sub, test_dataset_name, names, criterion):
    df_plot = df_sub.copy()
    cols_with_name = [col for col in df_plot['name'].unique() if names in col and test_dataset_name in col]
    cols_with_name = [test_dataset_name] + cols_with_name
    df_plot = df_plot.loc[df_plot['name'].isin(cols_with_name)]

    df_plot['truncname'] = df_plot['name'].str.replace('_adversarial', "").str.replace('_friendly', "")
    

    # relevant_names = [n for n in df_sub['name'].unique() if names in n and test_dataset_name in n]
    # relevant_names = [test_dataset_name] + relevant_names

    # names = [test_dataset_name] + ["_".join((test_dataset_name, name)) for name in names]
    df_plot = df_plot[df_plot['test_dataset'] == test_dataset_name]
    # 
    sns.boxplot(data=df_plot, x = criterion, y = 'truncname', hue = 'pert_type', order = np.sort(df_plot['truncname'].unique())).set_title('Quantitative_metrics Subject AM')



do_boxplot_with_ad_type_hue(df_sub, 'art', 'fgsm', 'dreamsim')
do_boxplot_with_ad_type_hue(df_sub, 'art', 'fgsm', 'inceptionv3_avgpool')
do_boxplot_with_ad_type_hue(df_sub, 'art', 'fgsm', 'PixCorr')

do_boxplot_with_ad_type_hue(df_sub, 'test', 'fgsm', 'dreamsim')
do_boxplot_with_ad_type_hue(df_sub, 'test', 'fgsm', 'inceptionv3_avgpool')
do_boxplot_with_ad_type_hue(df_sub, 'test', 'fgsm', 'PixCorr')

do_boxplot_with_ad_type_hue(df_sub, 'art', 'ic', 'dreamsim')
do_boxplot_with_ad_type_hue(df_sub, 'art', 'ic', 'inceptionv3_avgpool')
do_boxplot_with_ad_type_hue(df_sub, 'art', 'ic', 'PixCorr')

do_boxplot_with_ad_type_hue(df_sub, 'test', 'ic', 'dreamsim')
do_boxplot_with_ad_type_hue(df_sub, 'test', 'ic', 'inceptionv3_avgpool')
do_boxplot_with_ad_type_hue(df_sub, 'test', 'ic', 'PixCorr')
