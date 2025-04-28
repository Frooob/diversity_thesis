import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np


df_pattern_correlation = pd.read_csv('df_pattern_correlation.csv')
df_profile_correlation = pd.read_csv('df_profile_correlation.csv')
df_reconstruction = pd.read_csv('df_reconstruction.csv')


c1 = sns.color_palette()[0]
c2 = sns.color_palette()[1]

# Pattern Correlation

def pattern_corr_plot_art_vs_test(metric):
    sns.pointplot(
        data=df_pattern_correlation,
        x="dropout_ratio",
        y=f"{metric}",
        marker="o",
        linestyles="-",
        hue="test_dataset",
    ).set_title(f"{metric}")
    plt.show()


pattern_corr_plot_art_vs_test("conv1_1")
pattern_corr_plot_art_vs_test("conv4_1")
pattern_corr_plot_art_vs_test("fc8")

pattern_corr_plot_art_vs_test("vdvae")

pattern_corr_plot_art_vs_test("cliptext_final")
pattern_corr_plot_art_vs_test("cliptext_tokens")

pattern_corr_plot_art_vs_test("clipvision_final")
pattern_corr_plot_art_vs_test("clipvision_patches")

# Profile Correlation

test_dataset = "art"
group_cols = ['dataset', 'sub', 'test_dataset', 'dropout_algorithm', 'dropout_ratio', 'dropout_trial', 'ft_name']
fig,ax = plt.subplots(1)
df_profile_correlation[df_profile_correlation["ft_name"].isin([
# 'conv1_1', 
'clipvision_tokens',
 ])]
dfp = df_profile_correlation.groupby(group_cols)["profile_corr"].median().reset_index()
dfp = dfp[dfp["ft_name"].isin([
# 'conv1_1', 
# 'conv1_2', 
# 'conv2_1', 
# 'conv2_2', 
# 'conv3_1', 
# 'conv3_2',
# 'conv3_3', 
# 'conv3_4', 
# 'conv4_1', 
# 'conv4_2', 
# 'conv4_3', 
# 'conv4_4',
# 'conv5_1', 
# 'conv5_2', 
# 'conv5_3', 
# 'conv5_4', 
# 'fc6', 
# 'relu6', 
# 'fc7',
# 'relu7', 
# 'fc8'
# 'cliptext_final',
# 'cliptext_tokens',
# 'clipvision_final',
# 'clipvision_patches',
'vdvae',
 ])]

dfp = dfp[dfp["test_dataset"] == test_dataset]
sns.pointplot(data=dfp, x="dropout_ratio", y="profile_corr", hue="ft_name", ax=ax).set_title(f"dataset: {test_dataset}")


### Reconstruction

criterions = ["PixCorr_bd", "PixCorr_icnn"]
melted = pd.melt(df_reconstruction, ["dropout_ratio", "dropout_trial", 'dropout_algorithm'], value_vars=criterions)
recon_criterion = "clip_final_icnn"
sns.pointplot(data=melted, x="dropout_ratio", y="value", hue="variable").set_title("PixelCorrelation (Low-Level)")


criterions = ["dreamsim_bd", "dreamsim_icnn"]
melted = pd.melt(df_reconstruction, ["dropout_ratio", "dropout_trial", 'dropout_algorithm'], value_vars=criterions)
recon_criterion = "clip_final_icnn"
sns.pointplot(data=melted, x="dropout_ratio", y="value", hue="variable").set_title("Dreamsim (Mid-Level)")

criterions = ["clip_final_bd", "clip_final_icnn"]
melted = pd.melt(df_reconstruction, ["dropout_ratio", "dropout_trial", 'dropout_algorithm'], value_vars=criterions)
recon_criterion = "clip_final_icnn"
sns.pointplot(data=melted, x="dropout_ratio", y="value", hue="variable").set_title("ClipFinal (High-Level)")



# How about some double melting??

# criterions = ["clip_final_bd", "clip_final_icnn"]
# melted = pd.melt(df_reconstruction, ["dropout_ratio", "dropout_trial", 'dropout_algorithm', "test_dataset"], value_vars=criterions)
# double_melted = pd.melt(melted, ["dropout_ratio", "dropout_trial", 'dropout_algorithm'], value_vars=[])
# recon_criterion = "clip_final_icnn"
# sns.pointplot(data=melted, x="dropout_ratio", y="value", hue="variable").set_title("ClipFinal (High-Level)")


# split by dropout conditions
metrics_reg = [
    "conv1_1",
    "conv4_1",
    "fc8",
    "vdvae",
    "cliptext_final",
    "cliptext_tokens",
    "clipvision_final",
    "clipvision_patches"
]


metric = 'conv1_1'

test_dataset = 'test'
dfp = df_pattern_correlation[df_pattern_correlation['test_dataset'] == test_dataset]


sns.pointplot(
    data=dfp,
    x="dropout_ratio",
    y=f"{metric}",
    marker="o",
    linestyles="-",
    hue="dropout_algorithm",
).set_title(f"{metric}")
plt.show()


# reconstruction
metrics_rec = [
    'PixCorr_bd',
    'PixCorr_icnn',
    'dreamsim_bd',
    'dreamsim_icnn',
    'clip_final_bd',
    'clip_final_icnn'
]
metric = "dreamsim_bd"

test_dataset = 'art'
dfr = df_reconstruction[df_reconstruction['test_dataset'] == test_dataset]

sns.pointplot(
    data=dfr,
    x="dropout_ratio",
    y=f"{metric}",
    marker="o",
    linestyles="-",
    hue="dropout_algorithm",
).set_title(f"{metric}")
plt.show()
