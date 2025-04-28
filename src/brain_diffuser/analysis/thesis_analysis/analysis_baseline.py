import pandas as pd
import seaborn as sns
import os
import matplotlib.pyplot as plt
import numpy as np
sns.set_theme()

thesis_plots_path = "/Users/matt/ownCloud/gogo/MA/thesis/diversity_thesis/plots"

path_pattern_corr = "./data/df_pattern_correlation_baseline.csv"
path_profile_corr = "./data/df_profile_correlation_baseline.csv"
path_reconstruction = "./data/df_reconstruction_baseline.csv"

df_pattern_corr = pd.read_csv(path_pattern_corr).query("sub != 'JP'")
df_profile_corr = pd.read_csv(path_profile_corr).query("sub != 'JP'")
df_reconstruction = pd.read_csv(path_reconstruction).query("sub != 'JP'")

sub_renamer = {'TH':'S1', 'AM': 'S2', 'ES': 'S3', 'JK': 'S4', 'KS': 'S5'}
df_pattern_corr['sub'] = df_pattern_corr['sub'].replace(sub_renamer)
df_profile_corr['sub'] = df_profile_corr['sub'].replace(sub_renamer)
df_reconstruction['sub'] = df_reconstruction['sub'].replace(sub_renamer)
df_pattern_corr = df_pattern_corr.rename(columns={"id_acc":"id_acc_icnn"})

df_reconstruction['dreamsim_bd'] = 1 - df_reconstruction["dreamsim_bd"]
df_reconstruction['dreamsim_icnn'] = 1 - df_reconstruction["dreamsim_icnn"]

df_reconstruction['clip_sim_bd'] = 1 - df_reconstruction["clip_dist_bd"]
df_reconstruction['clip_sim_icnn'] = 1 - df_reconstruction["clip_dist_icnn"]

df_reconstruction['clip_final_bd'] = df_reconstruction['clip_final_bd'] - 0.5
df_reconstruction['clip_final_icnn'] = df_reconstruction['clip_final_icnn'] - 0.5

subject_col_renamer = {"sub": "Subject"}
df_pattern_corr = df_pattern_corr.rename(columns=subject_col_renamer)
df_profile_corr = df_profile_corr.rename(columns=subject_col_renamer)
df_reconstruction = df_reconstruction.rename(columns=subject_col_renamer)

hue_order_subs = ["S1", "S2", "S3", "S4", "S5"]


"""
Additional Plots:
A selection of natural images and artificial shapes  (in additional_plots.py)

"""

#### Decoder
test_dataset = "test"

def decoder_baseline_plot(test_dataset):
    df = df_pattern_corr.query(f"test_dataset == '{test_dataset}'")
    

    # Melt the dataset first
    criterions = ["id_acc_clipvision", "id_acc_cliptext", "id_acc_vdvae", "id_acc_icnn"]
    melted = pd.melt(df, id_vars = ["im", "Subject"], value_vars=criterions, value_name="id_acc", var_name="condition")
    melted["condition"] = melted["condition"].str.replace("id_acc_", "")

    ax = sns.pointplot(
        data=melted, x="condition", y="id_acc", hue="Subject", hue_order=hue_order_subs,
        dodge=.75, linestyle="none", errorbar="se", markersize=5, markeredgewidth=3,
    )

    if test_dataset == 'test':
        title = 'Baseline Translator performance natural test images'
    else:
        title = 'Baseline Translator performance artificial shapes'
    # ax.set_title(title)
    ax.set(xlabel='Translator')
    plt.ylabel('ID Accuracy')
    fig = ax.get_figure()
    fig.savefig(os.path.join(thesis_plots_path, f'baseline_translator_{test_dataset}.png')) 
    plt.show()
    plt.clf()

decoder_baseline_plot('test')
decoder_baseline_plot('art')



def decoder_baseline_plot_both_datasets():
    condition_renamer = {'clipvision': 'CLIP Vision', 'cliptext': 'CLIP Text', 'vdvae': 'VDVAE', 'icnn': 'iCNN'}
    fig, axes = plt.subplots(1,2, figsize=(10, 4), sharey=True)
    for i,test_dataset in enumerate(['test', 'art']):
        ax = axes[i]

        df = df_pattern_corr.query(f"test_dataset == '{test_dataset}'")
    
        # Melt the dataset first
        criterions = ["id_acc_clipvision", "id_acc_cliptext", "id_acc_vdvae", "id_acc_icnn"]
        melted = pd.melt(df, id_vars = ["im", "Subject"], value_vars=criterions, value_name="id_acc", var_name="condition")
        melted["condition"] = melted["condition"].str.replace("id_acc_", "")
        melted['condition'] = melted['condition'].replace(condition_renamer)

        sns.pointplot(
            data=melted, x="condition", y="id_acc", hue="Subject", hue_order=hue_order_subs,
            dodge=.75, linestyle="none", errorbar="se", markersize=5, markeredgewidth=3,ax=ax
        )

        if test_dataset == 'test':
            # title = 'Baseline Translator performance natural test images'
            title = 'Natural test images'
            ax.set_ylabel('ID Accuracy', fontweight='bold')
        else:
            # title = 'Baseline Translator performance artificial shapes'
            title = 'Artificial shapes'
            ax.set_ylabel(' ')
            ax.get_legend().remove()

        ax.set_title(title)
        ax.set_xlabel('Translator', fontweight='bold')

        # plt.ylabel('ID Accuracy')
    fig.savefig(os.path.join(thesis_plots_path, f'baseline_translator.png')) 
    plt.show()
    plt.clf()

decoder_baseline_plot_both_datasets()


## Results Table for the translator performance
# criterions = ["id_acc_clipvision", "id_acc_cliptext", "id_acc_vdvae", "id_acc_icnn"]
test_images_col_name = "test images"
criterions_name_mapper = {"id_acc_clipvision":"clipvision", "id_acc_cliptext":"cliptext", "id_acc_vdvae":"vdvae", "id_acc_icnn":"icnn", "test_dataset": test_images_col_name}
test_dataset_value_renamer={"test": "natural", "art": "artificial"}

df_translator= df_pattern_corr[list(criterions_name_mapper.keys())+["Subject"]].copy()
df_translator["test_dataset"] = df_translator["test_dataset"].replace(test_dataset_value_renamer)
group_per_sub = df_translator.groupby(["Subject", "test_dataset"])[list(criterions_name_mapper.keys())[:-1]].mean().reset_index().drop("Subject", axis=1).groupby("test_dataset")
group_per_sub_for_meta_data = group_per_sub.mean()

# Data for the table
table_mean = group_per_sub.mean().values.round(3).astype(str)
table_std = group_per_sub.std().values.round(3).astype(str)

vectorized_slice = np.vectorize(lambda s: s[1:]) # Get rid of the 0 in the std values
table_std = vectorized_slice(table_std)
table_data = table_mean + ' (' + table_std + ')'


df_table = pd.DataFrame(table_data, columns=group_per_sub_for_meta_data.columns, index=group_per_sub_for_meta_data.index)
# df_table = df_table.reset_index('test_dataset')
df_table = df_table.rename(columns=criterions_name_mapper)

df_table = df_table.sort_index(ascending=False)
df_table.index.name = 'test images'

with open(os.path.join(thesis_plots_path, 'baseline_translator_table.tex'), 'w') as f:
    f.write(df_table.to_latex())


#### Reconstruction results
test_dataset = "test"

def reconstruction_baseline_plot(test_dataset):
    df = df_reconstruction.query(f"test_dataset == '{test_dataset}'")
    df = df.rename(columns={"id_acc":"id_acc_icnn"})

    criterions = [f"PixCorr_bd",f"PixCorr_icnn", f"dreamsim_bd", f"dreamsim_icnn", f"clip_final_bd", f"clip_final_icnn"]
    melted = pd.melt(df, id_vars = ["im", "Subject"], value_vars=criterions, value_name="value", var_name="metric")
    melted["metric"] = melted["metric"].str.replace("final_", "")

    plt.figure(figsize=(10, 6))

    ax = sns.pointplot(
        data=melted, x="metric", y="value", hue="Subject", hue_order=hue_order_subs,
        dodge=.4, linestyle="none", errorbar='se', markersize=5, markeredgewidth=3,
    )
    ax.set_xticks(range(len(criterions)))
    ax.set_xticklabels(['bd', 'icnn','bd', 'icnn','bd', 'icnn'], fontsize=16)

    # Add secondary labels for metrics
    # Calculate the positions for metric labels (center of each metric pair)
    metrics = ['pixCorr', 'dreamsim', 'clip']
    num_datasets = 2  # 'bd' and 'icnn'

    for i, metric in enumerate(metrics):
        center = i * num_datasets + (num_datasets - 1) / 2
        ax.text(center, ax.get_ylim()[0] - 0.08*(ax.get_ylim()[1]-ax.get_ylim()[0]), 
                metric, ha='center', va='top', fontsize=18, fontweight='bold')
    ax.set(xlabel="")
    plt.subplots_adjust(bottom=0.2)

    if test_dataset == 'test':
        title = 'Baseline Reconstruction performance natural test images'
    else:
        title = 'Baseline Reconstruction performance artificial shapes'

    # plt.title(title, fontsize=16)
    plt.ylabel('performance', fontsize=18)
    plt.savefig(os.path.join(thesis_plots_path, f'baseline_reconstruction_{test_dataset}.png')) 
    plt.show()
    plt.clf()

# reconstruction_baseline_plot('test')
# reconstruction_baseline_plot('art')

def reconstruction_baseline_plot_transposed(test_dataset):
    df = df_reconstruction.query(f"test_dataset == '{test_dataset}'").copy()
    filter_cols = ['PixCorr_bd', 'PixCorr_icnn', "dreamsim_bd", "dreamsim_icnn","clip_final_bd", "clip_final_icnn", "Subject", "im"]
    
    df = df.loc[:, filter_cols]
    df_reset = df.reset_index()  # creates a new 'index' column
    df_long = pd.wide_to_long(df_reset,
                            stubnames=['PixCorr', 'dreamsim', 'clip_final'],
                            i='index',             # use the reset index
                            j='algorithm',
                            sep='_',
                            suffix='(bd|icnn)').reset_index()
    
    algorithm_renamer = {"bd": "Brain-Diffuser", "icnn": "iCNN"}
    df_long["algorithm"] = df_long["algorithm"].replace(algorithm_renamer)
    criteria = ["PixCorr", "dreamsim", "clip_final"]
    fig, axes = plt.subplots(3, 1, figsize=(8, 8), sharex=True, sharey=False)
    axes = axes.flatten()
    for i, criterion in enumerate(criteria):
        ax = axes[i]
        sns.pointplot(data=df_long, x='algorithm', y=criterion, hue="Subject", hue_order=hue_order_subs,errorbar='se', markersize=5, markeredgewidth=3, dodge=.3, linestyle="none",  order=algorithm_renamer.values(), ax=ax)
        if i == 0:
            ylabel='PixCorr'
        elif i == 1:
            ylabel='DreamSim'
        elif i == 2:
            ylabel='CLIP-acc'
        ax.set_xticks(range(len(algorithm_renamer.values())))
        ax.set_xticklabels(algorithm_renamer.values(), size=16)
        if i == 0:
            ...
            # ax.legend(loc='upper left')
        else:
            ax.get_legend().remove()
        ax.set_ylabel(ylabel, fontsize=15, fontweight='bold')
        ax.set(xlabel='')
    # fig.subplots_adjust(wspace=0)
    plt.show()
    fig.savefig(os.path.join(thesis_plots_path, f'baseline_reconstruction_{test_dataset}.png'), bbox_inches='tight')
    plt.clf()

reconstruction_baseline_plot_transposed('test')
reconstruction_baseline_plot_transposed('art')




def reconstruction_baseline_plot_transposed_both_datasets():
    fig, axes = plt.subplots(3, 2, figsize=(16, 8), sharex=True, sharey=False)
    # axes = axes.flatten()

    for col,test_dataset in enumerate(['test', 'art']):
        df = df_reconstruction.query(f"test_dataset == '{test_dataset}'").copy()
        filter_cols = ['PixCorr_bd', 'PixCorr_icnn', "dreamsim_bd", "dreamsim_icnn","clip_final_bd", "clip_final_icnn", "Subject", "im"]
        
        df = df.loc[:, filter_cols]
        df_reset = df.reset_index()  # creates a new 'index' column
        df_long = pd.wide_to_long(df_reset,
                                stubnames=['PixCorr', 'dreamsim', 'clip_final'],
                                i='index',             # use the reset index
                                j='algorithm',
                                sep='_',
                                suffix='(bd|icnn)').reset_index()
        
        algorithm_renamer = {"bd": "Brain-Diffuser", "icnn": "iCNN"}
        df_long["algorithm"] = df_long["algorithm"].replace(algorithm_renamer)
        criteria = ["PixCorr", "dreamsim", "clip_final"]
        
        
        for i, criterion in enumerate(criteria):
            ax = axes[i, col]
            sns.pointplot(data=df_long, x='algorithm', y=criterion, hue="Subject", hue_order=hue_order_subs,errorbar='se', markersize=5, markeredgewidth=3, dodge=.3, linestyle="none",  order=algorithm_renamer.values(), ax=ax)
            if i == 0:
                ylabel='PixCorr'
                ax.set_ylim([0.05, 0.35])
            elif i == 1:
                ylabel='DreamSim'
                ax.set_ylim([0.1, 0.4])
            elif i == 2:
                ylabel='CLIP-acc'
                ax.set_ylim([-0.05, 0.35])
            ax.set_xticks(range(len(algorithm_renamer.values())))
            ax.set_xticklabels(algorithm_renamer.values(), size=16)
            if col == 0:
                ax.set_ylabel(ylabel, fontsize=15, fontweight='bold')
            else:
                ax.set_ylabel(" ")
            ax.set(xlabel='')

            if test_dataset != 'test' or i !=0:
                ax.get_legend().remove()

            if test_dataset == 'test' and i == 0:
                # title = 'Baseline Translator performance natural test images'
                title = 'Natural test images'
                ax.set_title(title, fontsize = 20)
            elif test_dataset == 'art' and i == 0:
                # title = 'Baseline Translator performance artificial shapes'
                title = 'Artificial shapes'
                ax.set_title(title, fontsize = 20)
        # fig.subplots_adjust(wspace=0)
    plt.show()
    fig.savefig(os.path.join(thesis_plots_path, f'baseline_reconstruction.png'), bbox_inches='tight')
    plt.clf()




reconstruction_baseline_plot_transposed_both_datasets()

## Results Table for reconstruction results



df = df_reconstruction.copy()
# df_reconstruction.groupby(['test_dataset', "Subject"]).mean(err)

df = df.rename(columns={"id_acc":"id_acc_icnn"})
criterions = [f"PixCorr_bd",f"PixCorr_icnn", f"dreamsim_bd", f"dreamsim_icnn", f"clip_final_bd", f"clip_final_icnn"]

melted = pd.melt(df, id_vars = ["im", "Subject", 'test_dataset'], value_vars=criterions, value_name="value", var_name="metric")
melted['test_dataset'] = melted['test_dataset'].replace(test_dataset_value_renamer)
melted["metric"] = melted["metric"].str.replace("final_", "")

melted['algorithm'] = melted['metric'].str.split('_').str[-1]
melted['metric'] = melted['metric'].str.split('_').str[0]
melted = melted.rename(columns={'metric': ' '})

# unmelt metric to columns
melted = melted.pivot(index=["Subject", 'test_dataset', 'im', 'algorithm'], columns=' ', values='value').reset_index()
melted = melted.drop(['im'], axis=1)

grouped_mean = melted.groupby(["Subject", 'test_dataset', 'algorithm']).mean().reset_index().drop("Subject", axis=1)

grouped_mean = grouped_mean.rename(columns=criterions_name_mapper)

table_mean =  grouped_mean.groupby([test_images_col_name, 'algorithm']).mean()
table_mean.index.name = 'test images'

table_std =  grouped_mean.groupby([test_images_col_name, 'algorithm']).std().round(3).astype(str)
table_std = vectorized_slice(table_std)

table_data = table_mean.round(3).astype(str) + ' (' + table_std + ')'

df_table = pd.DataFrame(table_data)

df_table = df_table.rename(columns={"PixCorr":"pixCorr", "dreamsim":"dreamsim", "clip":"clip"})

df_table = df_table.sort_index(ascending=False)
df_table = df_table[['pixCorr', 'dreamsim', 'clip']]

with open(os.path.join(thesis_plots_path, 'baseline_reconstruction_table.tex'), 'w') as f:
    f.write(df_table.to_latex())

