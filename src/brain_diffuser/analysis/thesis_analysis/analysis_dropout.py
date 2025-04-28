import pandas as pd
import seaborn as sns
import os
import numpy as np
import matplotlib.pyplot as plt
from analysis_helpers import compute_output_name_for_dropout, competition_criterion
from matplotlib.ticker import FixedLocator
import pingouin as pg

sns.set_theme()
thesis_plots_path = "/Users/matt/ownCloud/gogo/MA/thesis/diversity_thesis/plots"

path_pattern_corr = "./data/df_pattern_correlation_dropout.csv"
path_profile_corr = "./data/df_profile_correlation_dropout.csv"
path_reconstruction = "./data/df_reconstruction_dropout.csv"

df_pattern_corr = pd.read_csv(path_pattern_corr).query("sub != 'JP'")
df_profile_corr = pd.read_csv(path_profile_corr).query("sub != 'JP'")
df_reconstruction = pd.read_csv(path_reconstruction).query("sub != 'JP'")

sub_renamer = {'TH':'S1', 'AM': 'S2', 'ES': 'S3', 'JK': 'S4', 'KS': 'S5'}
df_pattern_corr['sub'] = df_pattern_corr['sub'].replace(sub_renamer)
df_profile_corr['sub'] = df_profile_corr['sub'].replace(sub_renamer)
df_reconstruction['sub'] = df_reconstruction['sub'].replace(sub_renamer)

df_pattern_corr = df_pattern_corr.rename(columns={"id_acc":"id_acc_icnn"})

df_pattern_corr['data_fraction'] =  df_pattern_corr['dropout_ratio']
df_profile_corr['data_fraction'] =  df_profile_corr['dropout_ratio']
df_reconstruction['data_fraction'] =  df_reconstruction['dropout_ratio']

df_pattern_corr['dropout_ratio'] = 1 - df_pattern_corr["dropout_ratio"]
df_profile_corr['dropout_ratio'] = 1 - df_profile_corr["dropout_ratio"]
df_reconstruction['dropout_ratio'] = 1 - df_reconstruction["dropout_ratio"]

df_reconstruction['dreamsim_bd'] = 1 - df_reconstruction["dreamsim_bd"]
df_reconstruction['dreamsim_icnn'] = 1 - df_reconstruction["dreamsim_icnn"]

df_reconstruction['clip_final_bd'] = df_reconstruction['clip_final_bd'] - 0.5
df_reconstruction['clip_final_icnn'] = df_reconstruction['clip_final_icnn'] - 0.5

df_reconstruction['clip_sim_bd'] = 1 - df_reconstruction["clip_dist_bd"]
df_reconstruction['clip_sim_icnn'] = 1 - df_reconstruction["clip_dist_icnn"]

# Add 77 to all the dropout_trial of the quantized count conditions

df_reconstruction["dropout_trial"] = np.where(df_reconstruction["dropout_algorithm"].str.startswith("dropout-quantizedCount"), df_reconstruction["dropout_trial"] + 77, df_reconstruction["dropout_trial"])
df_pattern_corr["dropout_trial"] = np.where(df_pattern_corr["dropout_algorithm"].str.startswith("dropout-quantizedCount"), df_pattern_corr["dropout_trial"] + 77, df_pattern_corr["dropout_trial"])
df_profile_corr["dropout_trial"] = np.where(df_profile_corr["dropout_algorithm"].str.startswith("dropout-quantizedCount"), df_profile_corr["dropout_trial"] + 77, df_profile_corr["dropout_trial"])


df_reconstruction["dropout_trial"] = np.where(df_reconstruction["dropout_algorithm"].str.startswith("no-dropout"), df_reconstruction["dropout_trial"] + 77, df_reconstruction["dropout_trial"])
df_pattern_corr["dropout_trial"] = np.where(df_pattern_corr["dropout_algorithm"].str.startswith("no-dropout"), df_pattern_corr["dropout_trial"] + 77, df_pattern_corr["dropout_trial"])
df_profile_corr["dropout_trial"] = np.where(df_profile_corr["dropout_algorithm"].str.startswith("no-dropout"), df_profile_corr["dropout_trial"] + 77, df_profile_corr["dropout_trial"])


df_pattern_corr['all_icnn'] = df_pattern_corr[['conv1_1', 'conv1_2', 'conv2_1', 'conv2_2', 'conv3_1', 'conv3_2',
       'conv3_3', 'conv3_4', 'conv4_1', 'conv4_2', 'conv4_3', 'conv4_4',
       'conv5_1', 'conv5_2', 'conv5_3', 'conv5_4', 'fc6', 'fc7', 'fc8']].mean(axis=1)

df_pattern_corr['name'] = compute_output_name_for_dropout(df_pattern_corr)
df_reconstruction['name'] = compute_output_name_for_dropout(df_reconstruction)

competition_baseline_name = f'dropout-random_0.25_33'
competition_conditions = [
    'dropout-dreamsim_0.25_55',
    'dropout-clipvision_0.25_88',
    'dropout-pixels_0.25_44']

competition_criteria_pattern = ['id_acc_vdvae', 'id_acc_cliptext', 'id_acc_clipvision', 'id_acc_icnn', 'vdvae', 'cliptext_final', 'clipvision_final','all_icnn']
# competition_criteria = ['vdvae', 'cliptext_final', 'clipvision_final','all_icnn'] 
competition_criteria_recon = ['PixCorr_bd', 'PixCorr_icnn', 'dreamsim_bd', 'dreamsim_icnn', 'clip_final_bd', 'clip_final_icnn', "clip_sim_bd", "clip_sim_icnn"]
addition_cols_for_competition_df = ['dropout_algorithm']
df_competition_pattern = competition_criterion(df_pattern_corr, competition_baseline_name, competition_conditions, competition_criteria_pattern, addition_cols_for_competition_df)
df_competition_recon = competition_criterion(df_reconstruction, competition_baseline_name, competition_conditions, competition_criteria_recon, addition_cols_for_competition_df)

subject_col_renamer = {"sub": "Subject"}
df_pattern_corr = df_pattern_corr.rename(columns=subject_col_renamer)
df_profile_corr = df_profile_corr.rename(columns=subject_col_renamer)
df_reconstruction = df_reconstruction.rename(columns=subject_col_renamer)

hue_order_subs = ["S1", "S2", "S3", "S4", "S5"]

"""
Additional Plots for this experiment:
Qualitative Plots (created on smith)
UMAP Plot (in low_level_clustering::make_okay_plot_all_umaps)
Qualitative results of diversity subsampling (GIMP), raw version created back in the days with low_level_clustering::similarity_plot
Avg Min Distance Plot in validate_dropout_variance_increase::create_avg_min_distance_plot_all_spaces
Comparison of Monotone and homogeneous training images plain_background_subset::plot_two_rows_of_images

"""


### Validation

## Random Dropout (When do we see significant increase in performance?)
assert len(df_pattern_corr.query("dropout_algorithm == 'no-dropout'")['dropout_ratio'].value_counts()) == 1 # make sure in the no-dropout condition there's only trials where all of the data is used


# Plot1: Random dropout influence decoder
df_pattern_corr_random = df_pattern_corr.copy()
df_pattern_corr_random['dropout_algorithm'] = df_pattern_corr_random['dropout_algorithm'].replace({'no-dropout':'dropout-random'}) # Theoretically, a random dropout of 0% is the same as no dropout.

df_pattern_corr_random = df_pattern_corr_random.query("dropout_algorithm == 'dropout-random'")
df_pattern_corr_random = df_pattern_corr_random.query("Subject == 'S2'") # The random baseline subject
df_pattern_corr_random = df_pattern_corr_random.query("dropout_trial < 10")
df_pattern_corr_random["dropout_ratio"].value_counts()
df_pattern_corr_random["dropout_trial"].value_counts()
test_dataset = 'test'

def decoder_dropout_random_plot_both_datasets():
    fig, axes = plt.subplots(1, 2, figsize=(15, 5), sharex=True, sharey=True)

    for i,test_dataset in enumerate(['test', 'art']):
        ax = axes[i]
        df = df_pattern_corr_random.query(f"test_dataset == '{test_dataset}'")
        criterions = ["id_acc_clipvision", "id_acc_cliptext", "id_acc_vdvae", "id_acc_icnn"]
        melted = pd.melt(df, id_vars = ["im", "Subject", "data_fraction"], value_vars=criterions, value_name="id_acc", var_name="translator")
        melted["translator"] = melted["translator"].str.replace("id_acc_", "")
        translator_renamer = {"clipvision": "CLIP Vision", "cliptext": "CLIP Text", "vdvae": "VDVAE", "icnn": "iCNN"}
        melted["translator"] = melted["translator"].replace(translator_renamer)
        melted = melted.rename(columns={"translator": "Translator"})
        sns.pointplot(data=melted, x="data_fraction", y="id_acc", hue="Translator", errorbar="se", markersize=5, markeredgewidth=3, ax=ax)
        plt.setp(ax.get_legend().get_texts(), fontsize='16') # for legend text
        plt.setp(ax.get_legend().get_title(), fontsize='16') # for legend title
        if test_dataset == 'test':
            title = "Natural test images"
        else:
            title = "Artificial shapes"
        
        ax.set_title(title, fontsize = 18)  
        # ax.set(xlabel='Data fraction', ylabel='id_acc')
        ax.set_xlabel("Data fraction", fontsize = 18)
        ax.set_ylabel("ID Accuracy", fontsize = 18)
        ax.set_ylim([0.4, 1])
        ax.axhline(0.5, color='rosybrown', linestyle='-', linewidth=1, alpha=0.8)
        # ax.legend(prop={'size': 14})
    # fig.suptitle("Translator Performance Random Dropout", fontweight='bold')
    # plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()
    fig.savefig(os.path.join(thesis_plots_path, f'dropout_random_translator.png'), dpi=300)
    plt.clf()
decoder_dropout_random_plot_both_datasets()


df_reconstruction_random = df_reconstruction.copy()
df_reconstruction_random['dropout_algorithm'] = df_reconstruction_random['dropout_algorithm'].replace({'no-dropout':'dropout-random'}) 
df_reconstruction_random = df_reconstruction_random.query("dropout_algorithm == 'dropout-random'")
df_reconstruction_random = df_reconstruction_random.query("Subject == 'S2'") # The random baseline subject
df_reconstruction_random = df_reconstruction_random.query("dropout_trial < 10")


# Plot2: Random dropout influence reconstruction
def reconstruction_random_dropout_plot_all_datasets():
    fig, axes = plt.subplots(1, 2, figsize=(15, 5), sharex=True, sharey=True)
    for i,test_dataset in enumerate(['test', 'art']):
        ax = axes[i]
        df = df_reconstruction_random.query(f"test_dataset == '{test_dataset}'").copy()
        group_cols = ['dropout_trial', 'test_dataset', 'data_fraction']
        criterions = [f"PixCorr_bd",f"PixCorr_icnn", f"dreamsim_bd", f"dreamsim_icnn", f"clip_bd", f"clip_icnn"]
        criterion_renamer = {f"clip_final_bd":"clip_bd", f"clip_final_icnn":"clip_icnn"}
        # Rename the criterions for better readability
        df = df.rename(columns=criterion_renamer)

        df = df[group_cols+criterions].groupby(group_cols).mean().reset_index()
        colors = ["#B22222", "#DC143C",  # Reds: Crimson and Firebrick
                "#1E90FF", "#00BFFF",  # Blues: Dodger Blue and Deep Sky Blue
                "#556B2F", "#32CD32"]  # Greens: Dark Olive Green and Lime Green
        palette = {c:colors[i] for i,c in enumerate(criterions)}
        melted = pd.melt(df, id_vars = ["data_fraction"], value_vars=criterions, value_name="performance", var_name="criterion")
        # melted["criterion"] = melted["criterion"].str.replace("_bd", " bd")
        # melted["criterion"] = melted["criterion"].str.replace("_icnn", " icnn")
        sns.pointplot(data=melted, x="data_fraction", y="performance", hue="criterion", palette=palette, errorbar="se", markersize=5, markeredgewidth=3, ax=ax)
        # make the legend outside of the plot
        # plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)

        if test_dataset == 'test':
            title = 'Natural test images'
        else:
            title = 'Artificial shapes'
        ax.set_title(title, fontsize=18)
        # ax.set(xlabel='Data fraction', ylabel='Performance')
        ax.set_xlabel("Data fraction", fontsize = 18)
        ax.set_ylabel("Performance", fontsize = 18)


    # fig.suptitle("Reconstruction Performance Random Dropout", fontweight='bold')
    fig.savefig(os.path.join(thesis_plots_path, f'dropout_random_reconstruction.png'), dpi=300)
    plt.show()
    plt.clf()
# reconstruction_random_dropout_plot_all_datasets()

def reconstruction_random_dropout_plot_all_datasets_transposed():
    fig, axes = plt.subplots(3, 2, figsize=(13, 5), sharex=True)
    criteria_renamer = {"PixCorr":"PixCorr", "dreamsim":"DreamSim", "clip_final":"CLIP-acc"}
    criteria = list(criteria_renamer.values())
    filter_cols = ['PixCorr_bd', 'PixCorr_icnn', "dreamsim_bd", "dreamsim_icnn","clip_final_bd", "clip_final_icnn", "Subject", "im", "dropout_trial", "data_fraction"]
    algorithm_renamer = {"bd": "Brain-Diffuser", "icnn": "iCNN"}

    for i,test_dataset in enumerate(['test', 'art']):
        df = df_reconstruction_random.query(f"test_dataset == '{test_dataset}'").copy()
        df = df.loc[:, filter_cols]
        df_reset = df.reset_index()
        df = pd.wide_to_long(df_reset,
                            stubnames=['PixCorr', 'dreamsim', 'clip_final'],
                            i='index',             # use the reset index
                            j='algorithm',
                            sep='_',
                            suffix='(bd|icnn)').reset_index()
        df = df.rename(columns=criteria_renamer)
        df["algorithm"] = df["algorithm"].replace(algorithm_renamer)
        group_cols = ['dropout_trial', 'data_fraction', "algorithm"]
        df_criterion = df[group_cols+criteria].groupby(group_cols).mean().reset_index()
        for j, criterion in enumerate(criteria):
            ax = axes[j,i]
            sns.pointplot(data=df_criterion, x="data_fraction", y=criterion, hue="algorithm", errorbar="se", markersize=5, markeredgewidth=3, ax=ax)
            # make the legend outside of the plot
            # plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
            if i == 0 and j == 0:
                ...
                # ax.legend()
            else:
                ax.get_legend().remove()

            if test_dataset == 'test':
                title = 'Natural test images'
            else:
                title = 'Artificial shapes'
            if j == 0:
                ax.set_title(title, fontsize=18)
            # ax.set(xlabel='Data fraction', ylabel='Performance')
            ax.set_xlabel("Data fraction", fontsize = 18)
            if i == 0:
                ax.set_ylabel(criterion, fontsize = 18)
            else: 
                ax.set_ylabel("")


    # fig.suptitle("Reconstruction Performance Random Dropout", fontweight='bold')
    fig.savefig(os.path.join(thesis_plots_path, f'dropout_random_reconstruction.png'), dpi=300)
    plt.show()
    plt.clf()
reconstruction_random_dropout_plot_all_datasets_transposed()

# Plot3: Random dropout influence qualitative
# See qualitative_eval.py

## Subsampling Visualization in low_level_clustering.py
# UMAP Visualization for dreamsim, pixels and clip
# Similarity Plot of 10 images and the closest image in the corresponding feature space

## Sampling Validation in validate_dropout_variance_increase.py
# with avg-min-distance


### Final Experiment

## Decoder

# Plots
df_pattern_corr_eval = df_pattern_corr.copy()
df_pattern_corr_eval = df_pattern_corr_eval.query("dropout_trial > 10")
df_pattern_corr_eval = df_pattern_corr_eval.query("data_fraction == 0.25")

def set_ticks_and_axes(ax, xticklabels, title, ylabel, sethline=False, ticksize = 16, labelsize = 17, titlesize = 18):
    ax.set_xticks(range(len(xticklabels)))  # Ensure it aligns with categorical data
    ax.set_xticklabels(xticklabels, size=ticksize)
    ax.yaxis.get_label().set_fontsize(labelsize)
    ax.tick_params(axis='x', labelrotation=45)
    ax.xaxis.set_tick_params(pad=-5)
    ax.set_title(title, size=titlesize)
    ax.set(ylabel=ylabel)
    if sethline:
        ax.axhline(0.5, color='rosybrown', linestyle='-', linewidth=1, alpha=0.8)

def decoder_dropout_eval_plot(test_dataset, main_analysis=True):
    if main_analysis:
        df = df_pattern_corr_eval.query(f"test_dataset == '{test_dataset}'").copy()
    else:
        # df = df_pattern_corr_eval2.query(f"test_dataset == '{test_dataset}'").copy()
        df = df_pattern_corr_eval.query(f"test_dataset == '{test_dataset}'").copy()
    if test_dataset == 'test':
        title = 'Translator performance (0.25 data fraction) natural test images'
    else:
        title = 'Translator performance (0.25 data fraction) artificial shapes'
    criterions = ["id_acc_clipvision", "id_acc_cliptext", "id_acc_vdvae", "id_acc_icnn"]

    if main_analysis:
        dropout_algorithm_renamer = {'dropout-random':'Random', 'dropout-pixels':'Pixels', 'dropout-clipvision':'CLIP', 'dropout-dreamsim':'dreamsim'}
    else:
        dropout_algorithm_renamer = {'dropout-random':'Random', "dropout-quantizedCountParty": "Heterogeneous", "dropout-quantizedCountBoring": "Monotone"}

    df['dropout_algorithm'] = df['dropout_algorithm'].replace(dropout_algorithm_renamer)    
    criterions = ["id_acc_clipvision", "id_acc_cliptext", "id_acc_vdvae", "id_acc_icnn"]
    # fig, axes = plt.subplots(2, 2, figsize=(15, 10), sharex=True, sharey=True)
    fig, axes = plt.subplots(1,4, figsize=(16, 4), sharex=True, sharey=True)
    axes = axes.flatten()

    for i, criterion in enumerate(criterions):
        ax = axes[i]
        ax.set_ylim([0.4, 1])
        sns.pointplot(data=df, x='dropout_algorithm', y=criterion, hue="Subject", hue_order=hue_order_subs,errorbar='se', markersize=3, markeredgewidth=3, dodge=.3, linestyle="none", order=dropout_algorithm_renamer.values(), ax=ax, alpha=0.7)
        
        title = criterion.replace("id_acc_", "") + " translator"
        title = title.replace("clip", "CLIP ")
        title = title.replace("vision", "Vision")
        title = title.replace("text", "Text")
        title = title.replace("vdvae", "VDVAE")
        title = title.replace("icnn", "iCNN")
        set_ticks_and_axes(ax, dropout_algorithm_renamer.values(), title, ylabel='ID accuracy', sethline=True)
        ax.set(xlabel='')
        if i == 0:
            ...
            # ax.legend(loc='upper left')
        else:
            ax.get_legend().remove()
        if i == 3:
            ax.axvline(x=-0.48, color='black', linestyle='-', linewidth=1)
            ax.axvline(x=-0.44, color='black', linestyle='-', linewidth=1)
        ax.set_ylabel("ID Accuracy", fontsize=15, fontweight='bold')

        
    fig.text(0.5, -0.17, 'Dropout Strategy', ha='center', va='center', fontweight='bold', fontsize=18)
    # fig.suptitle(title, fontweight='bold')
    fig.subplots_adjust(wspace=0, hspace=0)
    plt.show()
    if main_analysis:
        fig.savefig(os.path.join(thesis_plots_path, f'dropout_eval_translator_{test_dataset}.png'), bbox_inches='tight')
    else:
        fig.savefig(os.path.join(thesis_plots_path, f'dropout_discussion_translator_{test_dataset}_monotone.png'), bbox_inches='tight')

    plt.clf()
    quant_results = df.groupby(["Subject", 'dropout_algorithm'])[criterions].mean().reset_index().groupby('dropout_algorithm')[criterions].agg(['mean', 'std']).transpose()
    print(f"Translator Results for {test_dataset} data.")
    print(quant_results)

# main_analysis = False
# decoder_dropout_eval_plot("test", main_analysis)
# decoder_dropout_eval_plot("art", main_analysis)

decoder_dropout_eval_plot('test')
decoder_dropout_eval_plot('art')

## Reconstruction

# Plots Quant
df_reconstruction_eval = df_reconstruction.copy()
df_reconstruction_eval = df_reconstruction_eval.query("dropout_trial > 10")
df_reconstruction_eval = df_reconstruction_eval.query("data_fraction == 0.25")


def reconstruction_dropout_eval_plot_transposed(test_dataset):
    df = df_reconstruction_eval.query(f"test_dataset == '{test_dataset}'").copy()
    dropout_algorithm_renamer = {'dropout-random':'Random', 'dropout-pixels':'Pixels', 'dropout-clipvision':'CLIP Vision', 'dropout-dreamsim':'dreamsim'}
    df['dropout_algorithm'] = df['dropout_algorithm'].replace(dropout_algorithm_renamer)

    criteria = ['PixCorr_bd', 'PixCorr_icnn', "dreamsim_bd", "dreamsim_icnn","clip_final_bd", "clip_final_icnn"]
    criteria_names = ["PixCorr brain-diffuser", "PixCorr ICNN", "dreamsim sim brain-diffuser", 'dreamsim sim ICNN', 'clip-accuracy brain-diffuser', 'clip-accuracy ICNN']
    fig, axes = plt.subplots(3, 2, figsize=(16, 8), sharex=True, sharey=False)

    axes = axes.flatten()
    for i, criterion in enumerate(criteria):
        ax = axes[i]
        sns.pointplot(data=df, x='dropout_algorithm', y=criterion, hue="Subject", hue_order=hue_order_subs, errorbar='se', markersize=5, markeredgewidth=3, dodge=.3, linestyle="none", order=dropout_algorithm_renamer.values(), ax=ax)

        if i == 0:
            ylabel='PixCorr'
        elif i == 2:
            ylabel='DreamSim'
        elif i == 4:
            ylabel='CLIP-acc'
        else:
            ylabel = ''

        if i == 0:
            title='Brain-Diffuser'
        elif i == 1:
            title='iCNN'
        else:
            title=''
        
        if test_dataset == 'test':
            if i == 0 or i == 1: # pixCorr
                ax.set_ylim([0.05, 0.35])
            if i == 2 or i == 3: # Dreamsim 
                ax.set_ylim([0.2, 0.35])
            elif i == 4 or i == 5: # clip-acc
                ax.set_ylim([-0.05, 0.35])
        elif test_dataset == 'art':
            if i == 0 or i == 1: # pixCorr
                ax.set_ylim([0.00, 0.3])
            if i == 2 or i == 3: # Dreamsim 
                ax.set_ylim([0.075, 0.225])
            elif i == 4 or i == 5: # clip-acc
                ax.set_ylim([-0.15, 0.3])

        if i == 1 or i == 3 or i == 5:
            ax.tick_params(axis='y', which='both', length=0, labelbottom=False, labelleft=False)

        set_ticks_and_axes(ax, dropout_algorithm_renamer.values(), criteria_names[i],ylabel=ylabel)

        ax.set_title(title, fontsize=18)
        if i == 0:
            ...
            # ax.legend(loc='upper left')
        else:
            ax.get_legend().remove()
        ax.set_ylabel(ylabel, fontsize=15, fontweight='bold')
        ax.set(xlabel='')
    fig.text(0.5, -0.013, 'Dropout Strategy', ha='center', va='center', fontweight='bold', fontsize=20)
    fig.subplots_adjust(wspace=0)
    plt.show()
    fig.savefig(os.path.join(thesis_plots_path, f'dropout_eval_reconstruction_{test_dataset}.png'), bbox_inches='tight')
    plt.clf()
    quant_for_stats = df.groupby(["Subject", 'dropout_algorithm'])[criteria].mean().reset_index()
    quant_metrics = quant_for_stats.groupby('dropout_algorithm')[criteria].agg(['mean', 'std']).transpose()


    """
    Also ich habe zwei Faktoren, einmal dropout Algorithmus und einmal Metric
    
    """
    qq = quant_for_stats.melt(["Subject", 'dropout_algorithm'], criteria)
    var = 'PixCorr_icnn'
    dropout_algorithms_to_test = ["Pixels", "dreamsim", "CLIP Vision"]

    for var in criteria:
        for dropout_algorithm in dropout_algorithms_to_test:
            s1 = qq.query(f"variable == '{var}' and dropout_algorithm == '{dropout_algorithm}'").sort_values("Subject")['value'].values
            s2 = qq.query(f"variable == '{var}' and dropout_algorithm == 'Random'").sort_values("Subject")['value'].values
            ttest = pg.ttest(s1, s2, True)
            p = np.round(ttest['p-val'].item(), 3)
            print(f"{test_dataset}: TTest for dropout space {dropout_algorithm} vs random for the criterion {var}: p= {p}")

    print(f"Recon Quant metrics for dataset {test_dataset}")
    print(quant_metrics)

reconstruction_dropout_eval_plot_transposed("test")
reconstruction_dropout_eval_plot_transposed("art")

# Plots Qual

# see qualitative_eval.py

# Table

# In this table there would be about a loot of values again. So maybe I shouldn't create it here?



##### Discussion Experiments with homogeneous/heterogeneous training data
df_pattern_corr_discussion = df_pattern_corr.copy()
df_pattern_corr_discussion = df_pattern_corr_discussion.query("dropout_trial > 10")
df_pattern_corr_discussion["dropout_algorithm"] = np.where(df_pattern_corr_discussion["dropout_ratio"] == 0, 'no-dropout', df_pattern_corr_discussion["dropout_algorithm"])
df_pattern_corr_discussion = df_pattern_corr_discussion.query("dropout_algorithm == 'no-dropout' or data_fraction == 0.25")


def discussion_plot_monotone_decoder(criterion = "id_acc_icnn"):
    df = df_pattern_corr_discussion.copy()
    test_datasets = ["test", "art"]
    dropout_algorithm_renamer = {'dropout-random':'Random', "dropout-quantizedCountParty": "Heterogeneous", "dropout-quantizedCountBoring": "Monotone", 'no-dropout': 'No Dropout'}
    df['dropout_algorithm'] = df['dropout_algorithm'].replace(dropout_algorithm_renamer)

    fig, axes = plt.subplots(1,2, figsize=(12, 4), sharex=True, sharey=True)
    axes = axes.flatten()

    for i, test_dataset in enumerate(test_datasets):
        dfd = df.query(f"test_dataset == '{test_dataset}'")
        ax = axes[i]
        ax.set_ylim([0.4, 1])
        sns.pointplot(data=dfd, x='dropout_algorithm', y=criterion, hue="Subject", hue_order=hue_order_subs,errorbar='se', markersize=3, markeredgewidth=3, dodge=.3, linestyle="none", order=dropout_algorithm_renamer.values(), ax=ax, alpha=0.7)
        
        # title = criterion.replace("id_acc_", "") + " translator"
        if test_dataset == 'test':
            title = "Natural test images"
        else:
            title = "Artificial shapes"

        set_ticks_and_axes(ax, dropout_algorithm_renamer.values(), title, ylabel='ID accuracy', sethline=True)
        ax.set(xlabel='')
        if i == 0:
            ...
            # ax.legend()
        else:
            ax.get_legend().remove()
        if i == 3:
            ax.axvline(x=-0.48, color='black', linestyle='-', linewidth=1)
            ax.axvline(x=-0.44, color='black', linestyle='-', linewidth=1)
        ax.set_ylabel("ID Accuracy", fontsize=15, fontweight='bold')

    fig.text(0.5, -0.17, 'Dropout Strategy', ha='center', va='center', fontweight='bold', fontsize=18)
    # fig.suptitle(title, fontweight='bold')
    fig.subplots_adjust(wspace=0, hspace=0)
    plt.show()

    fig.savefig(os.path.join(thesis_plots_path, f'dropout_discussion_translator_{criterion}.png'), bbox_inches='tight')

    plt.clf()
discussion_plot_monotone_decoder("id_acc_icnn")
discussion_plot_monotone_decoder("id_acc_cliptext")
discussion_plot_monotone_decoder("id_acc_clipvision")
discussion_plot_monotone_decoder("id_acc_vdvae")

df_reconstruction_discussion = df_reconstruction.copy()
df_reconstruction_discussion = df_reconstruction_discussion.query("dropout_trial > 10")
df_reconstruction_discussion["dropout_algorithm"] = np.where(df_reconstruction_discussion["dropout_ratio"] == 0, 'no-dropout', df_reconstruction_discussion["dropout_algorithm"])
df_reconstruction_discussion = df_reconstruction_discussion.query("dropout_algorithm == 'no-dropout' or data_fraction == 0.25")

def discussion_plot_monotone_reconstruction(algorithm):
    df = df_reconstruction_discussion.copy()
    dropout_algorithm_renamer = {'dropout-random':'Random', "dropout-quantizedCountParty": "Heterogeneous", "dropout-quantizedCountBoring": "Monotone", 'no-dropout': 'No Dropout'}
    df['dropout_algorithm'] = df['dropout_algorithm'].replace(dropout_algorithm_renamer)

    test_datasets = ["test", "art"]
    criterion = "dreamsim_bd" if algorithm == "bd" else "dreamsim_icnn"
    criterion_name = "dreamsim sim brain-diffuser" if algorithm == "bd" else 'dreamsim sim ICNN'

    fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharex=True, sharey=False)
    hue_order_subs = ["S1", "S2", "S3", "S4", "S5"]
    axes = axes.flatten()
    for i, test_dataset in enumerate(test_datasets):
        dfd = df.query(f"test_dataset == '{test_dataset}'")
        ax = axes[i]
        sns.pointplot(data=dfd, x='dropout_algorithm', y=criterion, hue="Subject", hue_order=hue_order_subs, errorbar='se', markersize=5, markeredgewidth=3, dodge=.3, linestyle="none", order=dropout_algorithm_renamer.values(), ax=ax)

        # if i == 1 or i == 3 or i == 5:
            # ax.tick_params(axis='y', which='both', length=0, labelbottom=False, labelleft=False)

        # set_ticks_and_axes(ax, dropout_algorithm_renamer.values(), criterion_name,ylabel=ylabel)
        if test_dataset == 'test':
            title = "Natural test images"
        else:
            title = "Artificial shapes"
        ylabel = "DreamSim"
        set_ticks_and_axes(ax, dropout_algorithm_renamer.values(), title, ylabel='ID accuracy', sethline=False)

        ax.set_title(title, fontsize=18)
        if i == 0:
            ...
            # ax.legend(label = "Subject")
        else:
            ax.get_legend().remove()
        ax.set_ylabel(ylabel, fontsize=15, fontweight='bold')
        ax.set(xlabel='')
    fig.text(0.5, -0.2, 'Dropout Strategy', ha='center', va='center', fontweight='bold', fontsize=20)
    # fig.subplots_adjust(wspace=0)
    plt.show()
    fig.savefig(os.path.join(thesis_plots_path, f'dropout_discussion_reconstruction_{algorithm}.png'), bbox_inches='tight')

algorithm = "bd"
discussion_plot_monotone_reconstruction("bd")
discussion_plot_monotone_reconstruction("icnn")


def discussion_plot_monotone_reconstruction_both_algorithms():
    fig, axes = plt.subplots(1, 4, figsize=(24, 4), sharex=True, sharey=False)
    axes = axes.flatten()
    i = 0
    for algorithm in ['bd', 'icnn']:
        df = df_reconstruction_discussion.copy()
        dropout_algorithm_renamer = {'dropout-random':'Random', "dropout-quantizedCountParty": "Heterogeneous", "dropout-quantizedCountBoring": "Monotone", 'no-dropout': 'No Dropout'}
        df['dropout_algorithm'] = df['dropout_algorithm'].replace(dropout_algorithm_renamer)

        test_datasets = ["test", "art"]
        criterion = "dreamsim_bd" if algorithm == "bd" else "dreamsim_icnn"
        criterion_name = "dreamsim sim brain-diffuser" if algorithm == "bd" else 'dreamsim sim ICNN'

        hue_order_subs = ["S1", "S2", "S3", "S4", "S5"]
        for test_dataset in test_datasets:
            dfd = df.query(f"test_dataset == '{test_dataset}'")
            ax = axes[i]
            sns.pointplot(data=dfd, x='dropout_algorithm', y=criterion, hue="Subject", hue_order=hue_order_subs, errorbar='se', markersize=5, markeredgewidth=3, dodge=.3, linestyle="none", order=dropout_algorithm_renamer.values(), ax=ax)

            # if i == 1 or i == 3 or i == 5:
                # ax.tick_params(axis='y', which='both', length=0, labelbottom=False, labelleft=False)

            # set_ticks_and_axes(ax, dropout_algorithm_renamer.values(), criterion_name,ylabel=ylabel)
            if test_dataset == 'test':
                title = "natural test images"
            else:
                title = "artificial shapes"
            ylabel = "DreamSim"
            set_ticks_and_axes(ax, dropout_algorithm_renamer.values(), title, ylabel='ID accuracy', sethline=False, ticksize=22, labelsize=22)

            algorithm_renamer = {'bd': "Brain-Diffuser", 'icnn': 'iCNN'}
            title = algorithm_renamer[algorithm] + ' ' +  title
            ax.set_title(title, fontsize=23)
            if i == 0:
                ...
                # ax.legend(label = "Subject")
                ax.set_ylabel(ylabel, fontsize=23, fontweight='bold')
                # plt.legend(fontsize=55)
                plt.setp(ax.get_legend().get_title(), fontsize='14') # for legend title
                plt.setp(ax.get_legend().get_texts(), fontsize='14') # for legend text
            else:
                ax.get_legend().remove()
                ax.set_ylabel(" ")
            
            ax.set(xlabel='')
            i += 1
    
    fig.text(0.5, -0.32, 'Dropout Strategy', ha='center', va='center', fontweight='bold', fontsize=25)
    # fig.subplots_adjust(wspace=0)
    plt.show()
    fig.savefig(os.path.join(thesis_plots_path, f'dropout_discussion_reconstruction.png'), bbox_inches='tight')

discussion_plot_monotone_reconstruction_both_algorithms()


########### Presentation
presentation_plots_folder = "/Users/matt/ownCloud/gogo/MA/thesis/diversity_thesis/presentation_plots"

######## Plot 1

# def presentation_plot_1(test_dataset, algorithm="bd", experiment="main"):
#     sns.set_context("talk")  # or "poster", "notebook", "paper"
#     title = ""
#     if test_dataset == "test":
#         title += "Natural test images "
#     else:
#         title += "Artificial shapes "

#     df = df_reconstruction_eval.query(f"test_dataset == '{test_dataset}'").copy()
    
#     if experiment == "main":
#         dropout_algorithm_renamer = {'dropout-random':'Random', 'dropout-dreamsim':'Diversity-based'}
#     else:
#         dropout_algorithm_renamer = {'dropout-random':'Random', "dropout-quantizedCountParty": "Heterogeneous", "dropout-quantizedCountBoring": "Monotone"}

#     df['dropout_algorithm'] = df['dropout_algorithm'].replace(dropout_algorithm_renamer)
#     criterion = "dreamsim_"+algorithm
#     fig, ax = plt.subplots(1, 1,)
#     sns.pointplot(data=df, x='dropout_algorithm', y=criterion, hue="Subject", hue_order=hue_order_subs, errorbar='se', markersize=5, markeredgewidth=3, dodge=.25, linestyle="none", order=dropout_algorithm_renamer.values(), ax=ax)
#     ax.set_ylabel("DreamSim")
#     ax.set(xlabel='Dropout strategy')
#     ax.set_title(title, fontweight='bold')
#     plt.legend(fontsize='small')  # options: 'x-small', 'small', 10, etc.
#     plt.show()
#     fig.savefig(os.path.join(presentation_plots_folder, f'exp1_{experiment}_{algorithm}_{test_dataset}.png'), bbox_inches='tight')
#     plt.clf()


# presentation_plot_1("test", "bd", "main")
# presentation_plot_1("art", "bd", "main")
# presentation_plot_1("test", "icnn", "main")
# presentation_plot_1("art", "icnn", "main")


# presentation_plot_1("art", "bd", "heterogeneous")
# presentation_plot_1("test", "bd", "heterogeneous")

# presentation_plot_1("art", "icnn", "heterogeneous")
# presentation_plot_1("test", "icnn", "heterogeneous")