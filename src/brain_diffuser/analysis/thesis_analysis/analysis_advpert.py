import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os

sns.set_theme()
thesis_plots_path = "/Users/matt/ownCloud/gogo/MA/thesis/diversity_thesis/plots"

path_pattern_corr = "./data/df_pattern_correlation_advpert.csv"
path_profile_corr = "./data/df_profile_correlation_advpert.csv"
path_reconstruction = "./data/df_reconstruction_advpert.csv"

df_pattern_corr = pd.read_csv(path_pattern_corr).query("sub != 'JP'")
df_profile_corr = pd.read_csv(path_profile_corr).query("sub != 'JP'")
df_reconstruction = pd.read_csv(path_reconstruction).query("sub != 'JP'")

sub_renamer = {'TH':'S1', 'AM': 'S2', 'ES': 'S3', 'JK': 'S4', 'KS': 'S5'}
df_pattern_corr['sub'] = df_pattern_corr['sub'].replace(sub_renamer)
df_profile_corr['sub'] = df_profile_corr['sub'].replace(sub_renamer)
df_reconstruction['sub'] = df_reconstruction['sub'].replace(sub_renamer)

df_reconstruction['dreamsim'] = 1 - df_reconstruction["dreamsim"]
df_reconstruction['clip_final'] = df_reconstruction["clip_final"] - 0.5

df_pattern_corr = df_pattern_corr.rename(columns={"id_acc":"id_acc_icnn"})

df_pattern_corr["pert_algorithm"].unique()
df_pattern_corr["pert_type"].unique()
df_pattern_corr.columns
df_pattern_corr.query("pert_algorithm =='ic'").groupby(["sub", "test_dataset", "pert_type", "pert_ratio"])["id_acc_clipvision"].mean().reset_index()

df_pattern_corr.query("pert_algorithm == 'fgsm'")

subject_col_renamer = {"sub": "Subject"}
df_pattern_corr = df_pattern_corr.rename(columns=subject_col_renamer)
df_profile_corr = df_profile_corr.rename(columns=subject_col_renamer)
df_reconstruction = df_reconstruction.rename(columns=subject_col_renamer)

hue_order_subs = ["S1", "S2", "S3", "S4", "S5"]


"""
Additional Plots
Perturbation Process (Powerpoint)
Perturbation validation Plots are in clip_module_testing.py EXECUTE ON SMITH
after execution: rsync -r matt@smith:"/home/matt/programming/recon_diffuser/src/recon_diffuser/adv_pert/thesis_plots/*" . (in plots folder)

"""

def set_ticks_and_axes(ax, xticklabels, title, ylabel, sethline=False):
    ax.set_xticks(range(len(xticklabels)))  # Ensure it aligns with categorical data
    ax.set_xticklabels(xticklabels, size=16)
    ax.yaxis.get_label().set_fontsize(17)
    ax.tick_params(axis='x', labelrotation=45)
    ax.xaxis.set_tick_params(pad=-5)
    ax.set_title(title, size=18)
    ax.set_ylabel(ylabel, fontweight='bold')
    if sethline:
        ax.axhline(0.5, color='rosybrown', linestyle='-', linewidth=1, alpha=0.8)


test_dataset = "test"
algorithm_criterion = "80-20"
algorithm="fgsm"
algorithm_criterion = 0.03
df = df_pattern_corr.query(f"test_dataset == '{test_dataset}'").copy()

def prepare_pert_df(df, algorithm, algorithm_criterion, fgsm_0_baseline):
    if fgsm_0_baseline:
        df = df.query("pert_algorithm != 'NO_PERT'").copy()

        baseline_indexer = df['pert_epsilon'] == 0
        df['pert_epsilon'] = np.where(baseline_indexer, np.nan, df['pert_epsilon'])
        df['pert_type'] = np.where(baseline_indexer, np.nan, df['pert_type'])
        df['pert_algorithm'] = np.where(baseline_indexer, 'NO_PERT', df['pert_algorithm'])

    if algorithm == "ic":
        ic_criterion = algorithm_criterion
        df = df.query(f"pert_ratio == '{ic_criterion}' or pert_algorithm == 'NO_PERT' ").copy()
        df["pert_ratio"] = df["pert_ratio"].fillna("")
        df["pert_type"] = df["pert_type"].fillna("")

        df["condition"] = df["pert_algorithm"].astype(str) + " " + df["pert_ratio"].astype(str) + " " + df["pert_type"].astype(str)

    elif algorithm == "fgsm":
        fgsm_epsilon = algorithm_criterion
        df = df.query(f"pert_epsilon == {fgsm_epsilon} or pert_algorithm == 'NO_PERT' ").copy()
        df["pert_epsilon"] = df["pert_epsilon"].fillna("")
        df["pert_type"] = df["pert_type"].fillna("")
        df["condition"] = df["pert_algorithm"].astype(str) + " " + df["pert_epsilon"].astype(str) + " " + df["pert_type"].astype(str)
    else:
        raise NotImplemented(f"{algorithm=}")
    condition_renamer = {
        'NO_PERT  ': 'No perturbation',  
        f'ic {algorithm_criterion} friendly': f'IC {algorithm_criterion} friendly', 
        f'ic {algorithm_criterion} adversarial': f'IC {algorithm_criterion} adversarial', 
        f'fgsm {algorithm_criterion} friendly': f'fgsm {algorithm_criterion} friendly', 
        f'fgsm {algorithm_criterion} adversarial': f'fgsm {algorithm_criterion} adversarial'}
    df['condition'] = df['condition'].replace(condition_renamer)

    return df, condition_renamer

def decoder_advpert_plot(algorithm="ic", algorithm_criterion=None, fgsm_0_baseline=True):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True, sharex=True)
    for i,test_dataset in enumerate(['test', 'art']):
        ax = axes[i]
        df = df_pattern_corr.query(f"test_dataset == '{test_dataset}'").copy()
        df["pert_algorithm"].value_counts() # 250 NO_PERT

        df, condition_renamer = prepare_pert_df(df, algorithm, algorithm_criterion, fgsm_0_baseline=fgsm_0_baseline)

        criterion = 'id_acc_clipvision'

        order_values = [val for val in condition_renamer.values() if val in df["condition"].unique()]
        print(df.columns)
        sns.pointplot(
            data=df, x="condition", y=criterion, hue="Subject", hue_order=hue_order_subs,
            dodge=.3, linestyle="none", errorbar="se", markersize=5, markeredgewidth=3, ax=ax, order=order_values)
        test_dataset_renamer = {'test': 'Natural images', 'art': 'Artificial shapes'}
        title = f'{test_dataset_renamer[test_dataset]}'
        set_ticks_and_axes(ax, order_values, title, 'CLIP Vision ID Accuracy', True)
        ax.set_xlabel("Training Data", fontsize=15, fontweight='bold')
        if i == 1:
            ax.legend(title='Subject', loc='upper right')
        else:
            ax.legend().remove()

    fig.subplots_adjust(wspace=0.1, hspace=0)
    # fig.suptitle('Cliptext-Translator Performance Different Captions')
    fig.savefig(os.path.join(thesis_plots_path, f'advpert_translator_{algorithm}_{algorithm_criterion}.png'), bbox_inches='tight')

# decoder_advpert_plot("ic", "90-10")
decoder_advpert_plot("ic", "80-20")
# decoder_advpert_plot("ic", "70-30")
# decoder_advpert_plot("ic", "50-50")
decoder_advpert_plot("fgsm", 0.03)
# decoder_advpert_plot("fgsm", 0.1)
# decoder_advpert_plot("fgsm", 0.0)


def reconstruction_quality_plot_both_mixings(test_dataset, algorithm="ic", algorithm_criterion=None, fgsm_0_baseline=True):
    df = df_reconstruction.query(f"test_dataset == '{test_dataset}'").copy()
    df["pert_algorithm"] = df["pert_algorithm"].fillna("NO_PERT")

    df, condition_renamer = prepare_pert_df(df, algorithm, algorithm_criterion, fgsm_0_baseline)

    order_values = [val for val in condition_renamer.values() if val in df["condition"].unique()]

    fig, axes = plt.subplots(3, 1, figsize=(13, 7), sharex=True)
    column_renamer = {"PixCorr":"PixCorr", "dreamsim": "DreamSim", "clip_final": "CLIP-acc"}
    df = df.rename(columns=column_renamer)
    for row, criterion in enumerate(column_renamer.values()):
        ax = axes[row]
        sns.pointplot(
            data=df, x="condition", y=criterion, hue="Subject", hue_order=hue_order_subs,
            dodge=.2, linestyle="none", errorbar="se", markersize=5, markeredgewidth=3, ax=ax, order=order_values)
        xticklabels = order_values
        ax.set_xticks(range(len(xticklabels)))  # Ensure it aligns with categorical data
        ax.set_xticklabels(xticklabels, size=18.5)

        ax.set_xlabel('Training Data', fontweight='bold', fontsize=18.5)
        ax.set_ylabel(criterion, fontsize=18.5)
        if row == 0:
            # ax.legend().remove()
            ...
        else:
            ax.legend().remove()
    fig.savefig(os.path.join(thesis_plots_path, f'advpert_reconstruction_{test_dataset}_{algorithm}_{algorithm_criterion}.png'), dpi=300, bbox_inches='tight')


# reconstruction_quality_plot_both_mixings("test", "fgsm", 0.03, fgsm_0_baseline=False)
reconstruction_quality_plot_both_mixings("test", "ic", "80-20")
reconstruction_quality_plot_both_mixings("art", "ic", "80-20")

reconstruction_quality_plot_both_mixings("test", "fgsm", 0.03)
reconstruction_quality_plot_both_mixings("art", "fgsm", 0.03)


#### Discussion results

## What happens with increasing perturbation?

def pert_ratio_exploration_test_vs_art():
    caption_type = 'friendly'
    criterion = 'id_acc_clipvision'
    df_base = df_pattern_corr
    
    # Plot1
    test_dataset = 'test'
    df_translator_baseline = prepare_pert_df(df_base.query(f"test_dataset == '{test_dataset}'"), 'fgsm', 0.03, fgsm_0_baseline=True)[0].query('pert_algorithm == "NO_PERT"')

    dfp = df_base.query(f"pert_type == '{caption_type}'")

    df_translator_baseline['pert_ratio'] = '0'
    df_ic = dfp.query('pert_algorithm == "ic"')
    df_ic = pd.concat((df_translator_baseline, df_ic))
    pert_ratio_renamer = {'50-50':'50', '70-30': '30', '80-20': '20', '90-10': '10'}


    df = df_ic.query(f"test_dataset == 'test'")
    df = df.groupby(["Subject", 'pert_ratio'])[criterion].mean().reset_index()
    df['pert_ratio_numeric'] = pd.to_numeric(df['pert_ratio'].replace(pert_ratio_renamer))
    fig, axs = plt.subplots(1,2, figsize=(12,5))

    sns.pointplot(data=df, x='pert_ratio_numeric', y=criterion, ax = axs[0])
    axs[0].set_xlabel('pert ratio')
    axs[0].set_title('Natural Test Images')

    # Plot2
    test_dataset='art'
    df_translator_baseline = prepare_pert_df(df_base.query(f"test_dataset == '{test_dataset}'"), 'fgsm', 0.03, fgsm_0_baseline=True)[0].query('pert_algorithm == "NO_PERT"')

    dfp = df_base.query(f"pert_type == '{caption_type}'")

    df_translator_baseline['pert_ratio'] = '0'
    df_ic = dfp.query('pert_algorithm == "ic"')
    df_ic = pd.concat((df_translator_baseline, df_ic))
    pert_ratio_renamer = {'50-50':'50', '70-30': '30', '80-20': '20', '90-10': '10'}

    df = df_ic.query(f"test_dataset == 'art'")
    df = df.groupby(["Subject", 'pert_ratio'])[criterion].mean().reset_index()
    df['pert_ratio_numeric'] = pd.to_numeric(df['pert_ratio'].replace(pert_ratio_renamer))

    sns.pointplot(data=df, x='pert_ratio_numeric', y=criterion, ax = axs[1])
    axs[1].set_xlabel('pert ratio')
    axs[1].set_ylabel('')
    axs[1].set_title('Artificial shapes')

    fig.savefig(os.path.join(thesis_plots_path, 'advpert_discussion_explore_pert_ratio_test_vs_art'))

def pert_ratio_exploration_test():
    test_dataset = 'test'
    caption_type = 'adversarial'
    criterion1 = 'id_acc_clipvision'

    # plot 1
    df_base = df_pattern_corr
    df_base = df_base.rename(columns={'pert_type': 'caption_type'})
    # df_translator_baseline = prepare_pert_df(df_base.query(f"test_dataset == '{test_dataset}'"), 'fgsm', 0.03, fgsm_0_baseline=True)[0].query('pert_algorithm == "NO_PERT"')
    dfp = df_base
    # dfp = df_base.query(f"pert_type == '{caption_type}'")
    # df_translator_baseline['pert_ratio'] = '0'
    df_ic = dfp.query('pert_algorithm == "ic"')
    # df_ic = pd.concat((df_translator_baseline, df_ic))
    pert_ratio_renamer = {'50-50':'50', '70-30': '30', '80-20': '20', '90-10': '10'}

    df = df_ic.query(f"test_dataset == 'test'")
    df = df.groupby(["Subject", 'pert_ratio', 'caption_type'])[criterion1].mean().reset_index()
    df['pert_ratio_numeric'] = pd.to_numeric(df['pert_ratio'].replace(pert_ratio_renamer))
    fig, axs = plt.subplots(1,2, figsize=(12,5))
    sns.pointplot(data=df, x='pert_ratio_numeric', y=criterion1, ax = axs[0], hue='caption_type')
    axs[0].set_xlabel('Perturbation Criterion')
    axs[0].set_title('Translator')
    axs[0].set_ylabel('CLIP Vision ID Accuracy')
    xticks = [0,1, 2,3]
    xticklabels = ["90/10", "80/20", "70/30", "50/50"]
    axs[0].set_xticks(xticks)
    axs[0].set_xticklabels(xticklabels)
    

    # plot2
    criterion2 = 'dreamsim'

    df_base = df_reconstruction
    df_base = df_base.rename(columns={'pert_type': 'caption_type'})

    # df_translator_baseline = prepare_pert_df(df_base.query(f"test_dataset == '{test_dataset}'"), 'fgsm', 0.03, fgsm_0_baseline=True)[0].query('pert_algorithm == "NO_PERT"')
    dfp = df_base
    # dfp = df_base.query(f"caption_type == '{caption_type}'")
    # df_translator_baseline['pert_ratio'] = '0'
    df_ic = dfp.query('pert_algorithm == "ic"')
    # df_ic = pd.concat((df_translator_baseline, df_ic))
    pert_ratio_renamer = {'50-50':'50', '70-30': '30', '80-20': '20', '90-10': '10'}


    df = df_ic.query(f"test_dataset == 'test'")
    df = df.groupby(["Subject", 'pert_ratio', 'caption_type'])[criterion2].mean().reset_index()
    df['pert_ratio_numeric'] = pd.to_numeric(df['pert_ratio'].replace(pert_ratio_renamer))

    sns.pointplot(data=df, x='pert_ratio_numeric', y=criterion2, ax = axs[1], hue='caption_type')
    axs[1].set_xlabel('Perturbation Criterion')
    axs[1].set_ylabel('DreamSim')
    axs[1].set_title('Reconstruction')
    xticks = [0,1, 2,3]
    xticklabels = ["90/10", "80/20", "70/30", "50/50"]
    axs[1].set_xticks(xticks)
    axs[1].set_xticklabels(xticklabels)
    
    fig.savefig(os.path.join(thesis_plots_path, 'advpert_discussion_explore_pert_ratio_test_translator_and_recon'))

pert_ratio_exploration_test()


"""
Presentation plots

"""


"""
Was will ich hier?

Einen Plot nur
Performance von low-level vs high-level vs human

"""
presentation_plots_folder = "/Users/matt/ownCloud/gogo/MA/thesis/diversity_thesis/presentation_plots"

def presentation_plot_advpert(test_dataset, algorithm="ic", algorithm_criterion=None, fgsm_0_baseline=True):
    sns.set_context("talk")  # or "poster", "notebook", "paper"

    df = df_reconstruction.query(f"test_dataset == '{test_dataset}'").copy()
    df["pert_algorithm"] = df["pert_algorithm"].fillna("NO_PERT")

    df, condition_renamer = prepare_pert_df(df, algorithm, algorithm_criterion, fgsm_0_baseline)
    
    new_condition_renamer = {'IC 80-20 friendly':'friendly', 'IC 80-20 adversarial': 'adversarial', 'No perturbation': 'No perturbation'}
    # condition_renamer['ic 80-20 friendly'] = 'friendly'
    # condition_renamer['ic 80-20 adversarial'] = 'adversarial'

    df['condition'] = df['condition'].replace(new_condition_renamer)

    order_values = ['No perturbation', 'friendly', 'adversarial']

    title = "Natural test images" if test_dataset == "test" else "Artificial Shapes"

    fig, ax = plt.subplots(1,1)
    column_renamer = {"dreamsim": "DreamSim"}
    criterion = "DreamSim"

    df = df.rename(columns=column_renamer)
    sns.pointplot(
        data=df, x="condition", y=criterion, hue="Subject", hue_order=hue_order_subs,
        dodge=.2, linestyle="none", errorbar="se", markersize=5, markeredgewidth=3, ax=ax, order=order_values)
    xticklabels = order_values
    ax.set_xticks(range(len(xticklabels)))  # Ensure it aligns with categorical data
    ax.set_xticklabels(xticklabels)

    ax.set_xlabel('Training Data', fontweight='bold')
    ax.set_title(title, fontweight='bold')


    ax.set_ylabel(criterion)
    plt.legend(fontsize='x-small')  # options: 'x-small', 'small', 10, etc.
    plt.show()


    fig.savefig(os.path.join(presentation_plots_folder, f'exp3_{test_dataset}.png'), dpi=300, bbox_inches='tight')


presentation_plot_advpert("test", "ic", "80-20")
presentation_plot_advpert("art", "ic", "80-20")
