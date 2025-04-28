import pandas as pd
import seaborn as sns
import os
import numpy as np
import matplotlib.pyplot as plt
import concurrent.futures
from PIL import Image
import pingouin as pg

sns.set_theme()
thesis_plots_path = "/Users/matt/ownCloud/gogo/MA/thesis/diversity_thesis/plots"

path_pattern_corr = "./data/df_pattern_correlation_aicap.csv"
path_profile_corr = "./data/df_profile_correlation_aicap.csv"
path_reconstruction = "./data/df_reconstruction_aicap.csv"

df_pattern_corr = pd.read_csv(path_pattern_corr).query("sub != 'JP'")
df_profile_corr = pd.read_csv(path_profile_corr).query("sub != 'JP'")
df_reconstruction = pd.read_csv(path_reconstruction).query("sub != 'JP'")

sub_renamer = {'TH':'S1', 'AM': 'S2', 'ES': 'S3', 'JK': 'S4', 'KS': 'S5'}
df_pattern_corr['sub'] = df_pattern_corr['sub'].replace(sub_renamer)
df_profile_corr['sub'] = df_profile_corr['sub'].replace(sub_renamer)
df_reconstruction['sub'] = df_reconstruction['sub'].replace(sub_renamer)

df_reconstruction['dreamsim'] = 1 - df_reconstruction["dreamsim"]
df_reconstruction['clip_final'] = df_reconstruction["clip_final"] - 0.5
df_reconstruction['clip_sim'] = 1 - df_reconstruction["clip_dist"]


df_pattern_corr = df_pattern_corr.rename(columns={"id_acc":"id_acc_icnn"})

# drop all rows where we have 'long' captions

df_pattern_corr = df_pattern_corr[~df_pattern_corr['prompt_name'].str.contains('_long')]
df_profile_corr = df_profile_corr[~df_profile_corr['prompt_name'].str.contains('_long')]
df_reconstruction = df_reconstruction[~df_reconstruction['prompt_name'].str.contains('_long')]

# Relevant Progress reports:

# First Captions
# https://www.notion.so/241220-Matthias-reducing-train-data-with-care-First-AI-captions-0a0a3c610dba4d9c84d4eda4fb1c53dd
# Results
# https://www.notion.so/250110-Matthias-AICaptions-Evaluation-053ef06c3e6e4858808c9266cc1bffc0
# Filling Gaps (Identification Accuracy, True Cliptext Features)
# https://www.notion.so/20250119-Matthias-Filling-The-gaps-2-a8f598e120de4b6ba4b20a7d0a2a9253#18e2cfd830264119bbfcd90d223b4cc0

subject_col_renamer = {"sub": "Subject"}
df_pattern_corr = df_pattern_corr.rename(columns=subject_col_renamer)
df_profile_corr = df_profile_corr.rename(columns=subject_col_renamer)
df_reconstruction = df_reconstruction.rename(columns=subject_col_renamer)

hue_order_subs = ["S1", "S2", "S3", "S4", "S5"]


### VALIDATION PLOTS

# in open_api.py
# First: AiCaption Samples
# aicap_caption_samples.png

"""
Additional Plots:
Example for different prompt types (Powerpoint)
"""


### DECODER PLOTS
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

#### Main results with mixing 0.4
def decoder_aicap_plot():
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True, sharex=True)
    for i,test_dataset in enumerate(['test', 'art']):
        ax = axes[i]
        df = df_pattern_corr.query(f"test_dataset == '{test_dataset}'")
        df = df.query("mixing == 0.4")
        df = df[~df['usetruecliptext']]
        df = df[~df['badshuffle']]

        criterion = 'id_acc_cliptext'
        condition_renamer = {'human_captions': 'human',  'low_level_short': 'AI low-level', 'high_level_short': 'AI high-level', 'human_captions_shuffled_single_caption': 'shuffled'}
        df['prompt_name'] = df['prompt_name'].replace(condition_renamer)
        sns.pointplot(
            data=df, x="prompt_name", y=criterion, hue="Subject", hue_order=hue_order_subs,
            dodge=.3, linestyle="none", errorbar="se", markersize=5, markeredgewidth=3, ax=ax, order=condition_renamer.values())
        test_dataset_renamer = {'test': 'Natural images', 'art': 'Artificial shapes'}
        title = f'{test_dataset_renamer[test_dataset]}'
        set_ticks_and_axes(ax, condition_renamer.values(), title, 'CLIP Text ID Accuracy', True)
        ax.set_xlabel("Caption Type", fontsize=15, fontweight='bold')
        # ax.axhline(0.5, color='rosybrown', linestyle='-', linewidth=1, alpha=0.8)
        # ax.set_title()
        if i == 1:
            ax.legend(title='Subject', loc='upper right')
        else:
            ax.legend().remove()

    fig.subplots_adjust(wspace=0.1, hspace=0)
    # fig.suptitle('Cliptext-Translator Performance Different Captions')
    fig.savefig(os.path.join(thesis_plots_path, f'aicap_translator.png'), bbox_inches='tight')
decoder_aicap_plot()

def decoder_quant():
    condition_renamer = {'human_captions': 'human',  'low_level_short': 'AI low-level', 'high_level_short': 'AI high-level', 'human_captions_shuffled_single_caption': 'shuffled'}

    for test_dataset in ['test', 'art']:
        df = df_pattern_corr.query("mixing == 0.4").copy()
        df = df.query(f"test_dataset == '{test_dataset}'").copy()
        # 1000 rows with test. Why?
        # 5 subjects * 4 prompt_names * 50 images = 1000 rows
        df['prompt_name'] = df['prompt_name'].replace(condition_renamer)
        quant_for_stats = df.groupby(["Subject", 'prompt_name'])['id_acc_cliptext'].mean().reset_index()
        quant_metrics = quant_for_stats.groupby('prompt_name')['id_acc_cliptext'].agg(['mean', 'std']).transpose()

        qq = quant_for_stats.melt(["Subject", 'prompt_name'], ['id_acc_cliptext'])
        print(f"Translator Results for {test_dataset} data.")
        print(quant_metrics)
        baselines = ['human', 'AI low-level']
        for prompt_name in condition_renamer.values():
            for baseline in baselines:
                if prompt_name == baseline:
                    continue
                s1 = qq.query(f"variable == 'id_acc_cliptext' and prompt_name == '{prompt_name}'").sort_values("Subject")['value'].values
                s2 = qq.query(f"variable == 'id_acc_cliptext' and prompt_name == '{baseline}'").sort_values("Subject")['value'].values
                ttest = pg.ttest(s1, s2, True)
                p = np.round(ttest['p-val'].item(), 3)
                print(f"{test_dataset}: {prompt_name} vs {baseline}: p= {p}")

decoder_quant()

## Reconstruction Quality

def reconstruction_quality_plot(test_dataset, mixing):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharex=True)
    df = df_reconstruction.query(f"test_dataset == '{test_dataset}'").copy()
    df = df.query(f"mixing == {mixing}")
    df = df[~df['usetruecliptext']]
    df = df[~df['badshuffle']]

    condition_renamer = {'human_captions': 'human',  'low_level_short': 'AI low-level', 'high_level_short': 'AI high-level', 'human_captions_shuffled_single_caption': 'shuffled'}
    df['prompt_name'] = df['prompt_name'].replace(condition_renamer)

    column_renamer = {"PixCorr":"PixCorr", "dreamsim": "dreamsim", "clip_final": "CLIP-acc"}
    df = df.rename(columns=column_renamer)

    for i, criterion in enumerate(column_renamer.values()):
        ax = axes[i]
        sns.pointplot(
            data=df, x="prompt_name", y=criterion, hue="Subject", hue_order=hue_order_subs,
            dodge=.3, linestyle="none", errorbar="se", markersize=5, markeredgewidth=3, ax=ax, order=condition_renamer.values())
        ax.set_title(criterion)
        ax.set(xlabel='Caption Type')
        if i == 0:
            ax.set(ylabel='Performance')
        else:
            ax.set(ylabel='')

        if i == 0:
            ax.legend(title='Subject', loc='upper left')
        else:
            ax.legend().remove()
        
    test_dataset_renamer = {'test': 'Natural images', 'art': 'Artificial shapes'}
    fig.suptitle(f'Reconstruction Quality {test_dataset_renamer[test_dataset]} cliptext-mixing {mixing}', fontweight='bold')

    fig.savefig(os.path.join(thesis_plots_path, f'aicap_reconstruction_{test_dataset}_mixing_{mixing}.png'), dpi=300)

# reconstruction_quality_plot('test', 0.4)
# reconstruction_quality_plot('art', 0.4)

# reconstruction_quality_plot('test', 0.8)
# reconstruction_quality_plot('art', 0.8)

def reconstruction_quality_plot_both_mixings(test_dataset):
    df = df_reconstruction.query(f"test_dataset == '{test_dataset}'").copy()
    df = df[~df['usetruecliptext']]
    df = df[~df['badshuffle']]
    df = df[df['mixing'].isin([0.4, 0.8])]
    
    name_renamer = {
        f'{test_dataset}_aicap_human_captions': 'human 0.4',
        f'{test_dataset}_aicap_human_captions-mix_0.8': 'human 0.8',
        f'{test_dataset}_aicap_low_level_short': 'AI low-level 0.4',
        f'{test_dataset}_aicap_low_level_short-mix_0.8': 'AI low-level 0.8',
        f'{test_dataset}_aicap_high_level_short': 'AI high-level 0.4',
        f'{test_dataset}_aicap_high_level_short-mix_0.8': 'AI high-level 0.8',
        f'{test_dataset}_aicap_human_captions_shuffled_single_caption': 'shuffled 0.4',
        f'{test_dataset}_aicap_human_captions_shuffled_single_caption-mix_0.8': 'shuffled 0.8',
    }
    df['prompt_name'] = df['name'].replace(name_renamer)

    fig, axes = plt.subplots(3, 1, figsize=(18, 5), sharex=True)
    column_renamer = {"PixCorr":"PixCorr", "dreamsim": "DreamSim", "clip_final": "CLIP-acc"}
    df = df.rename(columns=column_renamer)
    for row, criterion in enumerate(column_renamer.values()):
        ax = axes[row]
        sns.pointplot(
            data=df, x="prompt_name", y=criterion, hue="Subject", hue_order=hue_order_subs,
            dodge=.3, linestyle="none", errorbar="se", markersize=5, markeredgewidth=3, ax=ax, order=name_renamer.values())
        for xc in range(len(name_renamer.values())):
            if xc % 2 == 1:
                ax.axvline(x=xc+0.5, color='gray', linestyle='--', linewidth=2)
        xticklabels = [.4, .8, .4, .8, .4, .8, .4, .8]
        ax.set_xticks(range(len(xticklabels)))  # Ensure it aligns with categorical data
        ax.set_xticklabels(xticklabels, size=18.5)

        titles = ["Human captions", "Low-level AI captions", "High-level AI captions", "Shuffled captions",]
        if row == 0:
            ax.set_title("                  ".join(titles), fontsize=18.5)
        ax.set_xlabel('Mixing', fontweight='bold', fontsize=18.5)
        ax.set_ylabel(criterion, fontsize=18.5)
        if row == 0:
            # ax.legend(title='Subject', loc='upper left')
            # ax.legend().remove()
            ...
        else:
            ax.legend().remove()
    fig.savefig(os.path.join(thesis_plots_path, f'aicap_reconstruction_{test_dataset}_both_mixings.png'), dpi=300, bbox_inches='tight')

reconstruction_quality_plot_both_mixings('test')
reconstruction_quality_plot_both_mixings('art')

### DISCUSSION PLOTS

# Reconstruction Quant with: baseline, true shuffle, True Features

df = df_reconstruction.query('Subject == "S2"')
df.query('badshuffle')

df = df.query('test_dataset == "test"')
df.groupby('mixing')['name'].value_counts()

"""
Okay, die Storyline für die Discussion ist wie folgt:

1. Erstens: Reicht das Mixing überhaupt aus? Ab wann sehen wir wirklich Verbesserungen bzw. Verschlechterungen?
    a. Hier könnte man wirklich interessanterweise einen Plot machen, den ich leider noch berechnen muss. 
    Aber es wäre ein Plot mit drei verschiedenen Trajektorien. 
    1. Human captions (mixing 0-95)
    2. True Features (mixing 0-95)
    3. Shuffled captions (mixing 0-95)

    Das würde ich nur für dreamsim plotten. Einfach um zu sehen, wann passiert wirklich was mit den Bildern, das man überhaupt in den Metriken entdecken kann.
    Dann kommt hoffentlich raus, dass ab 0.8 oder so erst Unterschiede wirklich sichtbar werden.

    Gut also ich muss jetzt wohl doch noch ein paar conditions neu berechnen let's go. 

    HAHAHAHAHA deine Idee ist falsch.
    Wahrscheinlich hat Ozcelik das auch schon gemacht...

    Also: Wir haben schon den Probanden S2 genommen, der in dem Plot tatsächlich einen Unterschied hat.

    Wir wollen sehen: Wo wird dieser Unterschied größer, hat es einen Einfluss, wie die Mixing Ratio ist?

    Wir sehen in dem Plot (clipfinal, denn wir wollen ja insbesondere die semantische Qualität verbessern), dass schon bei einer kleineren Mixing Ratio die Performance maximiert ist. 
    Sie hat einfach nur einen extrem geringen Einfluss. 

    Es würde also keinen Sinn machen, die Mixing Ratio für die anderen captions noch weiter zu erhöhen, da nicht davon auszugehen ist, dass sich die Performance noch weiter verbessert.

    Außer wenn man noch viel näher an die True Features kommen würde. 

    Aber vielleicht zeige ich auch gleichzeitig den dreamsim Graphen. 
    Um zu zeigen: Selbst wenn man den True Features sehr nahe kommt würde das nichts bringen, weil man irgendwann wieder abfällt in der Performance. 
    Man muss also beides miteinander aufwiegen. So Wie Ozcelik es auch gemacht hat.
    Aber: Wir sehen, selbst mit Shuffled Features kommt man in einigen Fällen an die gleiche Performance. Also einen besonders großen Einfluss hat cliptext hier einfach nicht und es ist fraglich, ob es sich lohnt. 
"""

## Plots

# Main Question: Do the cliptext Features have ANY influence on the Reconstruction? If yes, at which mixing parameter does it start?


def download_reconstructed_images():

    host = 'matt@smith'
    opb = "/home/matt/programming/recon_diffuser/"

    results_path = 'results/versatile_diffusion/deeprecon/subjAM/'
    output_folder = '/Users/matt/programming/recon_diffuser/analysis/thesis_analysis/aicap_evolution'

    output_names = [
        'aicap_human_captions-mix_0.0',
        'aicap_human_captions-mix_0.25',
        'aicap_human_captions-mix_0.5',
        'aicap_human_captions-mix_0.75',
        'aicap_human_captions-mix_0.99',
        'aicap_human_captions-usetruecliptext_True-mix_0.0',
        'aicap_human_captions-usetruecliptext_True-mix_0.25',
        'aicap_human_captions-usetruecliptext_True-mix_0.5',
        'aicap_human_captions-usetruecliptext_True-mix_0.75',
        'aicap_human_captions-usetruecliptext_True-mix_0.99',
        'aicap_human_captions_shuffled_single_caption-mix_0.0',
        'aicap_human_captions_shuffled_single_caption-mix_0.25',
        'aicap_human_captions_shuffled_single_caption-mix_0.5',
        'aicap_human_captions_shuffled_single_caption-mix_0.75',
        'aicap_human_captions_shuffled_single_caption-mix_0.99'
        ]
    

    def scp_command(im_path, full_output_folder):
        os.system(f"scp {host}:{im_path} {full_output_folder}")

    im_num = 0
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        futures = []
        for test_dataset in ['test', 'art']:
            full_output_folder = os.path.join(output_folder, f"{test_dataset}_im_{im_num}")
            os.makedirs(full_output_folder, exist_ok=True)
            for output_name in output_names:
                full_im_name = f"{test_dataset}_{output_name}"
                im_path = os.path.join(opb, results_path, full_im_name, f"{im_num}.png")
                if 'aicap_human_captions-usetruecliptext_True-mix_0.5' in im_path:
                    ...
                else:
                    continue
                full_output_file = os.path.join(full_output_folder, f"{full_im_name}.png")
                futures.append(executor.submit(scp_command, im_path, full_output_file))
                print(f"Downloading {full_im_name}")
        concurrent.futures.wait(futures)

# download_reconstructed_images()

def reconstructed_qual_evolution_plot(im_num, test_dataset):
    true_img_folder = f'/Users/matt/programming/recon_diffuser/data/stimuli/deeprecon/{test_dataset}/'
    true_image_path = os.path.join(true_img_folder, f"{im_num}.png")
    true_image = Image.open(true_image_path)

    n_list = ['human_captions', 'human_captions-usetruecliptext_True', 'human_captions_shuffled_single_caption']
    m_list = ['0.0', '0.25', '0.5', '0.75', '0.99']
    
    n_names= ['Human captions', 'True Features', 'Shuffled']

    images = {}
    for alg in n_list:
        for ratio in m_list:
            # images[(alg, ratio)] = np.random.rand(100, 100)
            # if ratio == '0.5':
                # images[(alg, ratio)] = np.random.rand(100, 100)
            # else:
            images[(alg, ratio)] = Image.open(f'/Users/matt/programming/recon_diffuser/analysis/thesis_analysis/aicap_evolution/{test_dataset}_im_{im_num}/{test_dataset}_aicap_{alg}-mix_{ratio}.png') # Maybe correct

    n_rows = len(n_list)
    n_cols = len(m_list)
    fig, axs = plt.subplots(nrows=n_rows+1, 
                            ncols=n_cols, 
                            figsize=(12, 8))
    middle_col = n_cols // 2  # integer division to find middle index
    for col in range(n_cols):
        axs[0, col].axis('off')
    axs[0, middle_col].imshow(true_image, cmap='gray')
    axs[0, middle_col].set_title("True Image", fontsize=16)
    for row, alg in enumerate(n_list):
        for col, ratio in enumerate(m_list):
            ax = axs[row+1, col]
            ax.imshow(images[(alg, ratio)], cmap='gray')
            if col == 0:
                ax.set_ylabel(n_names[row], fontsize=15, rotation=0, labelpad=45)
            if row == len(n_list) - 1:
                ax.set_xlabel(xlabel=f"{ratio}", fontsize=16)
            ax.grid(False)
            ax.set(yticklabels=[])
            ax.set(xticklabels=[])
    fig.subplots_adjust(wspace=0, hspace=0.05)
    fig.text(0.5, 0.02, 'Mixing Ratio', ha='center', fontsize=16, fontweight='bold')
    plt.savefig(os.path.join(thesis_plots_path, f'aicap_reconstruction_evolution_{test_dataset}_{im_num}.JPEG'), bbox_inches='tight')
    plt.show()

reconstructed_qual_evolution_plot(0, 'test')
reconstructed_qual_evolution_plot(0, 'art')
# Quant Evolution Plot

def reconstruction_quant_evolution_plot(test_dataset):
    df = df_reconstruction.query(f"test_dataset == '{test_dataset}'").copy()
    df = df.query("Subject == 'S2'")
    output_names = [
        'aicap_human_captions-mix_0.0',
        'aicap_human_captions-mix_0.25',
        'aicap_human_captions-mix_0.5',
        'aicap_human_captions-mix_0.75',
        'aicap_human_captions-mix_0.99',
        'aicap_human_captions-usetruecliptext_True-mix_0.0',
        'aicap_human_captions-usetruecliptext_True-mix_0.25',
        'aicap_human_captions-usetruecliptext_True-mix_0.5',
        'aicap_human_captions-usetruecliptext_True-mix_0.75',
        'aicap_human_captions-usetruecliptext_True-mix_0.99',
        'aicap_human_captions_shuffled_single_caption-mix_0.0',
        'aicap_human_captions_shuffled_single_caption-mix_0.25',
        'aicap_human_captions_shuffled_single_caption-mix_0.5',
        'aicap_human_captions_shuffled_single_caption-mix_0.75',
        'aicap_human_captions_shuffled_single_caption-mix_0.99'
        ]
    
    full_output_names = [f"{test_dataset}_{output_name}" for output_name in output_names]
    df = df[df['name'].isin(full_output_names)]

    conditions = ['human_captions-mix', 'human_captions-usetruecliptext', 'human_captions_shuffled']
    condition_names = ['Human captions', 'True Features', 'Shuffled captions']
    df['condition_name'] = df['name'].apply(lambda x: next((condition_names[i] for i, condition in enumerate(conditions) if condition in x), 'Unknown'))

    column_renamer = {'clip_final': 'CLIP-acc','dreamsim': 'dreamsim', }
    df = df.rename(columns=column_renamer)

    fig, axes = plt.subplots(1, 2, figsize=(12,5))
    for i,metric in enumerate(column_renamer.values()):
        ax = axes[i]
        sns.lineplot(data=df, x='mixing', y=metric, hue='condition_name', ax=ax)
        # ax.set_title(f"Reconstruction Quality Evolution {test_dataset}", fontsize=16)
        if i == 0:
            ax.legend(title='Condition', loc='upper left')
            ax.set_xlabel("Mixing Ratio", fontsize=14)
        if i == 1:
            ax.set_xlabel("Mixing Ratio", fontsize=14)
            # disable legend
            ax.get_legend().remove()
        # ax.set_ylabel("CLIP-acc", fontsize=14)

    test_dataset_renamer = {'test': 'Natural images', 'art': 'Artificial Shapes'}
    
    fig.suptitle(f"Reconstruction Quality for different mixings ({test_dataset_renamer[test_dataset]})", fontsize=16)
    plt.savefig(os.path.join(thesis_plots_path, f'aicap_reconstruction_quant_evolution_{test_dataset}.JPEG'))

reconstruction_quant_evolution_plot('test')
reconstruction_quant_evolution_plot('art')




########### Presentation

presentation_plots_folder = "/Users/matt/ownCloud/gogo/MA/thesis/diversity_thesis/presentation_plots"

#### PLOT 1

def presentation_plot(test_dataset):
    sns.set_context("talk")  # or "poster", "notebook", "paper"
    df = df_reconstruction.query(f"test_dataset == '{test_dataset}'").copy()
    df = df[~df['usetruecliptext']]
    df = df[~df['badshuffle']]
    df = df[df['mixing'].isin([0.8])]

    title = "Natural test images" if test_dataset == "test" else "Artificial Shapes"

    name_renamer = {
        # f'{test_dataset}_aicap_human_captions-mix_0.8': 'human',
        f'{test_dataset}_aicap_low_level_short-mix_0.8': 'AI low-level',
        f'{test_dataset}_aicap_high_level_short-mix_0.8': 'AI high-level',
    }
    df['prompt_name'] = df['name'].replace(name_renamer)

    fig, ax = plt.subplots(1,1)
    column_renamer = {"dreamsim": "DreamSim"}
    criterion = "DreamSim"
    df = df.rename(columns=column_renamer)

    sns.pointplot(
        data=df, x="prompt_name", y=criterion, hue="Subject", hue_order=hue_order_subs,
        dodge=.3, linestyle="none", errorbar="se", markersize=5, markeredgewidth=3, ax=ax, order=name_renamer.values())
    ax.set(xlabel='Caption type')
    ax.set_title(title, fontweight='bold')
    plt.legend(fontsize='x-small')  # options: 'x-small', 'small', 10, etc.
    plt.show()
    fig.savefig(os.path.join(presentation_plots_folder, f'exp_2_{test_dataset}.png'), bbox_inches='tight')

presentation_plot("test")
presentation_plot("art")