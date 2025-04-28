import pandas as pd
import os
import seaborn as sns
import sys
from collections import defaultdict
sys.path.append('/home/matt/programming/recon_diffuser/src/recon_diffuser/adv_pert')
sys.path.append('/home/matt/programming/recon_diffuser/src/recon_diffuser/evaluation')
from visualize_results import visualize_results

from evaluate_pertgen_log import extract_pert_gen_info, extract_pertgen_all_from_folder

from extract_regression_scores_from_logs import extract_all_logs_from_directory

# Regression performance
log_folder_regression = "/home/matt/programming/recon_diffuser/analysis/progress_presentation_2/regression_log"
# We have 24 pert regression + 6 baseline regressions. Thus df has 30 lines
df_regression = extract_all_logs_from_directory(log_folder_regression)

# Reconstruction performance
csv_folder_reconstruction = "/home/matt/programming/recon_diffuser/analysis/progress_presentation_2/recon_quantitative_results"
# We have 24 + 6 configs. 90 Test (test + art) images each. Thus df has 2700 lines
df_reconstruction = pd.concat([pd.read_csv(os.path.join(csv_folder_reconstruction, f)) for f in os.listdir(csv_folder_reconstruction) if f.endswith('.csv')])

df_reconstruction = df_reconstruction[df_reconstruction["sub"] == "AM"].reset_index(drop=True)
df_regression = df_regression[df_regression["sub"] == "AM"].reset_index(drop=True)

import re
dropout_pattern = re.compile(r"(?P<test_dataset>.*?)_(?P<dropout_algorithm>.*?)_(?P<dropout_ratio>.*?)_(?P<dropout_trial>.*)")
# Get the dropout data from the output name
df_reconstruction = pd.concat((df_reconstruction,pd.DataFrame(list(df_reconstruction["name"].apply(lambda row: dropout_pattern.search(row).groupdict() if dropout_pattern.match(row) else {})))), axis=1)

df_reconstruction["test_dataset"] = df_reconstruction["name"].apply(lambda n: "test" if "test" in n else "art")
df_reconstruction["test_dataset"].value_counts() 
df_reconstruction["dropout_ratio"] = df_reconstruction["dropout_ratio"].fillna(1)
df_reconstruction["dropout_algorithm"] = df_reconstruction["dropout_algorithm"].fillna("no_dropout")
df_reconstruction["dropout_trial"] = df_reconstruction["dropout_trial"].fillna(0)
df_reconstruction = df_reconstruction.apply(pd.to_numeric, errors="ignore")


df_regression["dropout_ratio"] = df_regression["dropout_ratio"].fillna(1)
df_regression["dropout_algorithm"] = df_regression["dropout_algorithm"].fillna("no_dropout")
df_regression["dropout_trial"] = df_regression["dropout_trial"].fillna(0)
df_regression = df_regression.apply(pd.to_numeric, errors="ignore")


df_regression_random_dropout = df_regression[df_regression['dropout_algorithm'].isin(['dropout-random', 'no_dropout'])]
# Regression score plot

# # Magnitudes test
# criterions = [f"vdvae_test", f"clipvision_test_mean_reg_score", f"cliptext_test_mean_reg_score"]
# df_regression_random_dropout["id"] = df_regression_random_dropout.index
# melted = pd.melt(df_regression_random_dropout, ["dropout_ratio", "dropout_trial"], value_vars=criterions)
# sns.lineplot(data = melted, x="dropout_ratio", y="value", hue="variable").set_title("Random dropout Regression Performance")

# # magnitudes art
# criterions = [f"vdvae_art", f"clipvision_art_mean_reg_score", f"cliptext_art_mean_reg_score"]
# df_regression_random_dropout["id"] = df_regression_random_dropout.index
# melted = pd.melt(df_regression_random_dropout, ["dropout_ratio", "dropout_trial"], value_vars=criterions)
# sns.lineplot(data = melted, x="dropout_ratio", y="value", hue="variable").set_title("Random dropout Regression Performance")

# # VDVAE combined
# criterions = [f"vdvae_test", f"vdvae_art", ]
# df_regression_random_dropout["id"] = df_regression_random_dropout.index
# melted = pd.melt(df_regression_random_dropout, ["dropout_ratio", "dropout_trial"], value_vars=criterions)
# sns.lineplot(data = melted, x="dropout_ratio", y="value", hue="variable").set_title("Random dropout Regression Performance")

# ## Clipvision combined

# criterions = ["clipvision_test_mean_reg_score", "clipvision_art_mean_reg_score"]
# df_regression_random_dropout["id"] = df_regression_random_dropout.index
# melted = pd.melt(df_regression_random_dropout, ["dropout_ratio", "dropout_trial"], value_vars=criterions)
# sns.lineplot(data = melted, x="dropout_ratio", y="value", hue="variable").set_title("Random dropout Regression Performance")


# # clipvision test
# criterions = ["clipvision_test_mean_reg_score"]
# df_regression_random_dropout["id"] = df_regression_random_dropout.index
# melted = pd.melt(df_regression_random_dropout, ["dropout_ratio", "dropout_trial"], value_vars=criterions)
# sns.lineplot(data = melted, x="dropout_ratio", y="value", hue="variable").set_title("Random dropout Regression Performance")

# # clipvision art
# criterions = ["clipvision_art_mean_reg_score"]
# df_regression["id"] = df_regression.index
# melted = pd.melt(df_regression, ["dropout_ratio", "dropout_trial"], value_vars=criterions)
# sns.lineplot(data = melted, x="dropout_ratio", y="value", hue="variable").set_title("Random dropout Regression Performance")

# # cliptext combined
# criterions = ["cliptext_test_mean_reg_score", "cliptext_art_mean_reg_score"]
# df_regression_random_dropout["id"] = df_regression_random_dropout.index
# melted = pd.melt(df_regression_random_dropout, ["dropout_ratio", "dropout_trial"], value_vars=criterions)
# sns.lineplot(data = melted, x="dropout_ratio", y="value", hue="variable").set_title("Random dropout Regression Performance")

# # cliptext test
# criterions = ["cliptext_test_mean_reg_score"]
# df_regression_random_dropout["id"] = df_regression_random_dropout.index
# melted = pd.melt(df_regression_random_dropout, ["dropout_ratio", "dropout_trial"], value_vars=criterions)
# sns.lineplot(data = melted, x="dropout_ratio", y="value", hue="variable").set_title("Random dropout Regression Performance")

# # cliptext art
# criterions = ["cliptext_art_mean_reg_score"]
# df_regression_random_dropout["id"] = df_regression_random_dropout.index
# melted = pd.melt(df_regression_random_dropout, ["dropout_ratio", "dropout_trial"], value_vars=criterions)
# sns.lineplot(data = melted, x="dropout_ratio", y="value", hue="variable").set_title("Random dropout Regression Performance")

def competition_criterion(df, baseline_name, condition_name, criterion, operator):
    condition_name = baseline_name + "_" + condition_name
    df_baseline = df[df["name"] == baseline_name]
    df_condition = df[df["name"] == condition_name]
    ims_baseline = df_baseline["im"]
    ims_condition = df_condition["im"]

    assert all(ims_baseline.values == ims_condition.values)

    if operator == ">":
        competitions_won = (df_condition[criterion].values > df_baseline[criterion].values).sum()
    elif operator == "<":
        competitions_won = (df_condition[criterion].values < df_baseline[criterion].values).sum()
    else:
        raise ValueError(f"Operator {operator} not supported.")

    competition_ratio = competitions_won / len(df_baseline)
    return competition_ratio


def all_competitions(df_reconstruction, baseline_name, condition_name, criterion, operator):
    max_dropout_trial_for_condition = df_reconstruction[df_reconstruction["name"].str.contains(condition_name)]['dropout_trial'].max()
    competition_results = []
    for n in range(max_dropout_trial_for_condition):
        c = competition_criterion(df_reconstruction, baseline_name = baseline_name, condition_name = f"{condition_name}_0{n}", criterion=criterion, operator=operator)
        competition_results.append(c)
    return competition_results


def get_all_competitions_for_conditions(conditions, df_reconstruction, baseline_name, criterion, operator):
    all_conditions_competition_results = defaultdict(dict)
    
    for condition_name in conditions:
        dropout_rate = float(condition_name[condition_name.index("_")+1:])
        condition_algorithm = condition_name[:condition_name.index("_")]
        all_conditions_competition_results[condition_algorithm][dropout_rate] = all_competitions(df_reconstruction, baseline_name, condition_name, criterion, operator)
    return all_conditions_competition_results

def competition_plot(df_reconstruction, conditions, baseline_name, criterion, operator):
    all_conditions_competition_results = get_all_competitions_for_conditions(conditions, df_reconstruction, baseline_name, criterion, operator)

    dfs = []
    for condition_algorithm, condition_data in all_conditions_competition_results.items():
        condition_df = pd.DataFrame(condition_data)
        condition_df['algorithm'] = condition_algorithm
        dfs.append(condition_df)
    df_conditions = pd.concat(dfs)

    melted = pd.melt(df_conditions, ["algorithm"])
    ax = sns.pointplot(data=melted, x="variable", y="value", hue="algorithm", linestyle="none")
    ax.set_title(f"{baseline_name}: {criterion} competition vs baseline")
    ax.set_ylabel("Competitions Won")
    ax.set_xlabel("Dropout Ratio")

# Lowlevel (PixCorr)
conditions = ["dropout-random_0.1", "dropout-random_0.25", "dropout-random_0.5",  "dropout-random_0.75"]
baseline_name = "test"
criterion, operator = "PixCorr", ">"
competition_plot(df_reconstruction, conditions, baseline_name, criterion, operator)

baseline_name = "art"
competition_plot(df_reconstruction, conditions, baseline_name, criterion, operator)

# Midlevel (Dreamsim)
baseline_name = "test"
criterion, operator = "dreamsim", "<"
competition_plot(df_reconstruction, conditions, baseline_name, criterion, operator)

baseline_name = "art"
competition_plot(df_reconstruction, conditions, baseline_name, criterion, operator)

# Highlevel (clip_final)
baseline_name = "test"
criterion, operator = "clip_final", ">"
competition_plot(df_reconstruction, conditions, baseline_name, criterion, operator)

baseline_name = "art"
competition_plot(df_reconstruction, conditions, baseline_name, criterion, operator)





# Adding the ssimgreedy algorithm

criterions = [f"vdvae_test"]
df_regression["id"] = df_regression.index
melted = pd.melt(df_regression, ["dropout_ratio", "dropout_trial", 'dropout_algorithm'], value_vars=criterions)
melted['dropout_algorithm'] = melted['dropout_algorithm'].replace({"no_dropout": "dropout-random"})
sns.lineplot(data = melted, x="dropout_ratio", y="value", hue="dropout_algorithm").set_title("Random dropout Regression Performance")


criterions = ["clipvision_test_mean_reg_score", "clipvision_art_mean_reg_score"]
df_regression["id"] = df_regression.index
melted = pd.melt(df_regression, ["dropout_ratio", "dropout_trial", 'dropout_algorithm'], value_vars=criterions)
melted['dropout_algorithm'] = melted['dropout_algorithm'].replace({"no_dropout": "dropout-random"})
sns.lineplot(data = melted, x="dropout_ratio", y="value", hue="dropout_algorithm").set_title("Random dropout Regression Performance")


# # clipvision test
criterions = ["clipvision_test_mean_reg_score"]
df_regression["id"] = df_regression.index
melted = pd.melt(df_regression, ["dropout_ratio", "dropout_trial", 'dropout_algorithm'], value_vars=criterions)
melted['dropout_algorithm'] = melted['dropout_algorithm'].replace({"no_dropout": "dropout-random"})
sns.lineplot(data = melted, x="dropout_ratio", y="value", hue="dropout_algorithm").set_title("Random dropout Regression Performance")

# # clipvision art
criterions = ["clipvision_art_mean_reg_score"]
df_regression["id"] = df_regression.index
melted = pd.melt(df_regression, ["dropout_ratio", "dropout_trial", 'dropout_algorithm'], value_vars=criterions)
melted['dropout_algorithm'] = melted['dropout_algorithm'].replace({"no_dropout": "dropout-random"})
sns.lineplot(data = melted, x="dropout_ratio", y="value", hue="dropout_algorithm").set_title("Random dropout Regression Performance")

# # cliptext combined
criterions = ["cliptext_test_mean_reg_score", "cliptext_art_mean_reg_score"]
df_regression["id"] = df_regression.index
melted = pd.melt(df_regression, ["dropout_ratio", "dropout_trial", 'dropout_algorithm'], value_vars=criterions)
melted['dropout_algorithm'] = melted['dropout_algorithm'].replace({"no_dropout": "dropout-random"})
sns.lineplot(data = melted, x="dropout_ratio", y="value", hue="dropout_algorithm").set_title("Random dropout Regression Performance")

# # cliptext test
criterions = ["cliptext_test_mean_reg_score"]
df_regression["id"] = df_regression.index
melted = pd.melt(df_regression, ["dropout_ratio", "dropout_trial", 'dropout_algorithm'], value_vars=criterions)
melted['dropout_algorithm'] = melted['dropout_algorithm'].replace({"no_dropout": "dropout-random"})
sns.lineplot(data = melted, x="dropout_ratio", y="value", hue="dropout_algorithm").set_title("Random dropout Regression Performance")

# # cliptext art
criterions = ["cliptext_art_mean_reg_score"]
df_regression["id"] = df_regression.index
melted = pd.melt(df_regression, ["dropout_ratio", "dropout_trial", 'dropout_algorithm'], value_vars=criterions)
melted['dropout_algorithm'] = melted['dropout_algorithm'].replace({"no_dropout": "dropout-random"})
sns.lineplot(data = melted, x="dropout_ratio", y="value", hue="dropout_algorithm").set_title("Random dropout Regression Performance")


# criteria = [("dreamsim", "<"), ("clip_final", "<"), ("PixCorr", ">")]
conditions = ["dropout-random_0.1", "dropout-random_0.25", "dropout-random_0.5", 
              "dropout-ssimgreedy_0.1", "dropout-ssimgreedy_0.25", "dropout-ssimgreedy_0.5",]

# Lowlevel (PixCorr)

baseline_name = "test"
criterion, operator = "PixCorr", ">"
competition_plot(df_reconstruction, conditions, baseline_name, criterion, operator)

baseline_name = "art"
competition_plot(df_reconstruction, conditions, baseline_name, criterion, operator)

# Midlevel (Dreamsim)
baseline_name = "test"
criterion, operator = "dreamsim", "<"
competition_plot(df_reconstruction, conditions, baseline_name, criterion, operator)

baseline_name = "art"
competition_plot(df_reconstruction, conditions, baseline_name, criterion, operator)

# Highlevel (clip_final)
baseline_name = "test"
criterion, operator = "clip_final", ">"
competition_plot(df_reconstruction, conditions, baseline_name, criterion, operator)

baseline_name = "art"
competition_plot(df_reconstruction, conditions, baseline_name, criterion, operator)

