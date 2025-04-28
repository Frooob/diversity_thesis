import pandas as pd
import os
import seaborn as sns
import sys
sys.path.append('/home/matt/programming/recon_diffuser/src/recon_diffuser/adv_pert')
sys.path.append('/home/matt/programming/recon_diffuser/src/recon_diffuser/evaluation')
from visualize_results import visualize_results

from evaluate_pertgen_log import extract_pert_gen_info, extract_pertgen_all_from_folder

from extract_regression_scores_from_logs import extract_all_logs_from_directory


# Perturbation Generation
log_folder_pertgen = "/home/matt/programming/recon_diffuser/analysis/progress_presentation_1/pertgen_log"
# We have 24 log files, each created 6000 images. Thus the total pertgen df is 144000 lines.
df_pertgen = extract_pertgen_all_from_folder(log_folder_pertgen)

# Regression performance
log_folder_regression = "/home/matt/programming/recon_diffuser/analysis/progress_presentation_1/regression_log"
# We have 24 pert regression + 6 baseline regressions. Thus df has 30 lines
df_regression = extract_all_logs_from_directory(log_folder_regression)

# Reconstruction performance
csv_folder_reconstruction = "/home/matt/programming/recon_diffuser/analysis/progress_presentation_1/recon_quantitative_results"
# We have 24 + 6 configs. 90 Test (test + art) images each. Thus df has 2700 lines
df_reconstruction = pd.concat([pd.read_csv(os.path.join(csv_folder_reconstruction, f)) for f in os.listdir(csv_folder_reconstruction) if f.endswith('.csv')])
df_reconstruction["inputavg"] = df_reconstruction["name"].str.endswith("noinputavg")
df_reconstruction["name"] = df_reconstruction["name"].str.replace("__noinputavg","").str.replace("_noinputavg","")


# Plot 1: Quantitative Plot SSIM adversarial vs friendly
sns.boxplot(data = df_pertgen[df_pertgen['pert_type'] == 'ic'], x = 'caption_type', y = 'ssim').set_title('SSIM for friendly vs adversarial perturbation')

# Qualitative Plot recon GT vs baseline vs friendly vs adversarial
opb = "/home/matt/programming/recon_diffuser/"
visualize_results(
    opb, "deeprecon", "AM", "test", 
    ["test", "test_ic_adversarial_90-10_500_5","test_ic_friendly_90-10_500_5"], 
    include_vdvae=False)


visualize_results(
    opb, "deeprecon", "AM", "test", 
    ["test", "test_ic_adversarial_90-10_500_5","test_ic_friendly_90-10_500_5"], start_img_idx=10,
    include_vdvae=False)

# Quantitative Plot: dreamsim baseline vs friendly vs adversarial
replacer = {'art_ic_adversarial_90-10_500_5': 'adversarial', 'art_ic_friendly_90-10_500_5': 'friendly', 'art': 'baseline'}
selected = df_reconstruction[df_reconstruction['name'].isin(['art', 'art_ic_adversarial_90-10_500_5', 'art_ic_friendly_90-10_500_5'])].replace(replacer)
sns.boxplot(data=selected, x='name', y='dreamsim').set_title("INPUT AVG, ARTIFICIAL IMAGES")


replacer = {'test_ic_adversarial_90-10_500_5': 'adversarial', 'test_ic_friendly_90-10_500_5': 'friendly', 'test': 'baseline'}
selected = df_reconstruction[df_reconstruction['name'].isin(['test', 'test_ic_adversarial_90-10_500_5', 'test_ic_friendly_90-10_500_5'])].replace(replacer)
selected.groupby('name').mean()
sns.boxplot(data=selected, x='name', y='dreamsim').set_title("Input AVG, TEST IMAGES")

# Extra analysis noinputavg

##### TEST
replacer = {'test_ic_adversarial_90-10_500_5_noinputavg': 'adversarial', 'test_ic_friendly_90-10_500_5_noinputavg': 'friendly', 'test__noinputavg': 'baseline'}
selected = df_reconstruction[df_reconstruction['name'].isin(['test__noinputavg', 'test_ic_adversarial_90-10_500_5_noinputavg', 'test_ic_friendly_90-10_500_5_noinputavg'])].replace(replacer)
selected.groupby('name').mean()
sns.boxplot(data=selected, x='name', y='dreamsim').set_title("NO Input AVG, TEST IMAGES")


##### ART
replacer = {'art_ic_adversarial_90-10_500_5_noinputavg': 'adversarial', 'art_ic_friendly_90-10_500_5_noinputavg': 'friendly', 'art__noinputavg': 'baseline'}
selected = df_reconstruction[df_reconstruction['name'].isin(['art__noinputavg', 'art_ic_adversarial_90-10_500_5_noinputavg', 'art_ic_friendly_90-10_500_5_noinputavg'])].replace(replacer)
selected.groupby('name').mean()
sns.boxplot(data=selected, x='name', y='dreamsim').set_title("NO Input AVG, ARTIFICIAL IMAGES")


df_noinputavg = df_reconstruction[df_reconstruction["name"].str.endswith("noinputavg")]

df_noinputavg_test = df_noinputavg[df_noinputavg["name"].str.contains("test")]
df_noinputavg_art = df_noinputavg[df_noinputavg["name"].str.contains("art")]
sns.boxplot(data=df_noinputavg_test, y='name', x='dreamsim')
sns.boxplot(data=df_noinputavg_art, y='name', x='dreamsim')


# compare noinputavg vs inputavg

sns.boxplot(data=df_reconstruction[df_reconstruction["name"].str.contains("test")], y='name', x='dreamsim', hue="inputavg")


sns.boxplot(data=df_reconstruction[df_reconstruction["name"].str.contains("art")], y='name', x='dreamsim', hue="inputavg")


sns.boxplot(data=df_reconstruction[df_reconstruction["name"].str.contains("art") & ~df_reconstruction["inputavg"]], y='name', x='dreamsim')




