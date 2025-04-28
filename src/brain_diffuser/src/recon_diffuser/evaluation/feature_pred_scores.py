
import numpy as np
import pandas as pd
import os
from datetime import datetime
from numba import njit, prange
from tqdm import tqdm
import time
def simple_log(l):
    print(f"{datetime.now()}: {l}")
# Flatten each layer and turn into numpy array of shape (n_im, n_layer, n_feature)
# -> doesn't work, because the feature vectors aren't the same size.

def get_layer_map():
    layer_mapping = {
        'conv1_1': 'features[0]',
        'conv1_2': 'features[2]',
        'conv2_1': 'features[5]',
        'conv2_2': 'features[7]',
        'conv3_1': 'features[10]',
        'conv3_2': 'features[12]',
        'conv3_3': 'features[14]',
        'conv3_4': 'features[16]',
        'conv4_1': 'features[19]',
        'conv4_2': 'features[21]',
        'conv4_3': 'features[23]',
        'conv4_4': 'features[25]',
        'conv5_1': 'features[28]',
        'conv5_2': 'features[30]',
        'conv5_3': 'features[32]',
        'conv5_4': 'features[34]',
        'fc6': 'classifier[0]',
        'relu6': 'classifier[1]',
        'fc7': 'classifier[3]',
        'relu7': 'classifier[4]',
        'fc8': 'classifier[6]'}

    return layer_mapping

def get_inv_layer_map():
    layer_mapping = get_layer_map()
    inv_layer_mapping = {v: k for k, v in layer_mapping.items()}
    return inv_layer_mapping

def get_profile_corr_generator_icnn(extracted_icnn, predicted_icnn):
    # Iterates the features and yields all images for them

    inv_layer_map = get_inv_layer_map()
    for feat_name in extracted_icnn.keys():
        feat_name_nice = inv_layer_map[feat_name]
        if feat_name_nice in ["relu6", "relu7"]:
            continue
        extracted_feat = extracted_icnn[feat_name]
        # flatten all but first dimension in extracted feat
        extracted_feat_flat = extracted_feat.reshape(extracted_feat.shape[0], np.prod(extracted_feat.shape[1:]))

        predicted_feat = np.stack(predicted_icnn[feat_name], axis=1)
        predicted_feat_flat = predicted_feat.reshape(predicted_feat.shape[0], np.prod(predicted_feat.shape[1:]))

        assert extracted_feat_flat.shape == predicted_feat_flat.shape

        # yields 19 times
        yield feat_name_nice, extracted_feat_flat, predicted_feat_flat

def get_pattern_corr_generator_icnn(extracted_icnn, predicted_icnn):
    # yields whatever we need. 
    # Iterates the images and gives all features for them
    total_ims = next(iter(extracted_icnn.values())).shape[0]
    all_feature_names = extracted_icnn.keys()
    layer_mapping = get_layer_map()
    inv_layer_map = get_inv_layer_map()
    relevant_nice_names = [inv_layer_map[feat_name] for feat_name in all_feature_names if inv_layer_map[feat_name] not in ["relu6", "relu7"]]

    for im_num in range(total_ims):
        # Gives all feature for the image
        extracted_feat = [extracted_icnn[feat_name][im_num].flatten() for feat_name in all_feature_names if inv_layer_map[feat_name] not in ["relu6", "relu7"]]
        pred_feat = []
        for feat_name_nice in relevant_nice_names:
            feat_name = layer_mapping[feat_name_nice]
            [predicted_icnn[feat_name]]
            pred_feat.append(np.array([f[im_num] for f in predicted_icnn[feat_name]]).flatten())
        assert all([(s1.shape== s2.shape) for s1,s2 in zip(extracted_feat, pred_feat)])
        # yields 40/50 times
        # yield relevant_nice_names, f
        yield im_num, relevant_nice_names, extracted_feat, pred_feat
    
def icnn_to_arrays_for_pairwise_id_acc(pattern_corr_generator_icnn):
    n_imgs = len(pattern_corr_generator_icnn)
    n_feature_values = sum([pattern_corr_generator_icnn[0][2][n].shape[0] for n in range(len(pattern_corr_generator_icnn[0][2]))])

    big_extracted = np.zeros((n_imgs, n_feature_values))
    big_predicted = np.zeros((n_imgs, n_feature_values))

    for (im_num, ft_names, extracted_feat, pred_feat) in pattern_corr_generator_icnn:
        extracted_feat_flat = np.concatenate(extracted_feat)
        predicted_feat_flat = np.concatenate(pred_feat)
        big_extracted[im_num] = extracted_feat_flat
        big_predicted[im_num] = predicted_feat_flat
    return big_extracted, big_predicted
    

def get_profile_corr_generator_vdvae(extracted_vdvae, predicted_vdvae):
    for feat_name in ["vdvae"]:
        yield feat_name, extracted_vdvae, predicted_vdvae

def get_pattern_corr_generator_vdvae(extracted_vdvae, predicted_vdvae):
    total_ims = extracted_vdvae.shape[0]
    for im_num in range(total_ims):
        yield im_num, ["vdvae"], [extracted_vdvae[im_num]], [predicted_vdvae[im_num]]

def get_profile_corr_generator_clip(extracted_clip, predicted_clip):
    cliptype = "cliptext" if extracted_clip.shape[1] == 77 else "clipvision"

    patch_name = "tokens" if cliptype == "cliptext" else "patches"
    feat_names = [f"{cliptype}_final", f"{cliptype}_{patch_name}"]

    result = []
    
    # Features of the final Layer (cliptext of clipvision)
    final_feat_index = -1 if cliptype == "cliptext" else 0
    result.append((feat_names[0], extracted_clip[:, final_feat_index, :], predicted_clip[:, final_feat_index, :]))

    # TODO: append all cliptext/clipvision features one by one (not the final one though)

    # Features of the patches/tokens layers
    slicer_concat_patches = slice(None,-1) if cliptype == "cliptext" else slice(1,None)

    extracted_concat = extracted_clip[:, slicer_concat_patches, :]
    predicted_concat = predicted_clip[:, slicer_concat_patches, :]

    result.append((feat_names[1], extracted_concat.reshape(extracted_concat.shape[0], np.prod(extracted_concat.shape[1:])), predicted_concat.reshape(predicted_concat.shape[0], np.prod(predicted_concat.shape[1:]))))
    return result

def get_pattern_corr_generator_clip(extracted_clip, predicted_clip):
    cliptype = "cliptext" if extracted_clip.shape[1] == 77 else "clipvision"
    total_ims = extracted_clip.shape[0]

    patch_name = "tokens" if cliptype == "cliptext" else "patches"

    # Do the concatenation of all clip_patches
    feat_names = [f"{cliptype}_final", f"{cliptype}_{patch_name}"]
    final_feat_index = -1 if cliptype == "cliptext" else 0
    
    for im_num in range(total_ims):
        extracted_final_feature = extracted_clip[im_num, final_feat_index, :]
        extracted_concat_patches = extracted_clip[im_num, :-1, :].flatten() if final_feat_index == -1 else extracted_clip[im_num, 1:, :].flatten()

        predicted_final_feature = predicted_clip[im_num, final_feat_index, :]
        predicted_concat_patches = predicted_clip[im_num, :-1, :].flatten() if final_feat_index == -1 else predicted_clip[im_num, 1:, :].flatten()

        yield im_num, feat_names, [extracted_final_feature, extracted_concat_patches], [predicted_final_feature, predicted_concat_patches]


@njit(parallel=True)
def profile_correlation_numba_nan_aware(y_true, y_pred):
    n_samples, n_features = y_true.shape
    correlations = np.empty(n_features, dtype=np.float64)

    for i in prange(n_features):
        # First pass: compute means ignoring NaNs
        sumTrue = 0.0
        sumPred = 0.0
        count_pairs = 0
        for j in range(n_samples):
            Y = y_true[j, i]
            P = y_pred[j, i]
            if not (np.isnan(Y) or np.isnan(P)):
                sumTrue += Y
                sumPred += P
                count_pairs += 1

        if count_pairs == 0:
            # No valid pairs, correlation is NaN
            correlations[i] = np.nan
            continue

        y_true_mean = sumTrue / count_pairs
        y_pred_mean = sumPred / count_pairs

        # Second pass: compute numerator and denominator for correlation
        sum_num = 0.0
        sum_true_sq = 0.0
        sum_pred_sq = 0.0
        for j in range(n_samples):
            Y = y_true[j, i]
            P = y_pred[j, i]
            if not (np.isnan(Y) or np.isnan(P)):
                Yc = Y - y_true_mean
                Pc = P - y_pred_mean
                sum_num += Yc * Pc
                sum_true_sq += Yc * Yc
                sum_pred_sq += Pc * Pc

        denom = np.sqrt(sum_true_sq * sum_pred_sq)
        if denom == 0:
            # If standard deviations are zero or no variability
            correlations[i] = np.nan
        else:
            correlations[i] = sum_num / denom

    # Compute the mean of correlations ignoring NaNs
    # Equivalent to np.nanmean, but done manually
    sum_corr = 0.0
    count_valid = 0
    for i in range(n_features):
        c = correlations[i]
        if not np.isnan(c):
            sum_corr += c
            count_valid += 1

    if count_valid == 0:
        return np.nan  # if no valid correlations
    else:
        return sum_corr / count_valid

def compute_pairwise_identification_accuracy_to_df(extracted, predicted):
    pairwise_won = compute_pairwise_identification_accuracy_numba(extracted.copy(), predicted.copy())
    df = pd.DataFrame({"id_acc" : pairwise_won})
    return df

@njit
def cosine_distance(a, b):
    dot = 0.0
    norm_a = 0.0
    norm_b = 0.0
    for i in range(a.size):
        dot += a[i] * b[i]
        norm_a += a[i] * a[i]
        norm_b += b[i] * b[i]
    if norm_a == 0.0 or norm_b == 0.0:
        return 1.0  # Define distance as maximum if one vector is zero
    return 1.0 - (dot / (np.sqrt(norm_a) * np.sqrt(norm_b)))

@njit
def euclidean_distance(a, b):
    distance_sq = 0.0
    for i in range(a.size):
        diff = a[i] - b[i]
        distance_sq += diff * diff
    return np.sqrt(distance_sq)

@njit(parallel=True)
def compute_pairwise_identification_accuracy_numba(extracted, predicted):
    n_imgs = extracted.shape[0]
    extracted_reshaped = extracted.reshape(n_imgs, -1)
    predicted_reshaped = predicted.reshape(n_imgs, -1)
    pairwise_won = np.zeros(n_imgs, dtype=np.int64)
    for i in prange(n_imgs):
        dist_predicted = cosine_distance(predicted_reshaped[i], extracted_reshaped[i])
        n_won = 0
        for j in range(n_imgs):
            if j == i:
                continue
            dist_paired = cosine_distance(predicted_reshaped[j], extracted_reshaped[i])
            if dist_predicted < dist_paired:
                n_won += 1
        pairwise_won[i] = n_won
    return pairwise_won


def compute_profile_corrs(profile_corr_generator):
    simple_log("Computing profile corrs...")
    t1_numba = time.time()
    profile_corr_collector = {}
    for feat_name, feat_true, feat_pred in profile_corr_generator:
        profile_correlation = profile_correlation_numba_nan_aware(feat_true, feat_pred)
        profile_corr_collector[feat_name] = {"profile_corr": profile_correlation}
    print(f"numba took {time.time() - t1_numba} {profile_corr_collector}")

    simple_log("Done computing profile_corrs")
    return pd.DataFrame(profile_corr_collector).transpose()

def compute_pattern_corrs(pattern_corr_generator):
    pattern_corr_collector = {}
    for im_num, feat_names, feat_true, feat_pred in pattern_corr_generator:
        corrs = [np.corrcoef(true, pred) for true, pred in zip(feat_true, feat_pred)]
        corrs_shapes_correct = [c.shape == (2,2) for c in corrs]
        cors_values = [c[0,1] for c in corrs]
        assert all(corrs_shapes_correct)
        pattern_corr_collector[im_num] = {feat_name: corr for feat_name, corr in zip(feat_names, cors_values)}
    return pd.DataFrame(pattern_corr_collector).transpose()

def compute_profile_pattern_main(out_path_base, dataset, test_name, sub, output_name, algorithm):
    simple_log(f"Starting profile pattern main in: {out_path_base}, {dataset}, {test_name}, {sub}, {output_name}, {algorithm}")
    if algorithm == "bd":
        return compute_profile_pattern_bd(out_path_base, dataset, test_name, sub, output_name)
    elif algorithm == "icnn":
        return compute_profile_pattern_icnn(out_path_base, dataset, test_name, sub, output_name)
    simple_log("Finished.")


def compute_profile_pattern_bd(out_path_base, dataset, test_name, sub, output_name):
    if output_name.startswith("ic") or output_name.startswith("fgsm"):
        # Computing the data for adversarial perturbations. 
        # cliptext and vdvae are not created for these.
        advpert_trial = True
    else:
        advpert_trial = False

    if "aicap" in output_name:
        # VDVAE is not computed for aicaps
        aicap_trial = True
    else:
        aicap_trial = False

    if "dropout" in output_name:
        # Everything is computed for dropout lol
        dropout_trial = True
    else:
        dropout_trial = False

    # Extracted features, don't depend on trial. Shouldn't even depend on subject, but we do have some redundant data here.
    vdvae_extracted_path = os.path.join(out_path_base, "data", "extracted_features", dataset, f"subj{sub}", f"vdvae_features_{test_name}_31l.npz")
    cliptext_extracted_path = os.path.join(out_path_base, "data", "extracted_features", dataset, f"subj{sub}", f"cliptext_{test_name}.npy")
    clipvision_extracted_path = os.path.join(out_path_base, "data", "extracted_features", dataset, f"subj{sub}", f"clipvision_{test_name}.npy")

    # VDVAE is computed for dropout
    vdvae_pred_n = f"vdvae_general_pred_{test_name}_{output_name}_sub{sub}_31l_alpha50k.npz" if (dropout_trial) else f"vdvae_general_pred_{test_name}_sub{sub}_31l_alpha50k.npz"

    vdvae_predicted_path = os.path.join(out_path_base, "data", "predicted_features", dataset, f"subj{sub}", vdvae_pred_n)


    # cliptext is computed for aicap and dropout
    cliptext_pred_n = f"cliptext_pred{test_name}_{output_name}_general.npy" if (aicap_trial or dropout_trial) else f"cliptext_pred{test_name}_general.npy"

    cliptext_predicted_path = os.path.join(out_path_base, "data", "predicted_features", dataset, f"subj{sub}", cliptext_pred_n)
    
    # clipvision is computed for advpert and dropout
    clipvision_pred_n = f"clipvision_pred{test_name}_{output_name}_general.npy" if (advpert_trial or dropout_trial) else f"clipvision_pred{test_name}_general.npy"
    clipvision_predicted_path = os.path.join(out_path_base, "data", "predicted_features", dataset, f"subj{sub}", clipvision_pred_n)


    assert all([os.path.exists(vdvae_extracted_path), os.path.exists(vdvae_predicted_path), os.path.exists(cliptext_extracted_path), os.path.exists(cliptext_predicted_path), os.path.exists(clipvision_extracted_path), os.path.exists(clipvision_predicted_path)])

    # do the profile/pattern computation

    extracted_vdvae = np.load(vdvae_extracted_path)
    predicted_vdvae = np.load(vdvae_predicted_path)

    extracted_vdvae = extracted_vdvae["latents"]
    predicted_vdvae = predicted_vdvae["pred_latents"]
    assert extracted_vdvae.shape == predicted_vdvae.shape # == (50 x 91168)

    # vdvae_layer_cumsum = [   16,    32,   288,   544,   800,  1056,  2080,  3104,  4128,
    #     5152,  6176,  7200,  8224,  9248, 13344, 17440, 21536, 25632,
    #    29728, 33824, 37920, 42016, 46112, 50208, 54304, 58400, 62496,
    #    66592, 70688, 74784, 91168]
    # take the first n layers of the vdvae for the correlation, since adding the others adds too much noise on the whole vector
    # n_layers_vdvae = 31
    # extracted_vdvae = extracted_vdvae[:, :vdvae_layer_cumsum[n_layers_vdvae]]
    # predicted_vdvae = predicted_vdvae[:, :vdvae_layer_cumsum[n_layers_vdvae]]

    extracted_cliptext = np.load(cliptext_extracted_path)
    predicted_cliptext = np.load(cliptext_predicted_path)
    assert extracted_cliptext.shape == predicted_cliptext.shape # (50, 77, 768)

    extracted_clipvision = np.load(clipvision_extracted_path)
    predicted_clipvision = np.load(clipvision_predicted_path)
    assert extracted_clipvision.shape == predicted_clipvision.shape # (50, 257, 768)

    profile_corr_generator_vdvae = list(get_profile_corr_generator_vdvae(extracted_vdvae, predicted_vdvae))
    pattern_corr_generator_vdvae = list(get_pattern_corr_generator_vdvae(extracted_vdvae, predicted_vdvae))

    profile_corr_generator_cliptext = list(get_profile_corr_generator_clip(extracted_cliptext, predicted_cliptext))
    pattern_corr_generator_cliptext = list(get_pattern_corr_generator_clip(extracted_cliptext, predicted_cliptext))
    
    profile_corr_generator_clipvision = list(get_profile_corr_generator_clip(extracted_clipvision, predicted_clipvision))
    pattern_corr_generator_clipvision = list(get_pattern_corr_generator_clip(extracted_clipvision, predicted_clipvision))

    # return profile_corr_generator_cliptext
    profile_corrs_vdvae = compute_profile_corrs(profile_corr_generator_vdvae)
    profile_corrs_cliptext = compute_profile_corrs(profile_corr_generator_cliptext)
    profile_corrs_clipvision = compute_profile_corrs(profile_corr_generator_clipvision)

    pattern_corrs_vdvae = compute_pattern_corrs(pattern_corr_generator_vdvae)
    pattern_corrs_cliptext = compute_pattern_corrs(pattern_corr_generator_cliptext)
    pattern_corrs_clipvision = compute_pattern_corrs(pattern_corr_generator_clipvision)

    pairwise_id_acc_vdvae = compute_pairwise_identification_accuracy_to_df(extracted_vdvae, predicted_vdvae)
    pairwise_id_acc_cliptext = compute_pairwise_identification_accuracy_to_df(extracted_cliptext, predicted_cliptext)
    pairwise_id_acc_clipvision = compute_pairwise_identification_accuracy_to_df(extracted_clipvision, predicted_clipvision)

    n_test_ims = extracted_vdvae.shape[0]
    print('vdvae', (pairwise_id_acc_vdvae/(n_test_ims-1)).mean())
    print('cliptext', (pairwise_id_acc_cliptext/(n_test_ims-1)).mean())
    print('clipvision', (pairwise_id_acc_clipvision/(n_test_ims-1)).mean())

    final_folder = f"{test_name}" if not output_name else f"{test_name}_{output_name}"
    out_dir_results = os.path.join(out_path_base, "results", "versatile_diffusion", dataset, f"subj{sub}", final_folder, )

    # folder must exist
    assert os.path.exists(out_dir_results)
    profile_corrs_vdvae.to_csv(os.path.join(out_dir_results, f"vdvae_profile_corr.csv"))
    profile_corrs_cliptext.to_csv(os.path.join(out_dir_results, f"cliptext_profile_corr.csv"))
    profile_corrs_clipvision.to_csv(os.path.join(out_dir_results, f"clipvision_profile_corr.csv"))

    pattern_corrs_vdvae.to_csv(os.path.join(out_dir_results, f"vdvae_pattern_corr.csv"))
    pattern_corrs_cliptext.to_csv(os.path.join(out_dir_results, f"cliptext_pattern_corr.csv"))
    pattern_corrs_clipvision.to_csv(os.path.join(out_dir_results, f"clipvision_pattern_corr.csv"))

    pairwise_id_acc_vdvae.to_csv(os.path.join(out_dir_results, f"vdvae_pair_id_acc.csv"))
    pairwise_id_acc_cliptext.to_csv(os.path.join(out_dir_results, f"cliptext_pair_id_acc.csv"))
    pairwise_id_acc_clipvision.to_csv(os.path.join(out_dir_results, f"clipvision_pair_id_acc.csv"))


def compute_profile_pattern_icnn(out_path_base, dataset, test_name, sub, output_name):
    icnn_extracted_path = os.path.join(out_path_base, "data", "extracted_features", dataset, f"subj{sub}", f"{test_name}_icnn_extracted_features.npy")
    icnn_predicted_path = os.path.join(out_path_base, "data", "predicted_features", dataset, f"subj{sub}", f"pred_icnn-features_{output_name}.pkl")
    assert all([os.path.exists(icnn_extracted_path), os.path.exists(icnn_predicted_path)])

    simple_log("Loading extracted icnn features...")
    extracted_icnn = np.load(icnn_extracted_path, allow_pickle=True)
    simple_log("Loading predicted icnn features...")
    predicted_icnn = np.load(icnn_predicted_path, allow_pickle=True)
    simple_log("Done")

    extracted_icnn = extracted_icnn.item()
    predicted_icnn = predicted_icnn[test_name]

    assert extracted_icnn.keys() == predicted_icnn.keys() # names of 19 feature layers (non-human friendly yet)

    profile_corr_generator_icnn = list(get_profile_corr_generator_icnn(extracted_icnn, predicted_icnn))
    pattern_corr_generator_icnn = list(get_pattern_corr_generator_icnn(extracted_icnn, predicted_icnn))

    profile_corrs_icnn = compute_profile_corrs(profile_corr_generator_icnn)
    pattern_corrs_icnn = compute_pattern_corrs(pattern_corr_generator_icnn)
    simple_log("Computing pairwise identification accuracy")
    big_extracted, big_predicted = icnn_to_arrays_for_pairwise_id_acc(pattern_corr_generator_icnn)
    pairwise_id_acc_icnn = compute_pairwise_identification_accuracy_to_df(big_extracted, big_predicted)
    simple_log("Done")

    final_folder = f"{test_name}" if not output_name else f"{test_name}_{output_name}"

    out_dir_results = os.path.join(out_path_base, "results", "icnn", dataset, f"subj{sub}", final_folder) 
    assert os.path.exists(out_dir_results)

    profile_corrs_icnn.to_csv(os.path.join(out_dir_results, "icnn_profile_corr.csv"))
    pattern_corrs_icnn.to_csv(os.path.join(out_dir_results, "icnn_pattern_corr.csv"))
    pairwise_id_acc_icnn.to_csv(os.path.join(out_dir_results, "icnn_pair_id_acc.csv"))

if __name__ == "__main__":
    out_path_base = '/home/matt/programming/recon_diffuser/'
    dataset= "deeprecon"
    test_name= "test"
    sub="AM"

    algorithm='bd'
    output_name= ''
    pc_corr_generator1 = compute_profile_pattern_main(out_path_base, dataset, test_name, sub, output_name, algorithm)

    # algorithm='bd'
    # output_name= 'dropout-random_0.1_00'
    # pc_corr_generator1 = compute_profile_pattern_main(out_path_base, dataset, test_name, sub, output_name, algorithm)
    
    # output_name= 'dropout-random_0.1_01'
    # pc_corr_generator2 = compute_profile_pattern_main(out_path_base, dataset, test_name, sub, output_name, algorithm)

    # algorithm = "icnn"
    # output_name = 'dropout-random_0.1_00_size224_iter500_scaled'
    # pc_corr_generator3 = compute_profile_pattern_main(out_path_base, dataset, test_name, sub, output_name, algorithm)

    # output_name = 'dropout-random_0.1_01_size224_iter500_scaled'
    # pc_corr_generator3 = compute_profile_pattern_main(out_path_base, dataset, test_name, sub, output_name, algorithm)
