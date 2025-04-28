import numpy as np
import os

np.random.seed(0)

def get_all_input_imgs(out_path_base, dataset, im_suffix):
    input_img_folder = os.path.join(out_path_base, 'data/train_data', dataset)

    all_input_imgs = [f for f in os.listdir(input_img_folder) if f.endswith(im_suffix)]
    # strip the file ending
    all_input_filepaths = [os.path.join(input_img_folder, x) for x in all_input_imgs]
    all_input_imgs = [os.path.basename(x).split(".")[0] for x in all_input_imgs]
    return all_input_imgs, all_input_filepaths


def random_dropout_sample(out_path_base, dataset, sample_size, sample_n, im_suffix):
    # takes a random sample of the input images and saves it accordingly. 
    # sample size should be a float between 0 and 1.
    # sample n is just an integer for the file descriptor, such that we can have multiple samples with the same size. 

    all_input_imgs,_ = get_all_input_imgs(out_path_base, dataset, im_suffix)
    # take a random sample from the image indexes

    num_samples = int(np.round(sample_size * len(all_input_imgs)))

    sample = np.random.choice(all_input_imgs, num_samples, replace=False)

    path_dropout_samples = os.path.join(out_path_base, f'data/dropout_samples/{dataset}/')
    os.makedirs(path_dropout_samples, exist_ok=True)

    sample_filename = f"dropout-random_{sample_size}_{sample_n:0>2}"
    
    np.save(os.path.join(path_dropout_samples, sample_filename),sample)

def main_deeprecon():
    out_path_base = "/home/matt/programming/recon_diffuser/"
    dataset = "deeprecon"
    im_suffix = "JPEG"

    sample_sizes = [0.75, 0.5, 0.25, 0.1]
    n_samples = 9

    for sample_size in sample_sizes:
        for sample_n in range(n_samples):
            random_dropout_sample(out_path_base, dataset, sample_size, sample_n, im_suffix)


if __name__ == "__main__":
    main_deeprecon()
    ...