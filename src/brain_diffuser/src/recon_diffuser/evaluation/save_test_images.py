import numpy as np
import os
from PIL import Image
from tqdm import tqdm


def extract_test_images(dataset, out_path_base):
   if dataset != 'nsd':
      raise NotImplementedError(f"test images for dataset {dataset} cannot be extracted yet. Sorry!")
   #The same for all subjects
   images = np.load(os.path.join(out_path_base, 'data/processed_data/nsd/subj1/test_stim_sub1.npy'))

   test_images_dir = os.path.join(out_path_base, 'data/stimuli/nsd/test/')

   if not os.path.exists(test_images_dir):
      os.makedirs(test_images_dir)
   for i in tqdm(range(len(images)), desc="Extracting test images...", total=len(images)):
      im = Image.fromarray(images[i].astype(np.uint8))
      im.save('{}/{}.png'.format(test_images_dir,i))


if __name__ == "__main__":
   out_path_base = "/home/matt/programming/recon_diffuser/"
   dataset="nsd"

   extract_test_images(dataset, out_path_base)

   ...