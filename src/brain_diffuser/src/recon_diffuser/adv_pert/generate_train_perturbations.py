from datetime import datetime
import logging
from clip_module_testing import perturbations_all_images, out_name_from_perturb_params

logger = logging.getLogger("recdi_genpert")

if __name__ == "__main__":
    sub = "AM"
    out_path_base = "/home/matt/programming/recon_diffuser/"
    dataset="deeprecon"
    input_name = "train"

    # perturb_kinds: 'iterative_criterion', 'fgsm'

    perturb_params_iterative_criterion = {
        'perturb_kind': 'iterativeCriterion',
        'caption_type': 'friendly',
        'criterion': '50-50',
        'n_perts': 5,
        'max_pert_steps': 500
    }

    perturb_params_fgsm = {
        'perturb_kind' : 'fgsm',
        'caption_type' : 'friendly',
        'epsilon' : 0.01,
        'n_perts': 5,
    }

    perturb_params = perturb_params_fgsm

    out_name = out_name_from_perturb_params(perturb_params)

    fname_log = f"{datetime.now()}_train_pert_{sub}_{dataset}_{input_name}_{out_name}.log"
    
    logging.basicConfig(
        format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=[
            logging.FileHandler(fname_log),
            logging.StreamHandler()
        ],
        level=logging.INFO)
    logger.info(f"Let's go {fname_log}")

    perturbations_all_images(out_path_base, dataset, sub, input_name, perturb_params, collect_imgs = False)
    