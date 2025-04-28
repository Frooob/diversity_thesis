print("moin")
import os
from PIL import Image

from eval_utils import simple_log

THINGS_TEST_MASTER_ORDERING = [
    'candelabra_14s','pacifier_14s','mango_13s','cookie_15s','blind_19s','pear_13s','butterfly_16n','cow_16n',
    'kimono_14s','beer_14s','cufflink_16s','crayon_15s','coat_rack_13s','bean_13s','footprint_15s','wasp_15n',
    'spoon_13n','headlamp_14s','television_14n','umbrella_13s','axe_14n','piano_13n','microscope_14n','seesaw_13s',
    'hula_hoop_16s','helicopter_25s','graffiti_17s','watch_13s','sim_card_13s','dragonfly_13s','wig_13s','donut_15s',
    'iguana_20s','horse_17s','drawer_13s','speaker_16s','bench_16n','hippopotamus_16s','pan_14s','bulldozer_22n',
    'chipmunk_16n','bobsled_14s','t-shirt_13s','monkey_18n','earring_16n','urinal_13s','starfish_15n','cheese_14n',
    'bike_14s','whip_14s','beaver_13n','shredder_13s','banana_13s','alligator_14n','ribbon_13s','clipboard_14s',
    'altar_13s','crank_16s','grate_13s','jam_15s','pumpkin_21n','horseshoe_13n','typewriter_13s','streetlight_13s',
    'peach_14n','joystick_14s','quill_15s','kazoo_14s','boa_13s','jar_14s','dress_13s','mousetrap_14s','ashtray_14n',
    'mosquito_net_13s','tamale_16s','marshmallow_13s','brace_14s','boat_14n','tent_14n','bed_20n','rabbit_13n',
    'key_17s','bamboo_13s','ferris_wheel_20s','lemonade_14s','drain_16s','fudge_14s','lasagna_13s','grape_20s',
    'stalagmite_14s','guacamole_18s','wallpaper_13s','chest1_14s','dough_19s','uniform_14s','easel_15s',
    'hovercraft_13n','beachball_16s','brownie_14s','nest_13s'
]


def move_and_rename_imgs(out_path_base, img_root, dataset_name, name):
    if name == 'test':
        correct_stim_order = THINGS_TEST_MASTER_ORDERING
    else:
        raise NotImplementedError(f"{name} not supported.")
    
    file_renamer = {im_id:f"{n}.png" for n,im_id in enumerate(correct_stim_order)}

    images_in_img_root = os.listdir(img_root)

    for im_idx in correct_stim_order:
        suffix = '.jpg'
        fname = f"{im_idx}{suffix}"
        if fname not in images_in_img_root:
            raise ValueError(f"image {fname} doesn't exist in img root. CRITICAL!")
        
        fpath = os.path.join(img_root, fname)
        # open the image
        img = Image.open(fpath)
        img = img.resize((500,500))

        # out path 
        out_dir = os.path.join(out_path_base, f'data/stimuli/{dataset_name}/{name}')
        os.makedirs(out_dir, exist_ok=True)

        out_path = os.path.join(out_dir, file_renamer[im_idx])
        # save as png
        img.save(out_path)
        simple_log(f'saved {fname} to {out_path}')

if __name__ == "__main__":
    image_data_root = "/home/kiss/data/contents_shared"
    things_images_root = "/home/kiss/data/contents_shared/THINGS-fMRI1/source"

    out_path_base = "/home/matt/programming/recon_diffuser/"
    move_and_rename_imgs(out_path_base, things_images_root, "things", 'test')
