from visualize_results import visualize_results

if __name__ == "__main__":
    opb = "/home/matt/programming/recon_diffuser/"
    ############# THESIS PLOTS #################

    thesis_plot_folder = "/home/matt/programming/recon_diffuser/analysis/thesis_analysis/plots/qualitative"
    thesis_plot_base_settings = {
        "set_save_folder":thesis_plot_folder,
        "row_label_size":25,
        "save_dpi":100,
        "show_img":False,
        "save_format":"JPEG"}
    ## BASELINE
    # True Features TEST
    visualize_results(
        opb, "deeprecon", "AM", "test", 
        ['test_true_bd', 'icnn:test_true_icnn_size224_iter500_scaled'], 
        include_vdvae=True, 
        caption_names=['VDVAE','Brain-Diffuser', 'iCNN'],
        title=" ",
        true_features=True, 
        set_save_name="baseline_qual_true_recon_test",
        **thesis_plot_base_settings
        )

    # True Features ART
    visualize_results(
        opb, "deeprecon", "AM", "art", 
        ['art_true_bd', 'icnn:art_true_icnn_size224_iter500_scaled'], 
        include_vdvae=True, 
        caption_names=['VDVAE','Brain-Diffuser','iCNN'],
        title=" ",
        true_features=True, 
        set_save_name="baseline_qual_true_recon_art",
        **thesis_plot_base_settings
        )

    # Baseline result TEST
    visualize_results(
        opb, "deeprecon", "AM", "test", 
        ["test", 'icnn:test__size224_iter500_scaled'], 
        caption_names= ["VDVAE", "Brain-Diffuser", "iCNN"],
        title=" ",
        include_vdvae=True,
        set_save_name="baseline_qual_recon_test",
        **thesis_plot_base_settings
        )
    
    # Baseline result ART
    visualize_results(
        opb, "deeprecon", "AM", "art", 
        ["art", 'icnn:art__size224_iter500_scaled'], 
        caption_names= ["VDVAE", "Brain-Diffuser", "iCNN"],
        title=" ",
        include_vdvae=True,
        set_save_name="baseline_qual_recon_art",
        **thesis_plot_base_settings
        )
    

    ## DROPOUT

    # Random dropout
    
    visualize_results(
        opb, "deeprecon", "AM", "test", 
        ["test", "test_dropout-random_0.5_00","test_dropout-random_0.1_00", 'icnn:test__size224_iter500_scaled', "icnn:test_dropout-random_0.5_00_size224_iter500_scaled", "icnn:test_dropout-random_0.1_00_size224_iter500_scaled"], 
        caption_names = ['bd All data', 'bd 0.5 data', 'bd 0.1 data', 'icnn All data', 'icnn 0.5 data', 'icnn 0.1 data'],
        title=" ",
        include_vdvae=False, set_save_name='dropout_qual_random_test',
        **thesis_plot_base_settings)

    visualize_results(
        opb, "deeprecon", "AM", "art", 
        ["art", "art_dropout-random_0.5_00","art_dropout-random_0.1_00", 'icnn:art__size224_iter500_scaled', "icnn:art_dropout-random_0.5_00_size224_iter500_scaled", "icnn:art_dropout-random_0.1_00_size224_iter500_scaled"], 
        caption_names = ['bd All data', 'bd 0.5 data', 'bd 0.1 data', 'icnn All data', 'icnn 0.5 data', 'icnn 0.1 data'],
        title=" ",
        include_vdvae=False, set_save_name='dropout_qual_random_art',
        **thesis_plot_base_settings)
    
    # Dropout Eval
    # Brain-Diffuser
    visualize_results(
    opb, "deeprecon", "AM", "test", 
    ['test_dropout-random_0.25_33', 'test_dropout-pixels_0.25_44','test_dropout-clipvision_0.25_88', "test_dropout-dreamsim_0.25_55"], 
    caption_names = ['Random 0.25', 'Pixels 0.25', 'Clipvision 0.25', 'dreamsim 0.25'],
    title=" ",
    include_vdvae=False, set_save_name='dropout_qual_eval_bd_test', 
    **thesis_plot_base_settings)

    visualize_results(
    opb, "deeprecon", "AM", "art", 
    ['art_dropout-random_0.25_33', 'art_dropout-pixels_0.25_44','art_dropout-clipvision_0.25_88', "art_dropout-dreamsim_0.25_55"], 
    caption_names = ['Random 0.25', 'Pixels 0.25', 'Clipvision 0.25', 'dreamsim 0.25'],
    title=" ",
    include_vdvae=False, set_save_name='dropout_qual_eval_bd_art',
    **thesis_plot_base_settings)

    # icnn
    smaller_row_label_size_settings = thesis_plot_base_settings.copy()
    smaller_row_label_size_settings['row_label_size'] = 23
    visualize_results(
    opb, "deeprecon", "AM", "test", 
    ['icnn:test_dropout-random_0.25_33_size224_iter500_scaled', 'icnn:test_dropout-pixels_0.25_44_size224_iter500_scaled','icnn:test_dropout-clipvision_0.25_88_size224_iter500_scaled', "icnn:test_dropout-dreamsim_0.25_55_size224_iter500_scaled"], 
    caption_names = ['Random 0.25', 'Pixels 0.25', 'Clipvision 0.25', 'dreamsim 0.25'],
    title=" ",
    include_vdvae=False, set_save_name='dropout_qual_eval_ICNN_test', 
    **smaller_row_label_size_settings)

    visualize_results(
    opb, "deeprecon", "AM", "art", 
    ['icnn:art_dropout-random_0.25_33_size224_iter500_scaled', 'icnn:art_dropout-pixels_0.25_44_size224_iter500_scaled','icnn:art_dropout-clipvision_0.25_88_size224_iter500_scaled', "icnn:art_dropout-dreamsim_0.25_55_size224_iter500_scaled"], 
    caption_names = ['Random 0.25', 'Pixels 0.25', 'Clipvision 0.25', 'dreamsim 0.25'],
    title=" ",
    include_vdvae=False, set_save_name='dropout_qual_eval_ICNN_art',
    **smaller_row_label_size_settings)


    ## Add heterogeneous vs monotone images for bd and icnn
    visualize_results(
    opb, "deeprecon", "AM", "test", 
    ['icnn:test_dropout-quantizedCountBoring_0.25_00_size224_iter500_scaled', 
     'icnn:test_dropout-quantizedCountParty_0.25_00_size224_iter500_scaled',
     'test_dropout-quantizedCountBoring_0.25_00', 
    "test_dropout-quantizedCountParty_0.25_00"], 
    caption_names = ['Mono iCNN 0.25', 'Hetero iCNN 0.25', 'Mono bd 0.25', 'Hetero bd 0.25'], 
    title=" ",
    include_vdvae=False, 
    set_save_name='dropout_discussion_test',
    **smaller_row_label_size_settings)

    visualize_results(
    opb, "deeprecon", "AM", "art", 
    ['icnn:art_dropout-quantizedCountBoring_0.25_00_size224_iter500_scaled', 
     'icnn:art_dropout-quantizedCountParty_0.25_00_size224_iter500_scaled',
     'art_dropout-quantizedCountBoring_0.25_00', 
    "art_dropout-quantizedCountParty_0.25_00"], 
    caption_names = ['Mono iCNN 0.25', 'Hetero iCNN 0.25', 'Mono bd 0.25', 'Hetero bd 0.25'], 
    title=" ",
    include_vdvae=False, set_save_name='dropout_discussion_art',
    **smaller_row_label_size_settings)


    ## AICAP
    # Main results(low-level captions)
    visualize_results(
    opb, "deeprecon", "AM", "test", 
    ['test_aicap_human_captions', 'test_aicap_human_captions-mix_0.8', "test_aicap_low_level_short","test_aicap_low_level_short-mix_0.8"], 
    caption_names = ['baseline 0.4', 'baseline 0.8', 'AI low-level 0.4', 'AI low-level 0.8'],
    title=" ",
    include_vdvae=False, set_save_name='aicap_qual_test', 
    **thesis_plot_base_settings)

    visualize_results(
    opb, "deeprecon", "AM", "art", 
    ['art_aicap_human_captions', 'art_aicap_human_captions-mix_0.8', "art_aicap_low_level_short","art_aicap_low_level_short-mix_0.8"], 
    caption_names = ['baseline 0.4', 'baseline 0.8', 'AI low-level 0.4', 'AI low-level 0.8'],
    title=" ",
    include_vdvae=False, set_save_name='aicap_qual_art', 
    **thesis_plot_base_settings)

    # high level captions for the appendix
    visualize_results(
    opb, "deeprecon", "AM", "test", 
    ['test_aicap_human_captions', 'test_aicap_human_captions-mix_0.8', "test_aicap_high_level_short","test_aicap_high_level_short-mix_0.8"], 
    caption_names = ['baseline 0.4', 'baseline 0.8', 'AI high-level 0.4', 'AI high-level 0.8'],
    title=" ",
    include_vdvae=False, set_save_name='aicap_qual_test_highlevel_appendix', 
    **thesis_plot_base_settings)

    visualize_results(
    opb, "deeprecon", "AM", "art", 
    ['art_aicap_human_captions', 'art_aicap_human_captions-mix_0.8', "art_aicap_high_level_short","art_aicap_high_level_short-mix_0.8"], 
    caption_names = ['baseline 0.4', 'baseline 0.8', 'AI high-level 0.4', 'AI high-level 0.8'],
    title=" ",
    include_vdvae=False, set_save_name='aicap_qual_art_highlevel_appendix', 
    **thesis_plot_base_settings)

    ## ADVPERt
    visualize_results(
        opb, "deeprecon", "AM", "test", 
        ["test", "test_ic_friendly_80-20_500_5", "test_ic_adversarial_80-20_500_5", "test_fgsm_adversarial_0.03_5", "test_fgsm_adversarial_0.03_5"], 
        caption_names=['baseline', "IC 80/20 friendly", "IC 80/20 adv ", "fgsm 0.03 friendly", "fgsm 0.03 adv"],
        title=' ',
        set_save_name='advpert_qual_test',
        include_vdvae=False,
        **thesis_plot_base_settings)

    visualize_results(
        opb, "deeprecon", "AM", "art", 
        ["art", "art_ic_friendly_80-20_500_5", "art_ic_adversarial_80-20_500_5", "art_fgsm_adversarial_0.03_5", "art_fgsm_adversarial_0.03_5"], 
        caption_names=['baseline', "IC 80/20 friendly", "IC 80/20 adv ", "fgsm 0.03 friendly", "fgsm 0.03 adv"],
        title=' ',
        set_save_name='advpert_qual_test',
        include_vdvae=False,
        **thesis_plot_base_settings
        )


    ############# PRESENTATION PLOTS #################

    pres_plot_base_settings = {'title': " ", 'include_vdvae': False, 'num_images': 5, 'start_img_idx': 24}
    pres_plot_art_base_settings = {**pres_plot_base_settings.copy(), "sample_indices": [9,13,5,2,34]} 
    pres_plot_base_settings['sample_indices'] = [4, 6, 16, 24, 37]

    cont_indices = [20,21,22,23,25]

    ##### Exp 0 - Baseline
    visualize_results(
        opb, "deeprecon", "AM", "test", 
        ["test"], 
        caption_names= ["Brain-Diffuser"],
        set_save_name="exp0_qual_test",
        **pres_plot_base_settings,
        **thesis_plot_base_settings
        )
    
    # Baseline result ART
    visualize_results(
        opb, "deeprecon", "AM", "art", 
        ["art"], 
        caption_names= ["Brain-Diffuser"],
        set_save_name="exp0_qual_art",
        **pres_plot_art_base_settings,
        **thesis_plot_base_settings
        )

    ##### Exp 1 - Dropout

    # Test Brain-Diffuser
    visualize_results(
    opb, "deeprecon", "AM", "test", 
    ['test_dropout-random_0.25_33', "test_dropout-dreamsim_0.25_55"], 
    caption_names = ['Random', 'Diversity'], 
    set_save_name='exp1_qual_main_bd_test',
    **pres_plot_base_settings,
    **thesis_plot_base_settings)

    # Art Brain-Diffuser
    visualize_results(
    opb, "deeprecon", "AM", "art", 
    ['art_dropout-random_0.25_33', "art_dropout-dreamsim_0.25_55"], 
    caption_names = ['Random', 'Diversity'],
    set_save_name='exp1_qual_main_bd_art',
    **pres_plot_art_base_settings,
    **thesis_plot_base_settings)


    # Test iCNN
    visualize_results(
    opb, "deeprecon", "AM", "test", 
    ['icnn:test_dropout-random_0.25_33_size224_iter500_scaled', "icnn:test_dropout-dreamsim_0.25_55_size224_iter500_scaled"], 
    caption_names = ['Random', 'Diversity'],
    set_save_name='exp1_qual_main_icnn_test', 
    **pres_plot_base_settings,
    **thesis_plot_base_settings)

    # Art iCNN
    visualize_results(
    opb, "deeprecon", "AM", "art", 
    ['icnn:art_dropout-random_0.25_33_size224_iter500_scaled', "icnn:art_dropout-dreamsim_0.25_55_size224_iter500_scaled"], 
    caption_names = ['Random', 'Diversity'],
    set_save_name='exp1_qual_main_icnn_art',
    **pres_plot_art_base_settings,
    **thesis_plot_base_settings)


    # Hetero Test Brain-Diffuser
    visualize_results(
    opb, "deeprecon", "AM", "test", 
    ['test_dropout-quantizedCountBoring_0.25_00', 
    "test_dropout-quantizedCountParty_0.25_00"], 
    caption_names = ['Monotone', 'Heterogeneous'], 
    set_save_name='exp1_qual_hetero_bd_test',
    **pres_plot_base_settings,
    **thesis_plot_base_settings)

    # Hetero Art Brain-Diffuser
    visualize_results(
    opb, "deeprecon", "AM", "art", 
    ['art_dropout-quantizedCountBoring_0.25_00', 
    "art_dropout-quantizedCountParty_0.25_00"], 
    caption_names = ['Monotone', 'Heterogeneous'], 
    set_save_name='exp1_qual_hetero_bd_art',
    **pres_plot_art_base_settings,
    **thesis_plot_base_settings)


    # Hetero Test iCNN

    visualize_results(
    opb, "deeprecon", "AM", "test", 
    ['icnn:test_dropout-quantizedCountBoring_0.25_00_size224_iter500_scaled', 
     'icnn:test_dropout-quantizedCountParty_0.25_00_size224_iter500_scaled'], 
    caption_names = ['Monotone', 'Heterogeneous'], 
    set_save_name='exp1_qual_hetero_icnn_test',
    **pres_plot_base_settings,
    **thesis_plot_base_settings)

    # Hetero Art iCNN
    visualize_results(
    opb, "deeprecon", "AM", "art", 
    ['icnn:art_dropout-quantizedCountBoring_0.25_00_size224_iter500_scaled', 
     'icnn:art_dropout-quantizedCountParty_0.25_00_size224_iter500_scaled'], 
    caption_names = ['Monotone', 'Heterogeneous'], 
    set_save_name='exp1_qual_hetero_icnn_art',
    **pres_plot_art_base_settings,
    **thesis_plot_base_settings)


    ##### Exp 2 - AI cap
    visualize_results(
    opb, "deeprecon", "AM", "test", 
    [ "test_aicap_high_level_short-mix_0.8", "test_aicap_low_level_short-mix_0.8", 'test_aicap_human_captions-mix_0.8', ], 
    caption_names = ['High-level', 'Low-level', 'Human'],
    set_save_name='exp2_test', 
    **pres_plot_base_settings,
    **thesis_plot_base_settings)

    visualize_results(
    opb, "deeprecon", "AM", "art", 
    ['art_aicap_high_level_short-mix_0.8',"art_aicap_low_level_short-mix_0.8", 'art_aicap_human_captions-mix_0.8', ], 
    caption_names = ['High-level', 'Low-level', 'Human'],
    set_save_name='exp2_art', 
    **pres_plot_art_base_settings,
    **thesis_plot_base_settings)


    ### Cont images

    # cont_images_settings = pres_plot_art_base_settings.copy()
    # cont_images_settings['sample_indices'] = [4, 6, 16, 24, 37]
    # visualize_results(
    # opb, "deeprecon", "AM", "art", 
    # ['art_aicap_high_level_short-mix_0.8',"art_aicap_low_level_short-mix_0.8", 'art_aicap_human_captions-mix_0.8', ], 
    # caption_names = ['High-level', 'Low-level', 'Human'],
    # set_save_name='exp2_art_cont', 
    # **cont_images_settings,
    # **thesis_plot_base_settings)

    ##### Exp 3 - Adv pert
    
    # Test
    visualize_results(
    opb, "deeprecon", "AM", "test", 
    ["test", "test_ic_friendly_80-20_500_5", "test_ic_adversarial_80-20_500_5"], 
    caption_names=['baseline', "friendly", "adversarial "],
    set_save_name='exp3_qual_test',
    **pres_plot_base_settings,
    **thesis_plot_base_settings)

    # Art
    visualize_results(
    opb, "deeprecon", "AM", "art", 
    ["art", "art_ic_friendly_80-20_500_5", "art_ic_adversarial_80-20_500_5"], 
    caption_names=['baseline', "friendly", "adversarial "],
    set_save_name='exp3_qual_art',
    **pres_plot_base_settings,
    **thesis_plot_base_settings
    )


    ########## Cont Plots


    cont_images_settings = pres_plot_art_base_settings.copy()
    cont_images_settings['sample_indices'] = cont_indices


    # Test Brain-Diffuser
    visualize_results(
    opb, "deeprecon", "AM", "test", 
    ['test_dropout-random_0.25_33', "test_dropout-dreamsim_0.25_55"], 
    caption_names = ['Random', 'Diversity'], 
    set_save_name='exp1_qual_main_bd_test_cont',
    **cont_images_settings,
    **thesis_plot_base_settings)

    # Art Brain-Diffuser
    visualize_results(
    opb, "deeprecon", "AM", "art", 
    ['art_dropout-random_0.25_33', "art_dropout-dreamsim_0.25_55"], 
    caption_names = ['Random', 'Diversity'],
    set_save_name='exp1_qual_main_bd_art_cont',
    **cont_images_settings,
    **thesis_plot_base_settings)


    # Test iCNN
    visualize_results(
    opb, "deeprecon", "AM", "test", 
    ['icnn:test_dropout-random_0.25_33_size224_iter500_scaled', "icnn:test_dropout-dreamsim_0.25_55_size224_iter500_scaled"], 
    caption_names = ['Random', 'Diversity'],
    set_save_name='exp1_qual_main_icnn_test_cont', 
    **cont_images_settings,
    **thesis_plot_base_settings)

    # Art iCNN
    visualize_results(
    opb, "deeprecon", "AM", "art", 
    ['icnn:art_dropout-random_0.25_33_size224_iter500_scaled', "icnn:art_dropout-dreamsim_0.25_55_size224_iter500_scaled"], 
    caption_names = ['Random', 'Diversity'],
    set_save_name='exp1_qual_main_icnn_art_cont',
    **cont_images_settings,
    **thesis_plot_base_settings)


    # Hetero Test Brain-Diffuser
    visualize_results(
    opb, "deeprecon", "AM", "test", 
    ['test_dropout-quantizedCountBoring_0.25_00', 
    "test_dropout-quantizedCountParty_0.25_00"], 
    caption_names = ['Monotone', 'Heterogeneous'], 
    set_save_name='exp1_qual_hetero_bd_test_cont',
    **cont_images_settings,
    **thesis_plot_base_settings)

    # Hetero Art Brain-Diffuser
    visualize_results(
    opb, "deeprecon", "AM", "art", 
    ['art_dropout-quantizedCountBoring_0.25_00', 
    "art_dropout-quantizedCountParty_0.25_00"], 
    caption_names = ['Monotone', 'Heterogeneous'], 
    set_save_name='exp1_qual_hetero_bd_art_cont',
    **cont_images_settings,
    **thesis_plot_base_settings)


    # Hetero Test iCNN

    visualize_results(
    opb, "deeprecon", "AM", "test", 
    ['icnn:test_dropout-quantizedCountBoring_0.25_00_size224_iter500_scaled', 
     'icnn:test_dropout-quantizedCountParty_0.25_00_size224_iter500_scaled'], 
    caption_names = ['Monotone', 'Heterogeneous'], 
    set_save_name='exp1_qual_hetero_icnn_test_cont',
    **cont_images_settings,
    **thesis_plot_base_settings)

    # Hetero Art iCNN
    visualize_results(
    opb, "deeprecon", "AM", "art", 
    ['icnn:art_dropout-quantizedCountBoring_0.25_00_size224_iter500_scaled', 
     'icnn:art_dropout-quantizedCountParty_0.25_00_size224_iter500_scaled'], 
    caption_names = ['Monotone', 'Heterogeneous'], 
    set_save_name='exp1_qual_hetero_icnn_art_cont',
    **cont_images_settings,
    **thesis_plot_base_settings)


    ##### Exp 2 - AI cap
    visualize_results(
    opb, "deeprecon", "AM", "test", 
    [ "test_aicap_high_level_short-mix_0.8", "test_aicap_low_level_short-mix_0.8", 'test_aicap_human_captions-mix_0.8', ], 
    caption_names = ['High-level', 'Low-level', 'Human'],
    set_save_name='exp2_test_cont', 
    **cont_images_settings,
    **thesis_plot_base_settings)

    visualize_results(
    opb, "deeprecon", "AM", "art", 
    ['art_aicap_high_level_short-mix_0.8',"art_aicap_low_level_short-mix_0.8", 'art_aicap_human_captions-mix_0.8', ], 
    caption_names = ['High-level', 'Low-level', 'Human'],
    set_save_name='exp2_art_cont', 
    **cont_images_settings,
    **thesis_plot_base_settings)

    ##### Exp 3 - Adv pert
    
    # Test
    visualize_results(
    opb, "deeprecon", "AM", "test", 
    ["test", "test_ic_friendly_80-20_500_5", "test_ic_adversarial_80-20_500_5"], 
    caption_names=['baseline', "friendly", "adversarial "],
    set_save_name='exp3_qual_test_cont',
    **cont_images_settings,
    **thesis_plot_base_settings)

    # Art
    visualize_results(
    opb, "deeprecon", "AM", "art", 
    ["art", "art_ic_friendly_80-20_500_5", "art_ic_adversarial_80-20_500_5"], 
    caption_names=['baseline', "friendly", "adversarial "],
    set_save_name='exp3_qual_art_cont',
    **cont_images_settings,
    **thesis_plot_base_settings
    )

