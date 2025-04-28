def get_prompt_name_and_params_from_output_name(output_name):
    if not output_name.startswith("aicap"):
        raise ValueError(f"Output name needs to start with aicap.")

    if '-' in output_name:
        index_first_dash = output_name.index('-')
    else:
        index_first_dash = len(output_name)

    prompt_name = output_name[6:index_first_dash]

    params = output_name.split('-')

    params = params[1:]
    params = {p.split('_')[0]: eval(p.split('_')[1]) for p in params}

    return prompt_name, params

def parse_true_feat_output_name(output_name):
    if not output_name.startswith("true"):
        raise ValueError(f"Output name for true feature reconstruction must start with 'true'.")
    output_name_parts = output_name.split('-')
    params = {}
    for part in output_name_parts:
        key, value = part.split("_")
        try:
            params[key] = eval(value)
        except:
            params[key] = value
    algorithm = params.pop('true')
    return algorithm, params