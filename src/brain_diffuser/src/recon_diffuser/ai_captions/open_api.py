from openai import OpenAI
from set_api_key import set_api_key_env
import os
from PIL import Image
import base64
import io
import matplotlib.pyplot as plt
import clip
from tqdm import tqdm
import textwrap
from collections import defaultdict
import pandas as pd

# Haiku test request to see if the API is working
def test_request(client):
    completion = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "user","content": "Write a short story about an Elf. Make sure the output doesn't exceed 30 tokens."}],
        # max_tokens= 10,
        )
      
    msg = completion.choices[0].message.content
    return msg
# response_msg = test_request(client)

def count_tokens(msg):
    try:
        tokens = clip.tokenize(msg)
        n_tokens = (tokens != 0).sum().item()
    except:
        n_tokens = 99
    return n_tokens
# count_tokens(response_msg)

def get_all_img_paths(out_path_base, dataset, im_suffix): # duplicate from dropout_random.py for now
    input_img_folder = os.path.join(out_path_base, 'data/train_data', dataset)
    all_input_imgs = [f for f in os.listdir(input_img_folder) if f.endswith(im_suffix)]
    all_input_filepaths = [os.path.join(input_img_folder, x) for x in all_input_imgs]
    all_input_imgs = [os.path.basename(x).split(".")[0] for x in all_input_imgs]
    return all_input_imgs, all_input_filepaths

def encode_image_to_base64(image_path):
    """
    Encodes an image to a base64 string.

    Parameters:
        image_path (str): Path to the image file.

    Returns:
        str: Base64 encoded string of the image.
    """
    with Image.open(image_path) as img:
        buffered = io.BytesIO()
        img.save(buffered, format="PNG")
        img_bytes = buffered.getvalue()
        base64_image = base64.b64encode(img_bytes).decode("utf-8")
    return base64_image

def generate_image_description(image_path, prompt, client):
    """
    Generates a description of an image based on a given prompt.

    Parameters:
        image_path (str): Path to the image file.
        prompt (str): Instruction for describing the image.

    Returns:
        str: Generated description of the image.
    """
    base64_image = encode_image_to_base64(image_path)

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"},
                    },
                ],
            }
        ],
    )

    return response

def plot_image_with_multiple_descriptions(image, descriptions, 
                                          figure_width=8, figure_height=10, 
                                          max_font_size=16, min_font_size=6,
                                          wrap_chars=60):
    """
    Plots an image on the top and places multiple descriptions below it.
    Each description consists of a (caption, text) tuple.
    The text is automatically wrapped and the font size is decreased if necessary 
    so that all descriptions fit neatly.

    Parameters
    ----------
    image : np.ndarray or PIL.Image
        Image data to display (e.g., a NumPy array or a PIL Image).
    descriptions : list of tuples
        A list of (caption, text) tuples.
        Example: [("Caption 1", "This is the first description..."),
                  ("Caption 2", "This is another description...")]
    figure_width : float, optional
        Width of the figure in inches. Default is 8.
    figure_height : float, optional
        Height of the figure in inches. Default is 10.
    max_font_size : int, optional
        The maximum font size to try for the text. Default is 16.
    min_font_size : int, optional
        The minimum font size to try if texts don't fit. Default is 6.
    wrap_chars : int, optional
        The number of characters at which to wrap text lines. Default is 60.
    """

    n_desc = len(descriptions)
    ratios = [10] + [0.5]*n_desc

    fig, axes = plt.subplots(n_desc+1, 1, figsize=(figure_width, figure_height),
                             gridspec_kw={'height_ratios': ratios})
    ax_img = axes[0]
    ax_img.imshow(image.resize((image.width, image.height)))
    ax_img.axis('off')
    wrapped_blocks = []
    for caption, text in descriptions:
        wrapped_lines = textwrap.wrap(text, width=wrap_chars)
        wrapped_text = "\n".join(wrapped_lines)
        wrapped_blocks.append((caption, wrapped_text))
    def texts_fit(font_size):
        for ax in axes[1:]:
            ax.clear()
            ax.axis('off')
        for i, ((caption, wrapped_text), ax) in enumerate(zip(wrapped_blocks, axes[1:])):            
            combined_text = f"$\\bf{{{caption}}}$\n{wrapped_text}"
            txt_obj = ax.text(0.5, 0.5, combined_text, ha='center', va='center',
                              wrap=True, fontsize=font_size, transform=ax.transAxes)
        fig.canvas.draw_idle()
        renderer = fig.canvas.get_renderer()
        for ((caption, wrapped_text), ax) in zip(wrapped_blocks, axes[1:]):
            txt_obj = ax.texts[0]
            text_bbox = txt_obj.get_window_extent(renderer=renderer)
            ax_bbox = ax.get_window_extent(renderer=renderer)
            if (text_bbox.width > ax_bbox.width or text_bbox.height > ax_bbox.height):
                return False
        return True
    # for fs in range(max_font_size, min_font_size-1, -1):
    #     if texts_fit(fs):
    #         break
    fs = 15.5
    texts_fit(fs)
    plt.tight_layout()
    # plt.show()
    # plt.clf()


def plot_image_with_description(image_path, description, figure_width=8, figure_height=6, max_font_size=20, min_font_size=6, wrap_chars=80):
    image = Image.open(im_path)
    wrapped_lines = textwrap.wrap(description, width=wrap_chars)
    wrapped_text = "\n".join(wrapped_lines)
    fig, (ax_img, ax_txt) = plt.subplots(2, 1, figsize=(figure_width, figure_height), gridspec_kw={'height_ratios': [3, 1]})
    
    ax_img.imshow(image)
    ax_img.axis('off')
    def text_fits(font_size):
        ax_txt.clear()
        ax_txt.axis('off')
        txt_obj = ax_txt.text(0.5, 0.5, wrapped_text, ha='center', va='center', wrap=True, fontsize=font_size)
        fig.canvas.draw_idle()
        renderer = fig.canvas.get_renderer()
        text_bbox = txt_obj.get_window_extent(renderer=renderer)
        ax_bbox = ax_txt.get_window_extent(renderer=renderer)
        return (text_bbox.width <= ax_bbox.width) and (text_bbox.height <= ax_bbox.height)
    for fs in range(max_font_size, min_font_size - 1, -1):
        if text_fits(fs):
            break  # fs is suitable
    plt.tight_layout()
    plt.show()

def generate_caption_for_image(im_path, prompt, client):
    response = generate_image_description(im_path, prompt, client)
    caption = response.choices[0].message.content
    return caption


def generate_all_tasks(out_path_base, dataset):
    im_idx, im_paths = get_all_img_paths(out_path_base, dataset, "JPEG")

    prompt_highlevel = "Describe the visual contents of this picture. Focus on the semantic concepts in the image. What are the main subjects of the image and what is happening in it? "

    prompt_lowlevel = "Describe the visual contents of this picture without using the word representing its direct category or semantic meaning as much as possible. Instead, focus on the colors, shapes, and textures with their relative positions in the image. Do not interpret the scene just describe what is actually visible. "

    prompt_long = "Make sure the output doesn't exceed 70 tokens."
    prompt_short = "Make sure the output doesn't exceed 25 tokens."
    prompt_high_long = prompt_highlevel + prompt_long
    prompt_high_short = prompt_highlevel + prompt_short

    prompt_low_long = prompt_lowlevel + prompt_long
    prompt_low_short = prompt_lowlevel + prompt_short

    prompts_with_names = {
        "high_level_long": prompt_high_long,
        "high_level_short": prompt_high_short,
        "low_level_long": prompt_low_long,
        "low_level_short": prompt_low_short
    }

    all_tasks = []

    for im_path, im in tqdm(zip(im_paths, im_idx), total=len(im_paths), desc="Generating Tasks..."):
        for name, prompt in prompts_with_names.items():
            all_tasks.append((im, im_path, name, prompt))
    
    return all_tasks

def create_task_csv():
    with open("completed_tasks.txt", "r") as f:
        lines = f.readlines()
    completed_tasks = defaultdict(list)
    for line in lines:
        task_id = line.split(":::")[0]
        caption = line.split(":::")[1].strip()
        content_id, caption_name = task_id.split(",")
        csv_format_dict = {"content_id": content_id, "counter": 1, "caption": caption}
        completed_tasks[caption_name].append(csv_format_dict)

    for caption_name, tasks in completed_tasks.items():
        df_caption = pd.DataFrame(tasks)
        df_caption.to_csv(f"{caption_name}.csv", index=False)

def get_all_results_from_csvs():
    csv_files = [f for f in os.listdir() if f.endswith(".csv")]
    prompt_names = [f.split(".")[0] for f in csv_files]
    all_results_from_csv = {}
    for prompt_name in prompt_names:
        df = pd.read_csv(f"{prompt_name}.csv")
        all_results_from_csv[prompt_name] = df
    return all_results_from_csv

def get_img_id_and_description(all_results_from_csv):
    # get all prompt name descriptions for a randomly picked image 
    prompt_names = list(all_results_from_csv.keys())
    img_id = all_results_from_csv[prompt_names[0]]["content_id"].sample().values[0]    
    descriptions = []
    for prompt_name in prompt_names:
        caption = all_results_from_csv[prompt_name][all_results_from_csv[prompt_name]["content_id"] == img_id]["caption"].values[0]
        descriptions.append((img_id, prompt_name, caption))

    return img_id, descriptions

def plot_from_csv():
    all_results_from_csv = get_all_results_from_csvs()
    img_id, descriptions = get_img_id_and_description(all_results_from_csv)
    im_num = im_idx.index(img_id)
    # sort descriptions by prompt name
    descriptions = sorted(descriptions, key=lambda x: x[1])
    descriptions_for_plot = []
    for im, prompt_name, caption in descriptions:   
        if im == im_idx[im_num]:
            descriptions_for_plot.append((prompt_name.replace("_", "\_"), caption))
    plot_image_with_multiple_descriptions(Image.open(im_paths[im_num]), descriptions_for_plot)

def plot_from_csv_only_relevant_prompts():
    all_results_from_csv = get_all_results_from_csvs()
    n_images = 3
    thesis_plots_path = "/Users/matt/ownCloud/gogo/MA/thesis/diversity_thesis/plots"
    # set active ax
    img_id, descriptions = get_img_id_and_description(all_results_from_csv)
    im_num = im_idx.index(img_id)
    # sort descriptions by prompt name
    descriptions = sorted(descriptions, key=lambda x: x[1])
    relevant_prompts = ['human_captions', 'high_level_short', 'low_level_short']

    # Get the relevant descriptions and sort them by the relevant prompts
    descriptions = [x for x in descriptions if x[1] in relevant_prompts]
    descriptions = sorted(descriptions, key=lambda x: relevant_prompts.index(x[1]))

    # Remove the _short from the prompt names
    descriptions = [(x[0], x[1].replace("_short", ""), x[2]) for x in descriptions]
    print(descriptions)

    descriptions_for_plot = []
    for im, prompt_name, caption in descriptions:   
        if im == im_idx[im_num]:
            descriptions_for_plot.append((prompt_name.replace("_", "\_"), caption))
    image = Image.open(im_paths[im_num])
    descriptions = descriptions_for_plot
    plot_image_with_multiple_descriptions(image, descriptions_for_plot)

    plt.savefig(os.path.join(thesis_plots_path, f"aicap_caption_samples.png"))

def count_tokens(msg):
    try:
        tokens = clip.tokenize(msg)
        n_tokens = (tokens != 0).sum().item()
    except:
        n_tokens = 99
    return n_tokens

def get_df_tokenlength():
    all_results_from_csv = get_all_results_from_csvs()
    df_low_short = all_results_from_csv['low_level_short']
    df_low_short = df_low_short.rename(columns={'caption': "low_level_short"})

    df_high_short = all_results_from_csv['high_level_short']
    df_high_short = df_high_short.rename(columns={'caption': "high_level_short"})

    df = pd.merge(df_low_short, df_high_short, 'inner', on='content_id')
    df['n_tokens_low_short'] = df['low_level_short'].apply(count_tokens)
    df['n_tokens_high_short'] = df['high_level_short'].apply(count_tokens)
    return df


        


def check_all_token_lenghts():
    all_results_from_csv = get_all_results_from_csvs()
    for prompt_name, df in all_results_from_csv.items():
        df["n_tokens"] = df["caption"].apply(count_tokens)
        print(f"Prompt: {prompt_name}")
        print(df["n_tokens"].value_counts())

if __name__ == "'__main__'":
    ...
    clip_model, preprocess = clip.load("ViT-L/14")
    set_api_key_env()
    client = OpenAI()

    out_path_base = "/Users/matt/programming/recon_diffuser" 
    dataset = "deeprecon"
    im_idx, im_paths = get_all_img_paths(out_path_base, dataset, "JPEG")

    im_path = im_paths[0]

    captions_with_names = {}
    im_paths_short = im_paths[:2]
    im_idx_short = im_idx[:2]

    # all_tasks = generate_all_tasks(out_path_base, dataset) # 4 prompts per image

    # all_results_from_csv = get_all_results_from_csvs()
    # task_csv = create_task_csv()

    # all_results = []
    # for im, im_path, prompt_name, prompt in tqdm(all_tasks[:2]):
    #     caption = generate_caption_for_image(im_path, prompt, client)
    #     # caption = "Hey friends hows it going"
    #     all_results.append((im, prompt_name, caption))

    # im_num = 0
    # descriptions = []
    # for im, prompt_name, caption in all_results:
    #     if im == im_idx[im_num]:
    #         descriptions.append((prompt_name.replace("_", "\_"), caption))

    # plot_image_with_multiple_descriptions(Image.open(im_paths[im_num]), descriptions)

    # plot_from_csv()

    # Create the plot for the thesis
    df_tokenlength = get_df_tokenlength()
    import seaborn as sns
    melted = pd.melt(df_tokenlength, ['content_id'], ['n_tokens_low_short', 'n_tokens_high_short'])
    # sns.boxplot(data=melted, x='variable', y='value')
    print("Tokenlength comparison")
    print(melted.groupby(['variable'])['value'].agg(['mean', 'std']))
    
    plot_from_csv_only_relevant_prompts()