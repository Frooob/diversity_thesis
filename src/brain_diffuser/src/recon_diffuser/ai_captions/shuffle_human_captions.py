import os
import pandas as pd

df_human_captions = pd.read_csv('human_captions.csv')


df_human_captions_shuffled = df_human_captions.copy()

df_human_captions_shuffled['caption'] = df_human_captions_shuffled['caption'].sample(frac=1).reset_index(drop=True)
df_human_captions_shuffled.to_csv('human_captions_shuffled.csv', index=False)


df_human_captions_shuffled['caption'] = df_human_captions_shuffled['caption'].sample(frac=1).reset_index(drop=True)
df_human_captions_shuffled_single_caption = df_human_captions_shuffled[df_human_captions_shuffled['counter'] == 1]
df_human_captions_shuffled_single_caption.to_csv('human_captions_shuffled_single_caption.csv', index=False)
