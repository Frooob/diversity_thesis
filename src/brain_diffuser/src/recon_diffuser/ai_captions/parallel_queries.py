import time
import os
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from openai import OpenAI
from set_api_key import set_api_key_env
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm
import logging
from open_api import generate_all_tasks, generate_caption_for_image
from datetime import datetime
import clip
import pandas as pd

set_api_key_env()
client = OpenAI()

def count_tokens(msg):
    try:
        tokens = clip.tokenize(msg)
        n_tokens = (tokens != 0).sum().item()
    except:
        n_tokens = 99
    return n_tokens

def get_completed_tasks():
    with open("completed_tasks.txt", "r") as f:
        lines = f.readlines()
    completed_tasks = []
    for line in lines:
        task_id = line.split(":::")[0]
        caption = line.split(":::")[1]
        completed_tasks.append((task_id, caption))
    return completed_tasks

def extract_completed_tasks(logfile, max_long_tokens, max_short_tokens, min_long_tokens):
    with open(logfile, "r") as f:
        lines = f.readlines()
    completed_tasks = []
    not_validated_tasks = []
    for line in lines:
        if "CAPTION" in line:
            task_id = line.split(":::")[2]
            caption = line.split(":::")[3]
            n_tokens = count_tokens(caption)
            is_long_prompt = "long" in task_id
            max_tokens = max_long_tokens if is_long_prompt else max_short_tokens
            if n_tokens > max_tokens:
                # print(f"Task ID: {task_id} has {n_tokens} tokens. Skipping it.")
                not_validated_tasks.append((task_id, "too_long"))
                continue
            if is_long_prompt and n_tokens < min_long_tokens:
                # print(f"Task ID: {task_id} has {n_tokens} tokens. Skipping it.")
                not_validated_tasks.append((task_id, "too_short"))
                continue
            completed_tasks.append((task_id, caption))
    return completed_tasks, not_validated_tasks


def write_completed_tasks_to_file():
    max_long_tokens = 77
    min_long_tokens = 50
    max_short_tokens = 35
    logfiles = [logfile for logfile in os.listdir() if "worker_log" in logfile]
    completed_tasks = []
    not_validated_tasks = []
    for logfile in logfiles:
        new_completed_tasks, new_not_validated_tasks = extract_completed_tasks(logfile, max_long_tokens, max_short_tokens, min_long_tokens)
        completed_tasks += new_completed_tasks
        not_validated_tasks += new_not_validated_tasks
    
    # not_validated_tasks_distribution
    dist = [{"im_id":nv[0].split(",")[0], "prompt_name":nv[0].split(",")[1], "error":nv[1]} for nv in not_validated_tasks]
    df_dist = pd.DataFrame(dist)

    print(f"N-not validated is {len(not_validated_tasks)}")
    with open("completed_tasks.txt", "w") as f:
        for task in completed_tasks:
            f.write(f"{task[0]}:::{task[1]}")

def get_uncompleted_tasks(all_tasks, completed_tasks):
    completed_task_ids = [task[0] for task in completed_tasks]
    uncompleted_tasks = [task for task in all_tasks if ",".join((task[0],task[2])) not in completed_task_ids]
    print(f"Filtered out {len(all_tasks) - len(uncompleted_tasks)} completed tasks. ({len(uncompleted_tasks)} remaining)")
    return uncompleted_tasks


def worker_function(task, log_lock):
    """Wrapper function to log the result and return it."""
    im, im_path, prompt_name, prompt = task
    caption = generate_caption_for_image(im_path, prompt, client)
    # caption="Hehehe"
    with log_lock:  # Ensure only one thread writes to the log file at a time
        logging.info(f":::CAPTION:::{im},{prompt_name}:::{caption}")
    return caption

def main():
    # Configure logging to log to a file (all workers share this log file)
    logging.basicConfig(
        filename=f"{datetime.now()}_worker_log.log",
        filemode="a",
        format="%(asctime)s - %(levelname)s - %(message)s",
        level=logging.INFO
    )
    write_completed_tasks_to_file()
    num_workers = 5
    out_path_base = "/Users/matt/programming/recon_diffuser" 
    dataset = "deeprecon"

    all_tasks = generate_all_tasks(out_path_base, dataset)
    completed_tasks = get_completed_tasks()
    all_tasks = get_uncompleted_tasks(all_tasks, completed_tasks)

    # all_tasks = all_tasks[:500]

    # This lock ensures multiple threads don't corrupt the log file while writing
    log_lock = logging._lock

    results = []

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        # Submit all tasks to the executor
        future_to_task_id = {executor.submit(worker_function, task, log_lock): task for task in all_tasks}
        for future in tqdm(as_completed(future_to_task_id), total=len(all_tasks), desc="Processing Tasks..."):
            task_id = future_to_task_id[future]
            try:
                result = future.result()
                results.append(result)  # Collect the result in the shared list
            except Exception as exc:
                logging.error(f"Task {task_id} generated an exception: {exc}")
    write_completed_tasks_to_file()

    print(f"All {len(all_tasks)} tasks are complete.")
    # print("Results:", results)

if __name__ == "__main__":
    main()
