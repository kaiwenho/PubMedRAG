import pandas as pd
import json
from utils_3step import generate_prompt, extract_non_think, safe_extract_response, parse_llm_response, query_llm, query_llm_three_step
import logging
import time
import queue
from concurrent.futures import ThreadPoolExecutor
import threading
import requests
import copy
from datasets import load_dataset

MAX_WORKERS = 3
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logging.getLogger("httpx").setLevel(logging.WARNING)

# TODO: update the generate_prompt function for other prompt types
# TODO: add medgemma
LLMs = [
# ('gemma3:4b', 'standard', 'generate'),
# ('llama3.1:8b', 'standard', 'generate'),
# ('mistral:7b', 'standard', 'generate'),
# ('phi4:14b', 'standard', 'generate'),
# ('Granite3.3:8b', 'standard', 'generate'),
# ('Granite3.3:2b', 'standard', 'generate'),
# ('cogito:3b', 'standard', 'generate'),
# ('cogito:8b', 'standard', 'generate'),
# ('llama3.2:3b', 'standard', 'generate'),
# ('Phi4-mini-reasoning:3.8b', 'thinking', 'generate'),
# ('Granite3.3:2b', 'thinking', 'chat'),
# ('Granite3.3:8b', 'thinking', 'chat'),
# ('cogito:3b', 'thinking', 'chat'),
# ('cogito:8b', 'thinking', 'chat'),
('gpt-oss:20b', 'thinking', 'generate')
]

prompt_types = [
'naive',
'devils_advocate',
'pre-commitment_to_standards',
'conservative_evidence_search',
'three_step_analysis',
]
prompt_type = prompt_types[4]  # Using three_step_analysis
# Create directory if it doesn't exist
from pathlib import Path
Path(f"result/LLMs/{prompt_type}").mkdir(parents=True, exist_ok=True)

# Load the full dataset
bionli_data = load_dataset("presencesw/bionli")
bionli_data = bionli_data["train"]
# bionli_data = bionli_data[:10]
logging.info(f"number of Premise-Hypothesis pairs: {len(bionli_data)}")

premises = bionli_data['sentence1']
hypothesises = bionli_data['sentence2']
labels = bionli_data['gold_label']

prompts = []
for premise, hypothesis in list(zip(premises, hypothesises)):
    prompt = generate_prompt(hypothesis, [premise], prompt_type)
    prompts.append(prompt)
notes = [None] * len(prompts)

logging.info(f"number of prompts generated: {len(prompts)}")
logging.info(f"promt type: {prompt_type}")

def worker(prompt_queue, results, progress_lock, completed_count, which_notes, LLM_name, type):
    """Worker function that continuously processes prompts from the queue"""
    while True:
        try:
            # Get next prompt from queue (timeout prevents hanging)
            prompt_index, prompt = prompt_queue.get(timeout=1)

            # Check if this is a three-step analysis prompt
            if prompt_type == "three_step_analysis":
                # For three-step analysis, prompt is a tuple of (claim, context)
                claim, context = prompt
                actual_response, step_error = query_llm_three_step(claim, context, type, LLM_name)

                # If any step failed, don't parse - the response is incomplete
                if step_error is not None:
                    classification = None
                    which_notes[prompt_index] = step_error
                else:
                    # All steps succeeded, parse the final response
                    classification, parse_note = parse_llm_response(actual_response, prompt_type, LLM_name, type)
                    which_notes[prompt_index] = parse_note
            else:
                # Standard single prompt
                actual_response = query_llm(prompt, type, LLM_name)
                classification, parse_note = parse_llm_response(actual_response, prompt_type, LLM_name, type)
                which_notes[prompt_index] = parse_note

            results[prompt_index] = classification

            if classification is None:
                logging.info(f"Error: not a proper answer for prompt {prompt_index}: {actual_response}")

            # Update progress
            with progress_lock:
                completed_count[0] += 1
                if completed_count[0] % 500 == 0:
                    logging.info(f"Processing claim {completed_count[0]}/{len(prompts)}")

            # Mark task as done
            prompt_queue.task_done()

        except queue.Empty:
            # No more prompts to process
            break
        except Exception as e:
            logging.info(f"Worker error: {e}")
            prompt_queue.task_done()

timer = []
# LLM_name = "gemma3:4b"  # Replace with your actual model name
for LLM_name, type, api in LLMs:
    # Initialize data structures
    logging.info(f"Running the LLM {LLM_name} with {type} mode: ")
    scores = []
    notes_for_this_LLM = copy.deepcopy(notes)
    results = [None] * len(prompts)
    prompt_queue = queue.Queue()
    progress_lock = threading.Lock()
    completed_count = [0]  # Use list to make it mutable in the worker function

    start = time.time()

    # Add all prompts to the queue
    for i, prompt in enumerate(prompts):
        if prompt:
            prompt_queue.put((i, prompt))

    # Create and start worker threads
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        # Start 4 workers that will continuously process prompts
        workers = []
        for _ in range(MAX_WORKERS):
            worker_future = executor.submit(worker, prompt_queue, results, progress_lock, completed_count, notes_for_this_LLM, LLM_name, type)
            workers.append(worker_future)

        # Wait for all prompts to be processed
        prompt_queue.join()

        # Wait for all workers to complete
        for worker_future in workers:
            worker_future.result()

    # Convert results to scores
    for i, prompt in enumerate(prompts):
        if prompt and results[i] is not None:
            scores.append(results[i])
        else:
            scores.append(0)

    end = time.time()
    logging.info(f"Total processing time of {LLM_name} inferences: {end - start:.2f} seconds")
    timer.append((LLM_name, type, f"{end - start:.2f}"))

    LLM_file_name = LLM_name.replace(':',"_")
    with open(f"result/LLMs/{prompt_type}/bionli_preds_{LLM_file_name}_{type}.json", "w", encoding="utf-8") as f:
        json.dump(scores, f, ensure_ascii=False, indent=2)

    with open(f"result/LLMs//{prompt_type}/bionli_notes_{LLM_file_name}_{type}.json", "w", encoding="utf-8") as f:
        json.dump(notes_for_this_LLM, f, ensure_ascii=False, indent=2)

    with open(f"result/LLMs//{prompt_type}/bionli_timer.json", "w", encoding="utf-8") as f:
        json.dump(timer, f, ensure_ascii=False, indent=2)

with open(f"result/LLMs/{prompt_type}/bionli_labels.json", "w", encoding="utf-8") as f:
    json.dump(labels, f, ensure_ascii=False, indent=2)
