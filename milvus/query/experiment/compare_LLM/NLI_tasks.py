import pandas as pd
import json
from utils import generate_prompt, extract_non_think, safe_extract_response
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

# Load the full dataset
bionli_data = load_dataset("presencesw/bionli")
bionli_data = bionli_data["train"]
logging.info(f"number of Premise-Hypothesis pairs: {len(bionli_data)}")

premises = bionli_data['sentence1']
hypothesises = bionli_data['sentence2']
labels = bionli_data['gold_label']

prompts = []
notes = []
for premise, hypothesis in list(zip(premises, hypothesises)):
    prompt = generate_prompt(hypothesis, [premise])
    notes.append("-")
    prompts.append(prompt)

logging.info(f"number of prompts generated: {len(prompts)}")
logging.info(prompts[0])

def query_llm(prompt, type, LLM_name):
    headers = {"Content-Type": "application/json"}
    if type == 'thinking':
        if LLM_name.startswith('Granite'):
            data = {
                "model": LLM_name,
                "messages": [
                    {"role": "control", "content": "thinking"},
                    {"role": "user", "content": prompt}
                ],
                "stream": False,
                "options": {
                    "temperature": 0,
                    "num_predict": 1024}
            }
            LLM_url = "http://localhost:11434/api/chat"
        elif LLM_name.startswith('cogito'):
            data = {
                "model": LLM_name,
                "messages": [
                    {"role": "system", "content": "Enable deep thinking subroutine."},
                    {"role": "user", "content": prompt}
                ],
                "stream": False,
                "options": {
                    "temperature": 0,
                    "num_predict": 1024}
            }
            LLM_url = "http://localhost:11434/api/chat"
        else:
            data = {
                "model": LLM_name,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0,
                    "num_predict": 1024}
            }
            LLM_url = "http://localhost:11434/api/generate"

    elif type == 'standard':
        data = {
            "model": LLM_name,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0,
                "num_predict": 3
            }
        }
        LLM_url = "http://localhost:11434/api/generate"

    else:
        return f"Error unknown type: {type}"

    response = requests.post(LLM_url, headers=headers, data=json.dumps(data))

    if response.status_code == 200:
        actual_response, error = safe_extract_response(response)
        if error:
            return error
        if type == 'thinking':
            pure_response = extract_non_think(actual_response)
            return pure_response
        return actual_response
    else:
        return f"Error {response.status_code}: {response.text}"

def worker(prompt_queue, results, progress_lock, completed_count, which_notes, LLM_name, type):
    """Worker function that continuously processes prompts from the queue"""
    while True:
        try:
            # Get next prompt from queue (timeout prevents hanging)
            prompt_index, prompt = prompt_queue.get(timeout=1)

            # Process the prompt
            actual_response = query_llm(prompt, type, LLM_name)

            response_lower = actual_response.lower().strip()

            if response_lower.startswith('yes') or 'yes' in response_lower[:20]:
                results[prompt_index] = 1
            elif response_lower.startswith('no') or 'no' in response_lower[:20]:
                results[prompt_index] = 0
            else:
                logging.info(f"Error: not a proper answer for prompt {prompt_index}: {actual_response}")
                results[prompt_index] = 0
                which_notes[prompt_index] = "Bad LLM response"

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

LLMs = [
('gemma3:4b', 'standard', 'generate'),
('llama3.1:8b', 'standard', 'generate'),
('mistral:7b', 'standard', 'generate'),
('phi4:14b', 'standard', 'generate'),
('phi4-mini:3.8b', 'standard', 'generate'),
('Granite3.3:8b', 'standard', 'generate'),
('Granite3.3:2b', 'standard', 'generate'),
('cogito:3b', 'standard', 'generate'),
('cogito:8b', 'standard', 'generate'),
('llama3.2:3b', 'standard', 'generate'),
('Phi4-mini-reasoning:3.8b', 'thinking', 'generate'),
('Granite3.3:2b', 'thinking', 'chat'),
('Granite3.3:8b', 'thinking', 'chat'),
('cogito:3b', 'thinking', 'chat'),
('cogito:8b', 'thinking', 'chat')
]

# Create directory if it doesn't exist
from pathlib import Path
Path("result/LLMs").mkdir(parents=True, exist_ok=True)

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
    with open(f"result/LLMs/bionli_scores_{LLM_file_name}_{type}.json", "w", encoding="utf-8") as f:
        json.dump(scores, f, ensure_ascii=False, indent=2)

    with open(f"result/LLMs/bionli_notes_{LLM_file_name}_{type}.json", "w", encoding="utf-8") as f:
        json.dump(notes_for_this_LLM, f, ensure_ascii=False, indent=2)

    with open(f"result/LLMs/bionli_timer.json", "w", encoding="utf-8") as f:
        json.dump(timer, f, ensure_ascii=False, indent=2)

with open(f"result/LLMs/bionli_labels.json", "w", encoding="utf-8") as f:
    json.dump(labels, f, ensure_ascii=False, indent=2)
