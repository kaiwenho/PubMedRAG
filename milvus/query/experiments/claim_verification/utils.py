import re
from transformers import AutoTokenizer, AutoModel
import torch
import numpy as np
import requests
import json

LLM_url = "http://localhost:11434/api/generate"
headers = {"Content-Type": "application/json"}

_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = AutoTokenizer.from_pretrained(_MODEL_NAME)
model = AutoModel.from_pretrained(_MODEL_NAME).to(device)

def embed_sentences(sentences):
    """
    sentences: List[str]
    returns: np.ndarray of shape (len(sentences), hidden_size)
    """
    # 4) Tokenize + send to the same device
    batch = tokenizer(
        sentences,
        return_tensors="pt",
        padding=True,
        truncation=True,
    )
    batch = {k: v.to(device) for k, v in batch.items()}

    # 5) Forward once
    with torch.no_grad():
        last_hidden = model(**batch).last_hidden_state

    # 6) Mean‑pool & bring back to CPU + numpy
    return last_hidden.mean(dim=1).cpu().numpy()

def embed_sentence(sentence):
    input = tokenizer(sentence, return_tensors='pt', padding=True, truncation=True)
    embedding = model(**input).last_hidden_state.mean(dim=1).detach().numpy()
    return embedding

def extract_non_think(text):
    """
    Remove any text enclosed in <think>...</think> tags and return the remaining text.

    Args:
        text (str): The input string containing <think> blocks.

    Returns:
        str: The text with <think> blocks removed.
    """
    # Use re.DOTALL to ensure newline characters are included in the match.
    cleaned_text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
    return cleaned_text.strip()


# TODO: add qualifier to the returned sentence
def generate_claim(edge):
    """
    Convert an edge dictionary to a sentence.

    The function:
      - Extracts 'subject', 'predicate', and 'object' from the edge.
      - Removes underscores from the predicate.
      - Removes the "Biolink:" prefix if it exists.

    Args:
        edge (dict): A dictionary with keys 'subject', 'predicate', and 'object'.

    Returns:
        str: A sentence in the format "subject predicate object".

    Example:
        Input: {'subject': 'ginger', 'object': 'nausea', 'predicate': 'Biolink:treats'}
        Output: 'ginger treats nausea'
    """
    subject = edge.get('subject', '')
    predicate = edge.get('predicate', '')
    object = edge.get('object', '')

    # Remove underscores and unwanted prefix from the predicate.
    predicate = predicate.replace('_', ' ')
    if predicate.startswith("Biolink:"):
        predicate = predicate[len("Biolink:"):]

    return f"{subject} {predicate} {object}".strip()

def generate_treatment_sentences(df, positive):
    if {'source_name', 'target_name'}.issubset(df.columns):
        if positive:
            df['sentence'] = df.apply(lambda row: f"{row['source_name']} treats {row['target_name']}", axis=1)
        else:
            df['sentence'] = df.apply(lambda row: f"{row['source_name']} does not treat {row['target_name']}", axis=1)
        return df['sentence'].tolist()

    elif {'drug_name', 'disease_name'}.issubset(df.columns):
        if positive:
            df['sentence'] = df.apply(lambda row: f"{row['drug_name']} treats {row['disease_name']}", axis=1)
        else:
            df['sentence'] = df.apply(lambda row: f"{row['drug_name']} does not treat {row['disease_name']}", axis=1)
        return df['sentence'].tolist()

    else:
        raise ValueError("DataFrame must contain 'source_name' and 'target_name' columns or 'drug_name' and 'disease_name' columns.")


def generate_prompt(claim, context):
    prompt = f"""Claim: {claim}
Context:
{"\n".join(context)}
Question: Does the context support the claim? Just return Yes or No.
"""
    return prompt

def filter_sort_trim_evidence(evidence_list, max_tokens=15000, avg_tokens_per_word=1.5):
    max_words = int(max_tokens / avg_tokens_per_word)

    # Step 1: Group by pmid and keep the highest distance evidence per pmid
    pmid_to_best_evidence = {}

    for item in evidence_list:
        pmid = item["entity"]["pmid"]
        distance = item["distance"]

        if pmid not in pmid_to_best_evidence or distance > pmid_to_best_evidence[pmid]["distance"]:
            pmid_to_best_evidence[pmid] = item

    # Step 2: Get the best evidence from each pmid group
    unique_evidence = list(pmid_to_best_evidence.values())

    # Step 3: Sort by distance in descending order
    sorted_evidence = sorted(unique_evidence, key=lambda x: x["distance"], reverse=True)

    # Step 4: Select as many as possible without exceeding the token limit
    selected_evidence = []
    total_words = 0

    for item in sorted_evidence:
        sentence = item["entity"]["sentence"]
        word_count = len(sentence.split())

        if total_words + word_count > max_words:
            break

        selected_evidence.append(item)
        total_words += word_count

    return selected_evidence

def query_llm(llm_name, prompt):
    data = {
        "model": llm_name,
        "prompt": prompt,
        "stream": False
    }
    response = requests.post(LLM_url, headers=headers, data=json.dumps(data))
    if response.status_code == 200:
        response_data = response.json()
        actual_response = response_data['response'].strip()
        if llm_name == 'deepseek-r1:8b':
            actual_response = extract_non_think(actual_response)
        return llm_name, actual_response
    else:
        return llm_name, f"Error {response.status_code}: {response.text}"
