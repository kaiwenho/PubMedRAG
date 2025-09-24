from transformers import AutoTokenizer, AutoModel
import torch
import numpy as np
import requests
import json
import re

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
    If a <response>...</response> block is present, return only its content.

    Args:
        text (str): The input string containing <think> and optionally <response> blocks.

    Returns:
        str: The final cleaned text.
    """
    # Remove <think>...</think> blocks
    text_no_think = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)

    # Look for <response>...</response> and return the content if found
    response_match = re.search(r'<response>(.*?)</response>', text_no_think, flags=re.DOTALL)
    if response_match:
        return response_match.group(1).strip()

    # If no <response> block, return the text without <think> blocks
    return text_no_think.strip()



def clean_llm_text(text):
    """Cleans LLM-generated text by removing formatting symbols and newlines."""
    # Replace newlines and tabs with spaces
    text = text.replace('\n', ' ').replace('\r', ' ').replace('\t', ' ')

    # Remove common markdown/formatting symbols
    # text = re.sub(r'[*#_`~-]+', '', text)

    # Collapse multiple spaces into one
    cleaned = re.sub(r'\s+', ' ', text).strip()
    return cleaned

def safe_extract_response(response):
    try:
        response_data = response.json()
    except json.JSONDecodeError:
        return None, f"Error: Unable to parse JSON response: {response.text}"

    # Case 1: Standard response with top-level 'response' field
    if isinstance(response_data.get("response"), str):
        cleaned = clean_llm_text(response_data["response"])
        return cleaned, None

    # Case 2: Chat API response with nested 'message' -> 'content'
    try:
        content = response_data["message"]["content"]
        if isinstance(content, str):
            cleaned = clean_llm_text(content)
            return cleaned, None
    except (KeyError, TypeError):
        pass

    return None, f"Error: Unsupported response format: {response_data}"

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

def filter_sort_trim_evidence(evidence_list, max_tokens=4000, avg_tokens_per_word=4):
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
