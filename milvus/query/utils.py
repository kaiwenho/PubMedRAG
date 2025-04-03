import re
from transformers import AutoTokenizer, AutoModel

# Load the model for converting sentence to vector/embeddings
model_name = 'sentence-transformers/all-MiniLM-L6-v2'
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

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
