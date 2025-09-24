import numpy as np
import requests
import json
import re
import logging
from ollama import Client

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
    # Remove **Thinking...** ... **...done thinking.** blocks
    text_no_think = re.sub(
        r'\*\*Thinking\.\.\.\*\*.*?\*\*\.\.\.done thinking\.\*\*',
        '',
        text_no_think,
        flags=re.DOTALL
    )

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

def generate_prompt(claim, context, prompt_type):
    if prompt_type == "naive":
        prompt = f"""Claim: {claim}
Context:
{"\n".join(context)}
Question: Does the context support the claim? Just return Yes or No."""

    elif prompt_type == "devils_advocate":
        prompt = f"""Claim: {claim}
Context:
{"\n".join(context)}
First, analyze the context that SUPPORTS the claim.
Then, analyze the context that CONTRADICTS the claim.
Finally, analyze the context that is irrelevant to the claim.
Now what is your conclusion on the relation between the claim and the context? Select exactly ONE of these four options:
Support - if supporting evidence is strongest
Contradict - if contradicting evidence is strongest
Neutral - if evidence is weak or irrelevant
Maybe - if you cannot determine

Please format your final answer as:
###Final Answer: [Write the corresponding word]
Example: ###Final Answer: Neutral
"""

    elif prompt_type == "pre-commitment_to_standards":
        prompt = f"""Does the context support the claim?
Before reading the context, state: I will answer "Support" only with clear positive evidence, "Contradict" only with clear opposing evidence, and "Neutral" when evidence is absent, mixed, or irrelevant.
Claim: {claim}
Context: {"\n".join(context)}
Apply the pre-stated standard, then do reasoning and analyze if the context supports the claim.
Now you must select exactly ONE of these four options:
Support - if supporting evidence is strongest
Contradict - if contradicting evidence is strongest
Neutral - if evidence is mixed, weak, or irrelevant
Maybe - if you cannot determine

Please format your final answer as:
###Final Answer: [Write the corresponding word]
Example: ###Final Answer: Neutral
"""

    elif prompt_type == "conservative_evidence_search":
        prompt = f"""Does the context support the claim?
Claim: {claim}
Context:
{"\n".join(context)}
Actively search for contradictory evidence first, then supporting evidence.
- If clear contradictions found: "Contradict"
- If clear support found AND no contradictions: "Support"
- If evidence is absent, mixed, or irrelevant: "Neutral"
Now you must select exactly ONE of these four options:
Support - if supporting evidence is strongest
Contradict - if contradicting evidence is strongest
Neutral - if evidence is mixed, weak, or irrelevant
Maybe - if you cannot determine

Please format your final answer as:
###Final Answer: [Write the corresponding word]
Example: ###Final Answer: Neutral
"""

    elif prompt_type == "three_step_analysis":
        # For three-step analysis, return tuple of (claim, context) instead of single prompt
        return (claim, context)

    return prompt

def generate_three_step_prompts(claim, context):
    """
    Generate the three prompts for the three-step analysis approach.

    Args:
        claim (str): The claim to analyze
        context (list): List of context sentences

    Returns:
        tuple: (step1_prompt, step2_prompt, step3_template)
    """
    context_text = "\n".join(context)

    step1_prompt = f"""Claim: {claim}
Context:
{context_text}

Please analyze the context and identify any evidence that SUPPORTS the given claim.
Explain why this evidence supports the claim if you find any.
If you don't find supporting evidence, clearly state that no supporting evidence was found.
"""

    step2_prompt = f"""Claim: {claim}
Context:
{context_text}

Please analyze the context and identify any evidence that CONTRADICTS or goes AGAINST the given claim.
Explain why this evidence contradicts the claim if you find any.
If you don't find contradicting evidence, clearly state that no contradicting evidence was found.
"""

    step3_template = f"""Claim: {claim}
Context:
{context_text}

Analysis of Supporting Evidence:
{{step1_response}}

Analysis of Contradicting Evidence:
{{step2_response}}

Based on the analyses above, please classify the relationship between the claim and the context.
Consider both the supporting and contradicting evidence analyses.

Now you must select exactly ONE of these four options:
Support - if supporting evidence is strongest
Contradict - if contradicting evidence is strongest
Neutral - if evidence is mixed, weak, or irrelevant
Maybe - if you cannot determine

Please format your final answer as:
###Final Answer: [Write the corresponding word]
Example: ###Final Answer: Neutral
"""

    return step1_prompt, step2_prompt, step3_template

def query_llm_three_step(claim, context, type, LLM_name):
    """
    Perform three-step analysis:
    1. Find supporting evidence
    2. Find contradicting evidence
    3. Make final classification based on both analyses

    Args:
        claim (str): The claim to analyze
        context (list): List of context sentences
        type (str): LLM type ('standard' or 'thinking')
        LLM_name (str): Name of the LLM model

    Returns:
        tuple: (final_response, error_note)
        - final_response: The response from step 3, or error message
        - error_note: None if successful, or description of which step failed
    """
    try:
        # Generate the three prompts
        step1_prompt, step2_prompt, step3_template = generate_three_step_prompts(claim, context)

        # Step 1: Find supporting evidence
        step1_response = query_llm(step1_prompt, type, LLM_name)
        if step1_response.startswith("Error"):
            return step1_response, "step1_failed"

        # Step 2: Find contradicting evidence
        step2_response = query_llm(step2_prompt, type, LLM_name)
        if step2_response.startswith("Error"):
            return step2_response, "step2_failed"

        # Step 3: Final classification based on both analyses
        step3_prompt = step3_template.format(
            step1_response=step1_response,
            step2_response=step2_response
        )

        step3_response = query_llm(step3_prompt, type, LLM_name)
        if step3_response.startswith("Error"):
            return step3_response, "step3_failed"

        return step3_response, None

    except Exception as e:
        return f"Three-step analysis failed: {str(e)}", "exception_occurred"

def query_llm(prompt, type, LLM_name):
    if LLM_name.startswith('gpt-oss'):
        client = Client()
        messages = [
          {
            'role': 'user',
            'content': prompt,
          },
        ]
        response = client.chat(
            model=LLM_name,
            messages=messages,
            options={
                "num_ctx": 8192,
                "temperature": 0,
                "num_predict": 1024},  # 8192 is the recommended lower limit for the context window
          )
          # Extract response content from ChatResponse object
        if hasattr(response, 'message') and hasattr(response.message, 'content'):
            actual_response = response.message.content
        else:
            return f"Error: Unexpected gpt-oss response format: {response}"

        if type == 'thinking':
            pure_response = extract_non_think(actual_response)
            return pure_response

    else:
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
                    "num_predict": 1024
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

def parse_llm_response(response_text, prompt_type, LLM_name=None, type=None):
    """
    Enhanced parser with fallback LLM inference capability
    """
    if prompt_type == "naive":
        # For naive prompts, only accept Yes/No
        response_lower = response_text.lower().strip()
        if response_lower.startswith('yes') or 'yes' in response_lower[:20]:
            return 'Yes', None
        elif response_lower.startswith('no') or 'no' in response_lower[:20]:
            return 'No', None
        else:
            return None, "Bad LLM response"

    valid_answers = ['Support', 'Contradict', 'Neutral', "Maybe"]

    # Try primary extraction method
    classification, error_note = extract_final_answer_primary(response_text, valid_answers)

    # If primary method succeeded, return result
    if classification is not None:
        return classification, error_note

    # Primary method failed - try fallback LLM inference
    logging.info(f"Primary parsing failed: {error_note}. Attempting fallback LLM parsing...")

    if LLM_name is None or type is None:
        return None, f"Primary parsing failed: {error_note}. No fallback LLM specified."

    try:
        # Create fallback prompt
        fallback_prompt = create_fallback_prompt(response_text)

        # Make fallback LLM call
        fallback_response = query_llm(fallback_prompt, type, LLM_name)

        # Parse the fallback response
        fallback_classification, fallback_error = extract_final_answer_primary(fallback_response, valid_answers)

        if fallback_classification is not None:
            return fallback_classification, f"Fallback success (Primary failed: {error_note})"
        else:
            return None, f"Both primary and fallback failed. Primary: {error_note}, Fallback: {fallback_error}"

    except Exception as e:
        return None, f"Primary parsing failed: {error_note}. Fallback LLM error: {str(e)}"


def extract_final_answer_primary(response_text, valid_answers):
    """
    Primary extraction method (your existing logic)
    """
    # Search for all occurrences of 'Final Answer:' (case-insensitive)
    pattern = r'Final\s+Answer\s*:'
    matches = list(re.finditer(pattern, response_text, re.IGNORECASE))

    # Check if we found any matches
    if not matches:
        return None, "Bad LLM response"

    # If multiple matches found, use the last one
    if len(matches) > 1:
        final_answer_match = matches[-1]  # Get the last occurrence
        position = final_answer_match.end()  # Position after 'Final Answer:'
    else:
        final_answer_match = matches[0]
        position = final_answer_match.end()

    # Extract all text after 'Final Answer:'
    remaining_text = response_text[position:]

    # Find all valid answers that appear as substrings (case-insensitive)
    found_answers = []
    for valid in valid_answers:
        if valid.lower() in remaining_text.lower():
            found_answers.append(valid)

    # Check results
    if len(found_answers) == 0:
        return None, "No valid classification found after 'Final Answer:'"
    elif len(found_answers) > 1:
        return None, "Multiple classifications found"
    else:
        return found_answers[0], None  # Success - return the exact valid answer


def create_fallback_prompt(original_response):
    """
    Create a clear, structured prompt for the fallback LLM
    """
    fallback_prompt = f"""Please analyze the following response and provide a clear classification.

Original Response:
{original_response}

Based on the analysis in the original response above, select exactly ONE of these four options:
Support - if supporting evidence is strongest
Contradict - if contradicting evidence is strongest
Neutral - if evidence is mixed, weak, or irrelevant
Maybe - if you cannot determine

Please format your answer as
###Final Answer: [Write the corresponding word]

Example: ###Final Answer: Neutral

Please provide your classification now:"""

    return fallback_prompt
