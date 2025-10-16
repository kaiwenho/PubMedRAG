import re
from typing import List, Dict, Tuple, Optional
import pandas as pd
import unicodedata
from difflib import SequenceMatcher
import ast


def generate_correction_prompt(error_case):
    return f"""You are a helpful assistant. Your task is to map LLM-generated sentences back to their original source sentences in an abstract.

CONTEXT: An LLM was instructed to extract exact sentences from an abstract, but it sometimes paraphrased, combined, or slightly modified them instead.

INPUT:
- Unmatched sentences: {error_case['unmatched_sentences']} (these are the LLM's outputs that should have been exact quotes)
- Abstract sentences: {error_case['abstract_sentences']} (the original source text)

TASK: For each unmatched sentence, identify which abstract sentence(s) it corresponds to based on semantic content and text similarity.

IMPORTANT:
- Do not correct imperfect sentence segmentation in the abstract - use it as-is
- One unmatched sentence may map to multiple abstract sentences
- Focus on finding the source content, even if wording differs
- If you cannot find any correct matched sentences in the abstract for an unmatched sentence, return an empty list for that sentence

OUTPUT: Return a Python list of the corresponding abstract sentences for each unmatched sentence, in order.
Example: "[sentence1, sentence2, ...]"
If no matches are found, return: "[]"

Which abstract sentences do the unmatched sentences map to?
"""

def process_with_llm_fallback(row, abstracts_dict, client, model='gpt-oss:20b'):
    """
    Process a single row, using LLM correction if needed.
    Returns: (indices, success_flag)
    """
    # First attempt with process_edge
    indices, error = process_edge(row, abstracts_dict, return_partial_matches=True)

    if error is None:
        return indices, True

    # If there's an error, try LLM correction
    try:
        prompt = generate_correction_prompt(error)
        messages = [{
            'role': 'user',
            'content': prompt,
        }]
        response = client.chat(
            model=model,
            messages=messages,
            options={'num_ctx': 8192},
        )

        # Build corrected sentences
        matched_sents = [error['abstract_sentences'][idx]
                        for idx in error.get('partial_matches', [])]
        corrected_sents = parse_llm_list(response['message']['content'])
        corrected_sentences = matched_sents + corrected_sents

        # Create corrected row and retry
        corrected_row = row.copy()
        # Apply your correction logic here (depends on your apply_manual_corrections_dict implementation)
        corrected_row['support_abstract_sentences'] = corrected_sentences

        indices_retry, error_retry = process_edge(corrected_row, abstracts_dict, return_partial_matches=True)

        if error_retry is None:
            return indices_retry, True
        else:
            print(f"Warning: Row {row['edge_index']}, {row['pmid']} failed even after LLM correction")
            return indices_retry, False

    except Exception as e:
        print(f"Error processing row {row['edge_index']}, {row['pmid']} with LLM: {e}")
        return indices, False

def fix_specific_rows(df, row_indices, abstracts_dict, client, reprocess=True):
    """
    Fix specific rows without reprocessing the entire dataframe.

    Args:
        df: The dataframe
        row_indices: List of row indices to fix
        abstracts_dict: Your abstracts dictionary
        client: LLM client
        reprocess: Whether to reprocess with process_edge after manual correction

    Returns:
        Updated dataframe
    """
    df_copy = df.copy()

    for idx in row_indices:
        row = df_copy.loc[idx]

        # Process this single row
        indices, success = process_with_llm_fallback(row, abstracts_dict, client, model='gpt-oss:120b')

        # Update only this row
        df_copy.at[idx, 'gold_sent_idxs'] = indices
        df_copy.at[idx, 'mapping_success'] = success

        if success:
            print(f"Row {idx} fixed successfully")
        else:
            print(f"Row {idx} still failing")

    return df_copy

def parse_llm_list(llm_output):
    """
    Parse LLM output into a Python list, handling common formatting issues.

    Returns:
        List of strings on success
        Raises ValueError if parsing fails completely
    """
    original_output = llm_output  # Keep for error reporting

    # Strip whitespace
    llm_output = llm_output.strip()

    # Remove markdown code blocks if present
    if llm_output.startswith('```'):
        lines = llm_output.split('\n')
        lines = lines[1:]  # Remove first line (```python or ```)
        if lines and lines[-1].strip() == '```':
            lines = lines[:-1]  # Remove last line if it's ```
        llm_output = '\n'.join(lines).strip()

    # Try to find a list in the output
    list_match = re.search(r'\[.*\]', llm_output, re.DOTALL)
    if list_match:
        llm_output = list_match.group(0)

    try:
        result = ast.literal_eval(llm_output)
    except (SyntaxError, ValueError) as e:
        # If parsing fails, raise an exception with details
        print(f"Failed to parse LLM output: {original_output[:200]}...")
        print(f"Parse error: {e}")
        raise ValueError(f"Could not parse LLM response as Python list: {e}")

    def flatten_to_strings(item):
        if isinstance(item, str):
            return [item]
        elif isinstance(item, list):
            result = []
            for subitem in item:
                result.extend(flatten_to_strings(subitem))
            return result
        else:
            return []

    return flatten_to_strings(result)

def compare_llm_decisions(list1, list2):
    """
    Compare decisions from two LLMs for the same edges (same index and pmid).

    Args:
        list1: List of dictionaries from first LLM
        list2: List of dictionaries from second LLM

    Returns:
        Dictionary containing comparison results
    """
    # Create dictionaries for quick lookup by (index, pmid)
    llm1_decisions = {(item['index'], item['pmid']): item['extracted_data']['Support?']
                     for item in list1 if item.get('extraction_status') == 'success'}

    llm2_decisions = {(item['index'], item['pmid']): item['extracted_data']['Support?']
                     for item in list2 if item.get('extraction_status') == 'success'}

    # Find all unique (index, pmid) combinations
    all_keys = set(llm1_decisions.keys()) | set(llm2_decisions.keys())

    # Compare decisions
    agreements = []
    disagreements = []
    only_in_llm1 = []
    only_in_llm2 = []

    for key in all_keys:
        index, pmid = key
        llm1_decision = llm1_decisions.get(key)
        llm2_decision = llm2_decisions.get(key)

        if llm1_decision is None:
            only_in_llm2.append({
                'index': index,
                'pmid': pmid,
                'llm2_decision': llm2_decision
            })
        elif llm2_decision is None:
            only_in_llm1.append({
                'index': index,
                'pmid': pmid,
                'llm1_decision': llm1_decision
            })
        elif llm1_decision == llm2_decision:
            agreements.append({
                'index': index,
                'pmid': pmid,
                'decision': llm1_decision
            })
        else:
            disagreements.append({
                'index': index,
                'pmid': pmid,
                'llm1_decision': llm1_decision,
                'llm2_decision': llm2_decision
            })

    # Count disagreement patterns
    disagreement_patterns = {}
    for item in disagreements:
        pattern = f"{item['llm1_decision']} -> {item['llm2_decision']}"
        disagreement_patterns[pattern] = disagreement_patterns.get(pattern, 0) + 1

    return {
        'agreements': agreements,
        'disagreements': disagreements,
        'only_in_llm1': only_in_llm1,
        'only_in_llm2': only_in_llm2,
        'disagreement_patterns': disagreement_patterns,
        'summary': {
            'total_compared': len(agreements) + len(disagreements),
            'agreements': len(agreements),
            'disagreements': len(disagreements),
            'only_in_llm1': len(only_in_llm1),
            'only_in_llm2': len(only_in_llm2)
        }
    }

def get_disagreement_indices(results):
    """Get the (index, pmid) tuples where LLMs disagree."""
    return [(item['index'], item['pmid']) for item in results['disagreements']]

def get_keys_for_pattern(results, pattern):
    """
    Get (index, pmid) tuples for a specific disagreement pattern.

    Args:
        results: Results from compare_llm_decisions()
        pattern: String like "yes -> maybe" or "maybe -> no"

    Returns:
        List of (index, pmid) tuples matching the pattern
    """
    # Parse the pattern
    if " -> " not in pattern:
        print(f"Error: Pattern should be in format 'decision1 -> decision2', got: {pattern}")
        return []

    llm1_decision, llm2_decision = pattern.split(" -> ")
    llm1_decision = llm1_decision.strip()
    llm2_decision = llm2_decision.strip()

    # Find matching keys
    matching_keys = []
    for item in results['disagreements']:
        if item['llm1_decision'] == llm1_decision and item['llm2_decision'] == llm2_decision:
            matching_keys.append((item['index'], item['pmid']))

    return matching_keys

def show_pattern_keys(results, pattern):
    """
    Show (index, pmid) tuples for a specific disagreement pattern with summary.

    Args:
        results: Results from compare_llm_decisions()
        pattern: String like "yes -> maybe" or "maybe -> no"
    """
    keys = get_keys_for_pattern(results, pattern)

    if keys:
        print(f"=== Keys for pattern '{pattern}' ===")
        print(f"Count: {len(keys)}")
        print(f"Keys (index, pmid): {keys}")
    else:
        print(f"No disagreements found for pattern '{pattern}'")
        available_patterns = list(results['disagreement_patterns'].keys())
        if available_patterns:
            print(f"Available patterns: {available_patterns}")
        else:
            print("No disagreement patterns available.")

def print_comparison_summary(results):
    """Print a summary of the comparison results."""
    print("=== LLM Decision Comparison Summary ===")
    print(f"Total edges compared: {results['summary']['total_compared']}")
    print(f"Agreements: {results['summary']['agreements']}")
    print(f"Disagreements: {results['summary']['disagreements']}")
    print(f"Only in LLM1: {results['summary']['only_in_llm1']}")
    print(f"Only in LLM2: {results['summary']['only_in_llm2']}")

    if results['summary']['disagreements'] > 0:
        agreement_rate = results['summary']['agreements'] / results['summary']['total_compared']
        print(f"Agreement rate: {agreement_rate:.2%}")

    if results['disagreements']:
        print(f"\n=== Disagreements ===")
        print(results['disagreement_patterns'])

def normalize_sentence(sentence: str) -> str:
    """
    Normalize a sentence by:
    - Removing extra whitespace
    - Standardizing punctuation
    - Normalizing Unicode characters (hyphens, dashes, quotes, etc.)
    """
    # Strip leading/trailing whitespace
    sentence = sentence.strip()

    # Normalize Unicode characters to their closest ASCII equivalent
    # This handles special hyphens, dashes, quotes, etc.
    sentence = unicodedata.normalize('NFKD', sentence)

    # Replace various types of hyphens and dashes with standard hyphen
    # U+2010 (‐), U+2011 (‑), U+2012 (‒), U+2013 (–), U+2014 (—), U+2015 (―)
    sentence = re.sub(r'[\u2010-\u2015]', '-', sentence)

    # Replace various types of quotes with standard quotes
    sentence = re.sub(r'[''‚]', "'", sentence)  # Single quotes
    sentence = re.sub(r'[""„]', '"', sentence)  # Double quotes

    # Normalize backslash-escaped characters (common in text processing)
    sentence = sentence.replace('\\/', '/')  # \/ -> /
    sentence = sentence.replace('\\or', '/or')  # \or -> /or (for and\or -> and/or)
    sentence = sentence.replace('\\and', '/and')  # \and -> /and

    # Replace multiple spaces with single space
    sentence = re.sub(r'\s+', ' ', sentence)

    # Standardize common punctuation issues
    sentence = re.sub(r'\s+([.,;:!?])', r'\1', sentence)  # Remove space before punctuation

    return sentence

def is_exact_match(sent1: str, sent2: str) -> bool:
    """
    Check if two sentences match after normalization (ignoring case, extra spaces, punctuation).
    Returns True only if words are identical.
    """
    norm1 = normalize_sentence(sent1)
    norm2 = normalize_sentence(sent2)

    # Case-insensitive comparison
    if norm1.lower() == norm2.lower():
        return True

    # Also check with punctuation removed for edge cases
    words1 = re.findall(r'\b\w+\b', norm1.lower())
    words2 = re.findall(r'\b\w+\b', norm2.lower())

    return words1 == words2

def string_similarity(str1: str, str2: str) -> float:
    """Calculate similarity ratio between two strings (0-1)."""
    return SequenceMatcher(None, str1.lower(), str2.lower()).ratio()

def find_sentence_with_ellipsis(ellipsis_sent: str, sentences_list: List[str]) -> Optional[List[int]]:
    """
    Find sentence(s) that match the ellipsis pattern:
    - '...' at the end: find sentence that starts with the given text
    - '...' at the beginning: find sentence that ends with the given text
    - One or more '...' in the middle/throughout: find sentence(s) containing all parts in order
      AND include all sentences in between (the ellipsis implies continuous support)

    Returns the list of indices if found, None otherwise.
    Note: Can return multiple indices when ellipsis spans multiple sentences.
    """
    if '...' not in ellipsis_sent:
        return None

    norm_ellipsis = normalize_sentence(ellipsis_sent)

    # Case 1: Ellipsis at the end only (truncated ending)
    if norm_ellipsis.endswith('...') and norm_ellipsis.count('...') == 1:
        prefix = normalize_sentence(norm_ellipsis[:-3]).strip()
        if prefix:
            # Remove trailing period from prefix for matching
            prefix_no_period = prefix.rstrip('.')
            for idx, sentence in enumerate(sentences_list):
                norm_sent = normalize_sentence(sentence)
                if norm_sent.lower().startswith(prefix_no_period.lower()):
                    return [idx]
        return None

    # Case 2: Ellipsis at the beginning only (truncated beginning)
    if norm_ellipsis.startswith('...') and norm_ellipsis.count('...') == 1:
        suffix = normalize_sentence(norm_ellipsis[3:]).strip()
        if suffix:
            for idx, sentence in enumerate(sentences_list):
                norm_sent = normalize_sentence(sentence)
                if norm_sent.lower().endswith(suffix.lower()):
                    return [idx]
        return None

    # Case 3: One or more '...' throughout the sentence (omitted portions)
    # Split by '...' and check if all parts appear in order
    parts = [normalize_sentence(part).strip() for part in ellipsis_sent.split('...')]
    parts = [p.rstrip('.').lstrip('.').strip() for p in parts if p.strip()]  # Remove empty parts and periods

    if len(parts) < 2:
        return None

    # Strategy 1: Try to find a single sentence containing all parts
    for idx, sentence in enumerate(sentences_list):
        norm_sent = normalize_sentence(sentence).lower()

        # Check if all parts appear in order in this single sentence
        current_pos = 0
        all_found = True

        for part in parts:
            part_lower = part.lower().strip('.,;:')
            pos = norm_sent.find(part_lower, current_pos)
            if pos == -1:
                all_found = False
                break
            current_pos = pos + len(part_lower)

        if all_found:
            return [idx]

    # Strategy 2: Find multiple sentences that contain the parts
    # and return ALL indices from first to last (inclusive)
    matched_indices = []

    for part_idx, part in enumerate(parts):
        part_lower = part.lower().strip('.,;:')
        found_for_this_part = False

        for idx, sentence in enumerate(sentences_list):
            norm_sent = normalize_sentence(sentence).lower()

            # Check if this part appears in this sentence
            if part_lower in norm_sent:
                matched_indices.append(idx)
                found_for_this_part = True
                break

            # Check if part spans this sentence and the next (boundary case)
            if idx < len(sentences_list) - 1:
                # Combine current and next sentence
                next_sent = normalize_sentence(sentences_list[idx + 1]).lower()
                combined = norm_sent + ' ' + next_sent

                if part_lower in combined:
                    # Part spans both sentences
                    matched_indices.append(idx)
                    matched_indices.append(idx + 1)
                    found_for_this_part = True
                    break

        if not found_for_this_part:
            # If any part doesn't match, the whole thing fails
            return None

    # Verify that matched indices are in ascending order
    if matched_indices != sorted(matched_indices) or len(matched_indices) < 2:
        return None

    # Return ALL indices from first to last (inclusive)
    # The '...' implies all sentences in between are also relevant
    first_idx = matched_indices[0]
    last_idx = matched_indices[-1]
    return list(range(first_idx, last_idx + 1))

def find_matching_sentence_index(llm_sent: str, sentences_list: List[str]) -> Optional[int]:
    """
    Find the index of a matching sentence, including cases where:
    - LLM sentence is an exact match (normalized)
    - LLM sentence is a substring of an original sentence (prefix or anywhere within)
    - LLM sentence is very similar to original (fuzzy match for typos)

    Returns the index if found, None otherwise.
    """
    norm_llm = normalize_sentence(llm_sent)

    for idx, orig_sent in enumerate(sentences_list):
        norm_orig = normalize_sentence(orig_sent)

        # Exact match (case-insensitive, normalized)
        if is_exact_match(llm_sent, orig_sent):
            return idx

        # Prepare LLM sentence for substring matching
        llm_for_matching = norm_llm.lower()

        # Remove trailing period for matching
        if llm_for_matching.endswith('.'):
            llm_for_matching = llm_for_matching[:-1].strip()

        # Check if LLM sentence is a substring of the original sentence
        if llm_for_matching in norm_orig.lower():
            return idx

        # Fuzzy matching as fallback (for typos, minor differences)
        similarity = string_similarity(llm_for_matching, norm_orig)
        if similarity >= 0.95:  # 95% similar
            return idx

        # Also check if removing the period makes it an exact match
        if llm_for_matching == norm_orig.lower().rstrip('.'):
            return idx

    return None

def match_llm_sentences_to_indices(
    llm_sentences: List[str],
    abstract_dict: Dict[str, any],
    edge_index: any,
    pmid: str,
    return_partial_matches: bool = True  # NEW PARAMETER
) -> Tuple[List[int], Optional[Dict]]:
    """
    Match LLM-generated sentences to their indices in the abstract's sentence list.

    Args:
        llm_sentences: List of sentences generated by LLM
        abstract_dict: Dictionary with 'abstract' and 'sentences' keys
        edge_index: The edge identifier for error logging
        pmid: The PMID for error logging
        return_partial_matches: If True, return partial matches even when some sentences fail

    Returns:
        Tuple of (indices_list, error_dict)
        - indices_list: List of matched indices (empty if manual review needed and return_partial_matches=False)
        - error_dict: None if successful, otherwise dict with error info for manual review
    """
    sentences_list = abstract_dict['sentences']
    abstract_text = abstract_dict['abstract']

    # Step 1: Check if all LLM sentences match sentences in the list (exact or prefix match)
    indices = []
    unmatched_sentences = []

    for llm_sent in llm_sentences:
        idx = find_matching_sentence_index(llm_sent, sentences_list)
        if idx is not None:
            indices.append(idx)
        else:
            unmatched_sentences.append(llm_sent)

    # If all matched, return success
    if len(unmatched_sentences) == 0:
        return sorted(set(indices)), None  # Remove duplicates and sort

    # Step 2: Check if unmatched sentences are in the abstract (wrong segmentation)
    resegmented_indices = []
    still_unmatched = []

    for llm_sent in unmatched_sentences:
        norm_llm = normalize_sentence(llm_sent)
        norm_abstract = normalize_sentence(abstract_text)

        # Check if this sentence appears in the abstract (use normalized version)
        if norm_llm.lower() in norm_abstract.lower():
            # Try to find which sentence(s) it overlaps with
            # Don't break after first match - collect ALL matching sentences
            matched_indices = []

            for idx, orig_sent in enumerate(sentences_list):
                norm_orig = normalize_sentence(orig_sent)
                # Check for substantial overlap
                if norm_llm.lower() in norm_orig.lower() or norm_orig.lower() in norm_llm.lower():
                    matched_indices.append(idx)

            if matched_indices:
                resegmented_indices.extend(matched_indices)
            else:
                still_unmatched.append(llm_sent)
        else:
            still_unmatched.append(llm_sent)

    # Update indices with resegmented matches
    indices.extend(resegmented_indices)

    # If all matched after resegmentation, return success
    if len(still_unmatched) == 0:
        return sorted(set(indices)), None

    # Step 3: Check for ellipsis usage
    ellipsis_indices = []
    final_unmatched = []

    for llm_sent in still_unmatched:
        if '...' in llm_sent:
            result = find_sentence_with_ellipsis(llm_sent, sentences_list)
            if result is not None:
                # result is a list of indices, extend rather than append
                ellipsis_indices.extend(result)
            else:
                final_unmatched.append(llm_sent)
        else:
            final_unmatched.append(llm_sent)

    # Update indices with ellipsis matches
    indices.extend(ellipsis_indices)

    # If all matched after ellipsis handling, return success
    if len(final_unmatched) == 0:
        return sorted(set(indices)), None

    # Manual review needed - prepare error info
    error_dict = {
        'edge_index': edge_index,
        'pmid': pmid,
        'llm_generated_sentences': llm_sentences,
        'unmatched_sentences': final_unmatched,
        'abstract_sentences': sentences_list,
        'partial_matches': sorted(set(indices)) if indices else []
    }

    # NEW: Return partial matches if requested
    if return_partial_matches and indices:
        return sorted(set(indices)), error_dict
    else:
        return [], error_dict

# Convenience function for processing a dataframe row
def process_edge(edge_row: pd.Series, abstracts_dict: Dict[str, Dict],
                return_partial_matches: bool = True) -> Tuple[List[int], Optional[Dict]]:
    """
    Process a single edge row from the dataframe.

    Args:
        edge_row: A row from the edges dataframe
        abstracts_dict: Dictionary mapping PMID to abstract dictionaries
        return_partial_matches: If True, return partial matches even when some sentences fail

    Returns:
        Tuple of (indices_list, error_dict)
    """
    support_abstract = edge_row['abstract_support?']
    if support_abstract != 'yes':
        return [], None

    pmid = edge_row['pmid']
    edge_index = edge_row['edge_index']
    llm_sentences = edge_row['support_abstract_sentences']

    # Get the abstract dictionary for this PMID
    if pmid not in abstracts_dict:
        return [], {
            'edge_index': edge_index,
            'pmid': pmid,
            'error': f'PMID {pmid} not found in abstracts dictionary'
        }

    abstract_dict = abstracts_dict[pmid]

    return match_llm_sentences_to_indices(
        llm_sentences,
        abstract_dict,
        edge_index,
        pmid,
        return_partial_matches=return_partial_matches
    )
