import json
import pandas as pd
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
from nltk.tokenize import sent_tokenize
import asyncio
from pathlib import Path
import logging
from datetime import datetime

# Setup logging
def setup_logging(output_dir: Path):
    """Setup logging to both file and console."""
    log_file = output_dir / f"validation_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )

    # Suppress HTTP request logs from third-party libraries
    logging.getLogger('httpx').setLevel(logging.WARNING)
    logging.getLogger('httpcore').setLevel(logging.WARNING)
    logging.getLogger('urllib3').setLevel(logging.WARNING)
    logging.getLogger('requests').setLevel(logging.WARNING)

    return logging.getLogger(__name__)

class LLMValidator:
    """
    Sophisticated LLM validator with two-round validation and sentence mapping.
    Adapted to work with the pub_extension pipeline workflow.
    """

    def __init__(
        self,
        llm_client: Any,
        node_dict: Dict[str, Any],
        predicate_dict: Dict[str, Any],
        response_parser: Any,
        round1_model: str = 'gpt-oss:20b',
        round2_model: str = 'gpt-oss:120b',
        context_window: int = 8192,
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize the LLMValidator.

        Args:
            llm_client: LLM client with chat method (e.g., Ollama Client)
            node_dict: Dictionary with node details for enrichment
            predicate_dict: Dictionary with predicate details for enrichment
            response_parser: Parser with parse_file, save_results methods
            round1_model: Model name for initial validation
            round2_model: Model name for refined validation
            context_window: Context window size for LLM
            logger: Logger instance for progress tracking
        """
        self.llm_client = llm_client
        self.node_dict = node_dict
        self.predicate_dict = predicate_dict
        self.response_parser = response_parser
        self.round1_model = round1_model
        self.round2_model = round2_model
        self.context_window = context_window
        self.logger = logger or logging.getLogger(__name__)

        # Storage for intermediate results
        self.prompts = []
        self.round1_responses = []
        self.round2_responses = []
        self.round1_results = []
        self.round2_results = []
        self.merged_results = None
        self.validation_df = None

        # Track failures
        self.failed_pmids = []
        self.missing_info_edges = []

    async def get_abstracts_dict(
        self,
        pmids: List[str],
        batch_size: int = 100
    ) -> Dict[str, Dict]:
        """
        Retrieve abstracts from PubMed for a list of PMIDs and segment into sentences.

        Args:
            pmids: List of PMIDs (e.g., ['PMID:12345', 'PMID:67890'])
            batch_size: Number of PMIDs to process in each batch (default: 100)

        Returns:
            Dictionary mapping PMID to abstract data:
            {
                'PMID:12345': {
                    'abstract': 'Full abstract text...',
                    'sentences': ['Sentence 1.', 'Sentence 2.', ...]
                }
            }
        """
        from pubmed_client import get_publication_info

        self.logger.info(f"Retrieving abstracts for {len(pmids)} unique PMIDs")

        abstracts_dict = {}
        failed_pmids = []

        # Process PMIDs in batches to avoid API limits
        total_batches = (len(pmids) + batch_size - 1) // batch_size

        for i in range(0, len(pmids), batch_size):
            batch_pmids = pmids[i:i + batch_size]
            batch_num = i // batch_size + 1

            self.logger.info(f"Processing batch {batch_num}/{total_batches}: {len(batch_pmids)} PMIDs")

            try:
                abstracts_info = await get_publication_info(batch_pmids, 'placeholder')

                if abstracts_info['_meta']['n_results'] > 0:
                    abstracts = abstracts_info['results']

                    for pmid in batch_pmids:
                        abstract = abstracts.get(pmid, {})
                        abstract_text = abstract.get('abstract')

                        if abstract_text:
                            # Clean abstract and skip if it's empty placeholder
                            if abstract_text != '-\n':
                                # Tokenize into sentences
                                sentences = sent_tokenize(abstract_text)
                                abstracts_dict[pmid] = {
                                    'abstract': abstract_text,
                                    'sentences': sentences
                                }
                            else:
                                self.logger.warning(f"Empty placeholder abstract for {pmid}")
                                failed_pmids.append(pmid)
                        else:
                            self.logger.warning(f"No abstract found for {pmid}")
                            failed_pmids.append(pmid)
                else:
                    self.logger.warning(f"No results returned for batch {batch_num}")
                    failed_pmids.extend(batch_pmids)

            except Exception as e:
                self.logger.error(f"Error processing batch {batch_num}: {e}")
                failed_pmids.extend(batch_pmids)
                continue

        self.failed_pmids = failed_pmids
        self.logger.info(f"Successfully retrieved: {len(abstracts_dict)}/{len(pmids)} abstracts")
        if failed_pmids:
            self.logger.warning(f"Failed to retrieve: {len(failed_pmids)} abstracts")

        return abstracts_dict

    def generate_prompt(
        self,
        subj_info: Dict,
        obj_info: Dict,
        pred_info: Dict,
        pred: str,
        abstract: str
    ) -> str:
        """Generate a validation prompt for an edge-abstract pair."""
        return f"""Please analyze whether the provided abstract supports the following edge.
Carefully consider the subject, object, and predicate details.

Edge: {subj_info['name']} --{pred}-> {obj_info['name']}
Subject: {subj_info}
Object: {obj_info}
Predicate: {pred_info}

Abstract:
{abstract}

Instructions:
- Determine if the abstract provides evidence for this edge.
- Use "yes" if the relation is explicitly supported.
- Use "no" if the relation is not mentioned or contradicted.
- Use "maybe" if the evidence is indirect, ambiguous, or suggestive.
- If "Support?" is "yes", return one or more exact supporting sentences from the abstract.
- If "Support?" is "no" or "maybe", return an empty list for "Sentences".

Output Format: Return only a JSON object in the following structure:
{{
  "Support?": "yes" | "no" | "maybe",
  "Sentences": ["..."]  // one or more if yes, [] if no/maybe
}}
"""

    def prepare_prompts(
        self,
        edges_df: pd.DataFrame,
        abstracts_dict: Dict[str, Dict]
    ) -> List[Dict]:
        """
        Prepare validation prompts for edges with supporting PMIDs.

        NEW: Works with dataframe that has 'publications' column containing list of PMIDs.

        Args:
            edges_df: DataFrame of edges with 'publications' column containing list of PMIDs
            abstracts_dict: Dict mapping PMID to abstract data

        Returns:
            List of prompt dictionaries with keys: edge_index, pmid, prompt
        """
        self.logger.info("Preparing validation prompts...")

        prompts = []
        edges_processed = 0
        edges_skipped = 0
        prompts_generated = 0

        total_edges = len(edges_df)

        for idx, row in edges_df.iterrows():
            edges_processed += 1

            # Log progress every 100 edges
            if edges_processed % 100 == 0:
                self.logger.info(f"Processing edge {edges_processed}/{total_edges}")

            # Get subject, object, predicate info
            subj = row['subject']
            subj_info = self.node_dict.get(subj)

            if not subj_info:
                self.logger.debug(f"Edge {idx}: Missing subject info for {subj}")
                self.missing_info_edges.append({'edge_index': idx, 'reason': f'Missing subject: {subj}'})
                edges_skipped += 1
                continue

            obj = row['object']
            obj_info = self.node_dict.get(obj)

            if not obj_info:
                self.logger.debug(f"Edge {idx}: Missing object info for {obj}")
                self.missing_info_edges.append({'edge_index': idx, 'reason': f'Missing object: {obj}'})
                edges_skipped += 1
                continue

            pred = row['predicate']
            pred_info = self.predicate_dict.get(pred)

            if not pred_info:
                self.logger.debug(f"Edge {idx}: Missing predicate info for {pred}")
                self.missing_info_edges.append({'edge_index': idx, 'reason': f'Missing predicate: {pred}'})
                edges_skipped += 1
                continue

            # Get publications list
            publications = row['publications']

            # Filter for valid PMIDs
            if isinstance(publications, (list, np.ndarray)):
                pmids = [pmid for pmid in publications if isinstance(pmid, str) and pmid.startswith('PMID:')]
            else:
                self.logger.debug(f"Edge {idx}: publications is not a list/array: {type(publications)}")
                edges_skipped += 1
                continue

            if not pmids:
                self.logger.debug(f"Edge {idx}: No valid PMIDs found")
                edges_skipped += 1
                continue

            # Generate prompts for each PMID
            for pmid in pmids:
                # Get abstract from cache
                abstract_data = abstracts_dict.get(pmid)

                if not abstract_data:
                    self.logger.debug(f"Edge {idx}, PMID {pmid}: Abstract not in cache")
                    continue

                abstract = abstract_data.get('abstract')
                if not abstract:
                    continue

                # Generate prompt
                prompt = self.generate_prompt(
                    subj_info, obj_info, pred_info, pred, abstract
                )

                prompts.append({
                    'edge_index': idx,
                    'pmid': pmid,
                    'prompt': prompt
                })
                prompts_generated += 1

        self.prompts = prompts

        self.logger.info(f"Prompt generation complete:")
        self.logger.info(f"  Total edges: {total_edges}")
        self.logger.info(f"  Edges processed: {edges_processed}")
        self.logger.info(f"  Edges skipped (missing info): {edges_skipped}")
        self.logger.info(f"  Total prompts generated: {prompts_generated}")

        if self.missing_info_edges:
            self.logger.warning(f"  {len(self.missing_info_edges)} edges missing subject/object/predicate info")

        return prompts

    def run_validation_round(
        self,
        prompts: List[Dict],
        model: str
    ) -> Tuple[List[Dict], List[Dict]]:
        """
        Run a validation round using the specified model.

        Args:
            prompts: List of prompt dictionaries
            model: Model name to use

        Returns:
            Tuple of (responses, results)
        """
        responses = []
        results = []

        total = len(prompts)
        self.logger.info(f"Running validation with {model} on {total} prompts")

        for i, prompt_data in enumerate(prompts, 1):
            if i % 50 == 0:
                self.logger.info(f"  Progress: {i}/{total} prompts")

            try:
                response = self.llm_client.chat(
                    model=model,
                    messages=[{'role': 'user', 'content': prompt_data['prompt']}],
                    options={'temperature': 0}
                )

                response_text = response['message']['content']

                # Store response
                response_entry = {
                    'edge_index': prompt_data['edge_index'],
                    'pmid': prompt_data['pmid'],
                    'response': response_text
                }
                responses.append(response_entry)

                # Parse response
                try:
                    extracted_data = self.response_parser.parse_response(response_text)
                    result_entry = {
                        'edge_index': prompt_data['edge_index'],
                        'pmid': prompt_data['pmid'],
                        'extraction_status': 'success',
                        'extracted_data': extracted_data
                    }
                except Exception as parse_error:
                    self.logger.debug(f"Parse error for edge {prompt_data['edge_index']}, PMID {prompt_data['pmid']}: {parse_error}")
                    result_entry = {
                        'edge_index': prompt_data['edge_index'],
                        'pmid': prompt_data['pmid'],
                        'extraction_status': 'parse_failed',
                        'error': str(parse_error)
                    }

                results.append(result_entry)

            except Exception as e:
                self.logger.error(f"LLM call failed for edge {prompt_data['edge_index']}, PMID {prompt_data['pmid']}: {e}")
                responses.append({
                    'edge_index': prompt_data['edge_index'],
                    'pmid': prompt_data['pmid'],
                    'response': None,
                    'error': str(e)
                })
                results.append({
                    'edge_index': prompt_data['edge_index'],
                    'pmid': prompt_data['pmid'],
                    'extraction_status': 'llm_failed',
                    'error': str(e)
                })

        self.logger.info(f"Completed {model} validation round")
        return responses, results

    async def run_round1(self):
        """Run Round 1 validation."""
        self.logger.info("\n" + "="*50)
        self.logger.info("ROUND 1: Initial Validation")
        self.logger.info("="*50)

        self.round1_responses, self.round1_results = self.run_validation_round(
            self.prompts,
            self.round1_model
        )

        # Separate 'yes'/'maybe' and 'no' results
        supporting_results = []
        no_support_results = []

        for result in self.round1_results:
            if result['extraction_status'] == 'success':
                support = result['extracted_data'].get('support?', '').lower()
                if support in ['yes', 'maybe']:
                    supporting_results.append(result)
                elif support == 'no':
                    no_support_results.append(result)

        self.logger.info(f"\nRound 1 Results:")
        self.logger.info(f"  Supporting ('yes'/'maybe'): {len(supporting_results)}")
        self.logger.info(f"  Non-supporting ('no'): {len(no_support_results)}")
        self.logger.info(f"  Failed: {len(self.round1_results) - len(supporting_results) - len(no_support_results)}")

        self.no_support_results = no_support_results

        return supporting_results

    async def run_round2(self):
        """Run Round 2 validation on 'yes' and 'maybe' results."""
        self.logger.info("\n" + "="*50)
        self.logger.info("ROUND 2: Refined Validation")
        self.logger.info("="*50)

        # Filter for 'yes' and 'maybe' from Round 1
        round2_prompts = []
        for result, prompt_data in zip(self.round1_results, self.prompts):
            if result['extraction_status'] == 'success':
                support = result['extracted_data'].get('support?', '').lower()
                if support in ['yes', 'maybe']:
                    round2_prompts.append(prompt_data)

        self.logger.info(f"Running Round 2 on {len(round2_prompts)} supporting results")

        if not round2_prompts:
            self.logger.info("No supporting results to validate in Round 2")
            self.round2_responses = []
            self.round2_results = []
            return []

        self.round2_responses, self.round2_results = self.run_validation_round(
            round2_prompts,
            self.round2_model
        )

        # Count Round 2 outcomes
        r2_yes_maybe = sum(
            1 for r in self.round2_results
            if r['extraction_status'] == 'success'
            and r['extracted_data'].get('support?', '').lower() in ['yes', 'maybe']
        )
        r2_no = sum(
            1 for r in self.round2_results
            if r['extraction_status'] == 'success'
            and r['extracted_data'].get('support?', '').lower() == 'no'
        )

        self.logger.info(f"\nRound 2 Results:")
        self.logger.info(f"  Confirmed supporting: {r2_yes_maybe}")
        self.logger.info(f"  Changed to non-supporting: {r2_no}")

        return self.round2_results

    def merge_results(self):
        """Merge Round 1 and Round 2 results."""
        self.logger.info("\n" + "="*50)
        self.logger.info("MERGING RESULTS")
        self.logger.info("="*50)

        # Create lookup for Round 2 results
        round2_dict = {}
        for result in self.round2_results:
            key = (result['edge_index'], result['pmid'])
            round2_dict[key] = result

        merged = []
        additional_no_support = []

        for result in self.round1_results:
            if result['extraction_status'] == 'success':
                support = result['extracted_data'].get('support?', '').lower()

                if support in ['yes', 'maybe']:
                    key = (result['edge_index'], result['pmid'])
                    round2_result = round2_dict.get(key)

                    if round2_result and round2_result['extraction_status'] == 'success':
                        round2_support = round2_result['extracted_data'].get('support?', '').lower()
                        if round2_support == 'no':
                            additional_no_support.append(round2_result)
                        else:
                            merged.append(round2_result)
                    else:
                        merged.append(result)
                elif support == 'no':
                    pass  # Already in no_support_results

        if additional_no_support:
            self.logger.info(f"Round 2 changed {len(additional_no_support)} results from supporting to non-supporting")
            if not hasattr(self, 'no_support_results'):
                self.no_support_results = []
            self.no_support_results.extend(additional_no_support)

        self.merged_results = merged

        # Convert supporting results to DataFrame
        validation_data = []
        for result in merged:
            if result.get('extraction_status') == 'success':
                extracted = result['extracted_data']
                validation_data.append({
                    'edge_index': result['edge_index'],
                    'pmid': result['pmid'],
                    'abstract_support?': extracted.get('support?', 'error'),
                    'support_abstract_sentences': extracted.get('sentences', [])
                })

        self.validation_df = pd.DataFrame(validation_data)

        # Convert non-supporting results to DataFrame
        no_support_data = []
        for result in self.no_support_results:
            if result.get('extraction_status') == 'success':
                extracted = result['extracted_data']
                no_support_data.append({
                    'edge_index': result['edge_index'],
                    'pmid': result['pmid'],
                    'abstract_support?': 'no',
                    'explanation': extracted.get('explanation', '')
                })

        self.no_support_df = pd.DataFrame(no_support_data)

        self.logger.info(f"\nMerged Results Summary:")
        self.logger.info(f"  Supporting ('yes'/'maybe'): {len(merged)}")
        self.logger.info(f"  Non-supporting ('no'): {len(self.no_support_results)}")

        if not self.validation_df.empty:
            support_counts = self.validation_df['abstract_support?'].value_counts()
            self.logger.info("\nSupporting validation breakdown:")
            for support_type, count in support_counts.items():
                self.logger.info(f"  {support_type}: {count}")

        return self.merged_results

    def add_abstract_sentences_to_validation(
        self,
        abstracts_dict: Dict[str, Dict]
    ):
        """Add abstract sentences to validation DataFrame."""
        if self.validation_df is None:
            raise ValueError("No validation results. Run merge_results() first.")

        self.logger.info("Adding abstract sentences to validation data...")

        abstract_sentences_list = []
        for _, row in self.validation_df.iterrows():
            pmid = row['pmid']
            abstract_data = abstracts_dict.get(pmid, {})
            sentences = abstract_data.get('sentences', [])
            abstract_sentences_list.append(sentences)

        self.validation_df['abstract_sentences'] = abstract_sentences_list

    def map_sentences_to_indices(
        self,
        abstracts_dict: Dict[str, Dict]
    ) -> pd.DataFrame:
        """
        Map LLM-generated sentences to abstract sentence indices.

        Uses script-based matching with LLM fallback for failed cases.
        """
        from utils import process_with_llm_fallback
        from collections import Counter

        if self.validation_df is None:
            raise ValueError("No validation results. Run merge_results() first.")

        if 'abstract_sentences' not in self.validation_df.columns:
            raise ValueError("Abstract sentences not added. Run add_abstract_sentences_to_validation() first.")

        self.logger.info("Mapping LLM sentences to abstract indices...")

        indices_column = []
        success_flags = []

        total = len(self.validation_df)

        for i, row in self.validation_df.iterrows():
            if (i + 1) % 50 == 0:
                self.logger.info(f"  Mapping progress: {i + 1}/{total}")

            indices, success = process_with_llm_fallback(
                row, abstracts_dict, self.llm_client
            )
            indices_column.append(indices)
            success_flags.append(success)

        self.validation_df['gold_sent_idxs'] = indices_column
        self.validation_df['mapping_success'] = success_flags

        # Print statistics
        success_counter = Counter(success_flags)
        self.logger.info(f"\nMapping Statistics:")
        self.logger.info(f"  Successful: {success_counter[True]}")
        self.logger.info(f"  Failed: {success_counter[False]}")
        if len(success_flags) > 0:
            self.logger.info(f"  Success rate: {success_counter[True] / len(success_flags) * 100:.1f}%")

        return self.validation_df

    def fix_failed_mappings(
        self,
        abstracts_dict: Dict[str, Dict]
    ) -> pd.DataFrame:
        """Retry failed mappings with a more powerful model."""
        from utils import fix_specific_rows

        if self.validation_df is None or 'mapping_success' not in self.validation_df.columns:
            raise ValueError("No mapping results available. Run map_sentences_to_indices() first.")

        failed_indices = self.validation_df[
            ~self.validation_df['mapping_success']
        ].index.tolist()

        if not failed_indices:
            self.logger.info("No failed mappings to fix!")
            return self.validation_df

        self.logger.info(f"Fixing {len(failed_indices)} failed mappings with {self.round2_model}")

        self.validation_df = fix_specific_rows(
            self.validation_df,
            failed_indices,
            abstracts_dict,
            self.llm_client
        )

        return self.validation_df

    def get_final_results(self) -> pd.DataFrame:
        """
        Get the final validation results with gold sentence indices.

        Returns:
            DataFrame with columns: edge_index, pmid, abstract_support?,
                                   support_abstract_sentences, abstract_sentences,
                                   gold_sent_idxs
        """
        if self.validation_df is None:
            raise ValueError("No validation results available.")

        # Remove mapping_success column if present
        result_df = self.validation_df.copy()
        if 'mapping_success' in result_df.columns:
            result_df = result_df.drop(['mapping_success'], axis=1)

        return result_df

    async def validate_edges(
        self,
        edges: pd.DataFrame,
        abstracts_dict: Dict[str, Dict]
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Complete validation pipeline for edges.

        NEW: Works directly with edges dataframe (no separate supporting_pmids dict needed).

        Args:
            edges: DataFrame of edges with 'publications' column
            abstracts_dict: Dict mapping PMID to abstract data

        Returns:
            Tuple of (supporting_results_df, non_supporting_results_df)
        """
        self.logger.info("\n" + "="*70)
        self.logger.info("STARTING LLM VALIDATION PIPELINE")
        self.logger.info("="*70)

        # Step 1: Prepare prompts
        self.logger.info("\n[Step 1] Preparing validation prompts...")
        self.prepare_prompts(edges, abstracts_dict)
        self.logger.info(f"Generated {len(self.prompts)} prompts")

        # Step 2: Run Round 1
        self.logger.info("\n[Step 2] Running Round 1 validation...")
        await self.run_round1()

        # Step 3: Run Round 2 (validate 'yes' and 'maybe' for quality assurance)
        self.logger.info("\n[Step 3] Running Round 2 validation...")
        await self.run_round2()

        # Step 4: Merge results
        self.logger.info("\n[Step 4] Merging results from both rounds...")
        self.merge_results()

        # Step 5: Add abstract sentences to supporting results only
        self.logger.info("\n[Step 5] Adding abstract sentences to supporting results...")
        self.add_abstract_sentences_to_validation(abstracts_dict)

        # Step 6: Map sentences to indices (only for supporting results)
        self.logger.info("\n[Step 6] Mapping LLM sentences to abstract indices...")
        self.map_sentences_to_indices(abstracts_dict)

        # Step 7: Fix failed mappings
        self.logger.info("\n[Step 7] Fixing failed mappings...")
        self.fix_failed_mappings(abstracts_dict)

        self.logger.info("\n" + "="*70)
        self.logger.info("VALIDATION PIPELINE COMPLETE")
        self.logger.info("="*70)

        return self.get_final_results(), self.no_support_df


def create_llm_review_column(
    main_df: pd.DataFrame,
    info_df: pd.DataFrame,
    logger: Optional[logging.Logger] = None
) -> pd.DataFrame:
    """
    Add LLM_review_publications column to main_df based on validation results.

    Args:
        main_df: Original edges dataframe
        info_df: Validation results dataframe with columns:
                 ['edge_index', 'pmid', 'abstract_support?', 'gold_sent_idxs', 'abstract_sentences']
        logger: Logger instance

    Returns:
        main_df with added 'LLM_review_publications' column
    """
    if logger is None:
        logger = logging.getLogger(__name__)

    logger.info("Creating LLM_review_publications column...")

    # Initialize the new column with None
    main_df = main_df.copy()
    main_df['LLM_review_publications'] = None

    # Filter info_df to only include 'yes' and 'maybe'
    info_filtered = info_df[info_df['abstract_support?'].isin(['yes', 'maybe'])].copy()

    logger.info(f"Processing {len(info_filtered)} validated publications across edges")

    edges_updated = 0
    pmids_matched = 0
    pmids_not_found = 0

    # Group by edge_index to aggregate all PMIDs for each edge
    for edge_idx, group in info_filtered.groupby('edge_index'):
        # Check if this edge_index exists in main_df
        if edge_idx not in main_df.index:
            # Try converting types if needed
            try:
                edge_idx_converted = int(edge_idx) if isinstance(main_df.index[0], (int, np.integer)) else str(edge_idx)
                if edge_idx_converted not in main_df.index:
                    logger.warning(f"Edge index {edge_idx} not found in main dataframe")
                    continue
                edge_idx = edge_idx_converted
            except:
                logger.warning(f"Edge index {edge_idx} not found in main dataframe")
                continue

        # Get the publications array for this row
        pubs = main_df.loc[edge_idx, 'publications']

        # Skip if publications is None or not an array
        if pubs is None or not isinstance(pubs, (list, np.ndarray)):
            logger.warning(f"Edge {edge_idx}: publications is not a list/array: {type(pubs)}")
            continue

        # Convert publications array to set for faster lookup
        pubs_set = set(pubs)

        # Build the dictionary for this edge
        llm_review_dict = {}

        for _, row in group.iterrows():
            pmid = row['pmid']

            # Check if this PMID exists in the publications
            if pmid in pubs_set:
                pmids_matched += 1
                support_status = row['abstract_support?']

                if support_status == 'yes':
                    # Extract support sentences using gold_sent_idxs
                    gold_idxs = row['gold_sent_idxs']
                    abstract_sents = row['abstract_sentences']

                    # Extract the sentences (handle potential indexing issues)
                    if gold_idxs is not None and abstract_sents is not None:
                        try:
                            support_sents = [abstract_sents[idx] for idx in gold_idxs]
                        except (IndexError, TypeError) as e:
                            logger.warning(f"Edge {edge_idx}, PMID {pmid}: Error extracting sentences: {e}")
                            support_sents = []
                    else:
                        support_sents = []

                    llm_review_dict[pmid] = {
                        'abstract_support?': support_status,
                        'support_sentences_from_abstract': support_sents
                    }

                elif support_status == 'maybe':
                    llm_review_dict[pmid] = {
                        'abstract_support?': support_status
                    }
            else:
                pmids_not_found += 1
                logger.warning(f"PMID {pmid} not found in edge {edge_idx} publications")

        # Only assign if we found any matching PMIDs
        if llm_review_dict:
            main_df.at[edge_idx, 'LLM_review_publications'] = llm_review_dict
            edges_updated += 1

    logger.info(f"\nIntegration Summary:")
    logger.info(f"  Edges updated: {edges_updated}")
    logger.info(f"  PMIDs matched and integrated: {pmids_matched}")
    logger.info(f"  PMIDs not found in edge publications: {pmids_not_found}")

    return main_df


async def main(
    edges_file_path: str,
    node_dict: Dict[str, Any],
    predicate_dict: Dict[str, Any],
    llm_client: Any,
    response_parser: Any,
    output_dir: str = "./validation_output",
    round1_model: str = 'gpt-oss:20b',
    round2_model: str = 'gpt-oss:120b',
    batch_size: int = 100
):
    """
    Main function to run the complete validation pipeline.

    Args:
        edges_file_path: Path to parquet file containing edges with 'publications' column
        node_dict: Dictionary mapping node IDs to node information
        predicate_dict: Dictionary mapping predicates to predicate information
        llm_client: LLM client instance
        response_parser: Response parser instance
        output_dir: Directory to save output files
        round1_model: Model name for Round 1 validation
        round2_model: Model name for Round 2 validation
        batch_size: Batch size for PubMed API calls

    Returns:
        Tuple of (edges_with_reviews_df, validation_results_df, abstracts_dict)
    """
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Setup logging
    logger = setup_logging(output_path)

    logger.info("="*70)
    logger.info("LLM VALIDATION PIPELINE - MAIN")
    logger.info("="*70)
    logger.info(f"Edges file: {edges_file_path}")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Round 1 model: {round1_model}")
    logger.info(f"Round 2 model: {round2_model}")

    # Load edges
    logger.info("\nLoading edges dataframe...")
    edges_df = pd.read_parquet(edges_file_path)
    logger.info(f"Loaded {len(edges_df)} edges")
    logger.info(f"Columns: {list(edges_df.columns)}")

    # Verify required columns
    required_columns = ['subject', 'object', 'predicate', 'publications']
    missing_columns = [col for col in required_columns if col not in edges_df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    # Extract unique PMIDs from all edges
    logger.info("\nExtracting unique PMIDs from all edges...")
    all_pmids = set()
    edges_with_pmids = 0

    for idx, row in edges_df.iterrows():
        publications = row['publications']
        if isinstance(publications, (list, np.ndarray)):
            pmids = [pmid for pmid in publications if isinstance(pmid, str) and pmid.startswith('PMID:')]
            if pmids:
                all_pmids.update(pmids)
                edges_with_pmids += 1

    all_pmids = list(all_pmids)
    logger.info(f"Found {len(all_pmids)} unique PMIDs across {edges_with_pmids} edges")

    # Initialize validator
    logger.info("\nInitializing LLM Validator...")
    validator = LLMValidator(
        llm_client=llm_client,
        node_dict=node_dict,
        predicate_dict=predicate_dict,
        response_parser=response_parser,
        round1_model=round1_model,
        round2_model=round2_model,
        logger=logger
    )

    # Fetch abstracts for all unique PMIDs (batched)
    logger.info("\n" + "="*70)
    logger.info("FETCHING ABSTRACTS")
    logger.info("="*70)
    abstracts_dict = await validator.get_abstracts_dict(all_pmids, batch_size=batch_size)

    # Save abstracts_dict
    abstracts_file = output_path / "abstracts_dict.parquet"
    logger.info(f"\nSaving abstracts dictionary to {abstracts_file}")
    abstracts_df = pd.DataFrame([
        {'pmid': pmid, 'abstract': data['abstract'], 'sentences': data['sentences']}
        for pmid, data in abstracts_dict.items()
    ])
    abstracts_df.to_parquet(abstracts_file, index=False)

    # Run validation
    validation_results_df, no_support_df = await validator.validate_edges(
        edges_df,
        abstracts_dict
    )

    # Save validation results
    validation_file = output_path / "validation_results.parquet"
    logger.info(f"\nSaving validation results to {validation_file}")
    validation_results_df.to_parquet(validation_file, index=False)

    no_support_file = output_path / "no_support_results.parquet"
    logger.info(f"Saving non-supporting results to {no_support_file}")
    no_support_df.to_parquet(no_support_file, index=False)

    # Integrate validation results back into edges
    logger.info("\n" + "="*70)
    logger.info("INTEGRATING VALIDATION RESULTS INTO EDGES")
    logger.info("="*70)
    edges_with_reviews = create_llm_review_column(
        edges_df,
        validation_results_df,
        logger=logger
    )

    # Save final edges with LLM reviews
    final_edges_file = output_path / "edges_with_llm_reviews.parquet"
    logger.info(f"\nSaving final edges with LLM reviews to {final_edges_file}")
    edges_with_reviews.to_parquet(final_edges_file, index=False)

    summary = {
        'total_edges': len(edges_df),
        'edges_with_pmids': edges_with_pmids,
        'unique_pmids': len(all_pmids),
        'abstracts_retrieved': len(abstracts_dict),
        'abstracts_failed': len(validator.failed_pmids),
        'prompts_generated': len(validator.prompts),
        'edges_with_missing_info': len(validator.missing_info_edges),
        'validation_results': {
            'supporting': len(validation_results_df),
            'non_supporting': len(no_support_df),
            'yes': len(validation_results_df[validation_results_df['abstract_support?'] == 'yes']),
            'maybe': len(validation_results_df[validation_results_df['abstract_support?'] == 'maybe'])
        },
        'edges_updated_with_reviews': edges_with_reviews['LLM_review_publications'].notna().sum()
    }

    # Try to save as JSON, fallback to text if it fails
    try:
        summary_file = output_path / "validation_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        logger.info("Summary saved as JSON")
    except (TypeError, ValueError) as e:
        logger.warning(f"Failed to save as JSON ({e}), saving as text instead")
        summary_file = output_path / "validation_summary.txt"
        with open(summary_file, 'w') as f:
            f.write("VALIDATION SUMMARY\n")
            f.write("=" * 70 + "\n\n")
            for key, value in summary.items():
                if isinstance(value, dict):
                    f.write(f"{key}:\n")
                    for sub_key, sub_value in value.items():
                        f.write(f"  {sub_key}: {sub_value}\n")
                else:
                    f.write(f"{key}: {value}\n")
        logger.info(f"Summary saved as text to {summary_file}")

    logger.info("\n" + "="*70)
    logger.info("PIPELINE COMPLETE")
    logger.info("="*70)
    logger.info("\nOutput files created:")
    logger.info(f"  1. {abstracts_file}")
    logger.info(f"  2. {validation_file}")
    logger.info(f"  3. {no_support_file}")
    logger.info(f"  4. {final_edges_file}")
    logger.info(f"  5. {summary_file}")
    logger.info(f"  6. Log file in {output_dir}")

    return edges_with_reviews, validation_results_df, abstracts_dict


# Example usage
if __name__ == "__main__":
    """
    Example of how to use the main function.

    You'll need to provide:
    - edges_file_path: path to your edges parquet file
    - node_dict: dictionary with node information
    - predicate_dict: dictionary with predicate information
    - llm_client: your LLM client instance
    - response_parser: your response parser instance
    """

    import json
    def load_json(path):
        with open(path, 'r') as file:
            return json.load(file)

    # These would be loaded from your actual data sources
    edges_file_path = "data/arax_rtx_273_semmed_biomarker_for_edges.parquet"
    node_dict = load_json('dict/rtx-kg2_id_info_dictionary.json')
    predicate_dict = load_json('dict/biolink_pred_info_dictionary.json')

    from ollama import Client
    llm_client = Client()

    from response_parser import SimpleLLMResponseParser
    response_parser = SimpleLLMResponseParser()

    asyncio.run(main(
        edges_file_path=edges_file_path,
        node_dict=node_dict,
        predicate_dict=predicate_dict,
        llm_client=llm_client,
        response_parser=response_parser,
        output_dir="./validation_output"
    ))
