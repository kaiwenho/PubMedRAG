"""
Iterative Batch Optimization System

This implements the TRUE iterative workflow:
1000 gold edges -> params v0 -> process 50 edges -> add 32 new -> params v1 -> process next 50 -> ...

Key Difference from parameter_optimization_bridge.py:
- That script: Process ALL edges -> optimize -> repeat
- This script: Process BATCH -> optimize -> process NEXT batch -> optimize -> ...

This is a micro-feedback loop that optimizes parameters AS YOU PROCESS the edges,
rather than waiting for all edges to be processed.
"""

import asyncio
import pandas as pd
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from datetime import datetime
from dataclasses import dataclass, asdict
from nltk.tokenize import sent_tokenize
from biomed_eval import BiomedicalRetrievalEvaluator


# ============================================================================
# CONSTANTS AND UTILITIES
# ============================================================================

SEPARATOR_FULL = "=" * 80
SEPARATOR_SECTION = "=" * 70
SEPARATOR_SUBSECTION = "-" * 70
SEPARATOR_BOX = "#" * 80


def convert_numpy_types(obj):
    """
    Recursively convert numpy types to Python native types for JSON serialization.

    Args:
        obj: Object to convert

    Returns:
        Converted object with all numpy types as Python native types
    """
    import numpy as np

    if isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    else:
        return obj


# ============================================================================
# CONFIGURATION
# ============================================================================

@dataclass
class IterativeBatchConfig:
    """Configuration for iterative batch processing."""

    # Data paths
    base_evaluation_data: str  # Gold standard edges (your_data.pkl)
    nopub_edges_path: str  # Edges without publications (edges/treats_nopub.parquet)

    # Processing parameters
    batch_size: int = 50  # Process this many edges per iteration
    min_new_edges_for_reoptimization: int = 10  # Min new edges before re-optimizing
    max_batches: int = 20  # Maximum number of batches to process

    # Paths
    results_dir: str = 'result/iterative_batches'
    optimization_history_file: str = 'result/iterative_batches/optimization_history.json'
    checkpoint_file: str = 'result/iterative_batches/checkpoint.json'
    extended_edges_dir: str = 'result/iterative_batches/extended_edges'  # Per-batch extended edges

    # Predicate
    predicate: str = 'biolink:treats'

    # Evaluation parameters
    random_state: int = 42
    val_size: float = 0.5

    # Resume from checkpoint
    resume_from_checkpoint: bool = True  # Set to False to force fresh start


# ============================================================================
# EXTENDED EDGES MANAGER
# ============================================================================

class ExtendedEdgesManager:
    """
    Manages extended edges - original edges merged with validation results.

    Saves per-batch files and produces final concatenated result.
    """

    def __init__(self, extended_edges_dir: str, predicate: str):
        """Initialize extended edges manager."""
        self.extended_edges_dir = Path(extended_edges_dir)
        self.predicate = predicate

        # Create directory
        self.extended_edges_dir.mkdir(parents=True, exist_ok=True)

        # Track processed batches
        self.processed_batches = self._discover_processed_batches()

        print(f"Extended edges will be saved to: {self.extended_edges_dir}")
        if self.processed_batches:
            print(f"  Found {len(self.processed_batches)} previously processed batches")

    def _discover_processed_batches(self) -> List[int]:
        """Discover which batches have already been processed."""
        batch_files = list(self.extended_edges_dir.glob('batch_*_extended_edges.parquet'))

        batch_nums = []
        for f in batch_files:
            try:
                batch_num = int(f.stem.split('_')[1])
                batch_nums.append(batch_num)
            except (IndexError, ValueError):
                continue

        return sorted(batch_nums)

    def save_batch_extended_edges(self, batch_num: int,
                                  original_edges: pd.DataFrame,
                                  validated_edges: pd.DataFrame,
                                  cached_abstracts_path: Optional[str] = None) -> Dict:
        """
        Save extended edges for a batch (yes/maybe results only).

        MODIFIED: Only save supporting evidence (yes/maybe) for production use.

        Returns:
            Dict with statistics about what was saved
        """
        # Filter to only yes/maybe (supporting evidence)
        supporting_edges = validated_edges[
            validated_edges['abstract_support?'].isin(['yes', 'maybe'])
        ].copy()

        n_yes = len(validated_edges[validated_edges['abstract_support?'] == 'yes'])
        n_maybe = len(validated_edges[validated_edges['abstract_support?'] == 'maybe'])

        if len(supporting_edges) == 0:
            print(f"  No supporting edges (yes/maybe) to save for batch {actual_batch_num}")
            return {
                'n_extended_saved': 0,
                'n_yes': n_yes,
                'n_maybe': n_maybe,
                'skipped': True
            }

        print(f"  Saving {len(supporting_edges)} supporting edges ({n_yes} yes, {n_maybe} maybe)")

        # Load cached abstracts if available
        abstracts_dict = {}
        if cached_abstracts_path and Path(cached_abstracts_path).exists():
            try:
                with open(cached_abstracts_path, 'r') as f:
                    abstracts_dict = json.load(f)
            except Exception as e:
                print(f"  Warning: Could not load cached abstracts: {e}")

        # Merge original edges with validated results
        extended = []

        for _, val_row in supporting_edges.iterrows():
            edge_idx = val_row['edge_index']

            if edge_idx not in original_edges.index:
                print(f"  Warning: edge_index {edge_idx} not in original edges")
                continue

            orig_row = original_edges.loc[edge_idx]

            # Create extended edge record
            extended_edge = {
                # From original edge
                'edge_index': edge_idx,
                'subject': orig_row['subject'],
                'object': orig_row['object'],
                'predicate': orig_row['predicate'],

                # From validation
                'pmid': val_row['pmid'],
                'abstract_support?': val_row['abstract_support?']
            }

            for col in ['abstract_sentences', 'gold_sent_idxs', 'llm_response',
                       'supporting_sentences', 'sentence_indices']:
                if col in val_row:
                    try:
                        value = val_row[col]
                        # Skip None values
                        if value is not None:
                            # For scalar float values, check for NaN
                            if isinstance(value, float) and pd.isna(value):
                                continue
                            # Add the value
                            extended_edge[col] = value
                    except Exception as e:
                        # Skip problematic values
                        print(f"  Warning: Could not add column {col}: {e}")
                        continue

            extended.append(extended_edge)

        if not extended:
            print(f"  No extended edges after merge")
            return {
                'n_extended_saved': 0,
                'n_yes': n_yes,
                'n_maybe': n_maybe,
                'skipped': True
            }

        # Save as parquet
        extended_df = pd.DataFrame(extended)
        output_path = self.extended_edges_dir / f'batch_{batch_num:03d}_extended_edges.parquet'
        extended_df.to_parquet(output_path, index=False)

        print(f"   Saved {len(extended_df)} extended edges to {output_path}")

        return {
            'n_extended_saved': len(extended_df),
            'n_yes': n_yes,
            'n_maybe': n_maybe,
            'skipped': False
        }

    def save_complete_extended_edges(self) -> str:
        """Concatenate all batch extended edges into single file."""
        batch_files = sorted(self.extended_edges_dir.glob('batch_*_extended_edges.parquet'))

        if not batch_files:
            print("No batch files to concatenate")
            return None

        print(f"Concatenating {len(batch_files)} batch files...")

        all_extended = []
        for batch_file in batch_files:
            df = pd.read_parquet(batch_file)
            all_extended.append(df)

        complete_df = pd.concat(all_extended, ignore_index=True)

        # Remove duplicates (same edge_index + pmid)
        complete_df = complete_df.drop_duplicates(subset=['edge_index', 'pmid'], keep='first')

        # Save complete file
        pred_name = self.predicate.replace('biolink:', '')
        output_path = self.extended_edges_dir / f'{pred_name}_complete_extended_edges.parquet'
        complete_df.to_parquet(output_path, index=False)

        print(f" Saved complete extended edges: {output_path}")
        print(f"  Total unique edge-PMID pairs: {len(complete_df)}")
        print(f"  Yes: {len(complete_df[complete_df['abstract_support?'] == 'yes'])}")
        print(f"  Maybe: {len(complete_df[complete_df['abstract_support?'] == 'maybe'])}")

        return str(output_path)

    def load_all_extended_edges(self) -> pd.DataFrame:
        """Load and concatenate all per-batch extended edges."""
        all_edges = []

        for batch_num in sorted(self.processed_batches):
            filename = f"batch_{batch_num:03d}_extended_edges.parquet"
            filepath = self.extended_edges_dir / filename

            if filepath.exists():
                batch_edges = pd.read_parquet(filepath)
                all_edges.append(batch_edges)

        if not all_edges:
            return pd.DataFrame()

        return pd.concat(all_edges, ignore_index=True)

    def get_processed_batch_nums(self) -> List[int]:
        """Get list of processed batch numbers."""
        return sorted(self.processed_batches)


# ============================================================================
# CHECKPOINT MANAGER
# ============================================================================

class CheckpointManager:
    """
    Manages checkpointing for crash recovery.

    Saves state after each batch so processing can resume from last checkpoint.
    """

    def __init__(self, checkpoint_file: str):
        """
        Initialize checkpoint manager.

        Args:
            checkpoint_file: Path to checkpoint file
        """
        self.checkpoint_file = Path(checkpoint_file)
        self.checkpoint_file.parent.mkdir(parents=True, exist_ok=True)

    def save_checkpoint(self, state: Dict):
        """
        Save checkpoint state.

        Args:
            state: Dictionary with checkpoint state
        """
        # Add timestamp
        state['checkpoint_timestamp'] = datetime.now().isoformat()

        # Convert numpy types
        state_serializable = convert_numpy_types(state)

        # Write atomically (write to temp, then rename)
        temp_file = self.checkpoint_file.with_suffix('.json.tmp')

        with open(temp_file, 'w') as f:
            json.dump(state_serializable, f, indent=2)

        # Atomic rename
        temp_file.replace(self.checkpoint_file)

    def load_checkpoint(self) -> Optional[Dict]:
        """
        Load checkpoint state.

        Returns:
            Checkpoint state dict, or None if no checkpoint exists
        """
        if not self.checkpoint_file.exists():
            return None

        try:
            with open(self.checkpoint_file, 'r') as f:
                state = json.load(f)

            print(f" Loaded checkpoint from: {self.checkpoint_file}")
            print(f"  Timestamp: {state.get('checkpoint_timestamp', 'unknown')}")
            print(f"  Last batch: {state.get('last_completed_batch', 'unknown')}")

            return state

        except json.JSONDecodeError as e:
            print(f"  Corrupted checkpoint file: {e}")
            # Backup and start fresh
            backup_path = self.checkpoint_file.with_suffix('.json.corrupted')
            import shutil
            shutil.copy(self.checkpoint_file, backup_path)
            print(f"  Backed up to: {backup_path}")
            return None

    def delete_checkpoint(self):
        """Delete checkpoint file (e.g., after successful completion)."""
        if self.checkpoint_file.exists():
            self.checkpoint_file.unlink()
            print(f" Deleted checkpoint: {self.checkpoint_file}")


class EdgeBatchManager:
    """Manages batched processing of edges without publications."""

    def __init__(self, nopub_edges_path: str, batch_size: int = 50):
        """
        Initialize batch manager.

        Args:
            nopub_edges_path: Path to edges without publications
            batch_size: Number of edges per batch
        """
        self.nopub_edges_path = Path(nopub_edges_path)
        self.batch_size = batch_size

        # Load all edges
        if not self.nopub_edges_path.exists():
            raise FileNotFoundError(f"Edge file not found: {nopub_edges_path}")

        self.all_edges = pd.read_parquet(nopub_edges_path)
        self.total_edges = len(self.all_edges)

        # Track processed edges
        self.processed_edge_indices = set()
        self.current_batch_num = 0

        print(f"Loaded {self.total_edges} edges without publications")

    def get_next_batch(self) -> Optional[pd.DataFrame]:
        """
        Get next batch of unprocessed edges.

        Returns:
            DataFrame with next batch, or None if all edges processed
        """
        # Get unprocessed indices
        all_indices = set(self.all_edges.index)
        unprocessed_indices = all_indices - self.processed_edge_indices

        if not unprocessed_indices:
            print("All edges have been processed!")
            return None

        # Get batch
        batch_indices = list(unprocessed_indices)[:self.batch_size]
        batch = self.all_edges.loc[batch_indices].copy()

        self.current_batch_num += 1

        print(f"\nBatch {self.current_batch_num}:")
        print(f"  Batch size: {len(batch)}")
        print(f"  Processed so far: {len(self.processed_edge_indices)}")
        print(f"  Remaining: {len(unprocessed_indices) - len(batch_indices)}")

        return batch

    def mark_batch_processed(self, batch: pd.DataFrame):
        """Mark edges in batch as processed."""
        self.processed_edge_indices.update(batch.index)

    def get_progress(self) -> Dict:
        """Get processing progress statistics."""
        return {
            'total_edges': self.total_edges,
            'processed_edges': len(self.processed_edge_indices),
            'remaining_edges': self.total_edges - len(self.processed_edge_indices),
            'batches_completed': self.current_batch_num,
            'progress_pct': (len(self.processed_edge_indices) / self.total_edges * 100) if self.total_edges > 0 else 0
        }


# ============================================================================
# EVALUATION DATA MANAGER
# ============================================================================

class EvaluationDataManager:
    """Manages incremental addition of validated edges to evaluation dataset."""

    def __init__(self, base_data_path: str, predicate: str):
        """Initialize evaluation data manager."""
        self.base_data_path = Path(base_data_path)
        self.predicate = predicate

        # Load base data
        if self.base_data_path.suffix == '.pkl':
            self.base_data = pd.read_pickle(base_data_path)
        elif self.base_data_path.suffix == '.csv':
            self.base_data = pd.read_csv(base_data_path)
        else:
            raise ValueError(f"Unsupported format: {base_data_path}")

        # Filter to predicate
        self.base_data = self.base_data[
            self.base_data['predicate'] == predicate
        ].copy()

        print(f"Loaded {len(self.base_data)} base evaluation edges for {predicate}")

        # Track accumulated validated edges
        self.accumulated_validated = []
        self.version = 0

        self.cumulative_yes_added = 0
        self.cumulative_no_added = 0

        self.edge_text_cache = {}

        from edge_to_sentence import EdgeTextGenerator
        from ollama import Client
        from response_parser import SimpleLLMResponseParser
        llm_client = Client()
        response_parser = SimpleLLMResponseParser()
        self.text_generator = EdgeTextGenerator(
            node_dict_path='dict/rtx-kg2_id_info_dictionary.json',
            predicate_dict_path='dict/biolink_pred_info_dictionary.json',
            llm_client=llm_client,  # Will set later if needed
            response_parser=response_parser
        )



    def add_validated_edges(self, validated_edges: pd.DataFrame, original_edges: pd.DataFrame, cached_abstracts: Dict) -> Dict:
        """
        Add validated edges to evaluation dataset with CROSS-BATCH balancing.
        """
        if len(validated_edges) == 0:
            return {
                'n_new': 0, 'n_yes_raw': 0, 'n_maybe_raw': 0, 'n_no_raw': 0,
                'n_yes_added': 0, 'n_no_added': 0,
                'balance_before': self.cumulative_yes_added - self.cumulative_no_added,
                'balance_after': self.cumulative_yes_added - self.cumulative_no_added,
                'skipped': True, 'skip_reason': 'no_validation_results'
            }

        # Separate yes/maybe/no results
        yes_edges = validated_edges[validated_edges['abstract_support?'] == 'yes'].copy()
        maybe_edges = validated_edges[validated_edges['abstract_support?'] == 'maybe'].copy()
        no_edges = validated_edges[validated_edges['abstract_support?'] == 'no'].copy()

        n_yes = len(yes_edges)
        n_maybe = len(maybe_edges)
        n_no = len(no_edges)

        print(f"\n  Raw validated: {n_yes} yes, {n_maybe} maybe, {n_no} no")

        # Skip if no 'yes' results
        if n_yes == 0:
            print(f"    SKIP EVALUATION: No 'yes' results")
            return {
                'n_new': 0, 'n_yes_raw': n_yes, 'n_maybe_raw': n_maybe, 'n_no_raw': n_no,
                'n_yes_added': 0, 'n_no_added': 0,
                'balance_before': self.cumulative_yes_added - self.cumulative_no_added,
                'balance_after': self.cumulative_yes_added - self.cumulative_no_added,
                'skipped': True, 'skip_reason': 'no_yes_results'
            }

        # Calculate current balance
        balance_before = self.cumulative_yes_added - self.cumulative_no_added
        print(f"  Current balance: {balance_before:+d} (yes - no)")
        if balance_before > 0:
            print(f"     Need {balance_before} more 'no' to balance previous batches")
        elif balance_before < 0:
            print(f"     Need {-balance_before} more 'yes' to balance previous batches")
        else:
            print(f"     Perfectly balanced")

        # Decide what to add based on cross-batch balancing
        yes_to_add = yes_edges  # Always take ALL yes (never waste!)

        # Calculate how many 'no' we need
        if balance_before > 0:
            no_needed = balance_before + n_yes
            print(f"  Strategy: Need {no_needed} no ({balance_before} deficit + {n_yes} for current yes)")
        else:
            no_needed = n_yes
            print(f"  Strategy: Need {no_needed} no (to balance current yes)")

        # Take what we can from available 'no'
        if n_no >= no_needed:
            no_to_add = no_edges.sample(n=no_needed, random_state=self.version)
            print(f"   Taking {no_needed} no from {n_no} available")
            print(f"    Remaining no surplus: {n_no - no_needed} (saved for future)")
        else:
            no_to_add = no_edges
            deficit = no_needed - n_no
            print(f"    Only {n_no} no available, need {no_needed}")
            print(f"    Taking ALL {n_no} no")
            print(f"    Creating no deficit: {deficit} (will balance with future batches)")

        edges_to_add = pd.concat([yes_to_add, no_to_add], ignore_index=True)

        print(f"  Adding to evaluation: {len(yes_to_add)} yes + {len(no_to_add)} no = {len(edges_to_add)} pairs")

        # Enrich validated edges with original edge data
        enriched_edges = []
        cache_hits = 0
        cache_misses = 0

        for _, val_row in edges_to_add.iterrows():
            edge_idx = val_row['edge_index']

            if edge_idx not in original_edges.index:
                print(f"  Warning: edge_index {edge_idx} not found in original edges")
                continue

            orig_row = original_edges.loc[edge_idx]

            # Create enriched edge
            enriched = {
                'subject': orig_row['subject'],
                'object': orig_row['object'],
                'predicate': orig_row['predicate'],
                'pmid': val_row['pmid'],
                'abstract_support?': val_row['abstract_support?']
            }

            # Handle abstract_sentences
            pmid_str = str(val_row['pmid'])
            if pmid_str in cached_abstracts:
                abstract_data = cached_abstracts[pmid_str]
                if isinstance(abstract_data, dict):
                    if 'sentences' in abstract_data:
                        abstract_sentences = abstract_data['sentences']
                    elif 'abstract' in abstract_data:
                        abstract_sentences = sent_tokenize(abstract_data['abstract'])
                    else:
                        print(f"  Warning: Failed to fetch abstract for edge {edge_idx}")
                        continue
                else:
                    print(f"  Warning: Failed to fetch abstract for edge {edge_idx}")
                    continue
            else:
                print(f"  Warning: Failed to fetch abstract for edge {edge_idx}")
                continue
            enriched['abstract_sentences'] = abstract_sentences

            support_abstract = val_row['abstract_support?']
            if support_abstract == 'yes':
                enriched['gold_sent_idxs'] = val_row['gold_sent_idxs']
            else:
                enriched['gold_sent_idxs'] = []

            # Check cache first before generating LLM sentences
            if edge_idx in self.edge_text_cache:
                # Cache hit - use cached values
                cached_data = self.edge_text_cache[edge_idx]
                enriched['concat_sentence'] = cached_data['concat_sentence']
                enriched['ai_sentences'] = cached_data['ai_sentences']
                enriched_edges.append(enriched)
                cache_hits += 1
            else:
                # Cache miss - generate new sentences
                concat_sentence, ai_sentences = self.text_generator.generate_edge_text(
                    enriched['subject'],
                    enriched['predicate'],
                    enriched['object']
                )

                if ai_sentences:
                    enriched['concat_sentence'] = concat_sentence
                    enriched['ai_sentences'] = ai_sentences
                    enriched_edges.append(enriched)

                    # Save to in-memory cache
                    self.edge_text_cache[edge_idx] = {
                        'concat_sentence': concat_sentence,
                        'ai_sentences': ai_sentences
                    }
                    cache_misses += 1
                else:
                    print(f"  Warning: Failed to generate ai sentences for edge {edge_idx}")
                    continue

        # Print cache statistics
        total_requests = cache_hits + cache_misses
        if total_requests > 0:
            print(f"\n  Edge Text Cache: {cache_hits}/{total_requests} hits ({cache_hits/total_requests*100:.1f}%), {len(self.edge_text_cache)} total entries")

        if not enriched_edges:
            print(f"  No edges to add after enrichment")
            return {
                'n_new': 0, 'n_yes_raw': n_yes, 'n_maybe_raw': n_maybe, 'n_no_raw': n_no,
                'n_yes_added': 0, 'n_no_added': 0,
                'balance_before': balance_before,
                'balance_after': balance_before,
                'skipped': True, 'skip_reason': 'enrichment_failed'
            }

        # Validate enriched edges
        valid_enriched = []
        for edge in enriched_edges:
            if 'abstract_sentences' not in edge:
                print(f"  Warning: Missing abstract_sentences for edge")
                continue
            if not isinstance(edge['abstract_sentences'], list) or len(edge['abstract_sentences']) == 0:
                print(f"  Warning: Empty abstract_sentences for edge")
                continue
            valid_enriched.append(edge)

        if not valid_enriched:
            print(f"  No valid enriched edges")
            return {
                'n_new': 0, 'n_yes_raw': n_yes, 'n_maybe_raw': n_maybe, 'n_no_raw': n_no,
                'n_yes_added': 0, 'n_no_added': 0,
                'balance_before': balance_before,
                'balance_after': balance_before,
                'skipped': True, 'skip_reason': 'validation_failed'
            }

        enriched_df = pd.DataFrame(valid_enriched)

        # Deduplication
        existing_keys = set(
            self.base_data.apply(
                lambda r: (r['subject'], r['object'], r['pmid']), axis=1
            )
        )

        for prev_validated in self.accumulated_validated:
            prev_keys = set(
                prev_validated.apply(
                    lambda r: (r['subject'], r['object'], r['pmid']), axis=1
                )
            )
            existing_keys.update(prev_keys)

        new_keys = enriched_df.apply(
            lambda r: (r['subject'], r['object'], r['pmid']), axis=1
        )
        new_mask = ~new_keys.isin(existing_keys)
        new_edges = enriched_df[new_mask].copy()

        if len(new_edges) > 0:
            # Count what was actually added
            new_yes_added = len(new_edges[new_edges['abstract_support?'] == 'yes'])
            new_no_added = len(new_edges[new_edges['abstract_support?'] == 'no'])

            # Update cumulative tracking
            self.cumulative_yes_added += new_yes_added
            self.cumulative_no_added += new_no_added
            balance_after = self.cumulative_yes_added - self.cumulative_no_added

            # Save
            self.accumulated_validated.append(new_edges)
            self.version += 1

            print(f"   Added {len(new_edges)} new edges (version {self.version})")
            print(f"    Yes: {new_yes_added}, No: {new_no_added}")
            print(f"  NEW Balance: {balance_after:+d} (cumulative yes - no)")
            print(f"    Total yes: {self.cumulative_yes_added}, Total no: {self.cumulative_no_added}")

            return {
                'n_new': len(new_edges),
                'n_yes_raw': n_yes,
                'n_maybe_raw': n_maybe,
                'n_no_raw': n_no,
                'n_yes_added': new_yes_added,
                'n_no_added': new_no_added,
                'balance_before': balance_before,
                'balance_after': balance_after,
                'skipped': False,
                'skip_reason': None
            }
        else:
            print(f"  No new edges to add (all duplicates)")
            return {
                'n_new': 0,
                'n_yes_raw': n_yes,
                'n_maybe_raw': n_maybe,
                'n_no_raw': n_no,
                'n_yes_added': 0,
                'n_no_added': 0,
                'balance_before': balance_before,
                'balance_after': balance_before,
                'skipped': True,
                'skip_reason': 'all_duplicates'
            }

    def get_current_dataset(self) -> pd.DataFrame:
        """Get current augmented evaluation dataset."""
        if not self.accumulated_validated:
            return self.base_data.copy()

        # Combine all data
        all_data = [self.base_data] + self.accumulated_validated
        augmented = pd.concat(all_data, ignore_index=True)

        return augmented

    def get_stats(self) -> Dict:
        """Get statistics about current dataset."""
        current = self.get_current_dataset()

        return {
            'version': self.version,
            'base_edges': len(self.base_data),
            'accumulated_edges': sum(len(df) for df in self.accumulated_validated),
            'total_edges': len(current),
            'label_distribution': current['abstract_support?'].value_counts().to_dict(),
            'cumulative_yes_added': self.cumulative_yes_added,
            'cumulative_no_added': self.cumulative_no_added,
            'balance': self.cumulative_yes_added - self.cumulative_no_added
        }

# ============================================================================
# ITERATIVE BATCH OPTIMIZER
# ============================================================================

class IterativeBatchOptimizer:
    """
    Main orchestrator for iterative batch optimization.

    Workflow:
    1. Start with base evaluation data (1000 gold edges)
    2. Run evaluation -> get parameters v0
    3. Process batch 1 (50 edges) with params v0 -> get 32 validated
    4. Add 32 validated to evaluation data -> 1032 edges
    5. Re-run evaluation -> get parameters v1
    6. Process batch 2 (50 edges) with params v1 -> get 28 validated
    7. Add 28 validated to evaluation data -> 1060 edges
    8. ...continue until all edges processed or convergence
    """

    def __init__(self, config: IterativeBatchConfig):
        """Initialize optimizer."""
        self.config = config
        self.pred_name = config.predicate.replace("biolink:", "")

        # Create results directory
        Path(config.results_dir).mkdir(parents=True, exist_ok=True)
        Path(config.extended_edges_dir).mkdir(parents=True, exist_ok=True)

        # Initialize checkpoint manager
        self.checkpoint_manager = CheckpointManager(config.checkpoint_file)

        # Try to load checkpoint if resuming
        checkpoint = None
        if config.resume_from_checkpoint:
            checkpoint = self.checkpoint_manager.load_checkpoint()

        # Initialize managers
        self.batch_manager = EdgeBatchManager(
            config.nopub_edges_path,
            config.batch_size
        )

        self.data_manager = EvaluationDataManager(
            config.base_evaluation_data,
            config.predicate
        )

        self.extended_edges_manager = ExtendedEdgesManager(
            config.extended_edges_dir,
            config.predicate
        )

        # Optimization history
        self.history = self._load_history()

        # Current parameters (will be updated each iteration)
        self.current_params = None

        # Restore state from checkpoint if available
        if checkpoint:
            self._restore_from_checkpoint(checkpoint)

    def _load_history(self) -> List[Dict]:
        """Load optimization history from disk."""
        history_path = Path(self.config.optimization_history_file)

        if history_path.exists():
            try:
                with open(history_path, 'r') as f:
                    return json.load(f)
            except json.JSONDecodeError as e:
                print(f"  Warning: Corrupted history file at {history_path}")
                print(f"  Error: {e}")
                print(f"  Creating backup and starting fresh history")

                # Backup corrupted file
                backup_path = history_path.with_suffix('.json.corrupted.bak')
                import shutil
                shutil.copy(history_path, backup_path)
                print(f"  Corrupted file backed up to: {backup_path}")

                # Delete corrupted file
                history_path.unlink()

                return []

        return []

    def _restore_from_checkpoint(self, checkpoint: Dict):
        """
        Restore state from checkpoint.

        Args:
            checkpoint: Checkpoint state dictionary
        """
        print("\n" + "="*70)
        print("RESTORING FROM CHECKPOINT")
        print("="*70)

        # Restore batch progress
        completed_batches = checkpoint.get('completed_batches', [])
        if completed_batches:
            # Mark batches as processed
            processed_indices = set(checkpoint.get('processed_edge_indices', []))
            self.batch_manager.processed_edge_indices.update(processed_indices)
            self.batch_manager.current_batch_num = checkpoint.get('current_batch_num', 0)

            print(f"  Resuming from batch {self.batch_manager.current_batch_num + 1}")
            print(f"  Already processed: {len(completed_batches)} batches")

        # Restore parameters
        if 'current_params' in checkpoint and checkpoint['current_params']:
            self.current_params = checkpoint['current_params']
            print(f"  Restored parameters")

        # Load extended edges from disk (they're already saved)
        processed_batches = self.extended_edges_manager.get_processed_batch_nums()
        if processed_batches:
            print(f"  Found {len(processed_batches)} extended edge files")

        print("="*70 + "\n")

    def _save_checkpoint(self, batch_num: int, **extra_info):
        """
        Save checkpoint after batch completion.

        Args:
            batch_num: Just completed batch number
            **extra_info: Additional information to include in checkpoint
        """
        checkpoint_state = {
            'last_completed_batch': batch_num,
            'completed_batches': self.extended_edges_manager.get_processed_batch_nums(),
            'current_batch_num': self.batch_manager.current_batch_num,
            'processed_edge_indices': list(self.batch_manager.processed_edge_indices),
            'current_params': self.current_params,
            'data_stats': self.data_manager.get_stats(),
            'total_batches_planned': self.config.max_batches
        }

        checkpoint_state.update(extra_info)

        self.checkpoint_manager.save_checkpoint(checkpoint_state)

    def _save_history(self):
        """Save optimization history to disk."""
        history_path = Path(self.config.optimization_history_file)
        history_path.parent.mkdir(parents=True, exist_ok=True)

        # Convert numpy types to native Python types for JSON serialization
        history_serializable = convert_numpy_types(self.history)

        with open(history_path, 'w') as f:
            json.dump(history_serializable, f, indent=2)

    def _run_evaluation(self) -> Dict:
        """
        Run evaluation on current dataset to get optimized parameters.

        Returns:
            Dict with parameters and metrics
        """
        print("\n" + "="*70)
        print("RUNNING EVALUATION")
        print("="*70)

        # Get current dataset
        current_data = self.data_manager.get_current_dataset()

        # Save to temp file
        temp_path = Path(self.config.results_dir) / 'temp_eval_data.pkl'
        current_data.to_pickle(temp_path)

        try:
            # Initialize evaluator
            evaluator = BiomedicalRetrievalEvaluator(
                data_path=str(temp_path),
                predicate=self.config.predicate,
                random_state=self.config.random_state
            )

            # Split and run
            evaluator.split_data(val_size=self.config.val_size)
            best_config = evaluator.run_validation_pipeline()
            test_results = evaluator.run_test_evaluation()

            # Save results (this computes both thresholds correctly)
            results_path = Path(self.config.results_dir) / 'evaluation_results.json'
            evaluator.save_results(filepath=str(results_path))

            # Load the properly computed parameters from the saved file
            # This ensures we get both abstract_classification and sentence_search thresholds
            # which are computed separately by different methods
            from biomed_eval import load_search_parameters
            params = load_search_parameters(
                filepath=str(results_path),
                predicate=self.config.predicate
            )

            print(f"\nEvaluation Results:")
            print(f"  Test MRR: {params['test_metrics']['MRR']:.4f}")
            print(f"  Test PR-AUC: {params['test_metrics']['PR_AUC']:.4f}")
            print(f"  Sentence threshold: {params['sentence_search']['threshold']:.4f}")
            print(f"  Abstract threshold: {params['abstract_classification']['threshold']:.4f}")

            return params

        finally:
            # Cleanup temp file
            if temp_path.exists():
                temp_path.unlink()

    async def _run_pipeline_on_batch(self, batch: pd.DataFrame) -> Tuple[pd.DataFrame, Dict, Dict]:
        """
        Run pipeline on a batch of edges.

        Args:
            batch: DataFrame of edges to process

        Returns:
            (validated_edges, statistics, cached_abstracts)
        """
        print("\n" + "="*70)
        print(f"RUNNING PIPELINE ON BATCH (size={len(batch)})")
        print("="*70)

        from pub_extension_pipeline import PubExtensionPipeline
        from ollama import Client
        from response_parser import SimpleLLMResponseParser

        # Initialize LLM (adjust as needed)
        try:
            llm_client = Client()
            response_parser = SimpleLLMResponseParser()
        except Exception as e:
            print(f"Warning: LLM client initialization failed: {e}")
            llm_client = None
            response_parser = None

        # Create pipeline
        pipeline = PubExtensionPipeline(
            predicate=self.config.predicate,
            llm_client=llm_client,
            response_parser=response_parser,
            edges=batch
        )

        # Run pipeline
        await pipeline.run(max_edges=len(batch))

        # Load cached abstracts (needed for enrichment)
        # Use self.pred_name (calculated in __init__)
        pred_name = self.pred_name
        cached_abstracts_path = Path(f'result/classification/{pred_name}_cached_abstracts.json')

        cached_abstracts = {}
        if cached_abstracts_path.exists():
            with open(cached_abstracts_path, 'r') as f:
                cached_abstracts = json.load(f)
            print(f"  Loaded {len(cached_abstracts)} cached abstracts")
        else:
            print("    No cached abstracts found")

        # Load validation results - BOTH supporting and non-supporting
        # Load supporting results (yes/maybe)
        supporting_path = Path(f'result/validation/{pred_name}_validation_results.parquet')
        non_supporting_path = Path(f'result/validation/{pred_name}_non_supporting_results.parquet')

        all_validated = []

        # Load supporting results
        if supporting_path.exists():
            supporting = pd.read_parquet(supporting_path)
            # Filter to edges from this batch
            batch_indices = set(batch.index)
            supporting_from_batch = supporting[supporting['edge_index'].isin(batch_indices)].copy()
            all_validated.append(supporting_from_batch)
            print(f"  Loaded {len(supporting_from_batch)} supporting results (yes/maybe)")
        else:
            print("    No supporting results file found")

        # Load non-supporting results (no)
        if non_supporting_path.exists():
            non_supporting = pd.read_parquet(non_supporting_path)
            # Filter to edges from this batch
            batch_indices = set(batch.index)
            non_supporting_from_batch = non_supporting[non_supporting['edge_index'].isin(batch_indices)].copy()
            all_validated.append(non_supporting_from_batch)
            print(f"  Loaded {len(non_supporting_from_batch)} non-supporting results (no)")
        else:
            print("    No non-supporting results file found")

        # Combine all results
        if all_validated:
            validated_from_batch = pd.concat(all_validated, ignore_index=True)
        else:
            print("  No validation results found")
            return pd.DataFrame(), {'batch_size': len(batch), 'validated_total': 0}, {}

        # Calculate statistics on ALL results
        stats = {
            'batch_size': len(batch),
            'validated_total': len(validated_from_batch),
            'validated_yes': len(validated_from_batch[validated_from_batch['abstract_support?'] == 'yes']),
            'validated_no': len(validated_from_batch[validated_from_batch['abstract_support?'] == 'no']),
            'validated_maybe': len(validated_from_batch[validated_from_batch['abstract_support?'] == 'maybe'])
        }

        print(f"\nPipeline Results (COMPLETE):")
        print(f"  Validated: {stats['validated_total']} / {stats['batch_size']} edges")
        print(f"    Yes: {stats['validated_yes']}")
        print(f"    No: {stats['validated_no']}")
        print(f"    Maybe: {stats['validated_maybe']}")

        return validated_from_batch, stats, cached_abstracts

    def _save_batch_evaluation_pairs(self, batch_num: int, eval_stats: Dict):
        """Save enriched evaluation pairs (with LLM sentences) from this batch."""
        eval_pairs_dir = Path(self.config.results_dir) / 'evaluation_pairs'
        eval_pairs_dir.mkdir(parents=True, exist_ok=True)

        # Get the most recently added edges
        latest_edges = self.data_manager.accumulated_validated[-1]

        output_path = eval_pairs_dir / f'batch_{batch_num:03d}_eval_pairs.parquet'
        latest_edges.to_parquet(output_path, index=False)

        print(f"   Saved {len(latest_edges)} enriched evaluation pairs to {output_path}")

    async def run(self):
        """
        Execute complete iterative batch optimization.

        MODIFIED per user specifications:
        1. Skip batches with n_yes == 0 (mark as processed)
        2. Save extended edges (yes/maybe) if n_yes > 0
        3. Add to evaluation: ALL yes + balanced no (or ALL yes if no no)
        4. Accumulate small increments until reaching min_new_edges_for_reoptimization
        """
        print("\n" + "="*80)
        print("STARTING ITERATIVE BATCH OPTIMIZATION")
        print("="*80)
        print(f"Predicate: {self.config.predicate}")
        print(f"Batch size: {self.config.batch_size}")
        print(f"Base evaluation edges: {len(self.data_manager.base_data)}")
        print(f"Edges to process: {self.batch_manager.total_edges}")
        print(f"Max batches: {self.config.max_batches}")
        print(f"Min edges for re-optimization: {self.config.min_new_edges_for_reoptimization}")

        # Initial evaluation
        print("\n" + "#"*80)
        print("# ITERATION 0: Initial Evaluation")
        print("#"*80)

        self.current_params = self._run_evaluation()

        iteration_record = {
            'iteration': 0,
            'timestamp': datetime.now().isoformat(),
            'batch_num': 0,
            'data_stats': self.data_manager.get_stats(),
            'params': self.current_params,
            'batch_stats': None
        }

        self.history.append(iteration_record)
        self._save_history()

        # Batch processing loop with accumulation
        edges_since_last_optimization = 0  # ACCUMULATOR
        iteration = 0
        batches_skipped = 0

        # Track batches processed in THIS run (not total batches)
        batches_processed_this_run = 0

        while batches_processed_this_run < self.config.max_batches:
            # Get next batch (this increments batch_manager.current_batch_num)
            batch = self.batch_manager.get_next_batch()
            if batch is None:
                print("✓ All edges processed!")
                break

            # Use the actual batch number from the manager
            actual_batch_num = self.batch_manager.current_batch_num
            batches_processed_this_run += 1

            print("\n" + "#"*80)
            print(f"# BATCH {actual_batch_num} (iteration {batches_processed_this_run}/{self.config.max_batches})")
            print("#"*80)

            # Process batch with current parameters
            validated_edges, batch_stats, cached_abstracts = await self._run_pipeline_on_batch(batch)

            # Check validation results
            n_yes = len(validated_edges[validated_edges['abstract_support?'] == 'yes'])
            n_maybe = len(validated_edges[validated_edges['abstract_support?'] == 'maybe'])
            n_no = len(validated_edges[validated_edges['abstract_support?'] == 'no'])

            print(f"\nValidation Summary:")
            print(f"  Yes: {n_yes}, Maybe: {n_maybe}, No: {n_no}")

            # DECISION: Skip batch if no 'yes' results
            if n_yes == 0:
                print(f"\n  SKIPPING BATCH {actual_batch_num}: No 'yes' results")
                print(f"  - No supporting evidence found")
                print(f"  - No evaluation pairs to add")
                print(f"  - Marking batch as processed and moving to next")

                # Mark as processed (don't retry)
                self.batch_manager.mark_batch_processed(batch)
                batches_skipped += 1

                # Save checkpoint
                self._save_checkpoint(actual_batch_num, batch_skipped=True, skip_reason='no_yes_results')

                continue  # Skip to next batch

            # We have yes results - proceed with saving

            # 1. Save extended edges (yes/maybe) for PRODUCTION
            print(f"\n[1/2] Saving extended edges (yes/maybe for production)...")
            extended_stats = self.extended_edges_manager.save_batch_extended_edges(
                actual_batch_num,
                batch,
                validated_edges
            )

            # 2. Add to evaluation dataset (yes + balanced no, or all yes)
            print(f"\n[2/2] Adding to evaluation dataset (yes + no for optimization)...")
            eval_stats = self.data_manager.add_validated_edges(
                validated_edges,
                batch,
                cached_abstracts
            )

            if eval_stats['n_new'] > 0:
                self._save_batch_evaluation_pairs(actual_batch_num, eval_stats)

            # Mark batch as processed
            self.batch_manager.mark_batch_processed(batch)


            # ACCUMULATION: Add new edges to counter
            n_new = eval_stats['n_new']
            edges_since_last_optimization += n_new

            #  FIX: Enhanced status display with balance info
            print(f"\nStatus Summary:")
            print(f"  Accumulation: {edges_since_last_optimization}/{self.config.min_new_edges_for_reoptimization} edges")
            print(f"  Balance: {eval_stats.get('balance_after', 0):+d} (yes - no)")
            print(f"  Cumulative: {self.data_manager.cumulative_yes_added} yes, {self.data_manager.cumulative_no_added} no")


            # DECISION: Re-optimize if accumulated enough edges
            should_optimize = (
                edges_since_last_optimization >= self.config.min_new_edges_for_reoptimization
            )

            if should_optimize:
                print("\n" + "="*70)
                print(f" TRIGGER: Re-optimization ({edges_since_last_optimization} accumulated edges)")
                print("="*70)

                iteration += 1

                # Re-run evaluation with augmented data
                new_params = self._run_evaluation()

                # Compare with previous params
                param_change = self._calculate_param_change(
                    self.current_params,
                    new_params
                )

                print(f"\nParameter Changes:")
                print(f"  Sentence threshold: {self.current_params['sentence_search']['threshold']:.4f} "
                      f" {new_params['sentence_search']['threshold']:.4f} "
                      f"({param_change['sentence_threshold_pct_change']:.1f}%)")
                print(f"  Abstract threshold: {self.current_params['abstract_classification']['threshold']:.4f} "
                      f" {new_params['abstract_classification']['threshold']:.4f} "
                      f"({param_change['abstract_threshold_pct_change']:.1f}%)")
                print(f"  Test MRR: {self.current_params['test_metrics']['MRR']:.4f} "
                      f" {new_params['test_metrics']['MRR']:.4f}")

                # Update current params
                self.current_params = new_params

                # Record iteration
                iteration_record = {
                    'iteration': iteration,
                    'timestamp': datetime.now().isoformat(),
                    'batch_num': actual_batch_num,
                    'data_stats': self.data_manager.get_stats(),
                    'params': new_params,
                    'param_change': param_change,
                    'batch_stats': batch_stats,
                    'edges_since_last_opt': edges_since_last_optimization
                }

                self.history.append(iteration_record)
                self._save_history()

                # Reset accumulator
                edges_since_last_optimization = 0
                print(f"  Accumulator reset to 0")

                self._save_checkpoint(
                    actual_batch_num,
                    extended_stats=extended_stats,
                    eval_stats=eval_stats,
                    edges_accumulated=edges_since_last_optimization,
                    reoptimization_completed=True
                )
            else:
                print(f"\n Deferred: {edges_since_last_optimization}/{self.config.min_new_edges_for_reoptimization} edges accumulated")
                print(f"   Will re-optimize when threshold is reached")

                self._save_checkpoint(
                    actual_batch_num,
                    extended_stats=extended_stats,
                    eval_stats=eval_stats,
                    edges_accumulated=edges_since_last_optimization,
                    reoptimization_completed=False
                )

            # Print progress
            progress = self.batch_manager.get_progress()
            print(f"\nOverall Progress:")
            print(f"  Processed: {progress['processed_edges']}/{progress['total_edges']} ({progress['progress_pct']:.1f}%)")
            print(f"  Batches completed: {progress['batches_completed']}")
            print(f"  Batches skipped: {batches_skipped}")

        # Final processing
        print("\n" + "="*80)
        print("SAVING COMPLETE EXTENDED EDGES")
        print("="*80)
        complete_filepath = self.extended_edges_manager.save_complete_extended_edges()

        # Delete checkpoint
        self.checkpoint_manager.delete_checkpoint()

        # Generate final report
        self._generate_final_report(batches_skipped)

    def _calculate_param_change(self, old_params: Dict, new_params: Dict) -> Dict:
        """Calculate percentage change in parameters."""
        old_sent = old_params['sentence_search']['threshold']
        new_sent = new_params['sentence_search']['threshold']

        old_abs = old_params['abstract_classification']['threshold']
        new_abs = new_params['abstract_classification']['threshold']

        return {
            'sentence_threshold_old': old_sent,
            'sentence_threshold_new': new_sent,
            'sentence_threshold_pct_change': ((new_sent - old_sent) / old_sent * 100) if old_sent > 0 else 0,
            'abstract_threshold_old': old_abs,
            'abstract_threshold_new': new_abs,
            'abstract_threshold_pct_change': ((new_abs - old_abs) / old_abs * 100) if old_abs > 0 else 0
        }

    def _generate_final_report(self, batches_skipped: int):
        """Generate final summary report with balance tracking."""
        print("\n" + "="*80)
        print("FINAL SUMMARY")
        print("="*80)

        final_stats = self.data_manager.get_stats()
        progress = self.batch_manager.get_progress()

        total_batches = progress['batches_completed']
        productive_batches = total_batches - batches_skipped

        print(f"\nProcessing Statistics:")
        print(f"  Total batches processed: {total_batches}")

        if total_batches > 0:
            print(f"  Productive batches: {productive_batches} ({productive_batches/total_batches*100:.1f}%)")
            print(f"  Skipped batches (no yes): {batches_skipped} ({batches_skipped/total_batches*100:.1f}%)")
        else:
            print(f"  Productive batches: {productive_batches}")
            print(f"  Skipped batches (no yes): {batches_skipped}")

        print(f"  Edges processed: {progress['processed_edges']}/{progress['total_edges']}")

        print(f"\nEvaluation Dataset Evolution:")
        print(f"  Initial size: {self.history[0]['data_stats']['base_edges']}")
        print(f"  Final size: {final_stats['total_edges']}")
        print(f"  Added edges: {final_stats['accumulated_edges']}")
        print(f"    Yes: {final_stats.get('cumulative_yes_added', 0)}")
        print(f"    No: {final_stats.get('cumulative_no_added', 0)}")
        print(f"    Balance: {final_stats.get('balance', 0):+d} (yes - no)")

        balance = final_stats.get('balance', 0)
        if balance > 0:
            print(f"    {balance} more yes than no")
            print(f"       (Acceptable - 'yes' examples are valuable)")
        elif balance < 0:
            print(f"    {-balance} more no than yes")
        else:
            print(f"    Perfectly balanced!")

        print(f"  Optimization iterations: {len([h for h in self.history if h['iteration'] > 0])}")

        print(f"\nParameter Evolution:")
        for record in self.history:
            if record['iteration'] > 0:
                print(f"  Iteration {record['iteration']}:")
                print(f"    After batch {record['batch_num']}")
                print(f"    Dataset size: {record['data_stats']['total_edges']}")
                print(f"    Sentence threshold: {record['params']['sentence_search']['threshold']:.4f}")
                print(f"    Abstract threshold: {record['params']['abstract_classification']['threshold']:.4f}")
                print(f"    Test MRR: {record['params']['test_metrics']['MRR']:.4f}")

        print(f"\nFinal Parameters:")
        print(f"  Sentence threshold: {self.current_params['sentence_search']['threshold']:.4f}")
        print(f"  Abstract threshold: {self.current_params['abstract_classification']['threshold']:.4f}")
        print(f"  Test MRR: {self.current_params['test_metrics']['MRR']:.4f}")

        # Save report with balance info
        report_path = Path(self.config.results_dir) / 'final_report.txt'
        with open(report_path, 'w') as f:
            f.write("="*80 + "\n")
            f.write("ITERATIVE BATCH OPTIMIZATION - FINAL REPORT\n")
            f.write("="*80 + "\n\n")
            f.write(f"Productive batches: {productive_batches}/{total_batches}\n")
            f.write(f"Skipped batches: {batches_skipped}\n")
            f.write(f"Final dataset size: {final_stats['total_edges']}\n")
            f.write(f"Yes added: {final_stats.get('cumulative_yes_added', 0)}\n")
            f.write(f"No added: {final_stats.get('cumulative_no_added', 0)}\n")
            f.write(f"Balance: {final_stats.get('balance', 0):+d}\n")
            f.write(f"Final MRR: {self.current_params['test_metrics']['MRR']:.4f}\n")

        print(f"\n Report saved to: {report_path}")
        print("="*80)


# ============================================================================
# USAGE EXAMPLE
# ============================================================================

async def main():

    config = IterativeBatchConfig(
            base_evaluation_data='edges/gold_df.pkl',
            nopub_edges_path='edges/treats_nopub.parquet',
            predicate='biolink:treats',
            batch_size=2,
            min_new_edges_for_reoptimization=1,
            max_batches=2
        )

    optimizer = IterativeBatchOptimizer(config)
    await optimizer.run()


if __name__ == "__main__":
    asyncio.run(main())
