"""
Biomedical Literature Retrieval Evaluation System
Compares edge representations (concat vs AI) and embedding models (general vs biomedical)
for evidence sentence retrieval and abstract classification.
"""

import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    precision_recall_curve, roc_curve,
    precision_score, recall_score, f1_score
)
from scipy.stats import bootstrap
from typing import List, Dict, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')


class BiomedicalRetrievalEvaluator:
    """Evaluate edge-to-sentence retrieval for biomedical literature."""

    def __init__(self, data_path: str, predicate: Optional[str] = None,
                 random_state: int = 42, embedding_cache_dir: str = './embeddings_cache'):
        """
        Initialize evaluator.

        Args:
            data_path: Path to pickle (.pkl) or CSV file
            predicate: Optional predicate filter
            random_state: Random seed for reproducibility
            embedding_cache_dir: Directory for caching embeddings
        """
        self.random_state = random_state
        self.predicate_filter = predicate
        self.embedding_cache_dir = embedding_cache_dir  # Add this line
        self.data = self._load_and_preprocess(data_path)
        self.models = {
            'general': 'sentence-transformers/all-MiniLM-L6-v2',
            'biomedical': 'NeuML/pubmedbert-base-embeddings'
        }
        self.results = {}

        # Cache for computed similarities to avoid redundant computation
        self.similarity_cache = {}

    def clear_similarity_cache(self):
        """Clear cached similarities to free memory."""
        self.similarity_cache = {}
        if hasattr(self, 'val_similarities'):
            del self.val_similarities
        print("Similarity cache cleared.")

    def _load_and_preprocess(self, data_path: str) -> pd.DataFrame:
        """Load data and exclude Maybe labels."""
        # Load data based on file extension
        if data_path.endswith('.pkl') or data_path.endswith('.pickle'):
            df = pd.read_pickle(data_path)
        elif data_path.endswith('.csv'):
            df = pd.read_csv(data_path)

            # Parse string lists if needed for CSV
            for col in ['abstract_sentences', 'gold_sent_idxs', 'ai_sentences']:
                if col in df.columns and df[col].dtype == 'object':
                    # Check if first non-null value is a string that needs parsing
                    sample = df[col].dropna().iloc[0] if len(df[col].dropna()) > 0 else None
                    if sample is not None and isinstance(sample, str):
                        df[col] = df[col].apply(
                            lambda x: eval(x) if isinstance(x, str) else x
                        )
        else:
            raise ValueError(f"Unsupported file format: {data_path}. Use .pkl or .csv")

        # Exclude "Maybe" labels (using lowercase as per data)
        original_len = len(df)
        df = df[df['abstract_support?'].isin(['yes', 'no'])].copy()

        print(f"Loaded {original_len} edges total")
        print(f"After excluding 'maybe': {len(df)} edges")

        # Apply predicate filter if specified
        if self.predicate_filter is not None:
            df = df[df['predicate'] == self.predicate_filter].copy()
            print(f"After filtering for predicate '{self.predicate_filter}': {len(df)} edges")

            if len(df) == 0:
                raise ValueError(f"No edges found with predicate '{self.predicate_filter}'")

        print(f"Label distribution:\n{df['abstract_support?'].value_counts()}")

        return df

    def describe_dataset(self):
        """Print descriptive statistics about the dataset."""
        print("\n" + "="*60)
        print("DATASET DESCRIPTION")
        print("="*60)

        if self.predicate_filter is not None:
            print(f"\nFiltered by predicate: '{self.predicate_filter}'")

        # Basic counts
        print(f"\nTotal edges: {len(self.data)}")
        print(f"\nLabel distribution:")
        label_counts = self.data['abstract_support?'].value_counts()
        for label, count in label_counts.items():
            print(f"  {label}: {count} ({count/len(self.data)*100:.1f}%)")

        print(f"\nPredicate distribution:")
        pred_counts = self.data['predicate'].value_counts()
        for pred, count in pred_counts.items():
            print(f"  {pred}: {count} ({count/len(self.data)*100:.1f}%)")

        # Abstract statistics
        abstract_lengths = self.data['abstract_sentences'].apply(len)
        print(f"\nAbstract length (sentences):")
        print(f"  Mean: {abstract_lengths.mean():.1f}")
        print(f"  Median: {abstract_lengths.median():.1f}")
        print(f"  Std Dev: {abstract_lengths.std():.1f}")
        print(f"  Min/Max: {abstract_lengths.min()}/{abstract_lengths.max()}")

        # Gold sentence statistics (yes edges only)
        yes_edges = self.data[self.data['abstract_support?'] == 'yes']
        n_gold_sents = yes_edges['gold_sent_idxs'].apply(len)
        print(f"\nGold sentences per 'yes' edge:")
        print(f"  Mean: {n_gold_sents.mean():.2f}")
        print(f"  Median: {n_gold_sents.median():.0f}")
        print(f"  Std Dev: {n_gold_sents.std():.2f}")
        print(f"  Distribution: {n_gold_sents.value_counts().sort_index().to_dict()}")

        # Position of gold sentences (normalized to [0, 1])
        gold_positions = []
        for _, row in yes_edges.iterrows():
            n_sents = len(row['abstract_sentences'])
            for idx in row['gold_sent_idxs']:
                # Normalize position to [0, 1]
                gold_positions.append(idx / (n_sents - 1) if n_sents > 1 else 0)

        print(f"\nGold sentence positions (0=start, 1=end):")
        print(f"  Mean: {np.mean(gold_positions):.2f}")
        print(f"  Median: {np.median(gold_positions):.2f}")
        print(f"  Std Dev: {np.std(gold_positions):.2f}")

        # Edge text length statistics
        concat_lengths = self.data['concat_sentence'].apply(lambda x: len(x.split()))
        print(f"\nEdge text length (words):")
        print(f"  Mean: {concat_lengths.mean():.1f}")
        print(f"  Median: {concat_lengths.median():.1f}")
        print(f"  Min/Max: {concat_lengths.min()}/{concat_lengths.max()}")

        print("="*60)

    def show_example_edges(self, n_examples: int = 2):
        """Display example edges to illustrate the task."""
        print("\n" + "="*60)
        print("EXAMPLE EDGES")
        print("="*60)

        if self.predicate_filter is not None:
            print(f"Filtered by predicate: '{self.predicate_filter}'")

        yes_edges = self.data[self.data['abstract_support?'] == 'yes']

        if len(yes_edges) < n_examples:
            n_examples = len(yes_edges)

        examples = yes_edges.sample(n=n_examples, random_state=self.random_state)

        for i, (_, row) in enumerate(examples.iterrows(), 1):
            print(f"\n{'-'*60}")
            print(f"Example {i}")
            print(f"{'-'*60}")
            print(f"Predicate: {row['predicate']}")
            print(f"Subject: {row['subject']}")
            print(f"Object: {row['object']}")
            print(f"\nEdge (concat): {row['concat_sentence']}")

            if 'ai_sentences' in row and isinstance(row['ai_sentences'], list):
                print(f"\nAI-generated sentences:")
                for j, sent in enumerate(row['ai_sentences'], 1):
                    print(f"  {j}. {sent}")

            print(f"\nAbstract (PMID: {row['pmid']}, {len(row['abstract_sentences'])} sentences):")
            gold_idxs = set(row['gold_sent_idxs'])
            for j, sent in enumerate(row['abstract_sentences']):
                if j in gold_idxs:
                    print(f"  >>> [{j}] {sent}  <<<  GOLD")
                else:
                    print(f"      [{j}] {sent}")

        print("="*60)

    def split_data(self, val_size: float = 0.5, shuffle: bool = False):
        """
        Split data into val/test sets (50/50) stratified by both label and predicate.

        With limited data (~1000 edges), we use all data for evaluation rather than
        holding out unused training data. This ensures maximum statistical power.

        This ensures each predicate type is proportionally represented in both splits,
        which is important if different predicates have different retrieval characteristics.

        Args:
            val_size: Proportion for validation (default 0.5 for 50/50 split)
            shuffle: Whether to shuffle within each stratum before splitting
        """
        val_df_list = []
        test_df_list = []

        # Get unique predicates and support types
        predicates = self.data['predicate'].unique()
        support_types = ['yes', 'no']  # Using lowercase as per data

        print(f"\nSplitting data into validation/test by predicate and label...")
        print(f"Found {len(predicates)} unique predicates: {predicates.tolist()}")

        # Split each predicate + label combination
        for support_type in support_types:
            for predicate in predicates:
                # Get subset for this predicate + label
                mask = ((self.data['abstract_support?'] == support_type) &
                       (self.data['predicate'] == predicate))
                subset = self.data[mask].copy()

                if len(subset) == 0:
                    continue

                # Shuffle if requested (for randomization)
                if shuffle:
                    subset = subset.sample(frac=1, random_state=self.random_state)

                # Calculate split index
                n = len(subset)
                val_end = round(val_size * n)

                # Split into validation and test
                val = subset.iloc[:val_end]
                test = subset.iloc[val_end:]

                val_df_list.append(val)
                test_df_list.append(test)

                print(f"  {predicate} + {support_type}: {len(val)} val, {len(test)} test")

        # Concatenate all splits
        val = pd.concat(val_df_list, ignore_index=True)
        test = pd.concat(test_df_list, ignore_index=True)

        # Optionally shuffle the final splits
        if shuffle:
            val = val.sample(frac=1, random_state=self.random_state).reset_index(drop=True)
            test = test.sample(frac=1, random_state=self.random_state).reset_index(drop=True)

        self.splits = {'val': val, 'test': test}

        print(f"\n{'='*60}")
        print("Final split sizes:")
        for name, split in self.splits.items():
            label_dist = split['abstract_support?'].value_counts().to_dict()
            predicate_dist = split['predicate'].value_counts().to_dict()
            print(f"  {name}: {len(split)} edges")
            print(f"    Labels: {label_dist}")
            print(f"    Predicates: {predicate_dist}")
        print(f"{'='*60}")

        return self.splits

    def compute_similarities(self, df: pd.DataFrame,
                        model_name: str,
                        representation: str,
                        use_cache: bool = True,
                        embedding_cache_dir: Optional[str] = None) -> pd.DataFrame:
        """
        Compute similarity scores between edge and abstract sentences.
        Now uses disk-based embedding cache for efficiency.

        Args:
            df: DataFrame with edges
            model_name: 'general' or 'biomedical'
            representation: 'concat' or 'ai'
            use_cache: Whether to use cached results if available
            embedding_cache_dir: Directory for embedding cache files

        Returns:
            DataFrame with added similarity scores
        """
            # Use instance cache dir if not specified
        if embedding_cache_dir is None:
            embedding_cache_dir = self.embedding_cache_dir
        # Create cache key based on data indices, model, and representation
        cache_key = f"{model_name}_{representation}_{id(df)}"

        # Check in-memory similarity cache first (fastest)
        if use_cache and cache_key in self.similarity_cache:
            print(f"  Using cached similarities for {representation}_{model_name}")
            return self.similarity_cache[cache_key]

        print(f"\nComputing similarities: {representation} + {model_name}")

        # Step 1: Get or compute embeddings (with disk cache)
        df_with_embeddings = self.compute_and_cache_embeddings(
            df, model_name, representation, embedding_cache_dir
        )

        # Step 2: Compute similarities from embeddings
        df_out = self.compute_similarities_from_embeddings(df_with_embeddings)

        # Cache the final result in memory
        if use_cache:
            self.similarity_cache[cache_key] = df_out

        return df_out

    def compute_and_cache_embeddings(self, df: pd.DataFrame,
                                     model_name: str,
                                     representation: str,
                                     cache_dir: str = './embeddings_cache') -> pd.DataFrame:
        """
        Compute embeddings and cache them to disk for reuse.
        Uses per-example caching to support incremental updates.

        Args:
            df: DataFrame with edges
            model_name: 'general' or 'biomedical'
            representation: 'concat' or 'ai'
            cache_dir: Directory to store cached embeddings

        Returns:
            DataFrame with embeddings added
        """
        import os
        import pickle
        import hashlib

        os.makedirs(cache_dir, exist_ok=True)

        # Create subdirectory for this model/representation combination
        model_cache_dir = os.path.join(cache_dir, f"{representation}_{model_name}")
        os.makedirs(model_cache_dir, exist_ok=True)

        print(f"  Checking embedding cache: {representation} + {model_name}")

        # Load model only if needed
        model = None

        edge_embeddings = []
        abstract_embeddings = []
        n_cached = 0
        n_computed = 0

        for idx, row in df.iterrows():
            # Create unique hash for this specific example
            # Hash based on: pmid, predicate, subject, object, and representation
            example_key = f"{row.get('pmid', idx)}_{row['predicate']}_{row['subject']}_{row['object']}_{representation}"
            example_hash = hashlib.md5(example_key.encode()).hexdigest()

            cache_file = os.path.join(model_cache_dir, f"{example_hash}.pkl")

            # Try to load from cache
            if os.path.exists(cache_file):
                try:
                    with open(cache_file, 'rb') as f:
                        cached_example = pickle.load(f)

                    # Verify the cached data matches
                    if (cached_example.get('pmid') == row.get('pmid', idx) and
                        cached_example.get('predicate') == row['predicate']):

                        edge_embeddings.append(cached_example['edge_embedding'])
                        abstract_embeddings.append(cached_example['abstract_embeddings'])
                        n_cached += 1
                        continue
                except Exception as e:
                    # If cache is corrupted, recompute
                    pass

            # Not in cache or cache invalid - compute embeddings
            if model is None:
                print(f"  Loading model: {self.models[model_name]}")
                model = SentenceTransformer(self.models[model_name])

            # Compute edge embedding
            if representation == 'concat':
                edge_text = row['concat_sentence']
                edge_emb = model.encode([edge_text], convert_to_numpy=True)[0]
            else:  # ai
                ai_sents = row['ai_sentences']
                ai_embs = model.encode(ai_sents, convert_to_numpy=True)
                edge_emb = ai_embs.mean(axis=0)

            # Compute abstract embeddings
            abstract_sents = row['abstract_sentences']
            abstract_embs = model.encode(abstract_sents, convert_to_numpy=True)

            edge_embeddings.append(edge_emb)
            abstract_embeddings.append(abstract_embs)

            # Save to cache
            cache_data = {
                'pmid': row.get('pmid', idx),
                'predicate': row['predicate'],
                'subject': row['subject'],
                'object': row['object'],
                'edge_embedding': edge_emb,
                'abstract_embeddings': abstract_embs,
                'representation': representation,
                'model': model_name
            }

            try:
                with open(cache_file, 'wb') as f:
                    pickle.dump(cache_data, f)
                n_computed += 1
            except Exception as e:
                print(f"  Warning: Could not cache embedding for {example_hash}: {e}")

        print(f"  ✓ Embeddings: {n_cached} from cache, {n_computed} newly computed")

        # Add to dataframe
        df_out = df.copy()
        df_out['edge_embedding'] = edge_embeddings
        df_out['abstract_embeddings'] = abstract_embeddings

        return df_out


    def compute_similarities_from_embeddings(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute similarities from pre-computed embeddings.

        Args:
            df: DataFrame with 'edge_embedding' and 'abstract_embeddings' columns

        Returns:
            DataFrame with similarity scores added
        """
        results = []

        for idx, row in df.iterrows():
            edge_emb = row['edge_embedding']
            abstract_embs = row['abstract_embeddings']

            # Compute cosine similarities
            similarities = np.dot(abstract_embs, edge_emb) / (
                np.linalg.norm(abstract_embs, axis=1) * np.linalg.norm(edge_emb)
            )

            results.append({
                'index': idx,
                'similarities': similarities.tolist(),
                'max_sim': similarities.max(),
                'top2_mean_sim': np.sort(similarities)[-2:].mean() if len(similarities) >= 2 else similarities.mean(),
                'top3_mean_sim': np.sort(similarities)[-3:].mean() if len(similarities) >= 3 else similarities.mean()
            })

        # Merge back
        results_df = pd.DataFrame(results).set_index('index')
        df_out = df.copy()
        df_out['similarities'] = results_df['similarities']
        df_out['max_sim'] = results_df['max_sim']
        df_out['top2_mean_sim'] = results_df['top2_mean_sim']
        df_out['top3_mean_sim'] = results_df['top3_mean_sim']

        return df_out

    def precompute_all_embeddings(self, cache_dir: str = './embeddings_cache'):
        """
        Pre-compute and cache embeddings for all model/representation combinations.
        This saves time when running multiple evaluations or iterations.

        Args:
            cache_dir: Directory to store cached embeddings
        """
        print("\n" + "="*60)
        print("PRE-COMPUTING ALL EMBEDDINGS")
        print("="*60)

        if not hasattr(self, 'splits'):
            raise ValueError("Must run split_data() first")

        configs = [
            ('concat', 'general'), ('concat', 'biomedical'),
            ('ai', 'general'), ('ai', 'biomedical')
        ]

        for split_name in ['val', 'test']:
            df = self.splits[split_name]
            print(f"\n{split_name.upper()} set ({len(df)} examples):")

            for rep, model in configs:
                config_name = f"{rep}_{model}"
                print(f"  {config_name}...", end=' ', flush=True)

                # This will cache embeddings to disk
                _ = self.compute_and_cache_embeddings(
                    df, model, rep, cache_dir
                )
                print("✓")

        print("\n" + "="*60)
        print("All embeddings pre-computed and cached!")
        print(f"Cache location: {cache_dir}")
        print("="*60)

    def evaluate_sentence_level(self, df: pd.DataFrame,
                                only_yes: bool = True) -> Dict:
        """
        Evaluate sentence-level retrieval metrics.

        Args:
            df: DataFrame with similarities computed
            only_yes: Only evaluate on yes edges (with gold sentences)

        Returns:
            Dictionary of metrics
        """
        if only_yes:
            df = df[df['abstract_support?'] == 'yes'].copy()

        mrr_scores = []
        ranks = []
        recalls_at_k = {1: [], 3: [], 5: []}

        # For AUC metrics, collect all sentence-level predictions
        all_labels = []
        all_scores = []

        for _, row in df.iterrows():
            similarities = np.array(row['similarities'])
            gold_idxs = set(row['gold_sent_idxs'])

            # Rank sentences by similarity (descending)
            ranked_idxs = np.argsort(similarities)[::-1]

            # Find rank of first gold sentence
            first_gold_rank = None
            for rank, idx in enumerate(ranked_idxs, 1):
                if idx in gold_idxs:
                    first_gold_rank = rank
                    break

            if first_gold_rank:
                mrr_scores.append(1.0 / first_gold_rank)
                ranks.append(first_gold_rank)

            # Recall@k: proportion of gold sentences in top-k
            for k in [1, 3, 5]:
                top_k = set(ranked_idxs[:k])
                recall = len(top_k & gold_idxs) / len(gold_idxs)
                recalls_at_k[k].append(recall)

            # For AUC: binary labels for each sentence
            labels = np.array([1 if i in gold_idxs else 0
                              for i in range(len(similarities))])
            all_labels.extend(labels)
            all_scores.extend(similarities)

        # Compute metrics
        metrics = {
            'MRR': np.mean(mrr_scores),
            'mean_rank': np.mean(ranks),
            'recall_at_k_1': np.mean(recalls_at_k[1]),
            'recall_at_k_3': np.mean(recalls_at_k[3]),
            'recall_at_k_5': np.mean(recalls_at_k[5]),
        }

        # AUC metrics (if we have both positive and negative examples)
        if len(set(all_labels)) > 1:
            metrics['AUC_ROC'] = roc_auc_score(all_labels, all_scores)
            metrics['AUC_PR'] = average_precision_score(all_labels, all_scores)

        return metrics

    def evaluate_abstract_level(self, df: pd.DataFrame,
                                aggregation: str = 'max') -> Dict:
        """
        Evaluate abstract-level classification.

        Args:
            df: DataFrame with similarities computed
            aggregation: 'max' or 'top3_mean'

        Returns:
            Dictionary with scores for threshold tuning
        """
        # Get aggregated scores
        scores = df[f'{aggregation}_sim'].values
        labels = (df['abstract_support?'] == 'yes').astype(int).values

        # Compute AUC metrics
        roc_auc = roc_auc_score(labels, scores)
        pr_auc = average_precision_score(labels, scores)

        return {
            'scores': scores,
            'labels': labels,
            'ROC_AUC': roc_auc,
            'PR_AUC': pr_auc
        }

    def tune_threshold_for_precision(self, df: pd.DataFrame,
                                     aggregation: str = 'max',
                                     target_recall: float = 0.7) -> float:
        """
        Find threshold that maximizes precision at target recall.

        Args:
            df: Validation data with similarities
            aggregation: 'max' or 'top3_mean'
            target_recall: Minimum recall to maintain

        Returns:
            Optimal threshold
        """
        scores = df[f'{aggregation}_sim'].values
        labels = (df['abstract_support?'] == 'yes').astype(int).values

        precision, recall, thresholds = precision_recall_curve(labels, scores)

        # Find threshold where recall >= target_recall
        valid_idxs = np.where(recall >= target_recall)[0]
        if len(valid_idxs) == 0:
            print(f"Warning: Cannot achieve recall={target_recall}, using max precision threshold")
            best_idx = np.argmax(precision[:-1])  # Exclude last point
        else:
            # Among valid thresholds, pick one with max precision
            best_idx = valid_idxs[np.argmax(precision[valid_idxs])]

        optimal_threshold = thresholds[best_idx] if best_idx < len(thresholds) else 0.5

        print(f"  Optimal threshold: {optimal_threshold:.4f}")
        print(f"    Precision: {precision[best_idx]:.4f}")
        print(f"    Recall: {recall[best_idx]:.4f}")

        return optimal_threshold

    def evaluate_with_threshold(self, df: pd.DataFrame,
                               threshold: float,
                               aggregation: str = 'max') -> Dict:
        """Evaluate classification metrics at given threshold."""
        scores = df[f'{aggregation}_sim'].values
        labels = (df['abstract_support?'] == 'yes').astype(int).values
        predictions = (scores >= threshold).astype(int)

        return {
            'precision': precision_score(labels, predictions, zero_division=0),
            'recall': recall_score(labels, predictions, zero_division=0),
            'f1': f1_score(labels, predictions, zero_division=0)
        }

    def run_validation_pipeline(self):
        """
        Run full validation pipeline to select best configuration.
        Returns a comparison table and selects the best configuration.
        """
        print("\n" + "="*60)
        print("VALIDATION PHASE: Comparing configurations")
        print("="*60)

        if self.predicate_filter is not None:
            print(f"Filtered by predicate: '{self.predicate_filter}'")

        val_df = self.splits['val']

        results = {}
        comparison_data = []

        # Store validation similarities in cache for reuse
        self.val_similarities = {}

        for rep in ['concat', 'ai']:
            for model in ['general', 'biomedical']:
                config_name = f"{rep}_{model}"
                print(f"\n--- Configuration: {config_name} ---")

                # Compute similarities (will be cached)
                df_sim = self.compute_similarities(val_df, model, rep, use_cache=True)

                # Store for validation reference
                self.val_similarities[config_name] = df_sim

                # Sentence-level evaluation
                sent_metrics = self.evaluate_sentence_level(df_sim)

                # Abstract-level: try all aggregations
                best_agg = None
                best_pr_auc = 0
                best_roc_auc = 0
                for agg in ['max', 'top2_mean', 'top3_mean']:
                    abs_metrics = self.evaluate_abstract_level(df_sim, agg)
                    if abs_metrics['PR_AUC'] > best_pr_auc:
                        best_pr_auc = abs_metrics['PR_AUC']
                        best_roc_auc = abs_metrics['ROC_AUC']
                        best_agg = agg

                print(f"Best aggregation: {best_agg} (PR-AUC: {best_pr_auc:.4f})")

                # Tune threshold for best aggregation
                threshold = self.tune_threshold_for_precision(df_sim, best_agg)

                # Store results
                results[config_name] = {
                    'sentence_metrics': sent_metrics,
                    'best_aggregation': best_agg,
                    'abstract_pr_auc': best_pr_auc,
                    'abstract_roc_auc': best_roc_auc,
                    'optimal_threshold': threshold
                }

                # Prepare row for comparison table
                comparison_data.append({
                    'Configuration': config_name,
                    'Representation': rep,
                    'Model': model,
                    'MRR': sent_metrics['MRR'],
                    'Recall@1': sent_metrics['recall_at_k_1'],
                    'Recall@3': sent_metrics['recall_at_k_3'],
                    'Recall@5': sent_metrics['recall_at_k_5'],
                    'Sent-AUC-PR': sent_metrics.get('AUC_PR', 0),
                    'Best Agg': best_agg,
                    'Abs-PR-AUC': best_pr_auc,
                    'Abs-ROC-AUC': best_roc_auc,
                    'Threshold': threshold
                })

        # Create comparison DataFrame
        comparison_df = pd.DataFrame(comparison_data)

        # Sort by MRR (descending)
        comparison_df = comparison_df.sort_values('MRR', ascending=False).reset_index(drop=True)

        self.validation_results = results
        self.validation_comparison = comparison_df

        # Display the comparison table
        print("\n" + "="*60)
        print("VALIDATION RESULTS COMPARISON")
        print("="*60)
        print("\nSentence-Level Metrics:")
        print(comparison_df[['Configuration', 'MRR', 'Recall@1', 'Recall@3', 'Recall@5', 'Sent-AUC-PR']].to_string(index=False))

        print("\nAbstract-Level Metrics:")
        print(comparison_df[['Configuration', 'Best Agg', 'Abs-PR-AUC', 'Abs-ROC-AUC', 'Threshold']].to_string(index=False))

        # Select best configuration (by sentence-level MRR + abstract PR-AUC)
        best_idx = (comparison_df['MRR'] + comparison_df['Abs-PR-AUC']).idxmax()
        best_config_name = comparison_df.loc[best_idx, 'Configuration']

        print(f"\n{'='*60}")
        print(f"BEST CONFIGURATION: {best_config_name}")
        print(f"  Combined Score (MRR + Abs-PR-AUC): {comparison_df.loc[best_idx, 'MRR'] + comparison_df.loc[best_idx, 'Abs-PR-AUC']:.4f}")
        print(f"{'='*60}")

        self.best_config = {
            'name': best_config_name,
            'representation': best_config_name.split('_')[0],
            'model': best_config_name.split('_')[1],
            'aggregation': results[best_config_name]['best_aggregation'],
            'threshold': results[best_config_name]['optimal_threshold']
        }

        return self.best_config

    def run_test_evaluation(self):
        """Run final evaluation on test set with best configuration."""
        print("\n" + "="*60)
        print("TEST PHASE: Final evaluation with frozen configuration")
        print("="*60)

        if self.predicate_filter is not None:
            print(f"Filtered by predicate: '{self.predicate_filter}'")

        config = self.best_config
        test_df = self.splits['test']

        print(f"Configuration: {config['name']}")
        print(f"  Representation: {config['representation']}")
        print(f"  Model: {config['model']}")
        print(f"  Aggregation: {config['aggregation']}")
        print(f"  Threshold: {config['threshold']:.4f}")

        # Compute similarities (will be cached)
        df_sim = self.compute_similarities(
            test_df, config['model'], config['representation'], use_cache=True
        )

        # Sentence-level metrics
        sent_metrics = self.evaluate_sentence_level(df_sim)
        print(f"\nSentence-level metrics:")
        for k, v in sent_metrics.items():
            print(f"  {k}: {v:.4f}")

        # Abstract-level metrics
        abs_metrics = self.evaluate_abstract_level(df_sim, config['aggregation'])
        print(f"\nAbstract-level metrics:")
        print(f"  ROC-AUC: {abs_metrics['ROC_AUC']:.4f}")
        print(f"  PR-AUC: {abs_metrics['PR_AUC']:.4f}")

        # Classification at threshold
        clf_metrics = self.evaluate_with_threshold(
            df_sim, config['threshold'], config['aggregation']
        )
        print(f"\nClassification at threshold {config['threshold']:.4f}:")
        for k, v in clf_metrics.items():
            print(f"  {k}: {v:.4f}")

        # Bootstrap confidence intervals for key metrics
        print(f"\nBootstrap 95% CIs (1000 iterations):")
        ci_metrics = self._bootstrap_confidence_intervals(df_sim, config)

        # Store results for test analysis
        self.test_results = {
            'sentence_metrics': sent_metrics,
            'abstract_metrics': abs_metrics,
            'classification_metrics': clf_metrics,
            'confidence_intervals': ci_metrics,
            'df_with_predictions': df_sim
        }

        return self.test_results

    def _bootstrap_confidence_intervals(self, df: pd.DataFrame,
                                       config: Dict,
                                       n_iterations: int = 1000) -> Dict:
        """Compute bootstrap CIs for key metrics."""
        rng = np.random.default_rng(self.random_state)

        # Manual bootstrap implementation (more reliable for DataFrames)
        n_samples = len(df)

        mrr_scores = []
        pr_auc_scores = []

        print(f"  Running {n_iterations} bootstrap iterations...")

        for i in range(n_iterations):
            # Sample with replacement
            boot_indices = rng.choice(n_samples, size=n_samples, replace=True)
            df_boot = df.iloc[boot_indices].reset_index(drop=True)

            # Compute MRR
            try:
                sent_metrics = self.evaluate_sentence_level(df_boot)
                mrr_scores.append(sent_metrics['MRR'])
            except Exception as e:
                # Skip if bootstrap sample has issues
                print(f"    Warning: Bootstrap iteration {i+1} failed for MRR: {e}")

            # Compute PR-AUC
            try:
                abs_metrics = self.evaluate_abstract_level(df_boot, config['aggregation'])
                pr_auc_scores.append(abs_metrics['PR_AUC'])
            except Exception as e:
                # Skip if bootstrap sample has issues
                print(f"    Warning: Bootstrap iteration {i+1} failed for PR-AUC: {e}")

            # Progress indicator
            if (i + 1) % 200 == 0:
                print(f"    {i + 1}/{n_iterations} iterations complete")

        # Compute 95% confidence intervals
        mrr_ci = (np.percentile(mrr_scores, 2.5), np.percentile(mrr_scores, 97.5))
        pr_auc_ci = (np.percentile(pr_auc_scores, 2.5), np.percentile(pr_auc_scores, 97.5))

        cis = {
            'MRR': mrr_ci,
            'PR_AUC': pr_auc_ci
        }

        print(f"  MRR: [{mrr_ci[0]:.4f}, {mrr_ci[1]:.4f}]")
        print(f"  PR-AUC: [{pr_auc_ci[0]:.4f}, {pr_auc_ci[1]:.4f}]")

        return cis

    def analyze_sentence_threshold(self,
                                   representation: Optional[str] = None,
                                   model: Optional[str] = None):
        """
        Analyze sentence-level similarity thresholds for semantic search.
        Helps determine optimal range_filter for Milvus vector search.

        Args:
            representation: Edge representation to use ('concat' or 'ai').
                          If None, uses best_config from test phase.
            model: Embedding model to use ('general' or 'biomedical').
                  If None, uses best_config from test phase.

        Returns:
            DataFrame with threshold analysis results
        """
        print("\n" + "="*60)
        print("SENTENCE-LEVEL THRESHOLD ANALYSIS")
        print("="*60)
        print("Purpose: Find optimal similarity threshold for vector search filtering")

        if self.predicate_filter is not None:
            print(f"Filtered by predicate: '{self.predicate_filter}'")

        # Use provided config or default to best config from test
        if representation is None or model is None:
            if not hasattr(self, 'best_config'):
                raise ValueError("No configuration specified and no best_config available. "
                               "Run run_validation_pipeline() first or specify representation and model.")
            representation = representation or self.best_config['representation']
            model = model or self.best_config['model']

        print(f"\nUsing configuration: {representation}_{model}")

        if not hasattr(self, 'test_results'):
            print("Warning: Test results not found. Computing similarities on test set...")
            if not hasattr(self, 'splits'):
                raise ValueError("No test split available. Run split_data() first.")

            test_df = self.splits['test']
            test_df = self.compute_similarities(test_df, model, representation, use_cache=True)
        else:
            # Check if current test_results match the requested config
            if (hasattr(self, 'best_config') and
                representation == self.best_config['representation'] and
                model == self.best_config['model']):
                test_df = self.test_results['df_with_predictions']
            else:
                # Need to recompute with different config
                print(f"  Recomputing similarities for requested configuration...")
                test_df = self.splits['test']
                test_df = self.compute_similarities(test_df, model, representation, use_cache=True)

        # Collect all sentence similarities with labels
        all_gold_sims = []
        all_non_gold_sims = []

        for _, row in test_df.iterrows():
            sims = np.array(row['similarities'])
            if row['abstract_support?'] == 'yes':
                gold_idxs = set(row['gold_sent_idxs'])
                all_gold_sims.extend([sims[i] for i in gold_idxs])
                all_non_gold_sims.extend([sims[i] for i in range(len(sims)) if i not in gold_idxs])
            else:
                all_non_gold_sims.extend(sims)

        # Calculate statistics
        # print(f"\nSentence Similarity Statistics:")
        # print(f"  Gold sentences (n={len(all_gold_sims)}):")
        # print(f"    Mean: {np.mean(all_gold_sims):.3f}")
        # print(f"    Median: {np.median(all_gold_sims):.3f}")
        # print(f"    Std: {np.std(all_gold_sims):.3f}")
        # print(f"    Min/Max: {np.min(all_gold_sims):.3f} / {np.max(all_gold_sims):.3f}")
        # print(f"    25th/75th percentile: {np.percentile(all_gold_sims, 25):.3f} / {np.percentile(all_gold_sims, 75):.3f}")
        #
        # print(f"\n  Non-gold sentences (n={len(all_non_gold_sims)}):")
        # print(f"    Mean: {np.mean(all_non_gold_sims):.3f}")
        # print(f"    Median: {np.median(all_non_gold_sims):.3f}")
        # print(f"    Std: {np.std(all_non_gold_sims):.3f}")

        # Test different thresholds
        thresholds = np.arange(0.3, 0.8, 0.05)
        results = []

        for tau in thresholds:
            # At this threshold, what % of gold vs non-gold pass?
            gold_pass = np.sum(np.array(all_gold_sims) >= tau)
            gold_recall = gold_pass / len(all_gold_sims)

            non_gold_pass = np.sum(np.array(all_non_gold_sims) >= tau)

            # Precision = gold_pass / (gold_pass + non_gold_pass)
            precision = gold_pass / (gold_pass + non_gold_pass) if (gold_pass + non_gold_pass) > 0 else 0

            results.append({
                'threshold': tau,
                'gold_recall': gold_recall,
                'precision': precision,
                'gold_pass': gold_pass,
                'non_gold_pass': non_gold_pass
            })

        results_df = pd.DataFrame(results)

        # Find optimal thresholds for different goals
        print(f"\n{'='*60}")
        print("RECOMMENDED MILVUS PARAMETERS:")
        print(f"{'='*60}")

        # Recommend radius (lower bound) based on recall goals
        print(f"\nFor Milvus range search with COSINE metric:")
        print(f"  (Returns entities with similarity in [radius, range_filter])")

        # Goal: Balanced F1
        results_df['f1'] = 2 * (results_df['precision'] * results_df['gold_recall']) / (results_df['precision'] + results_df['gold_recall'])
        best_f1 = results_df.loc[results_df['f1'].idxmax()]
        print(f"\n  For BALANCED (Max F1):")
        print(f"    radius: {best_f1['threshold']:.4f}")
        print(f"    range_filter: 1.0")
        print(f"    → Precision: {best_f1['precision']:.4f}, Recall: {best_f1['gold_recall']:.4f}, F1: {best_f1['f1']:.4f}")

        print(f"\n{'='*60}")

        return results_df


    def _determine_best_sentence_threshold(self) -> Dict:
        """
        Determine best sentence-level threshold by comparing ai+general and concat+general.
        Selection criteria (in order):
        1. Higher F1
        2. If F1 equal, higher precision
        3. If precision equal, higher recall
        4. If all equal, prefer concat+general

        Returns:
            Dict with best threshold configuration
        """
        print("\n" + "="*60)
        print("DETERMINING BEST SENTENCE-LEVEL THRESHOLD")
        print("="*60)

        # Run threshold analysis for both representations with general model
        print("\nAnalyzing AI + General:")
        results_ai = self.analyze_sentence_threshold(
            representation='ai',
            model='general'
        )

        print("\nAnalyzing Concat + General:")
        results_concat = self.analyze_sentence_threshold(
            representation='concat',
            model='general'
        )

        # Find best F1 for each
        best_ai = results_ai.loc[results_ai['f1'].idxmax()]
        best_concat = results_concat.loc[results_concat['f1'].idxmax()]

        print(f"\n{'='*60}")
        print("COMPARISON:")
        print(f"{'='*60}")
        print(f"AI + General:")
        print(f"  Threshold: {best_ai['threshold']:.4f}")
        print(f"  F1: {best_ai['f1']:.4f}, Precision: {best_ai['precision']:.4f}, Recall: {best_ai['gold_recall']:.4f}")

        print(f"\nConcat + General:")
        print(f"  Threshold: {best_concat['threshold']:.4f}")
        print(f"  F1: {best_concat['f1']:.4f}, Precision: {best_concat['precision']:.4f}, Recall: {best_concat['gold_recall']:.4f}")

        # Selection logic
        if best_ai['f1'] > best_concat['f1']:
            selected = 'ai'
            best = best_ai
        elif best_concat['f1'] > best_ai['f1']:
            selected = 'concat'
            best = best_concat
        elif best_ai['precision'] > best_concat['precision']:
            selected = 'ai'
            best = best_ai
        elif best_concat['precision'] > best_ai['precision']:
            selected = 'concat'
            best = best_concat
        elif best_ai['gold_recall'] > best_concat['gold_recall']:
            selected = 'ai'
            best = best_ai
        elif best_concat['gold_recall'] > best_ai['gold_recall']:
            selected = 'concat'
            best = best_concat
        else:
            # All equal, prefer concat
            selected = 'concat'
            best = best_concat

        print(f"\n{'='*60}")
        print(f"SELECTED: {selected.upper()} + GENERAL")
        print(f"{'='*60}")

        return {
            'representation': selected,
            'model': 'general',
            'threshold': float(best['threshold']),
            'f1': float(best['f1']),
            'precision': float(best['precision']),
            'recall': float(best['gold_recall'])
        }

    def _prepare_evaluation_record(self) -> Dict:
        """
        Prepare a complete evaluation record with all results and parameters.

        Returns:
            Dict containing all evaluation information
        """
        if not hasattr(self, 'test_results') or not hasattr(self, 'best_config'):
            raise ValueError("Must run validation and test evaluation before saving results")

        # Determine best sentence-level threshold
        sentence_threshold_config = self._determine_best_sentence_threshold()

        # Prepare the record
        from datetime import datetime

        record = {
            'timestamp': datetime.now().isoformat(),
            'n_examples': len(self.data),
            'n_val': len(self.splits['val']),
            'n_test': len(self.splits['test']),
            'predicate': self.predicate_filter,

            # Best configuration from validation
            'best_config': {
                'name': self.best_config['name'],
                'representation': self.best_config['representation'],
                'model': self.best_config['model'],
                'model_path': self.models[self.best_config['model']],
                'aggregation': self.best_config['aggregation']
            },

            # Validation results (comparison table)
            'validation_comparison': self.validation_comparison.to_dict('records'),

            # Test results
            'test_results': {
                'sentence_level': {
                    'MRR': float(self.test_results['sentence_metrics']['MRR']),
                    'mean_rank': float(self.test_results['sentence_metrics']['mean_rank']),
                    'recall_at_1': float(self.test_results['sentence_metrics']['recall_at_k_1']),
                    'recall_at_3': float(self.test_results['sentence_metrics']['recall_at_k_3']),
                    'recall_at_5': float(self.test_results['sentence_metrics']['recall_at_k_5']),
                    'AUC_ROC': float(self.test_results['sentence_metrics'].get('AUC_ROC', 0)),
                    'AUC_PR': float(self.test_results['sentence_metrics'].get('AUC_PR', 0))
                },
                'abstract_level': {
                    'ROC_AUC': float(self.test_results['abstract_metrics']['ROC_AUC']),
                    'PR_AUC': float(self.test_results['abstract_metrics']['PR_AUC'])
                },
                'classification': {
                    'precision': float(self.test_results['classification_metrics']['precision']),
                    'recall': float(self.test_results['classification_metrics']['recall']),
                    'f1': float(self.test_results['classification_metrics']['f1'])
                },
                'confidence_intervals': {
                    'MRR': [float(x) for x in self.test_results['confidence_intervals']['MRR']],
                    'PR_AUC': [float(x) for x in self.test_results['confidence_intervals']['PR_AUC']]
                }
            },

            # Parameters for deployment
            'parameters': {
                'abstract_classification': {
                    'model': self.best_config['model'],
                    'model_path': self.models[self.best_config['model']],
                    'representation': self.best_config['representation'],
                    'aggregation': self.best_config['aggregation'],
                    'threshold': float(self.best_config['threshold'])
                },
                'sentence_search': {
                    'model': 'general',
                    'model_path': self.models['general'],
                    'representation': sentence_threshold_config['representation'],
                    'threshold': sentence_threshold_config['threshold'],
                    'f1': sentence_threshold_config['f1'],
                    'precision': sentence_threshold_config['precision'],
                    'recall': sentence_threshold_config['recall']
                }
            }
        }

        return record

    def save_results(self, filepath: str = 'evaluation_results.json'):
        """
        Save evaluation results and parameters to JSON file.
        Implements append/replace logic based on n_examples per predicate.
        Maintains latest parameters for each predicate separately.

        Args:
            filepath: Path to JSON file
        """
        import json
        import os
        from datetime import datetime

        # Prepare current evaluation record
        current_record = self._prepare_evaluation_record()
        current_predicate = current_record['predicate']

        # Load existing history if file exists
        if os.path.exists(filepath):
            with open(filepath, 'r') as f:
                data = json.load(f)

            history = data.get('evaluation_history', [])
            latest_by_predicate = data.get('latest_parameters_by_predicate', {})
        else:
            # New file
            print(f"\nCreating new evaluation history file")
            history = []
            latest_by_predicate = {}

        # Find if there's already a record for this predicate with same n_examples
        replaced = False
        for i, record in enumerate(history):
            if (record.get('predicate') == current_predicate and
                record.get('n_examples') == current_record['n_examples']):
                # Replace this record
                print(f"\nReplacing existing record for predicate '{current_predicate}' "
                      f"with n_examples={current_record['n_examples']}")
                history[i] = current_record
                replaced = True
                break

        if not replaced:
            # Append new record at the beginning (most recent first)
            print(f"\nAppending new record for predicate '{current_predicate}', "
                  f"n_examples={current_record['n_examples']}")
            history.insert(0, current_record)

        # Update latest parameters for this predicate
        latest_by_predicate[current_predicate] = {
            'timestamp': current_record['timestamp'],
            'n_examples': current_record['n_examples'],
            'n_val': current_record['n_val'],
            'n_test': current_record['n_test'],
            'abstract_classification': current_record['parameters']['abstract_classification'],
            'sentence_search': current_record['parameters']['sentence_search'],
            'test_metrics': {
                'MRR': current_record['test_results']['sentence_level']['MRR'],
                'recall_at_3': current_record['test_results']['sentence_level']['recall_at_3'],
                'PR_AUC': current_record['test_results']['abstract_level']['PR_AUC'],
                'classification_f1': current_record['test_results']['classification']['f1']
            }
        }

        # Get list of all predicates that have been evaluated
        predicates_evaluated = sorted(latest_by_predicate.keys())

        # Prepare output data
        output = {
            'evaluation_history': history,
            'latest_parameters_by_predicate': latest_by_predicate,
            'summary': {
                'total_evaluations': len(history),
                'predicates_evaluated': predicates_evaluated,
                'last_updated': current_record['timestamp'],
                'last_predicate_updated': current_predicate
            }
        }

        # Save to file
        with open(filepath, 'w') as f:
            json.dump(output, f, indent=2)

        print(f"\nResults saved to: {filepath}")
        print(f"Total evaluations in history: {len(history)}")
        print(f"Predicates with latest parameters: {predicates_evaluated}")

    def load_history(self, filepath: str = 'evaluation_results.json') -> Dict:
        """
        Load evaluation history from JSON file.

        Args:
            filepath: Path to JSON file

        Returns:
            Dict containing evaluation history
        """
        import json
        import os

        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Evaluation history file not found: {filepath}")

        with open(filepath, 'r') as f:
            data = json.load(f)

        print(f"\nLoaded evaluation history from: {filepath}")
        print(f"Total evaluations: {len(data.get('evaluation_history', []))}")

        if 'latest_metadata' in data:
            meta = data['latest_metadata']
            print(f"Latest evaluation:")
            print(f"  Timestamp: {meta['timestamp']}")
            print(f"  Predicate: {meta['predicate']}")
            print(f"  N examples: {meta['n_examples']}")

        return data

    def get_latest_parameters(self, filepath: str = 'evaluation_results.json', predicate: Optional[str] = None) -> Dict:
        """
        Get the latest parameters for semantic search and classification.

        Args:
            filepath: Path to JSON file
            predicate: Specific predicate to get parameters for.
                      If None, returns all predicates' parameters.

        Returns:
            Dict with parameters ready for deployment
        """
        data = self.load_history(filepath)

        latest_by_predicate = data.get('latest_parameters_by_predicate', {})

        if not latest_by_predicate:
            print("\nNo parameters found in file.")
            return {}

        # If specific predicate requested
        if predicate is not None:
            if predicate not in latest_by_predicate:
                available = list(latest_by_predicate.keys())
                raise ValueError(
                    f"No parameters found for predicate '{predicate}'. "
                    f"Available predicates: {available}"
                )

            params = latest_by_predicate[predicate]

            print("\n" + "="*60)
            print(f"LATEST PARAMETERS FOR PREDICATE: '{predicate}'")
            print("="*60)
            print(f"Timestamp: {params['timestamp']}")
            print(f"Based on {params['n_examples']} examples "
                  f"({params['n_val']} val, {params['n_test']} test)")

            print("\nSentence-Level Search:")
            print(f"  Model: {params['sentence_search']['model_path']}")
            print(f"  Representation: {params['sentence_search']['representation']}")
            print(f"  Threshold: {params['sentence_search']['threshold']:.3f}")
            print(f"  Expected F1: {params['sentence_search']['f1']:.3f}")
            print(f"  Expected Precision: {params['sentence_search']['precision']:.3f}")
            print(f"  Expected Recall: {params['sentence_search']['recall']:.3f}")

            print("\nAbstract-Level Classification:")
            print(f"  Model: {params['abstract_classification']['model_path']}")
            print(f"  Representation: {params['abstract_classification']['representation']}")
            print(f"  Aggregation: {params['abstract_classification']['aggregation']}")
            print(f"  Threshold: {params['abstract_classification']['threshold']:.3f}")

            print("\nTest Performance:")
            print(f"  MRR: {params['test_metrics']['MRR']:.3f}")
            print(f"  Recall@3: {params['test_metrics']['recall_at_3']:.3f}")
            print(f"  PR-AUC: {params['test_metrics']['PR_AUC']:.3f}")
            print(f"  Classification F1: {params['test_metrics']['classification_f1']:.3f}")
            print("="*60)

            return params

        # Return all predicates' parameters
        print("\n" + "="*60)
        print("LATEST PARAMETERS FOR ALL PREDICATES")
        print("="*60)

        for pred in sorted(latest_by_predicate.keys()):
            params = latest_by_predicate[pred]
            print(f"\n{pred}:")
            print(f"  Updated: {params['timestamp']}")
            print(f"  N examples: {params['n_examples']}")
            print(f"  Sentence threshold: {params['sentence_search']['threshold']:.3f} "
                  f"({params['sentence_search']['representation']} + {params['sentence_search']['model']})")
            print(f"  Abstract threshold: {params['abstract_classification']['threshold']:.3f} "
                  f"({params['abstract_classification']['representation']} + "
                  f"({params['abstract_classification']['model']} + "
                  f"{params['abstract_classification']['aggregation']})")
            print(f"  Test MRR: {params['test_metrics']['MRR']:.3f}, "
                  f"PR-AUC: {params['test_metrics']['PR_AUC']:.3f}")

        print("="*60)

        return latest_by_predicate

    def check_embedding_cache_status(self, cache_dir: str = './embeddings_cache'):
        """
        Check the status of the embedding cache.

        Args:
            cache_dir: Directory containing cached embeddings
        """
        import os

        print("\n" + "="*60)
        print("EMBEDDING CACHE STATUS")
        print("="*60)

        if not os.path.exists(cache_dir):
            print(f"Cache directory does not exist: {cache_dir}")
            return

        configs = [
            ('concat', 'general'), ('concat', 'biomedical'),
            ('ai', 'general'), ('ai', 'biomedical')
        ]

        total_files = 0
        total_size_mb = 0

        for rep, model in configs:
            model_cache_dir = os.path.join(cache_dir, f"{rep}_{model}")

            if os.path.exists(model_cache_dir):
                files = [f for f in os.listdir(model_cache_dir) if f.endswith('.pkl')]
                n_files = len(files)

                # Calculate total size
                size_bytes = sum(
                    os.path.getsize(os.path.join(model_cache_dir, f))
                    for f in files
                )
                size_mb = size_bytes / (1024 * 1024)

                print(f"\n{rep}_{model}:")
                print(f"  Cached examples: {n_files}")
                print(f"  Cache size: {size_mb:.2f} MB")

                total_files += n_files
                total_size_mb += size_mb
            else:
                print(f"\n{rep}_{model}: No cache")

        print(f"\n{'='*60}")
        print(f"TOTAL:")
        print(f"  Cached examples: {total_files}")
        print(f"  Total cache size: {total_size_mb:.2f} MB")
        print(f"{'='*60}")

# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    # Initialize evaluator
    evaluator = BiomedicalRetrievalEvaluator(
        data_path='your_data.pkl',  # Replace with your .pkl file path
        predicate=None,  # Set to specific predicate (e.g., 'treats') or None for all
        random_state=42
    )

    # Print basic dataset info
    print("\n" + "="*60)
    print("DATASET OVERVIEW")
    print("="*60)
    evaluator.describe_dataset()
    evaluator.show_example_edges(n_examples=2)

    # Split data into 50/50 validation/test
    print("\n" + "="*60)
    print("DATA SPLITTING")
    print("="*60)
    evaluator.split_data(val_size=0.5)

    # PHASE 1: VALIDATION (compare all configurations, select best)
    print("\n" + "="*60)
    print("PHASE 1: VALIDATION")
    print("="*60)
    best_config = evaluator.run_validation_pipeline()

    # PHASE 2: TEST (evaluate best configuration on unseen data)
    print("\n" + "="*60)
    print("PHASE 2: TEST")
    print("="*60)
    test_results = evaluator.run_test_evaluation()

    # Analyze sentence-level thresholds for Milvus deployment
    # Option 1: Use best config from test (default)
    threshold_analysis = evaluator.analyze_sentence_threshold()

    # Option 2: Use custom configuration
    # threshold_analysis = evaluator.analyze_sentence_threshold(
    #     representation='ai',
    #     model='biomedical'
    # )

    # Save results with parameters
    evaluator.save_results(filepath='evaluation_results.json')

    # FINAL SUMMARY
    print("\n" + "="*60)
    print("EVALUATION COMPLETE")
    print("="*60)
    print(f"\nBest configuration: {best_config['name']}")
    print(f"  Representation: {best_config['representation']}")
    print(f"  Model: {best_config['model']}")
    print(f"  Aggregation: {best_config['aggregation']}")
    print(f"  Threshold: {best_config['threshold']:.4f} (tuned on validation)")
    print(f"\nTest Results:")
    print(f"  MRR: {test_results['sentence_metrics']['MRR']:.4f}")
    print(f"  Recall@1: {test_results['sentence_metrics']['recall_at_k_1']:.4f}")
    print(f"  Recall@3: {test_results['sentence_metrics']['recall_at_k_3']:.4f}")
    print(f"  Recall@5: {test_results['sentence_metrics']['recall_at_k_5']:.4f}")
    print(f"  PR-AUC: {test_results['abstract_metrics']['PR_AUC']:.4f}")
    print(f"\nResults and parameters saved to: evaluation_results.json")
    print("="*60)

def load_search_parameters(filepath: str = 'evaluation_results.json',
                          predicate: Optional[str] = None) -> Dict:
    """
    Standalone function to load parameters in a different kernel/script.

    Usage in another kernel:
        from biomed_eval_cleaned import load_search_parameters

        # Get parameters for specific predicate
        params = load_search_parameters('evaluation_results.json', predicate='treats')

        # Get parameters for all predicates
        all_params = load_search_parameters('evaluation_results.json')

    Args:
        filepath: Path to JSON file
        predicate: Specific predicate to get parameters for.
                  If None, returns all predicates' parameters.

    Returns:
        Dict with parameters for deployment
    """
    import json
    import os

    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Evaluation results file not found: {filepath}")

    with open(filepath, 'r') as f:
        data = json.load(f)

    latest_by_predicate = data.get('latest_parameters_by_predicate', {})

    if not latest_by_predicate:
        raise ValueError("No parameters found in file")

    # If specific predicate requested
    if predicate is not None:
        if predicate not in latest_by_predicate:
            available = list(latest_by_predicate.keys())
            raise ValueError(
                f"No parameters found for predicate '{predicate}'. "
                f"Available predicates: {available}"
            )

        params = latest_by_predicate[predicate]
        print(f"Loaded parameters for predicate '{predicate}'")
        print(f"  Timestamp: {params['timestamp']}")
        print(f"  Based on {params['n_examples']} examples")

        return params

    # Return all predicates
    print(f"Loaded parameters for {len(latest_by_predicate)} predicates:")
    for pred in sorted(latest_by_predicate.keys()):
        params = latest_by_predicate[pred]
        print(f"  {pred}: {params['n_examples']} examples (updated {params['timestamp']})")

    return latest_by_predicate
