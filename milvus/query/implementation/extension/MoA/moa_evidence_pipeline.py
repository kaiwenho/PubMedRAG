"""
MoA Hypothesis Evidence Pipeline
A pipeline for finding PubMed evidence supporting mechanism of action (MoA) hypotheses.

Adapted from pub_extension_pipeline.py for a different use case:
- Input: A list of MoA sentences representing a multi-hop hypothesis path
  (e.g., "Metformin activates AMPK", "AMPK inhibits mTOR", "mTOR regulates autophagy")
- No predicate-specific thresholds or abstract classification step
- Enriched LLM validation with reasoning traces and confidence scores
- Final synthesis view assessing the overall hypothesis

Pipeline Steps:
1. Encode MoA sentences as query embeddings
2. Semantic search for relevant PubMed sentences (threshold + top-k per abstract)
3. Batch-fetch and cache abstracts
4. LLM validation with reasoning traces (two-round)
5. Hypothesis synthesis - overall assessment of the multi-hop path
"""

import json
import time
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from sentence_transformers import SentenceTransformer
from pymilvus import MilvusClient, connections, utility
from nltk.tokenize import sent_tokenize
from collections import Counter
import asyncio


# ============================================================================
# CONFIGURATION
# ============================================================================

class MoAConfig:
    """Configuration for the MoA evidence pipeline."""

    def __init__(
        self,
        ss_model_path: str = 'sentence-transformers/all-MiniLM-L6-v2',
        ss_threshold: float = 0.5,
        top_k_abstracts: int = 300,
        milvus_uri: str = "http://localhost:19530",
        milvus_token: str = "root:Milvus",
        num_collections: int = 10,
        round1_model: str = 'gpt-oss:20b',
        round2_model: str = 'gpt-oss:120b',
        context_window: int = 8192,
        output_dir: str = 'result/moa_evidence',
    ):
        """
        Args:
            ss_model_path: Sentence embedding model for semantic search
            ss_threshold: Minimum cosine similarity for semantic search hits
            top_k_abstracts: Maximum number of unique abstracts per MoA sentence
            milvus_uri: Milvus server URI
            milvus_token: Milvus authentication token
            num_collections: Number of Milvus collections to search
            round1_model: LLM model for round 1 validation
            round2_model: LLM model for round 2 validation
            context_window: LLM context window size
            output_dir: Directory for output files
        """
        self.ss_model_path = ss_model_path
        self.ss_threshold = ss_threshold
        self.top_k_abstracts = top_k_abstracts
        self.milvus_uri = milvus_uri
        self.milvus_token = milvus_token
        self.num_collections = num_collections
        self.round1_model = round1_model
        self.round2_model = round2_model
        self.context_window = context_window
        self.output_dir = output_dir


# ============================================================================
# DATA MODELS
# ============================================================================

class MoASentence:
    """
    A single MoA sentence with optional structured context.

    Required:
        sentence: Natural language statement (e.g., "Metformin activates AMPK")

    Optional context (improves LLM validation quality):
        subject_info: Dict with name, description, category, synonyms, etc.
        object_info: Dict with name, description, category, synonyms, etc.
        predicate_info: Dict with name, description, inverse, etc.
    """

    def __init__(
        self,
        sentence: str,
        subject_info: Optional[Dict[str, Any]] = None,
        object_info: Optional[Dict[str, Any]] = None,
        predicate_info: Optional[Dict[str, Any]] = None,
    ):
        self.sentence = sentence
        self.subject_info = subject_info
        self.object_info = object_info
        self.predicate_info = predicate_info

    @property
    def has_structured_context(self) -> bool:
        return any([self.subject_info, self.object_info, self.predicate_info])

    def __repr__(self):
        return f"MoASentence('{self.sentence}')"


class MoAHypothesis:
    """
    A multi-hop MoA hypothesis represented as an ordered list of MoA sentences.

    Example:
        Metformin -> activates -> AMPK -> inhibits -> mTOR -> regulates -> autophagy
        represented as:
        [
            MoASentence("Metformin activates AMPK"),
            MoASentence("AMPK inhibits mTOR"),
            MoASentence("mTOR regulates autophagy"),
        ]
    """

    def __init__(self, sentences: List[MoASentence], name: Optional[str] = None):
        """
        Args:
            sentences: Ordered list of MoA sentences forming the hypothesis path
            name: Optional human-readable name for the hypothesis
        """
        self.sentences = sentences
        self.name = name or self._auto_name()

    def _auto_name(self) -> str:
        """Generate a name from the first and last sentences."""
        if len(self.sentences) == 1:
            return self.sentences[0].sentence
        first = self.sentences[0].sentence
        last = self.sentences[-1].sentence
        return f"{first} ... {last}"

    @property
    def path_description(self) -> str:
        """Human-readable description of the full hypothesis path."""
        return " → ".join(s.sentence for s in self.sentences)

    def __len__(self):
        return len(self.sentences)

    def __repr__(self):
        return f"MoAHypothesis('{self.name}', {len(self.sentences)} hops)"


# ============================================================================
# QUERY ENCODING
# ============================================================================

class MoAQueryEncoder:
    """Encode MoA sentences into embedding vectors for semantic search."""

    def __init__(self, model: SentenceTransformer):
        self.model = model

    def encode_hypothesis(self, hypothesis: MoAHypothesis) -> List[Dict]:
        """
        Encode all sentences in a hypothesis.

        Returns:
            List of dicts with 'sentence_idx', 'sentence', and 'embedding'
        """
        query_vectors = []

        for idx, moa_sentence in enumerate(hypothesis.sentences):
            embedding = self.model.encode(
                [moa_sentence.sentence], convert_to_numpy=True
            )[0]

            query_vectors.append({
                'sentence_idx': idx,
                'sentence': moa_sentence.sentence,
                'embedding': embedding,
            })

        print(f"  Encoded {len(query_vectors)} MoA sentences")
        return query_vectors


# ============================================================================
# SEMANTIC SEARCH
# ============================================================================

class MoASemanticSearcher:
    """Perform semantic search in Milvus collections for MoA sentences."""

    def __init__(self, uri: str, token: str):
        self.uri = uri
        self.token = token
        self.client = None
        self._setup_connection()

    def _setup_connection(self):
        """Establish Milvus connection."""
        connections.connect(
            alias="default",
            uri=self.uri,
            token=self.token,
        )
        self._wait_for_node()
        self.client = MilvusClient(uri=self.uri, token=self.token)

    def _wait_for_node(self, resource_group: str = "__default_resource_group",
                       interval: int = 5):
        """Wait for Milvus node to be available."""
        while True:
            info = utility.describe_resource_group(name=resource_group)
            num_available = info.num_available_node
            print(f"  Node availability: {num_available}")
            if num_available >= 1:
                print("  Node is available — continuing execution.")
                return
            print(f"  No nodes available, retrying in {interval}s…")
            time.sleep(interval)

    def search(
        self,
        query_vectors: List[Dict],
        threshold: float,
        top_k_abstracts: int,
        num_collections: int = 10,
    ) -> Tuple[Dict[int, List[Dict]], Dict[int, List[Dict]]]:
        """
        Search across Milvus collections for each MoA sentence.

        Aggregates sentence-level hits, deduplicates to unique PMIDs per query,
        and applies the top-k abstract cutoff.

        Args:
            query_vectors: List of dicts with 'sentence_idx' and 'embedding'
            threshold: Minimum cosine similarity (radius parameter)
            top_k_abstracts: Maximum number of unique abstracts per sentence
            num_collections: Number of Milvus collections to search

        Returns:
            Tuple of:
              - sentence_pmids: Dict[sentence_idx -> List[Dict{pmid, max_score}]]
                (deduplicated, sorted, top-k abstracts)
              - search_counts: Dict[sentence_idx -> List[Dict]] (per-collection counts)
        """
        # Raw hits per sentence: sentence_idx -> list of {pmid, distance}
        raw_hits: Dict[int, List[Dict]] = {qv['sentence_idx']: [] for qv in query_vectors}
        search_counts: Dict[int, List[Dict]] = {qv['sentence_idx']: [] for qv in query_vectors}

        start = time.time()

        for i in range(num_collections):
            collection_name = f"pubmed_sentence_{i:02d}"
            print(f"  Loading collection: {collection_name}")
            self.client.load_collection(collection_name=collection_name)

            for qv in query_vectors:
                idx = qv['sentence_idx']
                emb = qv['embedding']

                results = self.client.search(
                    collection_name=collection_name,
                    data=[emb],
                    limit=200,
                    search_params={
                        "params": {
                            "radius": threshold,
                            "range_filter": 1.0,
                        }
                    },
                    output_fields=["pmid"],
                )

                hits = results[0]
                raw_hits[idx].extend(hits)
                search_counts[idx].append({collection_name: len(hits)})

            self.client.release_collection(collection_name=collection_name)
            print(f"  Finished and unloaded: {collection_name}")

        elapsed = time.time() - start
        print(f"  Semantic search time: {elapsed:.2f}s")

        # Aggregate to PMID level: keep max score per PMID per sentence
        sentence_pmids = {}
        for idx, hits in raw_hits.items():
            pmid_best: Dict[str, float] = {}
            for hit in hits:
                pmid = hit['entity']['pmid']
                score = hit['distance']
                if pmid not in pmid_best or score > pmid_best[pmid]:
                    pmid_best[pmid] = score

            # Sort by score descending, apply top-k cutoff
            sorted_pmids = sorted(pmid_best.items(), key=lambda x: x[1], reverse=True)
            top_k = sorted_pmids[:top_k_abstracts]

            sentence_pmids[idx] = [
                {'pmid': pmid, 'max_score': score} for pmid, score in top_k
            ]

            total_unique = len(pmid_best)
            kept = len(top_k)
            print(f"  Sentence {idx}: {total_unique} unique PMIDs → top-{kept} kept")

        return sentence_pmids, search_counts


# ============================================================================
# ABSTRACT CACHE
# ============================================================================

class AbstractCache:
    """
    Fetch and cache PubMed abstracts.
    Each abstract is fetched at most once across all MoA sentences.
    """

    def __init__(self):
        self.cache: Dict[str, Dict] = {}  # pmid -> {abstract, sentences}

    async def fetch_and_cache(
        self,
        sentence_pmids: Dict[int, List[Dict]],
        batch_size: int = 100,
    ) -> Dict[str, Dict]:
        """
        Collect all unique PMIDs across sentences, fetch abstracts, cache them.

        Args:
            sentence_pmids: Dict[sentence_idx -> List[Dict{pmid, max_score}]]
            batch_size: Number of PMIDs to fetch per API call

        Returns:
            The cache dict (pmid -> {abstract, sentences})
        """
        from pubmed_client import get_publication_info

        # Collect all unique PMIDs
        all_pmids = set()
        for idx, pmid_list in sentence_pmids.items():
            for entry in pmid_list:
                pmid = entry['pmid']
                # Normalize PMID format
                if not str(pmid).startswith('PMID:'):
                    pmid = f"PMID:{pmid}"
                all_pmids.add(pmid)

        # Remove already-cached PMIDs
        new_pmids = [p for p in all_pmids if p not in self.cache]
        print(f"  Total unique PMIDs: {len(all_pmids)}")
        print(f"  Already cached: {len(all_pmids) - len(new_pmids)}")
        print(f"  To fetch: {len(new_pmids)}")

        # Batch fetch
        new_pmids = list(new_pmids)
        for i in range(0, len(new_pmids), batch_size):
            batch = new_pmids[i:i + batch_size]
            batch_num = i // batch_size + 1
            total_batches = (len(new_pmids) - 1) // batch_size + 1
            print(f"  Fetching batch {batch_num}/{total_batches} ({len(batch)} PMIDs)...")

            try:
                result = await get_publication_info(batch, 'moa_pipeline')

                if result['_meta']['n_results'] > 0:
                    for pmid in batch:
                        abstract = result['results'].get(pmid, {}).get('abstract')
                        if abstract:
                            sentences = sent_tokenize(abstract)
                            self.cache[pmid] = {
                                'abstract': abstract,
                                'sentences': sentences,
                            }
            except Exception as e:
                print(f"  Warning: Batch {batch_num} failed: {e}")

        print(f"  Cache now holds {len(self.cache)} abstracts")
        return self.cache

    def get(self, pmid: str) -> Optional[Dict]:
        """Retrieve a cached abstract."""
        # Try both with and without PMID: prefix
        if pmid in self.cache:
            return self.cache[pmid]
        prefixed = f"PMID:{pmid}" if not str(pmid).startswith('PMID:') else pmid
        return self.cache.get(prefixed)


# ============================================================================
# LLM VALIDATOR
# ============================================================================

class MoALLMValidator:
    """
    LLM-based validation of abstract support for MoA sentences.

    Features:
    - Full hypothesis path context in every prompt
    - Optional structured entity/predicate context per sentence
    - Reasoning trace and confidence output
    - Two-round validation (small model filter → large model verification)
    """

    def __init__(
        self,
        llm_client: Any,
        response_parser: Any,
        round1_model: str = 'gpt-oss:20b',
        round2_model: str = 'gpt-oss:120b',
        context_window: int = 8192,
    ):
        self.llm_client = llm_client
        self.response_parser = response_parser
        self.round1_model = round1_model
        self.round2_model = round2_model
        self.context_window = context_window

    # ------------------------------------------------------------------
    # Prompt generation
    # ------------------------------------------------------------------

    def _build_context_block(self, moa_sentence: MoASentence) -> str:
        """Build optional structured context block for a single MoA sentence."""
        if not moa_sentence.has_structured_context:
            return ""

        parts = []
        if moa_sentence.subject_info:
            parts.append(f"Subject details: {json.dumps(moa_sentence.subject_info)}")
        if moa_sentence.object_info:
            parts.append(f"Object details: {json.dumps(moa_sentence.object_info)}")
        if moa_sentence.predicate_info:
            parts.append(f"Predicate details: {json.dumps(moa_sentence.predicate_info)}")

        return "\n".join(parts)

    def generate_validation_prompt(
        self,
        hypothesis: MoAHypothesis,
        sentence_idx: int,
        abstract: str,
    ) -> str:
        """
        Generate a validation prompt for one MoA sentence + one abstract.

        The prompt includes:
        - The full hypothesis path (for mechanistic context)
        - The specific sentence being evaluated (highlighted)
        - Optional structured context for the sentence
        - The abstract to evaluate
        - Instructions for reasoning, confidence, support, and sentence extraction
        """
        moa_sentence = hypothesis.sentences[sentence_idx]

        # Build the hypothesis path with the current sentence highlighted
        path_lines = []
        for i, s in enumerate(hypothesis.sentences):
            marker = "  >>>" if i == sentence_idx else "     "
            path_lines.append(f"{marker} Step {i+1}: {s.sentence}")
        path_block = "\n".join(path_lines)

        # Build structured context if available
        context_block = self._build_context_block(moa_sentence)
        context_section = ""
        if context_block:
            context_section = f"""
Additional context for the evaluated relationship:
{context_block}
"""

        return f"""Please analyze whether the provided abstract supports a specific step in a mechanism of action (MoA) hypothesis.

=== FULL HYPOTHESIS PATH ===
{path_block}

=== RELATIONSHIP BEING EVALUATED ===
"{moa_sentence.sentence}"
{context_section}
=== ABSTRACT ===
{abstract}

=== INSTRUCTIONS ===
1. First, reason step-by-step about whether the abstract provides evidence for the specific relationship "{moa_sentence.sentence}".
   Consider:
   - Does the abstract explicitly state this relationship?
   - Does the abstract describe experiments or observations that support it?
   - Is the evidence direct or indirect?
   - Are there any caveats, contradictions, or limitations?
   - Consider the broader hypothesis path — does the abstract discuss this relationship in a mechanistic context consistent with the hypothesis?

2. Based on your reasoning, provide:
   - "support": "yes" if the abstract explicitly supports the relationship
   - "support": "no" if the relationship is not mentioned or is contradicted
   - "support": "maybe" if the evidence is indirect, ambiguous, or suggestive
   - "confidence": "high", "medium", or "low" — your confidence in your judgment
   - "reasoning": A brief explanation (2-4 sentences) of your logic
   - "sentences": If "support" is "yes", provide one or more exact sentences from the abstract that support the relationship. If "no" or "maybe", return an empty list.

Output Format: Return ONLY a JSON object:
{{
  "support": "yes" | "no" | "maybe",
  "confidence": "high" | "medium" | "low",
  "reasoning": "...",
  "sentences": ["..."]
}}
"""

    # ------------------------------------------------------------------
    # Prompt preparation
    # ------------------------------------------------------------------

    def prepare_prompts(
        self,
        hypothesis: MoAHypothesis,
        sentence_pmids: Dict[int, List[Dict]],
        abstract_cache: AbstractCache,
    ) -> List[Dict]:
        """
        Prepare validation prompts for all sentence-PMID pairs.

        Args:
            hypothesis: The MoA hypothesis
            sentence_pmids: Dict[sentence_idx -> List[Dict{pmid, max_score}]]
            abstract_cache: The abstract cache

        Returns:
            List of prompt dicts with keys:
              sentence_idx, pmid, search_score, prompt
        """
        prompts = []

        for sentence_idx, pmid_entries in sentence_pmids.items():
            for entry in pmid_entries:
                pmid = entry['pmid']
                score = entry['max_score']

                # Normalize PMID
                pmid_key = pmid if str(pmid).startswith('PMID:') else f"PMID:{pmid}"

                abstract_data = abstract_cache.get(pmid_key)
                if not abstract_data:
                    continue

                abstract = abstract_data['abstract']
                prompt = self.generate_validation_prompt(
                    hypothesis, sentence_idx, abstract
                )

                prompts.append({
                    'sentence_idx': sentence_idx,
                    'pmid': pmid_key,
                    'search_score': score,
                    'prompt': prompt,
                })

        print(f"  Prepared {len(prompts)} validation prompts")
        for idx in sorted(sentence_pmids.keys()):
            count = sum(1 for p in prompts if p['sentence_idx'] == idx)
            print(f"    Sentence {idx}: {count} prompts")

        return prompts

    # ------------------------------------------------------------------
    # Validation rounds
    # ------------------------------------------------------------------

    def _run_validation_round(
        self, prompts: List[Dict], model: str
    ) -> Tuple[List[Dict], List[Dict]]:
        """
        Run a single validation round.

        Returns:
            Tuple of (raw_responses, parsed_results)
        """
        responses = []
        parsed_results = []

        for i, prompt_info in enumerate(prompts):
            messages = [{'role': 'user', 'content': prompt_info['prompt']}]

            try:
                response = self.llm_client.chat(
                    model=model,
                    messages=messages,
                    options={'num_ctx': self.context_window, 'temperature': 0},
                )
                response_text = response['message']['content']
            except Exception as e:
                print(f"  Warning: LLM call failed for sentence {prompt_info['sentence_idx']}, "
                      f"PMID {prompt_info['pmid']}: {e}")
                response_text = ""

            responses.append({
                'sentence_idx': prompt_info['sentence_idx'],
                'pmid': prompt_info['pmid'],
                'search_score': prompt_info['search_score'],
                'response': response_text,
            })

            # Parse
            parsed = self.response_parser.parse_response(response_text)
            if parsed:
                parsed_results.append({
                    'sentence_idx': prompt_info['sentence_idx'],
                    'pmid': prompt_info['pmid'],
                    'search_score': prompt_info['search_score'],
                    'extraction_status': 'success',
                    'extracted_data': parsed,
                })
            else:
                parsed_results.append({
                    'sentence_idx': prompt_info['sentence_idx'],
                    'pmid': prompt_info['pmid'],
                    'search_score': prompt_info['search_score'],
                    'extraction_status': 'failed',
                    'extracted_data': {},
                })

            if (i + 1) % 50 == 0:
                print(f"    Processed {i+1}/{len(prompts)} prompts")

        return responses, parsed_results

    def _print_round_summary(self, results: List[Dict], round_name: str):
        """Print summary for a validation round."""
        print(f"\n  {round_name} Summary:")
        print(f"    Total: {len(results)}")

        status_counts = Counter(r['extraction_status'] for r in results)
        print(f"    Parsed successfully: {status_counts.get('success', 0)}")
        print(f"    Parse failures: {status_counts.get('failed', 0)}")

        support_counts = Counter(
            r['extracted_data'].get('support')
            for r in results
            if r['extraction_status'] == 'success'
        )
        if support_counts:
            print(f"    Support distribution:")
            for stype, count in support_counts.most_common():
                print(f"      {stype}: {count}")

        confidence_counts = Counter(
            r['extracted_data'].get('confidence')
            for r in results
            if r['extraction_status'] == 'success'
        )
        if confidence_counts:
            print(f"    Confidence distribution:")
            for ctype, count in confidence_counts.most_common():
                print(f"      {ctype}: {count}")

    def run_round1(self, prompts: List[Dict]) -> Tuple[List[Dict], List[Dict]]:
        """
        Round 1: fast model, filter out definitive 'no' results.

        Returns:
            Tuple of (supporting_results, no_support_results)
        """
        print(f"\n  Running Round 1 with model: {self.round1_model}")
        _, parsed_results = self._run_validation_round(prompts, self.round1_model)
        self._print_round_summary(parsed_results, "Round 1")

        supporting = []
        no_support = []

        for result in parsed_results:
            if result['extraction_status'] == 'success':
                support = result['extracted_data'].get('support')
                if support in ['yes', 'maybe']:
                    supporting.append(result)
                else:
                    no_support.append(result)
            # Parse failures are dropped

        print(f"\n  Round 1 filtering:")
        print(f"    Supporting (yes/maybe): {len(supporting)} → proceed to Round 2")
        print(f"    Non-supporting (no): {len(no_support)} → filtered out")

        return supporting, no_support

    def run_round2(
        self, round1_supporting: List[Dict], prompts: List[Dict]
    ) -> Tuple[List[Dict], List[Dict]]:
        """
        Round 2: larger model validates yes/maybe cases for quality assurance.

        Returns:
            Tuple of (final_supporting, additional_no_support)
        """
        if not round1_supporting:
            print("  No supporting results to validate in Round 2")
            return [], []

        # Build prompt lookup
        prompt_lookup = {}
        for p in prompts:
            key = (p['sentence_idx'], p['pmid'])
            prompt_lookup[key] = p

        # Collect prompts for round 2
        r2_prompts = []
        for result in round1_supporting:
            key = (result['sentence_idx'], result['pmid'])
            if key in prompt_lookup:
                r2_prompts.append(prompt_lookup[key])

        print(f"\n  Running Round 2 with model: {self.round2_model}")
        print(f"  Validating {len(r2_prompts)} yes/maybe cases")

        _, r2_parsed = self._run_validation_round(r2_prompts, self.round2_model)
        self._print_round_summary(r2_parsed, "Round 2")

        # Build R2 lookup
        r2_lookup = {}
        for result in r2_parsed:
            key = (result['sentence_idx'], result['pmid'])
            r2_lookup[key] = result

        # Merge: R2 overrides R1
        final_supporting = []
        additional_no = []

        for r1_result in round1_supporting:
            key = (r1_result['sentence_idx'], r1_result['pmid'])
            r2_result = r2_lookup.get(key)

            if r2_result and r2_result['extraction_status'] == 'success':
                support = r2_result['extracted_data'].get('support')
                if support == 'no':
                    additional_no.append(r2_result)
                else:
                    final_supporting.append(r2_result)
            else:
                # No R2 result or parse failure — keep R1
                final_supporting.append(r1_result)

        print(f"\n  Round 2 results:")
        print(f"    Still supporting: {len(final_supporting)}")
        print(f"    Downgraded to no: {len(additional_no)}")

        return final_supporting, additional_no

    # ------------------------------------------------------------------
    # Sentence mapping (reused from utils.py)
    # ------------------------------------------------------------------

    def map_sentences_to_indices(
        self,
        supporting_results: List[Dict],
        abstract_cache: AbstractCache,
    ) -> List[Dict]:
        """
        Map LLM-extracted sentences back to abstract sentence indices.

        Uses the existing utils.py matching logic with LLM fallback.
        """
        from utils import match_llm_sentences_to_indices

        for result in supporting_results:
            extracted = result.get('extracted_data', {})
            support = extracted.get('support')
            llm_sentences = extracted.get('sentences', [])

            if support != 'yes' or not llm_sentences:
                result['gold_sent_idxs'] = []
                continue

            pmid = result['pmid']
            abstract_data = abstract_cache.get(pmid)
            if not abstract_data:
                result['gold_sent_idxs'] = []
                continue

            indices, error = match_llm_sentences_to_indices(
                llm_sentences,
                abstract_data,
                result['sentence_idx'],
                pmid,
                return_partial_matches=True,
            )

            result['gold_sent_idxs'] = indices
            if error:
                result['mapping_warning'] = (
                    f"{len(error.get('unmatched_sentences', []))} sentences could not be mapped"
                )

        return supporting_results

    # ------------------------------------------------------------------
    # Full validation pipeline
    # ------------------------------------------------------------------

    async def validate(
        self,
        hypothesis: MoAHypothesis,
        sentence_pmids: Dict[int, List[Dict]],
        abstract_cache: AbstractCache,
    ) -> Tuple[List[Dict], List[Dict]]:
        """
        Run the complete two-round validation pipeline.

        Returns:
            Tuple of (supporting_results, non_supporting_results)
            Each result dict contains:
              sentence_idx, pmid, search_score, extraction_status, extracted_data,
              gold_sent_idxs (for supporting 'yes' results)
        """
        print("\n" + "=" * 70)
        print("STARTING LLM VALIDATION")
        print("=" * 70)

        # Prepare prompts
        print("\n[Step 1] Preparing prompts...")
        prompts = self.prepare_prompts(hypothesis, sentence_pmids, abstract_cache)

        if not prompts:
            print("  No prompts to validate (no abstracts found)")
            return [], []

        # Round 1
        print("\n[Step 2] Round 1 validation...")
        r1_supporting, r1_no_support = self.run_round1(prompts)

        # Round 2
        print("\n[Step 3] Round 2 validation...")
        final_supporting, additional_no = self.run_round2(r1_supporting, prompts)

        # Combine non-supporting
        all_no_support = r1_no_support + additional_no

        # Sentence index mapping for 'yes' results
        print("\n[Step 4] Mapping supporting sentences to abstract indices...")
        final_supporting = self.map_sentences_to_indices(final_supporting, abstract_cache)

        print(f"\n  Final validation results:")
        print(f"    Supporting (yes/maybe): {len(final_supporting)}")
        print(f"    Non-supporting (no): {len(all_no_support)}")

        return final_supporting, all_no_support


# ============================================================================
# HYPOTHESIS SYNTHESIS
# ============================================================================

class HypothesisSynthesizer:
    """
    Synthesize per-hop evidence into an overall hypothesis assessment.

    Takes the validation results across all hops and produces:
    - Per-hop evidence summary
    - Overall hypothesis confidence
    - Identification of weak links
    """

    def __init__(self, llm_client: Any, response_parser: Any,
                 model: str = 'gpt-oss:120b', context_window: int = 8192):
        self.llm_client = llm_client
        self.response_parser = response_parser
        self.model = model
        self.context_window = context_window

    def _build_per_hop_summary(
        self,
        hypothesis: MoAHypothesis,
        supporting_results: List[Dict],
    ) -> str:
        """Build a structured summary of evidence per hop for the synthesis prompt."""
        lines = []

        for idx, moa_sentence in enumerate(hypothesis.sentences):
            hop_results = [r for r in supporting_results if r['sentence_idx'] == idx]

            yes_results = [
                r for r in hop_results
                if r.get('extracted_data', {}).get('support') == 'yes'
            ]
            maybe_results = [
                r for r in hop_results
                if r.get('extracted_data', {}).get('support') == 'maybe'
            ]

            lines.append(f"\n--- Step {idx + 1}: \"{moa_sentence.sentence}\" ---")
            lines.append(f"  Definitive support ('yes'): {len(yes_results)} abstracts")
            lines.append(f"  Suggestive support ('maybe'): {len(maybe_results)} abstracts")

            # Include top reasoning traces (up to 3 'yes', then up to 2 'maybe')
            shown = 0
            for r in yes_results[:3]:
                extracted = r.get('extracted_data', {})
                reasoning = extracted.get('reasoning', 'No reasoning provided')
                confidence = extracted.get('confidence', 'unknown')
                lines.append(f"  [yes, {confidence}] {r['pmid']}: {reasoning}")
                shown += 1

            for r in maybe_results[:2]:
                extracted = r.get('extracted_data', {})
                reasoning = extracted.get('reasoning', 'No reasoning provided')
                confidence = extracted.get('confidence', 'unknown')
                lines.append(f"  [maybe, {confidence}] {r['pmid']}: {reasoning}")
                shown += 1

            if not hop_results:
                lines.append("  ⚠ NO supporting evidence found for this step")

        return "\n".join(lines)

    def generate_synthesis_prompt(
        self,
        hypothesis: MoAHypothesis,
        supporting_results: List[Dict],
    ) -> str:
        """Generate the synthesis prompt."""
        per_hop_summary = self._build_per_hop_summary(hypothesis, supporting_results)

        return f"""You are an expert biomedical scientist assessing the evidence for a mechanism of action (MoA) hypothesis.

=== HYPOTHESIS ===
{hypothesis.name}

Full path:
{hypothesis.path_description}

=== EVIDENCE SUMMARY PER STEP ===
{per_hop_summary}

=== INSTRUCTIONS ===
Analyze the evidence across all steps of this hypothesis and provide:

1. **per_hop_assessment**: For each step, give:
   - "step": the step number
   - "sentence": the MoA sentence
   - "evidence_strength": "strong", "moderate", "weak", or "none"
   - "summary": 1-2 sentence summary of the evidence quality

2. **weakest_links**: Identify which step(s) have the weakest evidence and explain why.

3. **overall_confidence**: "high", "medium", or "low" confidence that the full hypothesis is supported by available literature.

4. **overall_assessment**: A paragraph (3-5 sentences) synthesizing the evidence. Discuss what is well-supported, what is speculative, and what additional evidence would strengthen the hypothesis.

Output Format: Return ONLY a JSON object:
{{
  "per_hop_assessment": [
    {{
      "step": 1,
      "sentence": "...",
      "evidence_strength": "strong" | "moderate" | "weak" | "none",
      "summary": "..."
    }}
  ],
  "weakest_links": "...",
  "overall_confidence": "high" | "medium" | "low",
  "overall_assessment": "..."
}}
"""

    def synthesize(
        self,
        hypothesis: MoAHypothesis,
        supporting_results: List[Dict],
    ) -> Optional[Dict]:
        """
        Run the synthesis and return parsed results.

        Returns:
            Parsed synthesis dict or None if LLM/parsing fails
        """
        print("\n" + "=" * 70)
        print("HYPOTHESIS SYNTHESIS")
        print("=" * 70)

        prompt = self.generate_synthesis_prompt(hypothesis, supporting_results)

        print(f"  Running synthesis with model: {self.model}")
        try:
            messages = [{'role': 'user', 'content': prompt}]
            response = self.llm_client.chat(
                model=self.model,
                messages=messages,
                options={'num_ctx': self.context_window, 'temperature': 0},
            )
            response_text = response['message']['content']
        except Exception as e:
            print(f"  Error: Synthesis LLM call failed: {e}")
            return None

        parsed = self.response_parser.parse_response(response_text)
        if not parsed:
            print("  Warning: Could not parse synthesis response")
            print(f"  Raw response: {response_text[:500]}")
            return None

        # Print synthesis results
        print("\n  === SYNTHESIS RESULTS ===")
        overall_confidence = parsed.get('overall_confidence', 'unknown')
        print(f"  Overall confidence: {overall_confidence}")

        per_hop = parsed.get('per_hop_assessment', [])
        for hop in per_hop:
            step = hop.get('step', '?')
            strength = hop.get('evidence_strength', '?')
            sentence = hop.get('sentence', '')
            print(f"    Step {step} [{strength}]: {sentence}")

        weakest = parsed.get('weakest_links', '')
        print(f"\n  Weakest links: {weakest}")

        overall = parsed.get('overall_assessment', '')
        print(f"\n  Overall assessment: {overall}")

        return parsed


# ============================================================================
# MAIN PIPELINE
# ============================================================================

class MoAEvidencePipeline:
    """
    Main pipeline orchestrator for MoA hypothesis evidence finding.

    Usage:
        hypothesis = MoAHypothesis([
            MoASentence("Metformin activates AMPK", subject_info={...}, ...),
            MoASentence("AMPK inhibits mTOR"),
            MoASentence("mTOR inhibition promotes autophagy"),
        ], name="Metformin-autophagy mechanism")

        pipeline = MoAEvidencePipeline(
            config=MoAConfig(),
            llm_client=ollama_client,
            response_parser=parser,
        )

        results = asyncio.run(pipeline.run(hypothesis))
    """

    def __init__(
        self,
        config: MoAConfig,
        llm_client: Any,
        response_parser: Any,
    ):
        self.config = config
        self.llm_client = llm_client
        self.response_parser = response_parser

        # Initialize components
        print("Initializing MoA Evidence Pipeline...")
        print(f"  Embedding model: {config.ss_model_path}")
        self.model = SentenceTransformer(config.ss_model_path)
        self.encoder = MoAQueryEncoder(self.model)

        print(f"  Connecting to Milvus: {config.milvus_uri}")
        self.searcher = MoASemanticSearcher(config.milvus_uri, config.milvus_token)

        self.abstract_cache = AbstractCache()

        self.validator = MoALLMValidator(
            llm_client=llm_client,
            response_parser=response_parser,
            round1_model=config.round1_model,
            round2_model=config.round2_model,
            context_window=config.context_window,
        )

        self.synthesizer = HypothesisSynthesizer(
            llm_client=llm_client,
            response_parser=response_parser,
            model=config.round2_model,
            context_window=config.context_window,
        )

        print("  Pipeline initialized.\n")

    async def run(self, hypothesis: MoAHypothesis) -> Dict[str, Any]:
        """
        Execute the complete pipeline for one MoA hypothesis.

        Args:
            hypothesis: The MoA hypothesis to find evidence for

        Returns:
            Dict with keys:
              hypothesis: the input hypothesis name and path
              search_counts: per-sentence search statistics
              supporting_results: list of supporting validation results
              non_supporting_results: list of non-supporting validation results
              synthesis: the overall hypothesis synthesis (or None)
              statistics: pipeline statistics
        """
        print("\n" + "=" * 70)
        print("MOA EVIDENCE PIPELINE")
        print(f"Hypothesis: {hypothesis.name}")
        print(f"Path: {hypothesis.path_description}")
        print(f"Hops: {len(hypothesis)}")
        print("=" * 70)

        # Step 1: Encode MoA sentences
        print("\nStep 1: Encoding MoA sentences...")
        query_vectors = self.encoder.encode_hypothesis(hypothesis)

        # Step 2: Semantic search
        print("\nStep 2: Semantic search...")
        print(f"  Threshold: {self.config.ss_threshold}")
        print(f"  Top-k abstracts: {self.config.top_k_abstracts}")
        sentence_pmids, search_counts = self.searcher.search(
            query_vectors,
            threshold=self.config.ss_threshold,
            top_k_abstracts=self.config.top_k_abstracts,
            num_collections=self.config.num_collections,
        )

        # Save intermediate search results
        self._save_json(sentence_pmids, 'semantic_search_results.json')
        self._save_json(search_counts, 'search_counts.json')

        # Step 3: Fetch and cache abstracts
        print("\nStep 3: Fetching abstracts...")
        await self.abstract_cache.fetch_and_cache(sentence_pmids)

        # Save abstract cache
        self._save_json(self.abstract_cache.cache, 'cached_abstracts.json')

        # Step 4: LLM validation
        print("\nStep 4: LLM validation...")
        supporting_results, non_supporting_results = await self.validator.validate(
            hypothesis, sentence_pmids, self.abstract_cache
        )

        # Save validation results
        self._save_json(supporting_results, 'supporting_results.json')
        self._save_json(non_supporting_results, 'non_supporting_results.json')

        # Step 5: Hypothesis synthesis
        print("\nStep 5: Hypothesis synthesis...")
        synthesis = self.synthesizer.synthesize(hypothesis, supporting_results)

        if synthesis:
            self._save_json(synthesis, 'hypothesis_synthesis.json')

        # Calculate statistics
        statistics = self._calculate_statistics(
            hypothesis, sentence_pmids, supporting_results, non_supporting_results
        )
        self._save_json(statistics, 'pipeline_statistics.json')

        print("\n" + "=" * 70)
        print("PIPELINE COMPLETE")
        print("=" * 70)
        self._print_final_summary(statistics)

        return {
            'hypothesis': {
                'name': hypothesis.name,
                'path': hypothesis.path_description,
                'sentences': [s.sentence for s in hypothesis.sentences],
            },
            'search_counts': search_counts,
            'supporting_results': supporting_results,
            'non_supporting_results': non_supporting_results,
            'synthesis': synthesis,
            'statistics': statistics,
        }

    # ------------------------------------------------------------------
    # Statistics and output
    # ------------------------------------------------------------------

    def _calculate_statistics(
        self,
        hypothesis: MoAHypothesis,
        sentence_pmids: Dict[int, List[Dict]],
        supporting_results: List[Dict],
        non_supporting_results: List[Dict],
    ) -> Dict:
        """Calculate pipeline statistics."""
        stats = {
            'hypothesis_name': hypothesis.name,
            'num_hops': len(hypothesis),
            'per_hop': [],
            'totals': {},
        }

        total_search = 0
        total_abstracts_fetched = len(self.abstract_cache.cache)
        total_supporting = 0
        total_non_supporting = 0

        for idx in range(len(hypothesis)):
            sentence = hypothesis.sentences[idx].sentence
            search_count = len(sentence_pmids.get(idx, []))
            hop_supporting = [r for r in supporting_results if r['sentence_idx'] == idx]
            hop_non_supporting = [r for r in non_supporting_results if r['sentence_idx'] == idx]

            yes_count = sum(
                1 for r in hop_supporting
                if r.get('extracted_data', {}).get('support') == 'yes'
            )
            maybe_count = sum(
                1 for r in hop_supporting
                if r.get('extracted_data', {}).get('support') == 'maybe'
            )

            stats['per_hop'].append({
                'sentence_idx': idx,
                'sentence': sentence,
                'search_results': search_count,
                'llm_yes': yes_count,
                'llm_maybe': maybe_count,
                'llm_no': len(hop_non_supporting),
            })

            total_search += search_count
            total_supporting += len(hop_supporting)
            total_non_supporting += len(hop_non_supporting)

        stats['totals'] = {
            'total_search_results': total_search,
            'total_abstracts_fetched': total_abstracts_fetched,
            'total_supporting': total_supporting,
            'total_non_supporting': total_non_supporting,
        }

        return stats

    def _print_final_summary(self, statistics: Dict):
        """Print a human-readable final summary."""
        print(f"\n  Hypothesis: {statistics['hypothesis_name']}")
        print(f"  Hops: {statistics['num_hops']}")
        print(f"\n  Per-hop summary:")
        for hop in statistics['per_hop']:
            print(
                f"    Step {hop['sentence_idx']}: "
                f"search={hop['search_results']}, "
                f"yes={hop['llm_yes']}, "
                f"maybe={hop['llm_maybe']}, "
                f"no={hop['llm_no']}  "
                f"| \"{hop['sentence']}\""
            )

        totals = statistics['totals']
        print(f"\n  Totals:")
        print(f"    Search results: {totals['total_search_results']}")
        print(f"    Abstracts fetched: {totals['total_abstracts_fetched']}")
        print(f"    Supporting (yes/maybe): {totals['total_supporting']}")
        print(f"    Non-supporting (no): {totals['total_non_supporting']}")

    def _save_json(self, data: Any, filename: str):
        """Save data to JSON in the output directory."""
        import os
        os.makedirs(self.config.output_dir, exist_ok=True)
        filepath = os.path.join(self.config.output_dir, filename)

        # Handle non-serializable types
        def default_serializer(obj):
            if isinstance(obj, np.floating):
                return float(obj)
            if isinstance(obj, np.integer):
                return int(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            raise TypeError(f"Object of type {type(obj)} is not JSON serializable")

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2, default=default_serializer)
        print(f"  Saved: {filepath}")


# ============================================================================
# USAGE EXAMPLE
# ============================================================================

if __name__ == "__main__":
    from ollama import Client
    from response_parser import SimpleLLMResponseParser

    # Initialize clients
    llm_client = Client()
    response_parser = SimpleLLMResponseParser()

    # Define the hypothesis
        hypothesis = MoAHypothesis(
        sentences=[
            MoASentence(
                "Metformin activates AMPK",
                subject_info={
                    "name": "Metformin",
                    "category": "small molecule",
                    "description": "An oral antidiabetic drug in the biguanide class",
                },
                object_info={
                    "name": "AMPK",
                    "category": "protein",
                    "description": "AMP-activated protein kinase, a cellular energy sensor",
                },
                predicate_info={
                    "name": "activates",
                    "description": "Increases the activity or expression of the target",
                },
            ),
            MoASentence(
                "AMPK inhibits mTOR signaling",
                subject_info={"name": "AMPK", "category": "protein"},
                object_info={"name": "mTOR", "category": "protein"},
            ),
            MoASentence(
                "mTOR inhibition promotes autophagy",
            ),
        ],
        name="Metformin-AMPK-mTOR-autophagy mechanism",
    )

    # Configure pipeline
    config = MoAConfig(
        ss_model_path='sentence-transformers/all-MiniLM-L6-v2',
        ss_threshold=0.5,
        top_k_abstracts=300,
        round1_model='gpt-oss:20b',
        round2_model='gpt-oss:120b',
        output_dir='result/moa_evidence/metformin_autophagy',
    )

    # Create and run pipeline
    pipeline = MoAEvidencePipeline(
        config=config,
        llm_client=llm_client,
        response_parser=response_parser,
    )

    results = asyncio.run(pipeline.run(hypothesis))

    # Access results
    print(f"\nSynthesis confidence: {results['synthesis']['overall_confidence']}")
    print(f"Overall: {results['synthesis']['overall_assessment']}")
