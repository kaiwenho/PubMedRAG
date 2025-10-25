"""
PubMed Extension Pipeline
A pipeline for linking edges without support to PubMed publications through semantic search and LLM validation.

Pipeline Steps:
1. Load parameters for specific predicate type
2. Generate query embeddings (concat or AI-generated paraphrases) for SEMANTIC SEARCH
3. Perform semantic search for relevant PMIDs
4. Retrieve and classify abstracts (generates NEW query embeddings with classification model)
5. LLM validation of edge-PMID support with two-round validation and sentence mapping
"""

import json
import time
import pandas as pd
from typing import List, Dict, Any, Optional, Tuple
from sentence_transformers import SentenceTransformer
from pymilvus import MilvusClient, connections, utility
from biomed_eval import load_search_parameters
import numpy as np
from nltk.tokenize import sent_tokenize
import asyncio


# ============================================================================
# CONFIGURATION AND SETUP
# ============================================================================

class PipelineConfig:
    """Configuration class for pipeline parameters"""

    def __init__(self, predicate: str, results_path: str = 'evaluation_results.json'):
        self.predicate = predicate
        self.params = load_search_parameters(results_path, predicate=predicate)

        # Semantic search parameters
        self.ss_model_path = self.params['sentence_search']['model_path']
        self.ss_threshold = self.params['sentence_search']['threshold']
        self.ss_representation = self.params['sentence_search']['representation']

        # Abstract classification parameters
        self.ac_model_path = self.params['abstract_classification']['model_path']
        self.ac_threshold = self.params['abstract_classification']['threshold']
        self.ac_representation = self.params['abstract_classification']['representation']
        self.ac_aggregation = self.params['abstract_classification']['aggregation']

        # Milvus configuration
        self.milvus_uri = "http://localhost:19530"
        self.milvus_token = "root:Milvus"

    def get_output_path(self, step: str) -> str:
        """Generate standardized output paths"""
        predicate = self.predicate.removeprefix('biolink:')
        paths = {
            'semantic_search': f"result/semantic_search/{predicate}_semantic_search.json",
            'search_counts': f"result/semantic_search/{predicate}_semantic_search_counts.json",
            'abstract_classification': f"result/classification/{predicate}_classified_abstracts.json",
            'cached_abstracts': f"result/classification/{predicate}_cached_abstracts.json",
            'llm_validation': f"result/validation/{predicate}_llm_validated.json",
            'validation_results': f"result/validation/{predicate}_validation_results.parquet"
        }
        return paths.get(step, f"result/{predicate}_{step}.json")


# ============================================================================
# DATA LOADING
# ============================================================================

class DataLoader:
    """Load reference dictionaries and edge data"""

    def __init__(self):
        self.predicate_dict = self._load_json('dict/biolink_pred_info_dictionary.json')
        self.node_dict = self._load_json('dict/rtx-kg2_id_info_dictionary.json')

    @staticmethod
    def _load_json(filepath: str) -> Dict:
        """Load JSON file"""
        with open(filepath, 'r') as f:
            return json.load(f)

    def load_edges(self, predicate: str) -> pd.DataFrame:
        """Load edges for specific predicate"""
        # Remove 'biolink:' prefix if present
        pred_name = predicate.replace('biolink:', '')
        return pd.read_parquet(f'edges/{pred_name}_nopub.parquet')


# ============================================================================
# QUERY GENERATION
# ============================================================================

class QueryGenerator:
    """Generate query embeddings for semantic search"""

    def __init__(self, model: SentenceTransformer, node_dict: Dict, predicate_dict: Dict):
        self.model = model
        self.node_dict = node_dict
        self.predicate_dict = predicate_dict

    def generate_queries(self, edges: pd.DataFrame, representation: str,
                        max_edges: int = 50, llm_client = None) -> List[Dict]:
        """Generate query vectors for edges"""
        if representation == 'ai':
            return self._generate_ai_queries(edges, max_edges, llm_client)
        elif representation == 'concat':
            return self._generate_concat_queries(edges, max_edges)
        else:
            raise ValueError(f"Unknown representation: {representation}")

    def _generate_concat_queries(self, edges: pd.DataFrame, max_edges: int) -> List[Dict]:
        """Generate concatenated subject-predicate-object queries"""
        query_vectors = []
        processed_count = 0

        for index, row in edges.iterrows():
            processed_count += 1

            # Stop if we've reached max_edges OR processed all edges
            if len(query_vectors) >= max_edges:
                break

            subj_info = self.node_dict.get(row['subject'])
            obj_info = self.node_dict.get(row['object'])

            if not subj_info or not obj_info:
                continue

            subj_name = subj_info.get('name')
            obj_name = obj_info.get('name')

            if not subj_name or not obj_name:
                continue

            pred = row['predicate'].replace("biolink:", "")
            concat_sentence = f"{subj_name} {pred} {obj_name}"

            edge_emb = self.model.encode([concat_sentence], convert_to_numpy=True)[0]
            query_vectors.append({"edge_id": index, "edge_emb": edge_emb})

            # Progress indicator for large datasets
            if processed_count % 100 == 0:
                print(f"  Processed {processed_count} edges, generated {len(query_vectors)} query vectors")

        print(f"  Final: Processed {processed_count} edges, generated {len(query_vectors)} query vectors")
        return query_vectors

    def _generate_ai_queries(self, edges: pd.DataFrame, max_edges: int,
                           llm_client) -> List[Dict]:
        """Generate AI-paraphrased queries"""
        from response_parser import SimpleLLMResponseParser

        parser = SimpleLLMResponseParser()
        query_vectors = []
        processed_count = 0

        for index, row in edges.iterrows():
            processed_count += 1

            # Stop if we've reached max_edges
            if len(query_vectors) >= max_edges:
                break

            prompt = self._create_prompt(row)
            if not prompt:
                continue

            try:
                messages = [{'role': 'user', 'content': prompt}]
                response = llm_client.chat(
                    model='gpt-oss:20b',
                    messages=messages,
                    options={'num_ctx': 8192, "temperature": 0}
                )
                response_text = response['message']['content']
                response_json = parser.parse_response(response_text)

                if response_json and 'sentences' in response_json:
                    sentences = response_json['sentences']
                    embeddings = self.model.encode(sentences, convert_to_numpy=True)
                    edge_emb = embeddings.mean(axis=0)
                    query_vectors.append({"edge_id": index, "edge_emb": edge_emb})

            except Exception as e:
                print(f"Error processing edge {index}: {e}")
                continue

            # Progress indicator
            if processed_count % 10 == 0:
                print(f"  Processed {processed_count} edges, generated {len(query_vectors)} query vectors")

        print(f"  Final: Processed {processed_count} edges, generated {len(query_vectors)} query vectors")
        return query_vectors

    def _create_prompt(self, row: pd.Series) -> Optional[str]:
        """Create prompt for LLM to generate sentence variations"""
        subj_info = self.node_dict.get(row['subject'])
        obj_info = self.node_dict.get(row['object'])
        pred_info = self.predicate_dict.get(row['predicate'])

        if not all([subj_info, obj_info, pred_info]):
            return None

        return f"""Convert the following biochemical edge into natural language sentences that express its meaning. Generate 3 different sentence variations that convey the same relationship using different phrasings or perspectives.

Input Format:
Edge: {subj_info['name']} --{row['predicate']}-> {obj_info['name']}
Subject: {subj_info}
Object: {obj_info}
Predicate: {pred_info}

Instructions:
- Generate 3 distinct, grammatically correct sentences
- Each sentence should accurately reflect the biochemical relationship
- Use the entity descriptions and predicate definition to ensure precision
- Vary the sentence structure and vocabulary while maintaining scientific accuracy

Output Format: Return your response as a JSON object:
{{
  "sentences": [
    "First sentence variation",
    "Second sentence variation",
    "Third sentence variation"
  ]
}}"""


# ============================================================================
# SEMANTIC SEARCH
# ============================================================================

class SemanticSearcher:
    """Perform semantic search in Milvus collections"""

    def __init__(self, uri: str, token: str):
        self.uri = uri
        self.token = token
        self.client = None
        self._setup_connection()

    def _setup_connection(self):
        """Establish Milvus connection"""
        connections.connect(
            alias="default",
            uri=self.uri,
            token=self.token,
        )
        self._wait_for_node()
        self.client = MilvusClient(uri=self.uri, token=self.token)

    def _wait_for_node(self, resource_group: str = "__default_resource_group",
                      interval: int = 5):
        """Wait for Milvus node to be available"""
        while True:
            info = utility.describe_resource_group(name=resource_group)
            num_available = info.num_available_node
            print(f"Node availability: {num_available}")
            if num_available >= 1:
                print("Node is available—continuing execution.")
                return
            print(f"No nodes available, retrying in {interval}s…")
            time.sleep(interval)

    def search_collections(self, query_vectors: List[Dict], threshold: float,
                          num_collections: int = 10) -> Tuple[Dict, Dict]:
        """
        Search across multiple Milvus collections.

        Args:
            query_vectors: List of dicts with 'edge_id' and 'edge_emb'
            threshold: Similarity threshold (radius parameter)
            num_collections: Number of collections to search

        Returns:
            Tuple of (edge_contexts, edge_context_counts)
        """
        edge_contexts = {}
        edge_context_counts = {}
        start = time.time()

        for i in range(num_collections):
            collection_name = f"pubmed_sentence_{i:02d}"
            print(f"Loading collection: {collection_name}")
            self.client.load_collection(collection_name=collection_name)

            for query in query_vectors:
                edge_id = query['edge_id']
                edge_emb = query['edge_emb']

                results = self.client.search(
                    collection_name=collection_name,
                    data=[edge_emb],
                    limit=200,
                    search_params={
                        "params": {
                            "radius": threshold,
                            "range_filter": 1.0
                        }
                    },
                    output_fields=["pmid"],
                )

                # Aggregate results
                if edge_id not in edge_contexts:
                    edge_contexts[edge_id] = results[0]
                    edge_context_counts[edge_id] = []
                else:
                    edge_contexts[edge_id].extend(results[0])

                edge_context_counts[edge_id].append({
                    collection_name: len(results[0])
                })

            self.client.release_collection(collection_name=collection_name)
            print(f"Finished and unloaded: {collection_name}")

        elapsed = time.time() - start
        print(f"Semantic search execution time: {elapsed:.2f}s")

        return edge_contexts, edge_context_counts


# ============================================================================
# ABSTRACT CLASSIFICATION
# ============================================================================

class AbstractClassifier:
    """Classify abstracts for edge support"""

    def __init__(self, model: SentenceTransformer, threshold: float,
                 aggregation: str, representation: str, query_generator: 'QueryGenerator'):
        self.model = model
        self.threshold = threshold
        self.aggregation = aggregation
        self.representation = representation
        self.query_generator = query_generator

    async def classify_abstracts_with_retrieval(self, edges: pd.DataFrame,
                                               edge_contexts: Dict,
                                               batch_size: int = 100,
                                               llm_client = None) -> Tuple[Dict, Dict]:
        """
        Classify abstracts AND cache them for later validation.

        Returns:
            Tuple of (supporting_pmids, abstracts_dict)
        """
        from pubmed_client import get_publication_info

        supporting_pmids = {}
        abstracts_dict = {}  # Cache abstracts here!

        # Generate query vectors using THIS model (classification model)
        query_vectors = self.query_generator.generate_queries(
            edges, self.representation, max_edges=len(edges), llm_client=llm_client
        )
        query_dict = {qv["edge_id"]: qv["edge_emb"] for qv in query_vectors}

        # Step 1: Collect all unique PMIDs across all edges
        print("\n  Collecting unique PMIDs...")
        all_unique_pmids = set()
        edge_pmid_mapping = {}  # Track which PMIDs belong to which edges

        for edge_id, contexts in edge_contexts.items():
            pmids = list(set([ctx['entity']['pmid'] for ctx in contexts]))
            # Add PMID: prefix if not present

            pmids = [f"PMID:{pmid}" if not str(pmid).startswith('PMID:') else pmid
                     for pmid in pmids]
            edge_pmid_mapping[edge_id] = pmids
            all_unique_pmids.update(pmids)

        print(f"  Total unique PMIDs to retrieve: {len(all_unique_pmids)}")

        # Step 2: Retrieve all abstracts in batches (only once!)
        print("\n  Retrieving abstracts from PubMed...")
        all_unique_pmids = list(all_unique_pmids)

        for i in range(0, len(all_unique_pmids), batch_size):
            batch_pmids = all_unique_pmids[i:i + batch_size]

            print(f"  Retrieving batch {i//batch_size + 1}/{(len(all_unique_pmids)-1)//batch_size + 1}...")

            abstracts_info = await get_publication_info(
                batch_pmids, 'placeholder'
            )

            if abstracts_info['_meta']['n_results'] > 0:
                abstracts = abstracts_info['results']

                for pmid in batch_pmids:
                    abstract = abstracts.get(pmid, {}).get('abstract')
                    if abstract:
                        sentences = sent_tokenize(abstract)
                        abstracts_dict[pmid] = {
                            'abstract': abstract,
                            'sentences': sentences
                        }

        print(f"  Retrieved {len(abstracts_dict)} abstracts")

        # Step 3: Classify abstracts for each edge
        print("\n  Classifying abstracts for each edge...")
        for edge_id, pmids in edge_pmid_mapping.items():
            if edge_id not in query_dict:
                continue

            edge_query_emb = query_dict[edge_id]

            for pmid in pmids:
                # Check if we have the abstract
                if pmid not in abstracts_dict:
                    continue

                sentences = abstracts_dict[pmid]['sentences']

                # Compute sentence embeddings
                sentence_embeddings = self.model.encode(
                    sentences, convert_to_numpy=True
                )

                # Calculate similarities
                similarities = np.dot(sentence_embeddings, edge_query_emb)

                # Apply aggregation and check threshold
                if self._is_supporting_with_similarities(similarities):
                    if edge_id not in supporting_pmids:
                        supporting_pmids[edge_id] = []
                    supporting_pmids[edge_id].append(pmid)

        print(f"\n  Classification complete:")
        print(f"  Edges with supporting PMIDs: {len(supporting_pmids)}")
        print(f"  Total supporting relationships: {sum(len(pmids) for pmids in supporting_pmids.values())}")

        return supporting_pmids, abstracts_dict

    def _is_supporting_with_similarities(self, similarities: np.ndarray) -> bool:
        """
        Determine if abstract supports edge based on sentence similarities.

        Args:
            similarities: Array of cosine similarities for each sentence

        Returns:
            True if abstract supports the edge, False otherwise
        """
        if len(similarities) == 0:
            return False

        # Sort similarities in descending order
        sorted_sims = np.sort(similarities)[::-1]

        # Apply aggregation strategy
        if self.aggregation == 'max':
            score = sorted_sims[0]
        elif self.aggregation == 'top2_mean':
            if len(sorted_sims) >= 2:
                score = sorted_sims[:2].mean()
            else:
                score = sorted_sims.mean()
        elif self.aggregation == 'top3_mean':
            if len(sorted_sims) >= 3:
                score = sorted_sims[:3].mean()
            else:
                score = sorted_sims.mean()
        elif self.aggregation == 'any':
            # Check if ANY sentence exceeds threshold
            return (similarities >= self.threshold).any()
        elif self.aggregation == 'mean':
            score = similarities.mean()
        else:
            # Default to max if unknown aggregation
            print(f"Warning: Unknown aggregation '{self.aggregation}', using 'max'")
            score = sorted_sims[0]

        return score >= self.threshold


# ============================================================================
# SOPHISTICATED LLM VALIDATOR (ADAPTED FOR PIPELINE)
# ============================================================================

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
        context_window: int = 8192
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
        """
        self.llm_client = llm_client
        self.node_dict = node_dict
        self.predicate_dict = predicate_dict
        self.response_parser = response_parser
        self.round1_model = round1_model
        self.round2_model = round2_model
        self.context_window = context_window

        # Storage for intermediate results
        self.prompts = []
        self.round1_responses = []
        self.round2_responses = []
        self.round1_results = []
        self.round2_results = []
        self.merged_results = None
        self.validation_df = None

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
        edges: pd.DataFrame,
        supporting_pmids: Dict[int, List[str]],
        abstracts_dict: Dict[str, Dict]
    ) -> List[Dict]:
        """
        Prepare validation prompts for edges with supporting PMIDs.

        Args:
            edges: DataFrame of edges
            supporting_pmids: Dict mapping edge_id to list of PMIDs
            abstracts_dict: Dict mapping PMID to abstract data

        Returns:
            List of prompt dictionaries with keys: edge_index, pmid, prompt
        """
        prompts = []

        for edge_id, pmids in supporting_pmids.items():
            # Get edge data
            row = edges.loc[edge_id]

            subj_info = self.node_dict.get(row['subject'])
            obj_info = self.node_dict.get(row['object'])
            pred_info = self.predicate_dict.get(row['predicate'])

            if not all([subj_info, obj_info, pred_info]):
                continue

            for pmid in pmids:
                # Get abstract from cache
                abstract_data = abstracts_dict.get(pmid)
                if not abstract_data:
                    continue

                abstract = abstract_data.get('abstract')
                if not abstract:
                    continue

                prompt = self.generate_prompt(
                    subj_info, obj_info, pred_info, row['predicate'], abstract
                )

                prompts.append({
                    'edge_index': edge_id,
                    'pmid': pmid,
                    'prompt': prompt
                })

        self.prompts = prompts
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
            Tuple of (responses, parsed_results)
        """
        responses = []
        parsed_results = []
        count = 0

        for prompt_info in prompts:
            edge_index = prompt_info['edge_index']
            pmid = prompt_info['pmid']
            prompt = prompt_info['prompt']

            messages = [{
                'role': 'user',
                'content': prompt,
            }]

            response = self.llm_client.chat(
                model=model,
                messages=messages,
                options={'num_ctx': self.context_window},
            )

            response_text = response['message']['content']
            responses.append({
                'edge_index': edge_index,
                'pmid': pmid,
                'response': response_text
            })

            # Parse the response
            try:
                parsed_data = self.response_parser.parse_response(response_text)
                parsed_results.append({
                    'edge_index': edge_index,
                    'pmid': pmid,
                    'extraction_status': 'success',
                    'extracted_data': parsed_data
                })
            except Exception as e:
                parsed_results.append({
                    'edge_index': edge_index,
                    'pmid': pmid,
                    'extraction_status': 'failed',
                    'error': str(e)
                })

            count += 1
            if count % 50 == 0:
                print(f"  Processed {count}/{len(prompts)} prompts")

        return responses, parsed_results

    async def run_round1(self) -> List[Dict]:
        """Run Round 1 validation with smaller model and filter out 'no' results."""
        print(f"Starting Round 1 with model: {self.round1_model}")
        self.round1_responses, self.round1_results = self.run_validation_round(
            self.prompts, self.round1_model
        )

        # Print summary
        self._print_validation_summary(self.round1_results, "Round 1")

        # Filter out 'no' results - we only care about supporting PMIDs
        filtered_results = []
        no_support_results = []

        for result in self.round1_results:
            if result.get('extraction_status') == 'success':
                support = result.get('extracted_data', {}).get('support?')
                if support in ['yes', 'maybe']:
                    filtered_results.append(result)
                elif support == 'no':
                    no_support_results.append(result)

        print(f"\nFiltering results:")
        print(f"  Supporting ('yes'/'maybe'): {len(filtered_results)}")
        print(f"  Non-supporting ('no'): {len(no_support_results)} (filtered out)")

        # Store filtered results
        self.round1_results = filtered_results
        self.no_support_results = no_support_results

        return self.round1_results

    def _print_validation_summary(self, results: List[Dict], round_name: str):
        """Print summary statistics for validation results."""
        from collections import Counter

        print(f"\n{round_name} Summary:")
        print(f"  Total: {len(results)}")

        status_counts = Counter(r.get('extraction_status') for r in results)
        print(f"  Successful: {status_counts.get('success', 0)}")
        print(f"  Failed: {status_counts.get('failed', 0)}")

        support_counts = Counter(
            r.get('extracted_data', {}).get('support?')
            for r in results
            if r.get('extraction_status') == 'success'
        )
        if support_counts:
            print(f"  Support distribution:")
            for support_type, count in support_counts.most_common():
                print(f"    {support_type}: {count}")

    async def run_round2(self) -> List[Dict]:
        """Run Round 2 validation on 'yes' and 'maybe' cases with larger model for quality assurance."""
        # Filter for 'yes' and 'maybe' cases from Round 1
        recheck_prompts = []

        for result in self.round1_results:
            if result.get('extraction_status') == 'success':
                support = result.get('extracted_data', {}).get('support?')
                if support in ['yes', 'maybe']:
                    # Find corresponding prompt
                    edge_index = result['edge_index']
                    pmid = result['pmid']

                    for prompt_info in self.prompts:
                        if (prompt_info['edge_index'] == edge_index and
                            prompt_info['pmid'] == pmid):
                            recheck_prompts.append(prompt_info)
                            break

        if not recheck_prompts:
            print("No 'yes' or 'maybe' cases to validate in Round 2")
            self.round2_responses = []
            self.round2_results = []
            return []

        print(f"Starting Round 2 with model: {self.round2_model}")
        print(f"Validating {len(recheck_prompts)} 'yes' and 'maybe' cases for quality assurance")

        self.round2_responses, self.round2_results = self.run_validation_round(
            recheck_prompts, self.round2_model
        )

        # Print summary
        self._print_validation_summary(self.round2_results, "Round 2")

        return self.round2_results

    def merge_results(self) -> Dict:
        """
        Merge results from both rounds.
        Round 2 results (from larger model) override Round 1 for quality assurance.
        Filter out any new 'no' results from Round 2.
        """
        print("\nMerging results from both rounds...")

        # Create lookup for Round 2 results
        round2_lookup = {}
        for result in self.round2_results:
            if result.get('extraction_status') == 'success':
                key = (result['edge_index'], result['pmid'])
                round2_lookup[key] = result

        # Merge: use Round 2 if available (it's the authoritative answer from larger model)
        merged = []
        additional_no_support = []

        for result in self.round1_results:
            key = (result['edge_index'], result['pmid'])
            if key in round2_lookup:
                round2_result = round2_lookup[key]
                # Check if Round 2 changed it to 'no'
                if (round2_result.get('extraction_status') == 'success' and
                    round2_result.get('extracted_data', {}).get('support?') == 'no'):
                    additional_no_support.append(round2_result)
                else:
                    merged.append(round2_result)
            else:
                # No Round 2 result, use Round 1
                merged.append(result)

        # Add new 'no' results to the no_support_results list
        if additional_no_support:
            print(f"  Round 2 identified {len(additional_no_support)} additional 'no' cases")
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

        # Print statistics
        print(f"\nMerged results:")
        print(f"  Supporting results ('yes'/'maybe'): {len(merged)}")
        print(f"  Non-supporting results ('no'): {len(self.no_support_results)}")

        if not self.validation_df.empty:
            support_counts = self.validation_df['abstract_support?'].value_counts()
            print("\nFinal supporting validation statistics:")
            for support_type, count in support_counts.items():
                print(f"  {support_type}: {count}")

        return self.merged_results

    def add_abstract_sentences_to_validation(
        self,
        abstracts_dict: Dict[str, Dict]
    ):
        """Add abstract sentences to validation DataFrame."""
        if self.validation_df is None:
            raise ValueError("No validation results. Run merge_results() first.")

        print("\nAdding abstract sentences to validation data...")

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

        print("\n=== Mapping LLM sentences to abstract indices ===")

        indices_column = []
        success_flags = []

        for i, row in self.validation_df.iterrows():
            indices, success = process_with_llm_fallback(
                row, abstracts_dict, self.llm_client
            )
            indices_column.append(indices)
            success_flags.append(success)

        self.validation_df['gold_sent_idxs'] = indices_column
        self.validation_df['mapping_success'] = success_flags

        # Print statistics
        success_counter = Counter(success_flags)
        print(f"\nMapping Statistics:")
        print(f"Successful: {success_counter[True]}")
        print(f"Failed: {success_counter[False]}")
        if len(success_flags) > 0:
            print(f"Success rate: {success_counter[True] / len(success_flags) * 100:.1f}%")

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
            print("No failed mappings to fix!")
            return self.validation_df

        print(f"\n=== Fixing {len(failed_indices)} failed mappings with {self.round2_model} ===")

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
        supporting_pmids: Dict[int, List[str]],
        abstracts_dict: Dict[str, Dict]
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Complete validation pipeline for edges.

        Args:
            edges: DataFrame of edges
            supporting_pmids: Dict mapping edge_id to list of PMIDs
            abstracts_dict: Dict mapping PMID to abstract data

        Returns:
            Tuple of (supporting_results_df, non_supporting_results_df)
        """
        print("\n" + "="*70)
        print("STARTING LLM VALIDATION PIPELINE")
        print("="*70)

        # Step 1: Prepare prompts
        print("\n[Step 1] Preparing validation prompts...")
        self.prepare_prompts(edges, supporting_pmids, abstracts_dict)
        print(f"Generated {len(self.prompts)} prompts")

        # Step 2: Run Round 1
        print("\n[Step 2] Running Round 1 validation...")
        await self.run_round1()

        # Step 3: Run Round 2 (validate 'yes' and 'maybe' for quality assurance)
        print("\n[Step 3] Running Round 2 validation...")
        await self.run_round2()

        # Step 4: Merge results
        print("\n[Step 4] Merging results from both rounds...")
        self.merge_results()

        # Step 5: Add abstract sentences to supporting results only
        print("\n[Step 5] Adding abstract sentences to supporting results...")
        self.add_abstract_sentences_to_validation(abstracts_dict)

        # Step 6: Map sentences to indices (only for supporting results)
        print("\n[Step 6] Mapping LLM sentences to abstract indices...")
        self.map_sentences_to_indices(abstracts_dict)

        # Step 7: Fix failed mappings
        print("\n[Step 7] Fixing failed mappings...")
        self.fix_failed_mappings(abstracts_dict)

        print("\n" + "="*70)
        print("VALIDATION PIPELINE COMPLETE")
        print("="*70)

        return self.get_final_results(), self.no_support_df


# ============================================================================
# MAIN PIPELINE
# ============================================================================

class PubExtensionPipeline:
    """Main pipeline orchestrator"""

    def __init__(self, predicate: str, llm_client=None, response_parser=None, edges = None):
        self.config = PipelineConfig(predicate)
        self.data_loader = DataLoader()
        self.llm_client = llm_client
        self.response_parser = response_parser

        # Initialize SEMANTIC SEARCH components
        self.ss_model = SentenceTransformer(self.config.ss_model_path)
        self.ss_query_generator = QueryGenerator(
            self.ss_model,
            self.data_loader.node_dict,
            self.data_loader.predicate_dict
        )

        # Initialize ABSTRACT CLASSIFICATION components
        self.ac_model = SentenceTransformer(self.config.ac_model_path)
        self.ac_query_generator = QueryGenerator(
            self.ac_model,
            self.data_loader.node_dict,
            self.data_loader.predicate_dict
        )

        # Initialize semantic searcher
        self.searcher = SemanticSearcher(
            self.config.milvus_uri,
            self.config.milvus_token
        )

        # Load edges
        if edges is not None:
            self.edges = edges
        else:
            self.edges = self.data_loader.load_edges(predicate)
        print(f"Loaded {len(self.edges)} edges for predicate: {predicate}")

    def _calculate_pipeline_statistics(
        self,
        edge_contexts: Dict,
        supporting_pmids: Dict,
        supporting_results: pd.DataFrame,
        non_supporting_results: pd.DataFrame
    ) -> Dict:
        """
        Calculate success rates at each pipeline stage.

        Returns:
            Dictionary with per-run statistics and per-edge details
        """
        statistics = {
            'per_run': {},
            'per_edge': []
        }

        # Stage 1: Semantic Search → Abstract Classification
        total_ss_pmids = 0
        total_ac_pmids = 0

        for edge_id in edge_contexts.keys():
            ss_pmids = set([ctx['entity']['pmid'] for ctx in edge_contexts.get(edge_id, [])])
            ac_pmids = set(supporting_pmids.get(edge_id, []))

            # Normalize PMIDs: convert to string and remove PMID: prefix
            ss_pmids_clean = {str(pmid).replace('PMID:', '') for pmid in ss_pmids}
            ac_pmids_clean = {str(pmid).replace('PMID:', '') for pmid in ac_pmids}

            passed_pmids = ss_pmids_clean & ac_pmids_clean

            total_ss_pmids += len(ss_pmids_clean)
            total_ac_pmids += len(ac_pmids_clean)

            edge_stat = {
                'edge_id': edge_id,
                'semantic_search_pmids': len(ss_pmids_clean),
                'classification_pmids': len(ac_pmids_clean),
                'ss_to_ac_passed': len(passed_pmids),
                'ss_to_ac_rate': len(passed_pmids) / len(ss_pmids_clean) if ss_pmids_clean else 0.0
            }

            # Stage 2: Abstract Classification → LLM Validation
            # Check if validation was actually run by checking if we have results
            if not supporting_results.empty or not non_supporting_results.empty:
                edge_validation = supporting_results[supporting_results['edge_index'] == edge_id]
                edge_yes = edge_validation[edge_validation['abstract_support?'] == 'yes']

                validation_pmids = set(edge_validation['pmid'].tolist())
                yes_pmids = set(edge_yes['pmid'].tolist())

                # Normalize PMIDs: convert to string and remove PMID: prefix
                validation_pmids_clean = {str(pmid).replace('PMID:', '') for pmid in validation_pmids}
                yes_pmids_clean = {str(pmid).replace('PMID:', '') for pmid in yes_pmids}

                edge_stat.update({
                    'validation_pmids': len(validation_pmids_clean),
                    'validation_yes_pmids': len(yes_pmids_clean),
                    'ac_to_llm_rate': len(yes_pmids_clean) / len(ac_pmids_clean) if ac_pmids_clean else 0.0
                })

            statistics['per_edge'].append(edge_stat)

        # Per-run aggregates
        statistics['per_run']['total_edges'] = len(edge_contexts)
        statistics['per_run']['semantic_search_total_pmids'] = total_ss_pmids
        statistics['per_run']['classification_total_pmids'] = total_ac_pmids
        statistics['per_run']['ss_to_ac_success_rate'] = (
            total_ac_pmids / total_ss_pmids if total_ss_pmids > 0 else 0.0
        )

        # Check if validation was run by checking if we have results
        if not supporting_results.empty or not non_supporting_results.empty:
            total_validation_pmids = len(supporting_results) + len(non_supporting_results)
            total_yes_pmids = len(supporting_results[supporting_results['abstract_support?'] == 'yes'])

            statistics['per_run']['validation_total_pmids'] = total_validation_pmids
            statistics['per_run']['validation_yes_pmids'] = total_yes_pmids
            statistics['per_run']['ac_to_llm_success_rate'] = (
                total_yes_pmids / total_ac_pmids if total_ac_pmids > 0 else 0.0
            )

        return statistics

    async def run(self, max_edges: int = 50):
        """
        Execute the complete pipeline.

        Steps:
        1. Generate query embeddings (for semantic search)
        2. Perform semantic search
        3. Retrieve and classify abstracts (generates its OWN query embeddings!)
        4. LLM validation with two-round validation and sentence mapping

        Args:
            max_edges: Maximum number of edges to process
        """
        print("\n" + "="*70)
        print("STARTING PUBMED EXTENSION PIPELINE")
        print("="*70 + "\n")

        # Step 1: Generate query embeddings FOR SEMANTIC SEARCH
        print("Step 1: Generating query embeddings for semantic search...")
        print(f"  Using: {self.config.ss_representation} + {self.config.ss_model_path}")
        query_vectors = self.ss_query_generator.generate_queries(
            self.edges,
            self.config.ss_representation,
            max_edges,
            self.llm_client
        )
        print(f"Generated {len(query_vectors)} query vectors\n")

        # Step 2: Semantic search
        print("Step 2: Performing semantic search...")
        edge_contexts, edge_context_counts = self.searcher.search_collections(
            query_vectors,
            self.config.ss_threshold
        )
        print(f"Search complete. Found contexts for {len(edge_contexts)} edges\n")

        # Save intermediate results
        self._save_json(edge_contexts, self.config.get_output_path('semantic_search'))
        self._save_json(edge_context_counts, self.config.get_output_path('search_counts'))

        # Step 3: Abstract classification with its OWN query embeddings
        print("Step 3: Classifying abstracts...")
        print(f"  Using: {self.config.ac_representation} + {self.config.ac_model_path}")

        classifier = AbstractClassifier(
            model=self.ac_model,
            threshold=self.config.ac_threshold,
            aggregation=self.config.ac_aggregation,
            representation=self.config.ac_representation,
            query_generator=self.ac_query_generator  # Use classification query generator
        )

        # Classify and get both results AND abstracts
        supporting_pmids, abstracts_dict = await classifier.classify_abstracts_with_retrieval(
            edges=self.edges,
            edge_contexts=edge_contexts,
            batch_size=100,
            llm_client=self.llm_client  # Needed for AI representation
        )

        print(f"Classification complete\n")

        self._save_json(
            supporting_pmids,
            self.config.get_output_path('abstract_classification')
        )

        # Save abstracts for potential manual inspection
        print(f"Caching {len(abstracts_dict)} abstracts for validation stage...")
        self._save_json(
            abstracts_dict,
            self.config.get_output_path('cached_abstracts')
        )

        # Step 4: LLM validation with sophisticated two-round pipeline
        if self.llm_client and self.response_parser:
            print("Step 4: LLM validation with two-round pipeline...")
            validator = LLMValidator(
                self.llm_client,
                self.data_loader.node_dict,
                self.data_loader.predicate_dict,
                self.response_parser
            )

            # Run complete validation pipeline
            supporting_results, non_supporting_results = await validator.validate_edges(
                self.edges,
                supporting_pmids,
                abstracts_dict
            )
            print(f"Validation complete\n")

            # Save supporting results (yes/maybe)
            if not supporting_results.empty:
                supporting_parquet_path = self.config.get_output_path('validation_results')
                supporting_results.to_parquet(supporting_parquet_path)
                print(f"Saved supporting results to: {supporting_parquet_path}")

                # Also save as JSON for inspection
                supporting_json = supporting_results.to_dict('records')
                supporting_json_path = self.config.get_output_path('llm_validation')
                self._save_json(supporting_json, supporting_json_path)

            # Save non-supporting results (no)
            if not non_supporting_results.empty:
                non_supporting_parquet_path = supporting_parquet_path.replace(
                    '_validation_results.parquet',
                    '_non_supporting_results.parquet'
                )
                non_supporting_results.to_parquet(non_supporting_parquet_path)
                print(f"Saved non-supporting results to: {non_supporting_parquet_path}")

                # Also save as JSON
                non_supporting_json = non_supporting_results.to_dict('records')
                non_supporting_json_path = supporting_json_path.replace(
                    '_llm_validated.json',
                    '_non_supporting.json'
                )
                self._save_json(non_supporting_json, non_supporting_json_path)

        statistics = self._calculate_pipeline_statistics(
            edge_contexts,
            supporting_pmids,
            supporting_results,
            non_supporting_results
        )

        statistics_path = self.config.get_output_path('semantic_search').replace(
                '_semantic_search.json',
                '_pipeline_statistics.json'
            )
        self._save_json(statistics, statistics_path)

        print("="*70)
        print("PIPELINE COMPLETE")
        print("="*70)

    @staticmethod
    def _save_json(data: Any, filepath: str):
        """Save data to JSON file"""
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"Saved results to: {filepath}")


# ============================================================================
# USAGE EXAMPLE
# ============================================================================

if __name__ == "__main__":
    # Example usage
    from ollama import Client
    from response_parser import SimpleLLMResponseParser
    import pandas as pd

    # Initialize clients
    llm_client = Client()
    response_parser = SimpleLLMResponseParser()

    edges = pd.read_parquet('edges/treats_nopub.parquet')
    edges = edges.head(50)

    # Run pipeline
    pipeline = PubExtensionPipeline(
        predicate='biolink:treats',
        llm_client=llm_client,
        response_parser=response_parser,
        edges=edges
    )

    # Run with async support
    import asyncio
    asyncio.run(pipeline.run(max_edges=50))
