# Biomedical Knowledge Graph PubMed Extension Pipeline

A pipeline for extending biomedical knowledge graphs by discovering, retrieving, and reviewing supporting evidence from PubMed literature. This tool takes edges without publication support and produces evidence-backed assertions through semantic search, abstract classification, and LLM review.

## Overview

This pipeline extends biomedical knowledge graphs (in KGX format following the Biolink model) by:
1. **Generating query representations** (concatenated or AI-paraphrased) for edges without publications
2. **Performing semantic search** across 38+ million PubMed abstracts using vector similarity
3. **Classifying candidate abstracts** to identify supporting evidence
4. **Reviewing edge-PMID pairs** through two-round LLM analysis with sentence-level evidence extraction
5. **Iteratively optimizing parameters** as new reviewed edges are discovered

### Key Features

- **Dual Representation System**: Supports both simple concatenated queries and AI-generated paraphrases for semantic search
- **Scalable Vector Search**: Leverages Milvus for efficient similarity search across millions of embeddings
- **Adaptive Parameter Optimization**: Continuously refines search and classification thresholds as validation data grows
- **Iterative Batch Processing**: Processes edges in batches with automatic parameter retuning between iterations
- **Comprehensive Evaluation Framework**: Validates configurations on sentence-level retrieval (F1) and abstract classification (PR-AUC)
- **Checkpoint Recovery**: Automatically resumes from last successful batch after interruptions
- **Production-Ready Outputs**: Generates extended edges with supporting PMIDs and gold sentence indices for downstream use

## Architecture

### Complete Pipeline Workflow

```
Input: Edges Without Publications (Parquet)
  ↓
[STAGE 1: SEMANTIC SEARCH]
  │
  ├─ Generate Query Embeddings
  │   • Concat: "subject_name predicate object_name"
  │   • AI: LLM-generated paraphrases (3 variations)
  │   • Model: General-purpose (sentence-transformers)
  │
  ├─ Search Milvus Collections (10 collections, 38M+ sentences)
  │   • Input: Query embedding
  │   • Search: Cosine similarity with threshold
  │   • Output: Candidate PMIDs per edge
  │
[STAGE 2: ABSTRACT CLASSIFICATION]
  │
  ├─ Retrieve Abstracts (Batched from PubMed)
  │   • 100 PMIDs per batch
  │   • Cached for efficiency
  │
  ├─ Classify Abstracts
  │   • Generate NEW embeddings (with classification model)
  │   • Compute sentence-level similarities
  │   • Aggregate: max, top2_mean, or top3_mean
  │   • Threshold: Tuned on validation set
  │   • Output: Supporting PMIDs per edge
  │
[STAGE 3: LLM REVIEW]
  │
  ├─ Two-Round Review
  │   • Round 1: Fast model (gpt-oss:20b) - all pairs
  │   • Round 2: Large model (gpt-oss:120b) - quality assurance on yes/maybe
  │   • Classification: yes, maybe, no
  │   • Evidence: Extract supporting sentences
  │
  ├─ Sentence Mapping
  │   • Map LLM sentences to abstract indices
  │   • Script-based + LLM fallback
  │   • Enables reproducibility and verification
  │
  ├─ Result Output
  │   • Supporting edges (yes/maybe) with evidence
  │   • Non-supporting edges (no) for analysis
  │
[STAGE 4: ITERATIVE OPTIMIZATION]
  │
  ├─ Process Batch Result
  │   • Get reviewed results (yes/maybe/no)
  │
  ├─ Add to Evaluation Dataset
  │   • Take ALL 'yes' results
  │   • Balance with 'no' results (maintain yes/no ratio)
  │   • Cross-batch balancing: Account for imbalance from previous batches
  │
  ├─ Accumulate Results
  │   • Track total new pairs added since last optimization
  │   • Continue processing batches until threshold reached
  │
  ├─ Re-optimize Parameters (when threshold reached, e.g., 50 pairs)
  │   • Re-run evaluation on augmented dataset
  │   • Tune search/classification thresholds
  │   • Update parameters for next batches
  │   • Reset accumulation counter
  │
  └─ Repeat Until Complete

Output: Edges with Extended Evidence (Parquet)
        • subject, object, predicate
        • pmid, abstract_support?
        • supporting_sentences, gold_sent_idxs
```

### Stage Descriptions

#### Stage 1: Semantic Search
Identifies candidate publications by comparing edge semantics to 38+ million PubMed abstract sentences. Uses configurable representation (concat vs AI) and general-purpose embedding model with tuned similarity thresholds.

#### Stage 2: Abstract Classification
Filters candidate PMIDs by re-embedding with the classification model (general-purpose vs biomedical-specific) and computing abstract-level support scores. Uses aggregation strategies (max, top2_mean, top3_mean) with thresholds optimized for precision-recall balance.

#### Stage 3: LLM Review

Reviews candidate edge-PMID pairs through two-round LLM analysis with sentence-level evidence extraction.

**Implementation:**
This pipeline includes the complete LLM review system integrated into the workflow. The review logic is the same as our standalone project [LLM-Based Knowledge Graph Edge Review Pipeline](https://github.com/kaiwenho/PubMedRAG/tree/main/milvus/query/implementation/validation/v1.0.0), adapted to work within the iterative optimization loop.

**Architecture:**
- Two-round review (fast model → quality assurance with larger model)
- Sentence-level evidence extraction
- yes/maybe/no classification
- Intelligent sentence mapping (script-based + LLM fallback)
- Comprehensive error handling

#### Stage 4: Iterative Optimization
Continuously improves pipeline parameters as new reviewed edges are discovered. Processes edges in batches, accumulates review results, and retunes thresholds to adapt to evolving data distribution.

## Core Components

### 1. Evaluation System (`biomed_eval.py`)

The foundation for parameter optimization through rigorous evaluation.

**Key Features:**
- **Data Splitting**: Stratified 50/50 val/test split by predicate and label
- **Sentence-Level Metrics**: MRR, Recall@k, AUC-ROC, AUC-PR
- **Abstract-Level Metrics**: ROC-AUC, PR-AUC, Precision, Recall, F1
- **Threshold Tuning**: Finds optimal thresholds for target recall/precision
- **Milvus Threshold Analysis**: Determines optimal similarity threshold (radius) for semantic search based on sentence-level retrieval performance
- **Bootstrap Confidence Intervals**: Statistical validation of results (1000 iterations)
- **Embedding Cache**: Disk-based caching for efficiency
- **History Tracking**: Maintains evaluation history per predicate with append/replace logic

**Configuration Comparison:**
The evaluator tests all combinations of:
- Representation: `concat`, `ai`
- Model: `general` (sentence-transformers/all-MiniLM-L6-v2), `biomedical` (PubMedBERT)
- Aggregation: `max`, `top2_mean`, `top3_mean`

Selects best based on combined sentence-level (MRR) and abstract-level (PR-AUC) performance.

### 2. PubMed Extension Pipeline (`pub_extension_pipeline.py`)

The main orchestrator that runs the complete semantic search and classification workflow.

**Outputs:**
- `result/semantic_search/{predicate}_semantic_search.json`: Raw semantic search results
- `result/classification/{predicate}_classified_abstracts.json`: Supporting PMIDs from abstract classifier
- `result/classification/{predicate}_cached_abstracts.json`: Retrieved abstracts
- `result/validation/{predicate}_validation_results.parquet`: LLM-reviewed edge-abstract pairs (yes/maybe) - pairs classified as supporting by abstract classifier
- `result/validation/{predicate}_non_supporting_results.parquet`: LLM-reviewed non-supporting pairs (no) - pairs classified as supporting by abstract classifier
- `result/pipeline_statistics.json`: Per-stage success rates

### 3. Iterative Batch Optimizer (`iterative_batch_optimizer.py`)

Orchestrates batch processing with continuous parameter optimization.

**Workflow:**
1. Start with base evaluation data (e.g., 1000 gold edges)
2. Optimize parameters → v0
3. Process batch 1 (100 edges) with v0 → review results
4. Add reviewed pairs to evaluation data (with cross-batch balancing)
5. Accumulate until threshold (e.g., 50 new pairs)
6. Re-optimize parameters → v1
7. Process next batch (batch 2, 3, 4, etc. - depends on when re-optimization threshold is met) with v1 → review results
8. Repeat until one of:
   - Max batches reached (configured limit)
   - All edges processed
   - Parameters converge (no significant change between iterations)*

**Key Point**: Optimization happens when **accumulated new pairs** reach threshold, not after every batch.

*Note: Convergence testing is not fully implemented yet - check optimization history to manually assess convergence if you would like to use it immediately, or wait for the next version which will have it implemented.

**Key Features:**
- **Cross-Batch Balancing**: Maintains yes/no balance across batches
- **Smart Accumulation**: Waits for sufficient new data before reoptimization
- **Skip Logic**: Skips batches with no 'yes' results (no supporting evidence)
- **Checkpoint Recovery**: Resumes from last successful batch
- **Extended Edges Output**: Saves yes/maybe results per batch + concatenated final
- **Evaluation Pairs Output**: Saves enriched validation pairs per batch

**Outputs:**
- `result/iterative_batches/extended_edges/batch_XXX_extended_edges.parquet`: Per-batch edges
- `result/iterative_batches/extended_edges/{predicate}_complete_extended_edges.parquet`: Final concatenated
- `result/iterative_batches/evaluation_pairs/batch_XXX_eval_pairs.parquet`: Enriched validation pairs
- `result/iterative_batches/optimization_history.json`: Complete parameter evolution
- `result/iterative_batches/checkpoint.json`: Recovery checkpoint
- `result/iterative_batches/final_report.txt`: Summary statistics

## Installation

### Prerequisites

- Python 3.8 or higher
- Ollama (for LLM features: AI queries, review)
- Milvus 2.x (for semantic search)

### Dependencies

```bash
pip install pandas numpy nltk httpx scikit-learn sentence-transformers pymilvus
```

### Setup Steps

1. **Download NLTK data**: `python -c "import nltk; nltk.download('punkt')"`
2. **Set up Ollama**: Install from https://ollama.ai and pull required models
3. **Set up Milvus**: Use Docker Compose or standalone installation (see Milvus docs)
4. **Obtain dictionaries and Milvus collections**: Contact authors for:
   - `dict/rtx-kg2_id_info_dictionary.json` (node information)
   - `dict/biolink_pred_info_dictionary.json` (predicate information)
   - Pre-built Milvus collections with PubMed sentence embeddings

## Input Data Format

This pipeline is designed for Knowledge Graphs in **KGX format** with edges following the **Biolink model**.

### Edges Without Publications (Parquet)

**Required Columns:**
- `subject`: Subject node ID (e.g., `CHEMBL.COMPOUND:CHEMBL1200879`)
- `object`: Object node ID (e.g., `MONDO:0005260`)
- `predicate`: Predicate ID (e.g., `biolink:treats`)

### Gold Standard Evaluation Data (Pickle or CSV; will add Parquet in the next version)

For initial parameter optimization, provide manually curated edges with:
- `subject`, `object`, `predicate`: Edge components
- `pmid`: Supporting PubMed ID
- `abstract_support?`: Label ('yes' or 'no')
- `abstract_sentences`: List of sentences from abstract
- `gold_sent_idxs`: Indices of supporting sentences (for 'yes' edges)
- `concat_sentence`: Concatenated edge text
- `ai_sentences`: List of AI-generated paraphrases

## Usage Examples

### Example 1: Initial Parameter Optimization

```python
from biomed_eval import BiomedicalRetrievalEvaluator

# Initialize evaluator with gold standard data
evaluator = BiomedicalRetrievalEvaluator(
    data_path='edges/gold_df.pkl',
    predicate='biolink:treats',
    random_state=42
)

# Split into validation/test (50/50)
evaluator.split_data(val_size=0.5)

# Optional: Pre-compute embeddings for faster evaluation
evaluator.precompute_all_embeddings(cache_dir='./embeddings_cache')

# Run validation to find best configuration
best_config = evaluator.run_validation_pipeline()

# Evaluate on test set
test_results = evaluator.run_test_evaluation()

# Analyze sentence thresholds for Milvus
threshold_analysis = evaluator._determine_best_sentence_threshold()

# Save results with parameter history
evaluator.save_results('evaluation_results.json')
```

### Example 2: Processing New Edges (Single Batch)

```python
import asyncio
import pandas as pd
from pub_extension_pipeline import PubExtensionPipeline
from ollama import Client
from response_parser import SimpleLLMResponseParser

# Load edges without publications
edges = pd.read_parquet('edges/treats_nopub.parquet')
edges_subset = edges.head(50)

# Initialize pipeline
pipeline = PubExtensionPipeline(
    predicate='biolink:treats',
    llm_client=Client(),
    response_parser=SimpleLLMResponseParser(),
    edges=edges_subset  # Edges to extend with PubMed support
)

# Run pipeline
await pipeline.run(max_edges=50)
```

### Example 3: Iterative Batch Optimization

```python
import asyncio
from iterative_batch_optimizer import IterativeBatchConfig, IterativeBatchOptimizer

# Configure iterative processing
config = IterativeBatchConfig(
    base_evaluation_data='edges/gold_df.pkl',
    nopub_edges_path='edges/treats_nopub.parquet',
    predicate='biolink:treats',
    batch_size=100,
    min_new_edges_for_reoptimization=50,
    max_batches=20,
    resume_from_checkpoint=True
)

# Run optimizer
optimizer = IterativeBatchOptimizer(config)
await optimizer.run()
```

### Example 4: Loading Optimized Parameters

```python
from biomed_eval import load_search_parameters

# Load latest parameters for a specific predicate
params = load_search_parameters(
    'evaluation_results.json',
    predicate='biolink:treats'
)

print(f"Sentence search threshold: {params['sentence_search']['threshold']}")
print(f"Abstract classification threshold: {params['abstract_classification']['threshold']}")
```

## Configuration

### Pipeline Parameters

**Semantic Search:**
- `representation`: 'concat' or 'ai'
- `model`: 'general' (sentence-transformers/all-MiniLM-L6-v2)
- `threshold`: Tuned on validation set

**Abstract Classification:**
- `representation`: 'concat' or 'ai'
- `model`: 'general' (sentence-transformers/all-MiniLM-L6-v2) or 'biomedical' (PubMedBERT)
- `aggregation`: 'max', 'top2_mean', or 'top3_mean'
- `threshold`: Tuned on validation set

**Iterative Optimization:**
- `batch_size`: Edges per iteration
- `min_new_edges_for_reoptimization`: Accumulation threshold
- `max_batches`: Maximum iterations (adjust based on dataset size)
- `val_size`: Validation split proportion (0.5 = 50/50)

## Performance Considerations

### Computational Requirements

**Milvus:**
- RAM: 40GB minimum (for 18M+ embeddings per collection)
- Storage: ~900GB for vector database

**LLM (Ollama):**
- GPU: Recommended for acceptable inference speed
  - 20B model: 13GB+ VRAM
  - 120B model: 48GB+ VRAM

**Embedding Models:**
- sentence-transformers/all-MiniLM-L6-v2: Lightweight general-purpose model, ~50MB in float16/bfloat16 format
- PubMedBERT (NeuML/pubmedbert-base-embeddings): Biomedical-specific model,  ~220MB if using float16/bfloat16

### Timing Estimates

**Measured Performance:**
- **Semantic search**: ~180 seconds per batch (across 10 collections)
**LLM review**: ~10.5 seconds per edge-abstract pair on average
  - Round 1 only ('no' results, 20B model): ~7-8 seconds
  - Both rounds ('yes'/'maybe' results, 20B + 120B models): ~13-14 seconds

*Note: Abstract classification and other component timings will be added as we collect more performance data.*

## Troubleshooting

### Common Issues

**Milvus connection errors:**
- Verify Milvus is running: `docker ps`
- Check connection settings in `pub_extension_pipeline.py`

**Missing node/predicate information:**
- Verify all IDs in edges exist in dictionaries
- Add missing entries or filter edges

**LLM review failures:**
- Verify Ollama is running: `ollama list`
- Try different models or adjust temperature

**Out of memory during evaluation:**
- Reduce batch size
- Process predicates separately

**Checkpoint corruption:**
- Delete `result/iterative_batches/checkpoint.json`
- Set `resume_from_checkpoint=False` to start fresh

## Contact and Support

For questions, issues, or to request dictionary files, open an issue in the repository.

## Acknowledgments

- **PubMed/NCBI**: Free access to biomedical literature
- **Biolink Model**: Standardized knowledge graph representation
- **KGX Format**: Knowledge graph exchange format
- **Milvus**: High-performance vector database
- **Ollama**: Local LLM inference

## License

MIT License

## Related Projects

- **[LLM-Based Knowledge Graph Edge Review Pipeline](https://github.com/kaiwenho/PubMedRAG/tree/main/milvus/query/implementation/validation/v1.0.0)**: Sophisticated two-round LLM review with sentence mapping
- **Biolink Model**: https://biolink.github.io/biolink-model/
- **KGX**: https://github.com/biolink/kgx
- **Milvus**: https://milvus.io/
