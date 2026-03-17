# MoA Hypothesis Evidence Pipeline

A pipeline for finding PubMed evidence supporting mechanism of action (MoA) hypotheses. Given a multi-hop hypothesis path expressed as natural language sentences (e.g., *Metformin activates AMPK → AMPK inhibits mTOR → mTOR inhibition promotes autophagy*), the pipeline searches PubMed literature, validates candidate abstracts with an LLM, and synthesizes the evidence into an overall hypothesis assessment.

Adapted from the PubMed Extension Pipeline (`pub_extension_pipeline.py`), which was designed for enriching knowledge graph edges with publication support. This pipeline is redesigned for interactive hypothesis exploration rather than large-scale batch processing.


## How It Works

The pipeline runs in five stages:

**1. Query Encoding** — Each MoA sentence is encoded into a dense vector using a biomedical sentence transformer (default: `sentence-transformers/all-MiniLM-L6-v2`).

**2. Semantic Search** — The query vectors are searched against Milvus collections of pre-indexed PubMed sentences. Hits are aggregated to the abstract (PMID) level by keeping the maximum similarity score per PMID, then sorted and capped at the `top_k_abstracts` cutoff (default: 300 per sentence).

**3. Abstract Fetching & Caching** — All unique PMIDs across all sentences are collected, deduplicated, and fetched from PubMed in batches. Each abstract is fetched at most once and cached for the duration of the run.

**4. LLM Validation (Two Rounds)** — Each sentence–abstract pair is evaluated by an LLM:
  - **Round 1** uses a smaller, faster model to filter out definitive "no" results.
  - **Round 2** uses a larger model to verify the remaining "yes" and "maybe" results.

  Every prompt includes the full hypothesis path for mechanistic context, optional structured entity/predicate information, and asks for a reasoning trace, confidence score, support judgment, and supporting sentences.

**5. Hypothesis Synthesis** — A final LLM call takes the per-hop evidence summaries and produces an overall assessment: per-hop evidence strength, weakest links, overall confidence, and a narrative synthesis.


## File Structure

```
moa_evidence/
├── moa_evidence_pipeline.py   # Main pipeline
├── pubmed_client.py           # PubMed abstract fetching via API
├── response_parser.py         # LLM JSON response parsing and repair
├── utils.py                   # Sentence matching utilities (index mapping, fuzzy matching)
└── README.md
```


## Prerequisites

**Python packages:**

```
sentence-transformers
pymilvus
nltk
pandas
numpy
httpx
ollama
```

**Infrastructure:**

- **Milvus** instance with pre-indexed PubMed sentence collections (named `pubmed_sentence_00` through `pubmed_sentence_09`).
- **Ollama** (or compatible LLM server) with the models specified in config (defaults: `gpt-oss:20b` for round 1, `gpt-oss:120b` for round 2).
- **NLTK data:**

```python
import nltk
nltk.download('punkt_tab')
```


## Usage

### Minimal Example

```python
import asyncio
from ollama import Client
from response_parser import SimpleLLMResponseParser
from moa_evidence_pipeline import (
    MoAConfig, MoASentence, MoAHypothesis, MoAEvidencePipeline
)

llm_client = Client()
response_parser = SimpleLLMResponseParser()

hypothesis = MoAHypothesis(
    sentences=[
        MoASentence("Metformin activates AMPK"),
        MoASentence("AMPK inhibits mTOR signaling"),
        MoASentence("mTOR inhibition promotes autophagy"),
    ],
    name="Metformin-autophagy mechanism",
)

config = MoAConfig(
    ss_threshold=0.5,
    top_k_abstracts=300,
    output_dir='result/moa_evidence/metformin_autophagy',
)

pipeline = MoAEvidencePipeline(
    config=config,
    llm_client=llm_client,
    response_parser=response_parser,
)

results = asyncio.run(pipeline.run(hypothesis))
```

### With Structured Context

Providing entity and predicate details improves LLM validation quality. Context is optional and can be provided per-sentence at any level of detail:

```python
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
        MoASentence("mTOR inhibition promotes autophagy"),
    ],
    name="Metformin-AMPK-mTOR-autophagy mechanism",
)
```

Sentences without structured context still work — the LLM prompt adapts to include or exclude the context block accordingly.


## Configuration

All configuration is managed through `MoAConfig`:

| Parameter | Default | Description |
|---|---|---|
| `ss_model_path` | `sentence-transformers/all-MiniLM-L6-v2` | Sentence transformer model for encoding queries |
| `ss_threshold` | `0.5` | Minimum cosine similarity for semantic search hits |
| `top_k_abstracts` | `300` | Max unique abstracts per MoA sentence after deduplication and ranking |
| `milvus_uri` | `http://localhost:19530` | Milvus server URI |
| `milvus_token` | `root:Milvus` | Milvus authentication token |
| `num_collections` | `10` | Number of Milvus collections to search |
| `round1_model` | `gpt-oss:20b` | LLM for round 1 (fast filtering) |
| `round2_model` | `gpt-oss:120b` | LLM for round 2 (verification) and synthesis |
| `context_window` | `8192` | LLM context window size |
| `output_dir` | `result/moa_evidence` | Directory for all output files |


## Output Files

All outputs are saved as JSON in the configured `output_dir`:

| File | Description |
|---|---|
| `semantic_search_results.json` | Per-sentence PMID lists with similarity scores |
| `search_counts.json` | Per-sentence, per-collection hit counts |
| `cached_abstracts.json` | All fetched abstracts (keyed by PMID) |
| `supporting_results.json` | Validation results for yes/maybe abstracts, including reasoning traces, confidence, and mapped sentence indices |
| `non_supporting_results.json` | Validation results for rejected abstracts |
| `hypothesis_synthesis.json` | Overall hypothesis assessment with per-hop strength, weakest links, and narrative |
| `pipeline_statistics.json` | Per-hop and aggregate statistics |


## Output Structure

The `pipeline.run()` method returns a dict:

```python
{
    "hypothesis": {
        "name": "...",
        "path": "sentence1 → sentence2 → ...",
        "sentences": ["...", "..."],
    },
    "search_counts": { ... },
    "supporting_results": [
        {
            "sentence_idx": 0,
            "pmid": "PMID:12345678",
            "search_score": 0.72,
            "extraction_status": "success",
            "extracted_data": {
                "support": "yes",
                "confidence": "high",
                "reasoning": "The abstract describes a direct experiment...",
                "sentences": ["Metformin treatment resulted in..."],
            },
            "gold_sent_idxs": [3, 4],
        },
        ...
    ],
    "non_supporting_results": [ ... ],
    "synthesis": {
        "per_hop_assessment": [
            {
                "step": 1,
                "sentence": "Metformin activates AMPK",
                "evidence_strength": "strong",
                "summary": "Multiple abstracts provide direct experimental evidence...",
            },
            ...
        ],
        "weakest_links": "Step 3 has the weakest evidence...",
        "overall_confidence": "medium",
        "overall_assessment": "The hypothesis is partially supported...",
    },
    "statistics": { ... },
}
```
