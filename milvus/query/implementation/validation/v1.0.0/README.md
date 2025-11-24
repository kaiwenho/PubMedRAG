# LLM-Based Knowledge Graph Edge Review Pipeline

A pipeline for reviewing biomedical knowledge graph edges against PubMed literature using Large Language Models (LLMs). This tool performs two-round review with sentence-level evidence extraction to identify high-quality, evidence-backed knowledge graph assertions and flag edges requiring expert review.

## Overview

This pipeline reviews edges (subject-predicate-object relationships) in biomedical knowledge graphs by:
1. Retrieving abstracts from PubMed for publications associated with each edge
2. Using LLMs to determine if abstracts support the claimed relationships
3. Extracting specific supporting sentences from abstracts
4. Mapping LLM-generated sentences to their exact locations in source abstracts
5. Producing a reviewed dataset with evidence provenance

**Key Features:**
- Two-round review with smaller and larger models for quality control
- Sentence-level evidence extraction with anti-hallucination verification
- Comprehensive logging and error handling

## Installation

### Prerequisites & Dependencies
```bash
# Python 3.8+, Ollama
pip install pandas numpy nltk httpx
python -c "import nltk; nltk.download('punkt')"

# Pull Ollama models (adjust names to your setup)
ollama pull gpt-oss:20b
ollama pull gpt-oss:120b
```

**Note**: The gpt-oss models support "thinking mode" for internal reasoning before responding, improving judgment quality.

## Input Data Format

Designed for Knowledge Graphs in **KGX format** with **Biolink model** edges.

### Edges File (Parquet)
**Required columns:** `subject`, `object`, `predicate`, `publications` (list of PMIDs)

**Example:**
```python
{
    'subject': 'CHEMBL.COMPOUND:CHEMBL1234789',
    'object': 'MONDO:0005432',
    'predicate': 'biolink:treats',
    'publications': ['PMID:12345678', 'PMID:23456789']
}
```

### Node & Predicate Dictionaries (JSON)
Maps node IDs and predicates to detailed information. **Contact the authors to request these files.**

**Node dictionary structure:**
```json
{
  "CHEMBL.COMPOUND:CHEMBL1200879": {
    "name": "WARFARIN SODIUM",
    "category": "biolink:SmallMolecule",
    "description": "..."
  }
}
```

**Predicate dictionary structure:**
```json
{
  "biolink:treats": {
    "description": "...",
    "domain": "...",
    "range": "..."
  }
}
```

## Usage

### Basic Usage

```python
import asyncio
import json
from ollama import Client
from response_parser import SimpleLLMResponseParser
from llm_validator import main

# Load data
edges_df = pd.read_parquet("data/edges.parquet")
with open('data/node_dict.json') as f:
    node_dict = json.load(f)
with open('data/predicate_dict.json') as f:
    predicate_dict = json.load(f)

# Run review pipeline
asyncio.run(main(
    edges_file_path="data/edges.parquet",
    node_dict=node_dict,
    predicate_dict=predicate_dict,
    llm_client=Client(),
    response_parser=SimpleLLMResponseParser(),
    output_dir="./review_output",
    round1_model='gpt-oss:20b',
    round2_model='gpt-oss:120b',
    batch_size=100
))
```

**Note**: `batch_size` controls how many PMIDs are fetched from PubMed per API batch (default: 100), not the number of input edges processed at once.

## Output Files

The pipeline creates 6 files in your output directory:

1. **`edges_with_llm_reviews.parquet`** - Main output with new `LLM_review_publications` column containing review results
  **`LLM_review_publications` Column Structure:**
  ```python
  {
    'PMID:12345678': {
      'abstract_support?': 'yes',
      'support_sentences_from_abstract': [
        'Sentence supporting the relationship.',
        'Another supporting sentence.'
      ]
    },
    'PMID:87654321': {
      'abstract_support?': 'maybe'
    }
  }
  ```
2. **`validation_results.parquet`** - Detailed results for supporting publications ('yes'/'maybe')
3. **`no_support_results.parquet`** - Publications that don't support edges ('no')
4. **`abstracts_dict.parquet`** - Cached abstracts (reusable)
5. **`validation_summary.json`** - Summary statistics
6. **`validation_log_*.log`** - Comprehensive log file

## Configuration Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `edges_file_path` | str | Required | Path to edges parquet file |
| `node_dict` | Dict | Required | Node ID → node info mapping |
| `predicate_dict` | Dict | Required | Predicate → predicate info mapping |
| `llm_client` | Any | Required | Ollama client instance |
| `response_parser` | Any | Required | Response parser instance |
| `output_dir` | str | `"./validation_output"` | Output directory |
| `round1_model` | str | `'gpt-oss:20b'` | Model for initial review |
| `round2_model` | str | `'gpt-oss:120b'` | Model for refined review |
| `batch_size` | int | `100` | PMIDs per PubMed batch |

## Troubleshooting

**Issue: `KeyError` when accessing dictionaries**
- Verify all node/predicate IDs in edges exist in dictionaries
- Check `validator.missing_info_edges` for problematic edges

**Issue: Low review success rate**
- Review prompt template clarity
- Test LLM on sample cases manually
- Verify abstracts aren't truncated

**Issue: Sentence mapping failures**
- Use `fix_failed_mappings()` with larger model
- Check for encoding issues in abstracts

**Issue: Out of memory**
- Process edges in chunks
- Reduce `batch_size` for abstract retrieval
- Use machine with more RAM

**Issue: Slow review speed**
- Ensure Ollama uses GPU acceleration
- Use smaller model for Round 1
- **Timing**: ~10.5 seconds per edge-abstract pair on average
  - Round 1 only ('no' results, 20B model): ~7-8 seconds
  - Both rounds ('yes'/'maybe' results, 20B + 120B models): ~13-14 seconds

## Frequently Asked Questions

### Q: How accurate is the review process?
A: The two-round approach with sentence extraction improves accuracy by requiring models to provide concrete evidence and having a larger model verify results. However, this is an automated review tool to assist human curators, not replace them:
- **'yes'** results: Supporting text identified, warrants inclusion  
- **'maybe'** results: Requires domain expert review (especially for full-text)
- **'no'** results: Lacks clear abstract-level support

Accuracy depends on model quality, prompt design, and the inherent limitations of abstract-only analysis.

### Q: Why "review" instead of "validation"?
A: This pipeline performs automated literature review and evidence extraction, not formal validation. While it provides quality controls through two-round assessment and sentence verification, it cannot guarantee correct interpretation which would require domain expert judgment.

### Q: What are the key limitations?
- **Abstract-only**: Full-text articles may contain critical context
- **Interpretation**: Models may extract real sentences but misinterpret relevance
- **Implicit reasoning**: Models use "thinking mode" internally, but reasoning isn't visible in outputs
- **No shared reasoning**: Round 2 provides independent assessment without access to Round 1's reasoning

**Recommendation**: Spot-check 'yes' results and manually review 'maybe' results.

### Q: Can I use cloud-based LLM APIs?
A: Yes, modify `llm_client` to work with your API (OpenAI, Anthropic, etc.). The interface needs a `chat()` method returning similar response format.

### Q: How do I interpret 'maybe' results?
A: 'Maybe' flags suggestive or indirect evidence requiring domain expert review. These are particularly important for edges with limited publications (e.g., single PMID), where full-text review may be needed.

## Architecture Details

### Pipeline Workflow
1. **Extract PMIDs** from edges' publications column
2. **Fetch abstracts** from PubMed (batched)
3. **Generate prompts** for each edge-abstract pair
4. **Round 1** (fast model): Classify as 'yes'/'maybe'/'no', extract sentences
5. **Round 2** (larger model): Re-review 'yes'/'maybe' for quality control
6. **Merge results**: Keep confirmed, collect all 'no'
7. **Map sentences**: Verify extracted sentences exist in abstracts
8. **Integrate**: Add results back to edges dataframe

### Two-Round Logic
- **Round 1**: Reviews all pairs, extracts sentences for 'yes'/'maybe'
- **Round 2**: Re-reviews only 'yes'/'maybe' as quality control, can downgrade if insufficient
- **Result**: Only pairs confirmed by Round 2 remain as 'yes'/'maybe'

### Anti-Hallucination Design
Sentence extraction + fuzzy matching verification ensures models don't fabricate quotes. Models must provide exact sentences from abstracts, which are then verified to actually exist.

---

## License
MIT License

## Contact and Support
For questions or to request node_dict/predicate_dict files, please open an issue in the repository.

## Acknowledgments
- PubMed/NCBI for biomedical literature access
- Biolink Model for standardized KG representation
- KGX format for knowledge graph exchange

## Version History
- **v1.0.0** (2025): Initial release with two-round review and sentence mapping

## Future Enhancements
- Batch processing for very large knowledge graphs
- Full-text article support
- Additional literature databases
- Optional exposure of model reasoning for interpretability
