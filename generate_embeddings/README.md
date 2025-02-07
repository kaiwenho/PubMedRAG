# PubMed 2024 Sentence Embeddings

This directory provides precomputed sentence embeddings for abstracts from the **[PubMed Baseline Repository](https://ftp.ncbi.nlm.nih.gov/pubmed/baseline/)** (updated December 2024). The embeddings were computed at the **sentence level**, meaning that each sentence from an abstract is treated individually, and the **title** of each publication is also considered a sentence.

Additionally, this directory includes the [script](https://github.com/kaiwenho/PubMedRAG/blob/main/generate_embeddings/pubmed_embeddings_from_xml.ipynb) used for generating the embeddings.

## Dataset & Embeddings
The precomputed sentence embeddings can be found at:
👉 **[Hugging Face Dataset](https://huggingface.co/datasets/biomedical-translator/pubmed2024_sentence_embeddings)**

## Embedding Computation Details
- **Sentence Tokenization Method**: `sent_tokenize`
- **Embedding Model Used**: `all-MiniLM-L6-v2`
- **Embedding Dimensions**: `384`
- **Number of Embeddings**: `185,145,351`
