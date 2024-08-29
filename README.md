# translator_QA_RAG
This project is a Flask-based API that answers user questions by retrieving relevant information from scientific publications and using that information to generate a natural language response.

## API Overview
### Input Data Structure
The API expects input data in JSON format as follows:
```
{
  "question": "<a question from the user>",
  "publication_ids": ["<PMID:000000>", "<PMC:000000>", ...] // Optional
}
```
- question: The user's question in natural language.
- publication_ids (Optional): A list of publication identifiers (e.g., PubMed ID, PMC ID). If provided, the system will answer the question based on the specified publications. If omitted, the system will answer the question using information from its entire database.

### Output Data Format
The API responds with text generated by a Language Model (LLM), **streamed** to the client one word at a time. This allows for a more responsive experience, especially for longer answers.


