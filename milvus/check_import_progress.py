from pymilvus.bulk_writer import list_import_jobs
import json


collection_name = "pubmed_sentence_02"
resp = list_import_jobs(
    url=f"http://127.0.0.1:19530",
    # collection_name=collection_name,
)

print(json.dumps(resp.json(), indent=4))
