from pymilvus import MilvusClient, connections, utility
import time
import requests
import json
from utils import generate_treatment_sentences, embed_sentence
import pandas as pd

# Establish connection once
connections.connect(
    alias="default",
    uri="http://localhost:19530",
    token="root:Milvus",
)

def wait_for_node(resource_group="__default_resource_group", interval=5):
    """
    Poll utility.describe_resource_group until num_available_node >= 1.
    """
    while True:
        info = utility.describe_resource_group(name=resource_group)
        num_available_node = info.num_available_node
        print(f"Node availability: {num_available_node}")
        if num_available_node >= 1:
            print("Node is available—continuing execution.")
            return
        else:
            print(f"No nodes available, retrying in {interval}s…")
            time.sleep(interval)

wait_for_node()

ground_truth_tp = pd.read_parquet('data/arax_true_positve.parquet')
claims = generate_treatment_sentences(ground_truth_tp, True)

start = time.time()
claim_vectors = list(map(embed_sentence, claims))
end = time.time()
print(f"'Map claim to vector' execution time: {(end - start):.2f}s")

client = MilvusClient(
    uri="http://localhost:19530",
    token="root:Milvus"
)

claim_contexts = {}
claim_context_counts = {}
start = time.time()
for i in range(10):
    which_collection = f"pubmed_sentence_{i:02d}"
    print(f"loading collection: {which_collection}")
    client.load_collection(collection_name=which_collection)

    for claim_text, claim_vector in list(zip(claims, claim_vectors)):
        res = client.search(
            collection_name=which_collection,
            data=claim_vector,
            limit=50,
            search_params={
                "params": {
                    "radius": 0.75,
                    "range_filter": 1.0
                }
            },
            output_fields=["sentence", "pmid"],  # specifies fields to be returned
        )

        if claim_text not in claim_contexts:
            claim_contexts[claim_text] = res[0]
        else:
            claim_contexts[claim_text].extend(res[0])

        if claim_text not in claim_context_counts:
            claim_context_counts[claim_text] = [{which_collection: len(res[0])}]
        else:
            claim_context_counts[claim_text].append({which_collection: len(res[0])})

    client.release_collection(collection_name=which_collection)
    print(f"finished and unload: {client.get_load_state(collection_name=which_collection)}")

end = time.time()
print(f"'Load collections and conduct semantic search' execution time: {(end - start):.2f}s")



with open("result/semantic_search/arax_claim_contexts.json", "w", encoding="utf-8") as f:
    json.dump(claim_contexts, f, ensure_ascii=False, indent=2)

with open("result/semantic_search/arax_claim_contexts_counts.json", "w", encoding="utf-8") as f:
    json.dump(claim_context_counts, f, ensure_ascii=False, indent=2)

print("semantic search results saved to 'result/semantic_search/arax_claim_contexts.json'")
