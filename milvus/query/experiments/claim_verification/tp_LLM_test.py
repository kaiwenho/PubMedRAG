import pandas as pd
import json
from utils import extract_non_think, generate_prompt, filter_sort_trim_evidence, query_llm
import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

ground_truth_tp = pd.read_parquet('data/arax_true_positve.parquet')
print(f"number of true positive drug-disease pairs: {len(ground_truth_tp)}")

with open('result/semantic_search/arax_claim_contexts.json', 'r') as file:
    claim_contexts = json.load(file)

for claim in claim_contexts:
    claim_contexts[claim] = filter_sort_trim_evidence(claim_contexts[claim])

claims = []
prompts = []
pmids_list = []
for claim, results in claim_contexts.items():

    pmids = set([i['entity']['pmid'] for i in results])
    context = [i['entity']['sentence'] for i in results]
    if not context:
        prompt = False
    else:
        prompt = generate_prompt(claim, context)

    prompts.append(prompt)
    pmids_list.append(pmids)
    claims.append(claim)

print(f"number of prompts generated: {len(prompts)}")

claims_no_evidence = sum(1 for x in prompts if x is False)
print(f"There are {claims_no_evidence} ({(claims_no_evidence/len(prompts)) * 100:.1f}% of the) claims that does not have evidence.")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

LLMs = ['gemma3:4b', 'llama3.1:8b', 'mistral:7b', 'deepseek-r1:8b', 'phi4']

scores = []
i = 0
start = time.time()
for prompt in prompts:
    i += 1
    if i % 10 == 0:
        logging.info(f"Processing claim {i}/{len(prompts)}")
    if prompt:
        results = []

        with ThreadPoolExecutor(max_workers=len(LLMs)) as executor:
            futures = [executor.submit(query_llm, llm, prompt) for llm in LLMs]

            for future in as_completed(futures):
                llm_name, actual_response = future.result()

                if actual_response.lower().startswith('yes'):
                    results.append(1)
                elif actual_response.lower().startswith('no'):
                    results.append(0)
                else:
                    # TODO: make the query again for the specific llm?
                    print(f"Error: not a proper answer from {llm_name}", actual_response)

        score = sum(results)/len(results)

    else:
        score = 0

    scores.append(score)

end = time.time()
print(f"LLMs execution time: {(end - start):.2f}s")

with open("result/LLMs/arax_tp_scores.json", "w", encoding="utf-8") as f:
    json.dump(scores, f, ensure_ascii=False, indent=2)

result_test_df = ground_truth_tp.copy()
result_test_df['assumption'] = claims
result_test_df['score'] = scores
result_test_df['pmids'] = pmids_list
result_test_df.to_csv('result/LLMs/arax_ground_truth_tp_result.tsv', sep='\t', index=False)
