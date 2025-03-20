# Prepare Source Data
# reference: https://milvus.io/docs/prepare-source-data.md

from pymilvus import MilvusClient, DataType
from pymilvus.bulk_writer import LocalBulkWriter, BulkFileType
from datasets import load_dataset
import os
import re
import json
from utils import create_milvus_schema

MAX_ROWS_PER_COLLECTION = 20_000_000
TOTAL_FILE_COUNT = 1219

schema = create_milvus_schema()

# print(schema.fields)
# print(schema.primary_field)

def get_writer_path_and_file_count(writer):
    pattern = re.compile(r'^(.*)/\d+\.parquet$')
    one_file = writer.batch_files[0][0]
    match = pattern.match(one_file)
    if match:
        directory_path = match.group(1)
        count_parquet = 0
        for filename in os.listdir(directory_path):
            if filename.endswith(".parquet"):
                count_parquet += 1
    else:
        raise Exception(f"file path not found from {one_file}")

    return [directory_path, count_parquet]

def create_new_writer():
    """Create or re-initiate a writer for a fresh collection."""
    # Example: Provide a unique name or other configuration per collection
    print(f"--- Creating new writer ---")
    writer = LocalBulkWriter(
        schema=schema,
        local_path='.',
        segment_size=512 * 1024 * 1024, # Default value
        file_type=BulkFileType.PARQUET
    )
    return writer

prepared_data_path_and_batch_count = []
count = 0
# Initialize the first writer
current_writer = create_new_writer()

for i in range(1, TOTAL_FILE_COUNT+1):
    filename = f"data/pubmed24n{i:04d}.parquet"
    print(f"Loading {filename}")

    # Load the dataset from Hugging Face
    data = load_dataset(
        "biomedical-translator/pubmed2024_sentence_embeddings",
        data_files=filename,
        trust_remote_code=True
    )
    df = data["train"].to_pandas()
    num_rows = len(df)

    # If adding these rows will exceed limit, commit + re-initiate writer
    if count + num_rows > MAX_ROWS_PER_COLLECTION:
        print(f"Reached {count} rows. Closing writer...")
        prepared_data_path_and_batch_count.append(get_writer_path_and_file_count(current_writer))
        with open('prepared_data_path_and_batch_count.json', 'w') as file:
            json.dump(prepared_data_path_and_batch_count, file)
        current_writer = create_new_writer()
        count = 0

    # Write new batch of rows
    docs = df['sentence'].tolist()
    vectors = df['embedding'].tolist()
    pmids = df['PMID'].tolist()
    pmids = list(map(int,pmids))

    for j in range(len(vectors)):
        current_writer.append_row({
            "pmid": pmids[j],
            "sentence": docs[j],
            "vector": vectors[j]
        })
        count += 1

    current_writer.commit()
    print(f'committed to folder {get_writer_path_and_file_count(current_writer)[0]} with total rows = {count}')

# Final commit for whatever is left in this last batch
# current_writer.commit()
path_and_count = get_writer_path_and_file_count(current_writer)
prepared_data_path_and_batch_count.append(path_and_count)
print(f"Committed final writer with total rows={count} to folder {path_and_count[0]}.")

# Saving the meta data of the prepared source data
with open('prepared_data_path_and_batch_count.json', 'w') as file:
    json.dump(prepared_data_path_and_batch_count, file)
