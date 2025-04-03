import json
from pymilvus.bulk_writer import bulk_import
from utils import create_milvus_schema, move_folder
from pymilvus.bulk_writer import LocalBulkWriter, BulkFileType
from pymilvus import MilvusClient

client = MilvusClient(
    uri="http://localhost:19530",
    token="root:Milvus"
)

index_params = client.prepare_index_params()

index_params.add_index(
    field_name="pmid",
    index_type="STL_SORT"
)

index_params.add_index(
    field_name="vector",
    index_type="AUTOINDEX",
    metric_type="COSINE"
)

buk_path_files = 'prepared_data_path_and_batch_count.json'
with open(buk_path_files, 'r') as file:
    paths = json.load(file)

print(f"There will be {len(paths)} collections.")

for i in range(len(paths)):
    collection_name = f"pubmed_sentence_{i:02d}"
    print(f"Creating collection {collection_name}...")
    directory_path = paths[i][0]
    count_parquet = paths[i][1]

    client.create_collection(
        collection_name=collection_name,
        schema=create_milvus_schema(),
        index_params=index_params
    )


    move_folder(directory_path, "../../../Milvus/volumes/milvus")
    folder_path = f"/var/lib/milvus/{directory_path}"
    file_names = [f"{n}.parquet" for n in range(1,(count_parquet+1))]
    files = [[f"{folder_path}/{f}"] for f in file_names]

    resp = bulk_import(
        url=f"http://127.0.0.1:19530",
        collection_name=collection_name,
        files=files,
    )

    job_id = resp.json()['data']['jobId']
    print(job_id)
