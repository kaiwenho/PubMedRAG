from pymilvus import MilvusClient, DataType
import shutil
import os

def create_milvus_schema():
    schema = MilvusClient.create_schema(
        auto_id=True,
        enable_dynamic_field=False
    )

    schema.add_field(field_name="id", datatype=DataType.INT64, is_primary=True, auto_id=True)
    schema.add_field(field_name="pmid", datatype=DataType.INT64)
    schema.add_field(field_name="sentence", datatype=DataType.VARCHAR, max_length=60535)
    schema.add_field(field_name="vector", datatype=DataType.FLOAT_VECTOR, dim=384)

    schema.verify()
    return schema

def move_folder(source_folder, destination_folder):
    """
    Moves a folder from source_folder to destination_folder.

    Parameters:
    source_folder (str): The path of the folder to be moved.
    destination_folder (str): The target directory where the folder will be moved.

    Returns:
    str: Success message or error message.
    """
    try:
        # Ensure source folder exists
        if not os.path.exists(source_folder):
            return f"Error: Source folder '{source_folder}' does not exist."

        # Ensure destination directory exists; create if necessary
        if not os.path.exists(destination_folder):
            return f"Error: Destination folder '{destination_folder}' does not exist."

        # Construct full destination path
        folder_name = os.path.basename(source_folder)
        target_path = os.path.join(destination_folder, folder_name)

        # Move the folder
        shutil.move(source_folder, target_path)
        return f"Successfully moved '{source_folder}' to '{target_path}'."

    except Exception as e:
        return f"Error: {e}"
