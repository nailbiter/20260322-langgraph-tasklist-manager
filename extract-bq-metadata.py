import json
import os
from google.cloud import bigquery

def export_bq_metadata(project_id, dataset_id, output_dir="bq_metadata"):
    """
    Iterates through all tables in a BigQuery dataset and dumps their 
    API representation to local JSON files.
    """
    # Initialize the BigQuery client
    # Note: Ensure you have active GCP credentials (e.g., via gcloud auth)
    client = bigquery.Client()
    
    dataset_ref = f"{project_id}.{dataset_id}"
    tables = client.list_tables(dataset_ref)
    
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created directory: {output_dir}")

    print(f"Starting metadata export for {dataset_ref}...")

    for table_item in tables:
        table_id = table_item.table_id
        table_full_name = f"{dataset_ref}.{table_id}"
        
        try:
            # Fetch the full table object to get schemas and descriptions
            table_obj = client.get_table(table_full_name)
            
            # Convert the table object to its API representation (dict)
            metadata_dict = table_obj.to_api_repr()
            
            # Define file path
            file_name = f"{table_id}.json"
            file_path = os.path.join(output_dir, file_name)
            
            # Write to local JSON file
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(metadata_dict, f, indent=2, ensure_ascii=False)
            
            print(f" Successfully exported: {file_name}")
            
        except Exception as e:
            print(f" Failed to export {table_id}: {e}")

    print(f"\nExport complete. Files are located in the '{output_dir}' folder.")

if __name__ == "__main__":
    # Parameters for the public thelook_ecommerce dataset
    PUBLIC_PROJECT = "bigquery-public-data"
    DATASET = "thelook_ecommerce"
    
    export_bq_metadata(PUBLIC_PROJECT, DATASET)