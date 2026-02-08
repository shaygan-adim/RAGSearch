import os
import yaml
import pandas as pd
from text_utils import TextCleaner
import product_builder
import query_builder
from tqdm import tqdm

def load_config(config_path="config.yaml"):
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def main():
    # 1. Setup
    current_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(current_dir)
    
    config = load_config()
    
    # Resolve absolute paths
    raw_path = os.path.abspath(os.path.join(current_dir, config['paths']['raw_data_dir']))
    processed_path = os.path.abspath(os.path.join(current_dir, config['paths']['processed_data_dir']))
    
    config['paths']['raw_data_dir'] = raw_path
    config['paths']['processed_data_dir'] = processed_path
    
    # Create output dir: data/preprocessed
    os.makedirs(processed_path, exist_ok=True)

    print(f"Raw Data Dir: {raw_path}")
    print(f"Output Data Dir: {processed_path}")

    # 2. Initialize Cleaner
    use_hazm = config['cleaning']['hazm_normalize']
    cleaner = TextCleaner(use_hazm=use_hazm)

    # 3. Run Product Pipeline
    df_products = product_builder.build_products(config, cleaner)
    
    print("Saving products...")
    prod_out_path = os.path.join(processed_path, 'preprocessed_products.parquet')
    df_products.to_parquet(prod_out_path, index=False)
    print(f"Saved products to: {prod_out_path}")

    print("-" * 30)

    # 4. Run Query Pipeline
    df_queries = query_builder.build_queries(config, cleaner)
    
    print("Saving queries...")
    query_out_path = os.path.join(processed_path, 'preprocessed_queries.parquet')
    df_queries.to_parquet(query_out_path, index=False)
    print(f"Saved queries to: {query_out_path}")

    print("-" * 30)
    print("Preprocessing pipeline completed successfully.")

if __name__ == "__main__":
    main()