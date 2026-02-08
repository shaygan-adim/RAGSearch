import pandas as pd
import os
import numpy as np
from tqdm import tqdm

def build_products(config, cleaner):
    """
    Loads raw product data, merges with categories and brands, 
    constructs the cleaned text, and prepares the final schema.
    """
    tqdm.pandas(desc="Product Processing")
    print("--- Starting Product Pipeline ---")
    
    raw_dir = config['paths']['raw_data_dir']
    
    # 1. Load Data
    print("Loading raw parquet files...")
    products_path = os.path.join(raw_dir, 'base_products.parquet')
    categories_path = os.path.join(raw_dir, 'categories.parquet')
    brands_path = os.path.join(raw_dir, 'brands.parquet')

    df_products = pd.read_parquet(products_path)
    df_categories = pd.read_parquet(categories_path)
    df_brands = pd.read_parquet(brands_path)

    # 2. Merge Data
    print("Merging categories and brands...")
    # Merge Categories
    df_merged = df_products.merge(
        df_categories[['id', 'title']].rename(columns={'title': 'category_name'}),
        left_on='category_id',
        right_on='id',
        how='left',
        suffixes=('', '_cat_drop')
    )

    # Merge Brands
    df_merged = df_merged.merge(
        df_brands[['id', 'title']].rename(columns={'title': 'brand_name'}),
        left_on='brand_id',
        right_on='id',
        how='left',
        suffixes=('', '_brand_drop')
    )

    # 3. Construct Text (Vectorized)
    print("Constructing cleaned text...")
    separator = config['text_construction']['product']['separator']
    include_brand = config['text_construction']['product']['include_brand']
    include_category = config['text_construction']['product']['include_category']

    # Start with product name
    df_merged['temp_text'] = df_merged['persian_name'].fillna('')

    if include_category:
        cat_suffix = np.where(
            df_merged['category_name'].notna(), 
            separator + df_merged['category_name'].astype(str), 
            ""
        )
        df_merged['temp_text'] = df_merged['temp_text'] + cat_suffix

    if include_brand:
        # LOGIC CHANGE: Only add brand if it exists AND brand_id is not -1
        brand_suffix = np.where(
            (df_merged['brand_name'].notna()) & (df_merged['brand_id'] != -1),
            separator + df_merged['brand_name'].astype(str),
            ""
        )
        df_merged['temp_text'] = df_merged['temp_text'] + brand_suffix

    # 4. Normalize
    print("Normalizing text (Applying Hazm)...")
    df_merged['cleaned_text'] = df_merged['temp_text'].progress_apply(cleaner.normalize)

    # 5. Final Selection
    print("Selecting final columns...")
    
    required_cols = [
        'random_key', 'persian_name', 'category_id', 'category_name', 
        'brand_id', 'brand_name', 'cleaned_text', 'image_url'
    ]
    
    # Ensure columns exist
    for col in required_cols:
        if col not in df_merged.columns:
            df_merged[col] = None

    output_df = df_merged[required_cols].copy()
    
    # Filter empty texts
    output_df = output_df[output_df['cleaned_text'].str.len() > 0]

    print(f"Product pipeline finished. Resulting shape: {output_df.shape}")
    return output_df