import pandas as pd
import os
import numpy as np
from tqdm import tqdm

def build_queries(config, cleaner):
    """
    Aggregates searches and views by UID, filters invalid/empty interactions,
    merges metadata, and prepares the cleaned query text.
    """
    tqdm.pandas(desc="Query Processing")
    print("--- Starting Query Pipeline ---")
    
    raw_dir = config['paths']['raw_data_dir']
    
    # 1. Load Data
    print("Loading raw parquet files...")
    searches_path = os.path.join(raw_dir, 'searches.parquet')
    categories_path = os.path.join(raw_dir, 'categories.parquet')
    views_path = os.path.join(raw_dir, 'base_views.parquet')

    df_searches = pd.read_parquet(searches_path)
    df_categories = pd.read_parquet(categories_path)
    df_views = pd.read_parquet(views_path)

    # 2. Aggregation Helper Functions
    def unique_union_list(series):
        """Combines lists from rows and returns unique elements as a list."""
        try:
            # Flatten list of lists and get unique using set
            flattened = [item for sublist in series if isinstance(sublist, (list, np.ndarray)) for item in sublist]
            return list(set(flattened))
        except TypeError:
            return []

    # 3. Process Searches (Aggregate by UID)
    print("Aggregating Searches by UID...")
    
    # We aggregate result_base_product_rks using set logic to ensure uniqueness (Task 4)
    search_agg = df_searches.groupby('uid').agg({
        'query': 'first',
        'category_id': 'first',
        'result_base_product_rks': unique_union_list
    }).rename(columns={'result_base_product_rks': 'result_base_products'})
    
    search_agg = search_agg.reset_index()

    # 4. Process Views (Aggregate by UID)
    print("Aggregating Views by UID...")
    
    # Map views to UIDs
    page_to_uid = df_searches[['id', 'uid']].drop_duplicates()
    
    df_views_mapped = df_views.merge(
        page_to_uid, 
        left_on='search_id', 
        right_on='id', 
        how='inner'
    )
    
    # Group by UID and aggregate viewed products (Task 4: Deduplicate using set)
    view_agg = df_views_mapped.groupby('uid')['base_product_rk'].apply(lambda x: list(set(x))).reset_index()
    view_agg.rename(columns={'base_product_rk': 'viewed_base_products'}, inplace=True)

    # 5. Merge Searches and Views
    print("Merging aggregated data...")
    final_df = search_agg.merge(view_agg, on='uid', how='left')
    
    # Fill NaN views with empty lists
    final_df['viewed_base_products'] = final_df['viewed_base_products'].apply(lambda d: d if isinstance(d, list) else [])

    # --- FILTERING LOGIC START ---
    
    # 6. Filter 1: Remove rows where viewed_base_products is empty (Task 2)
    print("Filtering empty views...")
    count_before_empty = len(final_df)
    
    final_df = final_df[final_df['viewed_base_products'].map(len) > 0]
    
    count_after_empty = len(final_df)
    dropped_empty = count_before_empty - count_after_empty
    perc_empty = (dropped_empty / count_before_empty) * 100 if count_before_empty > 0 else 0
    print(f"Dropped {dropped_empty} rows ({perc_empty:.2f}%) because viewed_base_products was empty.")

    # 7. Filter 2: Remove rows where viewed products are not in result products (Task 3)
    # This checks consistency: User cannot view what wasn't returned in search results.
    print("Checking consistency (viewed subset of results)...")
    
    def is_view_subset_of_result(row):
        # Convert to sets for O(1) lookups
        views = set(row['viewed_base_products'])
        results = set(row['result_base_products'])
        return views.issubset(results)

    count_before_subset = len(final_df)
    
    # Using progress_apply as this might be slower
    mask_consistent = final_df.progress_apply(is_view_subset_of_result, axis=1)
    final_df = final_df[mask_consistent]
    
    count_after_subset = len(final_df)
    dropped_subset = count_before_subset - count_after_subset
    perc_subset = (dropped_subset / count_before_subset) * 100 if count_before_subset > 0 else 0
    print(f"Dropped {dropped_subset} rows ({perc_subset:.2f}%) because viewed products were not in results.")
    
    # --- FILTERING LOGIC END ---

    # 8. Merge Category Names
    print("Merging category names...")
    final_df = final_df.merge(
        df_categories[['id', 'title']].rename(columns={'title': 'category_name'}),
        left_on='category_id',
        right_on='id',
        how='left'
    )

    # 9. Construct Cleaned Query Text
    print("Constructing and normalizing query text...")
    separator = config['text_construction']['query']['separator']
    append_category = config['text_construction']['query']['append_category']

    final_df['temp_text'] = final_df['query'].fillna('')

    if append_category:
        mask = (final_df['category_id'] != 0) & (final_df['category_id'].notna()) & (final_df['category_name'].notna())
        final_df.loc[mask, 'temp_text'] = (
            final_df.loc[mask, 'temp_text'] + separator + final_df.loc[mask, 'category_name'].astype(str)
        )

    # Normalize
    final_df['cleaned_query'] = final_df['temp_text'].progress_apply(cleaner.normalize)

    # 10. Filter Short Queries
    print("Filtering short queries...")
    min_len = config['cleaning']['min_query_length']
    initial_count = len(final_df)
    
    final_df = final_df[final_df['cleaned_query'].apply(lambda x: cleaner.filter_length(x, min_len))]
    
    print(f"Dropped {initial_count - len(final_df)} queries shorter than {min_len} chars.")

    # 11. Final Selection
    required_cols = [
        'uid', 'query', 'cleaned_query', 'result_base_products', 
        'category_id', 'category_name', 'viewed_base_products'
    ]
    
    output_df = final_df[required_cols].copy()

    print(f"Query pipeline finished. Resulting shape: {output_df.shape}")
    return output_df