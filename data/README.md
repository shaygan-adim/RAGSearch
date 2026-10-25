# Data

Nothing in this directory is committed except this file (see `.gitignore`): the dataset is large and
was supplied with the course, so you need to provide it yourself.

```
data/
├── raw/          # input - the original parquet tables (you provide these)
├── processed/    # created by `ragsearch preprocess`
└── images/       # product images, <random_key>.jpg (you provide these; see below)
```

Paths are configurable under `paths:` in [`configs/default.yaml`](../configs/default.yaml).

## `data/raw/`

Only the tables below are read by the pipeline. Other tables in the original dump (`members`, `shops`,
`cities`, `final_clicks`) are only used by the EDA notebook.

| File | Columns used | Notes |
|---|---|---|
| `base_products.parquet` | `random_key`, `persian_name`, `category_id`, `brand_id`, `image_url` | ~1.02M products. `brand_id == -1` means "no brand". |
| `categories.parquet` | `id`, `title` | 746 categories. `category_id == 0` in the logs means "unknown". |
| `brands.parquet` | `id`, `title` | 2,025 brands. |
| `searches.parquet` | `id`, `uid`, `query`, `category_id`, `result_base_product_rks` | One row per result *page*; `uid` identifies the whole search session. `result_base_product_rks` is a list of product keys. |
| `base_views.parquet` | `search_id`, `base_product_rk` | A product the user opened from a result page (`search_id` = `searches.id`). |

Ids in `categories`/`brands` must be unique; the loader fails loudly otherwise instead of duplicating rows.

## `data/processed/` (generated)

| File | One row per | Columns |
|---|---|---|
| `preprocessed_products.parquet` | product with non-empty text | `random_key`, `persian_name`, `category_id`, `category_name`, `brand_id`, `brand_name`, `cleaned_text`, `image_url` |
| `preprocessed_queries.parquet` | search session with at least one view | `uid`, `query`, `cleaned_query`, `result_base_products`, `viewed_base_products`, `category_id`, `category_name` |

`cleaned_text` is `persian_name [+ category] [+ brand]` (configurable) after Hazm normalisation.
`viewed_base_products` (the positives for training and evaluation) is always a subset of `result_base_products`.

## `data/images/`

Used by the image-only experiments and the demo. One JPEG per product, named after its key
(`<random_key>.jpg`, e.g. `chjknu.jpg`); products without a file are simply skipped by those stages.
The original experiments used a category-balanced sample of **9,151 products**. To recreate the folder,
download and sample images from `base_products.image_url`.
