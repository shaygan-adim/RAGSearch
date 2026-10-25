# Methodology and evaluation protocol

This document describes exactly what each pipeline stage does, how the reported numbers were produced,
and where the experimental design has weaknesses. Read it before quoting any number from the README.

## 1. Data

Raw tables are described in [`data/README.md`](../data/README.md). Two derived tables drive everything:

* **Products** - ~1.02M items. Retrieval text = `persian_name` + brand name (when the brand is known),
  normalised with [Hazm](https://github.com/roshan-research/hazm).
* **Query sessions** - one row per search `uid`: the query text, every product shown across all result pages
  (`result_base_products`) and the products the user opened (`viewed_base_products`).
  Sessions are dropped when nothing was viewed, when a viewed product was never in the results
  (inconsistent log), or when the cleaned query has fewer than 2 characters.

**Relevance signal.** There are no human relevance labels. A product is "relevant" to a query if the user
opened it. This is a noisy proxy: unopened products are not necessarily irrelevant, and sessions have only
about two opened products on average (199,916 views over 96,536 sessions), which caps precision and is why absolute
precision is low everywhere.

## 2. Text retrieval

### 2.1 MLM domain adaptation (`ragsearch mlm`)
`HooshvareLab/distilbert-fa-zwnj-base` is further trained with masked-language modelling (15% masking, 3 epochs,
lr 2e-5) on an even mix of product texts and query texts (both undersampled to the smaller set, 90/10 train/val).
The checkpoint with the lowest validation loss is kept. Goal: make the encoder familiar with product titles,
brand names and the short, informal style of search queries.

### 2.2 Dual encoder (`ragsearch contrastive`)
Two independent towers - one for queries, one for products - both initialised from the adapted model:

```
text -> DistilBERT -> [CLS] -+-> contrastive head (MLP) -> L2-normalised 128-d embedding   (retrieval)
                             '-> classification head (MLP) -> product-category logits     (auxiliary)
```

**Training pairs.** Every (session, viewed product) is a positive. Each positive gets one negative, sampled as
40% shown-but-never-opened for that query, 30% random product from the positive's category, 30% random product
from the whole catalogue (also the fallback when the first two have no candidate).

**Loss.** `InfoNCE + 0.1 * classification`.
For a batch, row *i* of the query/positive similarity matrix must pick positive *i*; the batch's negatives are
appended as extra columns (temperature 0.07). The classification loss predicts the product category for the
query (if it has one), the positive and the negative; it only shapes the representation and is unused at
inference. 5 epochs, batch 32, AdamW lr 2e-5, weight decay 0.01. The final-epoch weights are saved (no
best-epoch selection).

Sessions are split 81/9/10 into train/val/test (stratified by category) for the training loop's validation loss.

### 2.3 Text evaluation (`ragsearch eval-text`)
1. Keep sessions with at least one view; impute missing query categories from the opened products.
2. Subsample 50% of sessions (stratified by category).
3. Candidate pool = every product opened or shown in those sessions (623,636 products in the reported run).
4. Keep the sessions in the contrastive test split and split them 50/50 into validation and test, stratified
   (the reported run used 24,134 + 24,134 sessions).
5. For each model: encode the pool with the product tower, encode queries, exact inner-product search
   (FAISS `IndexFlatIP`), compute P/R/F1@{3,5,10,20,40}, MRR and NDCG against the opened products.
6. Pick the model with the best validation NDCG and report it once on the test split.

Compared systems: `heydariAI/persian-embeddings` (sentence-transformers), the untouched Hooshvare DistilBERT
(CLS pooling), the MLM-adapted DistilBERT (CLS pooling), and the trained dual encoder.

## 3. Image retrieval (`ragsearch eval-image`)

There is no image query in the logs, so the task is constructed from co-clicks: the **image of one product is
the query** and the **other products opened for the same text query are the relevant items**.

* *Pure* set: only opened products, from query texts with >= 2 opened products.
* *Augmented* set: opened products plus up to 5 random shown-but-not-opened ones per query text.
* Query *texts* are split 50/50 into validation / test so a text query never appears in both.
* Candidate pool: 9,151 products that have an image. The query product itself is removed from its results.
* Encoders are used zero-shot (no fine-tuning): CLIP ViT-B/16, ConvNeXtV2-base, SigLIP2 so400m NaFlex, DINOv3 ViT-S/16.
  Models are ranked by `Val_Augmented` NDCG; the winner (SigLIP2) is run on the test sets.

## 4. Differences from the original notebooks

The code was ported from Colab notebooks into the `ragsearch` package. The algorithms are unchanged; these
details differ:

| Change | Why |
|---|---|
| One metrics implementation (`evaluation/metrics.py`) instead of two near-copies | Same formulas, single source of truth (unit-tested against hand-computed values). |
| Dual-encoder class defined once; checkpoint key names kept | The class was pasted into 3 notebooks; old `.pth` files still load. |
| Products keyed by `random_key` (was `uid` in the contrastive-training notebook) | The repo's preprocessing output has no `uid` column for products; `random_key` is the identifier used everywhere else. |
| Type-2 negatives by rejection sampling | The original rebuilt a category-sized candidate list for every sample (very slow on large categories); same distribution. |
| Image index keys stay aligned when an image fails to load | The original mapped FAISS row -> key by position over *all* candidates, which silently shifts if one file is unreadable. |
| Image augmentation sampling and the validation/test split are deterministic | The original sampled from a `set` (order depends on `PYTHONHASHSEED`). |
| `max_length: null` for the sentence-transformers baseline | The original logged "64" but never applied it to that model; the config now says so. |
| The MLM mixing ratio option was removed | Only the 50/50 branch was ever implemented. |
| Demo: `share` is off by default (`--share` to enable), product text is HTML-escaped | A public tunnel should be opt-in; product names are untrusted text rendered as HTML. |
| `trust_remote_code` no longer used when loading image models | These models are natively supported by recent `transformers`. |
| Dropped: the notebook's in-line retrieval check (k = 1/5/10), category classification report, interactive t-SNE plots, Drive-to-local image copy, zip export | Superseded by `eval-text`, or Colab-specific. Still available in git history (commit `25ee2b6`). |
