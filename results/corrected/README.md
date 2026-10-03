# Corrected result tables

These are lightweight exports of the final corrected panel. Numeric fields
retain the original CSV strings and precision. Columns holding local server
paths were removed. No per-molecule labels, predictions, or embeddings are
included.

| File | Contents |
| --- | --- |
| `corrected_fold_auc.csv` | 16 neural configurations, eight datasets, five seeds, 640 rows |
| `corrected_dataset_summary.csv` | Dataset mean ROC-AUC and sample SD |
| `corrected_model_summary.csv` | Equal-weight mean of eight dataset means |
| `corrected_paired_contrasts.csv` | Dataset deltas and macro paired effects with bootstrap intervals |
| `coverage_audit.csv` | Recorded split coverage for final neural predictions |
| `checkpoint_selection.csv` | Recorded full-validation best epochs and selection metrics |
| `cpu_baseline_fold_auc.csv` | Five logistic-regression reference families, 200 rows |
| `cpu_baseline_dataset_summary.csv` | CPU reference dataset means and sample SDs |
| `cpu_baseline_model_summary.csv` | CPU reference macro means |

Folds 1 to 5 identify repeated scaffold splits with seeds 42 to 46, not a
single disjoint five-fold partition. Endpoint ROC-AUC is averaged within each
dataset and split after masking missing labels and omitting endpoints that
contain only one observed class. Dataset SD uses a divisor of four. Each
dataset receives equal weight in the macro mean.

Paired effects match dataset and fold. The hierarchical bootstrap resamples
eight datasets and then five fold deltas within each sampled dataset, using
10,000 draws and seed 42. The 2.5th and 97.5th percentiles form the interval.
Run `python scripts/summarize_reported_results.py --check` to rebuild summaries.

The coverage table reports checks made when raw predictions were aggregated.
The distributed AUC rows alone cannot independently establish sample coverage.
New training runs must retain their sample manifests and prediction files.
