# Amazon ML Challenge — Business Entity Resolution

## Project structure

```
Amazon ML Challenge/
├── dataset/
│   ├── train/                  # raw train TSVs (never modified by the pipeline)
│   └── test/                   # raw test TSVs (never modified by the pipeline)
├── cleaned_dataset/             # output of Stage 1 (data_cleaning.py)
│   ├── cleaned_source1.tsv
│   ├── cleaned_source2.tsv
│   ├── cleaned_source3.tsv
│   ├── cleaned_test_source1.tsv   # only when run with --include-test
│   ├── cleaned_test_source2.tsv   # only when run with --include-test
│   └── cleaned_test_source3.tsv   # only when run with --include-test
├── output/
│   └── splits/                  # output of Stage 2 (validation_split.py)
│       ├── train_source1_ids.txt
│       ├── val_source1_ids.txt
│       ├── train_ground_truth.tsv
│       ├── val_ground_truth.tsv
│       └── split_summary.json
├── logs/
│   ├── data_cleaning.log
│   ├── validation_split.log
│   └── data_quality/            # per-source JSON quality reports
├── src/
│   ├── config.py                # all paths, hyperparameters, rule dictionaries
│   ├── data_cleaning.py         # Stage 1: chunk-wise cleaning & normalization
│   └── validation_split.py      # Stage 2: ground-truth validation & stratified split
└── README.md
```

## Folder responsibilities

- **`dataset/`** — raw inputs only. Neither pipeline stage ever writes here.
- **`cleaned_dataset/`** — root-level home for all cleaned/normalized TSVs
  produced by Stage 1. Kept separate from `output/` so cleaned data is easy
  to find and isn't mixed in with split artifacts.
- **`output/splits/`** — Stage 2 artifacts only (ID lists, filtered
  ground-truth subsets, split summary). Source2/Source3 files are **not**
  duplicated here — downstream code loads the full `cleaned_dataset/cleaned_source{2,3}.tsv`
  once and filters by the ID lists at runtime.
- **`logs/`** — run logs and per-source data-quality JSON reports.

## Stage 1 — Data cleaning

```
python -m src.data_cleaning                    # clean all train sources → cleaned_dataset/
python -m src.data_cleaning --include-test      # also clean test sources
python -m src.data_cleaning --source1-only      # faster iteration, train_source1 only
python -m src.data_cleaning --output-dir /path  # override the output directory
```

Run from the project root (`Amazon ML Challenge/`) so `config.py`'s
`PROJECT_ROOT`-relative paths resolve correctly regardless of shell state.

## Stage 2 — Validation split

```
python -m src.validation_split
python -m src.validation_split --val-fraction 0.20 --seed 123
```

Writes to `output/splits/`. Does not touch `cleaned_dataset/`.