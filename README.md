# Entity Resolution Pipeline — Data Cleaning & Validation Split

## Overview

This module implements **Stage 1** (data cleaning/normalisation) and **Stage 2** (validation split) of an Entity Resolution pipeline for business records across multiple data sources and countries (US, India; test set adds France).

---

## Repository Layout

```
Amazon ML Challenge/
├── dataset/
│   ├── train/
│   │   ├── train_source1.tsv        (~2.2M rows)
│   │   ├── train_source2.tsv
│   │   ├── train_source3.tsv
│   │   └── train_ground_truth.tsv
│   └── test/
│       ├── test_source1.tsv
│       ├── test_source2.tsv
│       └── test_source3.tsv
├── src/
│   ├── config.py                    ← All paths, params, dicts
│   ├── data_cleaning.py             ← Stage 1
│   └── validation_split.py         ← Stage 2
├── output/
│   ├── cleaned/                     ← Cleaned TSVs (Stage 1 output)
│   └── splits/                      ← Split files (Stage 2 output)
└── logs/
    ├── data_cleaning.log
    ├── validation_split.log
    └── data_quality/
        ├── train_source1_quality_report.json
        ├── train_source2_quality_report.json
        └── train_source3_quality_report.json
```

---

## Stage 1: `data_cleaning.py`

### What it does

Reads each source TSV **in chunks** (configurable, default 50 000 rows) and writes a cleaned TSV with the original columns preserved plus new derived columns:

| New column | Source column | Description |
|---|---|---|
| `business_name_norm` | `business_name` | Lowercase, punct collapsed, `&→and`, legal suffixes → canonical tokens (NOT removed) |
| `business_name_core` | `business_name_norm` | Experimental: suffix-stripped variant. **Never used as sole identity signal.** |
| `business_name_alt` | `business_name` | DBA/trade-name extracted when detectable (`dba`, `t/a`, `trading as`, `aka`); `null` otherwise |
| `business_address_norm` | `business_address` | Lowercase, country-conditional abbreviation expansion (Rd→road, St→street, etc.) |
| `business_address_landmark` | `business_address` | Landmark phrase extracted when reliably detectable (`Near …`, `Opp …`); `null` otherwise |
| `country_norm` | `country` | Casing/whitespace standardised; **never remapped or bucketed** |

### Normalization design principles

- **Conservative**: ambiguous transforms produce *additional columns* rather than overwriting. Two distinct businesses collapsing to an identical normalised string is worse than under-normalising.
- **Country-open**: country selects which abbreviation dictionary to try first. Unknown countries (France, etc.) fall through to a generic dict — no crash, no skip.
- **No fabrication**: missing address components are left null; no city/state/PIN is invented.
- **No external services**: entirely dictionary/rule-based.

### Running

```bash
# All three train sources (default chunk size 50 000)
python -m src.data_cleaning

# Source1 only (faster iteration)
python -m src.data_cleaning --source1-only

# Custom chunk size
python -m src.data_cleaning --chunk-size 25000

# Also clean the test sources
python -m src.data_cleaning --include-test

# Custom output directory
python -m src.data_cleaning --output-dir /path/to/output
```

### Quality reports

After each source, a JSON report is written to `logs/data_quality/<source>_quality_report.json` containing:

- `total_rows`
- `null_counts_original_cols` — null rate per original column
- `null_entity_id_count` — null/empty entity_ids (rows kept, flagged)
- `duplicate_entity_id_count` — duplicate entity_ids within the file (rows kept, flagged)
- `country_distribution` — value counts for the `country` column
- `suffix_pattern_matches` — count of names containing legal suffix patterns
- `dba_pattern_matches` — count of names with detected DBA/trade-name
- `landmark_pattern_matches` — count of addresses with detected landmark phrases

> **entity_id is never modified.** Duplicates and nulls are **logged, never dropped or silently fixed**.

---

## Stage 2: `validation_split.py`

### Ground-truth validation

Every row in `train_ground_truth.tsv` is classified into **exactly one** of three categories:

| Category | Criterion |
|---|---|
| **Singleton** | `matched_entity_ids` is null or whitespace-only |
| **Matched** | One or more comma-separated IDs all matching `^(S2\|S3)-\d+$` |
| **Malformed** | Stray commas (empty tokens), non-`S2`/`S3` prefixes, or other parse failures |

Malformed rows are **logged and excluded from splitting** — never silently coerced into singleton or matched.

Source1 entity IDs present in ground truth but absent from `train_source1.tsv` are **flagged as missing references** and excluded from splitting (cannot stratify by country without a source record).

### Stratification

Split is stratified by **country × row_type** (e.g. `us::matched`, `india::singleton`).

**Small-stratum fallback**: if a stratum has fewer than `MIN_STRATUM_SIZE` (default 10) rows, all rows in that stratum are folded into train. The fallback is logged with the reason. No crash, no silent misassignment.

### What is NOT split: Source2/Source3

> **Source2 and Source3 files are NOT duplicated per split.**

Downstream code must load `output/cleaned/cleaned_source{2,3}.tsv` **once** and filter/join against `train_source1_ids.txt` or `val_source1_ids.txt` at runtime.

The same S2/S3 record appearing as a candidate for both train and val Source1 entities is **not leakage** — it mirrors the real matching problem where candidates are shared across queries. Only Source1 entity IDs and their ground-truth labels are separated between splits.

### Outputs

| File | Description |
|---|---|
| `output/splits/train_source1_ids.txt` | One S1 entity_id per line for the train split |
| `output/splits/val_source1_ids.txt` | One S1 entity_id per line for the val split |
| `output/splits/train_ground_truth.tsv` | GT rows for train S1 entities only |
| `output/splits/val_ground_truth.tsv` | GT rows for val S1 entities only |
| `output/splits/split_summary.json` | Full statistics (counts, distributions, fallback strata) |

### Running

```bash
# Default (15% val fraction, seed=42)
python -m src.validation_split

# Custom val fraction
python -m src.validation_split --val-fraction 0.20 --seed 123

# Custom paths
python -m src.validation_split --source1-path /path/to/source1.tsv \
    --gt-path /path/to/ground_truth.tsv \
    --output-dir /path/to/splits/
```

---

## Known Limitations

### France and unseen countries

France is **absent from training data**. It appears only in the test set.

This means:
- France cannot be stratified in the validation split.
- Validation metrics on US/India **cannot approximate** pipeline performance on France.
- Country stratification of US/India does **not** resolve or close this generalisation gap.

This limitation is explicitly logged at runtime and recorded in `split_summary.json`.

### Validation set scope

The validation split measures performance on held-out Source1 entities using only the countries and source records present in training. It is a within-distribution estimate only.

---

## Configuration (`src/config.py`)

All parameters are centralised here — never hard-coded in modules:

| Parameter | Default | Description |
|---|---|---|
| `CHUNK_SIZE` | 50 000 | Rows per chunk |
| `VAL_FRACTION` | 0.15 | Fraction of S1 entities for validation |
| `RANDOM_SEED` | 42 | Reproducibility seed |
| `MIN_STRATUM_SIZE` | 10 | Minimum stratum size before fallback |
| `LEGAL_SUFFIX_MAP` | see config | Raw → canonical suffix token mapping |
| `ABBREV_DICTS` | see config | US / India / generic abbreviation dicts |
| `LANDMARK_PATTERNS` | see config | Regex patterns for landmark extraction |
| `DBA_PATTERNS` | see config | Regex patterns for DBA/trade-name extraction |

---

## Design Notes

### Chunking and memory

- All three source files are read in configurable chunks. No single file is ever fully loaded into memory.
- The `QualityAccumulator` in `data_cleaning.py` holds all seen `entity_id` values in a Python `set` across chunks for duplicate detection. For 2.2M rows this is ~200 MB worst-case. An HLL sketch would reduce this to a few KB but loses exactness — documented trade-off.

### Vectorized operations

All hot-path transformations use pandas vectorised string operations (`str.replace`, `str.findall`, `str.split`, etc.). The only `.apply()` calls operate on already-processed list results (e.g., taking the first element of `str.findall` output), not on raw string cells.

### Ground-truth classification

The malformed row detection in `validation_split.py` uses `str.split().explode().str.match().groupby().all()` — a fully vectorised pandas pipeline — rather than per-row Python loops.
