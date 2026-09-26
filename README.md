# Amazon ML Challenge — Remaining pipeline using a friend's existing candidate TSV

This repo contains the implemented Stage 2 and Stage 3 logic for validating the train split and labeling existing candidate pairs. The newly developed blocking pipeline is intentionally skipped.

Important:
- Do not run `python -m src.blocking`
- Do not run `python -m src.generate_test_candidates`
- Do not rebuild the blocking pool
- Preserve the candidate file already produced by your friend at `processing/blocking/candidate_pairs.tsv`

The remaining workflow is:
1. Validate the existing Stage 2 train/validation split artifacts
2. Use the friend's `candidate_pairs.tsv` with `src/prepare_training_data.py`
3. Inspect labeled train/validation outputs and blocking recall diagnostics
4. Stop there unless a matching script already exists in this repo

## Required folder layout

```text
Amazon ML Challenge/
├── cleaned_dataset/
│   ├── cleaned_source1.tsv
│   ├── cleaned_source2.tsv
│   └── cleaned_source3.tsv
├── dataset/
│   └── train/
├── logs/
│   ├── prepare_training_data.log
│   ├── prepare_training_data_report.json
│   └── validation_split.log
├── processing/
│   ├── blocking/
│   │   └── candidate_pairs.tsv        # expected Stage 3 input from your friend
│   ├── splits/
│   │   ├── train_ground_truth.tsv
│   │   ├── val_ground_truth.tsv
│   │   ├── train_source1_ids.txt
│   │   ├── val_source1_ids.txt
│   │   └── split_summary.json
│   └── training_data/
│       ├── train_pairs.tsv
│       └── val_pairs.tsv
├── src/
│   ├── config.py
│   ├── prepare_training_data.py
│   ├── validation_split.py
│   └── ...
├── README.md
└── output/
```

## 1) Safe cleanup: keep the friend's candidate TSV and remove only blocking intermediates

Run this before using the friend's file if you need to clean out any partial/blocking artifacts.

```powershell
cd "C:\Amazon ML Challenge"

# 1. Stop any blocking or pool-building Python process that may still be running
Get-Process python, py -ErrorAction SilentlyContinue | Stop-Process -Force

# 2. Back up the friend's candidate TSV outside processing/blocking
$backupDir = "C:\Amazon ML Challenge\backup"
New-Item -ItemType Directory -Force -Path $backupDir | Out-Null
$backupPath = Join-Path $backupDir ("candidate_pairs_friend_" + (Get-Date -Format yyyyMMdd_HHmmss) + ".tsv")
Copy-Item "processing\blocking\candidate_pairs.tsv" $backupPath -Force

# 3. Delete all other contents inside processing/blocking while keeping the directory itself
Get-ChildItem "processing\blocking" -Force | Where-Object {
    $_.FullName -ne (Resolve-Path "processing\blocking\candidate_pairs.tsv").Path
} | Remove-Item -Recurse -Force

# 4. Restore the candidate TSV to the expected path
Copy-Item $backupPath "processing\blocking\candidate_pairs.tsv" -Force

# 5. Final sanity check
Get-ChildItem "processing\blocking" -Force
```

Important safety rules:
- Never delete `cleaned_dataset/`, `processing/splits/`, `src/`, or `processing/blocking/candidate_pairs.tsv`
- Only delete other files/folders within `processing/blocking`
- Keep the directory itself; do not remove `processing/blocking`

## 2) Verify the friend's candidate TSV before running Stage 3

The Stage 3 script expects a TSV with these columns:
- `source1_entity_id`
- `candidate_entity_ids`

A blank `candidate_entity_ids` value means zero candidates for that Source1 entity.

```powershell
cd "C:\Amazon ML Challenge"

# Header row (must include source1_entity_id and candidate_entity_ids)
Get-Content "processing\blocking\candidate_pairs.tsv" -TotalCount 1

# File size (safe: no full-file load)
Get-Item "processing\blocking\candidate_pairs.tsv" | Select-Object Name, @{Name='Bytes';Expression={$_.Length}}

# First few rows (safe: only first 5 lines)
Get-Content "processing\blocking\candidate_pairs.tsv" -TotalCount 5

# Confirm the candidate IDs still carry their original S2-/S3- prefixes
Get-Content "processing\blocking\candidate_pairs.tsv" -TotalCount 20 | Select-String 'S2-|S3-'

# Confirm blank candidate lists are represented correctly (blank after the tab)
Get-Content "processing\blocking\candidate_pairs.tsv" -TotalCount 20 | Select-String '^[^\t]+\t$'
```

A good output should show:
- the header includes `source1_entity_id` and `candidate_entity_ids`
- candidate IDs look like `S2-12345` or `S3-98765`
- blank candidate lists stay blank instead of becoming malformed rows

## 3) Validate the existing Stage 2 split artifacts

This script validates the train ground-truth and produces the train/validation split outputs used by the labeling stage.

```powershell
cd "C:\Amazon ML Challenge"

python -m src.validation_split --mode train --chunk-size 50000
```

Expected outputs:
- `processing/splits/train_source1_ids.txt`
- `processing/splits/val_source1_ids.txt`
- `processing/splits/train_ground_truth.tsv`
- `processing/splits/val_ground_truth.tsv`
- `processing/splits/split_summary.json`

Quick checks:

```powershell
Get-Item "processing\splits\train_ground_truth.tsv" | Select-Object Name, @{Name='Bytes';Expression={$_.Length}}
Get-Content "processing\splits\train_ground_truth.tsv" -TotalCount 3
Get-Content "processing\splits\val_ground_truth.tsv" -TotalCount 3
Get-Content "processing\splits\split_summary.json" -TotalCount 30
```

## 4) Prepare training and validation labeled pairs from the friend's candidate TSV

This is the actual downstream Stage 3 command using the existing candidate TSV and the validated Stage 2 split files.

```powershell
cd "C:\Amazon ML Challenge"

python -m src.prepare_training_data `
  --candidate-path "processing\blocking\candidate_pairs.tsv" `
  --split-dir "processing\splits" `
  --output-dir "processing\training_data" `
  --chunk-size 25000
```

The script writes:
- `processing/training_data/train_pairs.tsv`
- `processing/training_data/val_pairs.tsv`
- `logs/prepare_training_data_report.json`
- `logs/prepare_training_data.log`

Inspect the outputs:

```powershell
Get-Item "processing\training_data\train_pairs.tsv" | Select-Object Name, @{Name='Bytes';Expression={$_.Length}}
Get-Item "processing\training_data\val_pairs.tsv" | Select-Object Name, @{Name='Bytes';Expression={$_.Length}}
Get-Content "processing\training_data\train_pairs.tsv" -TotalCount 5
Get-Content "processing\training_data\val_pairs.tsv" -TotalCount 5
Get-Content "logs\prepare_training_data_report.json" -TotalCount 80
```

## 5) Blocking recall and zero-candidate diagnostics

The Stage 3 script reports blocking recall and zero-candidate statistics in `logs/prepare_training_data_report.json` when it processes the candidate file.

Examples:

```powershell
Select-String -Path "logs\prepare_training_data_report.json" -Pattern 'blocking_recall|zero_candidate_entities|recall|matched_but_missed'
```

This includes:
- train blocking recall
- validation blocking recall
- zero-candidate singleton counts
- zero-candidate matched entities that are missed by the candidate file

These are diagnostics from the existing implementation; they are not a model training step.

## 6) What is not implemented in this repo

The repository does not contain a downstream training script, a final validation script, or a final test-submission prediction script beyond the Stage 3 labeling step.

Specifically, no matching/training pipeline or final submission generator was found for:
- model training
- validation metrics beyond the Stage 3 report
- final test prediction generation
- Kaggle-style submission file creation

If those scripts exist elsewhere, they must be added explicitly; they are not part of the current checked-in repo.

## 7) Full command order to copy-paste

```powershell
cd "C:\Amazon ML Challenge"

# Optional: stop any existing blocking/pool job
Get-Process python, py -ErrorAction SilentlyContinue | Stop-Process -Force

# Back up the friend candidate file outside processing/blocking
$backupDir = "C:\Amazon ML Challenge\backup"
New-Item -ItemType Directory -Force -Path $backupDir | Out-Null
$backupPath = Join-Path $backupDir ("candidate_pairs_friend_" + (Get-Date -Format yyyyMMdd_HHmmss) + ".tsv")
Copy-Item "processing\blocking\candidate_pairs.tsv" $backupPath -Force

# Validate the friend's TSV before proceeding
Get-Content "processing\blocking\candidate_pairs.tsv" -TotalCount 1
Get-Item "processing\blocking\candidate_pairs.tsv" | Select-Object Name, @{Name='Bytes';Expression={$_.Length}}
Get-Content "processing\blocking\candidate_pairs.tsv" -TotalCount 5

# Stage 2: validate splits
python -m src.validation_split --mode train --chunk-size 50000

# Stage 3: label candidates using the existing friend TSV
python -m src.prepare_training_data `
  --candidate-path "processing\blocking\candidate_pairs.tsv" `
  --split-dir "processing\splits" `
  --output-dir "processing\training_data" `
  --chunk-size 25000

# Inspect outputs
Get-Content "processing\training_data\train_pairs.tsv" -TotalCount 5
Get-Content "processing\training_data\val_pairs.tsv" -TotalCount 5
Get-Content "logs\prepare_training_data_report.json" -TotalCount 80
```

This is the safe, memory-conscious workflow that uses the existing friend-produced candidate TSV and skips the blocking pool build entirely.