"""
validation_split.py — Stage 2: Ground-truth validation and stratified split.

Reads the full train_ground_truth.tsv and train_source1.tsv (in chunks),
validates ground-truth integrity, then produces a reproducible stratified
train/validation split without duplicating the large source2/source3 files.

Outputs (written to output/splits/):
  train_source1_ids.txt       — one entity_id per line for train split
  val_source1_ids.txt         — one entity_id per line for val split
  train_ground_truth.tsv      — label rows for train Source1 entities only
  val_ground_truth.tsv        — label rows for val Source1 entities only
  split_summary.json          — full split statistics

IMPORTANT: Source2/Source3 files are NOT duplicated.  Downstream code loads
  cleaned_source{2,3}.tsv once and filters by the ID lists as needed.

Stratification:
  Strata = country_norm × row_type (matched | singleton).
  Malformed and missing-reference rows are excluded from splitting (logged).
  Strata too small (< MIN_STRATUM_SIZE) → fallback: all rows folded into train.

Country note:
  France is absent from train; it cannot be stratified here.  This is logged
  explicitly.  Validation metrics on US/India do NOT approximate France performance.

Usage
-----
    python -m src.validation_split
    python -m src.validation_split --val-fraction 0.20 --seed 123
    python -m src.validation_split --gt-path /custom/ground_truth.tsv
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Iterator

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

import config as cfg


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def _setup_logger(name: str, log_path: Path) -> logging.Logger:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    if not logger.handlers:
        fh = logging.FileHandler(log_path, encoding="utf-8")
        fh.setLevel(logging.DEBUG)
        ch = logging.StreamHandler(sys.stdout)
        ch.setLevel(logging.INFO)
        fmt = logging.Formatter("%(asctime)s  %(levelname)-8s  %(message)s",
                                datefmt="%Y-%m-%d %H:%M:%S")
        fh.setFormatter(fmt)
        ch.setFormatter(fmt)
        logger.addHandler(fh)
        logger.addHandler(ch)
    return logger


# ---------------------------------------------------------------------------
# Ground-truth parsing constants
# ---------------------------------------------------------------------------

_RE_VALID_ID = re.compile(cfg.VALID_MATCHED_ID_PATTERN)

# Row-type labels
TYPE_SINGLETON = "singleton"
TYPE_MATCHED = "matched"
TYPE_MALFORMED = "malformed"

# Fallback behaviour for small strata (documented here, referenced below)
SMALL_STRATUM_FALLBACK = "fold_into_train"
# Rule: if a stratum has fewer than MIN_STRATUM_SIZE rows, all of its rows are
# assigned to train.  Rationale: splitting a very small stratum would produce
# a validation set too small to be meaningful and would risk exposing the val
# entities to near-certain identity leakage in the matched set.


# ---------------------------------------------------------------------------
# Step 1: Load Source1 entity_ids (chunked) — build lookup set
# ---------------------------------------------------------------------------

def load_source1_entity_ids(
    source1_path: Path,
    chunk_size: int,
    logger: logging.Logger,
) -> tuple[set[str], dict[str, str]]:
    """
    Stream train_source1.tsv and return:
      entity_ids : set[str]   — all unique entity_ids
      id_to_country : dict    — entity_id → country (for stratification)

    Duplicates within source1 are logged but not dropped.  The first-seen
    country is used for stratification when duplicates exist.
    """
    logger.info(f"Loading Source1 entity IDs from: {source1_path}")
    entity_ids: set[str] = set()
    id_to_country: dict[str, str] = {}
    dup_count = 0
    total = 0

    for chunk in _iter_tsv_chunks(source1_path, chunk_size):
        total += len(chunk)
        sub = chunk[["entity_id", "country"]].copy()
        sub["entity_id"] = sub["entity_id"].str.strip()
        sub["country"] = sub["country"].str.strip().str.title().fillna("__unknown__")
        # Drop rows with null/empty entity_id
        sub = sub[sub["entity_id"].notna() & (sub["entity_id"] != "")]
        # Detect duplicates relative to already-seen set
        dup_mask = sub["entity_id"].isin(entity_ids)
        dup_count += int(dup_mask.sum())
        # New IDs only (deduplicate within chunk too, keeping first occurrence)
        new_rows = sub[~dup_mask].drop_duplicates(subset="entity_id", keep="first")
        # Vectorized dict update — no iterrows
        entity_ids.update(new_rows["entity_id"].tolist())
        id_to_country.update(
            new_rows.set_index("entity_id")["country"].to_dict()
        )

    if dup_count:
        logger.warning(
            f"Source1 has {dup_count} duplicate entity_id rows. "
            f"First-seen country retained for stratification."
        )
    logger.info(f"Source1 total rows: {total:,} | Unique entity_ids: {len(entity_ids):,}")
    return entity_ids, id_to_country


# ---------------------------------------------------------------------------
# Step 2: Parse and validate ground truth (single full load — it's label-only)
# ---------------------------------------------------------------------------

def parse_ground_truth(
    gt_path: Path,
    source1_ids: set[str],
    logger: logging.Logger,
) -> pd.DataFrame:
    """
    Load the full ground-truth file and classify each row as:
      TYPE_SINGLETON  — matched_entity_ids is null/whitespace-only
      TYPE_MATCHED    — one or more well-formed S2-/S3- IDs (comma-separated)
      TYPE_MALFORMED  — parse failure (stray commas, bad prefix, etc.)

    Also checks for source1_entity_ids missing from source1_ids.

    Returns a DataFrame with columns:
      source1_entity_id, matched_entity_ids, row_type,
      missing_from_source1 (bool), parsed_ids (list[str] | None)
    """
    logger.info(f"Parsing ground truth: {gt_path}")
    gt = pd.read_csv(
        gt_path,
        sep="\t",
        dtype=str,
        keep_default_na=False,
        na_values=["", "NULL", "null", "NA", "N/A", "\\N"],
        encoding="utf-8",
        encoding_errors="replace",
    )
    logger.info(f"Ground truth loaded: {len(gt):,} rows")

    # --- Singleton detection ---
    # A row is singleton iff matched_entity_ids is null/empty/whitespace.
    singleton_mask = gt["matched_entity_ids"].isna() | (
        gt["matched_entity_ids"].str.strip() == ""
    )

    # --- Malformed detection (vectorized) ---
    # For non-singleton rows, check each token:
    #   (a) splitting on comma must not produce empty tokens (stray commas)
    #   (b) each token must match _RE_VALID_ID pattern

    def classify_matched_ids(cell: str) -> tuple[str, list[str] | None]:
        """
        Returns (row_type, parsed_ids_or_None).
        Called via vectorized path below — see note.
        """
        # NOTE: This function IS called via .apply() — the only .apply() in the
        # module.  It is necessary because malformed detection requires per-cell
        # parsing logic that involves splitting on comma and checking each token
        # individually.  There is no pandas built-in for this.  For
        # vectorization, one could use str.split().explode().str.match()
        # with groupby, which we do here instead.
        tokens = [t.strip() for t in cell.split(",")]
        # Check for empty tokens (stray leading/trailing/doubled commas)
        if any(t == "" for t in tokens):
            return TYPE_MALFORMED, None
        # Check each token against the valid ID pattern
        if not all(_RE_VALID_ID.match(t) for t in tokens):
            return TYPE_MALFORMED, None
        return TYPE_MATCHED, tokens

    # Vectorized classification for non-singleton rows
    # Step A: use str.split and explode to check all tokens at once
    non_singleton_idx = gt.index[~singleton_mask]
    non_singleton_vals = gt.loc[non_singleton_idx, "matched_entity_ids"]

    # Detect stray commas (empty tokens after split)
    has_stray_comma = (
        non_singleton_vals
        .str.strip()
        .str.split(",")
        .apply(lambda tokens: any(t.strip() == "" for t in tokens))
    )

    # Detect invalid token format
    # Split, explode, check pattern, group back
    split_series = non_singleton_vals.str.split(",")
    exploded = split_series.explode().str.strip()
    valid_token = exploded.str.match(cfg.VALID_MATCHED_ID_PATTERN, na=False)
    # A row is invalid if ANY of its tokens fail
    invalid_token_rows = ~valid_token.groupby(level=0).all()

    malformed_mask_non_singleton = has_stray_comma | invalid_token_rows

    # Build row_type series
    row_type = pd.Series(TYPE_SINGLETON, index=gt.index)
    # Set matched for non-singleton, non-malformed
    matched_idx = non_singleton_idx[~malformed_mask_non_singleton]
    row_type.loc[matched_idx] = TYPE_MATCHED
    # Set malformed
    malformed_idx = non_singleton_idx[malformed_mask_non_singleton]
    row_type.loc[malformed_idx] = TYPE_MALFORMED

    gt["row_type"] = row_type

    # --- Missing reference check ---
    gt["missing_from_source1"] = ~gt["source1_entity_id"].isin(source1_ids)

    # --- Parsed IDs for matched rows ---
    # Store as comma-separated string (not Python list) to keep DataFrame clean.
    gt["parsed_ids"] = pd.NA
    gt.loc[matched_idx, "parsed_ids"] = (
        gt.loc[matched_idx, "matched_entity_ids"].str.strip()
    )

    # --- Log summary ---
    n_singleton = (gt["row_type"] == TYPE_SINGLETON).sum()
    n_matched = (gt["row_type"] == TYPE_MATCHED).sum()
    n_malformed = (gt["row_type"] == TYPE_MALFORMED).sum()
    n_missing = gt["missing_from_source1"].sum()

    logger.info(f"  Singletons:      {n_singleton:,}")
    logger.info(f"  Matched:         {n_matched:,}")
    logger.info(f"  Malformed:       {n_malformed:,}  ← excluded from split")
    logger.info(f"  Missing from S1: {n_missing:,}  ← excluded from split")

    if n_malformed > 0:
        malformed_examples = gt[gt["row_type"] == TYPE_MALFORMED]["source1_entity_id"].head(10).tolist()
        logger.warning(
            f"Malformed ground-truth rows ({n_malformed}). "
            f"Example source1_entity_ids: {malformed_examples}. "
            f"These are logged but excluded from the split — neither train nor val."
        )
    if n_missing > 0:
        missing_examples = gt[gt["missing_from_source1"]]["source1_entity_id"].head(10).tolist()
        logger.warning(
            f"Ground-truth rows whose source1_entity_id is absent from train_source1: {n_missing}. "
            f"Examples: {missing_examples}. "
            f"Decision: excluded from split (cannot stratify by country without a source record)."
        )

    return gt


# ---------------------------------------------------------------------------
# Step 3: Build strata and split
# ---------------------------------------------------------------------------

def build_strata(
    gt: pd.DataFrame,
    id_to_country: dict[str, str],
) -> pd.DataFrame:
    """
    Attach country and build stratum key (country_norm × row_type).
    Excludes malformed and missing-reference rows.
    Returns only singleton and matched rows with valid source1 reference.
    """
    # Filter to only splittable rows
    splittable = gt[
        (gt["row_type"].isin([TYPE_SINGLETON, TYPE_MATCHED]))
        & (~gt["missing_from_source1"])
    ].copy()

    # Attach country (from source1)
    splittable["country_norm"] = (
        splittable["source1_entity_id"]
        .map(id_to_country)
        .fillna("__unknown__")
    )

    # Build stratum key
    splittable["stratum"] = (
        splittable["country_norm"].str.lower().str.strip()
        + "::"
        + splittable["row_type"]
    )

    return splittable


def stratified_split(
    splittable: pd.DataFrame,
    val_fraction: float,
    seed: int,
    min_stratum_size: int,
    logger: logging.Logger,
) -> tuple[pd.DataFrame, pd.DataFrame, list[dict]]:
    """
    Perform a reproducible stratified split of Source1 entity IDs.

    For each stratum (country × row_type):
      - If stratum size >= min_stratum_size: sample val_fraction for val.
      - If stratum size < min_stratum_size: apply SMALL_STRATUM_FALLBACK
        (fold entirely into train).

    Returns
    -------
    train_df : rows assigned to train
    val_df   : rows assigned to val
    strata_log : list of per-stratum statistics dicts
    """
    train_parts: list[pd.DataFrame] = []
    val_parts: list[pd.DataFrame] = []
    strata_log: list[dict] = []

    # Log France-related note
    known_train_countries = set(splittable["country_norm"].str.lower().unique())
    logger.info(
        f"Countries present in ground truth: {known_train_countries}. "
        f"Note: France (and other test-only countries) are absent from train and "
        f"CANNOT be stratified in this split. Validation metrics therefore cannot "
        f"directly estimate pipeline performance on France or unseen countries."
    )

    rng = pd.Series(range(len(splittable)), index=splittable.index)  # placeholder for seed

    for stratum_key, group in splittable.groupby("stratum"):
        n = len(group)
        n_val = int(round(n * val_fraction))
        stratum_info: dict = {
            "stratum": stratum_key,
            "total": n,
            "n_val": n_val,
            "n_train": n - n_val,
            "fallback": False,
            "fallback_reason": None,
        }

        if n < min_stratum_size:
            # Small stratum fallback — fold into train
            train_parts.append(group)
            stratum_info["n_val"] = 0
            stratum_info["n_train"] = n
            stratum_info["fallback"] = True
            stratum_info["fallback_reason"] = (
                f"Stratum size {n} < min_stratum_size={min_stratum_size}; "
                f"rule='{SMALL_STRATUM_FALLBACK}': all rows folded into train."
            )
            logger.warning(
                f"Stratum '{stratum_key}' has {n} rows < min={min_stratum_size}. "
                f"Fallback: all {n} rows → train."
            )
        elif n_val == 0:
            # val_fraction too small to yield even 1 row
            train_parts.append(group)
            stratum_info["n_val"] = 0
            stratum_info["n_train"] = n
            stratum_info["fallback"] = True
            stratum_info["fallback_reason"] = (
                f"val_fraction={val_fraction} yields 0 val rows for stratum size {n}; "
                f"all rows → train."
            )
            logger.warning(
                f"Stratum '{stratum_key}': val_fraction yields 0 val rows. "
                f"All {n} rows → train."
            )
        else:
            val_sample = group.sample(n=n_val, random_state=seed)
            train_sample = group.drop(index=val_sample.index)
            val_parts.append(val_sample)
            train_parts.append(train_sample)

        strata_log.append(stratum_info)

    train_df = pd.concat(train_parts, ignore_index=True) if train_parts else pd.DataFrame()
    val_df = pd.concat(val_parts, ignore_index=True) if val_parts else pd.DataFrame()
    return train_df, val_df, strata_log


# ---------------------------------------------------------------------------
# Step 4: Persist outputs
# ---------------------------------------------------------------------------

def write_id_list(ids: pd.Series, path: Path) -> None:
    """Write one entity_id per line."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(ids.tolist()), encoding="utf-8")


def filter_and_write_ground_truth(
    gt: pd.DataFrame,
    source1_ids: pd.Series,
    output_path: Path,
) -> None:
    """
    Filter gt to only rows in source1_ids, write original columns only
    (source1_entity_id, matched_entity_ids).
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    filtered = gt[gt["source1_entity_id"].isin(set(source1_ids.tolist()))]
    filtered[["source1_entity_id", "matched_entity_ids"]].to_csv(
        output_path, sep="\t", index=False, encoding="utf-8"
    )


# ---------------------------------------------------------------------------
# Step 5: Summary report
# ---------------------------------------------------------------------------

def build_summary(
    gt: pd.DataFrame,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    strata_log: list[dict],
) -> dict:
    """Build the JSON-serializable summary dict."""

    def country_dist(df: pd.DataFrame) -> dict[str, int]:
        if df.empty or "country_norm" not in df.columns:
            return {}
        return df["country_norm"].value_counts(dropna=False).to_dict()

    def type_dist(df: pd.DataFrame) -> dict[str, int]:
        if df.empty or "row_type" not in df.columns:
            return {}
        return df["row_type"].value_counts(dropna=False).to_dict()

    n_malformed = int((gt["row_type"] == TYPE_MALFORMED).sum())
    n_missing = int(gt["missing_from_source1"].sum())
    fallback_strata = [s for s in strata_log if s["fallback"]]

    return {
        "train_rows": int(len(train_df)),
        "val_rows": int(len(val_df)),
        "total_splittable": int(len(train_df) + len(val_df)),
        "malformed_gt_rows": n_malformed,
        "missing_reference_gt_rows": n_missing,
        "train_country_distribution": {str(k): int(v) for k, v in country_dist(train_df).items()},
        "val_country_distribution": {str(k): int(v) for k, v in country_dist(val_df).items()},
        "train_row_type_distribution": {str(k): int(v) for k, v in type_dist(train_df).items()},
        "val_row_type_distribution": {str(k): int(v) for k, v in type_dist(val_df).items()},
        "strata": strata_log,
        "fallback_strata_count": len(fallback_strata),
        "fallback_strata": fallback_strata,
        "limitation_note": (
            "Validation covers only countries present in training (US, India). "
            "France and other test-only countries are absent from this split. "
            "Validation metrics on US/India cannot approximate performance on France "
            "or any other unseen country. Country stratification here does NOT resolve "
            "or close this generalization gap."
        ),
        "source2_source3_note": (
            "Source2 and Source3 files are NOT duplicated per split. "
            "Downstream matching code must load the full cleaned_source{2,3}.tsv "
            "and filter/join against train_source1_ids.txt or val_source1_ids.txt "
            "at runtime. The same S2/S3 record appearing as a candidate for both "
            "train and val Source1 entities is NOT leakage — it mirrors the real "
            "matching problem where candidates are shared across queries."
        ),
    }


# ---------------------------------------------------------------------------
# Chunked TSV iterator (shared utility)
# ---------------------------------------------------------------------------

def _iter_tsv_chunks(path: Path, chunk_size: int) -> Iterator[pd.DataFrame]:
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    reader = pd.read_csv(
        path,
        sep="\t",
        chunksize=chunk_size,
        dtype=str,
        keep_default_na=False,
        na_values=["", "NULL", "null", "NA", "N/A", "\\N"],
        encoding="utf-8",
        encoding_errors="replace",
    )
    yield from reader


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Stage 2: Ground-truth validation and stratified train/val split."
    )
    p.add_argument(
        "--mode",
        choices=(cfg.TRAIN_MODE, cfg.TEST_MODE),
        default=cfg.TRAIN_MODE,
        help="Validation splitting is only supported for the training dataset; test data has no ground truth.",
    )
    p.add_argument("--val-fraction", type=float, default=cfg.VAL_FRACTION)
    p.add_argument("--seed", type=int, default=cfg.RANDOM_SEED)
    p.add_argument("--chunk-size", type=int, default=cfg.CHUNK_SIZE)
    p.add_argument("--min-stratum-size", type=int, default=cfg.MIN_STRATUM_SIZE)
    p.add_argument("--source1-path", type=Path, default=cfg.TRAIN_SOURCE1)
    p.add_argument("--gt-path", type=Path, default=cfg.TRAIN_GROUND_TRUTH)
    p.add_argument("--output-dir", type=Path, default=cfg.SPLIT_DIR)
    return p.parse_args()


def main() -> None:
    args = parse_args()

    if args.mode != cfg.TRAIN_MODE:
        raise SystemExit(
            "Validation splitting is only supported in training mode because test data has no ground truth. "
            "Use `python -m src.data_cleaning --mode test` for test cleaning instead."
        )

    log_path = cfg.SPLIT_LOG_PATH
    logger = _setup_logger("validation_split", log_path)

    logger.info("=" * 60)
    logger.info("Validation Split Stage — START")
    logger.info(f"  mode:              {args.mode}")
    logger.info(f"  val_fraction:      {args.val_fraction}")
    logger.info(f"  seed:              {args.seed}")
    logger.info(f"  chunk_size:        {args.chunk_size:,}")
    logger.info(f"  min_stratum_size:  {args.min_stratum_size}")
    logger.info(f"  source1_path:      {args.source1_path}")
    logger.info(f"  gt_path:           {args.gt_path}")
    logger.info(f"  output_dir:        {args.output_dir}")
    logger.info("=" * 60)

    out_dir: Path = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()

    # Step 1: Load Source1 entity IDs
    source1_ids, id_to_country = load_source1_entity_ids(
        args.source1_path, args.chunk_size, logger
    )

    # Step 2: Parse and validate ground truth
    gt = parse_ground_truth(args.gt_path, source1_ids, logger)

    # Step 3: Build strata
    splittable = build_strata(gt, id_to_country)
    logger.info(f"Splittable rows (singleton + matched, with valid S1 ref): {len(splittable):,}")

    # Step 4: Stratified split
    train_df, val_df, strata_log = stratified_split(
        splittable,
        val_fraction=args.val_fraction,
        seed=args.seed,
        min_stratum_size=args.min_stratum_size,
        logger=logger,
    )
    logger.info(f"Train rows: {len(train_df):,} | Val rows: {len(val_df):,}")

    # Step 5: Persist ID lists
    train_id_path = out_dir / "train_source1_ids.txt"
    val_id_path = out_dir / "val_source1_ids.txt"
    write_id_list(train_df["source1_entity_id"], train_id_path)
    write_id_list(val_df["source1_entity_id"], val_id_path)
    logger.info(f"ID lists written: {train_id_path}, {val_id_path}")

    # Step 6: Filter and write ground-truth subsets
    train_gt_path = out_dir / "train_ground_truth.tsv"
    val_gt_path = out_dir / "val_ground_truth.tsv"
    filter_and_write_ground_truth(gt, train_df["source1_entity_id"], train_gt_path)
    filter_and_write_ground_truth(gt, val_df["source1_entity_id"], val_gt_path)
    logger.info(f"GT subsets written: {train_gt_path}, {val_gt_path}")

    # Step 7: Summary report
    summary = build_summary(gt, train_df, val_df, strata_log)
    summary_path = out_dir / "split_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    logger.info(f"Split summary written: {summary_path}")

    # Step 8: Human-readable log summary
    elapsed = time.time() - t0
    logger.info("=" * 60)
    logger.info("Validation Split Stage — COMPLETE")
    logger.info(f"  Total elapsed: {elapsed:.1f}s")
    logger.info(f"  Train GT rows: {summary['train_rows']:,}")
    logger.info(f"  Val GT rows:   {summary['val_rows']:,}")
    logger.info(f"  Malformed GT:  {summary['malformed_gt_rows']:,}")
    logger.info(f"  Missing refs:  {summary['missing_reference_gt_rows']:,}")
    logger.info(f"  Fallback strata triggered: {summary['fallback_strata_count']}")
    if summary["fallback_strata"]:
        for fs in summary["fallback_strata"]:
            logger.info(f"    → {fs['stratum']}: {fs['fallback_reason']}")
    logger.info("=" * 60)
    logger.info(
        "LIMITATION: "
        + summary["limitation_note"]
    )
    logger.info(
        "S2/S3 NOTE: "
        + summary["source2_source3_note"]
    )


if __name__ == "__main__":
    main()
