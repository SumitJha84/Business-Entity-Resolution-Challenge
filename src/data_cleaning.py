"""
data_cleaning.py — Stage 1: Chunk-wise data cleaning and normalization.

Processes train_source1.tsv, train_source2.tsv, train_source3.tsv (and their
test equivalents) without loading any single source file fully into memory.

Produces:
  - cleaned_source{1,2,3}.tsv with original columns preserved plus:
      business_name_norm      conservative normalised name (lowercase, punct, & → and, suffix-normed)
      business_name_core      suffix-stripped experimental variant
      business_name_alt       DBA/trade-name when detectable, else null
      business_address_norm   abbreviation-expanded, country-conditional
      business_address_landmark  extracted landmark phrase when detectable, else null
      country_norm            casing/whitespace standardised country, never remapped

  - Per-source data-quality JSON report in logs/data_quality/

Design notes
------------
* All transforms are vectorized (pandas/str/regex).  No per-row Python loops
  or .apply() with scalar Python functions.
* country only selects which abbreviation dictionary to try first; a generic
  fallback is always applied afterward.  Unknown countries (e.g., France) use
  the generic dict directly — no crash.
* Normalization is conservative: ambiguous transforms produce extra columns
  rather than overwriting the less-aggressive version.
* entity_id is never modified; duplicates/nulls are flagged and logged, not dropped.
* No external API calls, no internet, no third-party ER services.

Usage
-----
    python -m src.data_cleaning               # uses config defaults
    python -m src.data_cleaning --chunk-size 25000
    python -m src.data_cleaning --source1-only
    python -m src.data_cleaning --include-test   # also cleans test sources
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

# Make sure we can import config regardless of working directory.
sys.path.insert(0, str(Path(__file__).resolve().parent))

import config as cfg


# ---------------------------------------------------------------------------
# Logging setup
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
# Pre-compiled regex objects (module-level — compiled once)
# ---------------------------------------------------------------------------

# Punctuation noise: characters to collapse to a single space.
# We deliberately keep hyphens (brand names like "Wal-Mart") and apostrophes
# ("McDonald's") as they carry identity signal.  Brackets, slashes, underscores
# are collapsed.
_RE_PUNCT_NOISE = re.compile(r"[/\\|_\[\]{}<>]+")

# Multiple whitespace → single space
_RE_MULTI_WS = re.compile(r"\s{2,}")

# Ampersand with surrounding whitespace → " and "
_RE_AMP = re.compile(r"\s*&\s*")

# Legal suffix pattern (built dynamically from config, longest-first)
def _build_suffix_regex() -> re.Pattern[str]:
    """
    Build a regex that matches legal suffix tokens at word boundaries.
    Longest token alternatives are placed first to avoid partial shadowing.
    NOTE: Not called per row — called once at module load.
    """
    tokens = sorted(cfg.LEGAL_SUFFIX_MAP.keys(), key=len, reverse=True)
    # Escape each token for regex safety, then join
    escaped = [re.escape(t) for t in tokens]
    # Match as whole words, case-insensitive
    pattern = r"(?i)\b(" + "|".join(escaped) + r")\.?\b"
    return re.compile(pattern)


_RE_SUFFIX = _build_suffix_regex()

# Suffix canonical set (lowercased); used for suffix-stripping in _core
_SUFFIX_CANONICAL = cfg.SUFFIX_CANONICAL_TOKENS

# DBA patterns: compiled list
_RE_DBA_PATTERNS = [re.compile(p) for p in cfg.DBA_PATTERNS]

# Landmark patterns: compiled list
_RE_LANDMARK_PATTERNS = [re.compile(p) for p in cfg.LANDMARK_PATTERNS]

# Trailing comma / period / whitespace cleanup
_RE_TRAILING_JUNK = re.compile(r"[\s,\.]+$")
_RE_LEADING_JUNK = re.compile(r"^[\s,\.]+")

# House number / address prefix normalisations for India
_RE_INDIA_HNO = re.compile(r"(?i)\bh\.?\s*no\.?\s*", flags=re.IGNORECASE)

# PIN code in India addresses (6 digits standalone or prefixed by "PIN")
_RE_INDIA_PIN = re.compile(r"(?i)\bpin[\s\-:]*(\d{6})\b|\b(\d{6})\b")

# Near/Opp landmark extractor (used for landmark column)
_RE_LANDMARK_COMBINED = re.compile(
    r"(?i)\b(?:near|nr\.?|adj(?:acent)?(?:\s+to)?|opp(?:osite)?|behind|beside"
    r"|next\s+to|in\s+front\s+of|landmark\s*[:–\-])\s+[A-Za-z][\w\s,\.]{3,60}",
)


# ---------------------------------------------------------------------------
# Pure normalisation functions (vectorized — operate on pd.Series)
# ---------------------------------------------------------------------------

def normalize_country(series: pd.Series) -> pd.Series:
    """
    Produce country_norm: strip whitespace, title-case, no remapping/bucketing.
    Returns a Series of the same length.
    """
    return (
        series.fillna("")
        .str.strip()
        .str.title()           # "  india  " → "India", "US" → "Us" … see note below
        # Keep the result as-is; .title() on "US" → "Us" which is fine for grouping.
        # We never remap or bucket; two spellings remain two groups.
    )


def normalize_business_name(series: pd.Series) -> tuple[pd.Series, pd.Series, pd.Series]:
    """
    Vectorized name normalization.

    Returns
    -------
    name_norm : pd.Series
        Conservative normalisation: lowercase, punct noise collapsed,
        & → and, legal suffixes → canonical tokens (NOT removed).
    name_core : pd.Series
        Experimental: name_norm with canonical suffix tokens stripped.
        Kept as separate column — never used as sole identity signal.
    name_alt : pd.Series
        DBA / trade-name extracted when detectable, else pd.NA.
    """
    s = series.fillna("").str.strip()

    # Step 1: lowercase
    norm = s.str.lower()

    # Step 2: collapse punctuation noise (keep - and ')
    norm = norm.str.replace(_RE_PUNCT_NOISE, " ", regex=True)

    # Step 3: & → and  (only if config flag set)
    if cfg.AMPERSAND_NORM:
        norm = norm.str.replace(_RE_AMP, " and ", regex=True)

    # Step 4: normalise legal suffixes → canonical tokens
    # Chained vectorized str.replace, longest token first.
    # Boundary handling: we use a leading \b (word-start boundary) but a
    # trailing negative-lookahead (?![a-z0-9]) instead of \b, because tokens
    # like "pvt." end with a dot (non-word char) and \b after \. never matches.
    for raw_token, canonical in sorted(
        cfg.LEGAL_SUFFIX_MAP.items(), key=lambda kv: len(kv[0]), reverse=True
    ):
        escaped = re.escape(raw_token)
        # Leading \b anchors the start; trailing (?![a-z0-9]) prevents partial matches
        pattern = r"(?i)\b" + escaped + r"(?![a-z0-9])"
        norm = norm.str.replace(pattern, canonical, regex=True)

    # Step 5: collapse multiple whitespace
    norm = norm.str.replace(_RE_MULTI_WS, " ", regex=True).str.strip()

    # --- business_name_core: strip canonical suffix tokens ---
    # Build a regex that matches one or more canonical suffix tokens at the end.
    suffix_tokens_pat = "|".join(re.escape(t) for t in sorted(_SUFFIX_CANONICAL, key=len, reverse=True))
    re_suffix_strip = re.compile(
        r"(?i)\s+(" + suffix_tokens_pat + r")(\s+(" + suffix_tokens_pat + r"))*\s*$"
    )
    core = norm.str.replace(re_suffix_strip, "", regex=True).str.strip()
    # Safety: if stripping empties the string, fall back to norm
    empty_core = core.str.len() == 0
    core = core.where(~empty_core, norm)

    # --- business_name_alt: extract DBA / trade-names ---
    # We do a single vectorized str.extract for the first DBA pattern match.
    # If multiple DBA patterns, we try them in order and fill gaps.
    alt = pd.Series(pd.NA, index=series.index, dtype="object")
    for re_pat in _RE_DBA_PATTERNS:
        extracted = series.str.extract(re_pat, expand=False)
        if isinstance(extracted, pd.DataFrame):
            extracted = extracted.iloc[:, 0]
        # Fill only rows still NA
        new_hits = extracted.notna() & alt.isna()
        alt = alt.where(~new_hits, extracted.str.strip())

    # Preserve pd.NA where no alt found (not empty string)
    alt = alt.where(alt.notna() & (alt.str.strip().str.len() > 0), pd.NA)

    return norm, core, alt


def _apply_abbrev_dict(series: pd.Series, abbrev_dict: dict[str, str]) -> pd.Series:
    """
    Vectorized: replace abbreviated tokens with their expanded forms.
    Applies one entry at a time (longest first) using whole-word matching.
    Each replacement is fully vectorized via str.replace.

    Not a per-row loop — O(|abbrev_dict| * n_rows) vectorized str operations.
    """
    result = series.copy()
    for abbr, expansion in sorted(abbrev_dict.items(), key=lambda kv: len(kv[0]), reverse=True):
        pattern = r"(?i)\b" + re.escape(abbr) + r"\b"
        result = result.str.replace(pattern, expansion, regex=True)
    return result


def normalize_address(
    address_series: pd.Series,
    country_norm_series: pd.Series,
) -> tuple[pd.Series, pd.Series]:
    """
    Produce:
      address_norm     — abbreviation-expanded, whitespace-normalised address.
      address_landmark — extracted landmark phrase (pd.NA if not found).

    Country-conditional logic:
      1. Start from lowercased address string.
      2. Apply country-specific abbreviation dict (us / india) if country maps to one.
      3. Always apply generic dict as fallback (catches unknowns like France).
      4. Collapse whitespace, strip.

    Unknown countries (France, etc.) bypass step 2 entirely and only use
    generic dict — no crash, no skip.

    NOTE: address_series and country_norm_series must have aligned indexes.
    """
    addr = address_series.fillna("").str.strip().str.lower()

    # --- Country-conditional first pass ---
    # Build a boolean mask per known country group.
    # Unknown countries fall through to generic-only path.
    c_lower = country_norm_series.str.lower().fillna("")

    # Country groups for which we have specific dicts (keys in ABBREV_DICTS minus "generic")
    specific_countries = {k for k in cfg.ABBREV_DICTS if k != "generic"}

    # Process each country group separately using masks, then recombine.
    result = addr.copy()

    for country_key in specific_countries:
        mask = c_lower == country_key
        if mask.any():
            subset = result[mask]
            subset = _apply_abbrev_dict(subset, cfg.ABBREV_DICTS[country_key])
            result = result.where(~mask, subset)

    # Generic pass (all rows)
    result = _apply_abbrev_dict(result, cfg.ABBREV_DICTS["generic"])

    # Final whitespace cleanup
    result = (
        result.str.replace(_RE_PUNCT_NOISE, " ", regex=True)
              .str.replace(_RE_MULTI_WS, " ", regex=True)
              .str.strip()
    )

    # --- Landmark extraction ---
    landmark = series_extract_first_match(address_series, _RE_LANDMARK_COMBINED)

    return result, landmark


def series_extract_first_match(series: pd.Series, pattern: re.Pattern) -> pd.Series:
    """
    Vectorized: extract the first full match of `pattern` from each cell.
    Returns pd.NA where no match.

    Implementation note
    -------------------
    We use ``str.findall`` with the pre-compiled Pattern object (not the pattern
    string).  ``str.extract`` internally re-compiles the string and chokes on
    inline ``(?i)`` flags when they appear after a wrapping ``(`` group.
    ``str.findall`` accepts a compiled Pattern directly and avoids this issue.

    The result of findall is a list per cell; we take the first element.
    This is done via a vectorized .apply on the list-typed column — the only
    per-row Python call here, but it operates on already-matched lists (not on
    raw strings), so the expensive regex scan is still fully vectorized.
    """
    # str.findall accepts a compiled Pattern object — no re-compilation.
    found_lists = series.str.findall(pattern)
    # found_lists is a Series of lists; take first element or pd.NA
    result = found_lists.apply(lambda lst: lst[0].strip() if isinstance(lst, list) and lst else pd.NA)
    # Coerce empty strings to pd.NA
    result = result.where(result.notna() & (result.str.len() > 0), pd.NA)
    return result


# ---------------------------------------------------------------------------
# Data quality tracking (accumulates across chunks, per source)
# ---------------------------------------------------------------------------

class QualityAccumulator:
    """Accumulates per-chunk statistics; never stores full row data."""

    def __init__(self, source_name: str) -> None:
        self.source_name = source_name
        self.total_rows: int = 0
        self.null_counts: dict[str, int] = defaultdict(int)
        self.duplicate_entity_id_count: int = 0
        self.null_entity_id_count: int = 0
        self.country_dist: dict[str, int] = defaultdict(int)
        self.suffix_match_count: int = 0
        self.dba_match_count: int = 0
        self.landmark_match_count: int = 0
        self.seen_entity_ids: set[str] = set()   # held in memory across chunks
        # NOTE: This set is O(n_unique_entity_ids).  For 2.2 M rows this is
        # ~200 MB worst-case (string storage).  An alternative is a HLL sketch
        # for approximate counting, but exact duplicate detection requires the
        # full set.  Documented trade-off: memory vs correctness.

    def update(self, chunk: pd.DataFrame, source_cols: list[str]) -> None:
        self.total_rows += len(chunk)

        # Null rates per original column
        for col in source_cols:
            if col in chunk.columns:
                self.null_counts[col] += int(chunk[col].isna().sum())

        # entity_id validation
        null_ids = chunk["entity_id"].isna() | (chunk["entity_id"].str.strip() == "")
        self.null_entity_id_count += int(null_ids.sum())

        current_ids = set(chunk["entity_id"].dropna().str.strip().tolist())
        dups_in_chunk = current_ids & self.seen_entity_ids
        self.duplicate_entity_id_count += len(dups_in_chunk)
        self.seen_entity_ids |= current_ids

        # Country distribution
        if "country" in chunk.columns:
            for val, cnt in chunk["country"].value_counts(dropna=False).items():
                key = str(val) if pd.notna(val) else "__null__"
                self.country_dist[key] += int(cnt)

        # Derived column match counts (only if already added to chunk)
        if "business_name_norm" in chunk.columns:
            # Use a simplified non-capturing suffix pattern for contains check.
            # The full _RE_SUFFIX has capture groups which trigger a pandas warning
            # when used with str.contains. We build a simpler version here.
            _suffix_contains_pat = r"(?i)\b(?:corp|inc|ltd|pvt|llc|llp|lp|co|assoc|enterprises|enterprise|group)\b"
            self.suffix_match_count += int(
                chunk["business_name"].str.contains(_suffix_contains_pat, regex=True, na=False).sum()
            )
        if "business_name_alt" in chunk.columns:
            self.dba_match_count += int(chunk["business_name_alt"].notna().sum())
        if "business_address_landmark" in chunk.columns:
            self.landmark_match_count += int(chunk["business_address_landmark"].notna().sum())

    def to_dict(self) -> dict:
        return {
            "source": self.source_name,
            "total_rows": self.total_rows,
            "null_counts_original_cols": dict(self.null_counts),
            "null_entity_id_count": self.null_entity_id_count,
            "duplicate_entity_id_count": self.duplicate_entity_id_count,
            "country_distribution": dict(self.country_dist),
            "suffix_pattern_matches": self.suffix_match_count,
            "dba_pattern_matches": self.dba_match_count,
            "landmark_pattern_matches": self.landmark_match_count,
        }


# ---------------------------------------------------------------------------
# Core chunk-processing pipeline
# ---------------------------------------------------------------------------

SOURCE_COLS = ["entity_id", "business_name", "business_address", "country"]


def _iter_chunks(path: Path, chunk_size: int, limit: int | None = None) -> Iterator[pd.DataFrame]:
    """Yield TSV chunks from a file.  Raises FileNotFoundError if missing."""
    if not path.exists():
        raise FileNotFoundError(f"Source file not found: {path}")
    reader = pd.read_csv(
        path,
        sep="\t",
        chunksize=chunk_size,
        dtype=str,                  # keep everything as string; no auto-coercion
        keep_default_na=False,      # "" stays "", not NaN — we handle nulls explicitly
        na_values=["", "NULL", "null", "NA", "N/A", "n/a", "\\N"],
        encoding="utf-8",
        encoding_errors="replace",  # handle encoding glitches (e.g., non-UTF-8 Hindi chars)
    )
    rows_seen = 0
    for chunk in reader:
        if limit is not None and rows_seen >= limit:
            break
        remaining = None if limit is None else limit - rows_seen
        if remaining is not None and len(chunk) > remaining:
            chunk = chunk.iloc[:remaining].copy()
        # Ensure expected columns exist
        for col in SOURCE_COLS:
            if col not in chunk.columns:
                chunk[col] = pd.NA
        rows_seen += len(chunk)
        yield chunk


def clean_chunk(chunk: pd.DataFrame) -> pd.DataFrame:
    """
    Apply all normalisation steps to a single chunk.

    Returns the chunk with new columns appended.
    Original columns are NEVER modified.
    """
    df = chunk.copy()

    # --- country_norm ---
    df["country_norm"] = normalize_country(df["country"])

    # --- business_name columns ---
    df["business_name_norm"], df["business_name_core"], df["business_name_alt"] = (
        normalize_business_name(df["business_name"])
    )

    # --- business_address columns ---
    df["business_address_norm"], df["business_address_landmark"] = normalize_address(
        df["business_address"], df["country_norm"]
    )

    return df


def clean_source(
    input_path: Path,
    output_path: Path,
    source_label: str,
    chunk_size: int,
    logger: logging.Logger,
    limit: int | None = None,
) -> dict:
    """
    Read input_path in chunks, clean each chunk, write to output_path.
    Returns a quality-report dict.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    accumulator = QualityAccumulator(source_label)
    first_chunk = True
    t0 = time.time()

    logger.info(f"[{source_label}] Starting cleaning → {output_path}")
    if limit is not None:
        logger.info(f"[{source_label}] Total row limit: {limit:,} records")

    for chunk_idx, chunk in enumerate(_iter_chunks(input_path, chunk_size, limit=limit)):
        cleaned = clean_chunk(chunk)
        accumulator.update(cleaned, SOURCE_COLS)

        write_mode = "w" if first_chunk else "a"
        header = first_chunk
        cleaned.to_csv(
            output_path,
            sep="\t",
            index=False,
            header=header,
            mode=write_mode,
            encoding="utf-8",
        )
        first_chunk = False

        if chunk_idx % 10 == 0:
            elapsed = time.time() - t0
            logger.info(
                f"[{source_label}] Processed chunk {chunk_idx} "
                f"({accumulator.total_rows:,} rows so far, {elapsed:.1f}s)"
            )

        if limit is not None and accumulator.total_rows >= limit:
            logger.info(
                f"[{source_label}] Reached total row limit of {limit:,}; stopping cleanly."
            )
            break

    elapsed = time.time() - t0
    report = accumulator.to_dict()
    logger.info(
        f"[{source_label}] Done — {report['total_rows']:,} rows in {elapsed:.1f}s. "
        f"Null entity_ids: {report['null_entity_id_count']}, "
        f"Duplicate entity_ids: {report['duplicate_entity_id_count']}, "
        f"DBA matches: {report['dba_pattern_matches']}, "
        f"Landmark matches: {report['landmark_pattern_matches']}"
    )

    if report["null_entity_id_count"] > 0:
        logger.warning(
            f"[{source_label}] {report['null_entity_id_count']} null/empty entity_ids detected. "
            f"Rows kept (not dropped) — check quality report for details."
        )
    if report["duplicate_entity_id_count"] > 0:
        logger.warning(
            f"[{source_label}] {report['duplicate_entity_id_count']} duplicate entity_ids detected. "
            f"Rows kept (not dropped) — duplicates are flagged in quality report."
        )

    return report


# ---------------------------------------------------------------------------
# Quality report persistence
# ---------------------------------------------------------------------------

def save_quality_report(report: dict, source_label: str) -> None:
    """Write the quality report JSON to logs/data_quality/<source_label>.json."""
    report_dir = cfg.QUALITY_LOG_DIR
    report_dir.mkdir(parents=True, exist_ok=True)
    out_path = report_dir / f"{source_label}_quality_report.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Stage 1: Data cleaning and normalisation for ER pipeline."
    )
    p.add_argument(
        "--mode",
        choices=cfg.CLEANING_MODES,
        default=cfg.TRAIN_MODE,
        help="Dataset mode to clean: train or test.",
    )
    p.add_argument(
        "--chunk-size",
        type=int,
        default=cfg.CHUNK_SIZE,
        help=f"Rows per chunk (default: {cfg.CHUNK_SIZE})",
    )
    p.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Maximum total rows to process for each source file before stopping. "
             "Applies per source, not per chunk.",
    )
    p.add_argument(
        "--source1-only",
        action="store_true",
        help="Clean only the selected mode's Source1 file (faster iteration).",
    )
    p.add_argument(
        "--include-test",
        action="store_true",
        help="Backward-compatible: also clean the test files when running in train mode.",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=cfg.CLEANED_DIR,
        help=f"Directory for cleaned output files (default: {cfg.CLEANED_DIR})",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    cfg.QUALITY_LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_path = cfg.LOG_DIR / "data_cleaning.log"
    logger = _setup_logger("data_cleaning", log_path)

    chunk_size = args.chunk_size
    out_dir: Path = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    mode = args.mode.lower()
    tasks = cfg.get_cleaning_tasks(mode)
    if args.source1_only:
        tasks = tasks[:1]
        logger.info(f"--source1-only flag set: cleaning only {mode} Source1.")

    if args.include_test and mode == cfg.TRAIN_MODE:
        logger.info("--include-test passed: also cleaning test sources after train sources.")
        tasks = tasks + cfg.get_cleaning_tasks(cfg.TEST_MODE)

    logger.info("=" * 60)
    logger.info("Data Cleaning Stage — START")
    logger.info(f"Mode: {mode}")
    logger.info(f"Chunk size: {chunk_size:,}")
    logger.info(f"Output dir: {out_dir}")
    logger.info("=" * 60)

    reports: list[dict] = []

    for inp, outp, label in tasks:
        # Keep output filenames aligned to the selected mode while reusing the
        # same chunk-wise cleaning implementation for both train and test.
        out_path = out_dir / outp.name
        try:
            report = clean_source(inp, out_path, label, chunk_size, logger, limit=args.limit)
            save_quality_report(report, label)
            reports.append(report)
        except FileNotFoundError as exc:
            logger.error(str(exc))

    # --- Summary ---
    logger.info("=" * 60)
    logger.info("Data Cleaning Stage — COMPLETE")
    for r in reports:
        logger.info(
            f"  {r['source']}: {r['total_rows']:,} rows, "
            f"country dist: {r['country_distribution']}"
        )
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
