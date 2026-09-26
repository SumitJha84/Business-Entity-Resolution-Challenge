"""
config.py — Centralised configuration for the Entity Resolution pipeline.

All paths, hyperparameters, and rule dictionaries live here.
Downstream modules import from this module — no hard-coded values inline.
"""

from __future__ import annotations

import os
from pathlib import Path

# ---------------------------------------------------------------------------
# Root paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "dataset"
TRAIN_DIR = DATA_DIR / "train"
TEST_DIR = DATA_DIR / "test"
OUTPUT_DIR = PROJECT_ROOT / "output"  # Final submission files only
LOG_DIR = PROJECT_ROOT / "logs"

# ---------------------------------------------------------------------------
# Intermediate processing paths (blocking, splits, and model-preparation data)
# ---------------------------------------------------------------------------
PROCESSING_DIR = PROJECT_ROOT / "processing"
BLOCKING_DIR = PROCESSING_DIR / "blocking"
TRAINING_DATA_DIR = PROCESSING_DIR / "training_data"

# ---------------------------------------------------------------------------
# Source file paths (train)
# ---------------------------------------------------------------------------
TRAIN_SOURCE1 = TRAIN_DIR / "train_source1.tsv"
TRAIN_SOURCE2 = TRAIN_DIR / "train_source2.tsv"
TRAIN_SOURCE3 = TRAIN_DIR / "train_source3.tsv"
TRAIN_GROUND_TRUTH = TRAIN_DIR / "train_ground_truth.tsv"

# ---------------------------------------------------------------------------
# Source file paths (test)
# ---------------------------------------------------------------------------
TEST_SOURCE1 = TEST_DIR / "test_source1.tsv"
TEST_SOURCE2 = TEST_DIR / "test_source2.tsv"
TEST_SOURCE3 = TEST_DIR / "test_source3.tsv"

# ---------------------------------------------------------------------------
# Cleaned output paths (data_cleaning.py writes these)
#
# Root-level folder (sibling to dataset/, output/, src/), not nested under
# output/, so cleaned data is easy to find and stays separate from the
# validation-split artifacts under output/splits/.  Resolved from
# PROJECT_ROOT so it is independent of the current working directory.
# ---------------------------------------------------------------------------
CLEANED_DIR = PROJECT_ROOT / "cleaned_dataset"
CLEANED_SOURCE1 = CLEANED_DIR / "cleaned_source1.tsv"
CLEANED_SOURCE2 = CLEANED_DIR / "cleaned_source2.tsv"
CLEANED_SOURCE3 = CLEANED_DIR / "cleaned_source3.tsv"

# ---------------------------------------------------------------------------
# Validation split output paths (validation_split.py writes these)
# Stored under processing/, not output/; output/ is reserved for final submissions.
# ---------------------------------------------------------------------------
SPLIT_DIR = PROCESSING_DIR / "splits"
TRAIN_SOURCE1_IDS = SPLIT_DIR / "train_source1_ids.txt"
VAL_SOURCE1_IDS = SPLIT_DIR / "val_source1_ids.txt"
TRAIN_GROUND_TRUTH_SPLIT = SPLIT_DIR / "train_ground_truth.tsv"
VAL_GROUND_TRUTH_SPLIT = SPLIT_DIR / "val_ground_truth.tsv"
SPLIT_SUMMARY = SPLIT_DIR / "split_summary.json"

# ---------------------------------------------------------------------------
# Chunking / IO
# ---------------------------------------------------------------------------
CHUNK_SIZE: int = 50_000          # rows per chunk for all source files

# ---------------------------------------------------------------------------
# Validation split hyperparameters
# ---------------------------------------------------------------------------
VAL_FRACTION: float = 0.15        # fraction of Source1 entities for validation
RANDOM_SEED: int = 42
MIN_STRATUM_SIZE: int = 10        # strata smaller than this trigger the fallback rule

# ---------------------------------------------------------------------------
# Ground-truth ID patterns
# ---------------------------------------------------------------------------
VALID_MATCHED_ID_PATTERN: str = r"^(S2|S3)-\d+$"   # accepted prefix+numeric pattern

# ---------------------------------------------------------------------------
# Legal / corporate suffix normalization
#
# Mapping: raw token (lowercased) → canonical token.
# Order matters for the regex: longer tokens should shadow shorter ones.
# These are normalized but NOT removed (business_name_norm keeps them).
# They ARE removed from business_name_core.
# ---------------------------------------------------------------------------
LEGAL_SUFFIX_MAP: dict[str, str] = {
    # Corporations
    "corporation":  "corp",
    "corp.":        "corp",
    # Incorporated
    "incorporated": "inc",
    "inc.":         "inc",
    # Limited
    "limited":      "ltd",
    "ltd.":         "ltd",
    # Private / Pvt
    "private":      "pvt",
    "pvt.":         "pvt",
    # Partnerships
    "llp":          "llp",
    "llc":          "llc",
    "lp":           "lp",
    "l.l.c.":       "llc",
    "l.l.p.":       "llp",
    "l.p.":         "lp",
    # Company
    "company":      "co",
    "co.":          "co",
    # Association
    "associates":   "assoc",
    "associate":    "assoc",
    "assoc.":       "assoc",
    # Enterprises
    "enterprises":  "enterprises",
    "enterprise":   "enterprise",
    # Group
    "group":        "group",
    # India-specific
    "pvt. ltd.":    "pvt ltd",
    "pvt ltd.":     "pvt ltd",
    "pvt.ltd.":     "pvt ltd",
}

# Set of canonical suffix tokens (used to strip in business_name_core)
SUFFIX_CANONICAL_TOKENS: frozenset[str] = frozenset({
    "corp", "inc", "ltd", "pvt", "llc", "llp", "lp", "co",
    "assoc", "enterprises", "enterprise", "group", "pvt ltd",
})

# ---------------------------------------------------------------------------
# Ampersand / conjunction normalisation (conservative — only unambiguous swap)
# ---------------------------------------------------------------------------
AMPERSAND_NORM: bool = True   # replace ' & ' with ' and '

# ---------------------------------------------------------------------------
# Address abbreviation dictionaries
#
# Structure:
#   ABBREV_DICTS["us"]     — applied first for country_norm == "us"
#   ABBREV_DICTS["india"]  — applied first for country_norm == "india"
#   ABBREV_DICTS["generic"]— applied as fallback for all other countries
#                            AND as a second pass for us/india after their own dict
#
# IMPORTANT: These are expansion maps (abbreviation → full form) applied to
# address tokens only.  We expand, then normalise.  Two-pass:
#   1. country-specific expansion
#   2. generic expansion (catches anything missed in step 1)
#
# Conservative: we only expand unambiguous, high-frequency tokens.
# ---------------------------------------------------------------------------
ABBREV_DICTS: dict[str, dict[str, str]] = {
    "us": {
        "st":    "street",
        "st.":   "street",
        "ave":   "avenue",
        "ave.":  "avenue",
        "blvd":  "boulevard",
        "blvd.": "boulevard",
        "rd":    "road",
        "rd.":   "road",
        "dr":    "drive",
        "dr.":   "drive",
        "ln":    "lane",
        "ln.":   "lane",
        "ct":    "court",
        "ct.":   "court",
        "pl":    "place",
        "pl.":   "place",
        "hwy":   "highway",
        "hwy.":  "highway",
        "pkwy":  "parkway",
        "pkwy.": "parkway",
        "fwy":   "freeway",
        "expy":  "expressway",
        "trl":   "trail",
        "trl.":  "trail",
        "ter":   "terrace",
        "ter.":  "terrace",
        "cir":   "circle",
        "cir.":  "circle",
        "sq":    "square",
        "sq.":   "square",
        "ste":   "suite",
        "ste.":  "suite",
        "apt":   "apartment",
        "apt.":  "apartment",
        "bldg":  "building",
        "bldg.": "building",
        "fl":    "floor",
        "frwy":  "freeway",
        "n":     "north",
        "s":     "south",
        "e":     "east",
        "w":     "west",
        "nw":    "northwest",
        "ne":    "northeast",
        "sw":    "southwest",
        "se":    "southeast",
        "po":    "po",          # PO Box — keep as-is
        "p.o.":  "po",
        "us":    "us",          # avoid expanding "US" to something else
        # State abbreviations — do NOT expand; they are canonical identifiers.
    },
    "india": {
        "rd":    "road",
        "rd.":   "road",
        "st":    "street",
        "st.":   "street",
        "ave":   "avenue",
        "ave.":  "avenue",
        "nagar": "nagar",       # keep locality markers canonical
        "marg":  "marg",
        "chowk": "chowk",
        "gali":  "gali",
        "mohalla": "mohalla",
        "colony": "colony",
        "sector": "sector",
        "plot":  "plot",
        "phase":  "phase",
        "h.no":  "house no",
        "h.no.": "house no",
        "hn":    "house no",
        "s.no":  "survey no",
        "s.no.": "survey no",
        "opp":   "opposite",
        "opp.":  "opposite",
        "nr":    "near",
        "nr.":   "near",
        "dist":  "district",
        "dist.": "district",
        "teh":   "tehsil",
        "teh.":  "tehsil",
        "taluk": "taluk",
        "vlg":   "village",
        "vlg.":  "village",
        "blk":   "block",
        "blk.":  "block",
        "p.o":   "po",
        "p.o.":  "po",
        "pin":   "pin",         # PIN code prefix — keep
    },
    "generic": {
        # Shared expansions safe for any country   
        "rd":    "road",
        "rd.":   "road",
        "st":    "street",
        "st.":   "street",
        "ave":   "avenue",
        "ave.":  "avenue",
        "blvd":  "boulevard",
        "blvd.": "boulevard",
        "dr":    "drive",
        "dr.":   "drive",
        "ln":    "lane",
        "ln.":   "lane",
        "ct":    "court",
        "ct.":   "court",
        "sq":    "square",
        "sq.":   "square",
        "no":    "number",      # "No. 5" → "number 5"
        "no.":   "number",
        "bldg":  "building",
        "bldg.": "building",
        "apt":   "apartment",
        "apt.":  "apartment",
        "opp":   "opposite",
        "opp.":  "opposite",
        "nr":    "near",
        "nr.":   "near",
    },
}

# ---------------------------------------------------------------------------
# Landmark detection patterns (regex, applied to address strings)
# These are extracted into business_address_landmark, NOT removed from address.
# ---------------------------------------------------------------------------
LANDMARK_PATTERNS: list[str] = [
    r"(?i)\b(near|nr\.?|adj(?:acent)?|opp(?:osite)?|behind|beside|next\s+to|in\s+front\s+of)\s+[A-Za-z][\w\s,\.]{3,60}",
    r"(?i)\b(landmark\s*[:–-]\s*)[\w\s,\.]{3,60}",
]

# ---------------------------------------------------------------------------
# DBA / trade-name extraction patterns
# ---------------------------------------------------------------------------
DBA_PATTERNS: list[str] = [
    r"(?i)\bdba\b\s*(.+?)(?:\s*,|\s*$)",
    r"(?i)\bt/a\b\s*(.+?)(?:\s*,|\s*$)",
    r"(?i)\btrading\s+as\b\s*(.+?)(?:\s*,|\s*$)",
    r"(?i)\bta\b\s*(.+?)(?:\s*,|\s*$)",
    r"(?i)\balso\s+known\s+as\b\s*(.+?)(?:\s*,|\s*$)",
    r"(?i)\baka\b\s*(.+?)(?:\s*,|\s*$)",
]

# ---------------------------------------------------------------------------
# Data quality log paths
# ---------------------------------------------------------------------------
QUALITY_LOG_DIR = LOG_DIR / "data_quality"
SPLIT_LOG_PATH = LOG_DIR / "validation_split.log"