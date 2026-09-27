"""
prepare_training_data.py — Stage 3: Candidate labeling for entity matching.

Joins blocking candidates with the validated Stage 2 ground truth and produces
labeled train/validation candidate pairs for the downstream matching model.

Inputs
------
  processing/blocking/candidate_pairs.tsv
      Columns: source1_entity_id, candidate_entity_ids
      One row per Source1 entity. candidate_entity_ids is a comma-separated
      list of S2-/S3- IDs; blank means zero candidates.

  processing/splits/train_ground_truth.tsv
  processing/splits/val_ground_truth.tsv
  processing/splits/train_source1_ids.txt
  processing/splits/val_source1_ids.txt

Outputs
-------
  processing/training_data/train_pairs.tsv
  processing/training_data/val_pairs.tsv
  logs/prepare_training_data_report.json
  logs/prepare_training_data.log

The script never re-splits Source1 entities. Train/validation assignment comes
only from the Stage 2 ID lists. Candidate pairs are processed in chunks so the
large blocking file is never loaded fully into memory. Similarity features use
RapidFuzz `cpdist`, which compares corresponding rows and returns one score per pair.

Usage
-----
    python -m src.prepare_training_data
    python -m src.prepare_training_data --chunk-size 25000
"""

from __future__ import annotations

import argparse
import json
import logging
import sqlite3
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Iterator

import pandas as pd
from rapidfuzz import fuzz, process

# Allow `python -m src.prepare_training_data` to import config.py in the same
# way as the existing Stage 1/2 scripts.
sys.path.insert(0, str(Path(__file__).resolve().parent))

import config as cfg


OUTPUT_COLUMNS = [
    "source1_entity_id",
    "candidate_entity_id",
    "source",
    "name_similarity",
    "address_similarity",
    "country_match",
    "label",
]

FEATURE_COLUMNS = [
    "name_similarity",
    "address_similarity",
    "country_match",
]

CLEANED_FEATURE_COLUMNS = [
    "entity_id",
    "business_name_norm",
    "business_address_norm",
    "country_norm",
]

CANDIDATE_POOL_PATH = cfg.BLOCKING_DIR / "train_pool.sqlite"
SOURCE1_FEATURE_DB_PATH = cfg.BLOCKING_DIR / "stage3_source1_feature_lookup.sqlite"


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
        fmt = logging.Formatter(
            "%(asctime)s  %(levelname)-8s  %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        fh.setFormatter(fmt)
        ch.setFormatter(fmt)
        logger.addHandler(fh)
        logger.addHandler(ch)

    return logger


# ---------------------------------------------------------------------------
# Chunked TSV helpers
# ---------------------------------------------------------------------------


def _iter_tsv_chunks(path: Path, chunk_size: int) -> Iterator[pd.DataFrame]:
    """Yield TSV chunks using the same reader conventions as Stages 1/2."""
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


def _read_id_list(path: Path) -> set[str]:
    """Read one exact Source1 entity_id per line; preserve ID characters."""
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    ids: set[str] = set()
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            # Remove only the file line ending; do not strip ID content.
            ids.add(line.rstrip("\r\n"))
    ids.discard("")
    return ids


# ---------------------------------------------------------------------------
# Step 1: Load Stage 2 lookup data
# ---------------------------------------------------------------------------


def _accumulate_ground_truth(
    gt_path: Path,
    true_match_pairs: set[str],
    true_match_by_s1: dict[str, set[str]],
    chunk_size: int,
    logger: logging.Logger,
) -> int:
    """Read a validated GT subset and accumulate exact true-match keys."""
    rows_read = 0
    matched_pairs_added = 0

    for chunk in _iter_tsv_chunks(gt_path, chunk_size):
        required = {"source1_entity_id", "matched_entity_ids"}
        missing = required - set(chunk.columns)
        if missing:
            raise ValueError(
                f"Ground-truth file {gt_path} is missing required columns: "
                f"{sorted(missing)}"
            )

        rows_read += len(chunk)

        # Split exactly on commas. Tokens are intentionally NOT stripped or
        # otherwise normalized: candidate and GT IDs are compared as supplied.
        matched_mask = chunk["matched_entity_ids"].notna() & (
            chunk["matched_entity_ids"] != ""
        )
        matched = chunk.loc[
            matched_mask, ["source1_entity_id", "matched_entity_ids"]
        ].copy()
        if matched.empty:
            continue

        matched["matched_id"] = matched["matched_entity_ids"].str.split(",")
        exploded = matched[["source1_entity_id", "matched_id"]].explode(
            "matched_id", ignore_index=True
        )

        # A blank token would mean the upstream Stage 2 output is inconsistent.
        blank_token_mask = exploded["matched_id"] == ""
        if blank_token_mask.any():
            bad_count = int(blank_token_mask.sum())
            logger.warning(
                f"Ground truth {gt_path} contains {bad_count} blank matched-ID "
                "tokens; those tokens are ignored."
            )
            exploded = exploded.loc[~blank_token_mask]

        if exploded.empty:
            continue

        keys = (
            exploded["source1_entity_id"].astype(str)
            + "::"
            + exploded["matched_id"].astype(str)
        )

        true_match_pairs.update(keys.tolist())

        # true_match_by_s1 is only used for entity-level zero-candidate
        # diagnostics, so keeping the true IDs grouped by Source1 is useful.
        for s1_id, matched_id in zip(
            exploded["source1_entity_id"].tolist(),
            exploded["matched_id"].tolist(),
        ):
            true_match_by_s1[s1_id].add(matched_id)

        matched_pairs_added += len(exploded)

    logger.info(
        f"Loaded GT subset {gt_path.name}: {rows_read:,} rows, "
        f"{matched_pairs_added:,} matched-ID entries."
    )
    return rows_read


def load_data(
    split_dir: Path,
    chunk_size: int,
    logger: logging.Logger,
) -> tuple[
    set[str],
    dict[str, set[str]],
    set[str],
    set[str],
]:
    """
    Load Stage 2 ground-truth subsets and Source1 split IDs.

    Returns:
      true_match_pairs : set of "source1_id::matched_id" keys
      true_match_by_s1 : source1_id -> set of true matched IDs
      train_ids        : validated train Source1 IDs
      val_ids          : validated validation Source1 IDs
    """
    train_gt_path = split_dir / "train_ground_truth.tsv"
    val_gt_path = split_dir / "val_ground_truth.tsv"
    train_ids_path = split_dir / "train_source1_ids.txt"
    val_ids_path = split_dir / "val_source1_ids.txt"

    train_ids = _read_id_list(train_ids_path)
    val_ids = _read_id_list(val_ids_path)

    overlap = train_ids & val_ids
    if overlap:
        raise ValueError(
            f"Stage 2 split is invalid: {len(overlap):,} Source1 IDs occur in "
            "both train and validation ID lists."
        )

    true_match_pairs: set[str] = set()
    true_match_by_s1: dict[str, set[str]] = defaultdict(set)

    _accumulate_ground_truth(
        train_gt_path,
        true_match_pairs,
        true_match_by_s1,
        chunk_size,
        logger,
    )
    _accumulate_ground_truth(
        val_gt_path,
        true_match_pairs,
        true_match_by_s1,
        chunk_size,
        logger,
    )

    logger.info(f"Train Source1 IDs: {len(train_ids):,}")
    logger.info(f"Val Source1 IDs:   {len(val_ids):,}")
    logger.info(f"Unique true-match pairs: {len(true_match_pairs):,}")

    return true_match_pairs, dict(true_match_by_s1), train_ids, val_ids


# ---------------------------------------------------------------------------
# Step 2: Candidate validation
# ---------------------------------------------------------------------------


def validate_candidates(
    chunk: pd.DataFrame,
    train_ids: set[str],
    val_ids: set[str],
    logger: logging.Logger,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """
    Validate candidate rows and return a cleaned candidate-token table.

    Validation is conservative:
      * only exact S2-/S3- prefixes are accepted;
      * Source1 IDs must already belong to the Stage 2 train/val universe;
      * blank candidate lists create no pair rows;
      * duplicate candidate IDs inside one Source1 row are logged and deduped.
    """
    required = {"source1_entity_id", "candidate_entity_ids"}
    missing = required - set(chunk.columns)
    if missing:
        raise ValueError(
            f"Candidate file is missing required columns: {sorted(missing)}"
        )

    work = chunk[["source1_entity_id", "candidate_entity_ids"]].copy()
    work["candidate_entity_ids"] = work["candidate_entity_ids"].fillna("")

    valid_s1 = set(train_ids) | set(val_ids)
    orphan_row_mask = ~work["source1_entity_id"].isin(valid_s1)
    orphan_rows = work.loc[orphan_row_mask]

    # Count candidate tokens belonging to orphan Source1 rows before exclusion.
    orphan_pair_count = 0
    if not orphan_rows.empty:
        orphan_pair_count = int(
            orphan_rows["candidate_entity_ids"]
            .str.split(",")
            .apply(lambda tokens: sum(t != "" for t in tokens))
            .sum()
        )

    work = work.loc[~orphan_row_mask].copy()

    # Entity-level zero-candidate inventory comes from the original blocking
    # row, before the list is exploded.
    zero_candidate_ids = set(
        work.loc[work["candidate_entity_ids"] == "", "source1_entity_id"].tolist()
    )

    nonblank = work.loc[work["candidate_entity_ids"] != ""].copy()
    if nonblank.empty:
        info = {
            "malformed_count": 0,
            "orphan_pair_count": orphan_pair_count,
            "duplicate_count": 0,
            "zero_candidate_ids": zero_candidate_ids,
            "candidate_rows_after_orphan_filter": len(work),
        }
        return pd.DataFrame(
            columns=["source1_entity_id", "candidate_entity_id"]
        ), info

    # Validate the token list without changing any surviving ID token. We use
    # a temporary exploded view only for diagnostics; label_candidates() performs
    # the actual explode for the output rows.
    temp = nonblank[["source1_entity_id", "candidate_entity_ids"]].copy()
    temp["candidate_entity_id"] = temp["candidate_entity_ids"].str.split(",")
    exploded = temp[["source1_entity_id", "candidate_entity_id"]].explode(
        "candidate_entity_id", ignore_index=True
    )

    duplicate_mask = exploded.duplicated(
        subset=["source1_entity_id", "candidate_entity_id"], keep="first"
    )
    duplicate_count = int(duplicate_mask.sum())
    if duplicate_count:
        logger.warning(
            f"Duplicate candidate IDs within Source1 rows: {duplicate_count:,} "
            "duplicate tokens detected; duplicates will be removed from labels."
        )

    valid_s2 = exploded["candidate_entity_id"].str.startswith("S2-")
    valid_s3 = exploded["candidate_entity_id"].str.startswith("S3-")
    valid_candidate_mask = valid_s2 | valid_s3
    malformed_count = int((~valid_candidate_mask).sum())

    if malformed_count:
        logger.warning(
            f"Malformed candidate IDs excluded: {malformed_count:,} "
            "(expected exact S2-/S3- prefixes)."
        )

    info = {
        "malformed_count": malformed_count,
        "orphan_pair_count": orphan_pair_count,
        "duplicate_count": duplicate_count,
        "zero_candidate_ids": zero_candidate_ids,
        "candidate_rows_after_orphan_filter": len(work),
    }
    return work, info


# ---------------------------------------------------------------------------
# Step 3: Feature lookup / extraction
# ---------------------------------------------------------------------------


def _sqlite_read_only_connection(path: Path) -> sqlite3.Connection:
    """Open a SQLite database read-only so Stage 3 cannot mutate the pool."""
    if not path.exists():
        raise FileNotFoundError(f"Required SQLite feature pool not found: {path}")
    uri = f"file:{path.resolve().as_posix()}?mode=ro"
    return sqlite3.connect(uri, uri=True)


def _source1_feature_db_is_current(path: Path) -> bool:
    """Return True when the cached Source1 lookup matches the current cleaned file."""
    if not path.exists() or not cfg.CLEANED_SOURCE1.exists():
        return False

    try:
        source_stat = cfg.CLEANED_SOURCE1.stat()
        with sqlite3.connect(path) as conn:
            row = conn.execute(
                """
                SELECT source_size, source_mtime_ns
                FROM metadata
                WHERE key = 'cleaned_source1'
                """
            ).fetchone()
        return bool(
            row
            and int(row[0]) == int(source_stat.st_size)
            and int(row[1]) == int(source_stat.st_mtime_ns)
        )
    except (sqlite3.Error, OSError, TypeError, ValueError):
        return False


def _build_source1_feature_lookup(
    db_path: Path,
    chunk_size: int,
    logger: logging.Logger,
) -> Path:
    """
    Build a disk-backed Source1 lookup from cleaned_source1.tsv.

    Only the fields required for feature extraction are stored. The cleaned
    Source1 file is read in chunks, so the full 2.2M-row file is never loaded
    into memory.
    """
    source_path = Path(cfg.CLEANED_SOURCE1)
    if not source_path.exists():
        raise FileNotFoundError(
            f"Cleaned Source1 file required for feature extraction was not found: "
            f"{source_path}"
        )

    db_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = db_path.with_name(db_path.name + ".tmp")
    temp_path.unlink(missing_ok=True)

    source_stat = source_path.stat()
    total_rows = 0
    ignored_duplicates = 0

    logger.info(
        f"Building Source1 feature lookup from {source_path} -> {db_path}"
    )

    try:
        with sqlite3.connect(temp_path) as conn:
            conn.execute("PRAGMA journal_mode=OFF")
            conn.execute("PRAGMA synchronous=OFF")
            conn.execute(
                """
                CREATE TABLE records (
                    entity_id TEXT PRIMARY KEY,
                    business_name_norm TEXT,
                    business_address_norm TEXT,
                    country_norm TEXT
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE metadata (
                    key TEXT PRIMARY KEY,
                    source_size INTEGER NOT NULL,
                    source_mtime_ns INTEGER NOT NULL,
                    source_rows INTEGER NOT NULL
                )
                """
            )

            insert_sql = """
                INSERT OR IGNORE INTO records (
                    entity_id,
                    business_name_norm,
                    business_address_norm,
                    country_norm
                )
                VALUES (?, ?, ?, ?)
            """

            for chunk_idx, chunk in enumerate(
                _iter_tsv_chunks(source_path, chunk_size), start=1
            ):
                missing = set(CLEANED_FEATURE_COLUMNS) - set(chunk.columns)
                if missing:
                    raise ValueError(
                        f"Cleaned Source1 file {source_path} is missing required "
                        f"feature columns: {sorted(missing)}"
                    )

                feature_chunk = chunk[CLEANED_FEATURE_COLUMNS].fillna("").astype(str)
                before = conn.total_changes
                conn.executemany(
                    insert_sql,
                    feature_chunk.itertuples(index=False, name=None),
                )
                inserted = conn.total_changes - before
                ignored_duplicates += len(feature_chunk) - inserted
                total_rows += len(feature_chunk)

                if chunk_idx % 10 == 0:
                    logger.info(
                        f"Source1 feature lookup: processed {total_rows:,} rows."
                    )

            conn.execute(
                """
                CREATE INDEX idx_records_entity_id
                ON records(entity_id)
                """
            )
            conn.execute(
                """
                INSERT INTO metadata (
                    key, source_size, source_mtime_ns, source_rows
                )
                VALUES (?, ?, ?, ?)
                """,
                (
                    "cleaned_source1",
                    int(source_stat.st_size),
                    int(source_stat.st_mtime_ns),
                    int(total_rows),
                ),
            )
            conn.commit()

        temp_path.replace(db_path)

    except Exception:
        temp_path.unlink(missing_ok=True)
        raise

    if ignored_duplicates:
        logger.warning(
            f"Source1 feature lookup ignored {ignored_duplicates:,} duplicate "
            "entity_id rows while building the cache."
        )

    logger.info(
        f"Source1 feature lookup ready: {total_rows:,} rows stored at {db_path}"
    )
    return db_path


def ensure_source1_feature_lookup(
    chunk_size: int,
    logger: logging.Logger,
) -> Path:
    """Create or reuse the cached Source1 feature lookup."""
    if _source1_feature_db_is_current(SOURCE1_FEATURE_DB_PATH):
        logger.info(
            f"Reusing current Source1 feature lookup: {SOURCE1_FEATURE_DB_PATH}"
        )
        return SOURCE1_FEATURE_DB_PATH

    logger.info("Source1 feature lookup is missing or stale; rebuilding it.")
    return _build_source1_feature_lookup(
        SOURCE1_FEATURE_DB_PATH,
        chunk_size,
        logger,
    )


def _validate_candidate_pool_schema(pool_conn: sqlite3.Connection) -> None:
    """Validate the existing train S2/S3 pool used for candidate generation."""
    try:
        rows = pool_conn.execute("PRAGMA table_info(pool)").fetchall()
    except sqlite3.Error as exc:
        raise RuntimeError(
            f"Could not inspect the candidate SQLite pool schema: {exc}"
        ) from exc

    columns = {str(row[1]) for row in rows}
    required = {"entity_id", "name", "address", "country"}
    missing = required - columns
    if missing:
        raise RuntimeError(
            "The existing train candidate pool does not contain the fields "
            f"required for feature extraction: {sorted(missing)}"
        )


def _load_source1_records(
    conn: sqlite3.Connection,
    entity_ids: set[str],
) -> pd.DataFrame:
    """Load only the Source1 records needed by one candidate chunk."""
    columns = [
        "entity_id",
        "business_name_norm",
        "business_address_norm",
        "country_norm",
    ]
    if not entity_ids:
        return pd.DataFrame(columns=columns)

    conn.execute(
        """
        CREATE TEMP TABLE IF NOT EXISTS requested_source1_ids (
            entity_id TEXT PRIMARY KEY
        )
        """
    )
    conn.execute("DELETE FROM requested_source1_ids")
    conn.executemany(
        "INSERT OR IGNORE INTO requested_source1_ids(entity_id) VALUES (?)",
        ((entity_id,) for entity_id in entity_ids),
    )

    query = """
        SELECT
            r.entity_id,
            r.business_name_norm,
            r.business_address_norm,
            r.country_norm
        FROM records AS r
        INNER JOIN requested_source1_ids AS q
            ON r.entity_id = q.entity_id
    """
    return pd.read_sql_query(query, conn).fillna("")


def _load_candidate_records(
    conn: sqlite3.Connection,
    entity_ids: set[str],
) -> pd.DataFrame:
    """
    Load only the candidate S2/S3 records needed by one candidate chunk.

    The existing training candidate pool stores normalized values in columns
    named country/name/address, so they are mapped back to the feature schema.
    """
    columns = [
        "entity_id",
        "business_name_norm",
        "business_address_norm",
        "country_norm",
    ]
    if not entity_ids:
        return pd.DataFrame(columns=columns)

    conn.execute(
        """
        CREATE TEMP TABLE IF NOT EXISTS requested_candidate_ids (
            entity_id TEXT PRIMARY KEY
        )
        """
    )
    conn.execute("DELETE FROM requested_candidate_ids")
    conn.executemany(
        "INSERT OR IGNORE INTO requested_candidate_ids(entity_id) VALUES (?)",
        ((entity_id,) for entity_id in entity_ids),
    )

    query = """
        SELECT
            p.entity_id,
            p.name AS business_name_norm,
            p.address AS business_address_norm,
            p.country AS country_norm
        FROM pool AS p
        INNER JOIN requested_candidate_ids AS q
            ON p.entity_id = q.entity_id
    """
    return pd.read_sql_query(query, conn).fillna("")


def _rapidfuzz_pairwise_ratio(
    left: pd.Series,
    right: pd.Series,
) -> pd.Series:
    """
    Compute corresponding RapidFuzz ratios for two aligned Series.

    Blank values are assigned 0.0 so that ratio("", "") never becomes 1.0.
    The two Series must have the same length and index so that the returned
    scores remain aligned with the input rows.
    """
    if len(left) != len(right):
        raise ValueError(
            "RapidFuzz pairwise inputs must have the same length: "
            f"left={len(left)}, right={len(right)}."
        )

    if not left.index.equals(right.index):
        raise ValueError(
            "RapidFuzz pairwise inputs must have identical indexes so that "
            "scores remain aligned with the corresponding candidate rows."
        )

    left = left.fillna("").astype(str)
    right = right.fillna("").astype(str)

    scores = pd.Series(0.0, index=left.index, dtype="float64")
    valid = left.ne("") & right.ne("")

    if valid.any():
        # cpdist() compares corresponding elements (row i with row i) and
        # returns one score per pair. It is intentionally preserved here;
        # replacing it with cdist() would create an unnecessary n x n matrix.
        values = process.cpdist(
            left.loc[valid].tolist(),
            right.loc[valid].tolist(),
            scorer=fuzz.ratio,
            workers=-1,
            dtype=float,
        )
        values = pd.Series(values, index=left.loc[valid].index, dtype="float64")
        scores.loc[valid] = values / 100.0

    return scores


def extract_pair_features(
    labeled: pd.DataFrame,
    source1_records: pd.DataFrame,
    candidate_records: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, int]]:
    """
    Add name/address/country features to already-labeled candidate pairs.

    The left join guarantees that missing business records do not discard
    candidate pairs. Missing fields simply produce zero-valued features.
    """
    if labeled.empty:
        empty = labeled.copy()
        empty["name_similarity"] = pd.Series(dtype="float64")
        empty["address_similarity"] = pd.Series(dtype="float64")
        empty["country_match"] = pd.Series(dtype="int8")
        return empty[OUTPUT_COLUMNS], {
            "missing_source1_records": 0,
            "missing_candidate_records": 0,
        }

    s1 = source1_records[
        [
            "entity_id",
            "business_name_norm",
            "business_address_norm",
            "country_norm",
        ]
    ].copy()
    s1.columns = [
        "source1_entity_id",
        "s1_business_name_norm",
        "s1_business_address_norm",
        "s1_country_norm",
    ]

    candidates = candidate_records[
        [
            "entity_id",
            "business_name_norm",
            "business_address_norm",
            "country_norm",
        ]
    ].copy()
    candidates.columns = [
        "candidate_entity_id",
        "candidate_business_name_norm",
        "candidate_business_address_norm",
        "candidate_country_norm",
    ]

    enriched = labeled.merge(
        s1,
        on="source1_entity_id",
        how="left",
        sort=False,
        validate="many_to_one",
    )
    enriched = enriched.merge(
        candidates,
        on="candidate_entity_id",
        how="left",
        sort=False,
        validate="many_to_one",
    )

    missing_source1 = int(enriched["s1_business_name_norm"].isna().sum())
    missing_candidates = int(
        enriched["candidate_business_name_norm"].isna().sum()
    )

    # read_sql_query().fillna("") makes existing blank values explicit; merged
    # missing records remain NaN and are safely converted to empty strings here.
    s1_names = enriched["s1_business_name_norm"].fillna("").astype(str)
    c_names = enriched["candidate_business_name_norm"].fillna("").astype(str)
    s1_addresses = enriched["s1_business_address_norm"].fillna("").astype(str)
    c_addresses = enriched["candidate_business_address_norm"].fillna("").astype(str)
    s1_countries = enriched["s1_country_norm"].fillna("").astype(str)
    c_countries = enriched["candidate_country_norm"].fillna("").astype(str)

    enriched["name_similarity"] = _rapidfuzz_pairwise_ratio(
        s1_names,
        c_names,
    ).astype("float64")

    enriched["address_similarity"] = _rapidfuzz_pairwise_ratio(
        s1_addresses,
        c_addresses,
    ).astype("float64")

    enriched["country_match"] = (
        s1_countries.ne("")
        & c_countries.ne("")
        & s1_countries.eq(c_countries)
    ).astype("int8")

    return enriched[OUTPUT_COLUMNS], {
        "missing_source1_records": missing_source1,
        "missing_candidate_records": missing_candidates,
    }


# ---------------------------------------------------------------------------
# Step 4: Candidate labeling
# ---------------------------------------------------------------------------


def label_candidates(
    chunk: pd.DataFrame,
    true_match_pairs: set[str],
) -> tuple[pd.DataFrame, set[str]]:
    """
    Assign binary match labels using vectorized membership in true_match_pairs.

    Returns the labeled candidate chunk plus the set of true-match keys observed
    in this chunk for the blocking-recall accumulator.
    """
    if chunk.empty:
        empty = pd.DataFrame(
            columns=["source1_entity_id", "candidate_entity_id", "source", "label"]
        )
        return empty, set()

    # Explode candidate_entity_ids into one row per candidate pair. This is the
    # only candidate-level expansion that feeds the output table.
    nonblank = chunk.loc[chunk["candidate_entity_ids"] != ""].copy()
    if nonblank.empty:
        empty = pd.DataFrame(
            columns=["source1_entity_id", "candidate_entity_id", "source", "label"]
        )
        return empty, set()

    nonblank["candidate_entity_id"] = nonblank["candidate_entity_ids"].str.split(",")
    exploded = nonblank[["source1_entity_id", "candidate_entity_id"]].explode(
        "candidate_entity_id", ignore_index=True
    )

    # Drop malformed IDs and deduplicate candidates within each Source1 row.
    # validate_candidates() has already counted/logged these conditions, but we
    # enforce the same filter here so no bad candidate can reach the model.
    valid_mask = exploded["candidate_entity_id"].str.startswith(("S2-", "S3-"))
    exploded = exploded.loc[valid_mask].copy()
    exploded = exploded.drop_duplicates(
        subset=["source1_entity_id", "candidate_entity_id"],
        keep="first",
    )

    if exploded.empty:
        empty = pd.DataFrame(
            columns=["source1_entity_id", "candidate_entity_id", "source", "label"]
        )
        return empty, set()

    # A singleton Source1 entity has no true-match IDs, so none of its candidate
    # keys can occur in true_match_pairs; those candidates therefore naturally
    # receive label 0 without any special-case logic.
    pair_keys = (
        exploded["source1_entity_id"]
        + "::"
        + exploded["candidate_entity_id"]
    )
    exploded["label"] = pair_keys.isin(true_match_pairs).astype("int8")
    exploded["source"] = exploded["candidate_entity_id"].str.slice(0, 2)
    labeled = exploded[
        ["source1_entity_id", "candidate_entity_id", "source", "label"]
    ]

    observed_true_match_keys = set(pair_keys[labeled["label"].eq(1)].tolist())
    return labeled, observed_true_match_keys


# ---------------------------------------------------------------------------
# Step 4: Split by existing Stage 2 IDs
# ---------------------------------------------------------------------------


def split_pairs(
    labeled_chunk: pd.DataFrame,
    train_ids: set[str],
    val_ids: set[str],
    logger: logging.Logger,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Assign candidate pairs to train/val without re-randomizing."""
    if labeled_chunk.empty:
        empty = labeled_chunk.copy()
        return empty, empty.copy()

    in_train = labeled_chunk["source1_entity_id"].isin(train_ids)
    in_val = labeled_chunk["source1_entity_id"].isin(val_ids)
    invalid = ~(in_train | in_val)

    if invalid.any():
        count = int(invalid.sum())
        logger.error(
            f"split_pairs received {count:,} rows whose Source1 ID belongs to "
            "neither train nor validation. This should have been filtered earlier."
        )
        raise ValueError("Invalid Source1 IDs reached split_pairs().")

    return labeled_chunk.loc[in_train].copy(), labeled_chunk.loc[in_val].copy()


# ---------------------------------------------------------------------------
# Step 5: Stream outputs
# ---------------------------------------------------------------------------


def _prepare_output_files(output_dir: Path) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    train_path = output_dir / "train_pairs.tsv"
    val_path = output_dir / "val_pairs.tsv"

    # Restart-safe: never append on top of a previous run.
    train_path.unlink(missing_ok=True)
    val_path.unlink(missing_ok=True)

    # Create both files immediately so an all-empty stage still produces valid
    # TSV outputs with the expected header.
    pd.DataFrame(columns=OUTPUT_COLUMNS).to_csv(
        train_path, sep="\t", index=False, encoding="utf-8"
    )
    pd.DataFrame(columns=OUTPUT_COLUMNS).to_csv(
        val_path, sep="\t", index=False, encoding="utf-8"
    )

    return train_path, val_path


def save_outputs(
    train_chunk: pd.DataFrame,
    val_chunk: pd.DataFrame,
    train_path: Path,
    val_path: Path,
) -> None:
    """Append labeled train/val rows to their restart-safe TSV outputs."""
    if not train_chunk.empty:
        train_chunk[OUTPUT_COLUMNS].to_csv(
            train_path,
            sep="\t",
            index=False,
            header=False,
            mode="a",
            encoding="utf-8",
        )

    if not val_chunk.empty:
        val_chunk[OUTPUT_COLUMNS].to_csv(
            val_path,
            sep="\t",
            index=False,
            header=False,
            mode="a",
            encoding="utf-8",
        )


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------


def _split_true_match_keys(
    true_match_pairs: set[str],
    train_ids: set[str],
    val_ids: set[str],
) -> tuple[set[str], set[str]]:
    """Partition ground-truth pair keys by their Source1 split."""
    train_pairs: set[str] = set()
    val_pairs: set[str] = set()

    for key in true_match_pairs:
        source1_id = key.split("::", 1)[0]
        if source1_id in train_ids:
            train_pairs.add(key)
        elif source1_id in val_ids:
            val_pairs.add(key)

    return train_pairs, val_pairs


def _entity_ids_with_missed_matches(
    missed_keys: set[str],
) -> set[str]:
    return {key.split("::", 1)[0] for key in missed_keys}


# ---------------------------------------------------------------------------
# CLI / main
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Stage 3: Join blocking candidate pairs with validated ground truth "
            "and produce labeled train/validation pairs."
        )
    )
    p.add_argument(
        "--chunk-size",
        type=int,
        default=cfg.CHUNK_SIZE,
        help=f"Rows per candidate chunk (default: {cfg.CHUNK_SIZE:,})",
    )
    p.add_argument(
        "--candidate-path",
        type=Path,
        default=cfg.BLOCKING_DIR / "candidate_pairs.tsv",
        help="Blocking candidate_pairs.tsv path.",
    )
    p.add_argument(
        "--split-dir",
        type=Path,
        default=cfg.SPLIT_DIR,
        help=f"Stage 2 split directory (default: {cfg.SPLIT_DIR})",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=cfg.TRAINING_DATA_DIR,
        help=f"Labeled pair output directory (default: {cfg.TRAINING_DATA_DIR})",
    )
    args = p.parse_args()

    if args.chunk_size <= 0:
        p.error("--chunk-size must be a positive integer.")

    return args


def main() -> None:
    args = parse_args()

    log_path = cfg.LOG_DIR / "prepare_training_data.log"
    report_path = cfg.LOG_DIR / "prepare_training_data_report.json"
    logger = _setup_logger("prepare_training_data", log_path)

    logger.info("=" * 60)
    logger.info("Prepare Training Data Stage — START")
    logger.info(f"Candidate path: {args.candidate_path}")
    logger.info(f"Split dir:      {args.split_dir}")
    logger.info(f"Output dir:     {args.output_dir}")
    logger.info(f"Chunk size:     {args.chunk_size:,}")
    logger.info("=" * 60)

    t0 = time.perf_counter()

    true_match_pairs, true_match_by_s1, train_ids, val_ids = load_data(
        args.split_dir,
        args.chunk_size,
        logger,
    )

    train_true_match_pairs, val_true_match_pairs = _split_true_match_keys(
        true_match_pairs,
        train_ids,
        val_ids,
    )

    train_path, val_path = _prepare_output_files(args.output_dir)

    observed_train_true_match_keys: set[str] = set()
    observed_val_true_match_keys: set[str] = set()

    total_candidate_pairs = 0
    malformed_candidate_count = 0
    orphan_source1_pair_count = 0
    duplicate_candidate_count = 0
    zero_candidate_total = 0
    zero_candidate_singleton = {"train": 0, "val": 0}
    zero_candidate_matched = {"train": 0, "val": 0}
    zero_candidate_matched_samples = {"train": [], "val": []}
    train_positive = train_negative = 0
    val_positive = val_negative = 0
    candidate_entity_rows = 0
    missing_source1_feature_records = 0
    missing_candidate_feature_records = 0

    # Feature extraction uses the same cleaned business records as the rest of
    # the pipeline. Source1 is indexed once on disk; S2/S3 records are read from
    # the existing train candidate pool used to generate candidate_pairs.tsv.
    source1_feature_db = ensure_source1_feature_lookup(args.chunk_size, logger)
    source1_conn = sqlite3.connect(source1_feature_db)
    pool_conn = _sqlite_read_only_connection(CANDIDATE_POOL_PATH)

    try:
        _validate_candidate_pool_schema(pool_conn)

        for chunk_idx, candidate_chunk in enumerate(
            _iter_tsv_chunks(args.candidate_path, args.chunk_size),
            start=1,
        ):
            candidate_entity_rows += len(candidate_chunk)

            # Main chunk-level validation. The zero-candidate entity inventory is
            # captured before explosion so matched entities with no candidates are
            # not lost from diagnostics.
            validated, info = validate_candidates(
                candidate_chunk,
                train_ids,
                val_ids,
                logger,
            )

            malformed_candidate_count += int(info["malformed_count"])
            orphan_source1_pair_count += int(info["orphan_pair_count"])
            duplicate_candidate_count += int(info["duplicate_count"])

            zero_ids = info["zero_candidate_ids"]
            zero_candidate_total += len(zero_ids)

            for s1_id in zero_ids:
                split = "train" if s1_id in train_ids else "val"
                if s1_id in true_match_by_s1:
                    zero_candidate_matched[split] += 1
                    if len(zero_candidate_matched_samples[split]) < 20:
                        zero_candidate_matched_samples[split].append(s1_id)
                else:
                    zero_candidate_singleton[split] += 1

            labeled, observed_keys = label_candidates(validated, true_match_pairs)

            # Extract features for the same labeled pair rows. Both lookups are
            # limited to IDs present in this chunk, so no full business dataset is
            # loaded into memory.
            source1_ids_for_features = set(
                labeled["source1_entity_id"].astype(str).unique().tolist()
            )
            candidate_ids_for_features = set(
                labeled["candidate_entity_id"].astype(str).unique().tolist()
            )

            source1_records = _load_source1_records(
                source1_conn,
                source1_ids_for_features,
            )
            candidate_records = _load_candidate_records(
                pool_conn,
                candidate_ids_for_features,
            )

            labeled, feature_info = extract_pair_features(
                labeled,
                source1_records,
                candidate_records,
            )

            missing_source1_feature_records += int(
                feature_info["missing_source1_records"]
            )
            missing_candidate_feature_records += int(
                feature_info["missing_candidate_records"]
            )

            if observed_keys:
                observed_train_true_match_keys.update(
                    observed_keys & train_true_match_pairs
                )
                observed_val_true_match_keys.update(
                    observed_keys & val_true_match_pairs
                )

            train_chunk, val_chunk = split_pairs(
                labeled,
                train_ids,
                val_ids,
                logger,
            )

            save_outputs(train_chunk, val_chunk, train_path, val_path)

            train_labels = train_chunk["label"].value_counts().to_dict()
            val_labels = val_chunk["label"].value_counts().to_dict()
            train_positive += int(train_labels.get(1, 0))
            train_negative += int(train_labels.get(0, 0))
            val_positive += int(val_labels.get(1, 0))
            val_negative += int(val_labels.get(0, 0))
            total_candidate_pairs += len(labeled)

            if chunk_idx % 10 == 0:
                elapsed = time.perf_counter() - t0
                logger.info(
                    f"Processed candidate chunk {chunk_idx:,}: "
                    f"{total_candidate_pairs:,} labeled pairs so far; "
                    f"{elapsed:.1f}s elapsed."
                )

    finally:
        source1_conn.close()
        pool_conn.close()

    # Blocking recall is computed against all true-match keys, not against the
    # negative candidate stream.
    train_missed = train_true_match_pairs - observed_train_true_match_keys
    val_missed = val_true_match_pairs - observed_val_true_match_keys

    train_missed_entities = _entity_ids_with_missed_matches(train_missed)
    val_missed_entities = _entity_ids_with_missed_matches(val_missed)

    train_recall = (
        len(observed_train_true_match_keys) / len(train_true_match_pairs)
        if train_true_match_pairs
        else 1.0
    )
    val_recall = (
        len(observed_val_true_match_keys) / len(val_true_match_pairs)
        if val_true_match_pairs
        else 1.0
    )

    report = {
        "stage": "prepare_training_data",
        "candidate_file": str(args.candidate_path),
        "split_dir": str(args.split_dir),
        "training_output_dir": str(args.output_dir),
        "train_pairs_file": str(train_path),
        "val_pairs_file": str(val_path),
        "candidate_entity_rows_processed": int(candidate_entity_rows),
        "total_candidate_pairs_processed_post_explode": int(total_candidate_pairs),
        "malformed_candidate_count": int(malformed_candidate_count),
        "orphan_source1_pair_count_excluded": int(orphan_source1_pair_count),
        "duplicate_candidate_within_row_count": int(duplicate_candidate_count),
        "feature_extraction": {
            "name_similarity": "RapidFuzz fuzz.ratio / 100 using cleaned business_name_norm",
            "address_similarity": "RapidFuzz fuzz.ratio / 100 using cleaned business_address_norm",
            "country_match": "1 only when both cleaned country_norm values are non-empty and equal",
            "missing_source1_record_count": int(missing_source1_feature_records),
            "missing_candidate_record_count": int(missing_candidate_feature_records),
            "feature_lookup_db": str(source1_feature_db),
            "candidate_pool_db": str(CANDIDATE_POOL_PATH),
        },
        "class_balance": {
            "train": {
                "positive": int(train_positive),
                "negative": int(train_negative),
            },
            "val": {
                "positive": int(val_positive),
                "negative": int(val_negative),
            },
        },
        "blocking_recall": {
            "train": {
                "true_match_pairs": int(len(train_true_match_pairs)),
                "observed_true_match_pairs": int(len(observed_train_true_match_keys)),
                "missed_true_match_pair_count": int(len(train_missed)),
                "missed_source1_entity_count": int(len(train_missed_entities)),
                "recall": float(train_recall),
            },
            "val": {
                "true_match_pairs": int(len(val_true_match_pairs)),
                "observed_true_match_pairs": int(len(observed_val_true_match_keys)),
                "missed_true_match_pair_count": int(len(val_missed)),
                "missed_source1_entity_count": int(len(val_missed_entities)),
                "recall": float(val_recall),
            },
        },
        "zero_candidate_entities": {
            "train": {
                "singleton": int(zero_candidate_singleton["train"]),
                "matched_but_missed": int(zero_candidate_matched["train"]),
                "matched_samples": zero_candidate_matched_samples["train"],
            },
            "val": {
                "singleton": int(zero_candidate_singleton["val"]),
                "matched_but_missed": int(zero_candidate_matched["val"]),
                "matched_samples": zero_candidate_matched_samples["val"],
            },
            "total_zero_candidate_entities": int(zero_candidate_total),
        },
        "notes": [
            "IDs are compared without stripping or normalization.",
            "Train/validation assignment comes exclusively from Stage 2 Source1 ID lists.",
            "Empty candidate lists produce zero pair rows; they are tracked only in entity-level diagnostics.",
            "Feature similarity is computed from cleaned business_name_norm and business_address_norm values using RapidFuzz fuzz.ratio.",
            "Missing business records or field values are retained and receive zero-valued similarity/country features.",
        ],
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    elapsed = time.perf_counter() - t0

    logger.info("=" * 60)
    logger.info("Prepare Training Data Stage — COMPLETE")
    logger.info(f"Candidate entity rows: {candidate_entity_rows:,}")
    logger.info(f"Labeled candidate pairs: {total_candidate_pairs:,}")
    logger.info(
        f"Train labels: +{train_positive:,} / -{train_negative:,}; "
        f"Val labels: +{val_positive:,} / -{val_negative:,}"
    )
    logger.info(
        "Feature extraction missing records — "
        f"Source1: {missing_source1_feature_records:,}; "
        f"S2/S3 candidates: {missing_candidate_feature_records:,}"
    )
    logger.info(
        f"Blocking recall — train: {train_recall:.6f} "
        f"({len(train_missed):,} missed pairs; {len(train_missed_entities):,} affected S1 entities)"
    )
    logger.info(
        f"Blocking recall — val:   {val_recall:.6f} "
        f"({len(val_missed):,} missed pairs; {len(val_missed_entities):,} affected S1 entities)"
    )
    logger.info(
        f"Zero-candidate matched entities — train: {zero_candidate_matched['train']:,}; "
        f"val: {zero_candidate_matched['val']:,}"
    )
    logger.info(
        f"Zero-candidate singletons — train: {zero_candidate_singleton['train']:,}; "
        f"val: {zero_candidate_singleton['val']:,}"
    )
    logger.info(f"Report: {report_path}")
    logger.info(f"Total elapsed: {elapsed:.1f}s")
    logger.info("=" * 60)

    if zero_candidate_matched["train"] or zero_candidate_matched["val"]:
        logger.warning(
            "BLOCKING WARNING: at least one matched Source1 entity received zero candidates. "
            "Those true matches are unrecoverable by the downstream model."
        )


if __name__ == "__main__":
    main()