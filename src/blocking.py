"""
blocking.py
===========
Reusable candidate-generation logic for the Business Entity Resolution
pipeline.

Responsibilities
----------------
1. Build a mode-specific SQLite candidate pool from cleaned Source 2 + Source 3.
2. Record a manifest/fingerprint for the pool so stale or wrongly-scoped pools
   are never silently reused.
3. Expose independent candidate-generation strategies that can be enabled or
   disabled by the orchestration module.
4. Generate a bounded, deduplicated candidate set for one Source 1 record.

Important contracts
-------------------
- Entity IDs are never changed, normalized, cast, or reconstructed.
- Country is an open string set. There is no US/India/France hard-coding.
- Ground truth is never imported or read here.
- Source files are read in chunks when building the pool.
- This module does not process the full Source 1 file. That belongs in
  generate_test_candidates.py.

Strategies
----------
A. exact_name      : exact normalized business name + country
B. exact_name_alt  : exact business_name_core / business_name_alt + country
C. exact_address   : exact normalized address + country
D. address_key     : house-number/address-token keys + country
E. fuzzy_name      : bounded fuzzy name search using prefix and name-anchor
                       blocks, followed by RapidFuzz ranking

The orchestration layer can enable/disable these independently.
"""

from __future__ import annotations

import hashlib
import re
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import pandas as pd
from rapidfuzz import fuzz

# ---------------------------------------------------------------------------
# Local project configuration
# ---------------------------------------------------------------------------
try:
    from src import config as cfg
except ImportError:  # pragma: no cover - supports running src/blocking.py directly
    import config as cfg


# ---------------------------------------------------------------------------
# Public configuration / constants
# ---------------------------------------------------------------------------
POOL_SCHEMA_VERSION = "2"

STRATEGY_EXACT_NAME = "exact_name"
STRATEGY_EXACT_NAME_ALT = "exact_name_alt"
STRATEGY_EXACT_ADDRESS = "exact_address"
STRATEGY_ADDRESS_KEY = "address_key"
STRATEGY_FUZZY_NAME = "fuzzy_name"

ALL_STRATEGIES: tuple[str, ...] = (
    STRATEGY_EXACT_NAME,
    STRATEGY_EXACT_NAME_ALT,
    STRATEGY_EXACT_ADDRESS,
    STRATEGY_ADDRESS_KEY,
    STRATEGY_FUZZY_NAME,
)

# Conservative defaults. The orchestration CLI can override them.
DEFAULT_CHUNK_SIZE = getattr(cfg, "CHUNK_SIZE", 50_000)
DEFAULT_MAX_CANDIDATES = 200
DEFAULT_EXACT_LIMIT = 100
DEFAULT_ADDRESS_KEY_MAX_FREQUENCY = 200
DEFAULT_ADDRESS_KEY_WORDS = 6
DEFAULT_FUZZY_PREFIX_LENGTH = 5
DEFAULT_FUZZY_BLOCK_LIMIT = 300
DEFAULT_FUZZY_TOP_K = 25
DEFAULT_FUZZY_THRESHOLD = 78.0
DEFAULT_NAME_ANCHOR_MIN_LENGTH = 5

# Words that are usually too generic to be useful by themselves as address keys.
# We deliberately do not hard-code country-specific address rules here.
ADDRESS_STOP_WORDS = frozenset(
    {
        "road",
        "street",
        "avenue",
        "boulevard",
        "lane",
        "highway",
        "parkway",
        "city",
        "district",
        "state",
        "near",
        "the",
        "and",
        "of",
        "block",
        "phase",
    }
)

_WORD_RE = re.compile(r"[^\W\d_]+", flags=re.UNICODE)
_NUMBER_RE = re.compile(r"\b\d+\b")
_TOKEN_RE = re.compile(r"[a-z0-9]+", flags=re.IGNORECASE)

_REQUIRED_POOL_COLUMNS = {
    "entity_id",
    "country_norm",
    "business_name_norm",
    "business_name_core",
    "business_name_alt",
    "business_address_norm",
    "business_address_landmark",
}


class PoolValidationError(RuntimeError):
    """Raised when an existing SQLite pool does not match the requested scope."""


@dataclass(frozen=True)
class PoolSource:
    """One cleaned Source 2 or Source 3 file used to build a pool."""

    source_name: str  # S2 or S3
    path: Path


@dataclass(frozen=True)
class PoolSpec:
    """Expected identity of a candidate pool."""

    mode: str
    source2: PoolSource
    source3: PoolSource


@dataclass(frozen=True)
class CandidateConfig:
    """Runtime controls for candidate retrieval."""

    enabled_strategies: tuple[str, ...] = ALL_STRATEGIES
    max_candidates: int | None = DEFAULT_MAX_CANDIDATES
    exact_limit: int = DEFAULT_EXACT_LIMIT
    address_key_max_frequency: int = DEFAULT_ADDRESS_KEY_MAX_FREQUENCY
    fuzzy_prefix_length: int = DEFAULT_FUZZY_PREFIX_LENGTH
    fuzzy_block_limit: int = DEFAULT_FUZZY_BLOCK_LIMIT
    fuzzy_top_k: int = DEFAULT_FUZZY_TOP_K
    fuzzy_threshold: float = DEFAULT_FUZZY_THRESHOLD
    name_anchor_min_length: int = DEFAULT_NAME_ANCHOR_MIN_LENGTH

    def __post_init__(self) -> None:
        invalid = set(self.enabled_strategies) - set(ALL_STRATEGIES)
        if invalid:
            raise ValueError(f"Unknown blocking strategies: {sorted(invalid)}")
        if self.max_candidates is not None and self.max_candidates <= 0:
            raise ValueError("max_candidates must be positive or None")
        if self.exact_limit <= 0:
            raise ValueError("exact_limit must be positive")
        if self.address_key_max_frequency <= 0:
            raise ValueError("address_key_max_frequency must be positive")
        if self.fuzzy_prefix_length <= 0:
            raise ValueError("fuzzy_prefix_length must be positive")
        if self.fuzzy_block_limit <= 0:
            raise ValueError("fuzzy_block_limit must be positive")
        if self.fuzzy_top_k <= 0:
            raise ValueError("fuzzy_top_k must be positive")
        if not 0 <= self.fuzzy_threshold <= 100:
            raise ValueError("fuzzy_threshold must be in [0, 100]")
        if self.name_anchor_min_length <= 0:
            raise ValueError("name_anchor_min_length must be positive")


# ---------------------------------------------------------------------------
# Mode/path helpers
# ---------------------------------------------------------------------------

def normalize_mode(mode: str) -> str:
    mode = mode.strip().lower()
    if mode not in {"train", "test"}:
        raise ValueError("mode must be exactly 'train' or 'test'")
    return mode


def cleaned_pool_sources(mode: str) -> tuple[PoolSource, PoolSource]:
    """Return cleaned Source 2/3 paths for the requested mode."""
    mode = normalize_mode(mode)

    if mode == "train":
        source2 = cfg.CLEANED_SOURCE2
        source3 = cfg.CLEANED_SOURCE3
    else:
        source2 = cfg.CLEANED_DIR / "cleaned_test_source2.tsv"
        source3 = cfg.CLEANED_DIR / "cleaned_test_source3.tsv"

    return (
        PoolSource("S2", Path(source2).resolve()),
        PoolSource("S3", Path(source3).resolve()),
    )


def pool_path(mode: str) -> Path:
    """Return the mode-specific SQLite pool path."""
    mode = normalize_mode(mode)
    return (cfg.BLOCKING_DIR / f"{mode}_pool.sqlite").resolve()


def candidate_output_path(mode: str) -> Path:
    """Return the mode-specific candidate-pairs output path."""
    mode = normalize_mode(mode)
    return (cfg.BLOCKING_DIR / f"candidate_pairs_{mode}.tsv").resolve()


def make_pool_spec(mode: str) -> PoolSpec:
    source2, source3 = cleaned_pool_sources(mode)
    return PoolSpec(mode=normalize_mode(mode), source2=source2, source3=source3)


# ---------------------------------------------------------------------------
# Fingerprinting / metadata validation
# ---------------------------------------------------------------------------

def file_fingerprint(path: Path, read_size: int = 8 * 1024 * 1024) -> tuple[int, str]:
    """
    Stream a file once and return (data_row_count, sha256_hex).

    The row count assumes a single header row, which matches the cleaned TSV
    contract. No file contents are loaded into memory.
    """
    path = Path(path).resolve()
    if not path.exists():
        raise FileNotFoundError(f"Required cleaned source file not found: {path}")

    digest = hashlib.sha256()
    newline_count = 0

    with path.open("rb") as handle:
        while True:
            block = handle.read(read_size)
            if not block:
                break
            digest.update(block)
            newline_count += block.count(b"\n")

    # Header + one line terminator is the expected TSV shape.
    data_rows = max(newline_count - 1, 0)
    return data_rows, digest.hexdigest()


def _metadata_pairs(spec: PoolSpec) -> dict[str, str]:
    s2_rows, s2_hash = file_fingerprint(spec.source2.path)
    s3_rows, s3_hash = file_fingerprint(spec.source3.path)

    return {
        "schema_version": POOL_SCHEMA_VERSION,
        "mode": spec.mode,
        "source2_name": spec.source2.source_name,
        "source2_path": str(spec.source2.path),
        "source2_rows": str(s2_rows),
        "source2_sha256": s2_hash,
        "source3_name": spec.source3.source_name,
        "source3_path": str(spec.source3.path),
        "source3_rows": str(s3_rows),
        "source3_sha256": s3_hash,
    }


def _read_metadata(connection: sqlite3.Connection) -> dict[str, str]:
    try:
        rows = connection.execute("SELECT key, value FROM metadata").fetchall()
    except sqlite3.DatabaseError as exc:
        raise PoolValidationError(
            "Existing pool has no readable metadata table; refusing to reuse it."
        ) from exc
    return {str(key): str(value) for key, value in rows}


def validate_pool(pool_db: Path, spec: PoolSpec) -> None:
    """
    Validate an existing pool against the requested mode and exact source files.

    On mismatch this raises PoolValidationError and explicitly reports every
    failed check. It never rebuilds the pool and never silently falls back.
    """
    pool_db = Path(pool_db).resolve()
    if not pool_db.exists():
        raise PoolValidationError(f"Pool does not exist: {pool_db}")

    expected = _metadata_pairs(spec)

    try:
        connection = sqlite3.connect(f"file:{pool_db}?mode=ro", uri=True)
    except sqlite3.Error as exc:
        raise PoolValidationError(f"Cannot open pool read-only: {pool_db}") from exc

    try:
        actual = _read_metadata(connection)

        mismatches: list[str] = []
        for key, expected_value in expected.items():
            actual_value = actual.get(key)
            if actual_value != expected_value:
                mismatches.append(
                    f"{key}: expected={expected_value!r}, actual={actual_value!r}"
                )

        required_tables = {"metadata", "pool", "address_keys", "name_anchor"}
        actual_tables = {
            row[0]
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
        }
        missing_tables = sorted(required_tables - actual_tables)
        if missing_tables:
            mismatches.append(f"missing tables: {missing_tables}")

        if mismatches:
            message = (
                f"Refusing to reuse existing pool {pool_db}. "
                "Pool identity/schema checks failed:\n- "
                + "\n- ".join(mismatches)
            )
            raise PoolValidationError(message)
    finally:
        connection.close()


def open_validated_pool(mode: str, db_path: Path | None = None) -> sqlite3.Connection:
    """Open a mode-specific pool only after identity validation succeeds."""
    spec = make_pool_spec(mode)
    db_path = Path(db_path).resolve() if db_path else pool_path(spec.mode)
    validate_pool(db_path, spec)
    connection = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    connection.row_factory = sqlite3.Row
    return connection


# ---------------------------------------------------------------------------
# Address/name key construction
# ---------------------------------------------------------------------------

def _clean_scalar(value: object) -> str:
    if value is None or pd.isna(value):
        return ""
    return str(value).strip().casefold()


def address_keys(address: object, max_words: int = DEFAULT_ADDRESS_KEY_WORDS) -> set[str]:
    """
    Produce deterministic address blocking keys for one Source 1 record.

    Word selection must match _vectorized_address_key_rows exactly (dedupe
    before truncating to max_words), or query-time keys will miss keys that
    were actually indexed during pool build.
    """
    text = _clean_scalar(address)
    if not text:
        return set()

    numbers = _NUMBER_RE.findall(text)

    seen_words: list[str] = []
    for word in _WORD_RE.findall(text):
        word = word.casefold()
        if len(word) < 4 or word in ADDRESS_STOP_WORDS or word in seen_words:
            continue
        seen_words.append(word)
        if len(seen_words) >= max_words:
            break

    keys: set[str] = set()

    if numbers:
        first_number = numbers[0].lstrip("0") or "0"
        for word in seen_words:
            keys.add(f"{first_number}|{word}")

    for number in numbers:
        if len(number) == 6:
            keys.add(f"pin|{number}")

    return keys


def _name_anchor_series(names: pd.Series, min_length: int) -> pd.Series:
    """
    Vectorized-ish chunk operation: choose one distinctive anchor token per row.

    The longest alphanumeric token is used. Ties are resolved lexicographically
    for deterministic output.
    """
    base = names.fillna("").astype(str).str.casefold()
    token_lists = base.str.findall(_TOKEN_RE)

    exploded = token_lists.rename("token").explode().reset_index()
    if exploded.empty:
        return pd.Series("", index=names.index, dtype="object")

    exploded.rename(columns={exploded.columns[0]: "row_index"}, inplace=True)
    exploded["token"] = exploded["token"].astype(str)
    exploded["length"] = exploded["token"].str.len()
    exploded = exploded[exploded["length"] >= min_length]

    if exploded.empty:
        return pd.Series("", index=names.index, dtype="object")

    exploded = exploded.sort_values(
        ["row_index", "length", "token"],
        ascending=[True, False, True],
        kind="mergesort",
    )
    best = exploded.drop_duplicates("row_index", keep="first").set_index("row_index")["token"]
    return best.reindex(names.index).fillna("")


def _vectorized_address_key_rows(
    chunk: pd.DataFrame,
    max_words: int = DEFAULT_ADDRESS_KEY_WORDS,
) -> pd.DataFrame:
    """Create (country, key, entity_id) rows without per-row Python apply()."""
    if chunk.empty:
        return pd.DataFrame(columns=["country", "key", "entity_id"])

    working = pd.DataFrame(
        {
            "entity_id": chunk["entity_id"].astype(str),
            "country": chunk["country_norm"].fillna("").astype(str).str.strip().str.casefold(),
            "address": chunk["business_address_norm"].fillna("").astype(str).str.strip().str.casefold(),
        },
        index=chunk.index,
    )
    working["row_index"] = working.index

    number_lists = working["address"].str.findall(_NUMBER_RE)
    working["first_number"] = number_lists.str[0].fillna("").str.lstrip("0")
    working.loc[working["first_number"] == "", "first_number"] = "0"

    word_lists = working["address"].str.findall(_WORD_RE)
    words = word_lists.rename("word").explode().reset_index()
    words.rename(columns={words.columns[0]: "row_index"}, inplace=True)
    words["word"] = words["word"].astype(str).str.casefold()
    words = words[words["word"].str.len() >= 4]
    words = words[~words["word"].isin(ADDRESS_STOP_WORDS)]
    words = words.drop_duplicates(["row_index", "word"])
    words = words.groupby("row_index", sort=False).head(max_words)

    if words.empty:
        word_keys = pd.DataFrame(columns=["row_index", "key"])
    else:
        word_keys = words.merge(
            working[["row_index", "first_number"]],
            on="row_index",
            how="inner",
        )
        word_keys["key"] = word_keys["first_number"] + "|" + word_keys["word"]
        word_keys = word_keys[["row_index", "key"]]

    pin_rows = []
    number_rows = working[["row_index", "country", "entity_id", "address"]].copy()
    number_rows["numbers"] = number_lists
    numbers = number_rows[["row_index", "numbers"]].explode("numbers")
    numbers = numbers.dropna(subset=["numbers"])
    numbers["numbers"] = numbers["numbers"].astype(str)
    numbers = numbers[numbers["numbers"].str.len() == 6]
    if not numbers.empty:
        numbers["key"] = "pin|" + numbers["numbers"]
        pin_rows = numbers[["row_index", "key"]]

    pieces = [piece for piece in (word_keys, *([pin_rows] if isinstance(pin_rows, pd.DataFrame) else [])) if not piece.empty]
    if not pieces:
        return pd.DataFrame(columns=["country", "key", "entity_id"])

    keys = pd.concat(pieces, ignore_index=True).drop_duplicates()
    result = keys.merge(
        working[["row_index", "country", "entity_id"]],
        on="row_index",
        how="left",
    )
    return result[["country", "key", "entity_id"]].drop_duplicates()


# ---------------------------------------------------------------------------
# Pool building
# ---------------------------------------------------------------------------

def _check_cleaned_columns(path: Path) -> None:
    header = pd.read_csv(path, sep="\t", nrows=0)
    missing = sorted(_REQUIRED_POOL_COLUMNS - set(header.columns))
    if missing:
        raise ValueError(
            f"Cleaned file {path} is missing required columns: {missing}"
        )


def _insert_pool_chunk(
    connection: sqlite3.Connection,
    chunk: pd.DataFrame,
    source_name: str,
    chunk_number: int,
) -> int:
    required = [
        "entity_id",
        "country_norm",
        "business_name_norm",
        "business_name_core",
        "business_name_alt",
        "business_address_norm",
        "business_address_landmark",
    ]

    working = chunk[required].copy()
    working = working.rename(
        columns={
            "business_name_norm": "name",
            "business_name_core": "name_core",
            "business_name_alt": "name_alt",
            "business_address_norm": "address",
            "business_address_landmark": "landmark",
        }
    )

    working["entity_id"] = working["entity_id"].fillna("").astype(str).str.strip()
    if working["entity_id"].eq("").any():
        raise ValueError(
            f"Blank entity_id found in {source_name}, chunk {chunk_number}; "
            "candidate pool construction refuses to proceed."
        )

    if working["entity_id"].duplicated().any():
        duplicate = working.loc[working["entity_id"].duplicated(), "entity_id"].iloc[0]
        raise ValueError(
            f"Duplicate entity_id {duplicate!r} found within {source_name}, "
            f"chunk {chunk_number}; candidate pool construction refuses to proceed."
        )

    for column in ["country_norm", "name", "name_core", "name_alt", "address", "landmark"]:
        working[column] = (
            working[column]
            .fillna("")
            .astype(str)
            .str.strip()
            .str.casefold()
        )

    rows = list(
        zip(
            working["entity_id"],
            [source_name] * len(working),
            working["country_norm"],
            working["name"],
            working["name_core"],
            working["name_alt"],
            working["address"],
            working["landmark"],
        )
    )

    try:
        connection.executemany(
            """
            INSERT INTO pool(
                entity_id, source, country, name, name_core, name_alt,
                address, landmark
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            rows,
        )
    except sqlite3.IntegrityError as exc:
        raise ValueError(
            f"Pool insert failed for {source_name}, chunk {chunk_number}; "
            "this usually means an entity_id collision across S2/S3. "
            "Entity IDs were not modified."
        ) from exc

    anchor_names = working["name_core"].where(
        working["name_core"].str.len() > 0,
        working["name"],
    )
    anchor_series = _name_anchor_series(anchor_names, DEFAULT_NAME_ANCHOR_MIN_LENGTH)
    anchor_df = pd.DataFrame(
        {
            "entity_id": working["entity_id"].values,
            "country": working["country_norm"].values,
            "anchor": anchor_series.values,
        }
    )
    anchor_df = anchor_df[anchor_df["anchor"].str.len() >= DEFAULT_NAME_ANCHOR_MIN_LENGTH]
    anchor_df = anchor_df.drop_duplicates()

    if not anchor_df.empty:
        connection.executemany(
            "INSERT OR IGNORE INTO name_anchor(country, anchor, entity_id) VALUES (?, ?, ?)",
            anchor_df[["country", "anchor", "entity_id"]].itertuples(index=False, name=None),
        )

    address_input = pd.DataFrame({
        "entity_id": working["entity_id"].values,
        "country_norm": working["country_norm"].values,
        "business_address_norm": working["address"].values,
    }, index=working.index)
    address_df = _vectorized_address_key_rows(
        address_input,
        max_words=DEFAULT_ADDRESS_KEY_WORDS,
    )
    if not address_df.empty:
        connection.executemany(
            "INSERT OR IGNORE INTO address_keys(country, key, entity_id) VALUES (?, ?, ?)",
            address_df[["country", "key", "entity_id"]].itertuples(index=False, name=None),
        )

    return len(working)


def _create_schema(connection: sqlite3.Connection) -> None:
    connection.executescript(
        """
        PRAGMA foreign_keys = ON;

        CREATE TABLE metadata (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL
        );

        CREATE TABLE pool (
            entity_id TEXT PRIMARY KEY,
            source TEXT NOT NULL,
            country TEXT NOT NULL,
            name TEXT NOT NULL,
            name_core TEXT NOT NULL,
            name_alt TEXT NOT NULL,
            address TEXT NOT NULL,
            landmark TEXT NOT NULL
        );

        CREATE TABLE address_keys (
            country TEXT NOT NULL,
            key TEXT NOT NULL,
            entity_id TEXT NOT NULL,
            PRIMARY KEY(country, key, entity_id),
            FOREIGN KEY(entity_id) REFERENCES pool(entity_id)
        );

        CREATE TABLE name_anchor (
            country TEXT NOT NULL,
            anchor TEXT NOT NULL,
            entity_id TEXT NOT NULL,
            PRIMARY KEY(country, anchor, entity_id),
            FOREIGN KEY(entity_id) REFERENCES pool(entity_id)
        );
        """
    )


def _create_indexes(connection: sqlite3.Connection) -> None:
    connection.executescript(
        """
        CREATE INDEX pool_country_name_idx
            ON pool(country, name);
        CREATE INDEX pool_country_core_idx
            ON pool(country, name_core);
        CREATE INDEX pool_country_alt_idx
            ON pool(country, name_alt);
        CREATE INDEX pool_country_address_idx
            ON pool(country, address);
        CREATE INDEX pool_source_country_idx
            ON pool(source, country);
        CREATE INDEX address_keys_lookup_idx
            ON address_keys(country, key);
        CREATE INDEX name_anchor_lookup_idx
            ON name_anchor(country, anchor);
        """
    )


def build_pool(
    mode: str,
    *,
    db_path: Path | None = None,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    overwrite: bool = False,
) -> Path:
    """
    Build a fresh mode-specific SQLite candidate pool.

    The build happens in a temporary SQLite file. The final file is replaced
    only after the complete build and metadata write succeed.

    Existing pools are NOT overwritten unless overwrite=True is passed
    explicitly by the orchestration layer (for example, --rebuild-pool).
    """
    spec = make_pool_spec(mode)
    final_path = Path(db_path).resolve() if db_path else pool_path(spec.mode)
    final_path.parent.mkdir(parents=True, exist_ok=True)

    if final_path.exists() and not overwrite:
        raise PoolValidationError(
            f"Pool already exists at {final_path}. Refusing to overwrite it. "
            "Validate it first or request an explicit rebuild."
        )

    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")

    for source in (spec.source2, spec.source3):
        if not source.path.exists():
            raise FileNotFoundError(
                f"Required cleaned pool source does not exist: {source.path}"
            )
        _check_cleaned_columns(source.path)

    metadata = _metadata_pairs(spec)
    temp_path = final_path.with_suffix(final_path.suffix + ".building")
    if temp_path.exists():
        temp_path.unlink()

    connection = sqlite3.connect(temp_path)
    try:
        connection.execute("PRAGMA journal_mode = DELETE")
        connection.execute("PRAGMA synchronous = NORMAL")
        connection.execute("PRAGMA temp_store = MEMORY")
        _create_schema(connection)

        connection.executemany(
            "INSERT INTO metadata(key, value) VALUES (?, ?)",
            metadata.items(),
        )

        for source in (spec.source2, spec.source3):
            total = 0
            for chunk_number, chunk in enumerate(
                pd.read_csv(
                    source.path,
                    sep="\t",
                    usecols=sorted(_REQUIRED_POOL_COLUMNS),
                    dtype=str,
                    chunksize=chunk_size,
                    keep_default_na=True,
                ),
                start=1,
            ):
                inserted = _insert_pool_chunk(
                    connection,
                    chunk,
                    source.source_name,
                    chunk_number,
                )
                total += inserted
                connection.commit()
                if total % 500_000 == 0:
                    print(
                        f"{source.source_name}: indexed {total:,} records",
                        flush=True,
                    )

            print(
                f"{source.source_name}: finished {total:,} records",
                flush=True,
            )

        _create_indexes(connection)
        connection.commit()
        connection.close()

        # Atomic-ish finalization on the same filesystem.
        if final_path.exists():
            if not overwrite:
                raise PoolValidationError(
                    f"Pool appeared during build at {final_path}; refusing to overwrite it."
                )
            final_path.unlink()
        temp_path.replace(final_path)

        # Final post-build validation. If validation fails, leave the final file
        # present but fail loudly; the caller must inspect rather than silently reuse.
        validate_pool(final_path, spec)
        print(f"Pool ready: {final_path}", flush=True)
        return final_path

    except Exception:
        try:
            connection.close()
        except Exception:
            pass
        if temp_path.exists():
            temp_path.unlink()
        raise


# ---------------------------------------------------------------------------
# Candidate retrieval helpers
# ---------------------------------------------------------------------------

def _exact_lookup(
    connection: sqlite3.Connection,
    sql: str,
    params: Sequence[object],
    limit: int,
) -> list[str]:
    rows = connection.execute(sql, tuple(params)).fetchall()
    return [str(row[0]) for row in rows[:limit]]


def exact_name_candidates(
    connection: sqlite3.Connection,
    country: object,
    name: object,
    *,
    limit: int = DEFAULT_EXACT_LIMIT,
) -> list[str]:
    country_value = _clean_scalar(country)
    name_value = _clean_scalar(name)
    if not country_value or not name_value:
        return []
    return _exact_lookup(
        connection,
        """
        SELECT entity_id
        FROM pool INDEXED BY pool_country_name_idx
        WHERE country = ? AND name = ?
        ORDER BY entity_id
        LIMIT ?
        """,
        (country_value, name_value, limit),
        limit,
    )


def exact_name_alt_candidates(
    connection: sqlite3.Connection,
    country: object,
    name_core: object,
    name_alt: object,
    *,
    limit: int = DEFAULT_EXACT_LIMIT,
) -> list[str]:
    country_value = _clean_scalar(country)
    core_value = _clean_scalar(name_core)
    alt_value = _clean_scalar(name_alt)
    if not country_value:
        return []

    candidates: list[str] = []

    if core_value:
        candidates.extend(
            _exact_lookup(
                connection,
                """
                SELECT entity_id
                FROM pool INDEXED BY pool_country_core_idx
                WHERE country = ? AND name_core = ?
                ORDER BY entity_id
                LIMIT ?
                """,
                (country_value, core_value, limit),
                limit,
            )
        )

    if alt_value:
        candidates.extend(
            _exact_lookup(
                connection,
                """
                SELECT entity_id
                FROM pool INDEXED BY pool_country_alt_idx
                WHERE country = ? AND name_alt = ?
                ORDER BY entity_id
                LIMIT ?
                """,
                (country_value, alt_value, limit),
                limit,
            )
        )

    return dedupe_candidate_ids(candidates)


def exact_address_candidates(
    connection: sqlite3.Connection,
    country: object,
    address: object,
    *,
    limit: int = DEFAULT_EXACT_LIMIT,
) -> list[str]:
    country_value = _clean_scalar(country)
    address_value = _clean_scalar(address)
    if not country_value or not address_value:
        return []
    return _exact_lookup(
        connection,
        """
        SELECT entity_id
        FROM pool INDEXED BY pool_country_address_idx
        WHERE country = ? AND address = ?
        ORDER BY entity_id
        LIMIT ?
        """,
        (country_value, address_value, limit),
        limit,
    )


def address_key_candidates(
    connection: sqlite3.Connection,
    country: object,
    address: object,
    *,
    max_frequency: int = DEFAULT_ADDRESS_KEY_MAX_FREQUENCY,
    max_words: int = DEFAULT_ADDRESS_KEY_WORDS,
) -> list[str]:
    country_value = _clean_scalar(country)
    if not country_value:
        return []

    candidates: list[str] = []
    for key in sorted(address_keys(address, max_words=max_words)):
        count_row = connection.execute(
            """
            SELECT COUNT(*)
            FROM address_keys INDEXED BY address_keys_lookup_idx
            WHERE country = ? AND key = ?
            """,
            (country_value, key),
        ).fetchone()
        frequency = int(count_row[0])
        if frequency == 0 or frequency > max_frequency:
            continue

        rows = connection.execute(
            """
            SELECT entity_id
            FROM address_keys INDEXED BY address_keys_lookup_idx
            WHERE country = ? AND key = ?
            ORDER BY entity_id
            LIMIT ?
            """,
            (country_value, key, max_frequency + 1),
        ).fetchall()
        candidates.extend(str(row[0]) for row in rows)

    return dedupe_candidate_ids(candidates)


def _name_prefix_range(prefix: str) -> tuple[str, str]:
    return prefix, prefix + "\uffff"


def fuzzy_name_candidates(
    connection: sqlite3.Connection,
    country: object,
    name: object,
    name_core: object,
    *,
    prefix_length: int = DEFAULT_FUZZY_PREFIX_LENGTH,
    block_limit: int = DEFAULT_FUZZY_BLOCK_LIMIT,
    top_k: int = DEFAULT_FUZZY_TOP_K,
    threshold: float = DEFAULT_FUZZY_THRESHOLD,
    anchor_min_length: int = DEFAULT_NAME_ANCHOR_MIN_LENGTH,
) -> list[str]:
    """Bounded fuzzy retrieval; all expensive scoring happens on a small block."""
    country_value = _clean_scalar(country)
    name_value = _clean_scalar(name)
    core_value = _clean_scalar(name_core)
    if not country_value or not name_value or len(name_value) < 3:
        return []

    block: dict[str, tuple[str, str, str]] = {}

    # Prefix block: protects runtime by never comparing against the full pool.
    prefix = name_value[:prefix_length]
    lower, upper = _name_prefix_range(prefix)
    for row in connection.execute(
        """
        SELECT entity_id, name, name_core, name_alt
        FROM pool
        WHERE country = ?
          AND name >= ?
          AND name < ?
        ORDER BY entity_id
        LIMIT ?
        """,
        (country_value, lower, upper, block_limit),
    ).fetchall():
        block[str(row[0])] = (str(row[1]), str(row[2]), str(row[3]))

    # Anchor block: helps when an optional prefix does not line up because of
    # leading stop words, token movement, abbreviations, etc.
    anchor_name = core_value or name_value
    anchor_tokens = [
        token.casefold()
        for token in _TOKEN_RE.findall(anchor_name)
        if len(token) >= anchor_min_length
    ]
    if anchor_tokens:
        anchor = sorted(anchor_tokens, key=lambda token: (-len(token), token))[0]
        for row in connection.execute(
            """
            SELECT p.entity_id, p.name, p.name_core, p.name_alt
            FROM name_anchor AS na
            JOIN pool AS p ON p.entity_id = na.entity_id
            WHERE na.country = ? AND na.anchor = ?
            ORDER BY p.entity_id
            LIMIT ?
            """,
            (country_value, anchor, block_limit),
        ).fetchall():
            block[str(row[0])] = (str(row[1]), str(row[2]), str(row[3]))

    scored: list[tuple[float, str]] = []
    for entity_id, (candidate_name, candidate_core, candidate_alt) in block.items():
        scores = [
            fuzz.ratio(name_value, candidate_name),
            fuzz.token_set_ratio(name_value, candidate_name),
        ]
        if core_value and candidate_core:
            scores.extend(
                [
                    fuzz.ratio(core_value, candidate_core),
                    fuzz.token_set_ratio(core_value, candidate_core),
                ]
            )
        if candidate_alt and name_value:
            scores.append(fuzz.token_set_ratio(name_value, candidate_alt))

        best_score = max(scores)
        if best_score >= threshold:
            scored.append((float(best_score), entity_id))

    scored.sort(key=lambda item: (-item[0], item[1]))
    return [entity_id for _, entity_id in scored[:top_k]]


# ---------------------------------------------------------------------------
# Candidate set composition
# ---------------------------------------------------------------------------

def dedupe_candidate_ids(candidate_ids: Iterable[object]) -> list[str]:
    """Deduplicate without changing IDs, returning deterministic sorted IDs."""
    cleaned = {str(candidate_id) for candidate_id in candidate_ids if candidate_id not in (None, "")}
    return sorted(cleaned)


def merge_candidate_sets(
    strategy_results: Mapping[str, Iterable[object]],
    *,
    max_candidates: int | None = DEFAULT_MAX_CANDIDATES,
) -> list[str]:
    """
    Merge candidates in deterministic strategy priority order.

    Strategy order is exact -> address -> fuzzy. Within fuzzy results, the
    order emitted by the strategy is retained. Exact candidate ordering is
    deterministic by entity ID.
    """
    ordered: list[str] = []
    seen: set[str] = set()

    priority = [
        STRATEGY_EXACT_NAME,
        STRATEGY_EXACT_NAME_ALT,
        STRATEGY_EXACT_ADDRESS,
        STRATEGY_ADDRESS_KEY,
        STRATEGY_FUZZY_NAME,
    ]

    for strategy in priority:
        values = strategy_results.get(strategy, ())
        if strategy == STRATEGY_FUZZY_NAME:
            iterable = values
        else:
            iterable = sorted({str(value) for value in values if value not in (None, "")})

        for candidate_id in iterable:
            candidate_id = str(candidate_id)
            if not candidate_id or candidate_id in seen:
                continue
            seen.add(candidate_id)
            ordered.append(candidate_id)
            if max_candidates is not None and len(ordered) >= max_candidates:
                return ordered

    return ordered


def generate_candidates(
    connection: sqlite3.Connection,
    *,
    country: object,
    name: object,
    name_core: object = "",
    name_alt: object = "",
    address: object = "",
    config: CandidateConfig | None = None,
) -> list[str]:
    """Generate a final deduplicated candidate list for one Source 1 record."""
    config = config or CandidateConfig()
    strategy_results: dict[str, list[str]] = {}

    enabled = set(config.enabled_strategies)

    if STRATEGY_EXACT_NAME in enabled:
        strategy_results[STRATEGY_EXACT_NAME] = exact_name_candidates(
            connection,
            country,
            name,
            limit=config.exact_limit,
        )

    if STRATEGY_EXACT_NAME_ALT in enabled:
        strategy_results[STRATEGY_EXACT_NAME_ALT] = exact_name_alt_candidates(
            connection,
            country,
            name_core,
            name_alt,
            limit=config.exact_limit,
        )

    if STRATEGY_EXACT_ADDRESS in enabled:
        strategy_results[STRATEGY_EXACT_ADDRESS] = exact_address_candidates(
            connection,
            country,
            address,
            limit=config.exact_limit,
        )

    if STRATEGY_ADDRESS_KEY in enabled:
        strategy_results[STRATEGY_ADDRESS_KEY] = address_key_candidates(
            connection,
            country,
            address,
            max_frequency=config.address_key_max_frequency,
        )

    if STRATEGY_FUZZY_NAME in enabled:
        strategy_results[STRATEGY_FUZZY_NAME] = fuzzy_name_candidates(
            connection,
            country,
            name,
            name_core,
            prefix_length=config.fuzzy_prefix_length,
            block_limit=config.fuzzy_block_limit,
            top_k=config.fuzzy_top_k,
            threshold=config.fuzzy_threshold,
            anchor_min_length=config.name_anchor_min_length,
        )

    return merge_candidate_sets(
        strategy_results,
        max_candidates=config.max_candidates,
    )


__all__ = [
    "ALL_STRATEGIES",
    "CandidateConfig",
    "PoolSpec",
    "PoolSource",
    "PoolValidationError",
    "STRATEGY_EXACT_ADDRESS",
    "STRATEGY_EXACT_NAME",
    "STRATEGY_EXACT_NAME_ALT",
    "STRATEGY_ADDRESS_KEY",
    "STRATEGY_FUZZY_NAME",
    "address_keys",
    "address_key_candidates",
    "build_pool",
    "candidate_output_path",
    "cleaned_pool_sources",
    "dedupe_candidate_ids",
    "exact_address_candidates",
    "exact_name_alt_candidates",
    "exact_name_candidates",
    "fuzzy_name_candidates",
    "generate_candidates",
    "make_pool_spec",
    "merge_candidate_sets",
    "normalize_mode",
    "open_validated_pool",
    "pool_path",
    "validate_pool",
]
