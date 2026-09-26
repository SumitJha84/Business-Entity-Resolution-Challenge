"""
generate_test_candidates.py
===========================
Stage 3A: Candidate generation orchestration for Business Entity Resolution.

This module intentionally contains orchestration only. The reusable blocking
logic lives in ``src/blocking.py``.

Contracts
---------
- Train and test are isolated by mode-specific cleaned Source 1/2/3 paths and
  mode-specific SQLite pools.
- Source 1 is always processed in chunks; ``--chunk-size`` is NOT a total-row
  limit.
- ``--limit`` is a strict maximum on the number of Source 1 records emitted
  to the candidate output. It is the primary safety control for smoke tests.
- Ground truth is never imported or read during candidate generation.
- Every processed Source 1 record produces exactly one output row, including
  rows with an empty candidate list.
- Entity IDs are written exactly as supplied; they are never normalized.
- Candidate output uses the contract expected by ``prepare_training_data.py``:
      source1_entity_id    candidate_entity_ids
- Existing pools are validated against their exact mode + source-file
  fingerprints before reuse. Mismatches fail loudly and are never silently
  rebuilt/reused.
- Pool construction is explicit via ``--build-pool`` or ``--rebuild-pool``;
  a smoke test will never silently trigger a full S2/S3 pool build.

Smoke-test safety
-----------------
For test-mode smoke tests, use ``--smoke-test --limit N``. This automatically
tries to include a few France rows when France exists in the cleaned test
Source 1 file. The total emitted Source 1 rows still never exceeds ``N``.
"""

from __future__ import annotations

import argparse
import csv
import logging
import sys
import time
from pathlib import Path
from typing import Iterable, Sequence

import pandas as pd

try:
    from src import blocking
    from src import config as cfg
except ImportError:  # pragma: no cover - supports direct execution from src/
    import blocking
    import config as cfg


# ---------------------------------------------------------------------------
# Paths / schema
# ---------------------------------------------------------------------------

SOURCE1_REQUIRED_COLUMNS = (
    "entity_id",
    "country_norm",
    "business_name_norm",
    "business_name_core",
    "business_name_alt",
    "business_address_norm",
)

OUTPUT_COLUMNS = ("source1_entity_id", "candidate_entity_ids")


class CandidateGenerationError(RuntimeError):
    """Raised when candidate generation cannot safely proceed."""


def normalize_mode(mode: str) -> str:
    mode = mode.strip().lower()
    if mode not in {"train", "test"}:
        raise ValueError("mode must be exactly 'train' or 'test'")
    return mode


def source1_path(mode: str) -> Path:
    mode = normalize_mode(mode)
    if mode == "train":
        return Path(cfg.CLEANED_SOURCE1).resolve()
    return (Path(cfg.CLEANED_DIR) / "cleaned_test_source1.tsv").resolve()


def default_output_path(mode: str, limit: int | None) -> Path:
    mode = normalize_mode(mode)
    blocking_dir = Path(cfg.BLOCKING_DIR).resolve()
    if limit is not None:
        return blocking_dir / f"smoke_candidate_pairs_{mode}_{limit}.tsv"
    if mode == "train":
        # Required default for prepare_training_data.py.
        return blocking_dir / "candidate_pairs.tsv"
    return blocking_dir / "candidate_pairs_test.tsv"


def validate_source1_columns(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Cleaned Source 1 file not found: {path}")

    header = pd.read_csv(path, sep="\t", nrows=0)
    missing = [column for column in SOURCE1_REQUIRED_COLUMNS if column not in header.columns]
    if missing:
        raise CandidateGenerationError(
            f"Source 1 file {path} is missing required cleaned columns: {missing}"
        )


def build_candidate_config(args: argparse.Namespace) -> blocking.CandidateConfig:
    disabled = {
        strategy
        for strategy, is_disabled in (
            (blocking.STRATEGY_EXACT_NAME, args.disable_exact_name),
            (blocking.STRATEGY_EXACT_NAME_ALT, args.disable_exact_name_alt),
            (blocking.STRATEGY_EXACT_ADDRESS, args.disable_exact_address),
            (blocking.STRATEGY_ADDRESS_KEY, args.disable_address_key),
            (blocking.STRATEGY_FUZZY_NAME, args.disable_fuzzy_name),
        )
        if is_disabled
    }

    enabled = tuple(
        strategy
        for strategy in blocking.ALL_STRATEGIES
        if strategy not in disabled
    )

    if not enabled:
        raise CandidateGenerationError(
            "All blocking strategies are disabled. Enable at least one strategy."
        )

    return blocking.CandidateConfig(
        enabled_strategies=enabled,
        max_candidates=args.max_candidates,
        exact_limit=args.exact_limit,
        address_key_max_frequency=args.address_key_max_frequency,
        fuzzy_prefix_length=args.fuzzy_prefix_length,
        fuzzy_block_limit=args.fuzzy_block_limit,
        fuzzy_top_k=args.fuzzy_top_k,
        fuzzy_threshold=args.fuzzy_threshold,
        name_anchor_min_length=args.name_anchor_min_length,
    )


# ---------------------------------------------------------------------------
# Pool lifecycle
# ---------------------------------------------------------------------------


def ensure_valid_pool(
    mode: str,
    *,
    chunk_size: int,
    build_pool: bool,
    rebuild_pool: bool,
    logger: logging.Logger,
) -> Path:
    """
    Validate or explicitly build the mode-specific pool.

    Safety rule:
      - Existing pool -> validate exact identity before reuse.
      - Missing pool -> refuse unless --build-pool/--rebuild-pool is explicit.
      - Existing mismatched pool -> refuse unless --rebuild-pool is explicit.
    """
    mode = normalize_mode(mode)
    db_path = blocking.pool_path(mode)
    spec = blocking.make_pool_spec(mode)

    if rebuild_pool and build_pool:
        raise CandidateGenerationError(
            "Use either --build-pool or --rebuild-pool, not both."
        )

    if rebuild_pool:
        logger.info("Explicit --rebuild-pool requested: rebuilding %s", db_path)
        blocking.build_pool(
            mode,
            db_path=db_path,
            chunk_size=chunk_size,
            overwrite=True,
        )
        return db_path

    if db_path.exists():
        logger.info("Existing pool found; validating identity before reuse: %s", db_path)
        try:
            blocking.validate_pool(db_path, spec)
        except blocking.PoolValidationError as exc:
            raise CandidateGenerationError(
                "Existing pool validation failed. Refusing to reuse or silently rebuild it. "
                f"Details:\n{exc}\n\n"
                "Use --rebuild-pool only after confirming that the cleaned source files "
                "for this mode are the intended inputs."
            ) from exc
        logger.info("Pool validation passed: %s", db_path)
        return db_path

    if not build_pool:
        raise CandidateGenerationError(
            f"Required {mode} pool does not exist: {db_path}\n"
            "Refusing to silently build a potentially full S2/S3 pool. "
            "Run again with --build-pool when you explicitly want to build it."
        )

    logger.info("No pool exists; explicit --build-pool requested: %s", db_path)
    blocking.build_pool(
        mode,
        db_path=db_path,
        chunk_size=chunk_size,
        overwrite=False,
    )
    return db_path


# ---------------------------------------------------------------------------
# Source 1 iteration / smoke selection
# ---------------------------------------------------------------------------


def _read_source1_chunks(path: Path, chunk_size: int) -> Iterable[pd.DataFrame]:
    return pd.read_csv(
        path,
        sep="\t",
        usecols=list(SOURCE1_REQUIRED_COLUMNS),
        dtype=str,
        chunksize=chunk_size,
        keep_default_na=False,
        na_filter=False,
    )


def _row_country_matches(row: object, wanted_country: str) -> bool:
    try:
        value = getattr(row, "country_norm")
    except AttributeError:
        return False
    return str(value).strip().casefold() == wanted_country.strip().casefold()


def iter_source1_rows(
    path: Path,
    *,
    chunk_size: int,
    limit: int | None,
    ensure_country: str | None,
    ensure_country_count: int,
    logger: logging.Logger,
) -> Iterable[object]:
    """
    Yield Source 1 rows without loading the file into memory.

    ``limit`` counts emitted Source 1 rows, not chunks read.

    When ensure_country is requested, the selector keeps at most ``limit`` rows
    in memory and continues reading until the requested country count is found
    or EOF is reached. It may therefore READ more than ``limit`` rows, but it
    will never GENERATE or EMIT more than ``limit`` candidate rows.
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if limit is not None and limit <= 0:
        raise ValueError("limit must be positive when provided")
    if ensure_country_count <= 0:
        raise ValueError("ensure_country_count must be positive")
    if ensure_country is not None and limit is None:
        raise CandidateGenerationError(
            "--ensure-country requires --limit so the smoke-test output remains bounded."
        )

    columns = list(SOURCE1_REQUIRED_COLUMNS)

    if ensure_country is None:
        emitted = 0
        for chunk in _read_source1_chunks(path, chunk_size):
            for row in chunk.itertuples(index=False, name="Source1Row"):
                yield row
                emitted += 1
                if limit is not None and emitted >= limit:
                    return
        return

    # Bounded smoke selector: preserve the first rows while reserving slots for
    # the requested country. We stop scanning as soon as the requested count is
    # reached. If the country does not exist, we scan to EOF and keep the first
    # ``limit`` rows only.
    selected: list[object] = []
    wanted = ensure_country.strip().casefold()
    target_count = 0
    emitted_base = 0
    scanned = 0
    seen_target_ids: set[str] = set()

    for chunk in _read_source1_chunks(path, chunk_size):
        for row in chunk.itertuples(index=False, name="Source1Row"):
            scanned += 1
            row_is_target = _row_country_matches(row, wanted)

            if len(selected) < limit:
                selected.append(row)
                emitted_base += 1
                if row_is_target:
                    target_count += 1
                    seen_target_ids.add(str(row.entity_id))
            elif row_is_target and target_count < ensure_country_count:
                # Replace the last non-target row. This keeps total selected
                # rows exactly bounded by ``limit``.
                replacement_index = next(
                    (
                        index
                        for index, existing in enumerate(selected)
                        if not _row_country_matches(existing, wanted)
                    ),
                    None,
                )
                if replacement_index is not None:
                    replaced = selected[replacement_index]
                    if _row_country_matches(replaced, wanted):
                        continue
                    selected[replacement_index] = row
                    target_count += 1
                    seen_target_ids.add(str(row.entity_id))

            if len(selected) >= limit and target_count >= ensure_country_count:
                break

        if len(selected) >= limit and target_count >= ensure_country_count:
            break

    logger.info(
        "Bounded smoke selector: scanned %s Source1 rows; selected %s rows; %s rows from country=%s.",
        f"{scanned:,}",
        f"{len(selected):,}",
        f"{target_count:,}",
        ensure_country,
    )

    # The selected list contains at most ``limit`` rows and each input row was
    # selected once. Sort nothing and preserve deterministic selection order.
    for row in selected:
        yield row

    if target_count < ensure_country_count:
        logger.warning(
            "Country safety check could not find %s rows for country=%r. "
            "Found %s. Candidate generation remains bounded by --limit.",
            ensure_country_count,
            ensure_country,
            target_count,
        )


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------


def _prepare_output_path(path: Path, overwrite: bool) -> tuple[Path, Path]:
    path = Path(path).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)

    if path.exists() and not overwrite:
        raise CandidateGenerationError(
            f"Output already exists: {path}\n"
            "Refusing to overwrite it. Use --overwrite-output explicitly."
        )

    temp_path = path.with_suffix(path.suffix + ".building")
    if temp_path.exists():
        temp_path.unlink()
    return path, temp_path


def _atomic_replace(temp_path: Path, final_path: Path) -> None:
    if final_path.exists():
        final_path.unlink()
    temp_path.replace(final_path)


# ---------------------------------------------------------------------------
# Main generation loop
# ---------------------------------------------------------------------------


def generate_candidate_file(
    *,
    mode: str,
    source1: Path,
    output_path: Path,
    pool_path: Path,
    chunk_size: int,
    limit: int | None,
    candidate_config: blocking.CandidateConfig,
    ensure_country: str | None,
    ensure_country_count: int,
    overwrite_output: bool,
    logger: logging.Logger,
) -> dict[str, object]:
    mode = normalize_mode(mode)
    validate_source1_columns(source1)

    final_path, temp_path = _prepare_output_path(output_path, overwrite_output)

    processed = 0
    nonempty = 0
    empty = 0
    total_candidates = 0
    max_seen_candidates = 0
    capped_rows = 0
    total_read_rows_before_limit = 0
    started = time.time()

    logger.info("Opening validated pool: %s", pool_path)
    connection = blocking.open_validated_pool(mode, db_path=pool_path)
    pool_counts_by_source = {
        str(row[0]): int(row[1])
        for row in connection.execute(
            "SELECT source, COUNT(*) FROM pool GROUP BY source"
        ).fetchall()
    }

    try:
        with temp_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
            writer.writerow(OUTPUT_COLUMNS)

            for row in iter_source1_rows(
                source1,
                chunk_size=chunk_size,
                limit=limit,
                ensure_country=ensure_country,
                ensure_country_count=ensure_country_count,
                logger=logger,
            ):
                source1_id = str(row.entity_id)
                if source1_id == "":
                    raise CandidateGenerationError(
                        f"Blank Source1 entity_id encountered after reading input: {source1}"
                    )

                candidates = blocking.generate_candidates(
                    connection,
                    country=row.country_norm,
                    name=row.business_name_norm,
                    name_core=row.business_name_core,
                    name_alt=row.business_name_alt,
                    address=row.business_address_norm,
                    config=candidate_config,
                )

                # ``generate_candidates`` already deduplicates. Still assert the
                # invariant here so a future blocking change cannot silently break
                # the output contract.
                if len(candidates) != len(set(candidates)):
                    raise CandidateGenerationError(
                        f"Duplicate candidate IDs generated for Source1 entity {source1_id!r}"
                    )

                writer.writerow((source1_id, ",".join(candidates)))

                processed += 1
                candidate_count = len(candidates)
                total_candidates += candidate_count
                max_seen_candidates = max(max_seen_candidates, candidate_count)

                if (
                    candidate_config.max_candidates is not None
                    and candidate_count == candidate_config.max_candidates
                ):
                    capped_rows += 1

                if candidate_count:
                    nonempty += 1
                else:
                    empty += 1

                if limit is not None and processed >= limit:
                    break

                if processed % 10_000 == 0:
                    elapsed = time.time() - started
                    logger.info(
                        "Processed Source1 records: %s | non-empty: %s | empty: %s | "
                        "candidates emitted: %s | elapsed: %.1fs",
                        f"{processed:,}",
                        f"{nonempty:,}",
                        f"{empty:,}",
                        f"{total_candidates:,}",
                        elapsed,
                    )

        if limit is not None and processed > limit:
            raise CandidateGenerationError(
                f"Internal limit violation: processed {processed:,} rows with limit={limit:,}"
            )

        _atomic_replace(temp_path, final_path)
    except Exception:
        try:
            temp_path.unlink(missing_ok=True)
        except OSError:
            pass
        raise
    finally:
        connection.close()

    elapsed = time.time() - started
    report = {
        "mode": mode,
        "source1_path": str(source1),
        "pool_path": str(pool_path),
        "output_path": str(final_path),
        "chunk_size": int(chunk_size),
        "limit": int(limit) if limit is not None else None,
        "processed_source1_rows": int(processed),
        "non_empty_candidate_rows": int(nonempty),
        "empty_candidate_rows": int(empty),
        "total_candidate_ids_emitted": int(total_candidates),
        "max_candidates_in_one_row": int(max_seen_candidates),
        "rows_at_candidate_cap": int(capped_rows),
        "pool_counts_by_source": pool_counts_by_source,
        "enabled_strategies": list(candidate_config.enabled_strategies),
        "elapsed_seconds": float(elapsed),
    }
    logger.info("Candidate generation complete: %s", report)
    return report


# ---------------------------------------------------------------------------
# Logging / CLI
# ---------------------------------------------------------------------------


def setup_logger() -> logging.Logger:
    logger = logging.getLogger("generate_test_candidates")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
    logger.addHandler(handler)
    return logger


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate blocking candidate lists for train or test Source1 data. "
            "Use --limit for a strict bounded smoke test."
        )
    )

    parser.add_argument("--mode", choices=("train", "test"), required=True)
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=cfg.CHUNK_SIZE,
        help=f"Source1 read chunk size only; not a total-row limit (default: {cfg.CHUNK_SIZE:,}).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Strict maximum number of Source1 records to generate. Omit only for an intentionally full run.",
    )
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Marks the run as a bounded smoke test. Requires --limit; in test mode it tries to include France rows.",
    )
    parser.add_argument(
        "--ensure-country",
        type=str,
        default=None,
        help="For bounded smoke tests, ensure up to this country appears in the sample when present (for example: France).",
    )
    parser.add_argument(
        "--ensure-country-count",
        type=int,
        default=3,
        help="Number of rows of --ensure-country to try to include (default: 3).",
    )
    parser.add_argument(
        "--source1-path",
        type=Path,
        default=None,
        help="Optional override for cleaned Source1 path. Mode-specific default is used when omitted.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional output TSV. Mode/limit-specific safe default is used when omitted.",
    )

    parser.add_argument(
        "--build-pool",
        action="store_true",
        help="Explicitly build a missing mode-specific S2/S3 SQLite pool.",
    )
    parser.add_argument(
        "--rebuild-pool",
        action="store_true",
        help="Explicitly replace the existing mode-specific pool after rebuilding it from current cleaned S2/S3 files.",
    )
    parser.add_argument(
        "--overwrite-output",
        action="store_true",
        help="Allow replacement of an existing candidate output file.",
    )

    # Independent strategy controls.
    parser.add_argument("--disable-exact-name", action="store_true")
    parser.add_argument("--disable-exact-name-alt", action="store_true")
    parser.add_argument("--disable-exact-address", action="store_true")
    parser.add_argument("--disable-address-key", action="store_true")
    parser.add_argument("--disable-fuzzy-name", action="store_true")

    # Retrieval tuning.
    parser.add_argument("--max-candidates", type=int, default=blocking.DEFAULT_MAX_CANDIDATES)
    parser.add_argument("--exact-limit", type=int, default=blocking.DEFAULT_EXACT_LIMIT)
    parser.add_argument(
        "--address-key-max-frequency",
        type=int,
        default=blocking.DEFAULT_ADDRESS_KEY_MAX_FREQUENCY,
    )
    parser.add_argument(
        "--fuzzy-prefix-length",
        type=int,
        default=blocking.DEFAULT_FUZZY_PREFIX_LENGTH,
    )
    parser.add_argument(
        "--fuzzy-block-limit",
        type=int,
        default=blocking.DEFAULT_FUZZY_BLOCK_LIMIT,
    )
    parser.add_argument("--fuzzy-top-k", type=int, default=blocking.DEFAULT_FUZZY_TOP_K)
    parser.add_argument(
        "--fuzzy-threshold",
        type=float,
        default=blocking.DEFAULT_FUZZY_THRESHOLD,
    )
    parser.add_argument(
        "--name-anchor-min-length",
        type=int,
        default=blocking.DEFAULT_NAME_ANCHOR_MIN_LENGTH,
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logger = setup_logger()

    mode = normalize_mode(args.mode)

    if args.chunk_size <= 0:
        raise CandidateGenerationError("--chunk-size must be positive")
    if args.limit is not None and args.limit <= 0:
        raise CandidateGenerationError("--limit must be positive when provided")
    if args.smoke_test and args.limit is None:
        raise CandidateGenerationError("--smoke-test requires an explicit --limit")
    if args.ensure_country_count <= 0:
        raise CandidateGenerationError("--ensure-country-count must be positive")
    if args.ensure_country is not None and args.limit is None:
        raise CandidateGenerationError("--ensure-country requires --limit")
    if args.smoke_test and mode == "test" and args.ensure_country is None:
        args.ensure_country = "France"

    source1 = Path(args.source1_path).resolve() if args.source1_path else source1_path(mode)
    output = (
        Path(args.output).resolve()
        if args.output
        else default_output_path(mode, args.limit)
    )

    candidate_config = build_candidate_config(args)

    logger.info("=" * 72)
    logger.info("Candidate Generation — START")
    logger.info("Mode:              %s", mode)
    logger.info("Source1:            %s", source1)
    logger.info("Output:             %s", output)
    logger.info("Chunk size:         %s", f"{args.chunk_size:,}")
    logger.info("TOTAL row limit:    %s", f"{args.limit:,}" if args.limit is not None else "NONE (full run requested)")
    logger.info("Enabled strategies: %s", ", ".join(candidate_config.enabled_strategies))
    logger.info("Pool path:          %s", blocking.pool_path(mode))
    logger.info("=" * 72)

    if args.limit is None:
        logger.warning(
            "No --limit supplied. This is an intentionally unbounded/full Source1 run. "
            "For smoke testing, stop and use an explicit --limit."
        )

    pool_db = ensure_valid_pool(
        mode,
        chunk_size=args.chunk_size,
        build_pool=args.build_pool,
        rebuild_pool=args.rebuild_pool,
        logger=logger,
    )

    report = generate_candidate_file(
        mode=mode,
        source1=source1,
        output_path=output,
        pool_path=pool_db,
        chunk_size=args.chunk_size,
        limit=args.limit,
        candidate_config=candidate_config,
        ensure_country=args.ensure_country,
        ensure_country_count=args.ensure_country_count,
        overwrite_output=args.overwrite_output,
        logger=logger,
    )

    logger.info("Output file: %s", report["output_path"])
    logger.info("Source1 rows emitted: %s", f"{report['processed_source1_rows']:,}")
    logger.info("Rows with candidates: %s", f"{report['non_empty_candidate_rows']:,}")
    logger.info("Rows without candidates: %s", f"{report['empty_candidate_rows']:,}")
    logger.info("=" * 72)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Interrupted by user.", file=sys.stderr)
        raise SystemExit(130)
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)
