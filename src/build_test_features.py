import argparse
import os
import sqlite3
import time
from pathlib import Path

import pandas as pd
from rapidfuzz import fuzz

from src.build_pair_features import fetch_candidates, load_s1, similarity


COLUMNS = [
    "source1_entity_id",
    "candidate_entity_id",
    "name_similarity",
    "name_token_similarity",
    "address_similarity",
    "address_token_similarity",
    "country_match",
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--source1", type=Path, required=True)
    parser.add_argument("--pool", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-parts", type=int, default=None)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    print("Loading test S1 lookup...", flush=True)
    s1 = load_s1(args.source1)
    connection = sqlite3.connect(
        args.pool.resolve().as_uri() + "?mode=ro", uri=True
    )

    started = time.monotonic()
    reader = pd.read_csv(
        args.candidates,
        sep="\t",
        dtype=str,
        keep_default_na=False,
        chunksize=500,
    )

    for part_number, chunk in enumerate(reader):
        if args.max_parts is not None and part_number >= args.max_parts:
            break

        output = args.output_dir / f"part_{part_number:05d}.tsv"
        if output.exists():
            print(f"Part {part_number}: already complete", flush=True)
            continue

        pairs = []
        for row in chunk.itertuples(index=False):
            if row.candidate_entity_ids:
                pairs.extend(
                    (row.source1_entity_id, candidate_id)
                    for candidate_id in row.candidate_entity_ids.split(",")
                    if candidate_id
                )

        candidates = fetch_candidates(
            connection, [candidate_id for _, candidate_id in pairs]
        )
        rows = []

        for s1_id, candidate_id in pairs:
            left = s1.get(s1_id)
            right = candidates.get(candidate_id)
            if left is None or right is None:
                raise ValueError(f"Missing record: {s1_id}, {candidate_id}")

            name1, address1, country1 = left
            name2, address2, country2 = right
            rows.append([
                s1_id,
                candidate_id,
                similarity(name1, name2, fuzz.ratio),
                similarity(name1, name2, fuzz.token_set_ratio),
                similarity(address1, address2, fuzz.ratio),
                similarity(address1, address2, fuzz.token_set_ratio),
                int(country1 == country2),
            ])

        temporary = output.with_suffix(".tsv.tmp")
        pd.DataFrame(rows, columns=COLUMNS).to_csv(
            temporary, sep="\t", index=False
        )
        os.replace(temporary, output)
        elapsed = (time.monotonic() - started) / 60
        print(
            f"Part {part_number}: {len(chunk)} S1, "
            f"{len(rows):,} pairs; {elapsed:.1f} minutes elapsed",
            flush=True,
        )

    connection.close()


if __name__ == "__main__":
    main()
