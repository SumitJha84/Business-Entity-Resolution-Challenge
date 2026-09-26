import argparse
import csv
import sqlite3
from rapidfuzz.fuzz import ratio
from pathlib import Path

import pandas as pd

from src.blocking import address_keys
from src.config import BLOCKING_DIR

DB_PATH = Path("output/blocking_test_pool.sqlite")
S1_PATH = Path("cleaned_dataset/cleaned_test_source1.tsv")
OUTPUT_PATH = BLOCKING_DIR / "candidate_pairs.tsv"


def get_candidates(connection, country, name, address):
    candidates = set()

    if name:
        candidates.update(
            entity_id
            for (entity_id,) in connection.execute(
                "SELECT entity_id FROM pool WHERE country = ? AND name = ? LIMIT 20",
                (country, name),
            )
        )

    if address:
        candidates.update(
            entity_id
            for (entity_id,) in connection.execute(
                "SELECT entity_id FROM pool WHERE country = ? AND address = ? LIMIT 20",
                (country, address),
            )
        )

    if len(name) >= 5:
        prefix = name[:5]
        possible = connection.execute(
            """
            SELECT entity_id, name FROM pool
            WHERE country = ? AND name >= ? AND name < ?
            LIMIT 300
            """,
            (country, prefix, prefix + "\uffff"),
        ).fetchall()

        ranked = sorted(
            possible,
            key=lambda item: ratio(name, item[1]),
            reverse=True,
        )
        candidates.update(entity_id for entity_id, _ in ranked[:20])

    for key in address_keys(address):
        rows = connection.execute(
            """
            SELECT entity_id FROM address_keys INDEXED BY address_key_cover_idx
            WHERE country = ? AND key = ?
            LIMIT 201
            """,
            (country, key),
        ).fetchall()

        if len(rows) <= 200:
            candidates.update(entity_id for (entity_id,) in rows)

    return sorted(candidates)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, help="Process only this many S1 rows")
    args = parser.parse_args()

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True)
    processed = 0

    with OUTPUT_PATH.open("w", newline="") as output:
        writer = csv.writer(output, delimiter="\t")
        writer.writerow(["source1_entity_id", "candidate_entity_ids"])

        for chunk in pd.read_csv(
            S1_PATH,
            sep="\t",
            usecols=[
                "entity_id",
                "country_norm",
                "business_name_norm",
                "business_address_norm",
            ],
            dtype=str,
            chunksize=10_000,
        ):
            for row in chunk.fillna("").itertuples(index=False):
                country = row.country_norm.strip().casefold()
                name = row.business_name_norm.strip().casefold()
                address = row.business_address_norm.strip().casefold()

                candidates = get_candidates(connection, country, name, address)
                writer.writerow([row.entity_id, ",".join(candidates)])
                processed += 1

                if processed % 1_000 == 0:
                    print(f"Processed {processed:,} test S1 records", flush=True)

                if args.limit and processed >= args.limit:
                    connection.close()
                    print(f"Sample written to {OUTPUT_PATH}")
                    return

    connection.close()
    print(f"Done: {processed:,} test S1 records written to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()