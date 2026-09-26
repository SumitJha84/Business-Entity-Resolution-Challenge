from collections import defaultdict
import sqlite3

import pandas as pd

from src.blocking import DB_PATH, address_keys
from difflib import SequenceMatcher
LIMIT_S1 = 5_000
MAX_RECORDS_PER_KEY = 200


def main():
    truth = pd.read_csv(
        "output/splits/val_ground_truth.tsv", sep="\t", dtype=str
    )
    truth_by_id = {
        row.source1_entity_id: set(
            row.matched_entity_ids.split(",")
            if pd.notna(row.matched_entity_ids) else []
        )
        for row in truth.itertuples(index=False)
    }

    # Select the same first 5,000 validation records as blocking.py.
    s1_info = {}
    s1_keys = {}
    for chunk in pd.read_csv(
        "cleaned_dataset/cleaned_source1.tsv",
        sep="\t",
        usecols=["entity_id","business_name_norm", "business_address_norm", "country_norm"],
        dtype=str,
        chunksize=50_000,
    ):
        for row in chunk.itertuples(index=False):
            if row.entity_id not in truth_by_id:
                continue

            name = (
                row.business_name_norm.strip().casefold()
                if pd.notna(row.business_name_norm) else ""
            )
            address = (
                row.business_address_norm.strip().casefold()
                if pd.notna(row.business_address_norm) else ""
            )
            
            country = (
                row.country_norm.strip().casefold()
                if pd.notna(row.country_norm) else ""
            )
            s1_info[row.entity_id] = (country, name, address)

            s1_keys[row.entity_id] = {
                (country, key) for key in address_keys(row.business_address_norm)
            }
            if len(s1_keys) == LIMIT_S1:
                break
        if len(s1_keys) == LIMIT_S1:
            break

    wanted_keys = set().union(*s1_keys.values())
    key_to_pool = defaultdict(list)
    crowded = set()

    # Scan the existing pool once. Store only records matching wanted keys.
    connection = sqlite3.connect(DB_PATH)
    cursor = connection.execute("SELECT entity_id, country, address FROM pool")
    scanned = 0

    while batch := cursor.fetchmany(50_000):
        for entity_id, country, address in batch:
            for key in address_keys(address):
                full_key = (country, key)
                if full_key not in wanted_keys or full_key in crowded:
                    continue

                records = key_to_pool[full_key]
                records.append(entity_id)
                if len(records) > MAX_RECORDS_PER_KEY:
                    crowded.add(full_key)
                    del key_to_pool[full_key]

        scanned += len(batch)
        if scanned % 1_000_000 < 50_000:
            print(f"Scanned {scanned:,} pool records")

    # Keep the connection open for the name/address lookups below.

    connection.close()

    connection = sqlite3.connect(DB_PATH)

    true_links = found_links = total_candidates = 0

    for s1_id, keys in s1_keys.items():
        country, name, address = s1_info[s1_id]

        # Address-key candidates from the pool scan.
        candidates = set()
        for key in keys:
            candidates.update(key_to_pool.get(key, []))

        # Existing exact-name route.
        if name:
            candidates.update(
                item[0]
                for item in connection.execute(
                    "SELECT entity_id FROM pool WHERE country = ? AND name = ? LIMIT 20",
                    (country, name),
                )
            )

        # Existing exact-address route.
        if address:
            candidates.update(
                item[0]
                for item in connection.execute(
                    "SELECT entity_id FROM pool WHERE country = ? AND address = ? LIMIT 20",
                    (country, address),
                )
            )

        # Existing five-character fuzzy-name route.
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
                key=lambda item: SequenceMatcher(None, name, item[1]).ratio(),
                reverse=True,
            )
            candidates.update(entity_id for entity_id, _ in ranked[:20])

        actual = truth_by_id[s1_id]
        true_links += len(actual)
        found_links += len(actual & candidates)
        total_candidates += len(candidates)

    connection.close()

    print(f"S1 records: {len(s1_keys):,}")
    print(f"Crowded address keys skipped: {len(crowded):,}")
    print(f"Combined true links found: {found_links:,} / {true_links:,}")
    print(f"Combined blocking recall: {found_links / true_links:.2%}")
    print(f"Average combined candidates per S1: {total_candidates / len(s1_keys):.1f}")

if __name__ == "__main__":
    main()