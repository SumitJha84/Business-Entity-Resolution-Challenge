import sqlite3
from pathlib import Path
import re

import pandas as pd
from difflib import SequenceMatcher

POOL_FILES = [
    Path("cleaned_dataset/cleaned_source2.tsv"),
    Path("cleaned_dataset/cleaned_source3.tsv"),
]
DB_PATH = Path("output/blocking_pool.sqlite")


ADDRESS_STOP_WORDS = {
    "road", "street", "saint", "city", "district", "state",
    "near", "west", "east", "north", "south", "of", "box"
}


def address_keys(address):
    if pd.isna(address):
        return set()

    text = str(address).casefold()
    numbers = re.findall(r"\b[0-9]+\b", text)
    if not numbers:
        return set()

    words = re.findall(r"[^\W\d_]+", text)
    useful = [
        word for word in words
        if len(word) >= 4 and word not in ADDRESS_STOP_WORDS
    ]

    number = str(int(numbers[0]))
    return {f"{number}|{word}" for word in useful[:6]}


def build_pool_index():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(DB_PATH)

    connection.execute("DROP TABLE IF EXISTS pool")
    connection.execute("""
        CREATE TABLE pool (
            entity_id TEXT PRIMARY KEY,
            country TEXT,
            name TEXT,
            address TEXT
        )
    """)

    for path in POOL_FILES:
        total = 0

        for chunk in pd.read_csv(
            path,
            sep="\t",
            usecols=[
    "entity_id",
    "country_norm",
    "business_name_norm",
    "business_address_norm",
],
            dtype=str,
            chunksize=50_000,
        ):
            rows = zip(
                chunk["entity_id"],
                chunk["country_norm"].fillna("").str.strip().str.casefold(),
                chunk["business_name_norm"].fillna("").str.strip().str.casefold(),
                chunk["business_address_norm"].fillna("").str.strip().str.casefold(),
            )

            connection.executemany(
                "INSERT INTO pool VALUES (?, ?, ?, ?)", rows
            )
            connection.commit()

            total += len(chunk)
            if total % 500_000 == 0:
                print(f"{path.name}: indexed {total:,} records")

        print(f"{path.name}: finished {total:,} records")

    connection.execute("CREATE INDEX pool_name_idx ON pool(country, name)")
    connection.commit()
    connection.close()
    print(f"Index ready: {DB_PATH}")


def preview_candidates():
    connection = sqlite3.connect(DB_PATH)

    validation_ids = set(
        Path("output/splits/val_source1_ids.txt").read_text().splitlines()
    )

    shown = 0
    for chunk in pd.read_csv(
        "cleaned_dataset/cleaned_source1.tsv",
        sep="\t",
        usecols=["entity_id", "country_norm", "business_name_norm"],
        dtype=str,
        chunksize=50_000,
    ):
        for row in chunk.itertuples(index=False):
            if row.entity_id not in validation_ids:
                continue

            country = (
                row.country_norm.strip().casefold()
                if pd.notna(row.country_norm) else ""
            )
            name = (
                row.business_name_norm.strip().casefold()
                if pd.notna(row.business_name_norm) else ""
            )

            matches = connection.execute(
                """
                SELECT entity_id
                FROM pool
                WHERE country = ? AND name = ?
                LIMIT 20
                """,
                (country, name),
            ).fetchall()

            print(row.entity_id, name, "→", [item[0] for item in matches])
            shown += 1
            if shown == 5:
                connection.close()
                return

    connection.close()


def measure_exact_name_recall(limit=20_000):
    truth = pd.read_csv(
        "output/splits/val_ground_truth.tsv",
        sep="\t",
        dtype=str,
    )
    truth_by_id = {
        row.source1_entity_id: set(
            row.matched_entity_ids.split(",")
            if pd.notna(row.matched_entity_ids) else []
        )
        for row in truth.itertuples(index=False)
    }

    connection = sqlite3.connect(DB_PATH)
    checked = true_matches = found_matches = total_candidates = max_candidates = 0
    

    for chunk in pd.read_csv(
        "cleaned_dataset/cleaned_source1.tsv",
        sep="\t",
        usecols=["entity_id", "country_norm", "business_name_norm", "business_address_norm"],
        dtype=str,
        chunksize=50_000,
    ):
        for row in chunk.itertuples(index=False):
            if row.entity_id not in truth_by_id:
                continue

            country = (
                row.country_norm.strip().casefold()
                if pd.notna(row.country_norm) else ""
            )
            name = (
                row.business_name_norm.strip().casefold()
                if pd.notna(row.business_name_norm) else ""
            )

            candidates = set()

            if name:
                candidates.update(
                    result[0]
                    for result in connection.execute(
                        """
                        SELECT entity_id FROM pool
                        WHERE country = ? AND name = ?
                        LIMIT 20
                        """,
                        (country, name),
                    )
                )

            address = (
                row.business_address_norm.strip().casefold()
                if pd.notna(row.business_address_norm) else ""
            )
            if address:
                candidates.update(
                    result[0]
                    for result in connection.execute(
                        """
                        SELECT entity_id FROM pool
                        WHERE country = ? AND address = ?
                        LIMIT 20
                        """,
                        (country, address),
                    )
                )

            if len(name) >= 5:
                prefix = name[:5]

                possible = connection.execute(
                    """
                    SELECT entity_id, name FROM pool
                    WHERE country = ?
                      AND name >= ?
                      AND name < ?
                    LIMIT 300
                    """,
                    (country, prefix, prefix + "\uffff"),
                ).fetchall()

                ranked = sorted(
                    possible,
                    key=lambda item: SequenceMatcher(
                        None, name, item[1]
                    ).ratio(),
                    reverse=True,
                )

                candidates.update(
                    entity_id for entity_id, _ in ranked[:20]
                )

            

            for key in address_keys(row.business_address_norm):
                rows = connection.execute(
                    """
                    SELECT entity_id FROM address_keys
                    WHERE country = ? AND key = ?
                    LIMIT 201
                    """,
                    (country, key),
                ).fetchall()

                # Match the pilot rule: ignore keys shared by over 200 records.
                if len(rows) <= 200:
                    candidates.update(item[0] for item in rows)

            total_candidates += len(candidates)
            max_candidates = max(max_candidates, len(candidates))

            actual = truth_by_id[row.entity_id]
            true_matches += len(actual)
            found_matches += len(actual & candidates)
            checked += 1

            if checked % 1_000 == 0:
                print(f"Checked {checked:,} S1 records", flush=True)

            if checked == limit:
                connection.close()
                recall = found_matches / true_matches if true_matches else 0
                print(f"S1 records checked: {checked:,}")
                print(f"True matching links: {true_matches:,}")
                print(f"Links found in candidates: {found_matches:,}")
                print(f"Combined blocking recall: {recall:.2%}")
                print(f"Average candidates per S1: {total_candidates / checked:.1f}")
                print(f"Maximum candidates for one S1: {max_candidates}")
                return
    connection.close()


if __name__ == "__main__":
    measure_exact_name_recall()