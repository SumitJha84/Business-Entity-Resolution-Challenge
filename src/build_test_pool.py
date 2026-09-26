import sqlite3
from pathlib import Path

import pandas as pd

from src.blocking import address_keys

DB_PATH = Path("output/blocking_test_pool.sqlite")
TEST_FILES = [
    Path("cleaned_dataset/cleaned_test_source2.tsv"),
    Path("cleaned_dataset/cleaned_test_source3.tsv"),
]


def main():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(DB_PATH)

    connection.executescript("""
        CREATE TABLE pool (
            entity_id TEXT PRIMARY KEY,
            country TEXT,
            name TEXT,
            address TEXT
        );
        CREATE TABLE address_keys (
            country TEXT NOT NULL,
            key TEXT NOT NULL,
            entity_id TEXT NOT NULL
        );
    """)

    for path in TEST_FILES:
        processed = 0

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
            chunk = chunk.fillna("")
            records = list(zip(
                chunk["entity_id"],
                chunk["country_norm"].str.strip().str.casefold(),
                chunk["business_name_norm"].str.strip().str.casefold(),
                chunk["business_address_norm"].str.strip().str.casefold(),
            ))

            connection.executemany(
                "INSERT INTO pool VALUES (?, ?, ?, ?)", records
            )
            connection.executemany(
                "INSERT INTO address_keys VALUES (?, ?, ?)",
                (
                    (country, key, entity_id)
                    for entity_id, country, _, address in records
                    for key in address_keys(address)
                ),
            )
            connection.commit()

            processed += len(records)
            if processed % 500_000 == 0:
                print(f"{path.name}: {processed:,} records", flush=True)

        print(f"{path.name}: finished {processed:,}", flush=True)

    print("Building test lookup indexes...", flush=True)
    connection.execute("CREATE INDEX pool_name_idx ON pool(country, name)")
    connection.execute("CREATE INDEX pool_address_idx ON pool(country, address)")
    connection.execute(
        "CREATE INDEX address_key_lookup ON address_keys(country, key)"
    )
    connection.execute(
    "CREATE INDEX address_key_cover_idx ON address_keys(country, key, entity_id)"
)
    connection.execute(
        "CREATE INDEX address_key_cover_idx ON address_keys(country, key, entity_id)"
    )
    connection.commit()
    connection.close()
    print("Test pool ready")


if __name__ == "__main__":
    main()