import sqlite3

from src.blocking import DB_PATH, address_keys


def main():
    connection = sqlite3.connect(DB_PATH)

    connection.execute("DROP TABLE IF EXISTS address_keys")
    connection.execute("""
        CREATE TABLE address_keys (
            country TEXT NOT NULL,
            key TEXT NOT NULL,
            entity_id TEXT NOT NULL
        )
    """)
    connection.commit()

    reader = connection.execute(
        "SELECT entity_id, country, address FROM pool"
    )
    writer = connection.cursor()
    scanned = 0

    while batch := reader.fetchmany(50_000):
        rows = [
            (country, key, entity_id)
            for entity_id, country, address in batch
            for key in address_keys(address)
        ]

        writer.executemany(
            "INSERT INTO address_keys VALUES (?, ?, ?)", rows
        )
        connection.commit()

        scanned += len(batch)
        if scanned % 1_000_000 < 50_000:
            print(f"Processed {scanned:,} pool records", flush=True)

    print("Building lookup index...", flush=True)
    connection.execute(
        "CREATE INDEX address_key_lookup "
        "ON address_keys(country, key)"
    )
    connection.commit()
    connection.close()
    print("Address-key index ready")


if __name__ == "__main__":
    main()