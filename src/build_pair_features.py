import argparse
import sqlite3
from pathlib import Path

import pandas as pd
from rapidfuzz import fuzz


def similarity(left, right, scorer):
    if not left or not right:
        return 0.0
    return scorer(left, right) / 100.0


def load_s1(path):
    records = {}
    columns = [
        "entity_id", "business_name_norm",
        "business_address_norm", "country_norm",
    ]
    for chunk in pd.read_csv(
        path, sep="\t", usecols=columns, dtype=str, chunksize=50_000
    ):
        for entity_id, name, address, country in chunk[columns].fillna("").itertuples(
            index=False, name=None
        ):
            records[entity_id] = (name, address, country.casefold())
    return records


def fetch_candidates(connection, ids):
    found = {}
    unique_ids = list(dict.fromkeys(ids))

    for start in range(0, len(unique_ids), 500):
        batch = unique_ids[start:start + 500]
        marks = ",".join("?" for _ in batch)
        query = (
            f"SELECT entity_id, name, address, country "
            f"FROM pool WHERE entity_id IN ({marks})"
        )
        for entity_id, name, address, country in connection.execute(query, batch):
            found[entity_id] = (name, address, country.casefold())

    return found


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pairs", type=Path, required=True)
    parser.add_argument("--source1", type=Path, required=True)
    parser.add_argument("--pool", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    s1 = load_s1(args.source1)
    connection = sqlite3.connect(
        args.pool.resolve().as_uri() + "?mode=ro", uri=True
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)

    columns = [
        "source1_entity_id", "candidate_entity_id", "source",
        "name_similarity", "name_token_similarity",
        "address_similarity", "address_token_similarity",
        "country_match", "label",
    ]
    pd.DataFrame(columns=columns).to_csv(args.output, sep="\t", index=False)

    processed = 0
    for chunk in pd.read_csv(args.pairs, sep="\t", chunksize=10_000):
        if args.limit is not None:
            chunk = chunk.iloc[:max(0, args.limit - processed)]
        if chunk.empty:
            break

        candidates = fetch_candidates(
            connection, chunk["candidate_entity_id"].tolist()
        )
        rows = []

        for row in chunk.itertuples(index=False):
            left = s1.get(row.source1_entity_id)
            right = candidates.get(row.candidate_entity_id)
            if left is None or right is None:
                raise ValueError(
                    f"Missing record: {row.source1_entity_id}, "
                    f"{row.candidate_entity_id}"
                )

            name1, address1, country1 = left
            name2, address2, country2 = right
            rows.append([
                row.source1_entity_id,
                row.candidate_entity_id,
                row.source,
                similarity(name1, name2, fuzz.ratio),
                similarity(name1, name2, fuzz.token_set_ratio),
                similarity(address1, address2, fuzz.ratio),
                similarity(address1, address2, fuzz.token_set_ratio),
                int(country1 == country2),
                row.label,
            ])

        pd.DataFrame(rows, columns=columns).to_csv(
            args.output, sep="\t", index=False, header=False, mode="a"
        )
        processed += len(chunk)
        print(f"Feature rows written: {processed:,}", flush=True)

        if args.limit is not None and processed >= args.limit:
            break

    connection.close()


if __name__ == "__main__":
    main()
