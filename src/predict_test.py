import argparse
import csv
import os
from pathlib import Path

import joblib
import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--features-dir", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    saved = joblib.load(args.model)
    model = saved["model"]
    features = saved["features"]
    threshold = saved["threshold"]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(".tsv.tmp")
    total_s1 = total_matches = 0

    with temporary.open("w", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerow(["source1_entity_id", "matched_entity_ids"])

        for part_number, chunk in enumerate(pd.read_csv(
            args.candidates,
            sep="\t",
            dtype=str,
            keep_default_na=False,
            chunksize=500,
        )):
            path = args.features_dir / f"part_{part_number:05d}.tsv"
            if not path.exists():
                raise FileNotFoundError(f"Test features not finished: {path}")

            pairs = pd.read_csv(path, sep="\t", dtype={
                "source1_entity_id": str,
                "candidate_entity_id": str,
            })
            expected = sum(
                len(ids.split(",")) if ids else 0
                for ids in chunk["candidate_entity_ids"]
            )
            if len(pairs) != expected:
                raise ValueError(
                    f"Part {part_number}: expected {expected} pairs, "
                    f"found {len(pairs)}"
                )

            selected_by_s1 = {}
            if not pairs.empty:
                probability = model.predict_proba(
                    pairs[features].fillna(0)
                )[:, 1]
                selected = pairs.loc[
                    probability >= threshold,
                    ["source1_entity_id", "candidate_entity_id"],
                ]
                selected_by_s1 = selected.groupby(
                    "source1_entity_id", sort=False
                )["candidate_entity_id"].agg(list).to_dict()

            for row in chunk.itertuples(index=False):
                matches = selected_by_s1.get(row.source1_entity_id, [])
                candidates = (
                    set(row.candidate_entity_ids.split(","))
                    if row.candidate_entity_ids else set()
                )
                if not set(matches).issubset(candidates):
                    raise ValueError(
                        f"Predicted match outside candidates: "
                        f"{row.source1_entity_id}"
                    )
                writer.writerow([row.source1_entity_id, ",".join(matches)])
                total_s1 += 1
                total_matches += len(matches)

            if (part_number + 1) % 100 == 0:
                print(f"Predicted {total_s1:,} S1 records", flush=True)

    os.replace(temporary, args.output)
    print(f"Done: {total_s1:,} S1; {total_matches:,} predicted links")
    print(f"Written: {args.output}")


if __name__ == "__main__":
    main()
