cat > src/train_model.py <<'PY'
import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier


FEATURES = [
    "name_similarity",
    "name_token_similarity",
    "address_similarity",
    "address_token_similarity",
    "country_match",
]
THRESHOLDS = np.arange(0.10, 1.00, 0.05)


def load_truth(ids_path, truth_path):
    val_ids = [line.strip() for line in ids_path.read_text().splitlines()
               if line.strip()]
    if len(val_ids) != len(set(val_ids)):
        raise ValueError("Duplicate S1 IDs in validation split")

    truth = pd.read_csv(truth_path, sep="\t", dtype=str).fillna("")
    counts = {}
    for row in truth.itertuples(index=False):
        matches = {x.strip() for x in row.matched_entity_ids.split(",")
                   if x.strip()}
        counts[row.source1_entity_id] = len(matches)

    if set(counts) != set(val_ids):
        raise ValueError(
            "Validation ID list and validation ground truth do not match"
        )

    return val_ids, np.array(
        [counts[entity_id] for entity_id in val_ids], dtype=np.int32
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--val", type=Path, required=True)
    parser.add_argument("--val-ids", type=Path, required=True)
    parser.add_argument("--truth", type=Path, required=True)
    parser.add_argument("--model-out", type=Path, required=True)
    parser.add_argument("--train-rows", type=int, default=1_000_000)
    args = parser.parse_args()

    train = pd.read_csv(
        args.train, sep="\t", usecols=FEATURES + ["label"],
        nrows=args.train_rows,
    )
    positives = int(train["label"].sum())
    print(f"Training rows: {len(train):,}; positives: {positives:,}")
    if train["label"].nunique() != 2:
        raise ValueError("Training sample needs both label 0 and label 1")

    model = LGBMClassifier(
        n_estimators=300,
        learning_rate=0.05,
        num_leaves=31,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(train[FEATURES].fillna(0), train["label"])
    del train

    val_ids, true_count = load_truth(args.val_ids, args.truth)
    position = {entity_id: i for i, entity_id in enumerate(val_ids)}
    tp = np.zeros((len(val_ids), len(THRESHOLDS)), dtype=np.int32)
    fp = np.zeros_like(tp)
    pairs_seen = 0

    for chunk in pd.read_csv(
        args.val,
        sep="\t",
        usecols=["source1_entity_id", "label"] + FEATURES,
        chunksize=50_000,
    ):
        mapped = chunk["source1_entity_id"].map(position)
        if mapped.isna().any():
            raise ValueError("Validation feature file contains a non-validation S1")

        indices = mapped.to_numpy(dtype=np.int32)
        labels = chunk["label"].to_numpy(dtype=np.int8)
        probability = model.predict_proba(
            chunk[FEATURES].fillna(0)
        )[:, 1]

        for j, threshold in enumerate(THRESHOLDS):
            selected = probability >= threshold
            tp[:, j] += np.bincount(
                indices[selected & (labels == 1)],
                minlength=len(val_ids),
            )
            fp[:, j] += np.bincount(
                indices[selected & (labels == 0)],
                minlength=len(val_ids),
            )

        pairs_seen += len(chunk)
        if pairs_seen % 1_000_000 < len(chunk):
            print(f"Validation pairs scored: {pairs_seen:,}", flush=True)

    if (tp > true_count[:, None]).any():
        raise ValueError("Duplicate positive pairs or inconsistent ground truth")

    fn = true_count[:, None] - tp
    denominator = 1.25 * tp + fp + 0.25 * fn
    per_s1 = np.divide(
        1.25 * tp,
        denominator,
        out=np.ones_like(denominator, dtype=float),
        where=denominator > 0,
    )
    scores = per_s1.mean(axis=0)
    best = int(scores.argmax())

    for threshold, score in zip(THRESHOLDS, scores):
        print(f"Threshold {threshold:.2f}: macro F0.5 = {score:.5f}")

    args.model_out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {"model": model, "features": FEATURES,
         "threshold": float(THRESHOLDS[best])},
        args.model_out,
    )
    print(f"Best: {THRESHOLDS[best]:.2f} → {scores[best]:.5f}")
    print(f"Saved: {args.model_out}")


if __name__ == "__main__":
    main()
