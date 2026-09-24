"""Audit the Beauty validation/test history lengths seen by the configured collator."""

import json
import argparse
from collections import Counter
from pathlib import Path

from tfrecord.reader import tfrecord_loader


def records(data_dir, split):
    paths = sorted((data_dir / split).glob("*.tfrecord.gz"))
    if not paths:
        raise FileNotFoundError(f"No TFRecord files in {data_dir / split}")
    for path in paths:
        yield from tfrecord_loader(str(path), None, compression_type="gzip")


def summarize(data_dir, split):
    lengths = {}
    for row in records(data_dir, split):
        user_id = int(row["user_id"][0])
        if user_id in lengths:
            raise ValueError(f"Duplicate {split} user ID {user_id}")
        # process_text_sequence uses seq[-sequence_length:-1]; Beauty sets sequence_length=20.
        lengths[user_id] = min(len(row["sequence_data"]) - 1, 19)
    if set(lengths) != set(range(len(lengths))):
        raise ValueError(f"{split} user IDs are not contiguous from zero")
    counts = Counter(lengths.values())
    return {"events": len(lengths), "length_counts": dict(sorted(counts.items())),
            "at_max_length_19": counts[19], "minimum": min(lengths.values()),
            "maximum": max(lengths.values())}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True,
                        help="Upstream Beauty data directory containing evaluation/ and testing/")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = {"source": "Beauty raw TFRecords, sequence_data; no model scores",
              "history_rule": "len(sequence_data[-20:-1])",
              "validation": summarize(args.data_dir, "evaluation"),
              "test": summarize(args.data_dir, "testing")}
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
