"""Fail-closed aggregate audit of Beauty sequential train/validation/test TFRecords.

The report contains aggregate counts and ranges only. It never emits user IDs,
raw sequences, text, or per-user rows.
"""

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from typing import Iterable

import numpy as np
from tfrecord.reader import tfrecord_loader

SPLITS = ("training", "evaluation", "testing")
VISIBLE_HISTORY_LIMIT = 19  # collator sequence_length=20; final event is target


def iter_records(data_dir: Path, split: str):
    paths = sorted((data_dir / split).glob("*.tfrecord.gz"))
    if not paths:
        raise FileNotFoundError(f"No gzipped TFRecords in {data_dir / split}")
    for path in paths:
        yield from tfrecord_loader(str(path), None, compression_type="gzip")


def _counter_dict(counter: Counter) -> dict[str, int]:
    return {str(key): int(value) for key, value in sorted(counter.items())}


def _summarize_split(rows: Iterable[dict], catalog_size: int) -> tuple[dict, dict[int, list[tuple[int, ...]]]]:
    by_user: dict[int, list[tuple[int, ...]]] = defaultdict(list)
    lengths, visible_lengths = Counter(), Counter()
    all_item_ids = []
    target_oob = 0
    short_sequences = 0
    for row in rows:
        if "user_id" not in row or "sequence_data" not in row:
            raise ValueError("TFRecord row lacks required user_id or sequence_data field")
        uid_values = np.asarray(row["user_id"]).reshape(-1)
        if uid_values.size != 1:
            raise ValueError("user_id must be a scalar field")
        uid = int(uid_values[0])
        sequence = tuple(int(v) for v in np.asarray(row["sequence_data"]).reshape(-1))
        by_user[uid].append(sequence)
        lengths[len(sequence)] += 1
        all_item_ids.extend(sequence)
        if len(sequence) < 2:
            short_sequences += 1
            visible_lengths[0] += 1
            continue
        visible_history = min(len(sequence) - 1, VISIBLE_HISTORY_LIMIT)
        visible_lengths[visible_history] += 1
        target_oob += int(sequence[-1] < 0 or sequence[-1] >= catalog_size)

    if not by_user:
        raise ValueError("Split contains no records")
    duplicate_records = sum(len(v) - 1 for v in by_user.values())
    user_ids = set(by_user)
    id_range = [min(user_ids), max(user_ids)]
    noncontiguous_users = user_ids != set(range(len(user_ids)))
    seq_sizes = [len(s) for sequences in by_user.values() for s in sequences]
    unique_sequences = {
        uid: seqs[0] for uid, seqs in by_user.items() if len(seqs) == 1
    }
    item_range = [min(all_item_ids), max(all_item_ids)] if all_item_ids else None
    item_oob = sum(item < 0 or item >= catalog_size for item in all_item_ids)
    report = {
        "records": sum(map(len, by_user.values())),
        "unique_users": len(user_ids),
        "duplicate_user_records": duplicate_records,
        "user_id_range": id_range,
        "user_ids_are_contiguous_from_zero": not noncontiguous_users,
        "sequence_length": {
            "minimum": min(seq_sizes),
            "maximum": max(seq_sizes),
            "mean": round(mean(seq_sizes), 4),
        },
        "visible_history_length_limit": VISIBLE_HISTORY_LIMIT,
        "visible_history_lengths": _counter_dict(visible_lengths),
        "users_truncated_to_limit": sum(
            count for raw_length, count in lengths.items()
            if max(0, raw_length - 1) > VISIBLE_HISTORY_LIMIT
        ),
        "raw_sequence_item_id_range": item_range,
        "out_of_catalog_range_event_ids": item_oob,
        "last_event_target_out_of_catalog_range": target_oob,
        "sequences_too_short_for_next_item_target": short_sequences,
    }
    return report, unique_sequences


def _compare_prefix(left: dict[int, tuple[int, ...]], right: dict[int, tuple[int, ...]]) -> dict:
    left_users, right_users = set(left), set(right)
    shared = left_users & right_users
    deltas = Counter()
    violations = 0
    strict_prefix_users = 0
    for uid in shared:
        a, b = left[uid], right[uid]
        delta = len(b) - len(a)
        deltas[delta] += 1
        if len(b) > len(a) and b[:len(a)] == a:
            strict_prefix_users += 1
        else:
            violations += 1
    return {
        "left_unique_users": len(left_users),
        "right_unique_users": len(right_users),
        "users_missing_from_right": len(left_users - right_users),
        "users_missing_from_left": len(right_users - left_users),
        "compared_users": len(shared),
        "strict_prefix_users": strict_prefix_users,
        "strict_prefix_violations": violations,
        "event_count_delta_right_minus_left": _counter_dict(deltas),
        "exactly_one_new_event_users": deltas[1],
    }


def _load_catalog(data_dir: Path) -> tuple[dict, int]:
    ids = []
    for row in iter_records(data_dir, "items"):
        values = np.asarray(row.get("id", [])).reshape(-1)
        if values.size != 1:
            raise ValueError("Catalog item id must be a scalar field")
        ids.append(int(values[0]))
    if not ids:
        raise ValueError("Item catalog is empty")
    count = len(ids)
    unique = set(ids)
    expected = set(range(1, count + 1))
    return {
        "records": count,
        "unique_ids": len(unique),
        "id_range": [min(ids), max(ids)],
        "duplicate_id_records": count - len(unique),
        "missing_ids_from_expected_1_to_catalog_size": len(expected - unique),
    }, count


def audit(data_dir: Path) -> dict:
    catalog, catalog_size = _load_catalog(data_dir)
    split_reports, sequences = {}, {}
    for split in SPLITS:
        split_reports[split], sequences[split] = _summarize_split(
            iter_records(data_dir, split), catalog_size
        )

    relationships = {
        "training_is_strict_prefix_of_evaluation": _compare_prefix(
            sequences["training"], sequences["evaluation"]
        ),
        "evaluation_is_strict_prefix_of_testing": _compare_prefix(
            sequences["evaluation"], sequences["testing"]
        ),
    }
    expected_users = set(sequences["training"])
    all_users_equal = all(set(sequences[s]) == expected_users for s in SPLITS)
    split_data_gates = all(
        report["duplicate_user_records"] == 0
        and report["user_ids_are_contiguous_from_zero"]
        and report["out_of_catalog_range_event_ids"] == 0
        and report["last_event_target_out_of_catalog_range"] == 0
        and report["sequences_too_short_for_next_item_target"] == 0
        for report in split_reports.values()
    )
    catalog_gate = catalog["duplicate_id_records"] == 0 and catalog["missing_ids_from_expected_1_to_catalog_size"] == 0
    prefix_gate = all(
        rel["users_missing_from_right"] == 0
        and rel["users_missing_from_left"] == 0
        and rel["strict_prefix_violations"] == 0
        and rel["exactly_one_new_event_users"] == rel["compared_users"]
        for rel in relationships.values()
    )
    # The source collator labels sequence[-1] and uses sequence[-20:-1] as history.
    report = {
        "source": "Beauty raw gzipped TFRecords; aggregate-only audit; no model scores",
        "target_definition": {
            "field": "sequence_data",
            "target_event": "last event in each split record",
            "visible_history": "sequence_data[-20:-1]",
            "visible_history_limit": VISIBLE_HISTORY_LIMIT,
            "collator_source": "src/experimental/data/loading/components/collate_functions.py:process_text_sequence",
            "dataloader_sequence_length": 20,
            "metadata_item_id_range": catalog["id_range"],
            "sequence_event_id_offset_from_metadata_id": -1,
        },
        "catalog": catalog,
        "splits": split_reports,
        "user_sets_identical_across_splits": all_users_equal,
        "sequence_relationships": relationships,
        "gates": {
            "catalog_ids_are_unique_and_cover_1_to_catalog_size": catalog_gate,
            "split_user_and_item_id_checks_pass": split_data_gates,
            "all_user_sets_identical": all_users_equal,
            "both_prefix_and_one_event_extension_checks_pass": prefix_gate,
        },
        "passed": catalog_gate and split_data_gates and all_users_equal and prefix_gate,
    }
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True,
                        help="Beauty raw data directory with training/, evaluation/, testing/, items/")
    parser.add_argument("--output", type=Path,
                        help="Optional aggregate JSON output path")
    args = parser.parse_args()
    report = audit(args.data_dir)
    serialized = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(serialized, encoding="utf-8")
    print(serialized, end="")
    return 0 if report["passed"] else 2


if __name__ == "__main__":
    sys.exit(main())
