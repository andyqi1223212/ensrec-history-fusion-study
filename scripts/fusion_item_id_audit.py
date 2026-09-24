"""Audit Beauty's raw item-ID to sequence-ID relation without the training stack."""

import argparse
import json
from collections import Counter
from pathlib import Path

from tfrecord.reader import tfrecord_loader


def records(data_dir, split):
    paths = sorted((data_dir / split).glob("*.tfrecord.gz"))
    if not paths:
        raise FileNotFoundError(f"No TFRecord files in {data_dir / split}")
    for path in paths:
        yield from tfrecord_loader(str(path), None, compression_type="gzip")


def summarize_mapping(item_rows, split_rows):
    """Summarize raw item IDs and per-event first-item text matches by offset."""
    import numpy as np

    item_ids = [int(row["id"][0]) for row in item_rows]
    item_text = {int(row["id"][0]): row["text"] for row in item_rows}
    counts = Counter(item_ids)
    if not item_ids:
        raise ValueError("Item catalog is empty")
    expected_item_ids = set(range(1, len(item_ids) + 1))
    if len(counts) != len(item_ids) or set(counts) != expected_item_ids:
        raise ValueError("Beauty item IDs must uniquely cover 1..item_count")

    split_reports = {}
    all_sequence_ids = set()
    for split, rows in split_rows.items():
        sequence_ids = set()
        first_item_ids = set()
        matches = {0: 0, 1: 0}
        missing = {0: 0, 1: 0}
        events = 0
        for row in rows:
            sequence = [int(value) for value in row["sequence_data"]]
            if not sequence:
                raise ValueError(f"{split}: empty sequence_data")
            first_id = sequence[0]
            first_item_ids.add(first_id)
            sequence_ids.update(sequence)
            target_text = row["text"]
            for offset in (0, 1):
                candidate_id = first_id + offset
                if candidate_id not in item_text:
                    missing[offset] += 1
                elif item_text[candidate_id] == target_text:
                    matches[offset] += 1
            events += 1
        if events == 0:
            raise ValueError(f"{split}: no sequence events found")
        all_sequence_ids.update(sequence_ids)
        split_reports[split] = {
            "events": events,
            "sequence_id_range": [min(sequence_ids), max(sequence_ids)],
            "sequence_id_unique_count": len(sequence_ids),
            "first_item_sequence_id_unique_count": len(first_item_ids),
            "first_item_text_matches_by_metadata_id_offset": {
                str(offset): matches[offset] for offset in (0, 1)
            },
            "missing_candidate_metadata_rows_by_offset": {
                str(offset): missing[offset] for offset in (0, 1)
            },
        }

    min_item, max_item = min(item_ids), max(item_ids)
    if not all_sequence_ids:
        raise ValueError("No sequence IDs found across splits")
    min_sequence, max_sequence = min(all_sequence_ids), max(all_sequence_ids)
    expected_sequence_ids = set(range(len(item_ids)))
    if all_sequence_ids != expected_sequence_ids:
        raise ValueError("Beauty sequence IDs must cover 0..item_count-1")
    for split, split_report in split_reports.items():
        matched = split_report["first_item_text_matches_by_metadata_id_offset"]["1"]
        events = split_report["events"]
        if events == 0 or matched != events:
            raise ValueError(
                f"{split}: first-item texts do not all map through metadata_id=sequence_id+1 "
                f"({matched}/{events})"
            )
    notebook_text_rows = [np.asarray([row["text"]]).astype(str) for row in item_rows]
    try:
        notebook_text_rows[0].encode("utf-8")
        first_notebook_error = None
    except AttributeError as error:
        first_notebook_error = f"{type(error).__name__}: {error}"
    repaired_item_to_text = {
        item_ids[index]: notebook_text_rows[index][0].encode("utf-8")
        for index in range(len(item_ids))
    }
    try:
        [repaired_item_to_text[index] for index in range(len(repaired_item_to_text))]
        second_notebook_error = None
    except KeyError as error:
        second_notebook_error = f"{type(error).__name__}: missing key {error.args[0]}"
    return {
        "source": "Beauty raw TFRecords; no model scores or embedding files",
        "item_catalog": {
            "rows": len(item_ids),
            "unique_ids": len(counts),
            "duplicate_rows": len(item_ids) - len(counts),
            "id_range": [min_item, max_item],
            "missing_ids_within_range": sorted(set(range(min_item, max_item + 1)) - set(counts)),
        },
        "sequence_ids_across_splits": {
            "range": [min_sequence, max_sequence],
            "unique_count": len(all_sequence_ids),
            "missing_ids_within_zero_based_range": sorted(
                set(range(0, max_sequence + 1)) - all_sequence_ids
            ),
        },
        "splits": split_reports,
        "mapping_gate": {
            "passed": True,
            "item_ids_cover_1_to_item_count_once": True,
            "sequence_ids_cover_0_to_item_count_minus_1": True,
            "all_first_item_texts_match_metadata_id_plus_1": True,
        },
        "notebook_failure_reproduction": {
            "cell_7_text_element_type": type(notebook_text_rows[0]).__name__,
            "cell_8_current_first_error": first_notebook_error,
            "after_extracting_scalar_text_cell_13_error": second_notebook_error,
            "dictionary_id_range": [min(item_ids), max(item_ids)],
        },
        "interpretation": (
            "Each split's first-item text matched metadata ID + 1 for all events, covering "
            "only the split's reported unique first-item sequence IDs (7,563 on the checked "
            "files), not every catalog item. Remaining items require a generation manifest "
            "and source-order contract. Event-level matches do not prove that a precomputed "
            "embedding file uses that row order."
        ),
    }


def audit(data_dir):
    item_rows = list(records(data_dir, "items"))
    splits = {
        split: records(data_dir, split)
        for split in ("training", "evaluation", "testing")
    }
    # Materialize each split independently while item rows remain small (Beauty has 12,101).
    split_rows = {split: list(rows) for split, rows in splits.items()}
    return summarize_mapping(item_rows, split_rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = audit(args.data_dir)
    serialized = json.dumps(report, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(serialized, encoding="utf-8")
    print(serialized, end="")


if __name__ == "__main__":
    main()
