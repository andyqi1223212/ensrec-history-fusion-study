"""Aggregate-only post hoc diagnostic for seen items in Beauty Text rankings."""

import argparse
import json
from pathlib import Path

import numpy as np

from beauty_cpu_proxy import (
    align_catalog,
    build_tfidf,
    choose_validation_cutoff,
    examples_for_split,
    history_profiles,
    load_tfrecords,
    topk_indices,
    validate_splits,
)


def _cohort_stats(examples, full_histories, text_matrix, item_ids, mask, k=10, batch_size=128):
    selected = [examples[i] for i in np.flatnonzero(mask)]
    visible_seen_targets = 0
    full_seen_targets = 0
    text_hits = 0
    top1_seen = 0
    seen_slots = 0

    for start in range(0, len(selected), batch_size):
        batch = selected[start:start + batch_size]
        histories = [history for _, history, _ in batch]
        profiles, _ = history_profiles(histories, text_matrix)
        similarities = (profiles @ text_matrix.T).tocsr()
        for row_index, (_, visible_history, target) in enumerate(batch):
            user_id = selected[start + row_index][0]
            full_history = full_histories[user_id]
            visible_set = set(visible_history)
            full_set = set(full_history)
            visible_seen_targets += target in visible_set
            full_seen_targets += target in full_set

            scores = similarities.getrow(row_index).toarray().reshape(-1)
            top = topk_indices(scores, item_ids, k)
            top_set = set(int(x) for x in top)
            text_hits += target in top_set
            top1_seen += int(top[0]) in visible_set
            seen_slots += sum(int(candidate) in visible_set for candidate in top)

    n = len(selected)
    if not n:
        raise ValueError("Empty history cohort")
    mean_seen_slots = seen_slots / n
    return {
        "users": n,
        "target_in_visible_history": {"count": int(visible_seen_targets), "rate": visible_seen_targets / n},
        "target_in_full_pre_target_history": {"count": int(full_seen_targets), "rate": full_seen_targets / n},
        "text_top1_is_visible_seen_item": {"count": int(top1_seen), "rate": top1_seen / n},
        "text_top10_visible_seen_slots": {
            "mean_count": mean_seen_slots,
            "mean_fraction": mean_seen_slots / k,
        },
        "text_top10_remaining_unseen_slots": {"mean_count": k - mean_seen_slots},
        "text_target_hits": {"count": int(text_hits), "recall_at_10": text_hits / n},
    }


def analyze(data):
    n_items = len(data["catalog"])
    audit = validate_splits({s: data[s] for s in ("training", "evaluation", "testing")}, n_items)
    catalog_texts = align_catalog(data["catalog"], n_items)
    _, text_matrix = build_tfidf(catalog_texts)
    item_ids = np.arange(n_items, dtype=np.int32)

    examples = {
        "validation": examples_for_split(data["evaluation"]),
        "test": examples_for_split(data["testing"]),
    }
    validation_lengths = np.asarray([len(history) for _, history, _ in examples["validation"]], dtype=np.int32)
    cutoff, _ = choose_validation_cutoff(validation_lengths)
    output = {
        "scope": "Post hoc potential mechanism diagnostic; aggregate metrics only",
        "protocol": {
            "catalog_items": n_items,
            "visible_history": "visible_history_target(sequence): latest at most 19 events before target",
            "full_pre_target_history": "all events before target in the record",
            "text_profile": "normalized sum of visible-history item TF-IDF vectors; all catalog items remain candidates",
            "text_ranking": "cosine similarity descending; item ID ascending for ties; top 10",
            "cohort_cutoff_source": "validation visible-history length median only",
            "validation_cutoff": cutoff,
            "short_rule": "visible history length <= validation median cutoff",
            "causal_claim": False,
        },
        "validation": {},
        "test": {},
    }
    for name, split_examples in examples.items():
        lengths = np.asarray([len(history) for _, history, _ in split_examples], dtype=np.int32)
        masks = {
            "all": np.ones(len(split_examples), dtype=bool),
            "short": lengths <= cutoff,
            "long": lengths > cutoff,
        }
        full_histories = {user_id: tuple(data["evaluation" if name == "validation" else "testing"][user_id][:-1])
                          for user_id, _, _ in split_examples}
        output[name] = {
            cohort: _cohort_stats(split_examples, full_histories, text_matrix, item_ids, mask)
            for cohort, mask in masks.items()
        }

    test_hits = output["test"]["all"]["text_target_hits"]["count"]
    short_hits = output["test"]["short"]["text_target_hits"]["count"]
    long_hits = output["test"]["long"]["text_target_hits"]["count"]
    if (test_hits, short_hits, long_hits) != (924, 404, 520):
        raise RuntimeError(
            "Text hit parity check failed: expected test all/short/long 924/404/520, "
            f"got {test_hits}/{short_hits}/{long_hits}; output not written"
        )
    output["parity_check"] = {"expected_test_text_hits": {"all": 924, "short": 404, "long": 520},
                              "passed": True}
    output["data_audit"] = audit
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("../data/beauty"))
    parser.add_argument("--output", type=Path, default=Path("outputs/beauty-seen-item-diagnostic.json"))
    args = parser.parse_args()
    result = analyze(load_tfrecords(args.data_dir))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "parity_check": result["parity_check"]}))


if __name__ == "__main__":
    main()
