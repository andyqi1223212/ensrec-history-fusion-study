"""Evaluate ID/Text complementarity and validation-selected history-aware fusion.

The input is the pair of independent, best-checkpoint exports from IdTextModule.
No test label is used to choose a weight or a history threshold.
"""

import argparse
import csv
import json
from pathlib import Path

import numpy as np


WEIGHTS = tuple(round(x / 10, 1) for x in range(11))
K = 10


def load_split(id_dir: Path, text_dir: Path, split: str):
    import torch

    def read(directory, name):
        tensor = torch.load(directory / name, map_location="cpu", weights_only=True)
        return (tensor.float() if tensor.is_floating_point() else tensor).numpy()

    result = {
        "id_users": read(id_dir, f"{split}_user_embeddings.pt"),
        "text_users": read(text_dir, f"{split}_user_embeddings.pt"),
        "id_items": read(id_dir, f"{split}_item_embeddings.pt"),
        "text_items": read(text_dir, f"{split}_item_embeddings.pt"),
        "labels": read(id_dir, f"{split}_user_labels.pt").astype(np.int64),
        "lengths": read(id_dir, f"{split}_history_lengths.pt").astype(np.int64),
        "ids": read(id_dir, f"{split}_user_ids.pt").astype(np.int64),
    }
    for name, file in (("labels", "user_labels"), ("lengths", "history_lengths"),
                       ("ids", "user_ids")):
        other = read(text_dir, f"{split}_{file}.pt").astype(np.int64)
        if not np.array_equal(result[name], other):
            raise ValueError(f"{split}: ID/Text {name} differ; exports are not aligned")
    check_split(result, split)
    return result


def check_split(data, split):
    n = len(data["labels"])
    if n == 0 or not np.array_equal(data["ids"], np.arange(n)):
        raise ValueError(f"{split}: user IDs must be contiguous from zero")
    if any(len(data[name]) != n for name in ("id_users", "text_users", "lengths")):
        raise ValueError(f"{split}: unequal event counts")
    m = len(data["id_items"])
    if m < K or len(data["text_items"]) != m:
        raise ValueError(f"{split}: candidate matrices must share at least {K} item IDs")
    if np.any(data["labels"] < 0) or np.any(data["labels"] >= m):
        raise ValueError(f"{split}: label outside candidate table")
    if np.any(data["lengths"] < 1):
        raise ValueError(f"{split}: history length must be positive")
    for name in ("id_users", "text_users", "id_items", "text_items"):
        if data[name].ndim != 2 or not np.isfinite(data[name]).all():
            raise ValueError(f"{split}: {name} must be a finite matrix")
    if data["id_users"].shape[1] != data["id_items"].shape[1]:
        raise ValueError(f"{split}: ID embedding dimensions differ")
    if data["text_users"].shape[1] != data["text_items"].shape[1]:
        raise ValueError(f"{split}: Text embedding dimensions differ")


def normalized(matrix):
    matrix = np.asarray(matrix, dtype=np.float32)
    return matrix / np.maximum(np.linalg.norm(matrix, axis=1, keepdims=True), 1e-12)


def topk(scores, k=K):
    # The tiny item-ID term breaks exact float32 ties before partial sorting.
    adjusted = scores.astype(np.float64) - np.arange(scores.shape[1]) * 1e-15
    selected = np.argpartition(-adjusted, kth=k - 1, axis=1)[:, :k]
    selected_scores = np.take_along_axis(adjusted, selected, axis=1)
    order = np.argsort(-selected_scores, axis=1)
    return np.take_along_axis(selected, order, axis=1)


def predictions(data, weights=WEIGHTS, batch_size=128):
    id_users = normalized(data["id_users"])
    text_users = normalized(data["text_users"])
    id_items = normalized(data["id_items"])
    text_items = normalized(data["text_items"])
    result = {weight: np.empty((len(id_users), K), dtype=np.int64) for weight in weights}
    for start in range(0, len(id_users), batch_size):
        end = min(start + batch_size, len(id_users))
        id_scores = id_users[start:end] @ id_items.T
        text_scores = text_users[start:end] @ text_items.T
        for weight in weights:
            result[weight][start:end] = topk(weight * id_scores + (1 - weight) * text_scores)
    return result


def ranks(prediction, labels):
    matches = prediction == labels[:, None]
    return np.where(matches.any(axis=1), matches.argmax(axis=1) + 1, 0)


def metrics(rank, mask):
    selected = rank[mask]
    if len(selected) == 0:
        raise ValueError("Empty cohort")
    discounts = np.zeros(len(selected), dtype=np.float64)
    hit = selected > 0
    discounts[hit] = 1 / np.log2(selected[hit] + 1)
    return {"n": int(len(selected)), "recall@10": float(hit.mean()),
            "ndcg@10": float(discounts.mean()), "hit@1": float((selected == 1).mean())}


def comparison(id_rank, text_rank, fused_rank, mask):
    id_hit, text_hit, fused_hit = id_rank > 0, text_rank > 0, fused_rank > 0
    added = int((~id_hit & fused_hit & mask).sum())
    lost = int((id_hit & ~fused_hit & mask).sum())
    n = int(mask.sum())
    text_only_hits = int((~id_hit & text_hit & mask).sum())
    text_only_recovered = int((~id_hit & text_hit & fused_hit & mask).sum())
    return {
        "id": metrics(id_rank, mask), "text": metrics(text_rank, mask),
        "fusion": metrics(fused_rank, mask),
        "text_only_hits": text_only_hits,
        "text_only_recovered": text_only_recovered,
        "text_only_recovery_rate": (text_only_recovered / text_only_hits
                                    if text_only_hits else None),
        "added_hits": added, "lost_hits": lost, "net_hits": added - lost,
        "recall_delta_vs_id": (added - lost) / n,
    }


def choose_weight(ranks_by_weight, mask):
    # Stable tie break: favor the simple equal-weight baseline.
    return max(WEIGHTS, key=lambda weight: (
        metrics(ranks_by_weight[weight], mask)["recall@10"],
        -abs(weight - 0.5), -weight,
    ))


def analyze(validation, test):
    for route in ("id_items", "text_items"):
        if not np.array_equal(validation[route], test[route]):
            raise ValueError(f"Validation/test {route} differ; candidate table changed")
    val_pred = predictions(validation)
    val_rank = {w: ranks(p, validation["labels"]) for w, p in val_pred.items()}
    val_all = np.ones(len(validation["labels"]), dtype=bool)
    test_all = np.ones(len(test["labels"]), dtype=bool)
    unique_lengths = np.unique(validation["lengths"])
    if len(unique_lengths) < 2:
        raise ValueError("Validation needs at least two distinct history lengths")
    counts = np.searchsorted(np.sort(validation["lengths"]), unique_lengths[:-1], side="right")
    threshold = int(unique_lengths[np.argmin(np.abs(counts - len(val_all) / 2))])
    val_short = validation["lengths"] <= threshold
    test_short = test["lengths"] <= threshold
    if val_short.all() or not val_short.any():
        raise ValueError("Validation lengths cannot form two nonempty cohorts")
    global_weight = choose_weight(val_rank, val_all)
    short_weight = choose_weight(val_rank, val_short)
    long_weight = choose_weight(val_rank, ~val_short)
    test_pred = predictions(test, weights=sorted({0.0, 0.5, 1.0, global_weight,
                                                  short_weight, long_weight}))
    test_rank = {w: ranks(p, test["labels"]) for w, p in test_pred.items()}
    conditional_rank = np.where(test_short, test_rank[short_weight], test_rank[long_weight])
    methods = {"id": test_rank[1.0], "text": test_rank[0.0],
               "equal": test_rank[0.5], "global": test_rank[global_weight],
               "history": conditional_rank}
    report = {
        "protocol": {"k": K, "weight_definition": "alpha * ID cosine + (1-alpha) * Text cosine",
                     "candidate_count": len(test["id_items"]), "validation_balanced_threshold": threshold,
                     "selected_on_validation": {"global": global_weight, "short": short_weight,
                                                "long": long_weight},
                     "short_rule": "history_length <= threshold",
                     "test_labels_used_for_selection": False},
        "validation": {
            "equal": comparison(val_rank[1.0], val_rank[0.0], val_rank[0.5], val_all),
            "short_equal": comparison(val_rank[1.0], val_rank[0.0], val_rank[0.5], val_short),
            "long_equal": comparison(val_rank[1.0], val_rank[0.0], val_rank[0.5], ~val_short),
            "weight_grid": {str(w): metrics(val_rank[w], val_all) for w in WEIGHTS},
        },
        "test": {
            "equal": comparison(methods["id"], methods["text"], methods["equal"], test_all),
            "short_equal": comparison(methods["id"], methods["text"], methods["equal"], test_short),
            "long_equal": comparison(methods["id"], methods["text"], methods["equal"], ~test_short),
            "methods": {name: metrics(rank, test_all) for name, rank in methods.items()},
        },
    }
    rows = []
    for i, label in enumerate(test["labels"]):
        row = {"event_id": int(test["ids"][i]), "label": int(label),
               "history_length": int(test["lengths"][i]),
               "cohort": "short" if test_short[i] else "long"}
        for name, rank in methods.items():
            row[f"{name}_rank_at_10"] = int(rank[i])
        row["equal_added_vs_id"] = int(methods["id"][i] == 0 and methods["equal"][i] > 0)
        row["equal_lost_vs_id"] = int(methods["id"][i] > 0 and methods["equal"][i] == 0)
        rows.append(row)
    return report, rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--id-dir", type=Path, required=True)
    parser.add_argument("--text-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    report, rows = analyze(load_split(args.id_dir, args.text_dir, "val"),
                           load_split(args.id_dir, args.text_dir, "test"))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    with (args.output_dir / "test_events.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps({"report": str(args.output_dir / "report.json"),
                      "events": str(args.output_dir / "test_events.csv")}, indent=2))


if __name__ == "__main__":
    main()
