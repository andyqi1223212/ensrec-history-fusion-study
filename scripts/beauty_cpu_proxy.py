"""Run a small Beauty next-item proxy experiment; this is not EnsRec training."""

import argparse
import json
import platform
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import scipy
import sklearn
from scipy.sparse import csr_matrix, vstack
from sklearn.feature_extraction.text import TfidfVectorizer

ALPHAS = tuple(round(x / 10, 1) for x in range(11))
K = 10
HISTORY_LIMIT = 19
POPULARITY_BACKOFF = 0.1
BOOTSTRAP_REPLICATES = 2000
BOOTSTRAP_SEED = 20260926
NEGATIVE_CONTROL_SEED = 2026


def _records(data_dir: Path, split: str):
    from tfrecord.reader import tfrecord_loader

    paths = sorted((data_dir / split).glob("*.tfrecord.gz"))
    if not paths:
        raise FileNotFoundError(f"No gzipped TFRecords in {data_dir / split}")
    for path in paths:
        yield from tfrecord_loader(str(path), None, compression_type="gzip")


def load_tfrecords(data_dir: Path) -> dict:
    catalog = []
    for row in _records(data_dir, "items"):
        text = row["text"]
        if isinstance(text, bytes):
            text = text.decode("utf-8", "replace")
        catalog.append((int(np.asarray(row["id"]).reshape(-1)[0]), str(text)))
    splits = {}
    for split in ("training", "evaluation", "testing"):
        users = {}
        for row in _records(data_dir, split):
            user_id = int(np.asarray(row["user_id"]).reshape(-1)[0])
            if user_id in users:
                raise ValueError(f"{split}: duplicate user record")
            users[user_id] = tuple(int(x) for x in np.asarray(row["sequence_data"]).reshape(-1))
        splits[split] = users
    return {"catalog": catalog, **splits}


def load_prepared_json(path: Path) -> dict:
    """Load a temporary source extraction; never write this input to the repository."""
    data = json.loads(path.read_text(encoding="utf-8"))
    data["catalog"] = [(int(item_id), text) for item_id, text in data["catalog"]]
    for split in ("training", "evaluation", "testing"):
        data[split] = {int(user_id): tuple(map(int, sequence))
                       for user_id, sequence in data[split].items()}
    return data


def align_catalog(rows: list[tuple[int, str]], n_items: int | None = None) -> list[str]:
    if n_items is None:
        n_items = len(rows)
    ids = [int(row[0]) for row in rows]
    expected = set(range(1, n_items + 1))
    if len(rows) != n_items or set(ids) != expected:
        raise ValueError("Catalog metadata IDs must uniquely cover 1..catalog_size")
    ordered = [None] * n_items
    for metadata_id, text in rows:
        ordered[metadata_id - 1] = text
    return ordered


def visible_history_target(sequence: tuple[int, ...]) -> tuple[tuple[int, ...], int]:
    if len(sequence) < 2:
        raise ValueError("Next-item record requires at least one history event and a target")
    return tuple(sequence[-(HISTORY_LIMIT + 1):-1]), int(sequence[-1])


def validate_splits(splits: dict, n_items: int) -> dict:
    names = ("training", "evaluation", "testing")
    expected_users = set(splits[names[0]])
    if not expected_users or expected_users != set(range(len(expected_users))):
        raise ValueError("User IDs must cover a nonempty contiguous range from zero")
    lengths = {}
    for split in names:
        users = splits[split]
        if set(users) != expected_users:
            raise ValueError(f"{split}: user set differs across splits")
        lengths[split] = []
        for sequence in users.values():
            if len(sequence) < 2:
                raise ValueError(f"{split}: sequence too short for next-item target")
            if any(item < 0 or item >= n_items for item in sequence):
                raise ValueError(f"{split}: event ID outside 0-based catalog range")
            lengths[split].append(len(sequence))
    for left, right in (("training", "evaluation"), ("evaluation", "testing")):
        violations = sum(
            not (len(splits[right][uid]) == len(splits[left][uid]) + 1
                 and splits[right][uid][:-1] == splits[left][uid])
            for uid in expected_users
        )
        if violations:
            raise ValueError(f"{left}/{right}: strict one-event prefix violations={violations}")
    return {
        "users": len(expected_users),
        "sequence_lengths": {
            split: {"minimum": min(vals), "maximum": max(vals), "mean": round(float(np.mean(vals)), 4)}
            for split, vals in lengths.items()
        },
    }


def fit_id_model(training: dict[int, tuple[int, ...]], n_items: int):
    transitions = [Counter() for _ in range(n_items)]
    popularity = np.zeros(n_items, dtype=np.int64)
    pair_count = 0
    for sequence in training.values():
        for item in sequence:
            popularity[item] += 1
        for source, target in zip(sequence, sequence[1:]):
            transitions[source][target] += 1
            pair_count += 1
    unique_edges = sum(len(row) for row in transitions)
    if popularity.sum() == 0 or pair_count == 0:
        raise ValueError("Training split contains no events/transitions")
    return transitions, popularity, {"training_events": int(popularity.sum()),
                                     "training_transitions": pair_count,
                                     "unique_transition_edges": unique_edges}


def id_scores(history: tuple[int, ...], transitions, popularity: np.ndarray) -> np.ndarray:
    n_items = len(popularity)
    scores = np.zeros(n_items, dtype=np.float32)
    sources = history[-HISTORY_LIMIT:]
    used = 0
    for source in sources:
        outgoing = transitions[source]
        total = sum(outgoing.values())
        if total:
            used += 1
            for target, count in outgoing.items():
                scores[target] += count / total
    if used:
        scores /= used
    popularity_prior = popularity.astype(np.float32) / float(popularity.sum())
    if used:
        scores += POPULARITY_BACKOFF * popularity_prior
    else:
        scores = popularity_prior
    return scores


def build_tfidf(catalog_texts: list[str]):
    vectorizer = TfidfVectorizer(
        lowercase=True,
        strip_accents="unicode",
        token_pattern=r"(?u)\b[a-zA-Z0-9][a-zA-Z0-9]+\b",
        min_df=2,
        max_df=0.85,
        sublinear_tf=True,
        smooth_idf=True,
        norm="l2",
        dtype=np.float32,
    )
    matrix = vectorizer.fit_transform(catalog_texts).tocsr()
    return vectorizer, matrix


def history_profiles(histories: list[tuple[int, ...]], item_text_matrix):
    profiles = []
    empty = 0
    for history in histories:
        rows = item_text_matrix[np.asarray(history, dtype=np.int32)]
        profile = csr_matrix(rows.sum(axis=0), dtype=np.float32)
        norm = float(np.sqrt(profile.multiply(profile).sum()))
        if norm:
            profile.data /= norm
        else:
            empty += 1
        profiles.append(profile)
    return vstack(profiles, format="csr"), empty


def shuffle_candidate_text(item_text_matrix, n_items: int, seed: int):
    permutation = np.random.default_rng(seed).permutation(n_items)
    return item_text_matrix[permutation], permutation


def rank_percentiles(scores: np.ndarray, item_ids: np.ndarray,
                     popularity: np.ndarray | None = None) -> np.ndarray:
    if popularity is None:
        order = np.lexsort((item_ids, -scores))
    else:
        # Primary: transition score; tie-break: training popularity; then item ID.
        order = np.lexsort((item_ids, -popularity, -scores))
    percentiles = np.empty(len(scores), dtype=np.float32)
    percentiles[order] = np.linspace(1.0, 0.0, len(scores), dtype=np.float32)
    return percentiles


def topk_indices(scores: np.ndarray, item_ids: np.ndarray, k: int = K) -> np.ndarray:
    if len(scores) < k:
        raise ValueError("Candidate catalog smaller than K")
    candidates = np.argpartition(scores, len(scores) - k)[-k:]
    cutoff = float(scores[candidates].min())
    above = np.flatnonzero(scores > cutoff)
    tied = np.flatnonzero(scores == cutoff)
    need = k - len(above)
    if need:
        tied = tied[np.argsort(item_ids[tied], kind="stable")[:need]]
        candidates = np.concatenate((above, tied))
    else:
        candidates = above
    order = np.lexsort((item_ids[candidates], -scores[candidates]))
    return candidates[order]


def target_rank(scores: np.ndarray, target_id: int, item_ids: np.ndarray) -> int:
    """One-based rank under descending score and ascending item-ID tie-break."""
    target_score = scores[target_id]
    ahead = (scores > target_score) | ((scores == target_score) & (item_ids < target_id))
    return int(np.count_nonzero(ahead)) + 1


def predict_hits(examples: list[tuple[int, tuple[int, ...], int]], item_text_matrix,
                 transitions, popularity, alphas: tuple[float, ...],
                 candidate_matrices: dict | None = None,
                 batch_size: int = 128) -> dict:
    n_items = item_text_matrix.shape[0]
    item_ids = np.arange(n_items, dtype=np.int32)
    candidate_matrices = candidate_matrices or {"aligned": item_text_matrix}
    histories = [history for _, history, _ in examples]
    profiles, empty_profiles = history_profiles(histories, item_text_matrix)
    variants = {
        name: {"fusion": {alpha: np.zeros(len(examples), dtype=bool) for alpha in alphas},
               "fusion_rank": {alpha: np.zeros(len(examples), dtype=np.int32) for alpha in alphas},
               "text": np.zeros(len(examples), dtype=bool),
               "text_rank": np.zeros(len(examples), dtype=np.int32)}
        for name in candidate_matrices
    }
    id_hits = np.zeros(len(examples), dtype=bool)
    id_ranks = np.zeros(len(examples), dtype=np.int32)
    for start in range(0, len(examples), batch_size):
        end = min(start + batch_size, len(examples))
        similarities = {
            name: (profiles[start:end] @ matrix.T).tocsr()
            for name, matrix in candidate_matrices.items()
        }
        for local, index in enumerate(range(start, end)):
            _, history, target = examples[index]
            score_id = id_scores(history, transitions, popularity)
            id_pct = rank_percentiles(score_id, item_ids, popularity)
            id_top = topk_indices(id_pct, item_ids)
            id_hits[index] = target in id_top
            id_ranks[index] = target_rank(id_pct, target, item_ids)
            for name, sim in similarities.items():
                row = sim.getrow(local).toarray().reshape(-1)
                text_pct = rank_percentiles(row, item_ids)
                variants[name]["text"][index] = target in topk_indices(text_pct, item_ids)
                variants[name]["text_rank"][index] = target_rank(text_pct, target, item_ids)
                for alpha in alphas:
                    fused = alpha * id_pct + (1.0 - alpha) * text_pct
                    variants[name]["fusion"][alpha][index] = target in topk_indices(fused, item_ids)
                    variants[name]["fusion_rank"][alpha][index] = target_rank(fused, target, item_ids)
    return {"id": id_hits, "id_rank": id_ranks, "variants": variants,
            "empty_history_text_profiles": empty_profiles}


def examples_for_split(split: dict[int, tuple[int, ...]]) -> list[tuple[int, tuple[int, ...], int]]:
    out = []
    for user_id in sorted(split):
        history, target = visible_history_target(split[user_id])
        out.append((user_id, history, target))
    return out


def select_validation_alpha(validation_hits: dict[float, np.ndarray]) -> float:
    # Validation-only choice; ties favor the equal-weight baseline, then larger ID weight.
    return max(validation_hits, key=lambda alpha: (
        float(validation_hits[alpha].mean()), -abs(alpha - 0.5), alpha
    ))


def select_validation_alpha_by_cohort(validation_hits: dict[float, np.ndarray],
                                      short_mask: np.ndarray) -> dict[str, float]:
    return {
        "short": select_validation_alpha({a: hits[short_mask] for a, hits in validation_hits.items()}),
        "long": select_validation_alpha({a: hits[~short_mask] for a, hits in validation_hits.items()}),
    }


def route_cohort_hits(fusion_hits: dict[float, np.ndarray], alphas: dict[str, float],
                      short_mask: np.ndarray) -> np.ndarray:
    routed = np.empty(len(short_mask), dtype=next(iter(fusion_hits.values())).dtype)
    routed[short_mask] = fusion_hits[alphas["short"]][short_mask]
    routed[~short_mask] = fusion_hits[alphas["long"]][~short_mask]
    return routed


def bootstrap_paired_hit_difference(candidate_hits: np.ndarray, baseline_hits: np.ndarray,
                                    replicates: int = BOOTSTRAP_REPLICATES,
                                    seed: int = BOOTSTRAP_SEED) -> dict:
    differences = candidate_hits.astype(np.float64) - baseline_hits.astype(np.float64)
    rng = np.random.default_rng(seed)
    samples = np.empty(replicates, dtype=np.float64)
    for i in range(replicates):
        samples[i] = differences[rng.integers(0, len(differences), size=len(differences))].mean()
    return {"mean_recall_difference": float(differences.mean()),
            "user_bootstrap_95ci": [float(x) for x in np.quantile(samples, [0.025, 0.975])],
            "bootstrap_replicates": int(replicates), "bootstrap_seed": int(seed)}


def choose_validation_cutoff(validation_lengths: np.ndarray) -> tuple[int, int]:
    cutoff = int(np.median(validation_lengths))
    n_short = int(np.sum(validation_lengths <= cutoff))
    if n_short == 0 or n_short == len(validation_lengths):
        raise ValueError("Validation median did not form two nonempty history cohorts")
    return cutoff, n_short


def _metric(hits: np.ndarray, mask: np.ndarray) -> dict:
    n = int(mask.sum())
    if n == 0:
        raise ValueError("Empty history cohort")
    count = int(np.sum(hits & mask))
    return {"users": n, "hits": count, "recall@10": count / n}


def _ranking_metric(ranks: np.ndarray, mask: np.ndarray) -> dict:
    n = int(mask.sum())
    if n == 0:
        raise ValueError("Empty ranking cohort")
    selected = ranks[mask]
    top1 = int(np.sum(selected == 1))
    top10 = int(np.sum(selected <= K))
    discounted = np.where(selected <= K, 1.0 / np.log2(selected + 1), 0.0)
    return {"users": n, "ndcg@10": float(discounted.mean()),
            "hit@1": top1 / n, "hit@1_count": top1, "hit@10_count": top10}


def _rescue_counts(id_hits: np.ndarray, text_hits: np.ndarray, mask: np.ndarray) -> dict:
    misses = mask & ~id_hits
    denominator = int(misses.sum())
    numerator = int(np.sum(misses & text_hits))
    return {"users": int(mask.sum()), "id_top10_misses": denominator,
            "text_hits_among_id_misses": numerator,
            "hit_rate": numerator / denominator if denominator else None}


def _difference_rate(short: dict, long: dict) -> float | None:
    if short["hit_rate"] is None or long["hit_rate"] is None:
        return None
    return short["hit_rate"] - long["hit_rate"]


def training_frequency_bands(popularity: np.ndarray) -> tuple[dict[str, np.ndarray], dict]:
    """Use fixed catalog-wide training-event frequency tertiles; empty bins stay empty."""
    lower, upper = (float(x) for x in np.quantile(popularity, [1 / 3, 2 / 3]))
    bands = {
        "low": popularity <= lower,
        "middle": (popularity > lower) & (popularity <= upper),
        "high": popularity > upper,
    }
    details = {"lower_cutpoint": lower, "upper_cutpoint": upper,
               "cutpoints_repeated": lower == upper,
               "method": "catalog item occurrence counts in training; numpy linear quantiles at 1/3 and 2/3; bins are <=lower, (lower,upper], >upper",
               "catalog_items_by_band": {name: int(mask.sum()) for name, mask in bands.items()},
               "empty_bands": [name for name, mask in bands.items() if not np.any(mask)]}
    return bands, details


def _write_event_diagnostics(path: Path, examples: list[tuple[int, tuple[int, ...], int]],
                             id_ranks: np.ndarray, text_ranks: np.ndarray,
                             equal_ranks: np.ndarray, global_ranks: np.ndarray,
                             conditional_ranks: np.ndarray) -> None:
    with path.open("w", encoding="utf-8") as out:
        for i, (user_id, history, target) in enumerate(examples):
            id_rank = int(id_ranks[i])
            row = {
                "user_id": int(user_id), "target_item_id": int(target),
                "visible_history_length": len(history),
                "target_rank": {"ID": id_rank if id_rank <= K else 0,
                                "Text": int(text_ranks[i]) if text_ranks[i] <= K else 0,
                                "equal": int(equal_ranks[i]) if equal_ranks[i] <= K else 0,
                                "global": int(global_ranks[i]) if global_ranks[i] <= K else 0,
                                "conditional": int(conditional_ranks[i]) if conditional_ranks[i] <= K else 0},
                "equal_vs_ID": {"added": bool(id_rank > K and equal_ranks[i] <= K),
                                "lost": bool(id_rank <= K and equal_ranks[i] > K)},
                "global_vs_ID": {"added": bool(id_rank > K and global_ranks[i] <= K),
                                 "lost": bool(id_rank <= K and global_ranks[i] > K)},
            }
            out.write(json.dumps(row, separators=(",", ":")) + "\n")


def _gain(id_hits: np.ndarray, fusion_hits: np.ndarray, mask: np.ndarray) -> dict:
    added = int(np.sum(~id_hits & fusion_hits & mask))
    lost = int(np.sum(id_hits & ~fusion_hits & mask))
    users = int(mask.sum())
    return {"added_hits": added, "lost_hits": lost, "net_hits": added - lost,
            "recall@10_net_gain": (added - lost) / users}


def _rescue_rate(id_hits: np.ndarray, text_hits: np.ndarray, mask: np.ndarray) -> dict:
    misses = mask & ~id_hits
    denominator = int(misses.sum())
    numerator = int(np.sum(misses & text_hits))
    return {"cohort_users": int(mask.sum()), "id_top10_misses": denominator,
            "text_top10_hits_among_id_misses": numerator,
            "hit_rate": numerator / denominator if denominator else None}


def _equal_recovery(id_hits: np.ndarray, text_hits: np.ndarray,
                    equal_hits: np.ndarray, mask: np.ndarray) -> dict:
    text_only = mask & ~id_hits & text_hits
    denominator = int(text_only.sum())
    recovered = int(np.sum(text_only & equal_hits))
    return {"text_only_hits": denominator, "recovered_by_equal": recovered,
            "recovery_rate": recovered / denominator if denominator else None}


def bootstrap_rescue_difference(id_hits: np.ndarray, text_hits: np.ndarray,
                                short_mask: np.ndarray, replicates: int = BOOTSTRAP_REPLICATES,
                                seed: int = BOOTSTRAP_SEED) -> dict:
    rng = np.random.default_rng(seed)
    groups = [np.flatnonzero(short_mask), np.flatnonzero(~short_mask)]
    rates = []
    samples = np.empty(replicates, dtype=np.float64)
    for group in groups:
        miss = ~id_hits[group]
        rescued = miss & text_hits[group]
        rates.append((miss, rescued))
    for replicate in range(replicates):
        pair = []
        for indices, (miss, rescued) in zip(groups, rates):
            sampled = rng.integers(0, len(indices), size=len(indices))
            denom = int(miss[sampled].sum())
            pair.append(float(rescued[sampled].sum() / denom) if denom else np.nan)
        samples[replicate] = pair[0] - pair[1]
    finite = samples[np.isfinite(samples)]
    if not len(finite):
        ci = [None, None]
    else:
        ci = [float(x) for x in np.quantile(finite, [0.025, 0.975])]
    observed_short = _rescue_rate(id_hits, text_hits, short_mask)
    observed_long = _rescue_rate(id_hits, text_hits, ~short_mask)
    observed_difference = (observed_short["hit_rate"] - observed_long["hit_rate"]
                          if observed_short["hit_rate"] is not None and observed_long["hit_rate"] is not None
                          else None)
    return {"short": observed_short, "long": observed_long,
            "short_minus_long": observed_difference,
            "difference_user_bootstrap_95ci": ci,
            "valid_bootstrap_replicates": int(len(finite)),
            "bootstrap_replicates": int(replicates), "bootstrap_seed": int(seed)}


def analyze(data: dict, bootstrap_replicates: int = BOOTSTRAP_REPLICATES,
            event_diagnostics_path: Path | None = None) -> dict:
    n_items = len(data["catalog"])
    catalog_texts = align_catalog(data["catalog"], n_items)
    data_audit = validate_splits({s: data[s] for s in ("training", "evaluation", "testing")}, n_items)
    transitions, popularity, training_summary = fit_id_model(data["training"], n_items)
    vectorizer, text_matrix = build_tfidf(catalog_texts)
    validation_examples = examples_for_split(data["evaluation"])
    test_examples = examples_for_split(data["testing"])
    validation_lengths = np.asarray([len(h) for _, h, _ in validation_examples], dtype=np.int32)
    test_lengths = np.asarray([len(h) for _, h, _ in test_examples], dtype=np.int32)
    cutoff, val_short_count = choose_validation_cutoff(validation_lengths)

    val_bundle = predict_hits(validation_examples, text_matrix, transitions, popularity, ALPHAS)
    val_result = val_bundle["variants"]["aligned"]
    selected_alpha = select_validation_alpha(val_result["fusion"])
    val_short = validation_lengths <= cutoff
    conditional_alphas = select_validation_alpha_by_cohort(val_result["fusion"], val_short)
    test_short = test_lengths <= cutoff
    validation_alpha_neighborhood = {
        str(alpha): {
            cohort: _metric(val_result["fusion"][alpha], mask)
            for cohort, mask in {"short": val_short, "long": ~val_short}.items()
        }
        for alpha in ALPHAS
    }
    test_weights = tuple(sorted({0.0, 0.5, selected_alpha, *conditional_alphas.values()}))
    control_matrix, _ = shuffle_candidate_text(text_matrix, n_items, NEGATIVE_CONTROL_SEED)
    test_bundle = predict_hits(test_examples, text_matrix, transitions, popularity, test_weights,
                               candidate_matrices={"aligned": text_matrix,
                                                   "candidate_text_permuted": control_matrix})
    test_result = test_bundle["variants"]["aligned"]
    control_result = test_bundle["variants"]["candidate_text_permuted"]
    test_id_hits = test_bundle["id"]
    conditional_hits = route_cohort_hits(test_result["fusion"], conditional_alphas, test_short)
    conditional_ranks = route_cohort_hits(test_result["fusion_rank"], conditional_alphas, test_short)

    val_grid = {str(a): _metric(val_result["fusion"][a], np.ones(len(validation_examples), dtype=bool))
                for a in ALPHAS}
    test_methods = {
        "ID-only": test_id_hits, "Text-only": test_result["text"],
        "equal-alpha-0.5": test_result["fusion"][0.5],
        "validation-selected-global": test_result["fusion"][selected_alpha],
    }
    cohorts = {"short": test_short, "long": ~test_short, "all": np.ones(len(test_short), dtype=bool)}
    metrics = {
        cohort: {name: _metric(hits, mask) for name, hits in test_methods.items()}
        for cohort, mask in cohorts.items()
    }
    gains = {
        cohort: {
            "equal_vs_ID": _gain(test_id_hits, test_result["fusion"][0.5], mask),
            "global_vs_ID": _gain(test_id_hits, test_result["fusion"][selected_alpha], mask),
        }
        for cohort, mask in cohorts.items()
    }
    conditional_comparisons = {}
    for cohort, mask in cohorts.items():
        conditional_comparisons[cohort] = {
            "metrics": _metric(conditional_hits, mask),
            "conditional_vs_global": _gain(test_result["fusion"][selected_alpha], conditional_hits, mask),
            "conditional_vs_equal": _gain(test_result["fusion"][0.5], conditional_hits, mask),
            "conditional_vs_ID": _gain(test_id_hits, conditional_hits, mask),
        }
    conditional_comparisons["all"]["conditional_vs_global_paired_user_bootstrap"] = \
        bootstrap_paired_hit_difference(conditional_hits, test_result["fusion"][selected_alpha],
                                        bootstrap_replicates)
    test_ranks = {
        "ID-only": test_bundle["id_rank"],
        "Text-only": test_result["text_rank"],
        "equal-alpha-0.5": test_result["fusion_rank"][0.5],
        "validation-selected-global": test_result["fusion_rank"][selected_alpha],
        "exploratory-conditional": conditional_ranks,
    }
    rank_metrics = {
        cohort: {name: _ranking_metric(ranks, mask) for name, ranks in test_ranks.items()}
        for cohort, mask in cohorts.items()
    }
    if event_diagnostics_path is not None:
        _write_event_diagnostics(
            event_diagnostics_path, test_examples, test_bundle["id_rank"],
            test_result["text_rank"], test_result["fusion_rank"][0.5],
            test_result["fusion_rank"][selected_alpha], conditional_ranks)

    frequency_bands, frequency_band_info = training_frequency_bands(popularity)
    targets = np.asarray([target for _, _, target in test_examples], dtype=np.int32)
    target_frequencies = popularity[targets]
    frequency_diagnostics = {}
    for name, band_mask in frequency_bands.items():
        short_counts = _rescue_counts(test_id_hits, test_result["text"], band_mask[targets] & test_short)
        long_counts = _rescue_counts(test_id_hits, test_result["text"], band_mask[targets] & ~test_short)
        frequency_diagnostics[name] = {
            "target_frequency_range": {
                "minimum": int(target_frequencies[band_mask[targets]].min()) if np.any(band_mask[targets]) else None,
                "maximum": int(target_frequencies[band_mask[targets]].max()) if np.any(band_mask[targets]) else None,
            },
            "short": short_counts, "long": long_counts,
            "short_minus_long": _difference_rate(short_counts, long_counts),
        }
    history_length_diagnostics = {}
    for length in range(1, HISTORY_LIMIT + 1):
        mask = test_lengths == length
        counts = _rescue_counts(test_id_hits, test_result["text"], mask)
        history_length_diagnostics[str(length)] = {
            **counts, "is_truncated_bucket": length == HISTORY_LIMIT,
            "visible_history_definition": "19 means 19 or more pre-target events were truncated to the latest 19" if length == HISTORY_LIMIT else "exact visible pre-target event count",
        }
    rescue = bootstrap_rescue_difference(test_id_hits, test_result["text"],
                                         test_short, bootstrap_replicates)
    equal_recovery = {
        cohort: _equal_recovery(test_id_hits, test_result["text"],
                                test_result["fusion"][0.5], mask)
        for cohort, mask in cohorts.items()
    }
    control_metrics = {
        "Text-only": {cohort: _metric(control_result["text"], mask) for cohort, mask in cohorts.items()},
        "equal-alpha-0.5": {cohort: _metric(control_result["fusion"][0.5], mask) for cohort, mask in cohorts.items()},
        "validation-selected-global-alpha": {
            "alpha": selected_alpha,
            "metrics": {cohort: _metric(control_result["fusion"][selected_alpha], mask)
                        for cohort, mask in cohorts.items()},
        },
    }
    return {
        "scope": "Beauty CPU proxy experiment; not EnsRec training or a paper reproduction",
        "runtime": {"python": platform.python_version(), "numpy": np.__version__,
                    "scipy": scipy.__version__, "scikit_learn": sklearn.__version__},
        "protocol": {
            "candidate_count": n_items, "candidate_sequence_ids": "0..N-1",
            "catalog_metadata_ids": "1..N; candidate_index=metadata_id-1",
            "target": "last item in evaluation/testing record; exactly one target per user",
            "visible_history": "sequence[-20:-1], maximum 19 events; target excluded",
            "id_model": "train-only item-to-next-item conditional transition probabilities averaged over available sources in the latest 19 history items; fixed 0.1 train-popularity backoff; when no source has transitions use popularity alone",
            "id_tiebreak": "train-only item popularity descending, then candidate ID ascending",
            "text_model": "catalog-only scikit-learn TfidfVectorizer: lowercase, unicode accent stripping, sublinear TF, smooth IDF, L2 normalization, min_df=2, max_df=0.85; query is normalized sum of visible-history item vectors",
            "fusion": "deterministic per-query candidate rank percentiles, higher is better; score ties broken by candidate ID; alpha*ID_percentile + (1-alpha)*Text_percentile",
            "alpha_grid": list(ALPHAS), "selected_alpha_is_id_weight": True,
            "validation_cutoff": cutoff, "short_rule": "visible history length <= validation median cutoff",
            "target_frequency_diagnostic": "post hoc only; bands use training catalog event-frequency tertile cutpoints and do not route prediction",
            "validation_short_users": val_short_count,
            "validation_long_users": int((~val_short).sum()),
            "weight_selection": "validation Recall@10 only; ties prefer alpha nearest 0.5, then larger ID alpha",
            "exploratory_followup": "after initial proxy results; validation-only short/long alpha selection, routed by visible history length; not confirmatory or preregistered",
            "test_used_for_selection": False,
            "primary_test_metric": "Text top-10 hit rate among ID top-10 misses, short minus long; user bootstrap 95% CI",
            "negative_control": "permute candidate-side catalog text vectors against item IDs with fixed seed; keep user history text profiles real; never used for validation selection",
            "negative_control_seed": NEGATIVE_CONTROL_SEED,
            "bootstrap_seed": BOOTSTRAP_SEED,
            "bootstrap_replicates": bootstrap_replicates,
            "validation_test_split_checks": data_audit,
            "training_model_counts": training_summary,
            "tfidf_vocabulary_size": len(vectorizer.vocabulary_),
            "tfidf_catalog_nonzero_entries": int(text_matrix.nnz),
            "empty_validation_text_profiles": val_bundle["empty_history_text_profiles"],
            "empty_test_text_profiles": test_bundle["empty_history_text_profiles"],
        },
        "validation": {"selected_alpha": selected_alpha, "recall_at_10_by_alpha": val_grid,
                        "recall_at_10_by_alpha_by_cohort": validation_alpha_neighborhood,
                        "exploratory_conditional_alpha": {
                            cohort: {"selected_alpha": alpha,
                                     "users": int((val_short if cohort == "short" else ~val_short).sum()),
                                     "recall_at_10": float(val_result["fusion"][alpha][val_short if cohort == "short" else ~val_short].mean())}
                            for cohort, alpha in conditional_alphas.items()}},
        "test": {"metrics_by_cohort": metrics, "ranking_metrics_by_cohort": rank_metrics,
                 "fusion_added_lost_net_hits": gains,
                 "exploratory_conditional_fusion": conditional_comparisons,
                 "posthoc_alternative_explanations": {
                     "target_item_training_frequency_tertiles": {
                         **frequency_band_info, "bands": frequency_diagnostics,
                         "difference_definition": "short text-rescue rate minus long text-rescue rate; null if either ID-miss denominator is empty",
                         "target_frequency_band_used_for_routing_or_alpha_selection": False,
                     },
                     "visible_history_length": history_length_diagnostics,
                     "text_rescue_definition": "Text top-10 hits among ID top-10 misses",
                 },
                 "equal_fusion_recovery_of_text_only_hits": equal_recovery,
                 "text_rescue_among_id_misses": rescue,
                 "candidate_text_permutation_control": control_metrics},
    }


def _percent(value):
    return "n/a" if value is None else f"{100 * value:.1f}%"


def write_svg(report: dict, path: Path):
    test = report["test"]
    methods = ["ID-only", "Text-only", "equal-alpha-0.5", "validation-selected-global"]
    colors = ["#52647a", "#bd6b43", "#2b8a78", "#7556a5"]
    W, H = 1000, 570
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}">',
             '<rect width="100%" height="100%" fill="#fff"/>',
             '<style>text{font-family:Arial,sans-serif;fill:#263238}.title{font-size:22px;font-weight:bold}.head{font-size:16px;font-weight:bold}.label{font-size:13px}.small{font-size:11px}</style>',
             '<text x="40" y="36" class="title">Beauty CPU proxy: next-item hit rates</text>',
             '<text x="40" y="59" class="small">Full 12,101-item catalog · rank-percentile fusion · not EnsRec training</text>']
    # Panel 1: test Recall@10 by method.
    x0, y0, chart_w, chart_h = 68, 105, 570, 165
    parts.append(f'<text x="{x0}" y="87" class="head">Test Recall@10</text>')
    for tick in range(0, 6):
        y = y0 + chart_h - tick * chart_h / 5
        parts.append(f'<line x1="{x0}" y1="{y:.1f}" x2="{x0+chart_w}" y2="{y:.1f}" stroke="#e2e7eb"/>')
        parts.append(f'<text x="{x0-10}" y="{y+4:.1f}" text-anchor="end" class="small">{tick*2}%</text>')
    gap, bw = 22, 82
    for i, method in enumerate(methods):
        value = test["metrics_by_cohort"]["all"][method]["recall@10"]
        x = x0 + 28 + i * (bw + gap)
        bar_h = chart_h * value / 0.10
        y = y0 + chart_h - bar_h
        parts.append(f'<rect x="{x}" y="{y:.1f}" width="{bw}" height="{bar_h:.1f}" rx="4" fill="{colors[i]}"/>')
        parts.append(f'<text x="{x+bw/2}" y="{y-7:.1f}" text-anchor="middle" class="small">{_percent(value)}</text>')
        short_label = {"ID-only": "ID", "Text-only": "Text", "equal-alpha-0.5": "Equal",
                       "validation-selected-global": f"Val-selected α={report['validation']['selected_alpha']}"}[method]
        parts.append(f'<text x="{x+bw/2}" y="{y0+chart_h+18}" text-anchor="middle" class="small">{short_label}</text>')
    # Panel 2: text rescue among ID misses by test cohort with bootstrap intervals.
    x1, y1, w1, h1 = 700, 105, 240, 165
    parts.append(f'<text x="{x1}" y="87" class="head">Text hit rate on ID misses</text>')
    rescue = test["text_rescue_among_id_misses"]
    for i, name in enumerate(("short", "long")):
        row = rescue[name]
        value = row["hit_rate"] or 0.0
        x = x1 + 22 + i * 112
        bar_h = h1 * value / 0.10
        y = y1 + h1 - bar_h
        parts.append(f'<rect x="{x}" y="{y:.1f}" width="70" height="{bar_h:.1f}" rx="4" fill="{colors[1]}"/>')
        parts.append(f'<text x="{x+35}" y="{y-7:.1f}" text-anchor="middle" class="small">{_percent(row["hit_rate"])}</text>')
        parts.append(f'<text x="{x+35}" y="{y1+h1+18}" text-anchor="middle" class="small">{name.title()}</text>')
        parts.append(f'<text x="{x+35}" y="{y1+h1+35}" text-anchor="middle" class="small">users={row["cohort_users"]}</text>')
        parts.append(f'<text x="{x+35}" y="{y1+h1+51}" text-anchor="middle" class="small">misses={row["id_top10_misses"]}</text>')
    # Lower table shows the key contrast, gains, and a semantic alignment control.
    parts.append('<text x="40" y="340" class="head">Complementarity and controls</text>')
    diff = rescue["short_minus_long"]
    ci = rescue["difference_user_bootstrap_95ci"]
    diff_ci = "n/a" if diff is None else f"{100*diff:+.2f} pp (95% CI {100*ci[0]:+.2f} to {100*ci[1]:+.2f} pp)"
    equal_gain = test["fusion_added_lost_net_hits"]["all"]["equal_vs_ID"]
    global_gain = test["fusion_added_lost_net_hits"]["all"]["global_vs_ID"]
    control_equal = test["candidate_text_permutation_control"]["equal-alpha-0.5"]["all"]["recall@10"]
    recovered = test["equal_fusion_recovery_of_text_only_hits"]["all"]
    lines = [
        f"Short − long Text rescue rate: {diff_ci}",
        f"Equal fusion added/lost/net hits vs ID: {equal_gain['added_hits']} / {equal_gain['lost_hits']} / {equal_gain['net_hits']:+d}",
        f"Validation-weighted fusion net hits vs ID: {global_gain['net_hits']:+d}",
        f"Shuffled candidate-text equal-fusion Recall@10 (alignment negative control): {_percent(control_equal)}",
        f"Equal fusion retained text-only hits: {recovered['recovered_by_equal']}/{recovered['text_only_hits']} ({_percent(recovered['recovery_rate'])})",
    ]
    for i, line in enumerate(lines):
        y = 376 + i * 34
        parts.append(f'<text x="48" y="{y}" class="label">{line}</text>')
        if i < len(lines) - 1:
            parts.append(f'<line x1="48" y1="{y+12}" x2="950" y2="{y+12}" stroke="#edf0f2"/>')
    parts.append('<text x="40" y="535" class="small">Scores are aggregate proxy results. Candidate-side text is shuffled only for the negative control; no user-level rows are included.</text>')
    parts.append('</svg>')
    path.write_text("\n".join(parts), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, help="Raw Beauty directory; requires tfrecord")
    parser.add_argument("--prepared-json", type=Path,
                        help="Temporary input extracted from TFRecords; contains private raw text and sequences")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-replicates", type=int, default=BOOTSTRAP_REPLICATES)
    parser.add_argument("--write-event-diagnostics", action="store_true",
                        help="Write private per-test-event JSONL into --output-dir (not for publication)")
    args = parser.parse_args()
    if bool(args.data_dir) == bool(args.prepared_json):
        parser.error("Specify exactly one of --data-dir or --prepared-json")
    if args.bootstrap_replicates < 100:
        parser.error("Use at least 100 bootstrap replicates")
    data = load_tfrecords(args.data_dir) if args.data_dir else load_prepared_json(args.prepared_json)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    event_path = args.output_dir / "beauty-cpu-proxy.event-diagnostics.jsonl" if args.write_event_diagnostics else None
    report = analyze(data, args.bootstrap_replicates, event_path)
    json_path = args.output_dir / "beauty-cpu-proxy-results.json"
    svg_path = args.output_dir / "beauty-cpu-proxy-summary.svg"
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_svg(report, svg_path)
    print(json.dumps({"results": str(json_path), "chart": str(svg_path),
                      "event_diagnostics": str(event_path) if event_path else None,
                      "selected_alpha": report["validation"]["selected_alpha"],
                      "test_primary": report["test"]["text_rescue_among_id_misses"]}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
