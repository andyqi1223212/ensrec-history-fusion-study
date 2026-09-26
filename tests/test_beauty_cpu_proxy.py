import tempfile
import unittest
from pathlib import Path

import numpy as np
from scipy.sparse import csr_matrix

from scripts.beauty_cpu_proxy import (
    _equal_recovery,
    align_catalog,
    analyze,
    bootstrap_rescue_difference,
    fit_id_model,
    rank_percentiles,
    route_cohort_hits,
    select_validation_alpha_by_cohort,
    shuffle_candidate_text,
    topk_indices,
    validate_splits,
    visible_history_target,
    write_svg,
)


def toy_data():
    n_items = 60
    catalog = [(i + 1, f"beauty category{i % 6} brand{i % 4} feature{i % 8}")
               for i in range(n_items)]
    training, evaluation, testing = {}, {}, {}
    for user in range(8):
        length = 3 + user
        train = tuple((user * 7 + step * (user + 1)) % n_items for step in range(length))
        evaluation[user] = train + ((user * 7 + length * (user + 1)) % n_items,)
        testing[user] = evaluation[user] + ((user * 7 + (length + 1) * (user + 1)) % n_items,)
        training[user] = train
    return {"catalog": catalog, "training": training,
            "evaluation": evaluation, "testing": testing}


class BeautyCpuProxyTests(unittest.TestCase):
    def test_target_is_excluded_from_capped_history(self):
        history, target = visible_history_target(tuple(range(30)))
        self.assertEqual(history, tuple(range(10, 29)))
        self.assertEqual(target, 29)
        self.assertNotIn(target, history)

    def test_catalog_metadata_id_maps_to_zero_based_candidate_row(self):
        texts = align_catalog([(2, "second"), (1, "first")])
        self.assertEqual(texts, ["first", "second"])

    def test_transition_model_uses_only_supplied_training_records(self):
        transitions, popularity, summary = fit_id_model({0: (0, 1, 2), 1: (0, 1, 3)}, 4)
        self.assertEqual(transitions[0], {1: 2})
        self.assertEqual(transitions[1], {2: 1, 3: 1})
        self.assertEqual(popularity.tolist(), [2, 2, 1, 1])
        self.assertEqual(summary["training_transitions"], 4)

    def test_split_gate_rejects_target_leakage_or_nonprefix(self):
        good = {
            "training": {0: (0, 1)},
            "evaluation": {0: (0, 1, 2)},
            "testing": {0: (0, 1, 2, 3)},
        }
        self.assertEqual(validate_splits(good, 4)["users"], 1)
        bad = {**good, "testing": {0: (0, 0, 2, 3)}}
        with self.assertRaisesRegex(ValueError, "prefix violations"):
            validate_splits(bad, 4)

    def test_rank_percentiles_and_top10_ties_use_item_id_order(self):
        ids = np.arange(12)
        scores = np.ones(12, dtype=np.float32)
        pct = rank_percentiles(scores, ids)
        np.testing.assert_array_equal(pct, np.linspace(1, 0, 12, dtype=np.float32))
        np.testing.assert_array_equal(topk_indices(scores, ids), np.arange(10))

    def test_test_targets_cannot_change_validation_alpha_or_cutoff(self):
        baseline = toy_data()
        changed_test = toy_data()
        changed_test["testing"] = {
            user: sequence[:-1] + ((sequence[-1] + 1) % len(changed_test["catalog"]),)
            for user, sequence in changed_test["testing"].items()
        }
        report_a = analyze(baseline, bootstrap_replicates=100)
        report_b = analyze(changed_test, bootstrap_replicates=100)
        self.assertEqual(report_a["validation"]["selected_alpha"],
                         report_b["validation"]["selected_alpha"])
        self.assertEqual(report_a["protocol"]["validation_cutoff"],
                         report_b["protocol"]["validation_cutoff"])
        self.assertEqual(report_a["validation"]["recall_at_10_by_alpha"],
                         report_b["validation"]["recall_at_10_by_alpha"])
        self.assertEqual(report_a["validation"]["exploratory_conditional_alpha"],
                         report_b["validation"]["exploratory_conditional_alpha"])
        self.assertIn("equal-alpha-0.5", report_a["test"]["metrics_by_cohort"]["all"])
        self.assertIn("validation-selected-global", report_a["test"]["metrics_by_cohort"]["all"])

    def test_cohort_selection_is_validation_only_and_routing_uses_cutoff_mask(self):
        hits = {
            0.0: np.array([True, True, False, False]),
            0.5: np.array([True, False, True, False]),
            1.0: np.array([False, False, True, True]),
        }
        short = np.array([True, True, False, False])
        alpha = select_validation_alpha_by_cohort(hits, short)
        self.assertEqual(alpha, {"short": 0.0, "long": 1.0})
        test_short_hits = {0.0: np.array([False, True, True, True]),
                           1.0: np.array([True, True, False, False])}
        routed = route_cohort_hits(test_short_hits, alpha, short)
        np.testing.assert_array_equal(routed, [False, True, False, False])

    def test_negative_control_only_permutates_candidate_side_deterministically(self):
        candidates = csr_matrix(np.arange(20).reshape(5, 4))
        shuffled_a, permutation_a = shuffle_candidate_text(candidates, 5, seed=2026)
        shuffled_b, permutation_b = shuffle_candidate_text(candidates, 5, seed=2026)
        np.testing.assert_array_equal(permutation_a, permutation_b)
        np.testing.assert_array_equal(shuffled_a.toarray(), shuffled_b.toarray())
        self.assertEqual(set(permutation_a.tolist()), set(range(5)))
        self.assertFalse(np.array_equal(shuffled_a.toarray(), candidates.toarray()))

    def test_user_bootstrap_captures_perfect_short_long_rescue_difference(self):
        id_hit = np.array([False, False, False, False])
        text_hit = np.array([True, True, False, False])
        short = np.array([True, True, False, False])
        report = bootstrap_rescue_difference(id_hit, text_hit, short, replicates=100, seed=7)
        self.assertEqual(report["short_minus_long"], 1.0)
        self.assertEqual(report["difference_user_bootstrap_95ci"], [1.0, 1.0])

    def test_equal_fusion_recovery_uses_only_text_hits_id_missed(self):
        id_hit = np.array([False, False, True, False])
        text_hit = np.array([True, True, True, False])
        equal_hit = np.array([True, False, False, True])
        result = _equal_recovery(id_hit, text_hit, equal_hit, np.ones(4, dtype=bool))
        self.assertEqual(result, {"text_only_hits": 2, "recovered_by_equal": 1,
                                  "recovery_rate": 0.5})

    def test_small_full_experiment_and_chart_are_aggregate_only(self):
        data = toy_data()
        report = analyze(data, bootstrap_replicates=100)
        self.assertEqual(report["protocol"]["candidate_count"], len(data["catalog"]))
        self.assertFalse(report["protocol"]["test_used_for_selection"])
        self.assertEqual(report["test"]["metrics_by_cohort"]["all"]["ID-only"]["users"], 8)
        self.assertIn("recovered_by_equal",
                      report["test"]["equal_fusion_recovery_of_text_only_hits"]["all"])
        self.assertNotIn("category", str(report))
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "summary.svg"
            write_svg(report, path)
            svg = path.read_text(encoding="utf-8")
            self.assertIn("Short − long Text rescue rate", svg)
            self.assertIn('y="410"', svg)
            self.assertIn('y="444"', svg)


if __name__ == "__main__":
    unittest.main()
