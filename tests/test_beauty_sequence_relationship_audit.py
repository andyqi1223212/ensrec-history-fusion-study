import unittest

import numpy as np

from scripts.beauty_sequence_relationship_audit import _compare_prefix, _summarize_split


class BeautySequenceRelationshipAuditTests(unittest.TestCase):
    def test_one_event_extensions_are_strict_prefixes(self):
        train = {0: (0, 7), 1: (2, 4, 5)}
        evaluation = {0: (0, 7, 9), 1: (2, 4, 5, 6)}
        result = _compare_prefix(train, evaluation)
        self.assertEqual(result["strict_prefix_violations"], 0)
        self.assertEqual(result["exactly_one_new_event_users"], 2)
        self.assertEqual(result["event_count_delta_right_minus_left"], {"1": 2})

    def test_nonprefix_is_counted_without_emitting_user_identity(self):
        result = _compare_prefix({123456: (0, 7)}, {123456: (0, 8)})
        self.assertEqual(result["strict_prefix_violations"], 1)
        self.assertNotIn("123456", str(result))

    def test_split_summary_counts_duplicate_and_out_of_range_events(self):
        rows = [
            {"user_id": np.array([0]), "sequence_data": np.array([0, 1, 9])},
            {"user_id": np.array([0]), "sequence_data": np.array([0, 2, 9])},
            {"user_id": np.array([2]), "sequence_data": np.array([4])},
        ]
        summary, unique_sequences = _summarize_split(rows, catalog_size=5)
        self.assertEqual(summary["duplicate_user_records"], 1)
        self.assertFalse(summary["user_ids_are_contiguous_from_zero"])
        self.assertEqual(summary["out_of_catalog_range_event_ids"], 2)
        self.assertEqual(summary["last_event_target_out_of_catalog_range"], 2)
        self.assertEqual(summary["sequences_too_short_for_next_item_target"], 1)
        self.assertEqual(set(unique_sequences), {2})
        self.assertNotIn("user_ids", summary)


if __name__ == "__main__":
    unittest.main()
