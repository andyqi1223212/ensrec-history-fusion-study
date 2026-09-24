import unittest

from scripts.fusion_item_id_audit import summarize_mapping


class FusionItemIdAuditTest(unittest.TestCase):
    def test_one_based_catalog_offset_matches_first_item_text(self):
        items = [
            {"id": [1], "text": b"first"},
            {"id": [2], "text": b"second"},
        ]
        splits = {
            "evaluation": [
                {"sequence_data": [0, 1], "text": b"first"},
                {"sequence_data": [1], "text": b"second"},
            ]
        }
        report = summarize_mapping(items, splits)
        self.assertEqual(report["item_catalog"]["duplicate_rows"], 0)
        self.assertEqual(report["item_catalog"]["missing_ids_within_range"], [])
        self.assertEqual(report["splits"]["evaluation"]["first_item_text_matches_by_metadata_id_offset"],
                         {"0": 0, "1": 2})
        notebook = report["notebook_failure_reproduction"]
        self.assertEqual(notebook["cell_8_current_first_error"].split(":", 1)[0], "AttributeError")
        self.assertEqual(notebook["after_extracting_scalar_text_cell_13_error"],
                         "KeyError: missing key 0")

    def test_empty_catalog_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "catalog is empty"):
            summarize_mapping([], {"evaluation": []})

    def test_first_item_text_mapping_must_match_or_fail_fast(self):
        items = [
            {"id": [1], "text": b"first"},
            {"id": [2], "text": b"second"},
        ]
        splits = {
            "evaluation": [
                {"sequence_data": [0], "text": b"wrong item text"},
                {"sequence_data": [1], "text": b"second"},
            ]
        }
        with self.assertRaisesRegex(ValueError, "do not all map through metadata_id"):
            summarize_mapping(items, splits)

    def test_catalog_id_gaps_are_rejected(self):
        items = [
            {"id": [1], "text": b"first"},
            {"id": [3], "text": b"third"},
        ]
        with self.assertRaisesRegex(ValueError, "uniquely cover 1..item_count"):
            summarize_mapping(items, {"evaluation": []})


if __name__ == "__main__":
    unittest.main()
