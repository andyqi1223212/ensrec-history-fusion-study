import tempfile
import unittest
import importlib.util
from pathlib import Path

import numpy as np

from scripts.fusion_history_experiment import analyze, comparison, load_split


class FusionHistoryExperimentTest(unittest.TestCase):
    def test_added_and_lost_hits_account_for_recall_delta(self):
        id_rank = np.array([1, 0, 4, 0])
        text_rank = np.array([0, 3, 0, 2])
        fused_rank = np.array([0, 2, 5, 0])
        result = comparison(id_rank, text_rank, fused_rank, np.ones(4, dtype=bool))
        self.assertEqual(result["text_only_hits"], 2)
        self.assertEqual(result["text_only_recovered"], 1)
        self.assertEqual(result["added_hits"], 1)
        self.assertEqual(result["lost_hits"], 1)
        self.assertEqual(result["recall_delta_vs_id"], 0)

    def test_test_labels_do_not_change_selected_weights(self):
        rng = np.random.default_rng(17)

        def split(labels):
            return {"id_users": rng.normal(size=(4, 3)),
                    "text_users": rng.normal(size=(4, 3)),
                    "id_items": rng.normal(size=(12, 3)),
                    "text_items": rng.normal(size=(12, 3)),
                    "labels": np.array(labels), "lengths": np.array([1, 1, 3, 3]),
                    "ids": np.arange(4)}

        validation = split([2, 3, 4, 5])
        test = split([2, 3, 4, 5])
        test["id_items"] = validation["id_items"].copy()
        test["text_items"] = validation["text_items"].copy()
        first, _ = analyze(validation, test)
        altered = {**test, "labels": np.array([5, 4, 3, 2])}
        second, _ = analyze(validation, altered)
        self.assertEqual(first["protocol"]["selected_on_validation"],
                         second["protocol"]["selected_on_validation"])

    def test_candidate_tables_must_match_between_splits(self):
        rng = np.random.default_rng(23)

        def split():
            return {"id_users": rng.normal(size=(4, 3)),
                    "text_users": rng.normal(size=(4, 3)),
                    "id_items": rng.normal(size=(12, 3)),
                    "text_items": rng.normal(size=(12, 3)),
                    "labels": np.array([2, 3, 4, 5]),
                    "lengths": np.array([1, 1, 3, 3]), "ids": np.arange(4)}

        validation, test = split(), split()
        test["id_items"] = validation["id_items"].copy()
        test["text_items"] = validation["text_items"].copy()
        test["text_items"][0, 0] += 1
        with self.assertRaisesRegex(ValueError, "candidate table changed"):
            analyze(validation, test)

    @unittest.skipUnless(importlib.util.find_spec("torch"), "PyTorch is needed to read .pt exports")
    def test_misaligned_labels_are_rejected(self):
        import torch

        with tempfile.TemporaryDirectory() as root:
            left, right = Path(root) / "id", Path(root) / "text"
            left.mkdir()
            right.mkdir()
            for directory in (left, right):
                torch.save(torch.ones(2, 3), directory / "val_user_embeddings.pt")
                torch.save(torch.ones(12, 3), directory / "val_item_embeddings.pt")
                torch.save(torch.tensor([1, 2]), directory / "val_history_lengths.pt")
                torch.save(torch.tensor([0, 1]), directory / "val_user_ids.pt")
            torch.save(torch.tensor([2, 3]), left / "val_user_labels.pt")
            torch.save(torch.tensor([2, 4]), right / "val_user_labels.pt")
            with self.assertRaisesRegex(ValueError, "labels differ"):
                load_split(left, right, "val")


if __name__ == "__main__":
    unittest.main()
