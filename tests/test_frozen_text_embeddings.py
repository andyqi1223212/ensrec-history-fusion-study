import unittest

import torch

from scripts.frozen_text_embeddings import (
    encode_catalog,
    fake_encode,
    text_hash,
    text_value,
    verify_table,
)


class FrozenTextEmbeddingTest(unittest.TestCase):
    def setUp(self):
        self.catalog = [(2, "second"), (1, "first")]
        self.table, self.rows = encode_catalog(
            self.catalog, lambda texts: fake_encode(texts, 4), 4, batch_size=1
        )

    def test_sorts_catalog_and_places_rows_after_two_zero_tokens(self):
        self.assertEqual(self.table.shape, (4, 4))
        self.assertTrue(torch.equal(self.table[:2], torch.zeros((2, 4))))
        self.assertEqual([(row["row"], row["metadata_id"]) for row in self.rows], [(2, 1), (3, 2)])
        self.assertEqual(self.rows[0]["source_text_hash"], text_hash("first"))
        verify_table(self.table, self.rows, item_count=2, dimension=4)

    def test_fake_encoder_is_deterministic_and_finite(self):
        first = fake_encode(["same text"], 17)
        second = fake_encode(["same text"], 17)
        self.assertTrue(torch.equal(first, second))
        self.assertTrue(torch.isfinite(first).all())

    def test_manifest_must_have_complete_ids_and_exact_row_mapping(self):
        with self.assertRaisesRegex(ValueError, "cover IDs"):
            verify_table(self.table, self.rows[:1], item_count=2, dimension=4)
        wrong_order = list(reversed(self.rows))
        with self.assertRaisesRegex(ValueError, "cover IDs"):
            verify_table(self.table, wrong_order, item_count=2, dimension=4)

    def test_zero_placeholders_and_finite_values_are_required(self):
        bad = self.table.clone()
        bad[0, 0] = 1
        with self.assertRaisesRegex(ValueError, "zero placeholder"):
            verify_table(bad, self.rows, item_count=2, dimension=4)
        bad = self.table.clone()
        bad[2, 0] = float("nan")
        with self.assertRaisesRegex(ValueError, "finite"):
            verify_table(bad, self.rows, item_count=2, dimension=4)

    def test_tfrecord_text_scalar_is_decoded_without_array_encode_bug(self):
        self.assertEqual(text_value(b"hello"), "hello")
        self.assertEqual(text_value([b"hello"]), "hello")


if __name__ == "__main__":
    unittest.main()
