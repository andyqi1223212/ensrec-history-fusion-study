# IdTextModule validation/test export probe

## Scope and source boundary

This is a static audit of the adjacent experiment source tree plus an environment probe. The clean public study repository at `d817879` does not contain `src/experimental/modules/id_text_module.py`, `src/train.py`, or the data-loading source. Those files are present in the neighboring reproduction workspace, whose Git tree was already dirty before this probe. No files in that neighboring tree were changed.

This note does not report model quality or recommend embeddings. Any values from a future smoke run must be labeled as random-initialization contract checks.

## What the source guarantees

- `IdTextModule.validation_step` and `test_step` collect user embeddings into dictionaries keyed by the input `user_id_list`, history lengths from `mask.sum(dim=1)`, and optional labels from `ItemData.item_ids`.
- Validation collection is enabled only while `trainer.validating` and `export_validation_embeddings` are both true. `on_validation_start` clears export buffers in that mode; `on_test_start` always clears them.
- `on_validation_epoch_end` writes the `val_*` files only in export mode. `on_test_epoch_end` writes the `test_*` files and the legacy `item_embeddings.pt` alias.
- During `eval_step`, the export collection path rejects duplicate user IDs within a batch and IDs already collected in an earlier batch. `_save_embedding_export` separately rejects multi-process export, empty user/candidate data, user-key sets that are not exactly contiguous `0..N-1`, and disagreement between exported user IDs and length/label keys. It writes user vectors, candidate vectors, history lengths, user IDs, and optional labels in sorted integer user-ID order. It does not check candidate row count or candidate ID coverage.
- `src/train.py` performs test first. A requested validation export then requires a best checkpoint and runs `trainer.validate` from that checkpoint.
- Beauty config places validation sequences in `data/beauty/evaluation` and test sequences in `data/beauty/testing`. The candidate dataloaders are separate stage configs named `generating_val_candidates` and `generating_test_candidates`.

## Candidate matrix limits

`get_candidate_embeddings` obtains the ID embedding table in item-ID row order. For dynamically encoded text/fused candidates, it allocates a matrix with the same row count and writes each batch at `label_data.item_ids` indices. Therefore dataloader row order does not define output row order; the item ID defines it.

The source does not assert candidate IDs are unique, cover every expected item ID, or that the final candidate row count equals a separately declared catalog size. Candidate completeness and row semantics still need an explicit fixture assertion or comparison to the source item-ID inventory. The legacy `item_embeddings.pt` alias is written only during test; val/test primary files have split-specific names.

## CPU probe outcome

No real IdTextModule forward/export run was completed. The repository root Python has torch 2.6.0 but lacks Lightning. `/private/tmp/ensrec-lifecycle-venv` has Lightning 2.5.0, torchmetrics 1.0.3, and Hydra but lacks torch. Combining the two read-only site-package directories successfully imports those packages together. Importing the real module then stops at missing `torchrec`; a subsequent import path also requires `fastavro`. The merged probe environments inspected here additionally have no TensorFlow package, so the repository TFRecord iterator cannot be exercised in those environments; this does not establish that TensorFlow is absent elsewhere on the machine. No dependencies were installed and no package environment was modified.

Exact import attempt (run from the reproduction workspace root):

```bash
PYTHONPATH="$PWD/.venv/lib/python3.11/site-packages:$PWD" \
  /private/tmp/ensrec-lifecycle-venv/bin/python -c \
  'from src.experimental.modules.id_text_module import IdTextModule'
```

The failure was `ModuleNotFoundError: No module named 'torchrec'`. A tiny custom synthetic batch could bypass TFRecord parsing but would still require the missing source import dependencies and faithful tower construction; adding import shims would not constitute a valid real-module probe, so execution stopped here.

## Minimal GPU smoke after the source environment is available

Run from the reproduction workspace root with its documented dependencies and Beauty data installed. This command scans the complete Beauty test split; it is a full export contract check, not a short CPU smoke. The model uses current/random initialization because training is disabled; inspect only file contracts, never retrieval scores. A one-batch limit was not added because the first batch user IDs have not been shown to include a complete contiguous `0..N-1` set, which the export hook requires.

```bash
export PROBE_OUT=/private/tmp/ensrec-id-text-export-smoke
mkdir -p "$PROBE_OUT"
python src/train.py \
  experiment=id_only/train_beauty \
  trainer=gpu \
  train=false test=true \
  model.embeddings_save_path="$PROBE_OUT" \
  model.save_user_labels=true \
  export_validation_embeddings=false \
  trainer.devices=1
```

For a true validation export, first produce a best checkpoint and then run the project's normal validation-export path with `export_validation_embeddings=true`; `src/train.py` intentionally rejects validation export when no best checkpoint exists. Do not infer source behavior by using a random test run to satisfy that checkpoint gate.

Contract checks for the smoke artifact:

1. Require test user vectors, item vectors, history lengths, user IDs, and user labels; require all first dimensions to agree with `N_users` and `test_user_ids == arange(N_users)`.
2. Check each saved label against the fixture/source test sequence's held-out target for that user ID. Check each saved history length against the count of non-padding events in that user's input sequence.
3. Require `test_item_embeddings.shape[0]` to match the expected candidate inventory size. Compare each output row with the model's candidate embedding at that item ID; do not infer row semantics from dataloader visitation order.
4. For a validation export, require the corresponding `val_*` files, verify them against evaluation fixtures, and confirm test files remain byte-identical. Check that the validation export does not consume or overwrite `item_embeddings.pt`.
5. Record that one process and one device are required by `_save_embedding_export`.
