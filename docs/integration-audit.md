# Integration audit

Status: static code and raw-data checks on 2026-09-24. No EnsRec training step, Lightning hook, checkpoint callback, or real ID/Text recommendation score has run in this environment.

## Beauty item mapping evidence

Reproduce from an upstream checkout with the downloaded Beauty TFRecords and this study repository:

```bash
python scripts/fusion_item_id_audit.py \
  --data-dir /path/to/EnsRec/data/beauty \
  --output results/beauty-item-id-mapping.json
```

The audit fails unless catalog IDs uniquely cover `1..N`, observed sequence IDs cover `0..N-1`, and every event's first-item text matches the catalog row at `sequence_id + 1`. On the checked files, the catalog has 12,101 unique IDs, 1–12,101, no duplicates or gaps; sequence IDs cover 0–12,100. For each of training, evaluation, and testing, the first-item text matches `items[sequence_data[0] + 1]` for all 22,363 events and matches the same numeric ID for 0 events. However, those events' first items cover only 7,563 unique sequence IDs, 0–12,095. Therefore the repeated event-level matches directly verify the raw semantic mapping for those 7,563 first-item identities only, not all 12,101 catalog items. The remaining items need a generation manifest and source-order contract; the raw-event check alone cannot verify their vector rows. In this TFRecord reader, `row['text']` is the first item's text, not a full text sequence.

The sequence feature uses `num_placeholder_tokens: 2`; `add_placeholder_tokens` adds two to raw sequence IDs. Thus labels occupy model IDs 2–12,102, within the 12,103-row ID table (0–12,102). The Beauty candidate dataset also configures two placeholders for `id`, and the generic preprocessing adds two to candidate IDs. That candidate transform is relevant to the dynamic Text path, which scatters vectors at `label_item_ids`. The frozen ID-only and Text-only routes used in this study instead return the ID table and precomputed Text table directly; neither uses candidate-loader IDs to reorder those tables.

## Frozen Text embedding notebook

`notebooks/item_text_embedding_gen.ipynb` reads `items.id` and `text`, converts sparse text rows with `tf.sparse.to_dense(...).numpy().astype(str)`, then cell 8 executes `text[i].encode('utf-8')`. The dense result is a NumPy array, so the first failure is `AttributeError: 'numpy.ndarray' object has no attribute 'encode'`. This first failure is reproduced by the audit script with the actual catalog text rows.

If scalar extraction is added at cell 8, the next code path builds `item_to_text` keyed by actual IDs 1–12,101. Cell 13 runs `[item_to_text[i] for i in range(len(item_to_text))]`, beginning with absent key 0; the second failure is `KeyError(0)`. The audit reproduces this path too. The cell numbers refer to the notebook's code-cell order; the expressions above identify the failure sites if notebook display numbering changes. No SentenceT5 vector file exists here, so no generated table's row semantics have been verified.

A minimal research-side repair candidate is to sort explicit `(metadata_id, text)` pairs by metadata ID, encode each text once, and prepend two zero vectors. The raw-data evidence suggests metadata ID `m` belongs at model row `m+1`, because the matching sequence ID is `m-1` and sequence preprocessing maps that to `m+1`. This is a proposal only; no notebook repair or row remapping has been applied.

Before training, generate a manifest with one `(row, metadata_id, text_hash)` record per real item. Gate the table on: IDs 1–12,101 each appearing exactly once; rows 2–12,102 following `metadata_id + 1`; rows 0 and 1 exactly matching the placeholder contract; expected shape (12,103 × 768 for the configured table); all finite values; and evaluation/test first-item text mapping passing for every event. Check the manifest and tensor together; tensor shape alone cannot prove row meaning.

## Checkpoint selection and export call chain

Beauty configs set `trainer.val_check_interval=2000`, `callbacks.model_checkpoint.every_n_train_steps=2001`, `save_top_k=1`, and `mode=max`. ID monitors `val/id_cosine/recall@10`; Text monitors `val/text_cosine/recall@10`. `src/train.py` calls `fit`, gets the first `trainer.checkpoint_callback.best_model_path`, warns and falls back to current weights if that path is empty, then passes the selected path to `trainer.test`. With validation export enabled, it rejects an empty path and calls `trainer.validate(..., ckpt_path=the_same_path)`; the module flag is reset in `finally`.

There are two separate claims to verify at runtime: (1) a best path exists; (2) the weights in that file correspond to the validation metric used to select it. Because checkpoint saves are scheduled 2001 steps apart while validation runs every 2000, a save can potentially consume the most recent validation metric while serializing weights from a later step. This static review does not establish Lightning hook order or whether the observed file has that drift. Lightning/Hydra/OmegaConf are unavailable here, so no callback run was possible.

In a complete environment, the first bounded ID-only integration run can stop at the first default save boundary:

```bash
python src/train.py experiment=id_only/train_beauty trainer=gpu seed=42 \
  trainer.max_steps=2001 trainer.val_check_interval=2000 \
  callbacks.model_checkpoint.every_n_train_steps=2001 \
  trainer.num_sanity_val_steps=0 export_validation_embeddings=true
```

Preserve the log showing the validation step/metric, checkpoint callback's best path and score, then inspect the saved checkpoint's `global_step`. Confirm `trainer.test` and the final `trainer.validate` both load that exact path. Repeat for Text after the embedding mapping gate passes. This is a short integration check, not a model result; the eventual full experiment must use the predeclared protocol.

## Text negative-sampling ablation

The Beauty Text-only ablation YAML sets `id_loss_weight=0`, `text_loss_weight=1`, `full_batch_text_negatives=false`, and `text_loss_function=InBatchContrastiveLoss(normalize=true, contrastive_tau=0.07)`. The Text-only base config defaults to full-batch negatives and `FullBatchCrossEntropyLoss`; the experiment override replaces the Text fields.

In `IdTextModule.model_step`, the Text branch tests `full_batch_text_negatives`: true selects the full `text_embeddings` table, false selects `item_text_embeddings`; it calls `text_loss_function` and scales by `text_loss_weight`. This statically confirms the intended active fields. PyYAML read the base/experiment values and showed the expected override. A CPU smoke instantiated the configured `InBatchContrastiveLoss`, ran a finite forward and backward pass on `[2, 8]` queries and `[6, 8]` keys; it did not instantiate `IdTextModule` or execute `model_step`. Hydra composition, actual key shapes, and training remain unverified.
