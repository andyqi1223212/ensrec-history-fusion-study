# Text-only negative-sampling ablation: configuration audit

Checked against the upstream EnsRec files in this study's local checkout on 2026-09-24.

Both `configs/experiment/text_only/ablate_negatives/train_beauty.yaml` and `train_steam.yaml` originally overrode `full_batch_id_negatives: false` and `id_loss_function: InBatchContrastiveLoss`. The equivalent ID-only ablation overrides those same ID fields, which matches its active branch.

The Text-only model configuration sets `id_loss_weight: 0`, `text_loss_weight: 1`, `full_batch_text_negatives: true`, and `text_loss_function: FullBatchCrossEntropyLoss`. In `IdTextModule.model_step`, the Text branch chooses candidate embeddings with `full_batch_text_negatives` and calls `text_loss_function`. The original Text ablation overrides therefore do not switch the Text training loss to in-batch negatives. The two intended overrides, if the goal is a Text in-batch-negative ablation, are `full_batch_text_negatives: false` and `text_loss_function: InBatchContrastiveLoss`; the local correction is in [`text-negative-ablation.patch`](../patches/text-negative-ablation.patch).

This is a static conclusion about the checked-in configuration and control flow. A complete Hydra configuration dump and a training-step observation of the Text key-embedding shape are still needed to confirm runtime behavior. It does not establish which configuration produced the paper's reported numbers.
