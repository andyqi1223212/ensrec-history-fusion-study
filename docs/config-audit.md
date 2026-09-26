# Text-only 负样本消融配置审计

本审计于 2026-09-24 对照本研究本地 checkout 中的 EnsRec 文件完成。

原 `configs/experiment/text_only/ablate_negatives/train_beauty.yaml` 和 `train_steam.yaml` 都设置了 `full_batch_id_negatives: false` 与 `id_loss_function: InBatchContrastiveLoss`。对应的 ID-only 消融也覆盖了这两个 ID 字段，与其当前生效的分支一致。

Text-only 模型配置设置 `id_loss_weight: 0`、`text_loss_weight: 1`、`full_batch_text_negatives: true` 和 `text_loss_function: FullBatchCrossEntropyLoss`。在 `IdTextModule.model_step` 中，Text 分支根据 `full_batch_text_negatives` 选择候选 embedding，并调用 `text_loss_function`。因此，原来的 Text 消融覆盖项不会把 Text 训练损失切换为 in-batch negatives。若目标是 Text in-batch 负样本消融，应改 Text 字段：`full_batch_text_negatives: false` 和 `text_loss_function: InBatchContrastiveLoss`。本地修正建议见 [`text-negative-ablation.patch`](../patches/text-negative-ablation.patch)。

后续已通过实际 Hydra compose 核对上述关键字段，记录见[集成检查](pretraining-integration-check.md)和[字段结果](../results/beauty-hydra-composed-fields.json)。完整 cfg job 仍受 resolver 限制；训练步骤中的 Text key embedding 形状和实际 loss 行为尚未运行验证。本审计仍无法确定论文报告实际使用了哪份配置。
