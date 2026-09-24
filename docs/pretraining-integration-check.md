# Pre-training configuration and lifecycle check

Status: source-aware YAML audit completed; Hydra composition and Lightning lifecycle smoke blocked by missing dependencies and unavailable package-index DNS. No EnsRec model or XXL text encoder was run.

## Beauty configuration fields

The reproducible source-aware overlay is in `scripts/audit_beauty_config_sources.py`; its captured output is `results/beauty-training-config-audit.json`. It reads the three experiment YAMLs, overlays the selected `model/*.yaml` base for audited fields, and records the source file for each value. It lists Hydra `defaults`, but does not claim to execute Hydra composition or resolve interpolations.

| Experiment | ID/Text loss weights | Negatives and Text loss | Checkpoint / validation |
| --- | --- | --- | --- |
| ID-only | `1.0 / 0.0` | `full_batch_id_negatives=true` | `val/id_cosine/recall@10`, max, top 1; save every 2001 train steps; validate every 2000; trainer group `ddp` |
| Text-only | `0.0 / 1.0` | `full_batch_text_negatives=true`; `FullBatchCrossEntropyLoss(normalize=true, contrastive_tau=0.07)` | `val/text_cosine/recall@10`, max, top 1; save every 2001; validate every 2000; trainer group `ddp` |
| Text-only negative ablation | `0.0 / 1.0` | `full_batch_text_negatives=false`; `InBatchContrastiveLoss(normalize=true, contrastive_tau=0.07)` | `val/text_cosine/recall@10`, max, top 1; save every 2001; validate every 2000; trainer group `ddp` |

Field-level source file paths are recorded in the JSON. Model loss defaults come from `configs/model/id_only.yaml` or `configs/model/text_only.yaml`; the negative-ablation Text loss and `full_batch_text_negatives=false` come from `configs/experiment/text_only/ablate_negatives/train_beauty.yaml`; monitor, interval, and max-step values come from each experiment file; `save_top_k=1` comes from `configs/callbacks/model_checkpoint.yaml`.

The experiment defaults select the `ddp` trainer group. The original documented training invocation overrides this with `trainer=gpu`; the preflight commands preserve that override when asking Hydra to resolve the job config.

## Why the Lightning smoke did not run

The available isolated Python 3.11.4 environment has PyTorch 2.6.0 but lacks `lightning`, `pytorch_lightning`, `hydra`, `omegaconf`, and `torchmetrics`. CUDA and MPS both report unavailable; `nvidia-smi` is not installed. The repository's pinned requirements include Hydra 1.3.2, Lightning 2.5.0, PyTorch Lightning 2.5.3, and torchmetrics 1.0.3.

An install to the temporary target `/private/tmp/ensrec-integration-deps` was attempted. Pip could not resolve the configured Tsinghua PyPI mirror hostname (`[Errno 8] nodename nor servname provided`), so no packages were installed. Static source inspection of `src/train.py` shows that it reads the first checkpoint callback's `best_model_path` and passes the same `ckpt_path` variable to `trainer.test` and, when requested, final `trainer.validate`. A synthetic Trainer result would be misleading without the actual Lightning hooks, so no substitute simulation is reported. In particular, the real callback ordering at validation step 2000 and save step 2001, the checkpoint's serialized `global_step`, and whether both hook invocations actually load that path remain unverified.

## GPU environment preflight

Run these commands in the full EnsRec checkout after installing its locked requirements. The `--cfg job --resolve` invocations compose configuration without starting training:

```bash
cd /path/to/EnsRec
python -m pip check
python - <<'PY'
import torch
import hydra
import omegaconf
import lightning
import pytorch_lightning
import torchmetrics
print("torch", torch.__version__)
print("cuda_available", torch.cuda.is_available())
assert torch.cuda.is_available(), "CUDA is required by the configured GPU trainer"
print("hydra", hydra.__version__)
print("omegaconf", omegaconf.__version__)
print("lightning", lightning.__version__)
print("pytorch_lightning", pytorch_lightning.__version__)
print("torchmetrics", torchmetrics.__version__)
PY
nvidia-smi
python src/train.py experiment=id_only/train_beauty trainer=gpu --cfg job --resolve > /tmp/ensrec-id-only-resolved.yaml
python src/train.py experiment=text_only/train_beauty trainer=gpu --cfg job --resolve > /tmp/ensrec-text-only-resolved.yaml
python src/train.py experiment=text_only/ablate_negatives/train_beauty trainer=gpu --cfg job --resolve > /tmp/ensrec-text-ablation-resolved.yaml
```

After preflight, a tiny standalone synthetic Lightning Trainer should be run with `val_check_interval=2000`, `ModelCheckpoint(every_n_train_steps=2001, monitor='val/score', save_top_k=1, mode='max')`, and `max_steps=2001`. Retain the validation step/score, callback `best_model_path`/score, checkpoint `global_step`, and explicit `ckpt_path` supplied to both test and final validate. This verifies framework lifecycle only; it is not an EnsRec run and cannot substitute for a later real GPU smoke.
