# Pre-training configuration and checkpoint lifecycle check

No EnsRec model, GPU run, or XXL text encoder was executed. This stage produced two bounded kinds of runtime evidence: Hydra-composed selected config fields and a synthetic CPU Lightning checkpoint lifecycle test.

## Beauty config composition

`scripts/compose_beauty_config_fields.py` uses Hydra 1.3.2 `compose()` with the project's custom resolvers and the same overrides used for training (`experiment=...`, `trainer=gpu`). Captured fields and source paths are in `results/beauty-hydra-composed-fields.json`. A cross-check against the source-aware YAML overlay in `results/beauty-training-config-audit.json` passed for all 14 overlapping fields in all three cases.

| Experiment | Effective loss fields | Checkpoint fields |
| --- | --- | --- |
| ID-only | `id_loss_weight=1`, `text_loss_weight=0`, `full_batch_id_negatives=true` | monitor `val/id_cosine/recall@10`, max/top 1, every 2001 steps; validate every 2000; `trainer=gpu` |
| Text-only | `id_loss_weight=0`, `text_loss_weight=1`, `full_batch_text_negatives=true`, `FullBatchCrossEntropyLoss(normalize=true, contrastive_tau=0.07)` | monitor `val/text_cosine/recall@10`, max/top 1, every 2001 steps; validate every 2000; `trainer=gpu` |
| Text-only negative ablation | `id_loss_weight=0`, `text_loss_weight=1`, `full_batch_text_negatives=false`, `InBatchContrastiveLoss(normalize=true, contrastive_tau=0.07)` | monitor `val/text_cosine/recall@10`, max/top 1, every 2001 steps; validate every 2000; `trainer=gpu` |

Per-field source files are in the composed JSON. The model defaults come from `configs/model/id_only.yaml` or `configs/model/text_only.yaml`; experiment overrides and monitor/cadence values come from the corresponding `configs/experiment/.../train_beauty.yaml`; `save_top_k=1` comes from `configs/callbacks/model_checkpoint.yaml`.

The ordinary `src/train.py ... --cfg job --resolve` command did not serialize the whole config in this no-run mode: the project's `replace_prefix` resolver received `None` from `hydra:runtime.output_dir` and raised `AttributeError`. Selected fields were instead composed directly with Hydra's API and resolved successfully. This is not a full training launch or an interpolation dump of every field.

## Synthetic Lightning lifecycle results

The reproducible synthetic model is `scripts/lightning_checkpoint_lifecycle_smoke.py`; the captured matrix is `results/lightning-checkpoint-lifecycle-smoke.json`. The JSON is an assembled summary of three separate CLI runs, not output from one invocation. The model has one scalar parameter, 2,001 one-sample CPU batches, a validation metric equal to the current global step, validation every 2,000 steps, and a maximum of 2,001 steps. It is a framework test, not an EnsRec run.

| Callback schedule | Validation result | Checkpoint result |
| --- | --- | --- |
| Source-like default: `every_n_train_steps=2001`, `save_on_train_epoch_end=null` | step 2000, score 2000.0 | best path saved at step 2001 with serialized `global_step=2001` and `best_model_score=2000.0`; test and final validate loaded that same path and step |
| Candidate A: `every_n_train_steps=2000`, other defaults | step 2000, score 2000.0 | no best path; at step 2000 the train-batch checkpoint hook warned: `ModelCheckpoint(monitor='val/probe_score') could not find the monitored key in the returned metrics: ['epoch', 'step']` |
| Candidate B: `every_n_train_steps=null`, `save_on_train_epoch_end=false` | step 2000, score 2000.0 | saved at step 2000 with `global_step=2000` and score 2000.0; test and final validate loaded that same path and step |

Candidate B is supported as a **study-side** checkpoint alignment override for both ID and Text runs:

```bash
callbacks.model_checkpoint.every_n_train_steps=null \
callbacks.model_checkpoint.save_on_train_epoch_end=false
```

Use these overrides on both the ID-only and Text-only study invocations (and the Text negative-ablation run if included). This leaves the author's reproduction configs and cadence untouched. Before using the override for reported EnsRec results, run a bounded real GPU smoke and confirm the callback path, metric, and serialized `global_step` in that environment.

Explicit main-route examples:

```bash
python src/train.py experiment=id_only/train_beauty trainer=gpu \
  callbacks.model_checkpoint.every_n_train_steps=null \
  callbacks.model_checkpoint.save_on_train_epoch_end=false
python src/train.py experiment=text_only/train_beauty trainer=gpu \
  callbacks.model_checkpoint.every_n_train_steps=null \
  callbacks.model_checkpoint.save_on_train_epoch_end=false
```

To reproduce the synthetic matrix in an environment with the full project's Lightning dependencies:

```bash
python scripts/lightning_checkpoint_lifecycle_smoke.py --output-dir /tmp/ensrec-smoke-default
python scripts/lightning_checkpoint_lifecycle_smoke.py --every-n-train-steps 2000 \
  --expect-no-checkpoint --output-dir /tmp/ensrec-smoke-2000
python scripts/lightning_checkpoint_lifecycle_smoke.py --save-on-validation-end \
  --output-dir /tmp/ensrec-smoke-validation-end
```

## Environment and remaining runtime boundary

The current Python 3.11.4 environment initially lacked Hydra/Lightning. An isolated temporary venv was populated from official PyPI; the first Tsinghua mirror attempt failed DNS, but official PyPI installation succeeded. The smoke ran with PyTorch 2.6.0, Hydra 1.3.2, OmegaConf 2.3.0, Lightning 2.5.0, PyTorch Lightning 2.5.3, Lightning Fabric 2.5.3, and torchmetrics 1.0.3. The existing PyTorch installation was exposed read-only to the temporary venv; no packages were installed into the user's existing venv. After aligning fsspec/packaging with Lightning's constraints, `pip check` reported no broken requirements.

This host has no CUDA, MPS, or `nvidia-smi`. Static source inspection of `src/train.py` shows the same selected `best_model_path` variable is passed to `trainer.test` and optional final `trainer.validate`; the synthetic smoke confirms Lightning loads a supplied path in both phases. The real EnsRec callback ordering, full datamodule/model behavior, and GPU checkpoint state remain untested.

GPU preflight in the full EnsRec checkout:

```bash
cd /path/to/EnsRec
python -m pip check
python - <<'PY'
import torch, hydra, omegaconf, lightning, pytorch_lightning, torchmetrics
print("torch", torch.__version__, "CUDA", torch.cuda.is_available())
print("hydra", hydra.__version__, "OmegaConf", omegaconf.__version__)
print("lightning", lightning.__version__, "pytorch_lightning", pytorch_lightning.__version__)
print("torchmetrics", torchmetrics.__version__)
assert torch.cuda.is_available(), "CUDA is required for the GPU run"
PY
nvidia-smi
```

Compose selected configs with Hydra without running EnsRec training:

```bash
python /path/to/ensrec-history-fusion-study/scripts/compose_beauty_config_fields.py \
  --ensrec-root /path/to/EnsRec
```
