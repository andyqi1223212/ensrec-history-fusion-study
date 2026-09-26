# Can item text help when interaction histories are short?

## Problem

In next-item recommendation, users with little history may have fewer useful ID transitions. This case study asks whether catalog text can recover some of the targets an interaction-only score misses, and whether a simple fusion preserves those text-only wins. It uses the Beauty data and audited item-ID mapping from [EnsRec](https://github.com/snap-research/EnsRec).

## Test

For each user, predict the final item in the validation and test sequence from the preceding events, capped at 19 visible history items. Fit ID transitions and popularity on training; fit TF-IDF on catalog text. Validation selects a median history-length cutoff and one global ID weight from a fixed grid; test is used once for reporting. The main comparison is Text top-10 hit rate among ID top-10 misses, short versus long histories. “Short” and “long” are cohorts split at the validation-selected cutoff.

This was a fixed analysis protocol, not a preregistered study. ID transition scores and TF-IDF similarities have different scales, so each is converted to a deterministic within-query candidate rank percentile before fusion. Equal fusion uses alpha 0.5; the validation-selected alpha is 0.9. This percentile fusion is a CPU proxy and differs from the EnsRec paper's cosine ensemble.

## Finding

![Beauty CPU proxy summary: text rescue by history length and fusion outcomes](results/beauty-cpu-proxy-summary.svg)

| Test result | Short histories | Long histories | All test users |
| --- | ---: | ---: | ---: |
| Text top-10 hit rate, conditioned on ID miss | 4.37% (296/6,776) | 2.56% (370/14,444) | — |
| Equal fusion added / lost / net hits vs ID | +211 / −154 / +57 | +325 / −361 / −36 | +536 / −515 / +21 |
| Text-only hits recovered by equal fusion | 69/296 (23.3%) | 99/370 (26.8%) | 168/666 (25.2%) |
| Validation-selected fusion net hits vs ID | — | — | +169 (alpha 0.9) |

Short minus long Text rescue was **+1.81 percentage points** (user-bootstrap 95% CI: **+1.25 to +2.35 pp**, 2,000 replicates). The measured complementarity is larger for the short-history cohort in this Beauty test split. Equal fusion's net change is only **+21 hits**: it recovers 168 of 666 Text-only wins, while losing 515 ID hits. This small aggregate difference does not establish a reliable uplift.

In the fixed-seed control, shuffling candidate-side text vectors reduced Text-only Recall@10 to **0.085%**, close to the uniform-candidate rate of **0.083%** (10/12,101). This is a sanity check that the observed text matches depend on item-text alignment, not evidence of generalization.

## What it means

This result supports a narrow descriptive claim: under this proxy scoring setup and this split, Text@10 more often rescues an ID miss for users in the short-history cohort. It also shows that a single equal rank-percentile blend does not reliably convert those wins into net gains for every cohort. It does not show that training EnsRec, using its learned representations, or deploying text will produce the same effect.

## Limitations

- This is one CPU proxy on one dataset, with deterministic sparse transition and catalog TF-IDF scores; it is not EnsRec model training or a reproduction of the paper's cosine fusion.
- Results are descriptive, not causal. The cohorts use visible history length, and validation chooses the cutoff and global alpha.
- The split contains adjacent sequences from the same users: training is a strict prefix of validation, and validation is a strict prefix of test. Each evaluation label is the last item of that split's sequence.
- Histories are truncated to the latest 19 events. The fixed candidate set has 12,101 items, with sequence ID `i` mapped to metadata ID `i+1`.
- A single deterministic scoring recipe and one Beauty test do not establish robustness, generalization, or online impact. Bootstrap intervals quantify user sampling uncertainty for this split, not model or dataset uncertainty.

## Run the CPU proxy

Python 3.10+ is required; no PyTorch or GPU is used by this proxy. Install its small dependency set and point it at the upstream Beauty data directory, containing `items/`, `training/`, `evaluation/`, and `testing/` TFRecords:

```bash
python -m pip install -r requirements-proxy.txt
python scripts/beauty_cpu_proxy.py \
  --data-dir /path/to/EnsRec/data/beauty \
  --output-dir outputs/beauty-cpu-proxy
```

The command writes aggregate JSON and SVG outputs. Keep source data, raw user sequences, text, per-user predictions, and generated vectors out of Git; this repository's `.gitignore` excludes common local data and output artifacts.

## Technical appendix

Earlier implementation evidence remains available for inspection:

- [Aggregated Beauty split-prefix and history-length audit](results/beauty-sequence-relationship-audit.json) ([audit script](scripts/beauty_sequence_relationship_audit.py))
- [ID/Text export hook and lifecycle probe](docs/id-text-export-probe.md)
- [Integration and embedding-row audit](docs/integration-audit.md)
- [Configuration audit](docs/config-audit.md)
- [Static protocol](docs/protocol.md)
- [EnsRec-aligned export patch](patches/ensrec-aligned-export.patch) and [Text negative-ablation patch](patches/text-negative-ablation.patch)

The experiment implementation is [scripts/beauty_cpu_proxy.py](scripts/beauty_cpu_proxy.py); aggregate machine-readable results are in [results/beauty-cpu-proxy-results.json](results/beauty-cpu-proxy-results.json). No raw Beauty records or user-level rows are committed.
