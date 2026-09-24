# History length and ID/Text complementarity in EnsRec

An independent, reproducible study of a question about [EnsRec](https://github.com/snap-research/EnsRec): does item text recover more ID-only misses for short interaction histories, and can a history-aware fusion rule convert that complementarity into better recommendations?

**Status:** analysis pipeline and export integration are implemented. The public Beauty data has been audited for history lengths. ID/Text training and real fusion metrics have **not** been run, so this repository makes no improvement claim.

## Research design

For the same next-item events and candidate catalog, compare ID-only, Text-only, the paper's equal-weight normalized concatenation, a single validation-selected weight, and validation-selected weights for short and long histories. Report Recall@10, NDCG@10, Hit@1, Text-only hits, hits gained and lost against ID-only, and per-event outcomes. Select the history threshold and fusion weights on validation data; use test labels only for final evaluation. See [the protocol](docs/protocol.md).

## Repository contents

| Path | Purpose |
| --- | --- |
| `scripts/fusion_history_experiment.py` | Checks event alignment, selects fusion weights on validation, writes `report.json` and `test_events.csv`. |
| `scripts/fusion_history_data_audit.py` | Counts the history lengths visible to the Beauty collator from raw TFRecords. |
| `patches/ensrec-aligned-export.patch` | Adds validation/test event IDs and history lengths to an EnsRec checkout, using the selected checkpoint. |
| `patches/text-negative-ablation.patch` | Corrects two Text-only negative-sampling experiment configs in a local EnsRec checkout. |
| `docs/config-audit.md` | Traces that configuration issue to the active Text loss branch. |
| `results/beauty-history-lengths.json` | A real input-data audit, not a model result. |

This repository contains the study's scripts, tests, aggregate audit, and integration patches. It does not contain the upstream training code, downloaded data, checkpoints, text embeddings, or private learning notes. Candidate tables are exported separately as `val_item_embeddings.pt` and `test_item_embeddings.pt`; the evaluator checks that each route's validation and test candidate matrices match. The original `item_embeddings.pt` test export is also retained for compatibility with the upstream notebook. ID and Text candidate rows are assumed to follow the upstream shared item-ID mapping; this export does not include a separate item-ID sidecar, so cross-route row semantics still depend on that mapping. The upstream code is MIT licensed; follow its license when applying the patches.

## Reproduce the current checks

Python 3.10+ is required. Install this repository's offline analysis dependencies with `pip install -r requirements.txt`.

```bash
python -m unittest discover -s tests -v
python scripts/fusion_history_data_audit.py --data-dir /path/to/EnsRec/data/beauty \
  --output results/beauty-history-lengths.json
```

To run model experiments, use a separate upstream EnsRec checkout with its full training environment and SentenceT5-XXL item embeddings. Apply `patches/ensrec-aligned-export.patch` there and train the ID and Text routes independently on one GPU with `export_validation_embeddings=true`. The upstream item embedding notebook writes `sentence-t5-xxl_item_embeddings.pt`, while the Text config expects `sentence-t5-xxl_embeddings.pt`; explicitly override `model.saved_embeddings_path` to the generated file. Once both routes have `val` and `test` exports:

```bash
python scripts/fusion_history_experiment.py \
  --id-dir /path/to/EnsRec/outputs/beauty/embeddings/id_only/seed42 \
  --text-dir /path/to/EnsRec/outputs/beauty/embeddings/text_only/seed42 \
  --output-dir outputs/beauty-seed42
```

The script stops if IDs, labels, or lengths differ between routes. Keep model checkpoint paths, configs, seed, environment versions, and logs with any reported result. Do not select a weight after looking at the test scores.

## Evidence so far

The Beauty validation and test splits each contain 22,363 events. With the upstream collator's `sequence_length=20`, a prediction sees at most 19 history items. Validation histories range from 3 to 19; test histories range from 4 to 19. The count at 19 includes truncated histories. These are properties of the input data only. They neither support nor refute the hypothesis about Text's marginal contribution.

## Attribution

This study builds on the [EnsRec paper and reference implementation](https://github.com/snap-research/EnsRec). The project also records an independently checked configuration discrepancy in the Text-only negative-sampling ablation. That finding concerns the checked-in configuration and code path; the training run that produced the paper's reported numbers has not been audited here.
