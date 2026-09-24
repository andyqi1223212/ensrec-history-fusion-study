# History length and ID/Text complementarity in EnsRec

An independent, reproducible study of a question about [EnsRec](https://github.com/snap-research/EnsRec): does item text recover more ID-only misses for short interaction histories, and can a history-aware fusion rule convert that complementarity into better recommendations?

**Status:** offline analysis and export integration are implemented, and a raw Beauty item-ID audit now exposes an unresolved preprocessing/embedding-generation boundary. ID/Text training and real fusion metrics have **not** been run, so this repository makes no improvement claim.

## Research design

For the same next-item events and candidate catalog, compare ID-only, Text-only, the paper's equal-weight normalized concatenation, a single validation-selected weight, and validation-selected weights for short and long histories. Report Recall@10, NDCG@10, Hit@1, Text-only hits, hits gained and lost against ID-only, and per-event outcomes. Select the history threshold and fusion weights on validation data; use test labels only for final evaluation. See [the protocol](docs/protocol.md).

## Repository contents

| Path | Purpose |
| --- | --- |
| `scripts/fusion_history_experiment.py` | Checks event alignment, selects fusion weights on validation, writes `report.json` and `test_events.csv`. |
| `scripts/fusion_history_data_audit.py` | Counts the history lengths visible to the Beauty collator from raw TFRecords. |
| `scripts/fusion_item_id_audit.py` | Audits item-catalog IDs, sequence IDs, and first-item text matches at raw ID offsets. |
| `scripts/frozen_text_embeddings.py` | Builds and validates a manifest-backed frozen Text table; fake mode needs no model download. |
| `docs/integration-audit.md` | Records mapping evidence, notebook failures, checkpoint selection, Text loss, and runtime limits. |
| `patches/ensrec-aligned-export.patch` | Adds validation/test event IDs and history lengths to an EnsRec checkout, using the selected checkpoint. |
| `patches/text-negative-ablation.patch` | Corrects two Text-only negative-sampling experiment configs in a local EnsRec checkout. |
| `docs/config-audit.md` | Traces that configuration issue to the active Text loss branch. |
| `results/beauty-history-lengths.json` | A real input-data audit, not a model result. |
| `results/beauty-item-id-mapping.json` | A real raw-data mapping audit; it does not verify any generated embedding file. |

This repository contains the study's scripts, tests, aggregate audit, and integration patches. It does not contain the upstream training code, downloaded data, checkpoints, text embeddings, or private learning notes. Candidate matrices are exported by split, and the evaluator checks each route's val/test matrices match. The original `item_embeddings.pt` test export remains for notebook compatibility. Cross-route row semantics are not certified until the generated table passes its manifest gate. `scripts/frozen_text_embeddings.py` is the research-side replacement for the broken notebook data/indexing path: it sorts explicit metadata IDs, writes two zero placeholders, places item `m` at row `m+1`, and emits only row IDs and SHA-256 text hashes. Its `fake` encoder mode supports download-free checks; it cannot produce research embeddings. The real mode uses the notebook's SentenceT5-XXL model. Validation hashes all current catalog texts and checks every raw event's first-item text against the corresponding manifest hash. The event evidence covers 7,563 unique first-item IDs per split; the catalog hash and explicit input order contract cover the remaining rows. Generated tensors/manifests and raw data must stay out of Git. The dynamic candidate path separately applies a +2 transform to item IDs before scattering Text vectors; it is not the frozen-table path used by this study. See the [integration audit](docs/integration-audit.md). The upstream code is MIT licensed; follow its license when applying patches.

## Reproduce the current checks

Python 3.10+ is required. Install this repository's offline analysis dependencies with `pip install -r requirements.txt`.

```bash
python -m unittest discover -s tests -v
python scripts/fusion_history_data_audit.py --data-dir /path/to/EnsRec/data/beauty \
  --output results/beauty-history-lengths.json
python scripts/fusion_item_id_audit.py --data-dir /path/to/EnsRec/data/beauty \
  --output results/beauty-item-id-mapping.json
python scripts/frozen_text_embeddings.py build --data-dir /path/to/EnsRec/data/beauty \
  --output /tmp/beauty-fake.pt --manifest /tmp/beauty-fake-manifest.json \
  --encoder fake --dimension 8
python scripts/frozen_text_embeddings.py validate --data-dir /path/to/EnsRec/data/beauty \
  --embeddings /tmp/beauty-fake.pt --manifest /tmp/beauty-fake-manifest.json
```

Do not interpret fusion scores until the Text embedding rows are linked to sequence/model IDs and the embedding tensor has been checked. The research-side builder now implements the candidate mapping: sort `(metadata_id, text)` pairs, encode them in that order, prepend two zero placeholder vectors, and place metadata ID `m` at row `m+1`, matching model ID `sequence_id+2` under the audited relation. It writes one `(row, metadata_id, source_text_hash)` manifest entry per item and validates shape, dimension, finiteness, placeholders, complete ID coverage, and row order. The download-free fake mode has been exercised against all Beauty records; real SentenceT5-XXL vector generation has not run. The upstream notebook writes `sentence-t5-xxl_item_embeddings.pt`, while the Text config expects `sentence-t5-xxl_embeddings.pt`; the documented real command uses the config's filename.

```bash
python scripts/fusion_history_experiment.py \
  --id-dir /path/to/EnsRec/outputs/beauty/embeddings/id_only/seed42 \
  --text-dir /path/to/EnsRec/outputs/beauty/embeddings/text_only/seed42 \
  --output-dir outputs/beauty-seed42
```

The script stops if IDs, labels, or lengths differ between routes. Keep model checkpoint paths, configs, seed, environment versions, and logs with any reported result. Do not select a weight after looking at the test scores.

## Evidence so far

The Beauty validation and test splits each contain 22,363 events. With the upstream collator's `sequence_length=20`, a prediction sees at most 19 history items. Validation histories range from 3 to 19; test histories range from 4 to 19. The count at 19 includes truncated histories. The item catalog has 12,101 unique IDs 1–12,101; sequence IDs span 0–12,100. In each split, every row's first sequence-item text matches catalog ID `sequence_id + 1` (22,363/22,363), but those first items cover only 7,563 unique IDs (0–12,095), not all catalog items. The same numeric catalog ID matches 0/22,363. The remaining items require a generation manifest and source-order contract to verify; this event-level check does not prove generated vector row order. The audit also reproduces two notebook failures in order: array `.encode()` raises `AttributeError`; extracting a scalar then reveals `KeyError(0)`. These data checks do not support or refute the fusion hypothesis.

## Attribution

This study builds on the [EnsRec paper and reference implementation](https://github.com/snap-research/EnsRec). The project also records an independently checked configuration discrepancy in the Text-only negative-sampling ablation. That finding concerns the checked-in configuration and code path; the training run that produced the paper's reported numbers has not been audited here.
