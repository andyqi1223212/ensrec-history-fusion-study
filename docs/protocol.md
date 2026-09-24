# Experiment protocol

## Question and prediction

Text-only may retrieve a next item that ID-only misses even when Text-only has lower overall Recall@10. The preregistered prediction is that this marginal help will be larger for short visible histories. A counterexample is equal or greater net help on long histories, or Text-only hits that equal-weight fusion fails to recover.

## Fair comparison

Train ID-only and Text-only independently only after establishing that their item-table rows represent the same items as the labels. Each route uses its own validation Recall@10 to select its checkpoint. Export vectors for the same validation and test events. Require identical event IDs, labels, and history lengths across routes, and require each route's validation/test candidate matrices to match. Beauty's raw data shows `items.id` in 1..12,101 and sequence IDs in 0..12,100. In each split, all 22,363 first-item texts match metadata ID `sequence_id + 1`, but the first items cover only 7,563 unique IDs (0–12,095). This does not verify the other catalog items; require a generation manifest plus source-order contract for full coverage. The frozen Text route projects a precomputed table directly. The author notebook first applies `.encode()` to an array; if fixed by extracting the scalar, its next cell indexes a 1-based dictionary from zero. Vector row semantics are unverified. Do not report fusion metrics until a per-row source-ID/text manifest verifies this mapping. The dynamic Text candidate path also adds two placeholders before using candidate IDs as matrix indices; it is a distinct path and must be checked separately.

The paper's normalized concatenation ranks items by `ID cosine + Text cosine`; this study writes it as `alpha * ID cosine + (1-alpha) * Text cosine`, with `alpha=0.5` giving the same ordering. Evaluate alpha on the grid 0.0, 0.1, …, 1.0. Choose the global alpha by validation Recall@10. Choose the short/long threshold from validation history lengths to balance group sizes, then choose one alpha per group using validation Recall@10. Break ties toward 0.5. Never route an event using whether either model actually hit its target.

Report ID, Text, equal fusion, global fusion, and history-aware fusion on the held-out test events. For each history cohort, also report equal fusion's added hits, lost hits, net hits, and the fraction of Text-only hits that equal fusion recovers. At one target per event, Recall@10 is Hit Rate@10. NDCG@10 describes rank within the top ten; Hit@1 is exploratory.

## Interpretation rules

Inspect the validation cohort diagnosis before interpreting history-aware weights. Compare against both equal and global weights, not just ID-only. Report cohort denominators because validation and test contain adjacent events for the same users and have different history distributions. The visible history is capped at 19; the capped cohort conflates exact length 19 with longer histories. Check nearby weights and additional seeds before claiming a stable effect. Compare any improvement with the cost of computing and scoring both routes. A negative result is still a result; do not tune on test to rescue the hypothesis.

## Current evidence boundary

The evaluation implementation and synthetic tests are available, and both Beauty history lengths and raw item-ID relations have been audited. No trained two-route embeddings or real recommendation outcomes have been evaluated. The Text vector row mapping is unresolved, so the model-score pipeline is not ready for interpretation.
