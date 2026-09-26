# Lightweight Beauty proxy probe

## Locked method

This is a CPU proxy experiment, not EnsRec training or a paper reproduction. Train sequences alone fit an item-to-next-item transition table and item popularity. For a query, average transition probabilities from sources in its latest 19 visible events, then add a fixed 0.1 train-popularity backoff (use popularity alone if no source has outgoing transitions). Fit scikit-learn TF-IDF on catalog descriptions only; score a query by the normalized sum of TF-IDF vectors for its visible history against all 12,101 candidates. Map raw sequence ID `i` to metadata item ID `i+1`.

ID and TF-IDF scores use different scales, so rank each query's entire candidate set into deterministic percentiles (higher is better; item ID breaks score ties), then fuse as `alpha * ID + (1-alpha) * Text`. Equal fusion is alpha 0.5. Validation alone selects a median-length cutoff and an ID-weight alpha from 0.0 through 1.0 in 0.1 steps, maximizing Recall@10; ties favor alpha nearest 0.5, then larger ID weight. This rank-percentile proxy is distinct from the EnsRec paper's cosine ensemble. For each validation/test record, the target is its final event and the input is `sequence[-20:-1]`.

The fixed-seed negative control permutes candidate-side text vectors against item IDs while keeping user history text profiles intact. It only checks whether correct text-to-item alignment matters; it does not select the fusion weight. Test is never used for tuning.

## Beauty run

The full-data CPU run used Python 3.13.3, scikit-learn 1.9.1, NumPy 2.4.4, and SciPy 1.18.1. Because the available TFRecord reader is in the separate Python 3.11 environment, records were staged as a temporary `/private/tmp` JSON input for the run and that raw text/sequence file was removed afterward. Validation selected a visible-history cutoff of 4 events (11,383 short / 10,980 long) and ID alpha 0.9. On all 22,363 test users, ID-only Recall@10 was 5.11%, Text-only 4.13%, equal fusion 5.20% (+21 net hits vs ID), and validation-selected fusion 5.87% (+169 net hits). Equal fusion added/lost 211/154 hits for short histories and 325/361 for long histories. Of 666 test targets hit by Text@10 but missed by ID@10, equal fusion retained 168 (25.2%): 69/296 short and 99/370 long.

Among ID top-10 misses, Text top-10 hit 296/6,776 short users (4.37%) and 370/14,444 long users (2.56%): short minus long is +1.81 percentage points (user-bootstrap 95% CI +1.25 to +2.35 pp; 2,000 replicates). With candidate text misaligned, Text-only Recall@10 fell to 0.085% and equal-fusion Recall@10 to 0.729%.

These are proxy results for the audited Beauty next-item split and fixed candidate inventory. They do not establish EnsRec model performance or generalize beyond this probe.
