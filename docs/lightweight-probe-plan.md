# Lightweight hypothesis-to-test plan

**Hypothesis.** Text similarity rescues more next-item targets missed by an ID-only behavior score for short histories than for long histories; adding text to ID scores improves held-out recall.

**Proxy experiment, not EnsRec training.** Learn an ID transition score from training sequences only. Build catalog text TF-IDF features and score a user's visible history against candidate item text. Since ID and TF-IDF scores have different scales, convert each query's candidate scores to deterministic rank percentiles (higher is better; break ties by item ID), then average percentiles for equal fusion. Use validation only to choose the short/long history cutoff and a nonnegative fusion weight on these percentiles; freeze both before testing. This rank-percentile proxy is distinct from the EnsRec paper's cosine ensemble.

On test, define ID misses as targets outside ID top-10. Report Text top-10 hit rate among those ID misses separately for short and long history cohorts, and the short-minus-long difference. Select the history cutoff using validation lengths only. Also report equal-weight fusion's net Recall@10 gain over ID-only and include ID-only, Text-only, validation-weighted fusion, and a shuffled-history-text control. Use the audited split prefixes and one last-event target per user. Do not use test results to tune any choice.
