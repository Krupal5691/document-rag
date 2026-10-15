# Evaluation: worldbank

Run 2026-10-15T22:46:42 · embeddings `local-bge-small-en-v1.5` · answers generated: False

| Variant | hit@k | mrr | context_recall | avg_latency_s |
|---|---|---|---|---|
| naive | 90% | 0.71 | 90% | 0.01 |
| advanced | 100% | 0.90 | 100% | 2.00 |
| advanced-no-rerank | 100% | 0.79 | 100% | 0.14 |
| advanced-no-hybrid | 100% | 0.90 | 100% | 1.95 |
| advanced-no-parent | 100% | 0.88 | 100% | 2.11 |
| advanced-no-rewrite-no-filters | 100% | 0.88 | 100% | 1.79 |

## Per question

| Question | naive | advanced | advanced-no-rerank | advanced-no-hybrid | advanced-no-parent | advanced-no-rewrite-no-filters |
|---|---|---|---|---|---|---|
| What is the growth outlook for South Asia in this report? | hit ✓ · facts 100% | hit ✓ · facts 100% | hit ✓ · facts 100% | hit ✓ · facts 100% | hit ✓ · facts 100% | hit ✓ · facts 100% |
| What does Chapter 3 say about sovereign debt in developing economies? | hit ✓ · facts 100% | hit ✓ · facts 100% | hit ✓ · facts 100% | hit ✓ · facts 100% | hit ✓ · facts 100% | hit ✓ · facts 100% |
| According to the executive summary, what are the main risks to the global outlook? | hit ✗ · facts 100% | hit ✓ · facts 100% | hit ✓ · facts 100% | hit ✓ · facts 100% | hit ✓ · facts 100% | hit ✓ · facts 100% |
| What is global growth projected to be in 2026? | hit ✓ · facts 100% | hit ✓ · facts 100% | hit ✓ · facts 100% | hit ✓ · facts 100% | hit ✓ · facts 100% | hit ✓ · facts 100% |
| What is India's growth forecast for fiscal year 2026/27? | hit ✓ · facts 100% | hit ✓ · facts 100% | hit ✓ · facts 100% | hit ✓ · facts 100% | hit ✓ · facts 100% | hit ✓ · facts 100% |
| What growth is projected for East Asia and Pacific in 2026? | hit ✓ · facts 100% | hit ✓ · facts 100% | hit ✓ · facts 100% | hit ✓ · facts 100% | hit ✓ · facts 100% | hit ✓ · facts 100% |
| What is the 2026 growth forecast for Sub-Saharan Africa? | hit ✓ · facts 0% | hit ✓ · facts 100% | hit ✓ · facts 100% | hit ✓ · facts 100% | hit ✓ · facts 100% | hit ✓ · facts 100% |
| By how much are sovereign spreads associated with the rise in EMDE debt since 2010? | hit ✓ · facts 100% | hit ✓ · facts 100% | hit ✓ · facts 100% | hit ✓ · facts 100% | hit ✓ · facts 100% | hit ✓ · facts 100% |
| SAR growth 2026 | hit ✓ · facts 100% | hit ✓ · facts 100% | hit ✓ · facts 100% | hit ✓ · facts 100% | hit ✓ · facts 100% | hit ✓ · facts 100% |
| What is Bangladesh's growth forecast for fiscal year 2026/27? | hit ✓ · facts 100% | hit ✓ · facts 100% | hit ✓ · facts 100% | hit ✓ · facts 100% | hit ✓ · facts 100% | hit ✓ · facts 100% |
| What is the population of Tokyo? | n/a | n/a | n/a | n/a | n/a | n/a |
