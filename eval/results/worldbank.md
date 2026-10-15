# Evaluation: worldbank

Run 2026-10-15T16:18:58 · embeddings `local-bge-small-en-v1.5` · answers generated: False

| Variant | hit@k | mrr | context_recall | avg_latency_s |
|---|---|---|---|---|
| naive | 89% | 0.68 | 89% | 0.01 |
| advanced-no-rewrite | 100% | 0.94 | 100% | 2.39 |
| advanced-no-rewrite-no-rerank | 100% | 0.85 | 100% | 0.14 |
| advanced-no-rewrite-no-hybrid | 100% | 0.94 | 100% | 2.64 |

## Per question

| Question | naive | advanced-no-rewrite | advanced-no-rewrite-no-rerank | advanced-no-rewrite-no-hybrid |
|---|---|---|---|---|
| What is the growth outlook for South Asia in this report? | hit ✓ · facts 100% | hit ✓ · facts 100% | hit ✓ · facts 100% | hit ✓ · facts 100% |
| What does Chapter 3 say about sovereign debt in developing economies? | hit ✓ · facts 100% | hit ✓ · facts 100% | hit ✓ · facts 100% | hit ✓ · facts 100% |
| According to the executive summary, what are the main risks to the global outlook? | hit ✗ · facts 100% | hit ✓ · facts 100% | hit ✓ · facts 100% | hit ✓ · facts 100% |
| What is global growth projected to be in 2026? | hit ✓ · facts 100% | hit ✓ · facts 100% | hit ✓ · facts 100% | hit ✓ · facts 100% |
| What is India's growth forecast for fiscal year 2026/27? | hit ✓ · facts 100% | hit ✓ · facts 100% | hit ✓ · facts 100% | hit ✓ · facts 100% |
| What growth is projected for East Asia and Pacific in 2026? | hit ✓ · facts 100% | hit ✓ · facts 100% | hit ✓ · facts 100% | hit ✓ · facts 100% |
| What is the 2026 growth forecast for Sub-Saharan Africa? | hit ✓ · facts 0% | hit ✓ · facts 100% | hit ✓ · facts 100% | hit ✓ · facts 100% |
| By how much are sovereign spreads associated with the rise in EMDE debt since 2010? | hit ✓ · facts 100% | hit ✓ · facts 100% | hit ✓ · facts 100% | hit ✓ · facts 100% |
| SAR growth 2026 | hit ✓ · facts 100% | hit ✓ · facts 100% | hit ✓ · facts 100% | hit ✓ · facts 100% |
| What is the population of Tokyo? | n/a | n/a | n/a | n/a |
