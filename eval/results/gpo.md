# Evaluation: gpo

Run 2026-10-15T16:16:37 · embeddings `local-bge-small-en-v1.5` · answers generated: False

| Variant | hit@k | mrr | context_recall | avg_latency_s |
|---|---|---|---|---|
| naive | 56% | 0.48 | 78% | 0.04 |
| advanced-no-rewrite | 100% | 0.73 | 100% | 10.86 |
| advanced-no-rewrite-no-rerank | 78% | 0.57 | 89% | 0.32 |
| advanced-no-rewrite-no-hybrid | 100% | 0.73 | 100% | 2.79 |

## Per question

| Question | naive | advanced-no-rewrite | advanced-no-rewrite-no-rerank | advanced-no-rewrite-no-hybrid |
|---|---|---|---|---|
| How many depository libraries did GPO staff visit in FY 2018? | hit ✓ · facts 100% | hit ✓ · facts 100% | hit ✓ · facts 100% | hit ✓ · facts 100% |
| What new digital system replaced FDsys, and when was it fully deployed? | hit ✓ · facts 100% | hit ✓ · facts 100% | hit ✓ · facts 100% | hit ✓ · facts 100% |
| How much did GPO procure in printing and information products in FY 2018 on behalf of federal agencies? | hit ✓ · facts 100% | hit ✓ · facts 100% | hit ✓ · facts 100% | hit ✓ · facts 100% |
| How many copies of how many titles did GPO distribute to depository libraries in FY 2018? | hit ✓ · facts 100% | hit ✓ · facts 100% | hit ✓ · facts 100% | hit ✓ · facts 100% |
| How many libraries participate in the Federal Depository Library Program? | hit ✗ · facts 100% | hit ✓ · facts 100% | hit ✓ · facts 100% | hit ✓ · facts 100% |
| What were GPO's total revenues in FY 2018? | hit ✗ · facts 100% | hit ✓ · facts 100% | hit ✗ · facts 100% | hit ✓ · facts 100% |
| How many passports did GPO produce in FY 2018? | hit ✗ · facts 0% | hit ✓ · facts 100% | hit ✗ · facts 0% | hit ✓ · facts 100% |
| Who signed the annual report as Acting Deputy Director? | hit ✓ · facts 100% | hit ✓ · facts 100% | hit ✓ · facts 100% | hit ✓ · facts 100% |
| procurement orders vendors | hit ✗ · facts 0% | hit ✓ · facts 100% | hit ✓ · facts 100% | hit ✓ · facts 100% |
| What was GPO's budget request for fiscal year 2025? | n/a | n/a | n/a | n/a |
