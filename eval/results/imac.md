# Evaluation: imac

Run 2026-10-15T16:19:50 · embeddings `local-bge-small-en-v1.5` · answers generated: False

| Variant | hit@k | mrr | context_recall | avg_latency_s |
|---|---|---|---|---|
| naive | 100% | 0.92 | 93% | 0.01 |
| advanced-no-rewrite | 100% | 0.87 | 100% | 14.65 |
| advanced-no-rewrite-no-rerank | 100% | 0.87 | 100% | 0.01 |
| advanced-no-rewrite-no-hybrid | 100% | 0.87 | 100% | 3.23 |

## Per question

| Question | naive | advanced-no-rewrite | advanced-no-rewrite-no-rerank | advanced-no-rewrite-no-hybrid |
|---|---|---|---|---|
| How much less energy does iMac use compared to the ENERGY STAR requirement? | hit ✓ · facts 100% | hit ✓ · facts 100% | hit ✓ · facts 100% | hit ✓ · facts 100% |
| What is the total carbon footprint of the iMac, and how much of that comes from product use versus manufacturing? | hit ✓ · facts 33% | hit ✓ · facts 100% | hit ✓ · facts 100% | hit ✓ · facts 100% |
| What material is the iMac stand made from, and what percentage of it is recycled? | hit ✓ · facts 100% | hit ✓ · facts 100% | hit ✓ · facts 100% | hit ✓ · facts 100% |
| What is the power consumption of iMac in sleep mode at 115V? | hit ✓ · facts 100% | hit ✓ · facts 100% | hit ✓ · facts 100% | hit ✓ · facts 100% |
| What share of the iMac packaging is fiber-based? | hit ✓ · facts 100% | hit ✓ · facts 100% | hit ✓ · facts 100% | hit ✓ · facts 100% |
| How much recycled content does iMac contain overall? | hit ✓ · facts 100% | hit ✓ · facts 100% | hit ✓ · facts 100% | hit ✓ · facts 100% |
| What share of the carbon footprint comes from transportation? | hit ✓ · facts 100% | hit ✓ · facts 100% | hit ✓ · facts 100% | hit ✓ · facts 100% |
| What recycled gold does iMac use and where? | hit ✓ · facts 100% | hit ✓ · facts 100% | hit ✓ · facts 100% | hit ✓ · facts 100% |
| What carbon footprint does the iMac four ports 512GB configuration have? | hit ✓ · facts 100% | hit ✓ · facts 100% | hit ✓ · facts 100% | hit ✓ · facts 100% |
| What is the retail price of the iMac? | n/a | n/a | n/a | n/a |
