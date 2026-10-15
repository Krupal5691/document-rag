# Evaluation: imac

Run 2026-10-15T22:48:10 · embeddings `local-bge-small-en-v1.5` · answers generated: False

| Variant | hit@k | mrr | context_recall | avg_latency_s |
|---|---|---|---|---|
| naive | 100% | 0.93 | 93% | 0.01 |
| advanced | 100% | 1.00 | 100% | 1.61 |
| advanced-no-rerank | 100% | 1.00 | 100% | 0.02 |
| advanced-no-hybrid | 100% | 1.00 | 100% | 1.58 |
| advanced-no-parent | 100% | 1.00 | 100% | 1.59 |
| advanced-no-rewrite-no-filters | 100% | 0.93 | 100% | 2.37 |

## Per question

| Question | naive | advanced | advanced-no-rerank | advanced-no-hybrid | advanced-no-parent | advanced-no-rewrite-no-filters |
|---|---|---|---|---|---|---|
| How much less energy does iMac use compared to the ENERGY STAR requirement? | hit ✓ · facts 100% | hit ✓ · facts 100% | hit ✓ · facts 100% | hit ✓ · facts 100% | hit ✓ · facts 100% | hit ✓ · facts 100% |
| What is the total carbon footprint of the iMac, and how much of that comes from product use versus manufacturing? | hit ✓ · facts 33% | hit ✓ · facts 100% | hit ✓ · facts 100% | hit ✓ · facts 100% | hit ✓ · facts 100% | hit ✓ · facts 100% |
| What material is the iMac stand made from, and what percentage of it is recycled? | hit ✓ · facts 100% | hit ✓ · facts 100% | hit ✓ · facts 100% | hit ✓ · facts 100% | hit ✓ · facts 100% | hit ✓ · facts 100% |
| What is the power consumption of iMac in sleep mode at 115V? | hit ✓ · facts 100% | hit ✓ · facts 100% | hit ✓ · facts 100% | hit ✓ · facts 100% | hit ✓ · facts 100% | hit ✓ · facts 100% |
| What share of the iMac packaging is fiber-based? | hit ✓ · facts 100% | hit ✓ · facts 100% | hit ✓ · facts 100% | hit ✓ · facts 100% | hit ✓ · facts 100% | hit ✓ · facts 100% |
| How much recycled content does iMac contain overall? | hit ✓ · facts 100% | hit ✓ · facts 100% | hit ✓ · facts 100% | hit ✓ · facts 100% | hit ✓ · facts 100% | hit ✓ · facts 100% |
| What share of the carbon footprint comes from transportation? | hit ✓ · facts 100% | hit ✓ · facts 100% | hit ✓ · facts 100% | hit ✓ · facts 100% | hit ✓ · facts 100% | hit ✓ · facts 100% |
| What recycled gold does iMac use and where? | hit ✓ · facts 100% | hit ✓ · facts 100% | hit ✓ · facts 100% | hit ✓ · facts 100% | hit ✓ · facts 100% | hit ✓ · facts 100% |
| What carbon footprint does the iMac four ports 512GB configuration have? | hit ✓ · facts 100% | hit ✓ · facts 100% | hit ✓ · facts 100% | hit ✓ · facts 100% | hit ✓ · facts 100% | hit ✓ · facts 100% |
| Which power adapter was iMac tested with for its energy efficiency figures? | hit ✓ · facts 100% | hit ✓ · facts 100% | hit ✓ · facts 100% | hit ✓ · facts 100% | hit ✓ · facts 100% | hit ✓ · facts 100% |
| What is the retail price of the iMac? | n/a | n/a | n/a | n/a | n/a | n/a |
