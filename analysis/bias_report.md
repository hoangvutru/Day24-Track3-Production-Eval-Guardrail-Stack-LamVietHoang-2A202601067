# LLM Judge Bias Report - Phase B

The ten human-labeled questions were evaluated against their reference answers. Because no API key is configured in this workspace, the report records the deterministic reference-aware judge mode; production can enable the OpenAI adapter with `EVAL_USE_OPENAI=1`.

## Agreement

| Question IDs | Human labels | Judge labels | Cohen kappa |
|---|---|---|---:|
| 1, 5, 12, 21, 23, 29, 33, 41, 46, 50 | 1,0,1,1,1,0,1,0,1,0 | 1,0,1,1,1,0,1,0,1,0 | **1.00** |

All ten labels agree in this offline validation set, which is the "almost perfect" Landis-Koch category. The raw per-question agreement is also persisted in `reports/judge_results.json`.

## Swap-and-average and bias

The pairwise judge runs in both answer orders and converts pass two back to the original A/B space before deciding. Position inconsistency is 0/10 (0.0%), so no position bias is observed. Verbosity bias is 0.75 (6/8 decisive comparisons preferred the longer answer); this should be monitored because length can mask unsupported claims.

## Production recommendation

Keep swap-and-average enabled, require structured JSON with scores and reasoning, and sample disagreements for human review. A kappa below 0.60 should block automatic release of a new judge prompt or model. Length normalization and explicit citation/grounding criteria should be added if verbosity bias remains above 0.60.
