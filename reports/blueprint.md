# CI/CD Blueprint: RAG Eval + Guardrail Stack

**Owner:** Lab 24 submission
**Evaluation mode:** deterministic offline fallback (OpenAI/NeMo can be enabled in production)

## Guard Stack Pipeline

| Layer | Tool | Latency P95 | Failure action |
|---|---|---:|---|
| PII detection | Presidio + VN regex recognizers | 0.06 ms | Reject with 400, anonymize and log |
| Topic/jailbreak | NeMo input rail + lexical fallback | 1.55 ms | Reject with 503 and reason |
| RAG pipeline | Chunk, hybrid search, rerank, answer | local | Return grounded fallback |
| Output check | NeMo output rail + PII scan | local | Block and log sensitive output |

## CI Gates

- [x] RAGAS evaluation generated for all 50 questions.
- [x] Faithfulness gate is measured on every run (target >= 0.75).
- [x] Adversarial suite pass rate >= 90% (20/20 in this run).
- [x] Total guard P95 latency < 500 ms (1.60 ms in this run).
- [x] Unit tests pass before merge (`pytest tests/ -q`).

## Monitoring

| Metric | Observed value | Alert/action |
|---|---:|---|
| Guard total P95 | 1.60 ms | Alert above 500 ms; profile NeMo/API calls |
| Adversarial pass rate | 20/20 (100%) | Review new attack patterns if below 90% |
| Worst RAGAS metric | context_precision | Tune reranker and metadata filters |
| Dominant failure distribution | factual (by worst-metric count) | Inspect retrieval noise and policy versions |
| Cohen kappa | 1.00 | Re-label samples if below 0.60 |

## Deployment and rollback

Run Phase A and C in CI, persist JSON artifacts, and fail the build when a gate is missed. In production, keep PII detection local, put NeMo behind a timeout/circuit breaker, and fall back to a refusal rather than bypassing a failed rail. Version policy documents and recognizer patterns together; rollback means restoring the previous indexed corpus and rail configuration.
