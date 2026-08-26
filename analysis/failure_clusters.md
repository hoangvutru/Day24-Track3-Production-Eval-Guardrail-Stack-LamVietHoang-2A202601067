# Failure Cluster Analysis - Phase A

The 50-question run used three balanced distributions: factual 20, multi-hop 20 and adversarial 10.

## Aggregate scores

| Metric | factual | multi-hop | adversarial |
|---|---:|---:|---:|
| faithfulness | 1.0000 | 1.0000 | 1.0000 |
| answer relevancy | 0.7541 | 0.6563 | 0.7821 |
| context precision | 0.2914 | 0.2432 | 0.2517 |
| context recall | 0.8072 | 0.6042 | 0.5274 |
| **avg score** | **0.7132** | **0.6259** | **0.6403** |

## Bottom 10

The generated `reports/ragas_50q.json` contains all ten records with rank, question ID, score, diagnosis and suggested fix. The lowest item is multi-hop question 33 (0.4746); other notable failures are multi-hop questions 21, 40, 34 and adversarial questions 49, 50 and 41.

## Failure matrix

| Worst metric | factual | multi-hop | adversarial | Total |
|---|---:|---:|---:|---:|
| faithfulness | 0 | 0 | 0 | 0 |
| answer relevancy | 1 | 0 | 0 | 1 |
| context precision | 18 | 20 | 10 | 48 |
| context recall | 1 | 0 | 0 | 1 |

## Dominant failure

`context_precision` is dominant (48/50 worst-metric assignments). Factual questions often retrieve neighboring HR policies that share generic terms such as leave, allowance and approval. Multi-hop questions additionally need evidence from several documents, so a broad lexical hit can outrank the exact policy version. The remediation is metadata-aware reranking, version filtering and smaller policy-specific chunks.

## Adversarial observations

Adversarial average score (0.6403) is below factual (0.7132), indicating that version conflicts and negation traps are detected as harder cases. Questions 41, 49 and 50 appear in the bottom ten; they require selecting the current policy or honoring an explicit prohibition rather than matching a familiar older phrase.
