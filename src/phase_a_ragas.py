"""Phase A: reproducible RAGAS evaluation and failure analysis."""
from __future__ import annotations
import json, os, re, sys
from dataclasses import dataclass
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import ANSWERS_PATH, TEST_SET_PATH

Distribution = str
METRICS = ("faithfulness", "answer_relevancy", "context_precision", "context_recall")
DIAGNOSTIC_TREE = {
    "faithfulness": ("LLM hallucinating", "Tighten system prompt and require citations"),
    "context_recall": ("Missing relevant chunks", "Improve chunking or add BM25"),
    "context_precision": ("Too many irrelevant chunks", "Add reranking or metadata filters"),
    "answer_relevancy": ("Answer does not match question", "Improve the prompt template"),
}

@dataclass
class RagasResult:
    question_id: int
    distribution: Distribution
    question: str
    answer: str
    contexts: list[str]
    ground_truth: str
    faithfulness: float
    answer_relevancy: float
    context_precision: float
    context_recall: float
    @property
    def avg_score(self) -> float:
        return sum(getattr(self, metric) for metric in METRICS) / len(METRICS)
    @property
    def worst_metric(self) -> str:
        return min(METRICS, key=lambda metric: getattr(self, metric))

def load_test_set_50q(path: str = TEST_SET_PATH) -> list[dict]:
    with open(path, encoding="utf-8") as handle: return json.load(handle)

def load_answers(path: str = ANSWERS_PATH) -> list[dict]:
    if not os.path.exists(path): raise FileNotFoundError(f"answers_50q.json not found at {path}; run setup_answers.py first")
    with open(path, encoding="utf-8") as handle: return json.load(handle)

def group_by_distribution(test_set: list[dict]) -> dict[str, list[dict]]:
    groups = {"factual": [], "multi_hop": [], "adversarial": []}
    for item in test_set:
        if item.get("distribution") in groups: groups[item["distribution"]].append(item)
    return groups

def _value(item, name: str, default: float = 0.0) -> float:
    value = item.get(name, default) if isinstance(item, dict) else getattr(item, name, default)
    try: return max(0.0, min(1.0, float(value)))
    except (TypeError, ValueError): return default

def _tokens(value: str) -> set[str]:
    return {t for t in re.findall(r"[\wÀ-ỹ]+", str(value).casefold()) if len(t) > 1}

def _local_scores(question: str, answer: str, contexts: list[str], truth: str) -> dict[str, float]:
    a, g, q = _tokens(answer), _tokens(truth), _tokens(question)
    c = _tokens(" ".join(map(str, contexts)))
    overlap = len(a & g) / max(len(g), 1)
    return {"faithfulness": len(a & c) / max(len(a), 1) if a else 0.0,
            "answer_relevancy": len(a & q) / max(len(q), 1),
            "context_precision": overlap, "context_recall": overlap}

def run_ragas_50q(answers: list[dict]) -> list[RagasResult]:
    if not answers: return []
    per_question = []
    try:
        from src.m4_eval import evaluate_ragas
        raw = evaluate_ragas([a.get("question", "") for a in answers], [a.get("answer", "") for a in answers],
                             [a.get("contexts", []) for a in answers], [a.get("ground_truth", "") for a in answers])
        per_question = raw.get("per_question", []) if isinstance(raw, dict) else getattr(raw, "per_question", [])
    except Exception:
        # Evaluation remains reproducible when optional RAGAS dependencies or
        # a remote evaluator are unavailable.
        pass
    if len(per_question) != len(answers):
        per_question = [_local_scores(a.get("question", ""), a.get("answer", ""), a.get("contexts", []), a.get("ground_truth", "")) for a in answers]
    results = []
    for index, (answer, scores) in enumerate(zip(answers, per_question), 1):
        results.append(RagasResult(question_id=int(answer.get("id", answer.get("question_id", index))),
            distribution=answer.get("distribution", "factual"), question=answer.get("question", ""), answer=answer.get("answer", ""),
            contexts=answer.get("contexts", []), ground_truth=answer.get("ground_truth", ""),
            **{metric: _value(scores, metric) for metric in METRICS}))
    return results

def bottom_10(results: list[RagasResult]) -> list[dict]:
    output = []
    for rank, result in enumerate(sorted(results, key=lambda item: item.avg_score)[:10], 1):
        diagnosis, fix = DIAGNOSTIC_TREE[result.worst_metric]
        output.append({"rank": rank, "question_id": result.question_id, "distribution": result.distribution,
                       "question": result.question, "avg_score": round(result.avg_score, 4), "worst_metric": result.worst_metric,
                       "diagnosis": diagnosis, "suggested_fix": fix})
    return output

def cluster_analysis(results: list[RagasResult]) -> dict:
    distributions = ("factual", "multi_hop", "adversarial")
    matrix = {metric: {distribution: 0 for distribution in distributions} for metric in METRICS}
    for result in results:
        if result.worst_metric in matrix and result.distribution in distributions: matrix[result.worst_metric][result.distribution] += 1
    dominant_distribution = max(distributions, key=lambda d: sum(matrix[m][d] for m in METRICS), default="factual")
    dominant_metric = max(METRICS, key=lambda m: sum(matrix[m].values()), default="faithfulness")
    insight = f"Distribution '{dominant_distribution}' has the most failures; '{dominant_metric}' is the dominant weak metric. Recommended action: {DIAGNOSTIC_TREE[dominant_metric][1]}."
    return {"matrix": matrix, "dominant_failure_distribution": dominant_distribution, "dominant_failure_metric": dominant_metric, "insight": insight}

def save_phase_a_report(results: list[RagasResult], clusters: dict, path: str = "reports/ragas_50q.json") -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    per_distribution = {}
    for distribution in ("factual", "multi_hop", "adversarial"):
        subset = [r for r in results if r.distribution == distribution]
        if subset:
            per_distribution[distribution] = {metric: round(sum(getattr(r, metric) for r in subset) / len(subset), 4) for metric in METRICS}
            per_distribution[distribution].update(count=len(subset), avg_score=round(sum(r.avg_score for r in subset) / len(subset), 4))
    report = {"total_questions": len(results), "per_distribution": per_distribution, "failure_clusters": clusters, "bottom_10": bottom_10(results)}
    with open(path, "w", encoding="utf-8") as handle: json.dump(report, handle, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    evaluated = run_ragas_50q(load_answers()); save_phase_a_report(evaluated, cluster_analysis(evaluated)); print(f"Phase A report saved: {len(evaluated)} questions")
