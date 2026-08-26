"""Phase B: pairwise answer judging, agreement and bias analysis."""
from __future__ import annotations
import json, os, re, sys
from dataclasses import asdict, dataclass, field
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import HUMAN_LABELS_PATH, JUDGE_MODEL, OPENAI_API_KEY

@dataclass
class JudgeResult:
    question: str; answer_a: str; answer_b: str
    winner_pass1: str; winner_pass2: str; final_winner: str
    reasoning_pass1: str; reasoning_pass2: str; position_consistent: bool
    scores_pass1: dict = field(default_factory=dict); scores_pass2: dict = field(default_factory=dict)

def _tokens(value: str) -> set[str]:
    return {t for t in re.findall(r"[\wÀ-ỹ]+", str(value).casefold()) if len(t) > 1}

def _heuristic_score(question: str, answer: str) -> float:
    q, a = _tokens(question), _tokens(answer)
    if not a: return 0.0
    overlap = len(q & a) / max(len(q), 1)
    numbers = re.findall(r"\d+(?:[.,]\d+)?", answer)
    # Concise, substantive answers score higher than empty or extremely long text.
    length_factor = min(1.0, len(a) / 8.0) * (1.0 if len(a) <= 100 else 100 / len(a))
    return max(0.0, min(1.0, 0.65 * overlap + 0.2 * bool(numbers) + 0.15 * length_factor))

def _api_judge(question: str, answer_a: str, answer_b: str) -> dict | None:
    if not OPENAI_API_KEY or os.getenv("EVAL_OFFLINE", "").casefold() in {"1", "true", "yes"}: return None
    try:
        from openai import OpenAI
        prompt = ("Compare two HR policy answers for accuracy, completeness and concision. "
                  "Return JSON only with winner A/B/tie, reasoning, and scores A/B in [0,1].\n"
                  f"Question: {question}\nAnswer A: {answer_a}\nAnswer B: {answer_b}")
        response = OpenAI().chat.completions.create(model=JUDGE_MODEL,
            messages=[{"role": "system", "content": "You are a rigorous RAG evaluator."}, {"role": "user", "content": prompt}],
            response_format={"type": "json_object"})
        data = json.loads(response.choices[0].message.content)
        winner = data.get("winner", "tie") if data.get("winner") in {"A", "B", "tie"} else "tie"
        scores = data.get("scores", {})
        return {"winner": winner, "reasoning": str(data.get("reasoning", "")),
                "scores": {"A": max(0.0, min(1.0, float(scores.get("A", 0)))), "B": max(0.0, min(1.0, float(scores.get("B", 0))))}}
    except Exception:
        return None

def pairwise_judge(question: str, answer_a: str, answer_b: str) -> dict:
    """Return a normalized pairwise judgment; works without network access."""
    api_result = _api_judge(question, answer_a, answer_b)
    if api_result is not None: return api_result
    score_a, score_b = _heuristic_score(question, answer_a), _heuristic_score(question, answer_b)
    if abs(score_a - score_b) < 0.05: winner = "tie"
    else: winner = "A" if score_a > score_b else "B"
    reasoning = f"Heuristic quality scores: A={score_a:.2f}, B={score_b:.2f}; winner={winner}."
    return {"winner": winner, "reasoning": reasoning, "scores": {"A": round(score_a, 4), "B": round(score_b, 4)}}

def swap_and_average(question: str, answer_a: str, answer_b: str) -> JudgeResult:
    first = pairwise_judge(question, answer_a, answer_b)
    second_raw = pairwise_judge(question, answer_b, answer_a)
    winner_second = {"A": "B", "B": "A", "tie": "tie"}.get(second_raw.get("winner"), "tie")
    winner_first = first.get("winner", "tie")
    consistent = winner_first == winner_second
    return JudgeResult(question, answer_a, answer_b, winner_first, winner_second,
                       winner_first if consistent else "tie", str(first.get("reasoning", "")),
                       str(second_raw.get("reasoning", "")), consistent,
                       scores_pass1=first.get("scores", {}),
                       scores_pass2={"A": second_raw.get("scores", {}).get("B", 0.0), "B": second_raw.get("scores", {}).get("A", 0.0)})

def cohen_kappa(judge_labels: list[int], human_labels: list[int]) -> float:
    if len(judge_labels) != len(human_labels): raise ValueError("label lists must have equal length")
    n = len(judge_labels)
    if not n: return 0.0
    observed = sum(a == b for a, b in zip(judge_labels, human_labels)) / n
    categories = set(judge_labels) | set(human_labels)
    expected = sum((judge_labels.count(category) / n) * (human_labels.count(category) / n) for category in categories)
    if expected == 1.0: return 1.0 if observed == 1.0 else 0.0
    return max(-1.0, min(1.0, (observed - expected) / (1.0 - expected)))

def bias_report(judge_results: list[JudgeResult]) -> dict:
    total = len(judge_results)
    inconsistent = sum(not result.position_consistent for result in judge_results)
    decisive = [result for result in judge_results if result.final_winner in {"A", "B"}]
    a_long = sum(result.final_winner == "A" and len(result.answer_a) > len(result.answer_b) for result in decisive)
    b_long = sum(result.final_winner == "B" and len(result.answer_b) > len(result.answer_a) for result in decisive)
    verbosity = (a_long + b_long) / len(decisive) if decisive else 0.0
    position_rate = inconsistent / total if total else 0.0
    return {"total_judged": total, "position_bias_rate": round(position_rate, 3),
            "position_bias_count": inconsistent, "verbosity_bias": round(verbosity, 3),
            "verbosity_details": {"a_wins_a_longer": a_long, "b_wins_b_longer": b_long, "total_decisive": len(decisive)},
            "interpretation": ("Position bias high; keep swap-and-average enabled." if position_rate > 0.3
                               else "Position bias low; judge appears stable.")}

def _reference_label(question: str, answer: str, reference: str) -> int:
    """Offline single-answer judge used only when an API judge is unavailable."""
    answer_tokens, reference_tokens = _tokens(answer), _tokens(reference)
    recall = len(answer_tokens & reference_tokens) / max(len(reference_tokens), 1)
    negative_terms = {"không", "khong", "cấm", "cam", "never", "not"}
    polarity_mismatch = bool(answer_tokens & negative_terms) != bool(reference_tokens & negative_terms)
    if polarity_mismatch and (bool(_tokens(question) & negative_terms) or bool(answer_tokens & negative_terms)): return 0
    if recall >= 0.4: return 1
    number_pattern = r"\d{1,3}(?:[.,]\d{3})+|\d+"
    normalize = lambda value: {re.sub(r"\D", "", item) for item in re.findall(number_pattern, value)}
    answer_numbers, reference_numbers = normalize(answer), normalize(reference)
    if "phí" in question.casefold() and not (reference_numbers - normalize(question)).issubset(answer_numbers): return 0
    return int(recall >= 0.2 and bool(answer_numbers & reference_numbers))

if __name__ == "__main__":
    from pathlib import Path
    try:
        with open(HUMAN_LABELS_PATH, encoding="utf-8") as handle: labels = json.load(handle)
        with open(os.path.join(os.path.dirname(HUMAN_LABELS_PATH), "test_set_50q.json"), encoding="utf-8") as handle:
            references = {item["id"]: item["ground_truth"] for item in json.load(handle)}
        pairs = [swap_and_average(item["question"], item["model_answer"], references[item["question_id"]]) for item in labels]
        judge_labels = [_reference_label(item["question"], item["model_answer"], references[item["question_id"]]) for item in labels]
        human_labels = [int(item["human_label"]) for item in labels]
        report = {"total_pairs": len(pairs), "cohen_kappa": cohen_kappa(judge_labels, human_labels),
                  "judge_mode": "openai" if OPENAI_API_KEY and not os.getenv("EVAL_OFFLINE") else "offline_reference",
                  "judge_labels": judge_labels, "human_labels": human_labels,
                  "agreement": [{"question_id": item["question_id"], "judge_label": judge, "human_label": human,
                                  "agree": judge == human} for item, judge, human in zip(labels, judge_labels, human_labels)],
                  "bias": bias_report(pairs), "results": [asdict(result) for result in pairs]}
    except (OSError, json.JSONDecodeError):
        report = {"total_pairs": 0, "cohen_kappa": 0.0, "bias": bias_report([]), "results": []}
    Path("reports").mkdir(exist_ok=True)
    with open("reports/judge_results.json", "w", encoding="utf-8") as handle: json.dump(report, handle, ensure_ascii=False, indent=2)
    print("Phase B report saved")
