"""Local RAG metric implementation compatible with the Phase A adapter."""
from __future__ import annotations
import json, re
from dataclasses import asdict, dataclass
from config import TEST_SET_PATH

def _tokens(text): return set(re.findall(r"[\wÀ-ỹ]+", str(text).casefold()))
@dataclass
class EvalResult:
    faithfulness: float; answer_relevancy: float; context_precision: float; context_recall: float

def load_test_set(path: str = TEST_SET_PATH):
    with open(path, encoding="utf-8") as handle: return json.load(handle)

def evaluate_ragas(questions, answers, contexts, ground_truths):
    per_question = []
    for question, answer, context, truth in zip(questions, answers, contexts, ground_truths):
        q, a, g, c = _tokens(question), _tokens(answer), _tokens(truth), _tokens(" ".join(context))
        per_question.append(EvalResult(len(a & c) / max(len(a), 1), len(a & q) / max(len(q), 1),
                                       len(a & g) / max(len(a), 1), len(a & g) / max(len(g), 1)))
    output = {metric: sum(getattr(item, metric) for item in per_question) / max(len(per_question), 1)
              for metric in ("faithfulness", "answer_relevancy", "context_precision", "context_recall")}
    output["per_question"] = per_question; return output

def save_report(results: dict, details: list, path: str):
    serializable = {key: ([asdict(item) for item in value] if key == "per_question" else value) for key, value in results.items()}
    serializable["details"] = details
    with open(path, "w", encoding="utf-8") as handle: json.dump(serializable, handle, ensure_ascii=False, indent=2)
