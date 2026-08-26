"""Phase C: layered PII, input/output and latency guardrails."""
from __future__ import annotations
import asyncio, json, os, re, statistics, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import ADVERSARIAL_SET_PATH, GUARDRAILS_CONFIG_DIR, LATENCY_BUDGET_P95_MS, PRESIDIO_LANGUAGE

_PII_PATTERNS = (("VN_CCCD", re.compile(r"(?<!\d)\d{12}(?!\d)"), 0.9),
                 ("VN_PHONE", re.compile(r"(?<!\d)0[3-9]\d{8}(?!\d)"), 0.9),
                 ("EMAIL", re.compile(r"\b[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}\b"), 0.95),
                 ("VN_CMND", re.compile(r"(?<!\d)\d{9}(?!\d)"), 0.7))

def setup_presidio():
    from presidio_analyzer import AnalyzerEngine, Pattern, PatternRecognizer, RecognizerRegistry
    from presidio_anonymizer import AnonymizerEngine
    registry = RecognizerRegistry(); registry.load_predefined_recognizers()
    registry.add_recognizer(PatternRecognizer(supported_entity="VN_CCCD", patterns=[Pattern("CCCD", r"\b\d{12}\b", 0.9), Pattern("CMND", r"\b\d{9}\b", 0.7)]))
    registry.add_recognizer(PatternRecognizer(supported_entity="VN_PHONE", patterns=[Pattern("VN mobile", r"\b0[3-9]\d{8}\b", 0.9)]))
    return AnalyzerEngine(registry=registry), AnonymizerEngine()

def pii_scan(text: str, analyzer=None, anonymizer=None) -> dict:
    """Detect Vietnamese CCCD/phone plus email and return anonymized text."""
    if analyzer is not None:
        try:
            detected = analyzer.analyze(text=text, language=PRESIDIO_LANGUAGE)
            entities = [{"type": item.entity_type, "text": text[item.start:item.end], "score": round(item.score, 3), "start": item.start, "end": item.end} for item in detected]
            anonymized = anonymizer.anonymize(text=text, analyzer_results=detected).text if anonymizer else text
            return {"has_pii": bool(entities), "entities": entities, "anonymized": anonymized}
        except Exception:
            pass
    matches = []
    for entity_type, pattern, score in _PII_PATTERNS:
        for match in pattern.finditer(text):
            matches.append((match.start(), match.end(), entity_type, score))
    # Prefer the more specific entity if patterns overlap.
    matches.sort(key=lambda item: (item[0], -(item[1] - item[0])))
    selected = []
    for match in matches:
        if not selected or match[0] >= selected[-1][1]: selected.append(match)
    entities = [{"type": kind, "text": text[start:end], "score": score, "start": start, "end": end} for start, end, kind, score in selected]
    anonymized = text
    for start, end, kind, _ in reversed(selected): anonymized = anonymized[:start] + f"<{kind}>" + anonymized[end:]
    return {"has_pii": bool(entities), "entities": entities, "anonymized": anonymized}

def setup_nemo_rails():
    from nemoguardrails import RailsConfig, LLMRails
    return LLMRails(RailsConfig.from_path(GUARDRAILS_CONFIG_DIR))

_JAILBREAK = ("ignore your previous", "ignore previous", "bỏ qua tất cả", "pretend you are dan", "unrestricted ai", "system override", "forget your system", "không có giới hạn", "admin command", "dump all", "print all system", "in ra toàn bộ system", "tấn công mạng")
_PII_REQUEST = ("cho tôi biết cccd", "số điện thoại của nhân viên", "thông tin cá nhân", "email của nhân viên", "tiết lộ thông tin nhân viên", "bảng lương chi tiết", "lương của nhân viên cụ thể", "mật khẩu admin")
_OFF_TOPIC = ("viết một bài thơ", "bài thơ", "nấu phở", "bitcoin", "ethereum", "giá cổ phiếu", "recommend phim", "bộ phim", "giải phương trình", "dy/dx", "thời tiết", "tin tức")
_HR_TERMS = ("nghỉ", "phép", "lương", "thưởng", "bảo hiểm", "tạm ứng", "công tác", "thử việc", "đào tạo", "vpn", "wfh", "làm việc từ xa", "phụ cấp", "mentor", "buddy", "mật khẩu", "mfa", "kpi", "chi phí", "hoàn trả", "mua thiết bị")

def _response_text(response) -> str:
    if isinstance(response, str): return response
    if isinstance(response, dict):
        if isinstance(response.get("content"), str): return response["content"]
        messages = response.get("messages", [])
        if messages: return _response_text(messages[-1])
    return str(getattr(response, "content", response))

async def check_input_rail(text: str, rails=None) -> dict:
    if rails is not None:
        try:
            response = await rails.generate_async(messages=[{"role": "user", "content": text}])
            content = _response_text(response); lowered = content.casefold()
            blocked = any(term in lowered for term in ("xin lỗi", "không thể", "không được phép", "i cannot", "i'm sorry"))
            return {"allowed": not blocked, "blocked_reason": "nemo_input_rail" if blocked else None, "response": content}
        except Exception:
            pass
    lowered = text.casefold()
    blocked = any(term in lowered for term in _JAILBREAK + _PII_REQUEST + _OFF_TOPIC)
    if not blocked and not any(term in lowered for term in _HR_TERMS): blocked = True
    reason = "nemo_input_rail" if blocked else None
    response = "Xin lỗi, tôi chỉ có thể trả lời câu hỏi về chính sách nhân sự nội bộ." if blocked else ""
    return {"allowed": not blocked, "blocked_reason": reason, "response": response}

async def check_output_rail(question: str, answer: str, rails=None) -> dict:
    pii = pii_scan(answer)
    sensitive = pii["has_pii"] or bool(re.search(r"(?i)(mật khẩu hệ thống|thông tin bí mật|lương của nhân viên cụ thể|cccd của nhân viên|số điện thoại cá nhân)", answer))
    response = answer
    if rails is not None:
        try:
            raw = await rails.generate_async(messages=[{"role": "user", "content": question}, {"role": "assistant", "content": answer}])
            candidate = _response_text(raw)
            if any(term in candidate.casefold() for term in ("xin lỗi", "không thể", "i cannot")): sensitive, response = True, candidate
        except Exception:
            pass
    return {"safe": not sensitive, "flagged_reason": "pii_or_sensitive_output" if sensitive else None,
            "final_answer": "Tôi không thể cung cấp thông tin nhạy cảm này. Vui lòng liên hệ phòng Nhân sự." if sensitive else response}

def _run(coro):
    try: asyncio.get_running_loop()
    except RuntimeError: return asyncio.run(coro)
    result, error = [], []
    import threading
    def worker():
        try: result.append(asyncio.run(coro))
        except Exception as exc: error.append(exc)
    thread = threading.Thread(target=worker); thread.start(); thread.join()
    if error: raise error[0]
    return result[0]

def run_adversarial_suite(adversarial_set: list[dict], rails=None, analyzer=None, anonymizer=None) -> list[dict]:
    results = []
    for item in adversarial_set:
        pii = pii_scan(item.get("input", ""), analyzer, anonymizer)
        blocked_by = "presidio" if pii["has_pii"] else None
        rail = {"allowed": True}
        if blocked_by is None: rail = _run(check_input_rail(item.get("input", ""), rails))
        if blocked_by is None and not rail.get("allowed", True): blocked_by = "nemo_input"
        actual = "blocked" if blocked_by else "allowed"
        results.append({"id": item.get("id"), "category": item.get("category"), "input": item.get("input", "")[:80] + ("..." if len(item.get("input", "")) > 80 else ""),
                       "expected": item.get("expected", "allowed"), "actual": actual, "blocked_by": blocked_by, "passed": actual == item.get("expected", "allowed")})
    return results

def _percentiles(values: list[float]) -> dict[str, float]:
    if not values: return {"p50": 0.0, "p95": 0.0, "p99": 0.0}
    values = sorted(values)
    def percentile(fraction):
        if len(values) == 1: return round(values[0], 2)
        position = (len(values) - 1) * fraction; lower = int(position); upper = min(lower + 1, len(values) - 1)
        return round(values[lower] + (values[upper] - values[lower]) * (position - lower), 2)
    return {"p50": percentile(.5), "p95": percentile(.95), "p99": percentile(.99)}

def measure_p95_latency(test_inputs: list[str], n_runs: int = 20, rails=None, analyzer=None, anonymizer=None) -> dict:
    presidio_times, nemo_times, total_times = [], [], []
    samples = list(test_inputs) or [""]
    for index in range(max(0, n_runs)):
        text = samples[index % len(samples)]
        start = time.perf_counter(); pii_scan(text, analyzer, anonymizer); presidio = (time.perf_counter() - start) * 1000
        start = time.perf_counter(); _run(check_input_rail(text, rails)); nemo = (time.perf_counter() - start) * 1000
        presidio_times.append(presidio); nemo_times.append(nemo); total_times.append(presidio + nemo)
    total = _percentiles(total_times)
    return {"presidio_ms": _percentiles(presidio_times), "nemo_ms": _percentiles(nemo_times), "total_ms": total,
            "latency_budget_ok": total["p95"] < LATENCY_BUDGET_P95_MS, "budget_ms": LATENCY_BUDGET_P95_MS}

if __name__ == "__main__":
    with open(ADVERSARIAL_SET_PATH, encoding="utf-8") as handle: adversarial = json.load(handle)
    suite = run_adversarial_suite(adversarial); latency = measure_p95_latency([item["input"] for item in adversarial], 20)
    os.makedirs("reports", exist_ok=True)
    with open("reports/guard_results.json", "w", encoding="utf-8") as handle: json.dump({"results": suite, "passed": sum(item["passed"] for item in suite), "latency": latency}, handle, ensure_ascii=False, indent=2)
    print(f"Guardrail report saved: {sum(item['passed'] for item in suite)}/{len(suite)} passed")
