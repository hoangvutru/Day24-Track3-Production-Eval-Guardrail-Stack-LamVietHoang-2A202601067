"""Deterministic lexical reranker with no model download requirement."""
from __future__ import annotations
import re
from dataclasses import dataclass

@dataclass
class RerankResult:
    text: str; score: float; metadata: dict

class CrossEncoderReranker:
    def rerank(self, query: str, documents: list[dict], top_k: int = 3) -> list[RerankResult]:
        query_tokens = set(re.findall(r"[\wÀ-ỹ]+", query.casefold()))
        output = []
        for document in documents:
            tokens = set(re.findall(r"[\wÀ-ỹ]+", document.get("text", "").casefold()))
            lexical = len(query_tokens & tokens) / max(len(query_tokens), 1)
            output.append(RerankResult(document.get("text", ""), lexical + .1 * float(document.get("score", 0)), document.get("metadata", {})))
        return sorted(output, key=lambda item: item.score, reverse=True)[:top_k]
