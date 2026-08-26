"""In-memory lexical search with the Day 18 public interface."""
from __future__ import annotations
import math, re
from collections import Counter
from dataclasses import dataclass

def _tokens(text: str): return re.findall(r"[\wÀ-ỹ]+", text.casefold())

@dataclass
class SearchResult:
    text: str; score: float; metadata: dict

class DenseSearch:
    def __init__(self): self.documents = []
    def index(self, chunks: list[dict], collection: str | None = None): self.documents = list(chunks)
    def search(self, query: str, top_k: int = 20, collection: str | None = None) -> list[SearchResult]:
        query_counts = Counter(_tokens(query)); scored = []
        for document in self.documents:
            counts = Counter(_tokens(document["text"])); common = sum(min(counts[t], n) for t, n in query_counts.items())
            score = common / math.sqrt(max(sum(counts.values()), 1) * max(sum(query_counts.values()), 1))
            scored.append(SearchResult(document["text"], score, document.get("metadata", {})))
        return sorted(scored, key=lambda item: item.score, reverse=True)[:top_k]

class HybridSearch(DenseSearch):
    pass
