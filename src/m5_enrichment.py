"""Metadata enrichment that remains usable without an external LLM."""
from dataclasses import dataclass

@dataclass
class EnrichedChunk:
    enriched_text: str
    auto_metadata: dict

def enrich_chunks(chunks: list[dict]) -> list[EnrichedChunk]:
    return [EnrichedChunk(chunk.get("text", ""), dict(chunk.get("metadata", {}))) for chunk in chunks]
