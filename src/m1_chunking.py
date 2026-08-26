"""Lightweight document loading and hierarchical chunking used by the lab."""
from __future__ import annotations
import os, re
from dataclasses import dataclass, field
from config import DATA_DIR, HIERARCHICAL_CHILD_SIZE, HIERARCHICAL_PARENT_SIZE

@dataclass
class Chunk:
    text: str
    metadata: dict = field(default_factory=dict)
    parent_id: str | None = None

def load_documents(data_dir: str = DATA_DIR) -> list[dict]:
    documents = []
    for name in sorted(os.listdir(data_dir)):
        path = os.path.join(data_dir, name); extension = os.path.splitext(name)[1].lower()
        text = ""
        if extension == ".md":
            with open(path, encoding="utf-8") as handle: text = handle.read()
        elif extension == ".pdf":
            try:
                from pypdf import PdfReader
                text = "\n".join(page.extract_text() or "" for page in PdfReader(path).pages)
            except Exception: continue
        if text.strip(): documents.append({"text": text, "metadata": {"source": name, "path": path}})
    return documents

def _windows(text: str, size: int, overlap: int = 40) -> list[str]:
    words = text.split(); step = max(1, size - overlap)
    return [" ".join(words[start:start + size]) for start in range(0, len(words), step) if words[start:start + size]]

def chunk_basic(text: str, metadata: dict | None = None) -> list[Chunk]:
    paragraphs = [part.strip() for part in re.split(r"\n\s*\n", text) if part.strip()]
    return [Chunk(part, dict(metadata or {})) for part in paragraphs]

def chunk_hierarchical(text: str, metadata: dict | None = None) -> tuple[list[Chunk], list[Chunk]]:
    parents, children = [], []
    parent_size = max(100, HIERARCHICAL_PARENT_SIZE // 4); child_size = max(50, HIERARCHICAL_CHILD_SIZE)
    for index, parent_text in enumerate(_windows(text, parent_size, 80)):
        parent_id = f"{(metadata or {}).get('source', 'doc')}:{index}"
        parents.append(Chunk(parent_text, dict(metadata or {}), parent_id))
        for child_text in _windows(parent_text, child_size, 40): children.append(Chunk(child_text, dict(metadata or {}), parent_id))
    return parents, children
