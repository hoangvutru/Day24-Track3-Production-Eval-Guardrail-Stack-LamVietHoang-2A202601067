"""Reusable local RAG retrieval pipeline."""
from src.m1_chunking import chunk_hierarchical, load_documents
from src.m2_search import HybridSearch
from src.m3_rerank import CrossEncoderReranker

class RAGPipeline:
    def __init__(self):
        chunks = []
        for document in load_documents():
            _, children = chunk_hierarchical(document["text"], document["metadata"])
            chunks.extend({"text": child.text, "metadata": child.metadata} for child in children)
        self.searcher = HybridSearch(); self.searcher.index(chunks); self.reranker = CrossEncoderReranker()
    def retrieve(self, question: str, top_k: int = 3):
        found = self.searcher.search(question)
        return self.reranker.rerank(question, [{"text": item.text, "score": item.score, "metadata": item.metadata} for item in found], top_k)
