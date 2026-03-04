from __future__ import annotations

import json
import logging
import math
from pathlib import Path
from typing import Any

try:
    import chromadb
except ImportError:  # pragma: no cover - exercised via runtime environment
    chromadb = None

from app.models.schemas import ChunkRecord, RetrievedChunk


logger = logging.getLogger(__name__)


class VectorStore:
    """Vector storage for chunk embeddings + metadata.

    Uses ChromaDB when available. Falls back to a local JSON index otherwise,
    so the project can run in constrained environments (e.g., no C++ build tools).
    """

    def __init__(self, persist_path: Path, collection_name: str) -> None:
        persist_path.mkdir(parents=True, exist_ok=True)

        self._backend = "chroma"
        self._index_file = persist_path / f"{collection_name}.json"
        self._entries: dict[str, dict[str, Any]] = {}

        if chromadb is None:
            self._backend = "local"
            self.client = None
            self.collection = None
            self._load_local_index()
            logger.warning("chromadb not available; using local JSON vector index backend.")
            return

        self.client = chromadb.PersistentClient(path=str(persist_path))
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def upsert_chunks(self, chunks: list[ChunkRecord], embeddings: list[list[float]]) -> None:
        if len(chunks) != len(embeddings):
            raise ValueError("Number of chunks and embeddings must match.")

        ids = [chunk.chunk_id for chunk in chunks]
        documents = [chunk.content for chunk in chunks]
        metadatas = [
            {
                "page_number": chunk.page_number,
                "source_type": chunk.source_type,
                "section_title": chunk.section_title or "",
                "table_id": chunk.table_id or "",
                "token_count": chunk.token_count,
            }
            for chunk in chunks
        ]

        if self._backend == "chroma":
            self.collection.upsert(
                ids=ids,
                documents=documents,
                metadatas=metadatas,
                embeddings=embeddings,
            )
            return

        for chunk, embedding, metadata in zip(chunks, embeddings, metadatas):
            self._entries[chunk.chunk_id] = {
                "content": chunk.content,
                "metadata": metadata,
                "embedding": embedding,
            }
        self._save_local_index()

    def query(self, query_embedding: list[float], top_k: int = 6) -> list[RetrievedChunk]:
        if self._backend == "chroma":
            result = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=top_k,
                include=["documents", "metadatas", "distances"],
            )

            ids = result.get("ids", [[]])[0]
            docs = result.get("documents", [[]])[0]
            metadatas = result.get("metadatas", [[]])[0]
            distances = result.get("distances", [[]])[0]

            retrieved: list[RetrievedChunk] = []
            for chunk_id, doc, metadata, distance in zip(ids, docs, metadatas, distances):
                score = max(0.0, 1.0 - float(distance))
                retrieved.append(
                    RetrievedChunk(
                        chunk_id=chunk_id,
                        page_number=int(metadata.get("page_number", -1)),
                        source_type=metadata.get("source_type", "text"),
                        section_title=metadata.get("section_title") or None,
                        table_id=metadata.get("table_id") or None,
                        content=doc,
                        score=score,
                    )
                )
            return retrieved

        retrieved: list[RetrievedChunk] = []
        ranked = sorted(
            (
                (
                    self._cosine_similarity(query_embedding, entry.get("embedding", [])),
                    chunk_id,
                    entry,
                )
                for chunk_id, entry in self._entries.items()
            ),
            key=lambda item: item[0],
            reverse=True,
        )

        for score, chunk_id, entry in ranked[:top_k]:
            metadata = entry.get("metadata", {})
            retrieved.append(
                RetrievedChunk(
                    chunk_id=chunk_id,
                    page_number=int(metadata.get("page_number", -1)),
                    source_type=metadata.get("source_type", "text"),
                    section_title=metadata.get("section_title") or None,
                    table_id=metadata.get("table_id") or None,
                    content=str(entry.get("content", "")),
                    score=max(0.0, float(score)),
                )
            )
        return retrieved

    def count(self) -> int:
        if self._backend == "chroma":
            return self.collection.count()
        return len(self._entries)

    def _load_local_index(self) -> None:
        if not self._index_file.exists():
            return
        try:
            data = json.loads(self._index_file.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                self._entries = data
        except json.JSONDecodeError:
            logger.warning("Local vector index is corrupt; starting from empty index.")
            self._entries = {}

    def _save_local_index(self) -> None:
        self._index_file.write_text(json.dumps(self._entries), encoding="utf-8")

    @staticmethod
    def _cosine_similarity(left: list[float], right: list[float]) -> float:
        if not left or not right or len(left) != len(right):
            return 0.0
        dot = sum(float(a) * float(b) for a, b in zip(left, right))
        left_norm = math.sqrt(sum(float(a) * float(a) for a in left))
        right_norm = math.sqrt(sum(float(b) * float(b) for b in right))
        if left_norm == 0.0 or right_norm == 0.0:
            return 0.0
        return dot / (left_norm * right_norm)
