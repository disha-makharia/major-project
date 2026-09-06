"""ChromaDB-backed vector store for the RAG pipeline."""
from __future__ import annotations

import threading

import chromadb

from backend.config import settings
from backend.rag.embeddings import get_embedding_function
from backend.utils.logger import get_logger

logger = get_logger(__name__)

COLLECTION_NAME = "documents"

_client = None
_client_lock = threading.Lock()


def get_chroma_client():
    global _client
    if _client is None:
        with _client_lock:
            if _client is None:
                _client = chromadb.PersistentClient(path=settings.chroma_dir)
    return _client


def get_collection():
    client = get_chroma_client()
    return client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=get_embedding_function(),
        metadata={"hnsw:space": "cosine"},
    )


def add_document_chunks(filename: str, chunks: list[str]) -> int:
    if not chunks:
        return 0
    collection = get_collection()
    ids = [f"{filename}::{i}" for i in range(len(chunks))]
    metadatas = [{"source": filename, "chunk_index": i} for i in range(len(chunks))]
    collection.add(ids=ids, documents=chunks, metadatas=metadatas)
    logger.info("rag_document_ingested", extra={"extra_fields": {"file": filename, "chunks": len(chunks)}})
    return len(chunks)


def delete_document(filename: str) -> None:
    collection = get_collection()
    collection.delete(where={"source": filename})


def search(query: str, top_k: int | None = None, min_similarity: float | None = None) -> list[dict]:
    top_k = top_k or settings.rag_top_k
    min_similarity = settings.rag_min_similarity if min_similarity is None else min_similarity
    collection = get_collection()
    if collection.count() == 0:
        return []

    result = collection.query(query_texts=[query], n_results=min(top_k, collection.count()))
    documents = result.get("documents", [[]])[0]
    metadatas = result.get("metadatas", [[]])[0]
    distances = result.get("distances", [[]])[0]

    hits = []
    for doc, meta, distance in zip(documents, metadatas, distances):
        similarity = 1.0 - distance  # cosine distance -> similarity, since hnsw:space=cosine
        if similarity < min_similarity:
            continue
        hits.append(
            {
                "source": meta.get("source", "unknown"),
                "chunk_index": meta.get("chunk_index", 0),
                "text": doc,
                "similarity": round(float(similarity), 4),
            }
        )
    logger.info(
        "rag_search",
        extra={"extra_fields": {"query": query, "hits": len(hits), "considered": len(documents)}},
    )
    return hits
