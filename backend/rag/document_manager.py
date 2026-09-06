"""Document ingestion orchestration: load -> chunk -> embed -> store, plus a catalog."""
from __future__ import annotations

import json
import threading
from datetime import datetime, timezone
from pathlib import Path

from backend.config import settings
from backend.models.schemas import DocumentInfo
from backend.rag import vector_store
from backend.rag.chunker import chunk_text
from backend.rag.document_loader import SUPPORTED_DOC_EXTENSIONS, detect_doc_type, extract_text
from backend.utils.logger import get_logger

logger = get_logger(__name__)


class DocumentManager:
    def __init__(self, document_dir: str | None = None):
        self.document_dir = Path(document_dir or settings.document_dir)
        self.document_dir.mkdir(parents=True, exist_ok=True)
        self.catalog_path = self.document_dir / "_catalog.json"
        self._lock = threading.Lock()
        self._catalog: dict[str, DocumentInfo] = {}
        self._load_catalog()

    def _load_catalog(self) -> None:
        if self.catalog_path.exists():
            try:
                raw = json.loads(self.catalog_path.read_text())
                for name, payload in raw.items():
                    self._catalog[name] = DocumentInfo(**payload)
            except Exception as exc:
                logger.info("doc_catalog_load_failed", extra={"extra_fields": {"error": str(exc)}})

    def _persist_catalog(self) -> None:
        payload = {name: json.loads(info.model_dump_json()) for name, info in self._catalog.items()}
        self.catalog_path.write_text(json.dumps(payload, indent=2))

    def bootstrap(self) -> None:
        for path in sorted(self.document_dir.iterdir()):
            if path.name.startswith("_") or path.suffix.lower() not in SUPPORTED_DOC_EXTENSIONS:
                continue
            if path.name in self._catalog:
                continue
            try:
                self._ingest_path(path, path.name)
            except Exception as exc:
                logger.info("doc_bootstrap_failed", extra={"extra_fields": {"file": path.name, "error": str(exc)}})

    def _ingest_path(self, path: Path, filename: str) -> DocumentInfo:
        doc_type = detect_doc_type(filename)
        text = extract_text(path, doc_type)
        chunks = chunk_text(text)
        chunk_count = vector_store.add_document_chunks(filename, chunks)
        info = DocumentInfo(
            filename=filename,
            doc_type=doc_type,
            chunk_count=chunk_count,
            uploaded_at=datetime.now(timezone.utc),
        )
        with self._lock:
            self._catalog[filename] = info
            self._persist_catalog()
        return info

    def upload(self, filename: str, content: bytes) -> DocumentInfo:
        detect_doc_type(filename)  # validates extension early
        dest = self.document_dir / filename
        if dest.exists():
            stem, suffix = Path(filename).stem, Path(filename).suffix
            dest = self.document_dir / f"{stem}_{int(datetime.now().timestamp())}{suffix}"
        dest.write_bytes(content)
        try:
            return self._ingest_path(dest, dest.name)
        except Exception:
            dest.unlink(missing_ok=True)
            raise

    def list_documents(self) -> list[DocumentInfo]:
        return list(self._catalog.values())


_manager: DocumentManager | None = None
_manager_lock = threading.Lock()


def get_document_manager() -> DocumentManager:
    global _manager
    if _manager is None:
        with _manager_lock:
            if _manager is None:
                _manager = DocumentManager()
                _manager.bootstrap()
    return _manager
