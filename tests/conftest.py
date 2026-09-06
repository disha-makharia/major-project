"""Test configuration: isolates every DataMind test run into a throwaway
temp directory so tests never touch the real data/, generated/, or .env of
a developer's checkout, and never require Ollama or network access to pass.

Environment variables MUST be set before backend.config is imported for the
first time anywhere in the test session (Settings is a module-level
singleton), so this happens at conftest *module import* time, not inside a
fixture function.
"""
from __future__ import annotations

import os
import shutil
import sys
import tempfile
from pathlib import Path

TEST_ROOT = Path(tempfile.mkdtemp(prefix="datamind_test_"))
os.environ["DATA_DIR"] = str(TEST_ROOT / "datasets")
os.environ["DOCUMENT_DIR"] = str(TEST_ROOT / "documents")
os.environ["CHROMA_DIR"] = str(TEST_ROOT / "chroma")
os.environ["CHART_DIR"] = str(TEST_ROOT / "charts")
os.environ["DUCKDB_PATH"] = str(TEST_ROOT / "test.duckdb")
# Deliberately unreachable so every test exercises deterministic fallback
# behavior rather than depending on a real Ollama server being up.
os.environ.setdefault("OLLAMA_BASE_URL", "http://127.0.0.1:1")
os.environ.setdefault("LOG_LEVEL", "WARNING")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SAMPLE_DATASETS_DIR = PROJECT_ROOT / "data" / "datasets"
SAMPLE_DOCUMENTS_DIR = PROJECT_ROOT / "data" / "documents"
sys.path.insert(0, str(PROJECT_ROOT))

import pytest  # noqa: E402


@pytest.fixture(autouse=True)
def _reset_singletons():
    """Every test gets fresh managers/engine/vector-store so uploads in one
    test never leak into another."""
    import backend.data.dataset_manager as dm
    import backend.database.duckdb_engine as db
    import backend.rag.document_manager as docm
    import backend.rag.vector_store as vs

    dm._manager = None
    docm._manager = None
    db._engine = None
    vs._client = None
    yield
    dm._manager = None
    docm._manager = None
    if db._engine is not None:
        db._engine.close()
    db._engine = None
    vs._client = None


@pytest.fixture
def tmp_data_dirs(tmp_path, monkeypatch):
    """Point every storage path at a fresh per-test tmp directory."""
    from backend import config as cfg

    data_dir = tmp_path / "datasets"
    doc_dir = tmp_path / "documents"
    chroma_dir = tmp_path / "chroma"
    chart_dir = tmp_path / "charts"
    duckdb_path = tmp_path / "test.duckdb"

    monkeypatch.setattr(cfg.settings, "data_dir", str(data_dir))
    monkeypatch.setattr(cfg.settings, "document_dir", str(doc_dir))
    monkeypatch.setattr(cfg.settings, "chroma_dir", str(chroma_dir))
    monkeypatch.setattr(cfg.settings, "chart_dir", str(chart_dir))
    monkeypatch.setattr(cfg.settings, "duckdb_path", str(duckdb_path))
    cfg.settings.ensure_directories()
    return {"data_dir": data_dir, "doc_dir": doc_dir, "chroma_dir": chroma_dir, "chart_dir": chart_dir}


@pytest.fixture
def sample_sales_csv_bytes() -> bytes:
    return (
        b"order_id,region,product,quantity,unit_price,revenue,month\n"
        b"1,North,Widget,10,5.0,50.0,2024-01\n"
        b"2,South,Widget,4,5.0,20.0,2024-01\n"
        b"3,North,Gadget,3,20.0,60.0,2024-01\n"
        b"4,East,Gadget,8,20.0,160.0,2024-02\n"
        b"5,West,Widget,20,5.0,100.0,2024-02\n"
    )


@pytest.fixture
def embedding_available() -> bool:
    """True if the local embedding model can actually be loaded in this
    environment (requires either a cached model or network access to
    Hugging Face). Tests that need real embeddings should skip when False
    instead of failing, since that reflects an environment limitation, not
    a code defect."""
    try:
        from backend.rag.embeddings import LocalEmbeddingFunction

        fn = LocalEmbeddingFunction()
        fn(["healthcheck"])
        return True
    except Exception:
        return False
