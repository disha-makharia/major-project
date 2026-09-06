"""Central configuration for DataMind, loaded from environment variables / .env."""
from __future__ import annotations

import os
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

# Resolve project root as two levels up from this file (backend/config.py -> project root)
PROJECT_ROOT = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # --- Ollama LLM configuration ---
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "llama3.1"
    ollama_timeout_seconds: float = 120.0

    # --- Embeddings ---
    embedding_model: str = "all-MiniLM-L6-v2"

    # --- Storage directories ---
    data_dir: str = str(PROJECT_ROOT / "data" / "datasets")
    document_dir: str = str(PROJECT_ROOT / "data" / "documents")
    chroma_dir: str = str(PROJECT_ROOT / "data" / "chroma")
    chart_dir: str = str(PROJECT_ROOT / "generated" / "charts")
    duckdb_path: str = str(PROJECT_ROOT / "data" / "datamind.duckdb")

    # --- RAG ---
    chunk_size: int = 800
    chunk_overlap: int = 150
    rag_top_k: int = 4
    rag_min_similarity: float = 0.15

    # --- App ---
    app_host: str = "0.0.0.0"
    app_port: int = 8000
    log_level: str = "INFO"
    max_sql_retries: int = 2
    cors_origins: str = "*"

    def ensure_directories(self) -> None:
        for path in (self.data_dir, self.document_dir, self.chroma_dir, self.chart_dir):
            Path(path).mkdir(parents=True, exist_ok=True)
        Path(self.duckdb_path).parent.mkdir(parents=True, exist_ok=True)


settings = Settings()
settings.ensure_directories()
