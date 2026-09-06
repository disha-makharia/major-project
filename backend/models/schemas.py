"""Pydantic request/response models for the DataMind API."""
from __future__ import annotations

from datetime import datetime
from typing import Any, Optional

from pydantic import BaseModel, Field


class ColumnInfo(BaseModel):
    name: str
    dtype: str


class DatasetInfo(BaseModel):
    name: str
    file_type: str
    file_name: str
    row_count: int
    columns: list[ColumnInfo]
    sample_records: list[dict[str, Any]]
    uploaded_at: datetime


class DatasetUploadResponse(BaseModel):
    message: str
    dataset: DatasetInfo


class DatasetListResponse(BaseModel):
    datasets: list[DatasetInfo]


class DocumentInfo(BaseModel):
    filename: str
    doc_type: str
    chunk_count: int
    uploaded_at: datetime


class DocumentUploadResponse(BaseModel):
    message: str
    document: DocumentInfo


class DocumentListResponse(BaseModel):
    documents: list[DocumentInfo]


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=3, description="Natural language business question")
    dataset_name: Optional[str] = Field(
        default=None, description="Optionally force a specific dataset instead of auto-detection"
    )


class RetrievedChunk(BaseModel):
    source: str
    chunk_index: int
    text: str
    similarity: float


class SqlExecution(BaseModel):
    query: Optional[str] = None
    columns: list[str] = Field(default_factory=list)
    rows: list[dict[str, Any]] = Field(default_factory=list)
    row_count: int = 0
    dataset_used: Optional[str] = None
    error: Optional[str] = None
    attempts: int = 0


class VisualizationResult(BaseModel):
    generated: bool
    chart_type: Optional[str] = None
    chart_path: Optional[str] = None
    chart_url: Optional[str] = None
    reason: Optional[str] = None


class ValidationResult(BaseModel):
    passed: bool
    checks: dict[str, bool] = Field(default_factory=dict)
    issues: list[str] = Field(default_factory=list)


class QueryResponse(BaseModel):
    question: str
    answer: str
    plan: list[str] = Field(default_factory=list)
    sql: Optional[SqlExecution] = None
    retrieved_documents: list[RetrievedChunk] = Field(default_factory=list)
    insights: list[str] = Field(default_factory=list)
    visualization: Optional[VisualizationResult] = None
    validation: ValidationResult
    errors: list[str] = Field(default_factory=list)


class HealthResponse(BaseModel):
    status: str
    ollama_available: bool
    ollama_message: str
    embedding_model: str
    datasets_loaded: int
    documents_indexed: int
