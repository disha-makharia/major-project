"""All DataMind REST endpoints."""
from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, File, HTTPException, UploadFile
from fastapi.responses import FileResponse

from backend.config import settings
from backend.data.dataset_manager import DatasetError, get_dataset_manager
from backend.graph.workflow import run_workflow
from backend.models.schemas import (
    DatasetListResponse,
    DatasetUploadResponse,
    DocumentListResponse,
    DocumentUploadResponse,
    HealthResponse,
    QueryRequest,
    QueryResponse,
    RetrievedChunk,
    SqlExecution,
    ValidationResult,
    VisualizationResult,
)
from backend.rag.document_manager import get_document_manager
from backend.rag.document_loader import DocumentLoadError
from backend.utils.llm_client import OllamaClient
from backend.utils.logger import get_logger

logger = get_logger(__name__)
router = APIRouter()


@router.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    client = OllamaClient()
    available, message = client.check_availability()
    dataset_manager = get_dataset_manager()
    document_manager = get_document_manager()
    return HealthResponse(
        status="ok",
        ollama_available=available,
        ollama_message=message,
        embedding_model=settings.embedding_model,
        datasets_loaded=len(dataset_manager.list_datasets()),
        documents_indexed=len(document_manager.list_documents()),
    )


@router.post("/upload/dataset", response_model=DatasetUploadResponse)
async def upload_dataset(file: UploadFile = File(...)) -> DatasetUploadResponse:
    logger.info("request_upload_dataset", extra={"extra_fields": {"filename": file.filename}})
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file provided.")
    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")
    try:
        info = get_dataset_manager().upload(file.filename, content)
    except DatasetError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        logger.info("upload_dataset_error", extra={"extra_fields": {"error": str(exc)}})
        raise HTTPException(status_code=500, detail=f"Failed to process dataset: {exc}") from exc
    return DatasetUploadResponse(message=f"Dataset '{info.name}' uploaded and registered.", dataset=info)


@router.post("/upload/document", response_model=DocumentUploadResponse)
async def upload_document(file: UploadFile = File(...)) -> DocumentUploadResponse:
    logger.info("request_upload_document", extra={"extra_fields": {"filename": file.filename}})
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file provided.")
    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")
    try:
        info = get_document_manager().upload(file.filename, content)
    except DocumentLoadError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        logger.info("upload_document_error", extra={"extra_fields": {"error": str(exc)}})
        raise HTTPException(status_code=500, detail=f"Failed to process document: {exc}") from exc
    return DocumentUploadResponse(message=f"Document '{info.filename}' ingested into the knowledge base.", document=info)


@router.get("/datasets", response_model=DatasetListResponse)
def list_datasets() -> DatasetListResponse:
    return DatasetListResponse(datasets=get_dataset_manager().list_datasets())


@router.get("/documents", response_model=DocumentListResponse)
def list_documents() -> DocumentListResponse:
    return DocumentListResponse(documents=get_document_manager().list_documents())


@router.get("/charts/{filename}")
def get_chart(filename: str):
    safe_name = Path(filename).name  # prevent path traversal
    path = Path(settings.chart_dir) / safe_name
    if not path.exists():
        raise HTTPException(status_code=404, detail="Chart not found.")
    return FileResponse(path, media_type="image/png")


@router.post("/query", response_model=QueryResponse)
def query(request: QueryRequest) -> QueryResponse:
    logger.info("request_query", extra={"extra_fields": {"question": request.question}})
    try:
        state = run_workflow(request.question, forced_dataset=request.dataset_name)
    except Exception as exc:
        logger.info("query_workflow_error", extra={"extra_fields": {"error": str(exc)}})
        raise HTTPException(status_code=500, detail=f"Query processing failed: {exc}") from exc

    sql_execution = None
    if state.get("needs_sql"):
        sql_execution = SqlExecution(
            query=state.get("sql_query"),
            columns=state.get("sql_columns", []),
            rows=state.get("sql_rows", []),
            row_count=state.get("sql_row_count", 0),
            dataset_used=state.get("selected_dataset"),
            error=state.get("sql_error"),
            attempts=state.get("sql_attempts", 0),
        )

    visualization = None
    if state.get("chart_type") is not None or state.get("chart_reason") is not None:
        visualization = VisualizationResult(
            generated=state.get("chart_generated", False),
            chart_type=state.get("chart_type"),
            chart_path=state.get("chart_path"),
            chart_url=state.get("chart_url"),
            reason=state.get("chart_reason"),
        )

    retrieved = [RetrievedChunk(**hit) for hit in state.get("retrieved_documents", [])]

    return QueryResponse(
        question=request.question,
        answer=state.get("final_answer", ""),
        plan=state.get("plan", []),
        sql=sql_execution,
        retrieved_documents=retrieved,
        insights=state.get("insights", []),
        visualization=visualization,
        validation=ValidationResult(
            passed=state.get("validation_passed", False),
            checks=state.get("validation_checks", {}),
            issues=state.get("validation_issues", []),
        ),
        errors=state.get("errors", []),
    )
