"""DataMind FastAPI application entrypoint.

Run with:  uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000
(run from the project root so the `backend` package resolves).
"""
from __future__ import annotations

# Windows note: chromadb pulls in onnxruntime as a hard dependency (it builds a
# default embedding function eagerly at class-definition time, even though we
# always pass our own local embedding function and never use it). On Windows,
# onnxruntime's native extension can fail to initialize ("DLL load failed while
# importing onnxruntime_pybind11_state") if other native-extension packages
# (DuckDB, PyArrow) have already loaded a conflicting copy of a shared runtime
# DLL (commonly the OpenMP runtime) into the process first. Importing chromadb
# here, before any other backend module gets a chance to import duckdb/pyarrow,
# ensures its native DLLs are the ones that win the Windows DLL search order.
import chromadb  # noqa: F401

from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from backend.api.routes import router
from backend.config import PROJECT_ROOT, settings
from backend.utils.llm_client import OllamaClient
from backend.utils.logger import get_logger

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    from backend.data.dataset_manager import get_dataset_manager
    from backend.rag.document_manager import get_document_manager

    get_dataset_manager()
    get_document_manager()

    client = OllamaClient()
    available, message = client.check_availability()
    if available:
        logger.info("startup_ollama_ok", extra={"extra_fields": {"message": message}})
    else:
        logger.info("startup_ollama_unavailable", extra={"extra_fields": {"message": message}})
    logger.info(
        "startup_complete",
        extra={"extra_fields": {"data_dir": settings.data_dir, "chroma_dir": settings.chroma_dir}},
    )
    yield


app = FastAPI(
    title="DataMind",
    description="Multi-Agent RAG platform for AWS-style data analytics: ask questions over "
    "your own datasets and documents in plain English.",
    version="1.0.0",
    lifespan=lifespan,
)

origins = [o.strip() for o in settings.cors_origins.split(",")] if settings.cors_origins != "*" else ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)

FRONTEND_DIR = PROJECT_ROOT / "frontend"
if FRONTEND_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")

    @app.get("/")
    def serve_frontend():
        return FileResponse(str(FRONTEND_DIR / "index.html"))


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    logger.info("unhandled_exception", extra={"extra_fields": {"path": str(request.url), "error": str(exc)}})
    return JSONResponse(status_code=500, content={"detail": "An unexpected server error occurred."})
