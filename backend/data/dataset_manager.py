"""Dataset ingestion: CSV / JSON / Parquet -> Pandas -> DuckDB.

Handles format detection, schema extraction, sample-record generation, and
keeps a small on-disk catalog (data/datasets/_catalog.json) so uploaded
datasets and metadata survive a server restart.
"""
from __future__ import annotations

import json
import math
import re
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from backend.config import settings
from backend.database.duckdb_engine import DuckDBEngine, get_duckdb_engine
from backend.models.schemas import ColumnInfo, DatasetInfo
from backend.utils.logger import get_logger

logger = get_logger(__name__)

SUPPORTED_EXTENSIONS = {".csv": "csv", ".json": "json", ".parquet": "parquet", ".pq": "parquet"}


class DatasetError(ValueError):
    """Raised for anything wrong with an uploaded dataset (bad format, empty, etc.)."""


def sanitize_table_name(stem: str) -> str:
    name = re.sub(r"[^a-zA-Z0-9_]", "_", stem.strip().lower())
    name = re.sub(r"_+", "_", name).strip("_")
    if not name:
        name = "dataset"
    if name[0].isdigit():
        name = f"t_{name}"
    return name


def detect_file_type(filename: str) -> str:
    ext = Path(filename).suffix.lower()
    if ext not in SUPPORTED_EXTENSIONS:
        raise DatasetError(
            f"Unsupported file type '{ext}'. Only CSV, JSON, and Parquet files are supported."
        )
    return SUPPORTED_EXTENSIONS[ext]


def load_dataframe(path: Path, file_type: str) -> pd.DataFrame:
    try:
        if file_type == "csv":
            df = pd.read_csv(path)
        elif file_type == "json":
            df = pd.read_json(path)
            if isinstance(df, pd.Series):
                df = df.to_frame()
        elif file_type == "parquet":
            df = pd.read_parquet(path, engine="pyarrow")
        else:  # pragma: no cover - guarded by detect_file_type
            raise DatasetError(f"Unsupported file type '{file_type}'.")
    except DatasetError:
        raise
    except Exception as exc:
        raise DatasetError(f"Failed to parse {file_type.upper()} file: {exc}") from exc

    if df.shape[1] == 0:
        raise DatasetError("Dataset has no columns.")
    if df.shape[0] == 0:
        raise DatasetError("Dataset is empty (0 rows).")
    return df


def _json_safe(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, float) and math.isnan(value):
        return None
    if isinstance(value, (pd.Timestamp, datetime)):
        return value.isoformat()
    if hasattr(value, "item"):  # numpy scalar
        return _json_safe(value.item())
    return value


def dataframe_columns(df: pd.DataFrame) -> list[ColumnInfo]:
    return [ColumnInfo(name=str(col), dtype=str(df[col].dtype)) for col in df.columns]


def dataframe_samples(df: pd.DataFrame, n: int = 5) -> list[dict[str, Any]]:
    records = df.head(n).to_dict(orient="records")
    return [{k: _json_safe(v) for k, v in record.items()} for record in records]


class DatasetManager:
    def __init__(self, engine: DuckDBEngine | None = None, data_dir: str | None = None):
        self.engine = engine or get_duckdb_engine()
        self.data_dir = Path(data_dir or settings.data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.catalog_path = self.data_dir / "_catalog.json"
        self._lock = threading.Lock()
        self._catalog: dict[str, DatasetInfo] = {}
        self._load_catalog()

    def _load_catalog(self) -> None:
        if self.catalog_path.exists():
            try:
                raw = json.loads(self.catalog_path.read_text())
                for name, payload in raw.items():
                    self._catalog[name] = DatasetInfo(**payload)
            except Exception as exc:
                logger.info("catalog_load_failed", extra={"extra_fields": {"error": str(exc)}})

    def _persist_catalog(self) -> None:
        payload = {name: json.loads(info.model_dump_json()) for name, info in self._catalog.items()}
        self.catalog_path.write_text(json.dumps(payload, indent=2))

    def bootstrap(self) -> None:
        """Ingest any dataset files already present in data_dir that aren't in the catalog yet."""
        for path in sorted(self.data_dir.iterdir()):
            if path.name.startswith("_") or path.suffix.lower() not in SUPPORTED_EXTENSIONS:
                continue
            table_name = sanitize_table_name(path.stem)
            if table_name in self._catalog and table_name in self.engine.list_tables():
                continue
            try:
                self._ingest_path(path, path.name, table_name)
            except DatasetError as exc:
                logger.info("bootstrap_ingest_failed", extra={"extra_fields": {"file": path.name, "error": str(exc)}})

    def _ingest_path(self, path: Path, original_filename: str, table_name: str) -> DatasetInfo:
        file_type = detect_file_type(original_filename)
        df = load_dataframe(path, file_type)
        self.engine.register_dataframe(table_name, df)
        info = DatasetInfo(
            name=table_name,
            file_type=file_type,
            file_name=original_filename,
            row_count=len(df),
            columns=dataframe_columns(df),
            sample_records=dataframe_samples(df),
            uploaded_at=datetime.now(timezone.utc),
        )
        with self._lock:
            self._catalog[table_name] = info
            self._persist_catalog()
        return info

    def upload(self, filename: str, content: bytes) -> DatasetInfo:
        file_type = detect_file_type(filename)  # raises DatasetError early for bad extensions
        stem = Path(filename).stem
        table_name = sanitize_table_name(stem)

        dest = self.data_dir / filename
        if dest.exists():
            suffix = Path(filename).suffix
            dest = self.data_dir / f"{stem}_{int(datetime.now().timestamp())}{suffix}"
        dest.write_bytes(content)

        try:
            return self._ingest_path(dest, filename, table_name)
        except DatasetError:
            dest.unlink(missing_ok=True)
            raise

    def list_datasets(self) -> list[DatasetInfo]:
        return list(self._catalog.values())

    def get_dataset(self, name: str) -> DatasetInfo | None:
        return self._catalog.get(name)

    def schema_context(self) -> dict[str, list[ColumnInfo]]:
        return {name: info.columns for name, info in self._catalog.items()}


_manager: DatasetManager | None = None
_manager_lock = threading.Lock()


def get_dataset_manager() -> DatasetManager:
    global _manager
    if _manager is None:
        with _manager_lock:
            if _manager is None:
                _manager = DatasetManager()
                _manager.bootstrap()
    return _manager
