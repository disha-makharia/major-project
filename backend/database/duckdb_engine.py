"""DuckDB engine: the single SQL execution surface for DataMind.

All dataset tables/views live in one persistent DuckDB database file so the
SQL Agent can join across uploaded datasets. Only validated, read-only SQL
(see backend.utils.sql_safety) ever reaches `execute_query`.
"""
from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import Any

import duckdb
import pandas as pd

from backend.config import settings
from backend.utils.logger import get_logger
from backend.utils.sql_safety import validate_sql

logger = get_logger(__name__)


class SqlExecutionError(RuntimeError):
    pass


class SqlSafetyError(RuntimeError):
    pass


@dataclass
class QueryResult:
    columns: list[str]
    rows: list[dict[str, Any]]
    row_count: int


class DuckDBEngine:
    """A process-wide, thread-safe DuckDB connection wrapper."""

    def __init__(self, db_path: str | None = None):
        self.db_path = db_path or settings.duckdb_path
        self._lock = threading.Lock()
        self._conn = duckdb.connect(self.db_path)

    def register_dataframe(self, table_name: str, df: pd.DataFrame) -> None:
        """Persist a pandas DataFrame as a real DuckDB table (survives across requests)."""
        with self._lock:
            self._conn.register("_incoming_df", df)
            safe_name = quote_identifier(table_name)
            self._conn.execute(f"CREATE OR REPLACE TABLE {safe_name} AS SELECT * FROM _incoming_df")
            self._conn.unregister("_incoming_df")
        logger.info("dataset_registered", extra={"extra_fields": {"table": table_name, "rows": len(df)}})

    def drop_table(self, table_name: str) -> None:
        with self._lock:
            self._conn.execute(f"DROP TABLE IF EXISTS {quote_identifier(table_name)}")

    def list_tables(self) -> list[str]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT table_name FROM information_schema.tables WHERE table_schema = 'main'"
            ).fetchall()
        return [r[0] for r in rows]

    def table_schema(self, table_name: str) -> list[tuple[str, str]]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT column_name, data_type FROM information_schema.columns "
                "WHERE table_name = ? ORDER BY ordinal_position",
                [table_name],
            ).fetchall()
        return [(r[0], r[1]) for r in rows]

    def execute_query(self, sql: str, row_limit: int = 1000) -> QueryResult:
        """Validate and execute a read-only SQL statement. Raises SqlSafetyError / SqlExecutionError."""
        validation = validate_sql(sql)
        if not validation.is_safe:
            logger.info("sql_rejected", extra={"extra_fields": {"reason": validation.reason, "sql": sql}})
            raise SqlSafetyError(validation.reason)

        clean_sql = validation.cleaned_sql
        try:
            with self._lock:
                cursor = self._conn.execute(clean_sql)
                columns = [desc[0] for desc in cursor.description] if cursor.description else []
                fetched = cursor.fetchmany(row_limit)
            rows = [dict(zip(columns, row)) for row in fetched]
            logger.info(
                "sql_executed",
                extra={"extra_fields": {"sql": clean_sql, "row_count": len(rows)}},
            )
            return QueryResult(columns=columns, rows=rows, row_count=len(rows))
        except SqlSafetyError:
            raise
        except Exception as exc:
            logger.info("sql_execution_failed", extra={"extra_fields": {"sql": clean_sql, "error": str(exc)}})
            raise SqlExecutionError(str(exc)) from exc

    def close(self) -> None:
        with self._lock:
            self._conn.close()


def quote_identifier(identifier: str) -> str:
    """Safely quote a DuckDB identifier (table/column name)."""
    escaped = identifier.replace('"', '""')
    return f'"{escaped}"'


_engine: DuckDBEngine | None = None
_engine_lock = threading.Lock()


def get_duckdb_engine() -> DuckDBEngine:
    global _engine
    if _engine is None:
        with _engine_lock:
            if _engine is None:
                _engine = DuckDBEngine()
    return _engine
