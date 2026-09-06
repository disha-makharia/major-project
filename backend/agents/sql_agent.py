"""SQL Agent: turns a natural-language question into DuckDB SQL, executes it,
and self-corrects on failure. Never invents columns - the prompt is built
exclusively from the real schema context produced by the Data Agent, and any
hallucinated identifier gets caught by DuckDB's binder and fed back for a
correction attempt.
"""
from __future__ import annotations

import re

from backend.agents.state import AgentState
from backend.config import settings
from backend.database.duckdb_engine import SqlExecutionError, SqlSafetyError, get_duckdb_engine
from backend.utils.llm_client import OllamaClient, OllamaUnavailableError
from backend.utils.logger import get_logger

logger = get_logger(__name__)

_SYSTEM_PROMPT = (
    "You are the SQL Agent in a data analytics system. You write DuckDB SQL for analytical, "
    "read-only questions. Rules:\n"
    "1. Use ONLY the exact table and column names given in the schema - never invent one.\n"
    "2. Write a single SELECT (or WITH ... SELECT) statement. Never use DDL/DML (no INSERT, "
    "UPDATE, DELETE, DROP, ALTER, CREATE).\n"
    "3. Use DuckDB SQL syntax.\n"
    "4. Respond with ONLY the SQL query, no explanation, no markdown fences."
)

_SQL_FENCE_RE = re.compile(r"```(?:sql)?\s*(.*?)```", re.IGNORECASE | re.DOTALL)


def _extract_sql(text: str) -> str:
    fenced = _SQL_FENCE_RE.search(text)
    candidate = fenced.group(1) if fenced else text
    candidate = candidate.strip()
    # Drop a leading "sql" language hint some models add outside fences.
    candidate = re.sub(r"^sql\s*\n", "", candidate, flags=re.IGNORECASE)
    return candidate.strip()


def _build_prompt(question: str, schema_context: str, prior_sql: str | None, prior_error: str | None) -> str:
    parts = [f"Schema:\n{schema_context}", f'Question: "{question}"']
    if prior_sql and prior_error:
        parts.append(
            f"Your previous query failed:\n{prior_sql}\n\nError: {prior_error}\n"
            "Write a corrected query using only the real table/column names above."
        )
    parts.append("SQL query:")
    return "\n\n".join(parts)


def sql_agent_node(state: AgentState) -> AgentState:
    question = state["user_question"]
    schema_context = state.get("schema_context", "")
    errors = list(state.get("errors", []))

    if not state.get("datasets"):
        return {
            **state,
            "sql_query": None,
            "sql_columns": [],
            "sql_rows": [],
            "sql_row_count": 0,
            "sql_error": "No datasets are uploaded.",
            "sql_attempts": 0,
            "errors": errors + ["SQL agent skipped: no datasets uploaded."],
        }

    engine = get_duckdb_engine()
    client = OllamaClient()
    available, msg = client.check_availability()
    if not available:
        return {
            **state,
            "sql_query": None,
            "sql_columns": [],
            "sql_rows": [],
            "sql_row_count": 0,
            "sql_error": msg,
            "sql_attempts": 0,
            "errors": errors + [f"SQL agent could not run: {msg}"],
        }

    max_attempts = settings.max_sql_retries + 1
    prior_sql, prior_error = None, None
    sql_query, columns, rows, row_count, last_error = None, [], [], 0, None

    for attempt in range(1, max_attempts + 1):
        prompt = _build_prompt(question, schema_context, prior_sql, prior_error)
        try:
            response = client.generate(prompt, system=_SYSTEM_PROMPT, temperature=0.0)
        except OllamaUnavailableError as exc:
            last_error = str(exc)
            break

        candidate = _extract_sql(response.text)
        if not candidate:
            prior_sql, prior_error = response.text, "No SQL statement found. Respond with ONLY the SQL query."
            last_error = prior_error
            continue

        try:
            result = engine.execute_query(candidate)
            sql_query, columns, rows, row_count, last_error = candidate, result.columns, result.rows, result.row_count, None
            break
        except SqlSafetyError as exc:
            prior_sql, prior_error = candidate, f"Rejected for safety: {exc}"
            last_error = str(exc)
        except SqlExecutionError as exc:
            prior_sql, prior_error = candidate, f"Execution failed: {exc}"
            last_error = str(exc)

        logger.info(
            "sql_agent_retry",
            extra={"extra_fields": {"attempt": attempt, "error": last_error}},
        )

    if sql_query is None and last_error:
        errors = errors + [f"SQL agent failed after {attempt} attempt(s): {last_error}"]

    return {
        **state,
        "sql_query": sql_query,
        "sql_columns": columns,
        "sql_rows": rows,
        "sql_row_count": row_count,
        "sql_error": last_error if sql_query is None else None,
        "sql_attempts": attempt,
        "errors": errors,
    }
