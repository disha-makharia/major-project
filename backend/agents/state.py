"""Shared LangGraph state carried between every agent node."""
from __future__ import annotations

from typing import Any, TypedDict

from backend.models.schemas import DatasetInfo


class AgentState(TypedDict, total=False):
    # Input
    user_question: str
    forced_dataset: str | None

    # Supervisor output
    plan: list[str]
    needs_sql: bool
    needs_rag: bool
    supervisor_reasoning: str

    # Data agent output
    datasets: list[DatasetInfo]
    selected_dataset: str | None
    relevant_columns: list[str]
    schema_context: str

    # SQL agent output
    sql_query: str | None
    sql_columns: list[str]
    sql_rows: list[dict[str, Any]]
    sql_row_count: int
    sql_error: str | None
    sql_attempts: int

    # RAG agent output
    retrieved_documents: list[dict[str, Any]]

    # Analysis agent output
    insights: list[str]
    analysis_summary: str

    # Visualization agent output
    chart_generated: bool
    chart_type: str | None
    chart_path: str | None
    chart_url: str | None
    chart_reason: str | None

    # Response / validation
    final_answer: str
    validation_passed: bool
    validation_checks: dict[str, bool]
    validation_issues: list[str]
    response_revision_count: int

    # Bookkeeping
    errors: list[str]
