"""End-to-end tests that drive the full LangGraph workflow through
run_workflow(), exercising Supervisor -> Data -> SQL/RAG -> Analysis ->
Visualization -> Response together.

Ollama is unreachable in the CI/sandbox environment (no network egress to
pull a model), so these tests verify the *graceful, real* degraded path:
every agent still runs, DuckDB/RAG/visualization code is genuinely
exercised, and the final response honestly reports that SQL generation
could not run rather than fabricating an answer. This is exactly the
behavior described in the spec for when Ollama is unavailable.
"""
from __future__ import annotations

from backend.data.dataset_manager import get_dataset_manager
from backend.graph.workflow import run_workflow


def test_workflow_runs_all_expected_nodes_with_dataset(tmp_data_dirs, sample_sales_csv_bytes):
    manager = get_dataset_manager()
    manager.upload("sales.csv", sample_sales_csv_bytes)

    state = run_workflow("What were total sales?")

    assert state["plan"][0] == "data_agent"
    assert "sql_agent" in state["plan"]
    assert state["selected_dataset"] == "sales"
    # Ollama unavailable -> SQL agent reports why rather than crashing.
    assert state["sql_query"] is None
    assert state["sql_error"] is not None
    assert state["final_answer"]
    assert state["validation_passed"] is False
    assert any("SQL" in issue for issue in state["validation_issues"])


def test_workflow_with_no_datasets_or_documents_still_returns_answer(tmp_data_dirs):
    state = run_workflow("What were total sales?")
    assert state["final_answer"]
    assert state.get("needs_sql") in (False, None) or state["sql_error"] is not None


def test_workflow_visualization_only_runs_when_sql_data_present(tmp_data_dirs, sample_sales_csv_bytes):
    manager = get_dataset_manager()
    manager.upload("sales.csv", sample_sales_csv_bytes)
    state = run_workflow("Compare sales between regions.")
    # No SQL result rows (Ollama down) -> visualization agent must not fabricate a chart.
    assert state["chart_generated"] is False


def test_forced_dataset_is_honored(tmp_data_dirs, sample_sales_csv_bytes):
    manager = get_dataset_manager()
    manager.upload("sales.csv", sample_sales_csv_bytes)
    manager.upload("sales_backup.csv", sample_sales_csv_bytes)
    state = run_workflow("total revenue", forced_dataset="sales_backup")
    assert state["selected_dataset"] == "sales_backup"
