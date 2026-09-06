from backend.agents.response_agent import _build_fact_pool, _numbers_supported, response_agent_node


def test_numbers_supported_when_grounded():
    pool = {100.0, 50.0}
    supported, unsupported = _numbers_supported("Revenue was 100 in the north and 50 in the south.", pool)
    assert supported
    assert unsupported == []


def test_numbers_unsupported_when_fabricated():
    pool = {100.0}
    supported, unsupported = _numbers_supported("Revenue was 999999.", pool)
    assert not supported
    assert "999999.0" in unsupported


def test_fact_pool_includes_sql_rows_and_question_numbers():
    state = {
        "user_question": "Top 5 products",
        "sql_rows": [{"revenue": 250.5}],
        "insights": ["Total is 500"],
        "retrieved_documents": [{"text": "growth of 12%"}],
    }
    pool = _build_fact_pool(state)
    assert 5.0 in pool
    assert 250.5 in pool
    assert 500.0 in pool
    assert 12.0 in pool


def test_fact_pool_includes_sql_error_digits():
    state = {
        "user_question": "What were total sales?",
        "sql_rows": [],
        "insights": [],
        "retrieved_documents": [],
        "sql_error": "Cannot reach Ollama at http://localhost:11434.",
    }
    pool = _build_fact_pool(state)
    assert 11434.0 in pool


def test_response_agent_node_passes_validation_when_grounded():
    state = {
        "user_question": "What were total sales?",
        "needs_sql": True,
        "needs_rag": False,
        "sql_query": "SELECT SUM(revenue) AS total FROM sales",
        "sql_columns": ["total"],
        "sql_rows": [{"total": 1000}],
        "sql_row_count": 1,
        "sql_error": None,
        "insights": ["The query returned a single result: total = 1,000."],
        "analysis_summary": "Total sales were 1,000.",
        "retrieved_documents": [],
        "chart_generated": False,
    }
    result = response_agent_node(state)
    assert result["validation_checks"]["sql_success"] is True
    assert result["validation_passed"] is True
    assert result["final_answer"]


def test_response_agent_flags_sql_failure():
    state = {
        "user_question": "What were total sales?",
        "needs_sql": True,
        "needs_rag": False,
        "sql_query": None,
        "sql_columns": [],
        "sql_rows": [],
        "sql_row_count": 0,
        "sql_error": "Cannot reach Ollama at http://127.0.0.1:1",
        "insights": [],
        "analysis_summary": "",
        "retrieved_documents": [],
        "chart_generated": False,
    }
    result = response_agent_node(state)
    assert result["validation_checks"]["sql_success"] is False
    assert result["validation_passed"] is False
    assert any("SQL execution did not succeed" in issue for issue in result["validation_issues"])
    # The port number in the quoted error message is a grounded fact, not a fabrication.
    assert result["validation_checks"]["numeric_claims_supported"] is True
