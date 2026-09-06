from backend.agents.supervisor import _heuristic_plan


def test_data_question_with_datasets_needs_sql():
    needs_sql, needs_rag, _ = _heuristic_plan("What were total sales?", has_datasets=True, has_documents=False)
    assert needs_sql is True
    assert needs_rag is False


def test_document_question_needs_rag():
    needs_sql, needs_rag, _ = _heuristic_plan(
        "What does the sales report say about the decline in Q3?", has_datasets=True, has_documents=True
    )
    assert needs_rag is True


def test_no_resources_available_disables_both():
    needs_sql, needs_rag, _ = _heuristic_plan("What were total sales?", has_datasets=False, has_documents=False)
    assert needs_sql is False
    assert needs_rag is False


def test_mixed_question_can_need_both():
    needs_sql, needs_rag, _ = _heuristic_plan(
        "Why did sales decrease in Q3 and what does our sales report say about it?",
        has_datasets=True,
        has_documents=True,
    )
    assert needs_sql is True
    assert needs_rag is True
