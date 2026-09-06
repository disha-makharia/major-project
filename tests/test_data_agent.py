from backend.agents.data_agent import build_schema_context, data_agent_node
from backend.models.schemas import ColumnInfo, DatasetInfo
from datetime import datetime, timezone


def _dataset(name, columns):
    return DatasetInfo(
        name=name,
        file_type="csv",
        file_name=f"{name}.csv",
        row_count=10,
        columns=[ColumnInfo(name=c, dtype="VARCHAR") for c in columns],
        sample_records=[{c: "x" for c in columns}],
        uploaded_at=datetime.now(timezone.utc),
    )


def test_selects_best_matching_dataset_by_column_overlap():
    sales = _dataset("sales", ["region", "product", "revenue"])
    hr = _dataset("employees", ["employee_id", "department", "salary"])
    state = {"user_question": "What is the total revenue by region?", "datasets": [sales, hr], "forced_dataset": None}
    result = data_agent_node(state)
    assert result["selected_dataset"] == "sales"
    assert "region" in result["relevant_columns"]


def test_forced_dataset_overrides_scoring():
    sales = _dataset("sales", ["region", "revenue"])
    hr = _dataset("employees", ["employee_id", "salary"])
    state = {"user_question": "revenue", "datasets": [sales, hr], "forced_dataset": "employees"}
    result = data_agent_node(state)
    assert result["selected_dataset"] == "employees"


def test_no_datasets_available():
    state = {"user_question": "anything", "datasets": [], "forced_dataset": None}
    result = data_agent_node(state)
    assert result["selected_dataset"] is None
    assert "No datasets" in result["schema_context"]


def test_schema_context_includes_real_column_names_only():
    sales = _dataset("sales", ["region", "revenue"])
    context = build_schema_context([sales])
    assert "region" in context
    assert "revenue" in context
    assert "sales" in context
