from backend.agents.analysis_agent import compute_statistical_insights


def test_single_scalar_result():
    insights = compute_statistical_insights(["total_revenue"], [{"total_revenue": 1000}])
    assert len(insights) == 1
    assert "1,000" in insights[0]


def test_ranking_insight_identifies_top_and_bottom():
    columns = ["product", "revenue"]
    rows = [
        {"product": "Laptop", "revenue": 1000},
        {"product": "Mouse", "revenue": 100},
        {"product": "Monitor", "revenue": 500},
    ]
    insights = compute_statistical_insights(columns, rows)
    joined = " ".join(insights)
    assert "Laptop" in joined and "highest" in joined
    assert "Mouse" in joined and "lowest" in joined
    assert "1,600" in joined  # total


def test_trend_insight_for_time_series():
    columns = ["month", "revenue"]
    rows = [
        {"month": "2024-01", "revenue": 100},
        {"month": "2024-02", "revenue": 150},
        {"month": "2024-03", "revenue": 50},
    ]
    insights = compute_statistical_insights(columns, rows)
    joined = " ".join(insights)
    assert "%" in joined


def test_no_insights_for_empty_rows():
    assert compute_statistical_insights(["a"], []) == []


def test_correlation_insight_for_two_numeric_columns():
    columns = ["price", "quantity"]
    rows = [{"price": p, "quantity": p * 2} for p in range(1, 10)]
    insights = compute_statistical_insights(columns, rows)
    assert any("correlation" in i for i in insights)
