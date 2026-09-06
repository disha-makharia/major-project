from pathlib import Path

from backend.visualization.chart_generator import decide_chart, render_chart


def test_bar_chart_for_category_and_numeric():
    columns = ["region", "revenue"]
    rows = [{"region": "North", "revenue": 100}, {"region": "South", "revenue": 50}, {"region": "East", "revenue": 75}]
    chart_type, reason = decide_chart("Compare sales between regions", columns, rows)
    assert chart_type == "bar"
    assert reason is None


def test_line_chart_for_time_series():
    columns = ["month", "revenue"]
    rows = [{"month": "2024-01", "revenue": 100}, {"month": "2024-02", "revenue": 150}]
    chart_type, _ = decide_chart("What were the monthly sales trends?", columns, rows)
    assert chart_type == "line"


def test_pie_chart_for_share_question_with_few_categories():
    columns = ["region", "revenue"]
    rows = [{"region": "North", "revenue": 100}, {"region": "South", "revenue": 50}]
    chart_type, _ = decide_chart("What is the revenue share by region?", columns, rows)
    assert chart_type == "pie"


def test_no_chart_for_single_scalar_result():
    chart_type, reason = decide_chart("What were total sales?", ["total_revenue"], [{"total_revenue": 1000}])
    assert chart_type is None
    assert "single aggregate value" in reason


def test_no_chart_for_empty_result():
    chart_type, reason = decide_chart("anything", ["a"], [])
    assert chart_type is None


def test_scatter_for_two_numeric_columns():
    columns = ["unit_price", "quantity"]
    rows = [{"unit_price": p, "quantity": 100 - p} for p in range(5, 40, 5)]
    chart_type, _ = decide_chart("Is there a relationship between price and quantity?", columns, rows)
    assert chart_type == "scatter"


def test_histogram_for_single_numeric_column_many_rows():
    columns = ["revenue"]
    rows = [{"revenue": i * 3.3} for i in range(20)]
    chart_type, _ = decide_chart("What is the distribution of order revenue?", columns, rows)
    assert chart_type == "histogram"


def test_render_bar_chart_creates_file(tmp_path):
    columns = ["region", "revenue"]
    rows = [{"region": "North", "revenue": 100}, {"region": "South", "revenue": 50}]
    result = render_chart("bar", "Compare regions", columns, rows, chart_dir=str(tmp_path))
    assert result.generated
    assert result.chart_path is not None
    assert Path(result.chart_path).exists()
    assert result.chart_url.startswith("/charts/")
