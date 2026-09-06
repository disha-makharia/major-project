import pandas as pd
import pytest

from backend.database.duckdb_engine import DuckDBEngine, SqlExecutionError, SqlSafetyError


@pytest.fixture
def engine(tmp_path):
    eng = DuckDBEngine(db_path=str(tmp_path / "t.duckdb"))
    yield eng
    eng.close()


def test_register_and_query(engine):
    df = pd.DataFrame({"region": ["North", "South"], "revenue": [100, 50]})
    engine.register_dataframe("sales", df)

    result = engine.execute_query("SELECT region, revenue FROM sales ORDER BY revenue DESC")
    assert result.columns == ["region", "revenue"]
    assert result.row_count == 2
    assert result.rows[0]["region"] == "North"


def test_table_schema_reflects_real_columns(engine):
    df = pd.DataFrame({"a": [1], "b": ["x"]})
    engine.register_dataframe("t1", df)
    schema = engine.table_schema("t1")
    names = [c[0] for c in schema]
    assert names == ["a", "b"]


def test_list_tables(engine):
    df = pd.DataFrame({"a": [1]})
    engine.register_dataframe("t1", df)
    engine.register_dataframe("t2", df)
    assert set(engine.list_tables()) == {"t1", "t2"}


def test_dangerous_sql_raises_safety_error(engine):
    df = pd.DataFrame({"a": [1]})
    engine.register_dataframe("t1", df)
    with pytest.raises(SqlSafetyError):
        engine.execute_query("DROP TABLE t1")


def test_invented_column_raises_execution_error(engine):
    df = pd.DataFrame({"a": [1]})
    engine.register_dataframe("t1", df)
    with pytest.raises(SqlExecutionError):
        engine.execute_query("SELECT nonexistent_column FROM t1")


def test_aggregation_join_and_group_by(engine):
    sales = pd.DataFrame(
        {"product": ["Widget", "Widget", "Gadget"], "region": ["N", "S", "N"], "revenue": [50.0, 20.0, 60.0]}
    )
    products = pd.DataFrame({"product": ["Widget", "Gadget"], "category": ["Tools", "Gizmos"]})
    engine.register_dataframe("sales", sales)
    engine.register_dataframe("products", products)

    result = engine.execute_query(
        "SELECT p.category, SUM(s.revenue) AS total FROM sales s "
        "JOIN products p ON s.product = p.product GROUP BY p.category ORDER BY total DESC"
    )
    totals = {row["category"]: row["total"] for row in result.rows}
    assert totals["Tools"] == 70.0
    assert totals["Gizmos"] == 60.0
