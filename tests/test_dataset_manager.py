import json

import pytest

from backend.data.dataset_manager import DatasetError, DatasetManager, sanitize_table_name
from backend.database.duckdb_engine import DuckDBEngine


@pytest.fixture
def manager(tmp_path):
    engine = DuckDBEngine(db_path=str(tmp_path / "t.duckdb"))
    mgr = DatasetManager(engine=engine, data_dir=str(tmp_path / "datasets"))
    yield mgr
    engine.close()


def test_sanitize_table_name():
    assert sanitize_table_name("Sales Report 2024.csv") == "sales_report_2024_csv"
    assert sanitize_table_name("2024sales") == "t_2024sales"
    assert sanitize_table_name("") == "dataset"


def test_upload_csv_detects_schema(manager, sample_sales_csv_bytes):
    info = manager.upload("sales.csv", sample_sales_csv_bytes)
    assert info.name == "sales"
    assert info.row_count == 5
    assert info.file_type == "csv"
    column_names = {c.name for c in info.columns}
    assert {"order_id", "region", "product", "revenue"} <= column_names
    assert len(info.sample_records) <= 5
    assert info.sample_records[0]["region"] == "North"


def test_upload_json(manager):
    payload = json.dumps([{"x": 1, "y": "a"}, {"x": 2, "y": "b"}]).encode()
    info = manager.upload("things.json", payload)
    assert info.row_count == 2
    assert info.file_type == "json"


def test_upload_parquet(manager):
    import io

    import pandas as pd

    df = pd.DataFrame({"x": [1, 2, 3], "y": ["a", "b", "c"]})
    buf = io.BytesIO()
    df.to_parquet(buf, engine="pyarrow")
    info = manager.upload("things.parquet", buf.getvalue())
    assert info.row_count == 3
    assert info.file_type == "parquet"


def test_unsupported_extension_rejected(manager):
    with pytest.raises(DatasetError):
        manager.upload("evil.exe", b"not a real dataset")


def test_empty_csv_rejected(manager):
    with pytest.raises(DatasetError):
        manager.upload("empty.csv", b"col1,col2\n")


def test_dataset_registered_in_duckdb_and_queryable(manager, sample_sales_csv_bytes):
    manager.upload("sales.csv", sample_sales_csv_bytes)
    result = manager.engine.execute_query("SELECT COUNT(*) AS n FROM sales")
    assert result.rows[0]["n"] == 5


def test_catalog_persists_across_manager_instances(tmp_path, sample_sales_csv_bytes):
    engine = DuckDBEngine(db_path=str(tmp_path / "t.duckdb"))
    mgr1 = DatasetManager(engine=engine, data_dir=str(tmp_path / "datasets"))
    mgr1.upload("sales.csv", sample_sales_csv_bytes)

    mgr2 = DatasetManager(engine=engine, data_dir=str(tmp_path / "datasets"))
    assert any(d.name == "sales" for d in mgr2.list_datasets())
    engine.close()


def test_list_datasets_multiple(manager, sample_sales_csv_bytes):
    manager.upload("sales.csv", sample_sales_csv_bytes)
    manager.upload("sales2.csv", sample_sales_csv_bytes)
    names = {d.name for d in manager.list_datasets()}
    assert "sales" in names and "sales2" in names
