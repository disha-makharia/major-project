import io

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def client(tmp_data_dirs):
    from backend.main import app

    return TestClient(app)


def test_health_endpoint(client):
    res = client.get("/health")
    assert res.status_code == 200
    body = res.json()
    assert body["status"] == "ok"
    assert "ollama_available" in body
    assert body["datasets_loaded"] == 0
    assert body["documents_indexed"] == 0


def test_upload_dataset_endpoint(client, sample_sales_csv_bytes):
    res = client.post(
        "/upload/dataset",
        files={"file": ("sales.csv", io.BytesIO(sample_sales_csv_bytes), "text/csv")},
    )
    assert res.status_code == 200, res.text
    body = res.json()
    assert body["dataset"]["name"] == "sales"
    assert body["dataset"]["row_count"] == 5


def test_upload_dataset_rejects_bad_extension(client):
    res = client.post(
        "/upload/dataset",
        files={"file": ("virus.exe", io.BytesIO(b"not a dataset"), "application/octet-stream")},
    )
    assert res.status_code == 400
    assert "Unsupported file type" in res.json()["detail"]


def test_upload_dataset_rejects_empty_file(client):
    res = client.post(
        "/upload/dataset",
        files={"file": ("empty.csv", io.BytesIO(b""), "text/csv")},
    )
    assert res.status_code == 400


def test_datasets_listing_after_upload(client, sample_sales_csv_bytes):
    client.post("/upload/dataset", files={"file": ("sales.csv", io.BytesIO(sample_sales_csv_bytes), "text/csv")})
    res = client.get("/datasets")
    assert res.status_code == 200
    names = [d["name"] for d in res.json()["datasets"]]
    assert "sales" in names


def test_chart_not_found_returns_404(client):
    res = client.get("/charts/does-not-exist.png")
    assert res.status_code == 404


def test_query_with_no_data_returns_graceful_response(client):
    res = client.post("/query", json={"question": "What were total sales?"})
    assert res.status_code == 200
    body = res.json()
    assert "answer" in body
    assert "validation" in body


def test_query_after_uploading_dataset_runs_sql_branch(client, sample_sales_csv_bytes):
    client.post("/upload/dataset", files={"file": ("sales.csv", io.BytesIO(sample_sales_csv_bytes), "text/csv")})
    res = client.post("/query", json={"question": "What were total sales?"})
    assert res.status_code == 200
    body = res.json()
    assert body["sql"] is not None
    # Ollama is unreachable in the test environment, so SQL generation should
    # fail gracefully rather than crash the request.
    assert body["sql"]["error"] is not None
    assert body["validation"]["passed"] is False
