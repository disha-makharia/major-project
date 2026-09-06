from backend.utils.sql_safety import validate_sql


def test_allows_simple_select():
    result = validate_sql("SELECT * FROM sales")
    assert result.is_safe
    assert result.cleaned_sql == "SELECT * FROM sales"


def test_allows_with_cte():
    result = validate_sql("WITH t AS (SELECT 1) SELECT * FROM t")
    assert result.is_safe


def test_rejects_drop():
    result = validate_sql("DROP TABLE sales")
    assert not result.is_safe
    assert "Only read-only" in result.reason


def test_rejects_delete_disguised_in_lowercase():
    result = validate_sql("delete from sales where id = 1")
    assert not result.is_safe


def test_rejects_embedded_dangerous_keyword():
    result = validate_sql("SELECT * FROM sales; DROP TABLE sales;")
    assert not result.is_safe
    assert "Multiple SQL statements" in result.reason


def test_rejects_insert():
    result = validate_sql("INSERT INTO sales VALUES (1,2,3)")
    assert not result.is_safe


def test_rejects_update():
    assert not validate_sql("UPDATE sales SET revenue = 0").is_safe


def test_rejects_alter():
    assert not validate_sql("ALTER TABLE sales ADD COLUMN x INT").is_safe


def test_rejects_pragma_and_attach():
    assert not validate_sql("PRAGMA database_list").is_safe
    assert not validate_sql("ATTACH 'evil.db' AS evil").is_safe


def test_column_named_like_keyword_is_fine():
    # 'created_at' contains 'create' as a substring but must not be flagged.
    result = validate_sql("SELECT created_at FROM sales")
    assert result.is_safe


def test_string_literal_containing_keyword_is_fine():
    result = validate_sql("SELECT * FROM sales WHERE region = 'DROP ZONE'")
    assert result.is_safe


def test_empty_query_rejected():
    assert not validate_sql("").is_safe
    assert not validate_sql("   ").is_safe


def test_non_select_start_rejected():
    result = validate_sql("EXPLAIN SELECT * FROM sales")
    assert not result.is_safe
