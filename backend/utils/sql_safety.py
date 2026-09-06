"""Guardrails that keep the SQL Agent to safe, read-only analytical queries.

DuckDB is our only SQL engine. We never want generated SQL to be able to
mutate data, touch the filesystem, or change the database/session, so every
query is validated here before it reaches DuckDB.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

# Statement types that are always dangerous, regardless of position.
_FORBIDDEN_KEYWORDS = [
    "DROP", "DELETE", "UPDATE", "INSERT", "ALTER", "CREATE", "TRUNCATE",
    "ATTACH", "DETACH", "COPY", "EXPORT", "IMPORT", "PRAGMA", "INSTALL",
    "LOAD", "CALL", "SET", "GRANT", "REVOKE", "VACUUM", "CHECKPOINT",
    "REPLACE", "MERGE", "EXEC", "EXECUTE",
]
_ALLOWED_START = ("SELECT", "WITH")

_KEYWORD_RE = {kw: re.compile(rf"(?<![A-Za-z0-9_]){kw}(?![A-Za-z0-9_])", re.IGNORECASE) for kw in _FORBIDDEN_KEYWORDS}


def _strip_comments_and_strings(sql: str) -> str:
    """Remove string literals and comments so keyword scanning ignores their contents."""
    no_line_comments = re.sub(r"--[^\n]*", " ", sql)
    no_block_comments = re.sub(r"/\*.*?\*/", " ", no_line_comments, flags=re.DOTALL)
    no_strings = re.sub(r"'(?:[^']|'')*'", "''", no_block_comments)
    no_strings = re.sub(r'"(?:[^"]|"")*"', '""', no_strings)
    return no_strings


@dataclass
class SqlValidationResult:
    is_safe: bool
    reason: str | None = None
    cleaned_sql: str | None = None


def validate_sql(sql: str) -> SqlValidationResult:
    """Validate that `sql` is a single, safe, read-only analytical statement."""
    if not sql or not sql.strip():
        return SqlValidationResult(False, "SQL query is empty.")

    cleaned = _strip_comments_and_strings(sql).strip()
    if not cleaned:
        return SqlValidationResult(False, "SQL query contains no executable statement.")

    # Reject multiple statements (stacked queries). Allow a single trailing semicolon.
    body = cleaned[:-1] if cleaned.endswith(";") else cleaned
    if ";" in body:
        return SqlValidationResult(False, "Multiple SQL statements are not allowed.")

    first_token_match = re.match(r"\s*([A-Za-z_]+)", body)
    if not first_token_match:
        return SqlValidationResult(False, "Could not determine the SQL statement type.")
    first_token = first_token_match.group(1).upper()
    if first_token not in _ALLOWED_START:
        return SqlValidationResult(
            False,
            f"Only read-only SELECT/WITH queries are allowed, got '{first_token}'.",
        )

    for keyword, pattern in _KEYWORD_RE.items():
        if pattern.search(body):
            return SqlValidationResult(False, f"Query contains forbidden keyword '{keyword}'.")

    return SqlValidationResult(True, cleaned_sql=body.strip())
