"""Analysis Agent: combines SQL results and RAG context into grounded insights.

Every insight is computed directly from the actual query result rows or
quotes retrieved document text - nothing here is fabricated. The LLM (when
available) is only used to phrase a narrative summary of facts we already
computed; it is explicitly instructed not to introduce new numbers.
"""
from __future__ import annotations

from typing import Any

import pandas as pd

from backend.agents.state import AgentState
from backend.utils.llm_client import OllamaClient, OllamaUnavailableError
from backend.utils.logger import get_logger

logger = get_logger(__name__)

_SYSTEM_PROMPT = (
    "You are the Analysis Agent in a data analytics system. You are given facts that were "
    "already computed from real query results and real document excerpts. Write a short "
    "(3-5 sentence) plain-English analysis that only restates and connects those facts. "
    "Do NOT invent any number, statistic, or claim that is not explicitly present in the facts given."
)


def _fmt_number(value: Any) -> str:
    try:
        as_float = float(value)
    except (TypeError, ValueError):
        return str(value)
    if as_float.is_integer():
        return f"{int(as_float):,}"
    return f"{as_float:,.2f}"


def compute_statistical_insights(columns: list[str], rows: list[dict[str, Any]]) -> list[str]:
    if not rows or not columns:
        return []

    df = pd.DataFrame(rows)
    numeric_cols = [c for c in columns if pd.api.types.is_numeric_dtype(df[c])]
    categorical_cols = [c for c in columns if c not in numeric_cols]
    insights: list[str] = []

    if len(rows) == 1:
        row = rows[0]
        facts = ", ".join(f"{col} = {_fmt_number(row[col])}" for col in columns)
        insights.append(f"The query returned a single result: {facts}.")
        return insights

    if numeric_cols and categorical_cols:
        cat_col, val_col = categorical_cols[0], numeric_cols[0]
        sorted_df = df.sort_values(val_col, ascending=False)
        top, bottom = sorted_df.iloc[0], sorted_df.iloc[-1]
        insights.append(
            f"'{top[cat_col]}' has the highest {val_col} at {_fmt_number(top[val_col])}."
        )
        if len(df) > 1 and top[cat_col] != bottom[cat_col]:
            insights.append(
                f"'{bottom[cat_col]}' has the lowest {val_col} at {_fmt_number(bottom[val_col])}."
            )
        total = df[val_col].sum()
        insights.append(f"Total {val_col} across all {len(df)} rows is {_fmt_number(total)}.")

        date_like = [c for c in categorical_cols if any(h in c.lower() for h in ("date", "month", "year", "quarter", "period"))]
        if date_like:
            time_col = date_like[0]
            ordered = df.sort_values(time_col)
            first_val, last_val = ordered.iloc[0][val_col], ordered.iloc[-1][val_col]
            if first_val:
                pct_change = ((last_val - first_val) / abs(first_val)) * 100
                direction = "increased" if pct_change >= 0 else "decreased"
                insights.append(
                    f"{val_col} {direction} by {abs(pct_change):.1f}% from {ordered.iloc[0][time_col]} "
                    f"({_fmt_number(first_val)}) to {ordered.iloc[-1][time_col]} ({_fmt_number(last_val)})."
                )
    elif len(numeric_cols) >= 2:
        col_a, col_b = numeric_cols[0], numeric_cols[1]
        corr = df[col_a].corr(df[col_b])
        if pd.notna(corr):
            insights.append(f"{col_a} and {col_b} have a correlation of {corr:.2f} across {len(df)} rows.")

    return insights


def _fallback_summary(question: str, insights: list[str], doc_snippets: list[str]) -> str:
    parts = []
    if insights:
        parts.append(" ".join(insights))
    if doc_snippets:
        parts.append("Related document context: " + " ".join(doc_snippets))
    if not parts:
        parts.append("No data or document context was available to analyze this question.")
    return " ".join(parts)


def analysis_agent_node(state: AgentState) -> AgentState:
    question = state["user_question"]
    sql_columns = state.get("sql_columns", [])
    sql_rows = state.get("sql_rows", [])
    retrieved_documents = state.get("retrieved_documents", [])

    insights = compute_statistical_insights(sql_columns, sql_rows) if sql_rows else []
    doc_snippets = [
        f'"{hit["text"][:220].strip()}" (source: {hit["source"]})' for hit in retrieved_documents[:3]
    ]

    summary = None
    if insights or doc_snippets:
        client = OllamaClient()
        available, _ = client.check_availability()
        if available:
            facts = "Computed facts:\n" + "\n".join(f"- {i}" for i in insights) if insights else "Computed facts: none."
            docs = "\nDocument excerpts:\n" + "\n".join(f"- {d}" for d in doc_snippets) if doc_snippets else ""
            prompt = f'Question: "{question}"\n\n{facts}{docs}\n\nWrite the analysis now.'
            try:
                response = client.generate(prompt, system=_SYSTEM_PROMPT, temperature=0.2)
                summary = response.text.strip()
            except OllamaUnavailableError as exc:
                logger.info("analysis_llm_fallback", extra={"extra_fields": {"error": str(exc)}})

    if not summary:
        summary = _fallback_summary(question, insights, doc_snippets)

    return {**state, "insights": insights, "analysis_summary": summary}
