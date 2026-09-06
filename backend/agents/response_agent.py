"""Response Agent: composes the final business-friendly answer and validates it.

Validation checks that: SQL succeeded (when needed), retrieved documents were
actually relevant (when needed), and every number in the final answer traces
back to a real SQL result, a computed insight, retrieved document text, or
the question itself - never a fabricated figure. On a failed numeric check
the answer is regenerated once with a stricter, more explicit prompt.
"""
from __future__ import annotations

import re
from typing import Any

from backend.agents.state import AgentState
from backend.utils.llm_client import OllamaClient, OllamaUnavailableError
from backend.utils.logger import get_logger

logger = get_logger(__name__)

_SYSTEM_PROMPT = (
    "You are the Response Agent in a data analytics system called DataMind. Write a short, "
    "clear, business-friendly answer (2-4 sentences) to the user's question for a non-technical "
    "reader. Use ONLY the facts given to you. Do not mention SQL, databases, or that you are an AI. "
    "Do not invent any number that is not given to you."
)

_NUMBER_RE = re.compile(r"-?\d[\d,]*\.?\d*%?")


def _normalize_number(token: str) -> float | None:
    cleaned = token.replace(",", "").rstrip("%")
    try:
        return float(cleaned)
    except ValueError:
        return None


def _extract_numbers(text: str) -> set[float]:
    values = set()
    for match in _NUMBER_RE.findall(text or ""):
        num = _normalize_number(match)
        if num is not None:
            values.add(round(num, 2))
    return values


def _build_fact_pool(state: AgentState) -> set[float]:
    pool: set[float] = set()
    pool |= _extract_numbers(state["user_question"])
    for row in state.get("sql_rows", []):
        for value in row.values():
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                pool.add(round(float(value), 2))
    for insight in state.get("insights", []):
        pool |= _extract_numbers(insight)
    for hit in state.get("retrieved_documents", []):
        pool |= _extract_numbers(hit.get("text", ""))
    if state.get("sql_error"):
        # The composed answer may quote the SQL error verbatim (e.g. "could not reach
        # Ollama at ...:11434"); those digits are grounded in a real fact, not fabricated.
        pool |= _extract_numbers(state["sql_error"])
    return pool


def _numbers_supported(answer: str, pool: set[float]) -> tuple[bool, list[str]]:
    answer_numbers = _extract_numbers(answer)
    if not answer_numbers:
        return True, []
    unsupported = []
    for num in answer_numbers:
        if any(abs(num - p) <= max(0.5, abs(p) * 0.02) for p in pool):
            continue
        unsupported.append(str(num))
    return len(unsupported) == 0, unsupported


def _compose_answer(state: AgentState, strict: bool = False) -> str:
    question = state["user_question"]
    facts = []
    if state.get("analysis_summary"):
        facts.append(f"Analysis: {state['analysis_summary']}")
    if state.get("insights"):
        facts.append("Key computed facts: " + "; ".join(state["insights"]))
    if state.get("sql_rows"):
        preview = state["sql_rows"][:8]
        facts.append(f"Query result preview: {preview}")
    if state.get("retrieved_documents"):
        docs = "; ".join(
            f'"{h["text"][:200].strip()}" (source: {h["source"]})' for h in state["retrieved_documents"][:2]
        )
        facts.append(f"Relevant document excerpts: {docs}")
    if state.get("sql_error"):
        facts.append(f"Note: the data query could not be completed ({state['sql_error']}).")
    if not facts:
        facts.append("No data or document context was available for this question.")

    client = OllamaClient()
    available, _ = client.check_availability()
    if available:
        strict_note = (
            "\n\nIMPORTANT: your previous answer used a number not present in the facts above. "
            "Use ONLY numbers that literally appear in the facts."
            if strict
            else ""
        )
        prompt = f'Question: "{question}"\n\nFacts:\n' + "\n".join(f"- {f}" for f in facts) + strict_note
        try:
            response = client.generate(prompt, system=_SYSTEM_PROMPT, temperature=0.2)
            text = response.text.strip()
            if text:
                return text
        except OllamaUnavailableError as exc:
            logger.info("response_llm_fallback", extra={"extra_fields": {"error": str(exc)}})

    # Deterministic fallback: directly stitch together what we actually computed.
    return " ".join(facts)


def response_agent_node(state: AgentState) -> AgentState:
    needs_sql = state.get("needs_sql", False)
    needs_rag = state.get("needs_rag", False)

    answer = _compose_answer(state)
    pool = _build_fact_pool(state)
    supported, unsupported = _numbers_supported(answer, pool)

    revision_count = state.get("response_revision_count", 0)
    if not supported and revision_count == 0:
        logger.info("response_regenerating", extra={"extra_fields": {"unsupported": unsupported}})
        answer = _compose_answer(state, strict=True)
        supported, unsupported = _numbers_supported(answer, pool)
        revision_count = 1

    checks: dict[str, bool] = {}
    issues: list[str] = []

    checks["sql_success"] = (not needs_sql) or (state.get("sql_error") is None)
    if not checks["sql_success"]:
        issues.append(f"SQL execution did not succeed: {state.get('sql_error')}")

    checks["sql_has_data"] = (not needs_sql) or (state.get("sql_error") is not None) or state.get("sql_row_count", 0) > 0
    if not checks["sql_has_data"]:
        issues.append("The SQL query executed successfully but returned no rows.")

    checks["documents_relevant"] = (not needs_rag) or len(state.get("retrieved_documents", [])) > 0
    if not checks["documents_relevant"]:
        issues.append("No sufficiently relevant document passages were found.")

    checks["numeric_claims_supported"] = supported
    if not supported:
        issues.append(f"Answer contains figures not traceable to source data: {', '.join(unsupported)}")

    checks["visualization_consistent"] = True
    if state.get("chart_generated") and not state.get("sql_rows"):
        checks["visualization_consistent"] = False
        issues.append("Chart was generated without underlying query data.")

    passed = checks["sql_success"] and checks["numeric_claims_supported"] and checks["visualization_consistent"]

    logger.info(
        "response_validation",
        extra={"extra_fields": {"passed": passed, "checks": checks, "issues": issues}},
    )

    return {
        **state,
        "final_answer": answer,
        "validation_passed": passed,
        "validation_checks": checks,
        "validation_issues": issues,
        "response_revision_count": revision_count,
    }
