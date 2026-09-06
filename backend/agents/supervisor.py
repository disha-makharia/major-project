"""Supervisor Agent: decides which downstream agents this question actually needs.

The workflow must not blindly run every agent for every question, so the
supervisor inspects the question (with the LLM when available, and a
deterministic keyword heuristic otherwise) and only turns on SQL / RAG when
they're likely to matter, given what data/documents actually exist.
"""
from __future__ import annotations

import json
import re

from backend.agents.state import AgentState
from backend.utils.llm_client import OllamaClient, OllamaUnavailableError
from backend.utils.logger import get_logger

logger = get_logger(__name__)

_DOC_HINTS = re.compile(
    r"\b(report|document|say|says|according to|explain|why|reason|policy|memo|summary|narrative)\b",
    re.IGNORECASE,
)
_DATA_HINTS = re.compile(
    r"\b(total|sum|average|avg|count|top|highest|lowest|revenue|sales|trend|compare|comparison|"
    r"growth|decline|rank|ranking|by region|by product|by month|how many|how much)\b",
    re.IGNORECASE,
)

_SYSTEM_PROMPT = (
    "You are the Supervisor Agent in a multi-agent data analytics system called DataMind. "
    "Given a user's business question and what resources exist, decide which specialist agents "
    "are actually needed. Respond ONLY with a compact JSON object: "
    '{"needs_sql": true/false, "needs_rag": true/false, "reasoning": "one short sentence"}. '
    "needs_sql should be true only if answering requires computing something from tabular data. "
    "needs_rag should be true only if answering benefits from company documents/reports."
)


def _heuristic_plan(question: str, has_datasets: bool, has_documents: bool) -> tuple[bool, bool, str]:
    needs_sql = has_datasets and (bool(_DATA_HINTS.search(question)) or not _DOC_HINTS.search(question))
    needs_rag = has_documents and bool(_DOC_HINTS.search(question))
    if not has_datasets:
        needs_sql = False
    if not has_documents:
        needs_rag = False
    if not needs_sql and not needs_rag:
        # Fall back to whichever resource exists so the question is still attempted.
        needs_sql = has_datasets
        needs_rag = has_documents and not needs_sql
    reasoning = "Heuristic routing based on question keywords (Ollama unavailable or ambiguous response)."
    return needs_sql, needs_rag, reasoning


def supervisor_node(state: AgentState) -> AgentState:
    question = state["user_question"]
    has_datasets = bool(state.get("datasets"))
    has_documents = state.get("_has_documents", False)

    needs_sql, needs_rag, reasoning = _heuristic_plan(question, has_datasets, has_documents)

    client = OllamaClient()
    available, _ = client.check_availability()
    if available:
        try:
            resources = []
            if has_datasets:
                names = ", ".join(d.name for d in state["datasets"])
                resources.append(f"Available datasets (DuckDB tables): {names}")
            else:
                resources.append("No datasets are uploaded.")
            resources.append("Company documents are indexed for search." if has_documents else "No documents are indexed.")

            prompt = (
                f"Resources:\n{chr(10).join(resources)}\n\n"
                f'User question: "{question}"\n\n'
                "Decide needs_sql and needs_rag as instructed."
            )
            response = client.generate(prompt, system=_SYSTEM_PROMPT, temperature=0.0, format_json=True)
            parsed = json.loads(response.text)
            needs_sql = bool(parsed.get("needs_sql", needs_sql)) and has_datasets
            needs_rag = bool(parsed.get("needs_rag", needs_rag)) and has_documents
            reasoning = str(parsed.get("reasoning", "LLM-routed plan."))
        except (OllamaUnavailableError, json.JSONDecodeError, Exception) as exc:
            logger.info("supervisor_llm_fallback", extra={"extra_fields": {"error": str(exc)}})

    plan = []
    plan.append("data_agent")
    if needs_sql:
        plan.append("sql_agent")
    if needs_rag:
        plan.append("rag_agent")
    plan.append("analysis_agent")
    plan.append("visualization_agent")
    plan.append("response_agent")

    logger.info(
        "supervisor_plan",
        extra={"extra_fields": {"needs_sql": needs_sql, "needs_rag": needs_rag, "plan": plan}},
    )

    return {
        **state,
        "needs_sql": needs_sql,
        "needs_rag": needs_rag,
        "plan": plan,
        "supervisor_reasoning": reasoning,
        "errors": state.get("errors", []),
    }
