"""Data Agent: inspects real uploaded datasets and builds the schema context
that the SQL Agent will use verbatim, so it never has to guess table or
column names.
"""
from __future__ import annotations

import re

from backend.agents.state import AgentState
from backend.models.schemas import DatasetInfo
from backend.utils.logger import get_logger

logger = get_logger(__name__)

_WORD_RE = re.compile(r"[a-zA-Z0-9]+")


def _tokenize(text: str) -> set[str]:
    return {t.lower() for t in _WORD_RE.findall(text) if len(t) > 2}


def _score_dataset(question_tokens: set[str], dataset: DatasetInfo) -> int:
    name_tokens = _tokenize(dataset.name)
    column_tokens: set[str] = set()
    for col in dataset.columns:
        column_tokens |= _tokenize(col.name)
    return len(question_tokens & name_tokens) * 2 + len(question_tokens & column_tokens)


def build_schema_context(datasets: list[DatasetInfo]) -> str:
    """Render every dataset's real schema + a couple of sample rows, verbatim,
    so downstream agents (mainly the SQL Agent) only ever see real identifiers."""
    if not datasets:
        return "No datasets are currently uploaded."

    blocks = []
    for ds in datasets:
        cols = ", ".join(f"{c.name} ({c.dtype})" for c in ds.columns)
        sample = ds.sample_records[:2]
        blocks.append(
            f"Table: {ds.name}  (source file: {ds.file_name}, {ds.row_count} rows)\n"
            f"Columns: {cols}\n"
            f"Sample rows: {sample}"
        )
    return "\n\n".join(blocks)


def data_agent_node(state: AgentState) -> AgentState:
    datasets = state.get("datasets", [])
    question = state["user_question"]
    forced = state.get("forced_dataset")

    selected_dataset = None
    relevant_columns: list[str] = []

    if forced and any(d.name == forced for d in datasets):
        selected_dataset = forced
    elif datasets:
        tokens = _tokenize(question)
        scored = sorted(datasets, key=lambda d: _score_dataset(tokens, d), reverse=True)
        selected_dataset = scored[0].name

    if selected_dataset:
        ds = next(d for d in datasets if d.name == selected_dataset)
        tokens = _tokenize(question)
        relevant_columns = [c.name for c in ds.columns if _tokenize(c.name) & tokens]
        if not relevant_columns:
            relevant_columns = [c.name for c in ds.columns]

    schema_context = build_schema_context(datasets)

    logger.info(
        "data_agent_selection",
        extra={"extra_fields": {"selected_dataset": selected_dataset, "relevant_columns": relevant_columns}},
    )

    return {
        **state,
        "selected_dataset": selected_dataset,
        "relevant_columns": relevant_columns,
        "schema_context": schema_context,
    }
