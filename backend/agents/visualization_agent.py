"""Visualization Agent: decides whether a chart adds value and renders it from
the actual SQL result rows - never a placeholder chart."""
from __future__ import annotations

from backend.agents.state import AgentState
from backend.visualization.chart_generator import decide_chart, render_chart


def visualization_agent_node(state: AgentState) -> AgentState:
    question = state["user_question"]
    columns = state.get("sql_columns", [])
    rows = state.get("sql_rows", [])

    chart_type, reason = decide_chart(question, columns, rows)
    if chart_type is None:
        return {
            **state,
            "chart_generated": False,
            "chart_type": None,
            "chart_path": None,
            "chart_url": None,
            "chart_reason": reason,
        }

    result = render_chart(chart_type, question, columns, rows)
    return {
        **state,
        "chart_generated": result.generated,
        "chart_type": result.chart_type,
        "chart_path": result.chart_path,
        "chart_url": result.chart_url,
        "chart_reason": result.reason,
    }
