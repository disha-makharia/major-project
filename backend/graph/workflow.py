"""The LangGraph orchestration graph that wires every DataMind agent together.

Supervisor -> Data -> (SQL and/or RAG, conditionally) -> Analysis -> Visualization -> Response

The supervisor decides, per question, whether the SQL and RAG branches are
needed at all - the graph does not blindly run every agent for every query.
"""
from __future__ import annotations

from langgraph.graph import END, START, StateGraph

from backend.agents.analysis_agent import analysis_agent_node
from backend.agents.data_agent import data_agent_node
from backend.agents.rag_agent import rag_agent_node
from backend.agents.response_agent import response_agent_node
from backend.agents.sql_agent import sql_agent_node
from backend.agents.state import AgentState
from backend.agents.supervisor import supervisor_node
from backend.agents.visualization_agent import visualization_agent_node
from backend.data.dataset_manager import get_dataset_manager
from backend.rag.document_manager import get_document_manager
from backend.utils.logger import get_logger

logger = get_logger(__name__)


def _route_after_data(state: AgentState) -> list[str]:
    targets = []
    if state.get("needs_sql"):
        targets.append("sql_agent")
    if state.get("needs_rag"):
        targets.append("rag_agent")
    if not targets:
        targets.append("analysis_agent")
    logger.info("graph_route", extra={"extra_fields": {"from": "data_agent", "to": targets}})
    return targets


def build_graph():
    graph = StateGraph(AgentState)

    graph.add_node("supervisor", supervisor_node)
    graph.add_node("data_agent", data_agent_node)
    graph.add_node("sql_agent", sql_agent_node)
    graph.add_node("rag_agent", rag_agent_node)
    graph.add_node("analysis_agent", analysis_agent_node)
    graph.add_node("visualization_agent", visualization_agent_node)
    graph.add_node("response_agent", response_agent_node)

    graph.add_edge(START, "supervisor")
    graph.add_edge("supervisor", "data_agent")
    graph.add_conditional_edges("data_agent", _route_after_data, ["sql_agent", "rag_agent", "analysis_agent"])
    graph.add_edge("sql_agent", "analysis_agent")
    graph.add_edge("rag_agent", "analysis_agent")
    graph.add_edge("analysis_agent", "visualization_agent")
    graph.add_edge("visualization_agent", "response_agent")
    graph.add_edge("response_agent", END)

    return graph.compile()


_compiled_graph = None


def get_compiled_graph():
    global _compiled_graph
    if _compiled_graph is None:
        _compiled_graph = build_graph()
    return _compiled_graph


def run_workflow(question: str, forced_dataset: str | None = None) -> AgentState:
    dataset_manager = get_dataset_manager()
    document_manager = get_document_manager()

    datasets = dataset_manager.list_datasets()
    has_documents = len(document_manager.list_documents()) > 0

    initial_state: AgentState = {
        "user_question": question,
        "forced_dataset": forced_dataset,
        "datasets": datasets,
        "_has_documents": has_documents,  # type: ignore[typeddict-item]
        "errors": [],
    }

    logger.info("workflow_start", extra={"extra_fields": {"question": question}})
    graph = get_compiled_graph()
    final_state = graph.invoke(initial_state)
    logger.info(
        "workflow_end",
        extra={"extra_fields": {"question": question, "validation_passed": final_state.get("validation_passed")}},
    )
    return final_state
