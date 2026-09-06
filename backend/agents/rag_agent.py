"""RAG Agent: semantic search over ingested company documents via ChromaDB."""
from __future__ import annotations

from backend.agents.state import AgentState
from backend.rag import vector_store
from backend.rag.embeddings import EmbeddingUnavailableError
from backend.utils.logger import get_logger

logger = get_logger(__name__)


def rag_agent_node(state: AgentState) -> AgentState:
    question = state["user_question"]
    errors = list(state.get("errors", []))

    try:
        hits = vector_store.search(question)
    except EmbeddingUnavailableError as exc:
        logger.info("rag_agent_embedding_unavailable", extra={"extra_fields": {"error": str(exc)}})
        return {**state, "retrieved_documents": [], "errors": errors + [f"RAG agent could not run: {exc}"]}
    except Exception as exc:
        logger.info("rag_agent_error", extra={"extra_fields": {"error": str(exc)}})
        return {**state, "retrieved_documents": [], "errors": errors + [f"RAG agent error: {exc}"]}

    if not hits:
        errors = errors + ["No sufficiently relevant document passages were found for this question."]

    return {**state, "retrieved_documents": hits, "errors": errors}
