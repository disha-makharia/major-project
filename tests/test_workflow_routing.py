from backend.graph.workflow import _route_after_data


def test_routes_to_both_branches():
    targets = _route_after_data({"needs_sql": True, "needs_rag": True})
    assert set(targets) == {"sql_agent", "rag_agent"}


def test_routes_to_sql_only():
    assert _route_after_data({"needs_sql": True, "needs_rag": False}) == ["sql_agent"]


def test_routes_to_rag_only():
    assert _route_after_data({"needs_sql": False, "needs_rag": True}) == ["rag_agent"]


def test_routes_to_analysis_when_neither_needed():
    assert _route_after_data({"needs_sql": False, "needs_rag": False}) == ["analysis_agent"]


def test_graph_compiles():
    from backend.graph.workflow import build_graph

    graph = build_graph()
    assert graph is not None
