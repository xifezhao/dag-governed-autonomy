"""
Unit tests verifying Stage II Cycle Gating Rule & Whitelist Pattern Matching.
Paper Reference: Section III-B, Equation 5.
Compatible with both `pytest` and standard library `python3 -m unittest discover -s tests`.
"""

import unittest

from src.gating.cycle_gate import WhitelistedCycleGate
from src.graph.hpg_model import (
    CodeEntityNode,
    DependencyEdge,
    EdgeType,
    GranularityLayer,
    HierarchicalAttributedGraph,
)


def test_acyclic_graph_passes() -> None:
    """Verify that a valid DAG without cycles passes the cycle gate."""
    g = HierarchicalAttributedGraph(name="AcyclicRepo")
    n1 = CodeEntityNode(node_id="n1", name="N1", layer=GranularityLayer.CLASS, file_path="f1.py", architectural_tier="Presentation")
    n2 = CodeEntityNode(node_id="n2", name="N2", layer=GranularityLayer.CLASS, file_path="f2.py", architectural_tier="Domain")
    g.add_node(n1)
    g.add_node(n2)
    g.add_edge(DependencyEdge(source_id="n1", target_id="n2", edge_type=EdgeType.INVOKE))

    gate = WhitelistedCycleGate()
    passed, violations = gate.evaluate(g)

    assert passed is True
    assert len(violations) == 0


def test_cyclic_graph_detected() -> None:
    """Verify Equation 5: Detects non-trivial strongly connected components (|C| > 1)."""
    g = HierarchicalAttributedGraph(name="CyclicRepo")
    n1 = CodeEntityNode(node_id="n1", name="N1", layer=GranularityLayer.CLASS, file_path="f1.py", architectural_tier="Domain")
    n2 = CodeEntityNode(node_id="n2", name="N2", layer=GranularityLayer.CLASS, file_path="f2.py", architectural_tier="Domain")
    g.add_node(n1)
    g.add_node(n2)
    g.add_edge(DependencyEdge(source_id="n1", target_id="n2", edge_type=EdgeType.INVOKE))
    g.add_edge(DependencyEdge(source_id="n2", target_id="n1", edge_type=EdgeType.INVOKE))

    gate = WhitelistedCycleGate(whitelist_config_path="config/arch_whitelist.yaml")
    passed, violations = gate.evaluate(g)

    assert passed is False
    assert len(violations) == 1
    assert violations[0].component_node_ids == frozenset({"n1", "n2"})


def test_cross_layer_cycle_anti_gaming_block() -> None:
    """Verify that cycles crossing tier boundaries are strictly blocked by anti-gaming rules."""
    g = HierarchicalAttributedGraph(name="CrossTierCycle")
    pres = CodeEntityNode(node_id="pres_view", name="PresentationView", layer=GranularityLayer.CLASS, file_path="view.py", architectural_tier="Presentation")
    infra = CodeEntityNode(node_id="infra_dao", name="InfrastructureDAO", layer=GranularityLayer.CLASS, file_path="dao.py", architectural_tier="Infrastructure")
    g.add_node(pres)
    g.add_node(infra)
    g.add_edge(DependencyEdge(source_id="pres_view", target_id="infra_dao", edge_type=EdgeType.INVOKE))
    g.add_edge(DependencyEdge(source_id="infra_dao", target_id="pres_view", edge_type=EdgeType.INVOKE))

    gate = WhitelistedCycleGate(whitelist_config_path="config/arch_whitelist.yaml")
    passed, violations = gate.evaluate(g)

    assert passed is False
    assert len(violations) >= 1
    assert "Anti-Gaming" in violations[0].rejection_reason


class TestCycleGate(unittest.TestCase):
    def test_acyclic_passes(self) -> None:
        test_acyclic_graph_passes()

    def test_cyclic_detected(self) -> None:
        test_cyclic_graph_detected()

    def test_cross_layer_cycle_blocked(self) -> None:
        test_cross_layer_cycle_anti_gaming_block()


if __name__ == "__main__":
    unittest.main()
