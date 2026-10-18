"""
Unit tests verifying Stage III Fine-Grained Exhaustive Attribution & Innocent Shielding.
Paper Reference: Section III-C, Equation 8.
Compatible with both `pytest` and standard library `python3 -m unittest discover -s tests`.
"""

import unittest

from src.attribution.exhaustive_attributor import ExhaustiveAttributor
from src.gating import ViolationInstance, ViolationType
from src.graph.hpg_model import DependencyEdge, EdgeType
from src.graph.incremental_operator import ASTDiffPayload


def test_attribution_culpable_and_innocent_separation() -> None:
    """Verify Equation 8: Correctly isolates the culpable patch while shielding the innocent patch."""
    attributor = ExhaustiveAttributor()

    edge_violating = DependencyEdge("src.presentation.View", "src.infrastructure.DB", EdgeType.INVOKE)

    # Patch 1: Added the violating cross-layer edge (Culpable)
    diff_p1 = ASTDiffPayload(
        patch_id="p1",
        agent_id="Agent_1",
        nodes_added=frozenset(),
        nodes_removed_ids=frozenset(),
        nodes_modified=frozenset(),
        edges_added=frozenset([edge_violating]),
        edges_removed=frozenset(),
        reclassified_node_ids=frozenset(),
        re_evaluated_edges=frozenset(),
    )

    # Patch 2: Added a safe domain internal edge (Innocent)
    safe_edge = DependencyEdge("src.domain.User", "src.domain.Address", EdgeType.AGGREGATE)
    diff_p2 = ASTDiffPayload(
        patch_id="p2",
        agent_id="Agent_2",
        nodes_added=frozenset(),
        nodes_removed_ids=frozenset(),
        nodes_modified=frozenset(),
        edges_added=frozenset([safe_edge]),
        edges_removed=frozenset(),
        reclassified_node_ids=frozenset(),
        re_evaluated_edges=frozenset(),
    )

    layer_v = ViolationInstance(
        type=ViolationType.LAYER,
        rule_id="NO-DIRECT-INFRASTRUCTURE-ACCESS",
        edges=frozenset([edge_violating]),
        nodes=frozenset(["src.presentation.View", "src.infrastructure.DB"]),
    )

    report = attributor.attribute_batch(
        violations=[layer_v],
        per_patch_diffs={"p1": diff_p1, "p2": diff_p2},
        agtd_profiles={},
        tau_batch=5.0,
    )

    assert "p1" in report.culpable_patch_ids
    assert "p2" in report.innocent_patch_ids
    assert len(report.innocent_patch_ids) == 1


class TestAttribution(unittest.TestCase):
    def test_culpable_and_innocent_separation(self) -> None:
        test_attribution_culpable_and_innocent_separation()


if __name__ == "__main__":
    unittest.main()
