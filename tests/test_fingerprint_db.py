"""
Unit tests verifying Stage III Topological State Fingerprint DB & Anti-Oscillation Detection.
Paper Reference: Section III-C, Equation 9 & Algorithm 1 (DetectCycle).
Compatible with both `pytest` and standard library `python3 -m unittest discover -s tests`.
"""

import unittest

from src.attribution.fingerprint_db import CryptographicFingerprintDB, TopologicalFingerprintDB
from src.gating import ViolationInstance, ViolationType
from src.graph.hpg_model import DependencyEdge, EdgeType


def test_canonical_hash_rotation_invariance() -> None:
    """Verify Equation 9: Hashing is invariant to cyclic graph traversal entry points."""
    db = TopologicalFingerprintDB()

    edges_from_a = frozenset([
        DependencyEdge("A", "B", EdgeType.INVOKE),
        DependencyEdge("B", "C", EdgeType.INVOKE),
        DependencyEdge("C", "A", EdgeType.INVOKE),
    ])
    v1 = ViolationInstance(
        type=ViolationType.CYCLE,
        rule_id="CYCLE",
        edges=edges_from_a,
        nodes=frozenset(["A", "B", "C"]),
    )

    edges_from_c = frozenset([
        DependencyEdge("C", "A", EdgeType.INVOKE),
        DependencyEdge("A", "B", EdgeType.INVOKE),
        DependencyEdge("B", "C", EdgeType.INVOKE),
    ])
    v2 = ViolationInstance(
        type=ViolationType.CYCLE,
        rule_id="CYCLE",
        edges=edges_from_c,
        nodes=frozenset(["C", "A", "B"]),
    )

    hash1 = db.compute_fingerprint([v1])
    hash2 = db.compute_fingerprint([v2])

    assert hash1 == hash2


def test_livelock_oscillation_detection() -> None:
    """Verify that visiting a non-adjacent recurring topological state (X -> Y -> X) flags livelock."""
    db = CryptographicFingerprintDB()

    edge_ab = DependencyEdge("ModuleA", "ModuleB", EdgeType.INVOKE)
    edge_ba = DependencyEdge("ModuleB", "ModuleA", EdgeType.INVOKE)
    v = ViolationInstance(
        type=ViolationType.CYCLE,
        rule_id="CYCLE",
        edges=frozenset([edge_ab, edge_ba]),
        nodes=frozenset(["ModuleA", "ModuleB"]),
    )

    # Iteration 1: First visit (State X) -> Not oscillating
    is_osc1, rec1 = db.record_and_check(agent_id="Agent_1", iteration=1, violations=[v])
    assert is_osc1 is False

    # Iteration 2: Different state (State Y) -> Not oscillating
    edge_ac = DependencyEdge("ModuleA", "ModuleC", EdgeType.INVOKE)
    edge_ca = DependencyEdge("ModuleC", "ModuleA", EdgeType.INVOKE)
    v_alt = ViolationInstance(
        type=ViolationType.CYCLE,
        rule_id="CYCLE",
        edges=frozenset([edge_ac, edge_ca]),
        nodes=frozenset(["ModuleA", "ModuleC"]),
    )
    is_osc2, rec2 = db.record_and_check(agent_id="Agent_1", iteration=2, violations=[v_alt])
    assert is_osc2 is False

    # Iteration 3: Returns to State X (X -> Y -> X) -> Livelock Detected!
    is_osc3, rec3 = db.record_and_check(agent_id="Agent_1", iteration=3, violations=[v])
    assert is_osc3 is True
    assert rec3.state_hash == rec1.state_hash


class TestFingerprintDB(unittest.TestCase):
    def test_rotation_invariance(self) -> None:
        test_canonical_hash_rotation_invariance()

    def test_oscillation_detection(self) -> None:
        test_livelock_oscillation_detection()


if __name__ == "__main__":
    unittest.main()
