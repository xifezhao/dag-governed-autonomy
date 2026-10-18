"""
Unit tests verifying Stage I Aging Priority Queue & Starvation-Free Lower Bound Theorem.
Paper Reference: Section III-A, Equations 2 & 3, and Theorem 1.
Compatible with both `pytest` and standard library `python3 -m unittest discover -s tests`.
"""

import unittest

from src.arbitration.aging_priority_queue import AgingPriorityQueue, StarvationBoundViolation
from src.arbitration.conflict_detector import AgentPatch


def test_aging_priority_queue_dynamic_priority() -> None:
    """Verify Equation 2: Priority(ΔC_i) = BasePriority(A_i) + λ * DeferralCount."""
    pq = AgingPriorityQueue(
        lambda_factor=0.50,
        max_retries=10,
        min_base_priority=1.0,
        max_base_priority=5.0,
    )

    patch = AgentPatch(patch_id="p1", agent_id="Agent_A", raw_diff="--- a\n+++ b\n")
    pq.push(patch, base_priority=1.0, deferral_count=0)

    # After 4 deferrals, priority should be 1.0 + 0.5 * 4 = 3.0
    entry = pq.pop()
    entry = pq.defer_and_age(entry)  # count=1 -> priority=1.5
    entry = pq.defer_and_age(entry)  # count=2 -> priority=2.0
    entry = pq.defer_and_age(entry)  # count=3 -> priority=2.5
    entry = pq.defer_and_age(entry)  # count=4 -> priority=3.0

    assert entry.deferral_count == 4
    assert abs(entry.effective_priority - 3.0) < 1e-3


def test_starvation_bound_violation_assertion() -> None:
    """Verify Equation 3: StarvationBoundViolation raised when λ < Δ_max / (K - 1)."""
    # Δ_max = 5.0 - 1.0 = 4.0; K = 10; λ_min = 4.0 / 9 ≈ 0.4444
    raised = False
    try:
        AgingPriorityQueue(
            lambda_factor=0.30,  # Below 0.4444 -> mathematically unsafe!
            max_retries=10,
            min_base_priority=1.0,
            max_base_priority=5.0,
            enforce_strict_bound=True,
        )
    except StarvationBoundViolation:
        raised = True
    assert raised is True


def test_starvation_bound_satisfied_and_theorem1_bound() -> None:
    """Verify that λ >= λ_min initializes cleanly and computes Theorem 1 deferral bound (28)."""
    pq = AgingPriorityQueue(
        lambda_factor=0.50,  # 0.50 >= 0.4444 -> safe
        max_retries=10,
        min_base_priority=1.0,
        max_base_priority=5.0,
        enforce_strict_bound=True,
    )
    assert abs(pq.theoretical_minimum_lambda - (4.0 / 9.0)) < 1e-3
    assert pq.compute_theorem1_max_deferral_bound(max_conflict_cliques_m=3) == 28


def test_deterministic_tie_breaking() -> None:
    """Verify deterministic total-order tie-breaking when two patches have identical priority."""
    pq = AgingPriorityQueue(
        lambda_factor=0.50,
        max_retries=10,
        min_base_priority=1.0,
        max_base_priority=5.0,
    )

    patch_a = AgentPatch(patch_id="pA", agent_id="Agent_A", raw_diff="")
    patch_b = AgentPatch(patch_id="pB", agent_id="Agent_B", raw_diff="")

    pq.push(patch_a, base_priority=2.0, deferral_count=0)
    pq.push(patch_b, base_priority=2.0, deferral_count=0)

    first_out = pq.pop()
    second_out = pq.pop()

    assert first_out.patch.patch_id == "pA"
    assert second_out.patch.patch_id == "pB"


class TestAgingQueue(unittest.TestCase):
    def test_dynamic_priority(self) -> None:
        test_aging_priority_queue_dynamic_priority()

    def test_starvation_bound_violation(self) -> None:
        test_starvation_bound_violation_assertion()

    def test_starvation_bound_and_theorem1(self) -> None:
        test_starvation_bound_satisfied_and_theorem1_bound()

    def test_tie_breaking(self) -> None:
        test_deterministic_tie_breaking()


if __name__ == "__main__":
    unittest.main()
