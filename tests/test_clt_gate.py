"""
Unit tests verifying Stage II AgTD Metric & Variance-Scaled Statistical Gating.
Paper Reference: Section III-B, Equations 6 & 7.
Compatible with both `pytest` and standard library `python3 -m unittest discover -s tests`.
"""

import math
import unittest

from src.gating.agtd_calculator import AgTDCalculator
from src.gating.statistical_gate import StatisticalCLTGate
from src.graph.hpg_model import StaticMetricsPayload


def test_agtd_calculator_non_negative_clipping() -> None:
    """Verify Equation 6: Normalized complexity calculation preserves relative ordering."""
    calc = AgTDCalculator()
    base_m = StaticMetricsPayload(mccabe_complexity=3.0, cognitive_complexity=2.0, fan_out=2)
    hypo_improved = StaticMetricsPayload(mccabe_complexity=1.0, cognitive_complexity=1.0, fan_out=1)
    hypo_degraded = StaticMetricsPayload(mccabe_complexity=6.0, cognitive_complexity=5.0, fan_out=4)

    c_base = calc.compute_normalized_complexity(base_m)
    c_improved = calc.compute_normalized_complexity(hypo_improved)
    c_degraded = calc.compute_normalized_complexity(hypo_degraded)

    assert c_improved < c_base
    assert c_degraded > c_base
    assert max(0.0, c_improved - c_base) == 0.0
    assert max(0.0, c_degraded - c_base) > 0.0


def test_clt_dynamic_threshold_scaling() -> None:
    """Verify Equation 7: Threshold scales with sqrt(k) under explicit z_score and finite-sample t."""
    gate = StatisticalCLTGate(
        historical_mu=0.50,
        historical_sigma=0.30,
        z_score=1.96,
        scaling_mode="sqrt_n",
    )

    tau_n4 = gate.compute_dynamic_threshold(batch_size=4)
    expected_n4 = 4 * 0.50 + 1.96 * math.sqrt(4) * 0.30
    assert abs(tau_n4 - expected_n4) < 1e-3

    tau_n9 = gate.compute_dynamic_threshold(batch_size=9)
    expected_n9 = 9 * 0.50 + 1.96 * math.sqrt(9) * 0.30
    assert abs(tau_n9 - expected_n9) < 1e-3


def test_clt_gate_rejection_and_acceptance() -> None:
    """Verify that batch debt exceeding dynamic threshold is rejected."""
    gate = StatisticalCLTGate(
        historical_mu=0.50,
        historical_sigma=0.30,
        z_score=1.96,
        scaling_mode="sqrt_n",
    )

    pass_ok, _, _ = gate.evaluate(total_agtd_delta=0.80, batch_size=2)
    assert pass_ok is True

    pass_fail, _, violation = gate.evaluate(total_agtd_delta=3.50, batch_size=2)
    assert pass_fail is False
    assert violation is not None
    assert violation.excess_debt > 0.0


class TestCLTGate(unittest.TestCase):
    def test_non_negative_clipping(self) -> None:
        test_agtd_calculator_non_negative_clipping()

    def test_dynamic_threshold_scaling(self) -> None:
        test_clt_dynamic_threshold_scaling()

    def test_rejection_and_acceptance(self) -> None:
        test_clt_gate_rejection_and_acceptance()


if __name__ == "__main__":
    unittest.main()
