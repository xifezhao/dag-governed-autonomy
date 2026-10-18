"""
================================================================================
Stage II: Variance-Scaled AgTD Gating Rule (r_debt, Equation 7)
================================================================================

Paper Reference:
  Section III-B: Architectural Constraint Calculus and Complexity Bounds
  Equation 7:
    r_debt(G') <=> ∑_{i=1}^k AgTD_i <= τ_batch(k)
    τ_batch(k) = k * μ_clean + t_{1-α, ν} * σ_comp(k)
    σ_comp(k) = sqrt( (k^2 * σ_clean^2 / N_hist) + k * σ_clean^2 * (1 + (k - 1) * ρ) )
  "Governed Autonomy: Halting Architectural Entropy in Agentic Software 
   Workflows via Dynamic Guardrails" (SANER 2026)

Standards:
  Compliant with IEEE/ACM Artifact Evaluation Guidelines (Badge: Reusable)
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Any, Final, Sequence

logger = logging.getLogger("dag.gating.statistical_gate")


@dataclass(frozen=True, slots=True)
class AgTDSpikeViolation:
    """
    Structured diagnostic report emitted when the aggregate batch debt
    exceeds the variance-scaled dynamic threshold (Eq. 7).
    """
    total_agtd_delta: float         # Observed cumulative debt: ∑_i AgTD_i
    dynamic_threshold_tau: float    # Computed statistical cap: τ_batch(k)
    excess_debt: float              # Net overflow: total_delta - tau_batch
    batch_size: int                 # Number of concurrent orthogonal patches (k)
    z_score: float                  # Critical quantile t_{1-α, ν} or z
    historical_mu: float            # Unit-commit empirical mean from H_clean
    historical_sigma: float         # Unit-commit empirical standard deviation
    standard_error: float           # Compound standard error σ_comp(k)

    def __repr__(self) -> str:
        return (
            f"<AgTDSpikeViolation: Observed=+{self.total_agtd_delta:.4f} > "
            f"Threshold τ={self.dynamic_threshold_tau:.4f} (Excess=+{self.excess_debt:.4f}) "
            f"across k={self.batch_size} agents (quantile={self.z_score:.3f})>"
        )


class StatisticalCLTGate:
    """
    Implementation of Variance-Scaled Gating (VSG) Rule (r_debt, Equation 7):
        r_debt(G') <=> ∑_{i=1}^k AgTD_i <= k * μ_clean + t_{1-α, ν} * σ_comp(k)
    """

    def __init__(
        self,
        historical_mu: float = 0.42,
        historical_sigma: float = 0.85,
        alpha: float = 0.05,
        inter_agent_rho: float = 0.15,
        n_hist: int = 500,
        scaling_mode: str = "sqrt_n",
        z_score: float | None = None,
        **kwargs: Any,
    ) -> None:
        """
        Parameters:
            historical_mu: Expected mean AgTD delta per human commit derived from H_clean (0.42).
            historical_sigma: Standard deviation of AgTD delta derived from H_clean (0.85).
            alpha: Single-tailed significance level (default 0.05 for 95% upper prediction bound).
            inter_agent_rho: Intra-batch correlation coefficient bound between concurrent agents (ρ = 0.15).
            n_hist: Size of historical benchmark corpus H_clean (N_hist = 500, ν = 499).
            scaling_mode: "sqrt_n" (Variance-scaled finite-sample) or "linear_n" (Ablation baseline).
            z_score: Optional explicit quantile override (when provided, uses pure CLT z*sqrt(k)*σ).
        """
        if historical_sigma <= 0:
            raise ValueError(f"historical_sigma must be strictly positive (got {historical_sigma}).")
        if not (0 < alpha < 0.5):
            raise ValueError(f"alpha must be in (0, 0.5) for upper-tail gating (got {alpha}).")

        self.mu: Final[float] = float(historical_mu)
        self.sigma: Final[float] = float(historical_sigma)
        self.alpha: Final[float] = float(alpha)
        self.rho: Final[float] = float(inter_agent_rho)
        self.n_hist: Final[int] = int(n_hist)
        self.nu: Final[int] = max(1, self.n_hist - 1)
        self.scaling_mode: Final[str] = scaling_mode.lower()
        self.explicit_z: Final[float | None] = float(z_score) if z_score is not None else None
        # Student-t quantile t_{0.95, 499} = 1.6479 (~1.648) for α = 0.05, ν = 499
        self.t_quantile: Final[float] = 1.6479 if abs(self.alpha - 0.05) < 1e-6 else 1.960
        self.z: Final[float] = self.explicit_z if self.explicit_z is not None else self.t_quantile

        logger.debug(
            "VarianceScaledGate initialized (μ=%.4f, σ=%.4f, α=%.2f, ν=%d, t=%.4f, ρ=%.2f, mode=%s)",
            self.mu,
            self.sigma,
            self.alpha,
            self.nu,
            self.z,
            self.rho,
            self.scaling_mode,
        )

    def compute_compound_se(self, batch_size: int, inter_agent_rho: float | None = None) -> float:
        """
        Computes σ_comp(k) per Equation 7:
            σ_comp(k) = sqrt( (k^2 * σ^2 / N_hist) + k * σ^2 * (1 + (k - 1) * ρ) )
        """
        if batch_size <= 0:
            return 0.0
        k = float(batch_size)
        if self.explicit_z is not None:
            # Pure asymptotic CLT mode when explicit z_score is requested
            return math.sqrt(k) * self.sigma
        rho = self.rho if inter_agent_rho is None else float(inter_agent_rho)
        hist_var = (k ** 2) * (self.sigma ** 2) / float(self.n_hist)
        batch_var = k * (self.sigma ** 2) * (1.0 + max(0.0, k - 1.0) * rho)
        return math.sqrt(hist_var + batch_var)

    def compute_dynamic_threshold(self, batch_size: int, inter_agent_rho: float | None = None) -> float:
        """
        Computes the dynamic gating boundary τ_batch(k) for concurrency level k (Equation 7).
        """
        if batch_size <= 0:
            return 0.0

        k = float(batch_size)
        critical_val = self.z

        if self.scaling_mode == "sqrt_n":
            compound_se = self.compute_compound_se(batch_size, inter_agent_rho=inter_agent_rho)
            tau_batch = (k * self.mu) + (critical_val * compound_se)
        elif self.scaling_mode == "linear_n":
            # Naive Linear Baseline (Evaluated in Ablation Studies: w/o-VarianceScaling)
            tau_batch = k * (self.mu + (critical_val * self.sigma))
        else:
            raise ValueError(f"Unknown statistical scaling mode: '{self.scaling_mode}'")

        return tau_batch

    def evaluate(
        self,
        total_agtd_delta: float,
        batch_size: int,
    ) -> tuple[bool, float, AgTDSpikeViolation | None]:
        """
        Evaluates Equation 7 over the aggregated batch debt.
        """
        if batch_size <= 0:
            return True, 0.0, None

        tau_batch = self.compute_dynamic_threshold(batch_size)
        passed = total_agtd_delta <= tau_batch

        if passed:
            logger.debug(
                "Statistical Gating PASSED (Eq. 7): Debt=+%.4f <= Threshold τ=%.4f (k=%d)",
                total_agtd_delta,
                tau_batch,
                batch_size,
            )
            return True, tau_batch, None

        std_err = (
            self.compute_compound_se(batch_size)
            if self.scaling_mode == "sqrt_n"
            else float(batch_size) * self.sigma
        )
        excess = total_agtd_delta - tau_batch

        violation = AgTDSpikeViolation(
            total_agtd_delta=total_agtd_delta,
            dynamic_threshold_tau=tau_batch,
            excess_debt=excess,
            batch_size=batch_size,
            z_score=self.z,
            historical_mu=self.mu,
            historical_sigma=self.sigma,
            standard_error=std_err,
        )

        logger.warning(
            "Statistical Gating REJECTED (Eq. 7): Debt=+%.4f > Threshold τ=%.4f across k=%d agents. Excess=+%.4f",
            total_agtd_delta,
            tau_batch,
            batch_size,
            excess,
        )

        return False, tau_batch, violation

    def evaluate_heldout_human_batches(
        self, human_batches: Sequence[dict[str, Any]]
    ) -> dict[str, Any]:
        """
        Empirically calibrates the false-rejection rate of r_debt (Eq. 7) on held-out
        clean human multi-commit batches from H_clean against nominal α = 0.05
        (Paper Section III-B & Section V-A2).
        """
        if not human_batches:
            return {"total_batches": 0, "rejected_batches": 0, "false_rejection_rate": 0.0}

        rejected = 0
        per_k_stats: dict[int, dict[str, int]] = {}
        for batch in human_batches:
            k = int(batch.get("batch_size", len(batch.get("commit_deltas", []))))
            total_delta = float(batch.get("total_agtd_delta", sum(batch.get("commit_deltas", []))))
            passed, _, _ = self.evaluate(total_agtd_delta=total_delta, batch_size=k)
            if k not in per_k_stats:
                per_k_stats[k] = {"total": 0, "rejected": 0}
            per_k_stats[k]["total"] += 1
            if not passed:
                rejected += 1
                per_k_stats[k]["rejected"] += 1

        fpr = float(rejected) / float(len(human_batches))
        return {
            "total_batches": len(human_batches),
            "rejected_batches": rejected,
            "false_rejection_rate": round(fpr, 4),
            "nominal_alpha": self.alpha,
            "per_k_breakdown": per_k_stats,
        }


# Formal alias matching Paper Section III-B terminology
VarianceScaledGate = StatisticalCLTGate
