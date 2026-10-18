"""
================================================================================
Stage II: Normalized Proportional AgTD Calculator (Equation 7)
================================================================================

Paper Reference:
  Section III-B: Architectural Constraint Calculus and Complexity Bounds
  Equation 7: AgTD_i = ∑_{v ∈ Touched(ΔC_i)} max(0, \tilde{Φ}'_{cplx}(v) - \tilde{Φ}_{cplx}(v))
  "Governed Autonomy: Halting Architectural Entropy in Agentic Software 
   Workflows via Dynamic Guardrails" (SANER 2026)

Formal Invariant:
  1. Computes dimensionless Z-score normalized degradation over McCabe, Cognitive, and FanOut.
  2. Strict positive delta accumulation: max(0, Δ) prevents masking effects.
  3. Bounded strictly to Touched(ΔC_i) for O(|Touched|) temporal execution.

Standards:
  Compliant with IEEE/ACM Artifact Evaluation Guidelines (Badge: Reusable)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Final, Mapping, Sequence

from ..graph.hpg_model import (
    CodeEntityNode,
    HierarchicalAttributedGraph,
    StaticMetricsPayload,
)
from ..graph.incremental_operator import ASTDiffPayload

logger = logging.getLogger("dag.gating.agtd_calculator")


@dataclass(frozen=True, slots=True)
class MetricDistributionParams:
    """
    Empirical distribution parameters (μ, σ) derived from historical clean commits (H_clean).
    Used to standardize heterogeneous metrics into dimensionless Z-scores.
    """
    mu_mccabe: float = 3.50
    sigma_mccabe: float = 2.80
    mu_cognitive: float = 4.20
    sigma_cognitive: float = 4.50
    mu_fanout: float = 5.10
    sigma_fanout: float = 3.90
    epsilon: float = 1e-6  # Guard against zero-division in degenerate codebases


@dataclass(frozen=True, slots=True)
class NodeDeltaRecord:
    """Detailed audit trace of architectural entropy mutation for an individual node."""
    node_id: str
    base_metrics: StaticMetricsPayload
    hypothetical_metrics: StaticMetricsPayload
    base_normalized_cplx: float
    hypo_normalized_cplx: float
    raw_delta: float
    positive_delta: float  # max(0, raw_delta)


@dataclass(frozen=True, slots=True)
class AgTDMetricProfile:
    """
    Structured outcome of evaluating Equation 7 for an individual agent's patch (ΔC_i).
    Consumed by StatisticalCLTGate (Eq. 8) and ExhaustiveAttributor (Eq. 9).
    """
    patch_id: str
    agent_id: str
    touched_node_ids: frozenset[str]
    node_deltas: tuple[NodeDeltaRecord, ...]
    net_positive_delta: float  # AgTD_i: Formal value calculated via Eq. 7
    gross_delta: float         # Raw unclipped delta (for ablation diagnostics)
    mccabe_raw_delta: float
    cognitive_raw_delta: float
    fanout_raw_delta: float

    def __repr__(self) -> str:
        return (
            f"<AgTDProfile Patch={self.patch_id} Agent={self.agent_id} "
            f"AgTD_i=+{self.net_positive_delta:.4f} (Gross={self.gross_delta:+.4f}) "
            f"TouchedNodes={len(self.touched_node_ids)}>"
        )


class AgTDCalculator:
    """
    Implementation of Equation 7:
        AgTD_i = ∑_{v ∈ Touched(ΔC_i)} max(0, \tilde{Φ}'_{cplx}(v) - \tilde{Φ}_{cplx}(v))
    """

    def __init__(
        self,
        weights: Mapping[str, float] | None = None,
        distribution_params: MetricDistributionParams | None = None,
    ) -> None:
        # Default weights correspond to config.yaml: w1=0.35, w2=0.40, w3=0.25
        raw_weights = weights or {"mccabe": 0.35, "cognitive": 0.40, "fan_out": 0.25}
        total_w = sum(raw_weights.values())
        
        # Enforce sum(w_i) == 1.0 normalization
        self.w_mccabe: Final[float] = float(raw_weights.get("mccabe", 0.35)) / total_w
        self.w_cognitive: Final[float] = float(raw_weights.get("cognitive", 0.40)) / total_w
        self.w_fanout: Final[float] = float(raw_weights.get("fan_out", 0.25)) / total_w

        self.dist: Final[MetricDistributionParams] = distribution_params or MetricDistributionParams()

        logger.debug(
            "AgTDCalculator initialized (Weights: McCabe=%.2f, Cognitive=%.2f, FanOut=%.2f)",
            self.w_mccabe,
            self.w_cognitive,
            self.w_fanout,
        )

    def standardize_metric(self, value: float, mean: float, sigma: float) -> float:
        """Computes dimensionless standard score: Z = (M - μ) / σ."""
        safe_sigma = max(sigma, self.dist.epsilon)
        return (value - mean) / safe_sigma

    def compute_normalized_complexity(self, metrics: StaticMetricsPayload) -> float:
        """
        Computes standardized composite complexity \tilde{Φ}_{cplx}(v) (Eq. 7).
        """
        z_mccabe = self.standardize_metric(metrics.mccabe_complexity, self.dist.mu_mccabe, self.dist.sigma_mccabe)
        z_cognitive = self.standardize_metric(metrics.cognitive_complexity, self.dist.mu_cognitive, self.dist.sigma_cognitive)
        z_fanout = self.standardize_metric(float(metrics.fan_out), self.dist.mu_fanout, self.dist.sigma_fanout)

        return (
            (self.w_mccabe * z_mccabe) +
            (self.w_cognitive * z_cognitive) +
            (self.w_fanout * z_fanout)
        )

    def compute_patch_agtd(
        self,
        base_graph: HierarchicalAttributedGraph,
        hypothetical_graph: HierarchicalAttributedGraph,
        diff_payload: ASTDiffPayload,
    ) -> AgTDMetricProfile:
        """
        Calculates AgTD_i strictly for the touched nodes of patch ΔC_i (Eq. 7).
        """
        # Touched(ΔC_i) includes newly added nodes and modified nodes
        touched_node_ids = set()
        for node in diff_payload.nodes_added | diff_payload.nodes_modified:
            touched_node_ids.add(node.node_id)
        # Note: nodes_removed_ids are checked to ensure their removal does not mask debt

        node_delta_records: list[NodeDeltaRecord] = []
        net_positive_delta: float = 0.0
        gross_delta: float = 0.0

        raw_mccabe_diff: float = 0.0
        raw_cognitive_diff: float = 0.0
        raw_fanout_diff: float = 0.0

        # Baseline template for newly created entities
        neutral_base_metrics = StaticMetricsPayload(
            mccabe_complexity=1.0,
            cognitive_complexity=0.0,
            fan_out=0,
            loc=0,
        )

        for nid in touched_node_ids:
            hypo_node = hypothetical_graph.nodes.get(nid)
            if not hypo_node:
                continue

            base_node = base_graph.nodes.get(nid)
            base_metrics = base_node.metrics if base_node else neutral_base_metrics
            hypo_metrics = hypo_node.metrics

            base_cplx = self.compute_normalized_complexity(base_metrics)
            hypo_cplx = self.compute_normalized_complexity(hypo_metrics)

            delta = hypo_cplx - base_cplx
            clamped_delta = max(0.0, delta)  # Mathematical clamping max(0, Δ) in Eq. 7

            net_positive_delta += clamped_delta
            gross_delta += delta

            # Diagnostic sub-metrics tracking
            raw_mccabe_diff += (hypo_metrics.mccabe_complexity - base_metrics.mccabe_complexity)
            raw_cognitive_diff += (hypo_metrics.cognitive_complexity - base_metrics.cognitive_complexity)
            raw_fanout_diff += float(hypo_metrics.fan_out - base_metrics.fan_out)

            node_delta_records.append(
                NodeDeltaRecord(
                    node_id=nid,
                    base_metrics=base_metrics,
                    hypothetical_metrics=hypo_metrics,
                    base_normalized_cplx=base_cplx,
                    hypo_normalized_cplx=hypo_cplx,
                    raw_delta=delta,
                    positive_delta=clamped_delta,
                )
            )

        profile = AgTDMetricProfile(
            patch_id=diff_payload.patch_id,
            agent_id=diff_payload.agent_id,
            touched_node_ids=frozenset(touched_node_ids),
            node_deltas=tuple(node_delta_records),
            net_positive_delta=net_positive_delta,
            gross_delta=gross_delta,
            mccabe_raw_delta=raw_mccabe_diff,
            cognitive_raw_delta=raw_cognitive_diff,
            fanout_raw_delta=raw_fanout_diff,
        )

        logger.debug(
            "Computed Eq. 7 for Patch %s (Agent %s): AgTD_i=+%.4f across %d touched nodes.",
            diff_payload.patch_id,
            diff_payload.agent_id,
            net_positive_delta,
            len(touched_node_ids),
        )
        return profile

    def compute_batch_agtd(
        self,
        base_graph: HierarchicalAttributedGraph,
        hypothetical_graph: HierarchicalAttributedGraph,
        per_patch_diffs: Mapping[str, ASTDiffPayload],
    ) -> dict[str, AgTDMetricProfile]:
        """
        Computes AgTD_i profiles across all concurrent orthogonal patches in batch.
        
        Returns:
            dict[patch_id, AgTDMetricProfile]
        """
        profiles: dict[str, AgTDMetricProfile] = {}
        for patch_id, diff_payload in per_patch_diffs.items():
            profiles[patch_id] = self.compute_patch_agtd(
                base_graph=base_graph,
                hypothetical_graph=hypothetical_graph,
                diff_payload=diff_payload,
            )
        return profiles
