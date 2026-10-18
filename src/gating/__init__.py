"""
================================================================================
Stage II: Architectural Constraint Calculus & Multi-Gate Gating Engine
================================================================================

Paper Reference:
  Section III-B: Architectural Constraint Calculus and Complexity Bounds
  "Governed Autonomy: Halting Architectural Entropy in Agentic Software 
   Workflows via Dynamic Guardrails" (SANER 2026)

Formal Implementation Mapping:
--------------------------------------------------------------------------------
Predicate | Equation | Architectural Invariant               | Implementation Class
--------------------------------------------------------------------------------
r_cycle   | Eq. 4    | Whitelisted SCC Non-Trivial Cycles    | WhitelistedCycleGate
r_layer   | Eq. 5    | Mined Forbidden Dependency Bounds     | EmpiricalLayerGate
r_debt    | Eq. 6    | Normalized Touched Node AgTD Delta    | AgTDCalculator
r_debt    | Eq. 7    | Variance-Scaled Batch Gating          | StatisticalCLTGate
--------------------------------------------------------------------------------

Licensing:
  Licensed under the Apache License, Version 2.0 (the "License").
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Final

from .cycle_gate import WhitelistedCycleGate, CycleViolation
from .layer_gate import EmpiricalLayerGate, LayerViolation
from .agtd_calculator import AgTDCalculator, AgTDMetricProfile
from .statistical_gate import StatisticalCLTGate, VarianceScaledGate, AgTDSpikeViolation

from ..graph.hpg_model import DependencyEdge, HierarchicalAttributedGraph
from ..graph.incremental_operator import IncrementalUpdateResult


class ViolationType(str, Enum):
    """Enumeration of architectural violation categories."""
    CYCLE = "cycle"
    LAYER = "layer"
    AGTD_SPIKE = "AgTD_Spike"


@dataclass(frozen=True, slots=True)
class ViolationInstance:
    r"""
    Formal representation of a single violation instance v \in \mathcal{V}(G').
    Directly consumed by Equation 8 (Attr(v)) and Equation 9 (State Fingerprinting).
    """
    type: ViolationType
    rule_id: str
    edges: frozenset[DependencyEdge] = field(default_factory=frozenset)  # Offending edges
    nodes: frozenset[str] = field(default_factory=frozenset)             # Incident node IDs
    signature: str = ""                                                  # Deterministic identifier for hashing
    context: dict[str, Any] = field(default_factory=dict, hash=False)   # Diagnostic payload for Templated Nudging

    def __post_init__(self) -> None:
        if not self.signature:
            edge_repr = ";".join(sorted(f"{e.source_id}->{e.target_id}" for e in self.edges))
            node_repr = ";".join(sorted(self.nodes))
            object.__setattr__(
                self,
                "signature",
                f"[{self.type.value}]:{self.rule_id}:edges({edge_repr}):nodes({node_repr})",
            )

    def __repr__(self) -> str:
        return f"<ViolationInstance [{self.type.value}] Rule={self.rule_id} Sig={self.signature}>"


@dataclass(frozen=True, slots=True)
class GatingEvaluationResult:
    """
    Encapsulates the complete outcome of evaluating R over hypothetical graph G'.
    """
    is_safe: bool                                  # True iff V(G') == ∅
    triggered_rules: frozenset[str]                # \mathcal{K}(G') = {r_k | ¬r_k(G')}
    violations: tuple[ViolationInstance, ...]      # \mathcal{V}(G')
    total_agtd_delta: float                        # Aggregate sum of AgTD_i across orthogonal batch
    dynamic_threshold_tau: float                   # \tau_{batch} calculated via Eq. 7
    cycle_violations_count: int
    layer_violations_count: int
    agtd_violation_present: bool
    agtd_profiles: dict[str, AgTDMetricProfile] = field(default_factory=dict, hash=False)

    def __bool__(self) -> bool:
        return self.is_safe


class CompositeGatingPipeline:
    """
    Unified evaluator running all architectural constraint gates sequentially.
    """

    def __init__(
        self,
        cycle_gate: WhitelistedCycleGate,
        layer_gate: EmpiricalLayerGate,
        agtd_calculator: AgTDCalculator,
        statistical_gate: StatisticalCLTGate,
    ) -> None:
        self.cycle_gate = cycle_gate
        self.layer_gate = layer_gate
        self.agtd_calculator = agtd_calculator
        self.statistical_gate = statistical_gate

    def evaluate(
        self,
        base_graph: HierarchicalAttributedGraph,
        update_result: IncrementalUpdateResult,
    ) -> GatingEvaluationResult:
        """
        Executes structural and statistical constraint gating over G'.
        Filters out pre-existing cycle violations already present in base_graph G.
        """
        hypothetical_g = update_result.hypothetical_graph
        aggregated_diff = update_result.aggregated_diff
        orthogonal_patches_count = len(update_result.per_patch_diffs)

        violations: list[ViolationInstance] = []
        triggered_rules: set[str] = set()

        # 1. Evaluate r_cycle (Eq. 4), ignoring pre-existing cycles in base_graph G
        _, base_cycle_violations = self.cycle_gate.evaluate(base_graph)
        pre_existing_cycle_nodes = {cv.component_node_ids for cv in base_cycle_violations}

        cycle_passed, raw_cycle_violations = self.cycle_gate.evaluate(hypothetical_g)
        new_cycle_violations = [
            cv for cv in raw_cycle_violations if cv.component_node_ids not in pre_existing_cycle_nodes
        ]
        if new_cycle_violations:
            triggered_rules.add("r_cycle")
            for cv in new_cycle_violations:
                violations.append(
                    ViolationInstance(
                        type=ViolationType.CYCLE,
                        rule_id="r_cycle",
                        edges=cv.offending_edges,
                        nodes=cv.component_node_ids,
                        context={"scc_size": len(cv.component_node_ids)},
                    )
                )

        # 2. Evaluate r_layer (Eq. 5)
        layer_passed, raw_layer_violations = self.layer_gate.evaluate(
            hypothetical_g,
            added_edges=aggregated_diff.edges_added,
            re_evaluated_edges=aggregated_diff.re_evaluated_edges,
        )
        if not layer_passed:
            triggered_rules.add("r_layer")
            for lv in raw_layer_violations:
                violations.append(
                    ViolationInstance(
                        type=ViolationType.LAYER,
                        rule_id="r_layer",
                        edges=frozenset([lv.offending_edge]),
                        nodes=frozenset([lv.offending_edge.source_id, lv.offending_edge.target_id]),
                        context={
                            "source_tier": lv.source_tier,
                            "target_tier": lv.target_tier,
                            "forbidden_rule": lv.forbidden_rule,
                        },
                    )
                )

        # 3. Compute Per-Agent AgTD_i and Batch Aggregate (Eq. 6 & 7)
        agtd_profiles = self.agtd_calculator.compute_batch_agtd(
            base_graph=base_graph,
            hypothetical_graph=hypothetical_g,
            per_patch_diffs=update_result.per_patch_diffs,
        )
        total_agtd_delta = sum(profile.net_positive_delta for profile in agtd_profiles.values())

        # 4. Evaluate r_debt with Variance-Scaled Gating (Eq. 7)
        debt_passed, tau_batch, raw_debt_violation = self.statistical_gate.evaluate(
            total_agtd_delta=total_agtd_delta,
            batch_size=orthogonal_patches_count,
        )
        if not debt_passed and raw_debt_violation:
            triggered_rules.add("r_debt")
            violations.append(
                ViolationInstance(
                    type=ViolationType.AGTD_SPIKE,
                    rule_id="r_debt",
                    nodes=update_result.touched_node_ids,
                    context={
                        "total_delta": total_agtd_delta,
                        "tau_batch": tau_batch,
                        "batch_size": orthogonal_patches_count,
                        "per_agent_contributions": {
                            pid: prof.net_positive_delta for pid, prof in agtd_profiles.items()
                        },
                    },
                )
            )

        is_safe = len(violations) == 0

        return GatingEvaluationResult(
            is_safe=is_safe,
            triggered_rules=frozenset(triggered_rules),
            violations=tuple(violations),
            total_agtd_delta=total_agtd_delta,
            dynamic_threshold_tau=tau_batch,
            cycle_violations_count=len(new_cycle_violations),
            layer_violations_count=len(raw_layer_violations) if not layer_passed else 0,
            agtd_violation_present=not debt_passed,
            agtd_profiles=agtd_profiles,
        )


__all__: Final[list[str]] = [
    "WhitelistedCycleGate",
    "CycleViolation",
    "EmpiricalLayerGate",
    "LayerViolation",
    "AgTDCalculator",
    "AgTDMetricProfile",
    "StatisticalCLTGate",
    "VarianceScaledGate",
    "AgTDSpikeViolation",
    "ViolationType",
    "ViolationInstance",
    "GatingEvaluationResult",
    "CompositeGatingPipeline",
]
