"""
================================================================================
Stage II: Empirical Layer Violation Gate (r_layer, Equation 6)
================================================================================

Paper Reference:
  Section III-B: Architectural Constraint Calculus and Complexity Bounds
  Equation 6: r_layer(G') <=> ∀ (u, v) ∈ E^+ ∪ E_{re-eval}, v ∉ B(u)
  "Governed Autonomy: Halting Architectural Entropy in Agentic Software 
   Workflows via Dynamic Guardrails" (SANER 2026)

Formal Invariant:
  No dependency edge in the set of newly introduced edges (E^+) or re-evaluated
  incident edges of reclassified nodes (E_{re-eval}) may target an entity in the
  forbidden dependency mapping B(u) mined via Association Rule Mining (ARM).

Design Distinction:
  Explicitly designated as an Empirical Lower Bound (ELB). Operates in strictly
  local O(|E^+| + |E_{re-eval}|) time complexity.

Standards:
  Compliant with IEEE/ACM Artifact Evaluation Guidelines (Badge: Reusable)
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final, Iterable, Sequence

from ..graph.hpg_model import (
    CodeEntityNode,
    DependencyEdge,
    HierarchicalAttributedGraph,
)

logger = logging.getLogger("dag.gating.layer_gate")


@dataclass(frozen=True, slots=True)
class LayerViolation:
    """
    Structured descriptor for an unauthorized cross-layer dependency edge (Eq. 6).
    Directly consumed by CompositeGatingPipeline to construct ViolationInstance.
    """
    offending_edge: DependencyEdge
    source_tier: str
    target_tier: str
    forbidden_rule: str
    rejection_reason: str
    is_reclassification_violation: bool = False  # True if triggered via E_{re-eval}

    def __repr__(self) -> str:
        origin = "Reclassification [E_re-eval]" if self.is_reclassification_violation else "Added Edge [E^+]"
        return (
            f"<LayerViolation ({origin}): {self.offending_edge.source_id} ({self.source_tier}) -> "
            f"{self.offending_edge.target_id} ({self.target_tier}) violates '{self.forbidden_rule}'>"
        )


class EmpiricalLayerGate:
    """
    Operationalizes Equation 6:
        r_layer(G') <=> ∀ (u, v) ∈ E^+ ∪ E_{re-eval}, v ∉ B(u)
    """

    # Default architectural rules (Clean / Hexagonal architecture reference baseline)
    # Used when mined rules file is uninitialized or in testing environments
    DEFAULT_FORBIDDEN_TIER_MAPPINGS: Final[dict[str, list[str]]] = {
        "Presentation": ["Infrastructure", "Database", "Driver"],
        "Infrastructure": ["Presentation", "UI", "Controller"],
        "Domain": ["Presentation", "UI", "Infrastructure_Driver"],
    }

    def __init__(
        self,
        mined_rules_path: Path | str | None = "data/architecture_rules/mined_rules_elb.json",
        treat_as_elb: bool = True,
        enforce_reclassification_check: bool = True,
    ) -> None:
        self.mined_rules_path = Path(mined_rules_path) if mined_rules_path else None
        self.treat_as_elb: Final[bool] = treat_as_elb
        self.enforce_reclassification_check: Final[bool] = enforce_reclassification_check

        # B(u) mappings loaded from ARM mining
        self.tier_blacklist: dict[str, set[str]] = {}
        self.package_blacklist: dict[str, set[str]] = {}
        self._load_mined_rules()

        logger.debug(
            "EmpiricalLayerGate initialized (ELB=%s, ReclassificationCheck=%s)",
            self.treat_as_elb,
            self.enforce_reclassification_check,
        )

    def _load_mined_rules(self) -> None:
        """Loads association rules mined from H_clean historical commits."""
        if not self.mined_rules_path or not self.mined_rules_path.exists():
            logger.warning(
                "Mined ELB rules not found at %s. Initializing with default standard tier mappings.",
                self.mined_rules_path,
            )
            self.tier_blacklist = {k: set(v) for k, v in self.DEFAULT_FORBIDDEN_TIER_MAPPINGS.items()}
            return

        try:
            with open(self.mined_rules_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            
            raw_tiers = data.get("forbidden_tier_dependencies", {})
            self.tier_blacklist = {k: set(v) for k, v in raw_tiers.items()}

            raw_pkgs = data.get("forbidden_package_dependencies", {})
            self.package_blacklist = {k: set(v) for k, v in raw_pkgs.items()}

            logger.info(
                "Mined rules B(u) loaded: %d tier bounds, %d package rules.",
                len(self.tier_blacklist),
                len(self.package_blacklist),
            )
        except Exception as err:
            logger.error("Failed to parse mined architectural rules JSON: %s. Using default mappings.", err)
            self.tier_blacklist = {k: set(v) for k, v in self.DEFAULT_FORBIDDEN_TIER_MAPPINGS.items()}

    def evaluate(
        self,
        graph: HierarchicalAttributedGraph,
        added_edges: Iterable[DependencyEdge],
        re_evaluated_edges: Iterable[DependencyEdge] | None = None,
    ) -> tuple[bool, list[LayerViolation]]:
        """
        Evaluates Equation 6 over the hypothetical graph state G'.
        
        Parameters:
            graph: The hypothetical graph G' constructed by the incremental operator ⊕.
            added_edges: The newly introduced edges E^+ parsed from AST diffs.
            re_evaluated_edges: Incident edges E_{re-eval} of entities whose architectural
                                tier metadata was modified (V_{re-tag}).
                                
        Returns:
            tuple[bool, list[LayerViolation]]: (passed, list_of_layer_violations)
        """
        violations: list[LayerViolation] = []
        set_added_edges = set(added_edges)
        set_re_eval_edges = set(re_evaluated_edges or []) if self.enforce_reclassification_check else set()

        # Formal set union: E_{eval} = E^+ ∪ E_{re-eval}
        eval_candidate_edges = set_added_edges | set_re_eval_edges

        for edge in eval_candidate_edges:
            is_reclass = edge in set_re_eval_edges and edge not in set_added_edges

            source_node = graph.nodes.get(edge.source_id)
            target_node = graph.nodes.get(edge.target_id)

            if not source_node or not target_node:
                # External library boundaries or unresolved placeholders are deferred to external validity
                continue

            src_tier = source_node.architectural_tier
            tgt_tier = target_node.architectural_tier

            # 1. Tier-Level Forbidden Check: target_tier ∈ B(source_tier)
            forbidden_target_tiers = self.tier_blacklist.get(src_tier, set())
            if tgt_tier in forbidden_target_tiers:
                violation = LayerViolation(
                    offending_edge=edge,
                    source_tier=src_tier,
                    target_tier=tgt_tier,
                    forbidden_rule=f"TierBlacklist: {src_tier} -> {tgt_tier}",
                    rejection_reason=(
                        f"Architectural Layer Breach (Eq. 6): Entity '{source_node.name}' in tier "
                        f"'{src_tier}' is strictly forbidden from importing/invoking '{target_node.name}' "
                        f"in tier '{tgt_tier}'."
                    ),
                    is_reclassification_violation=is_reclass,
                )
                violations.append(violation)
                logger.warning("Architectural Violation (Eq. 6): %r", violation)
                continue

            # 2. Package-Level Granular Check (Pattern-specific rules from ARM)
            src_pkg = self._extract_package_prefix(source_node.node_id)
            tgt_pkg = self._extract_package_prefix(target_node.node_id)

            forbidden_target_pkgs = self.package_blacklist.get(src_pkg, set())
            if any(tgt_pkg.startswith(forbidden) for forbidden in forbidden_target_pkgs):
                violation = LayerViolation(
                    offending_edge=edge,
                    source_tier=src_tier,
                    target_tier=tgt_tier,
                    forbidden_rule=f"PackageBlacklist: {src_pkg} -> {tgt_pkg}",
                    rejection_reason=(
                        f"Package Access Violation: Package '{src_pkg}' cannot reference '{tgt_pkg}'."
                    ),
                    is_reclassification_violation=is_reclass,
                )
                violations.append(violation)
                logger.warning("Package Layer Violation (Eq. 6): %r", violation)

        passed = len(violations) == 0
        return passed, violations

    @staticmethod
    def _extract_package_prefix(node_id: str) -> str:
        """Extracts top-level or sub-package prefix from fully qualified node ID."""
        parts = node_id.split(".")
        if len(parts) > 2:
            return ".".join(parts[:2])
        return parts[0] if parts else ""
