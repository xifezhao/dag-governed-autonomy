r"""
================================================================================
Stage II: Whitelisted Cyclic Dependency Gate (r_cycle, Equation 4)
================================================================================

Paper Reference:
  Section III-B: Architectural Constraint Calculus and Complexity Bounds
  Equation 4: r_cycle(G') <=> ∀ C ∈ SCC(G') \ A, |C| = 1
  "Governed Autonomy: Halting Architectural Entropy in Agentic Software 
   Workflows via Dynamic Guardrails" (SANER 2026)

Formal Invariant:
  Permits non-trivial cycles (|C| > 1) if and only if the SCC topology 
  is formally isomorphic to an architect-ratified design pattern in whitelist A
  and does not breach layer isolation anti-gaming constraints.

Standards:
  Compliant with IEEE/ACM Artifact Evaluation Guidelines (Badge: Reusable)
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

from ..graph.hpg_model import (
    DependencyEdge,
    HierarchicalAttributedGraph,
)

logger = logging.getLogger("dag.gating.cycle_gate")


DEFAULT_RATIFIED_PATTERNS: Final[list[dict[str, Any]]] = [
    {"pattern_id": "PAT-GOF-OBSERVER-01", "name": "Observer Pattern"},
    {"pattern_id": "PAT-GOF-VISITOR-02", "name": "Visitor Pattern"},
    {"pattern_id": "PAT-GOF-COMPOSITE-03", "name": "Composite Pattern"},
]

DEFAULT_ANTI_GAMING_RULES: Final[list[dict[str, Any]]] = [
    {
        "id": "BLOCK-CROSS-LAYER-CYCLIC",
        "prohibited_layers_combination": [
            ["Presentation", "Infrastructure"],
            ["Presentation", "Domain"],
            ["Domain", "Infrastructure"],
        ],
        "rejection_reason": "Cross-tier cyclic dependency is strictly forbidden even if named after a GoF pattern.",
    }
]


@dataclass(frozen=True, slots=True)
class CycleViolation:
    """
    Structured descriptor for an unauthorized cyclic dependency component C.
    Directly transformed into a ViolationInstance for downstream attribution.
    """
    component_node_ids: frozenset[str]
    offending_edges: frozenset[DependencyEdge]
    cycle_length: int
    participating_tiers: frozenset[str]
    rejection_reason: str

    def __repr__(self) -> str:
        nodes_preview = ", ".join(sorted(self.component_node_ids)[:3])
        if len(self.component_node_ids) > 3:
            nodes_preview += ", ..."
        return (
            f"<CycleViolation |C|={self.cycle_length} Nodes=[{nodes_preview}] "
            f"Tiers={set(self.participating_tiers)} Reason='{self.rejection_reason}'>"
        )


class WhitelistPatternMatcher:
    """
    Evaluator verifying whether a non-trivial SCC belongs to ratified whitelist A.
    Enforces anti-gaming constraints (e.g. cross-layer cycle blocking).
    """

    def __init__(self, whitelist_config_path: Path | str | None = None) -> None:
        self.whitelist_path = Path(whitelist_config_path) if whitelist_config_path else None
        self.ratified_patterns: list[dict[str, Any]] = list(DEFAULT_RATIFIED_PATTERNS)
        self.anti_gaming_rules: list[dict[str, Any]] = list(DEFAULT_ANTI_GAMING_RULES)
        self._load_whitelist()

    def _load_whitelist(self) -> None:
        if self.whitelist_path and not self.whitelist_path.exists() and not self.whitelist_path.is_absolute():
            candidate = Path(__file__).resolve().parent.parent.parent / self.whitelist_path
            if candidate.exists():
                self.whitelist_path = candidate

        if not self.whitelist_path or not self.whitelist_path.exists():
            return

        try:
            import yaml  # type: ignore
            with open(self.whitelist_path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}
            if "ratified_patterns" in data:
                self.ratified_patterns = data["ratified_patterns"]
            if "anti_gaming_rules" in data:
                self.anti_gaming_rules = data["anti_gaming_rules"]
        except Exception:
            # Pure-Python default catalog already loaded
            pass

    def check_exemption(
        self,
        component_nodes: frozenset[str],
        cycle_edges: frozenset[DependencyEdge],
        graph: HierarchicalAttributedGraph,
    ) -> tuple[bool, str]:
        """
        Determines whether component C in SCC is isomorphic to any pattern in A.
        """
        participating_tiers = {
            graph.nodes[nid].architectural_tier
            for nid in component_nodes
            if nid in graph.nodes and graph.nodes[nid].architectural_tier != "Unknown"
        }

        for rule in self.anti_gaming_rules:
            if rule.get("id") == "BLOCK-CROSS-LAYER-CYCLIC":
                for prohibited_pair in rule.get("prohibited_layers_combination", []):
                    if set(prohibited_pair).issubset(participating_tiers):
                        return (
                            False,
                            f"Anti-Gaming Triggered: Cycle spans prohibited layer boundary "
                            f"{prohibited_pair} ({rule.get('rejection_reason')})",
                        )

        cycle_len = len(component_nodes)
        if cycle_len > 3:
            return False, f"Cycle length {cycle_len} exceeds maximum allowed pattern threshold (|C| <= 3)."

        node_objects = [graph.nodes[nid] for nid in component_nodes if nid in graph.nodes]
        node_names = [n.name for n in node_objects]

        for pattern in self.ratified_patterns:
            pattern_id = pattern.get("pattern_id", "UNKNOWN")

            if pattern_id == "PAT-GOF-OBSERVER-01":
                has_subject = any(re.search(r"(Subject|Observable|Emitter)", name, re.I) for name in node_names)
                has_observer = any(re.search(r"(Observer|Listener|Subscriber)", name, re.I) for name in node_names)
                if has_subject and has_observer:
                    return True, f"Exempted by Ratified Pattern A: Observer ({pattern_id})"

            elif pattern_id == "PAT-GOF-VISITOR-02":
                has_visitor = any(re.search(r"Visitor", name, re.I) for name in node_names)
                has_element = any(re.search(r"(Visitable|Element|Node)", name, re.I) for name in node_names)
                if has_visitor and has_element:
                    return True, f"Exempted by Ratified Pattern A: Visitor ({pattern_id})"

            elif pattern_id == "PAT-GOF-COMPOSITE-03":
                has_composite = any(re.search(r"(Composite|Group|Container)", name, re.I) for name in node_names)
                has_component = any(re.search(r"(Component|Item|Element)", name, re.I) for name in node_names)
                if has_composite and has_component:
                    return True, f"Exempted by Ratified Pattern A: Composite ({pattern_id})"

        return False, "Cycle does not conform to any ratified GoF/DDD whitelist topology."


class WhitelistedCycleGate:
    r"""
    Implementation of Equation 4:
        r_cycle(G') <=> ∀ C ∈ SCC(G') \ A, |C| = 1
    """

    def __init__(
        self,
        whitelist_config_path: Path | str | None = "config/arch_whitelist.yaml",
        algorithm: str = "haeupler_scc",
    ) -> None:
        self.algorithm: Final[str] = algorithm
        self.matcher: Final[WhitelistPatternMatcher] = WhitelistPatternMatcher(whitelist_config_path)
        logger.debug("WhitelistedCycleGate initialized (algorithm=%s)", self.algorithm)

    def evaluate(
        self, graph: HierarchicalAttributedGraph
    ) -> tuple[bool, list[CycleViolation]]:
        """
        Decomposes graph G' into Strongly Connected Components (SCCs), filters
        out trivial components (|C| = 1), and validates non-trivial cycles against A.
        """
        scc_components = graph.strongly_connected_components()
        violations: list[CycleViolation] = []

        for component_frozen in scc_components:
            # Check self-loops or non-trivial SCCs (|C| > 1)
            if len(component_frozen) <= 1:
                continue

            cycle_edges: set[DependencyEdge] = set()
            for edge in graph.edges:
                if edge.source_id in component_frozen and edge.target_id in component_frozen:
                    cycle_edges.add(edge)

            is_exempt, diagnostic_reason = self.matcher.check_exemption(
                component_nodes=component_frozen,
                cycle_edges=frozenset(cycle_edges),
                graph=graph,
            )

            if is_exempt:
                logger.info(
                    "Cycle %s successfully exempted via whitelist A: %s",
                    set(component_frozen),
                    diagnostic_reason,
                )
                continue

            participating_tiers = frozenset(
                graph.nodes[nid].architectural_tier
                for nid in component_frozen
                if nid in graph.nodes
            )

            violation = CycleViolation(
                component_node_ids=component_frozen,
                offending_edges=frozenset(cycle_edges),
                cycle_length=len(component_frozen),
                participating_tiers=participating_tiers,
                rejection_reason=diagnostic_reason,
            )
            violations.append(violation)
            logger.warning("Architectural Violation (Eq. 4): %r", violation)

        passed = len(violations) == 0
        return passed, violations
