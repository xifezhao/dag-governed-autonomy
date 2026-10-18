"""
================================================================================
Stage III: Exhaustive Bifurcated Fault Attribution Engine (Equation 8)
================================================================================

Paper Reference:
  Section III-C: Exhaustive Attribution and Oscillation Control
  Equation 8: Bifurcated Attribution Mapping Attr(v)
  Algorithm 1: Line 10 (Targeted Agent Violation Extraction V_i)
  "Governed Autonomy: Halting Architectural Entropy in Agentic Software 
   Workflows via Dynamic Guardrails" (SANER 2026)

Formal Implementation:
  Attr(v) =
    Branch 1 [Structural]: {ΔC_i | e ∈ E_i^+ ∨ {x, y} ∩ V_{re,i} != ∅}
    Branch 2 [Scalar Debt]: {ΔC_i | AgTD_i >= η * (τ_{batch}(k) / k)}
                            (Fallback: argmax_{ΔC_i} AgTD_i if all < η * τ/k)
    Pre-existing violations present before the batch receive Attr(v) = ∅.

Standards:
  Compliant with IEEE/ACM Artifact Evaluation Guidelines (Badge: Reusable)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Final, Mapping, Sequence

from ..gating import ViolationInstance, ViolationType
from ..gating.agtd_calculator import AgTDMetricProfile
from ..graph.incremental_operator import ASTDiffPayload

logger = logging.getLogger("dag.attribution.exhaustive_attributor")


@dataclass(frozen=True, slots=True)
class AttributionResult:
    """
    Immutable envelope of multi-agent fault attribution across batch ΔC_orth.
    Directly consumed by Algorithm 1 lines 10-24 to drive targeted nudging.
    """
    culpable_patch_ids: frozenset[str]                           # Patches receiving fault attribution
    culpable_agent_ids: frozenset[str]                           # Agents tasked with targeted regeneration
    innocent_patch_ids: frozenset[str]                           # Non-culpable patches preserved for merging
    per_patch_violations: dict[str, tuple[ViolationInstance, ...]] # V_i: Violations attributed to patch i
    per_violation_attributions: dict[str, frozenset[str]]         # Mapping: v.signature -> {culpable_patch_ids}
    total_violations_attributed: int
    wasted_tokens_prevented_ratio: float = field(init=False)     # Telemetry metric for RQ2 evaluation

    def __post_init__(self) -> None:
        total_patches = len(self.culpable_patch_ids) + len(self.innocent_patch_ids)
        ratio = (
            float(len(self.innocent_patch_ids)) / float(total_patches)
            if total_patches > 0
            else 0.0
        )
        object.__setattr__(self, "wasted_tokens_prevented_ratio", ratio)

    def __repr__(self) -> str:
        return (
            f"<AttributionResult: CulpablePatches={len(self.culpable_patch_ids)} "
            f"InnocentPatches={len(self.innocent_patch_ids)} "
            f"ShieldingRatio={self.wasted_tokens_prevented_ratio:.2%}>"
        )


class ExhaustiveAttributor:
    """
    Implementation of Equation 8:
        Bifurcated attribution mapping Attr(v) isolating structural blame
        and proportional scalar technical debt accumulation.
    """

    def __init__(self, tolerance_coefficient_eta: float = 1.20) -> None:
        """
        Parameters:
            tolerance_coefficient_eta: Tolerance factor η in Equation 8 (Branch 2).
                                       Defaults to 1.20 (120% of fair batch share).
        """
        if tolerance_coefficient_eta <= 0:
            raise ValueError(f"tolerance_coefficient_eta must be strictly positive (got {tolerance_coefficient_eta}).")

        self.eta: Final[float] = float(tolerance_coefficient_eta)
        logger.debug("ExhaustiveAttributor initialized (η=%.2f)", self.eta)

    def attribute_violation(
        self,
        violation: ViolationInstance,
        per_patch_diffs: Mapping[str, ASTDiffPayload],
        agtd_profiles: Mapping[str, AgTDMetricProfile] | None = None,
        tau_batch: float = 0.0,
        batch_size: int = 1,
        pre_existing_signatures: frozenset[str] = frozenset(),
    ) -> frozenset[str]:
        """
        Computes Equation 8 for a single violation instance v.
        """
        # Pre-existing violations already present in base graph G receive Attr(v) = ∅
        if violation.signature in pre_existing_signatures or violation.context.get("pre_existing", False):
            return frozenset()

        culpable_patches: set[str] = set()

        # ======================================================================
        # Branch 1: Structural Violations (r_cycle and r_layer)
        # Condition: e ∈ E_i^+ ∨ {x, y} ∩ V_{re,i} != ∅
        # ======================================================================
        if violation.type in (ViolationType.CYCLE, ViolationType.LAYER):
            for patch_id, diff in per_patch_diffs.items():
                has_offending_edge = bool(violation.edges.intersection(diff.edges_added))
                has_reclassified_node = bool(violation.nodes.intersection(diff.reclassified_node_ids))

                if has_offending_edge or has_reclassified_node:
                    culpable_patches.add(patch_id)
                    logger.info(
                        "Attributed structural violation %s to Patch %s (Agent %s). "
                        "[EdgeMatch=%s, ReclassMatch=%s]",
                        violation.rule_id,
                        patch_id,
                        diff.agent_id,
                        has_offending_edge,
                        has_reclassified_node,
                    )

            # Fallback for emergent node-level interactions if no single patch added the explicit edge
            if not culpable_patches:
                for patch_id, diff in per_patch_diffs.items():
                    touched_nodes = {n.node_id for n in diff.nodes_added | diff.nodes_modified}
                    if violation.nodes.intersection(touched_nodes):
                        culpable_patches.add(patch_id)

        # ======================================================================
        # Branch 2: Scalar Debt Spikes (r_debt)
        # Condition: AgTD_i >= η * (τ_{batch}(k) / k)
        # Fallback: argmax_{ΔC_i} AgTD_i if all < η * (τ_{batch}(k) / k)
        # ======================================================================
        elif violation.type == ViolationType.AGTD_SPIKE:
            profiles = agtd_profiles or {}
            k = max(1, batch_size)
            fair_share_threshold = self.eta * (tau_batch / float(k))

            for patch_id, profile in profiles.items():
                if profile.net_positive_delta >= fair_share_threshold:
                    culpable_patches.add(patch_id)
                    logger.info(
                        "Attributed AgTD spike to Patch %s (Agent %s): "
                        "AgTD_i=+%.4f >= Threshold=%.4f (η=%.2f, τ/k=%.4f)",
                        patch_id,
                        profile.agent_id,
                        profile.net_positive_delta,
                        fair_share_threshold,
                        self.eta,
                        tau_batch / float(k),
                    )

            # Formal Fallback (Section III-C):
            # If ∑ AgTD_i > τ_batch(k) and ∀i, AgTD_i < η * τ_batch(k)/k,
            # blame argmax_{ΔC_i} AgTD_i so that no debt violation goes unattributed.
            if not culpable_patches and profiles:
                max_val = max(p.net_positive_delta for p in profiles.values())
                for patch_id, profile in profiles.items():
                    if abs(profile.net_positive_delta - max_val) <= 1e-9:
                        culpable_patches.add(patch_id)
                logger.warning(
                    "Distributed AgTD spike argmax fallback: Attributed to top contributor(s): %s",
                    culpable_patches,
                )

        return frozenset(culpable_patches)

    def attribute_batch(
        self,
        violations: Sequence[ViolationInstance],
        per_patch_diffs: Mapping[str, ASTDiffPayload],
        agtd_profiles: Mapping[str, AgTDMetricProfile] | None = None,
        tau_batch: float = 0.0,
        pre_existing_signatures: frozenset[str] = frozenset(),
    ) -> AttributionResult:
        """
        Executes exhaustive multi-violation attribution across the entire orthogonal batch.
        Synthesizes V_i subsets strictly for culpable agents (Algorithm 1, Line 10).
        """
        batch_size = len(per_patch_diffs)
        per_violation_attributions: dict[str, frozenset[str]] = {}
        per_patch_violations_map: dict[str, list[ViolationInstance]] = {
            pid: [] for pid in per_patch_diffs.keys()
        }
        all_culpable_patches: set[str] = set()

        for v in violations:
            culpable_set = self.attribute_violation(
                violation=v,
                per_patch_diffs=per_patch_diffs,
                agtd_profiles=agtd_profiles,
                tau_batch=tau_batch,
                batch_size=batch_size,
                pre_existing_signatures=pre_existing_signatures,
            )
            per_violation_attributions[v.signature] = culpable_set
            all_culpable_patches.update(culpable_set)

            for pid in culpable_set:
                if pid in per_patch_violations_map:
                    per_patch_violations_map[pid].append(v)

        culpable_agent_ids = {
            per_patch_diffs[pid].agent_id
            for pid in all_culpable_patches
            if pid in per_patch_diffs
        }

        all_patch_ids = set(per_patch_diffs.keys())
        innocent_patch_ids = all_patch_ids - all_culpable_patches

        final_per_patch_violations = {
            pid: tuple(v_list)
            for pid, v_list in per_patch_violations_map.items()
            if v_list
        }

        logger.info(
            "Batch Attribution Complete (Eq. 8): %d violations mapped across %d patches. "
            "Culpable=%d, Innocent=%d (Shielded).",
            len(violations),
            batch_size,
            len(all_culpable_patches),
            len(innocent_patch_ids),
        )

        return AttributionResult(
            culpable_patch_ids=frozenset(all_culpable_patches),
            culpable_agent_ids=frozenset(culpable_agent_ids),
            innocent_patch_ids=frozenset(innocent_patch_ids),
            per_patch_violations=final_per_patch_violations,
            per_violation_attributions=per_violation_attributions,
            total_violations_attributed=len(violations),
        )
