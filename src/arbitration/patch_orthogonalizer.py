"""
================================================================================
Stage I: Patch Orthogonalizer & Maximum Weight Independent Set Partitioning
================================================================================

Paper Reference:
  Section III-A: Incremental Graph Calculus and Bounded Arbitration
  Algorithm 1: Line 4 (AgingArbitrate)
  "Governed Autonomy: Halting Architectural Entropy in Agentic Software 
   Workflows via Dynamic Guardrails" (SANER 2026)

Formal Logic:
  Partitions concurrent proposals ΔC_set into:
    1. ΔC_orth : Mutually orthogonal (disjoint Touched ranges), highest-priority subset
    2. ΔC_defer: Conflicting patches scheduled for deferral and aging increment

Standards:
  Compliant with IEEE/ACM Artifact Evaluation Guidelines (Badge: Reusable)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Final, Iterable, Sequence

from .aging_priority_queue import AgingPriorityQueue, ArbitrationEntry
from .conflict_detector import AgentPatch, ConflictDetector, PatchConflict

logger = logging.getLogger("dag.arbitration.patch_orthogonalizer")


@dataclass(frozen=True, slots=True)
class OrthogonalPartitionResult:
    """
    Immutable structured result returned by the orthogonalization pipeline.
    Directly inspected by DAG loop driver (Algorithm 1) and AE test suites.
    """
    orthogonal_patches: tuple[AgentPatch, ...]  # ΔC_orth: safe to merge via incremental AST update ⊕
    deferred_patches: tuple[AgentPatch, ...]    # ΔC_defer: conflicted, subject to aging priority boost
    conflicts_encountered: tuple[PatchConflict, ...] # Audit records of Equation 1 violations
    total_proposals_count: int
    orthogonal_acceptance_ratio: float = field(init=False)

    def __post_init__(self) -> None:
        ratio = (
            len(self.orthogonal_patches) / float(self.total_proposals_count)
            if self.total_proposals_count > 0
            else 1.0
        )
        object.__setattr__(self, "orthogonal_acceptance_ratio", ratio)

    def __repr__(self) -> str:
        return (
            f"<OrthogonalPartitionResult: |ΔC_orth|={len(self.orthogonal_patches)}, "
            f"|ΔC_defer|={len(self.deferred_patches)}, "
            f"AcceptanceRatio={self.orthogonal_acceptance_ratio:.2%}>"
        )


class PatchOrthogonalizer:
    """
    Orchestrates syntactic conflict detection (Eq. 1) and aging-aware priority
    queuing (Eq. 2 & 3) to extract the maximal non-conflicting patch subset.
    """

    def __init__(
        self,
        conflict_detector: ConflictDetector,
        priority_queue: AgingPriorityQueue,
    ) -> None:
        self.conflict_detector: Final[ConflictDetector] = conflict_detector
        self.priority_queue: Final[AgingPriorityQueue] = priority_queue

    def partition(
        self,
        candidate_patches: Sequence[AgentPatch],
        base_priorities: dict[str, float] | None = None,
    ) -> OrthogonalPartitionResult:
        """
        Executes priority-guided independent set partitioning.
        
        Algorithm:
          1. Enqueue all candidate patches into AgingPriorityQueue.
          2. Greedily extract highest effective-priority candidate.
          3. Check syntactic conflicts against already accepted orthogonal subset.
          4. If disjoint (Eq. 1 == False), admit to ΔC_orth.
          5. Else, reject and route to ΔC_defer (with deferral count increment).
        """
        if not candidate_patches:
            return OrthogonalPartitionResult(
                orthogonal_patches=(),
                deferred_patches=(),
                conflicts_encountered=(),
                total_proposals_count=0,
            )

        priorities = base_priorities or {}
        self.priority_queue.clear()

        # Step 1: Ingest batch and compute dynamic priorities via Eq. 2
        for patch in candidate_patches:
            base_p = priorities.get(patch.agent_id, self.priority_queue.min_base_priority)
            self.priority_queue.push(patch, base_priority=base_p)

        accepted_entries: list[ArbitrationEntry] = []
        deferred_entries: list[ArbitrationEntry] = []
        detected_conflicts: list[PatchConflict] = []

        logger.debug(
            "Commencing orthogonalization for %d candidate patches.",
            len(candidate_patches),
        )

        # Step 2-4: Greedy conflict-free extraction based on effective priority order
        while self.priority_queue:
            candidate = self.priority_queue.pop()
            candidate_patch = candidate.patch

            has_conflict = False
            for accepted in accepted_entries:
                conflict = self.conflict_detector.check_conflict(
                    candidate_patch, accepted.patch
                )
                if conflict:
                    has_conflict = True
                    detected_conflicts.append(conflict)
                    logger.info(
                        "Arbitration Conflict (Eq. 1): Patch %s (Agent %s, P=%.2f) "
                        "conflicts with accepted %s (Agent %s, P=%.2f). Deferring.",
                        candidate_patch.patch_id,
                        candidate_patch.agent_id,
                        candidate.effective_priority,
                        accepted.patch.patch_id,
                        accepted.patch.agent_id,
                        accepted.effective_priority,
                    )
                    break

            if not has_conflict:
                accepted_entries.append(candidate)
            else:
                deferred_entries.append(candidate)

        # Step 5: Age deferred patches for subsequent epochs (Section III-A)
        # Note: Aging increment is persisted on the patch dataclass
        aged_deferred_patches: list[AgentPatch] = []
        for entry in deferred_entries:
            updated_patch = AgentPatch(
                patch_id=entry.patch.patch_id,
                agent_id=entry.patch.agent_id,
                touched_ranges=entry.patch.touched_ranges,
                raw_diff=entry.patch.raw_diff,
                deferral_count=entry.deferral_count + 1,
            )
            aged_deferred_patches.append(updated_patch)

        orthogonal_patches = tuple(e.patch for e in accepted_entries)
        deferred_patches = tuple(aged_deferred_patches)

        logger.info(
            "Orthogonalization finished: %d admitted (ΔC_orth), %d deferred (ΔC_defer).",
            len(orthogonal_patches),
            len(deferred_patches),
        )

        return OrthogonalPartitionResult(
            orthogonal_patches=orthogonal_patches,
            deferred_patches=deferred_patches,
            conflicts_encountered=tuple(detected_conflicts),
            total_proposals_count=len(candidate_patches),
        )

    def extract_orthogonal_subset(
        self, candidate_patches: Sequence[AgentPatch]
    ) -> tuple[AgentPatch, ...]:
        """
        Convenience wrapper returning strictly the admitted ΔC_orth subset.
        """
        return self.partition(candidate_patches).orthogonal_patches
