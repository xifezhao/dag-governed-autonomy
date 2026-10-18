"""
================================================================================
Stage III: Single-Point Fault Isolation & Graceful Degradation Protocol
================================================================================

Paper Reference:
  Section III-C: Exhaustive Attribution and Semantic Decoupling
  Algorithm 1: Lines 15-18 (Single-Point Isolation & Partial Merge)
  Section V-C: Answering RQ3 (Oscillation Prevention & Isolation Efficacy)
  "Governed Autonomy: Halting Architectural Entropy in Agentic Software 
   Workflows via Dynamic Guardrails" (SANER 2026)

Formal Invariant:
  Transforms systemic multi-agent livelocks into localized patch quarantine:
    ΔC_{orth} <- ΔC_{orth} \\ {ΔC_i}
  Permits orthogonal, structurally sound contributions to merge safely into G_{safe}
  while isolating non-convergent agents without halting the global CI/CD pipeline.

Standards:
  Compliant with IEEE/ACM Artifact Evaluation Guidelines (Badge: Reusable)
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Final, Iterable, Sequence

from ..arbitration.conflict_detector import AgentPatch

logger = logging.getLogger("dag.attribution.isolation_manager")


class BatchDegradationStatus(str, Enum):
    """
    Taxonomy of batch completion health post-isolation evaluation.
    """
    OPTIMAL = "OPTIMAL"                   # 100% of candidate orthogonal patches merged cleanly
    PARTIAL_MERGE = "PARTIAL_MERGE"       # Graceful degradation: isolated chronically flawed agents, merged healthy subset
    TOTAL_QUARANTINE = "TOTAL_QUARANTINE" # All patches in batch were isolated; requires human fallback
    EMPTY_INPUT = "EMPTY_INPUT"           # Zero candidate proposals in initial batch


@dataclass(frozen=True, slots=True)
class IsolationDecision:
    """
    Immutable forensic record of a Single-Point Isolation action (Algorithm 1, Line 16).
    """
    isolated_patch_id: str
    isolated_agent_id: str
    recurring_state_hash: str
    iteration: int
    trigger_reason: str
    cycle_trajectory: tuple[str, ...] = field(default_factory=tuple)
    timestamp_epoch: float = field(default_factory=time.time)

    def __repr__(self) -> str:
        traj_preview = " -> ".join(self.cycle_trajectory[:3]) if self.cycle_trajectory else "N/A"
        return (
            f"<IsolationDecision: Agent '{self.isolated_agent_id}' (Patch {self.isolated_patch_id}) "
            f"QUARANTINED at Iter {self.iteration}. Reason: {self.trigger_reason}. Trajectory: [{traj_preview}]>"
        )


@dataclass(frozen=True, slots=True)
class IsolationReport:
    """
    Complete state outcome after applying single-point isolation to an orthogonal batch.
    """
    retained_patches: tuple[AgentPatch, ...]       # Clean orthogonal subset ΔC_{orth} permitted to commit
    quarantined_patches: tuple[AgentPatch, ...]    # Culpable patches purged from current epoch
    isolation_decisions: tuple[IsolationDecision, ...] # Formal audit trail records
    degradation_status: BatchDegradationStatus
    isolation_rate: float = field(init=False)      # Ratio of isolated patches over original orthogonal set

    def __post_init__(self) -> None:
        total = len(self.retained_patches) + len(self.quarantined_patches)
        rate = float(len(self.quarantined_patches)) / float(total) if total > 0 else 0.0
        object.__setattr__(self, "isolation_rate", rate)

    def is_executable(self) -> bool:
        """Returns True if there is at least one healthy patch to commit."""
        return len(self.retained_patches) > 0


class SinglePointIsolationManager:
    """
    Executes Single-Point Isolation and manages multi-agent graceful degradation.
    """

    def __init__(
        self,
        fallback_on_empty_merge: bool = True,
        quarantine_log_path: Path | str | None = None,
    ) -> None:
        """
        Parameters:
            fallback_on_empty_merge: If True, raises warning/alert when all patches are quarantined.
            quarantine_log_path: Optional path to write structured isolation JSON records.
        """
        self.fallback_on_empty: Final[bool] = fallback_on_empty_merge
        self.quarantine_log_path = Path(quarantine_log_path) if quarantine_log_path else None
        
        # Historical ledger of all isolation events across the process lifetime
        self._isolation_ledger: list[IsolationDecision] = []
        self._quarantined_agent_ids: set[str] = set()

        logger.debug(
            "SinglePointIsolationManager initialized (FallbackOnEmpty=%s)",
            self.fallback_on_empty,
        )

    def create_isolation_decision(
        self,
        patch: AgentPatch,
        recurring_hash: str,
        iteration: int,
        cycle_trajectory: Sequence[str] | None = None,
        reason: str = "Whack-a-Mole oscillation trap detected",
    ) -> IsolationDecision:
        """
        Constructs and records a formal isolation judgment for a culpable patch.
        """
        decision = IsolationDecision(
            isolated_patch_id=patch.patch_id,
            isolated_agent_id=patch.agent_id,
            recurring_state_hash=recurring_hash,
            iteration=iteration,
            trigger_reason=reason,
            cycle_trajectory=tuple(cycle_trajectory or ()),
        )

        self._isolation_ledger.append(decision)
        self._quarantined_agent_ids.add(patch.agent_id)

        logger.warning(
            "SINGLE-POINT ISOLATION TRIGGERED (Algorithm 1, Line 16): %r", decision
        )
        return decision

    def apply_isolation(
        self,
        current_orthogonal_patches: Sequence[AgentPatch],
        culpable_decisions: Sequence[IsolationDecision],
    ) -> IsolationReport:
        """
        Applies mathematical set exclusion:
            ΔC_{orth} <- ΔC_{orth} \\ {ΔC_culpable}
            
        Returns:
            IsolationReport containing the partitioned subsets and degradation health status.
        """
        if not current_orthogonal_patches:
            return IsolationReport(
                retained_patches=(),
                quarantined_patches=(),
                isolation_decisions=(),
                degradation_status=BatchDegradationStatus.EMPTY_INPUT,
            )

        culpable_patch_ids = {d.isolated_patch_id for d in culpable_decisions}

        retained: list[AgentPatch] = []
        quarantined: list[AgentPatch] = []

        for patch in current_orthogonal_patches:
            if patch.patch_id in culpable_patch_ids:
                quarantined.append(patch)
            else:
                retained.append(patch)

        # Classify degradation health status
        if not quarantined:
            status = BatchDegradationStatus.OPTIMAL
        elif retained:
            status = BatchDegradationStatus.PARTIAL_MERGE
            logger.info(
                "Graceful degradation achieved: Merging %d healthy patches; quarantined %d oscillating patches.",
                len(retained),
                len(quarantined),
            )
        else:
            status = BatchDegradationStatus.TOTAL_QUARANTINE
            logger.error(
                "All %d candidate patches in orthogonal batch were quarantined. Halting to human review fallback.",
                len(quarantined),
            )

        return IsolationReport(
            retained_patches=tuple(retained),
            quarantined_patches=tuple(quarantined),
            isolation_decisions=tuple(culpable_decisions),
            degradation_status=status,
        )

    def is_agent_quarantined(self, agent_id: str) -> bool:
        """O(1) query checking if an agent is currently under active quarantine."""
        return agent_id in self._quarantined_agent_ids

    def get_isolation_history(self) -> list[IsolationDecision]:
        """Returns the full chronological history of isolation decisions for RQ3 audits."""
        return list(self._isolation_ledger)

    def get_quarantined_agents(self) -> frozenset[str]:
        """Returns the set of all unique agent IDs that have triggered isolation."""
        return frozenset(self._quarantined_agent_ids)

    def reset_task_state(self) -> None:
        """Resets per-task isolation state while retaining cumulative telemetry."""
        self._quarantined_agent_ids.clear()

    def export_summary(self) -> dict[str, Any]:
        """
        Generates telemetry metrics directly verifying Section V-C (RQ3).
        """
        return {
            "total_isolations_count": len(self._isolation_ledger),
            "unique_agents_isolated_count": len(self._quarantined_agent_ids),
            "isolation_events": [
                {
                    "patch_id": d.isolated_patch_id,
                    "agent_id": d.isolated_agent_id,
                    "iteration": d.iteration,
                    "hash": d.recurring_state_hash[:10],
                    "reason": d.trigger_reason,
                }
                for d in self._isolation_ledger
            ],
        }
