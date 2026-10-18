"""
================================================================================
Stage I: Aging-Aware Priority Queue & Starvation-Free Arbitration
================================================================================

Paper Reference:
  Section III-A: Incremental Graph Calculus and Bounded Arbitration
  "Governed Autonomy: Halting Architectural Entropy in Agentic Software 
   Workflows via Dynamic Guardrails" (SANER 2026)

Formal Implementation:
  Eq. 2: Priority(ΔC_i) = BasePriority(A_i) + λ * d_i
  Eq. 3: λ >= Δ_max / (K - 1), where Δ_max = max(BasePriority) - min(BasePriority)
  Theorem 1: Maximum Deferral Bound d_max <= ceil(Δ_max / λ) + (M - 1) * K

Standards:
  Compliant with IEEE/ACM Artifact Evaluation Guidelines (Badge: Reusable)
"""

from __future__ import annotations

import heapq
import itertools
import logging
import math
from dataclasses import dataclass, field
from typing import Final, Iterable, Iterator

from .conflict_detector import AgentPatch

logger = logging.getLogger("dag.arbitration.aging_priority_queue")


class StarvationBoundViolation(ValueError):
    """
    Raised when the configured aging factor λ fails to satisfy 
    the mathematical lower bound established in Equation 3.
    """
    pass


@dataclass(order=True)
class ArbitrationEntry:
    """
    Heap-orderable container for candidate patches implementing the total
    lexicographic priority order defined in Section III-A (Theorem 1):
      1. Higher effective Priority(ΔC_i) = BasePriority(A_i) + λ * d_i
      2. Larger deferral count d_i
      3. Earlier arrival sequence_id (FIFO)
      4. Lexicographically smaller agent_id
    """
    sort_key: tuple[float, int, int, str] = field(init=False)
    patch: AgentPatch = field(compare=False)
    base_priority: float = field(compare=False)
    deferral_count: int = field(compare=False, default=0)
    effective_priority: float = field(init=False, compare=False)
    sequence_id: int = field(compare=False, default=0)

    def __post_init__(self) -> None:
        pass

    def compute_priority(self, lambda_factor: float) -> float:
        """Calculates effective priority per Equation 2 and sets total order sort_key."""
        self.effective_priority = self.base_priority + (lambda_factor * float(self.deferral_count))
        self.sort_key = (
            -round(self.effective_priority, 8),
            -int(self.deferral_count),
            int(self.sequence_id),
            str(self.patch.agent_id),
        )
        return self.effective_priority


class AgingPriorityQueue:
    """
    Deterministic, bounded-aging priority queue implementing Equations 2, 3 and Theorem 1.
    Guarantees starvation-freeness under bounded retry regime K.
    """

    def __init__(
        self,
        lambda_factor: float,
        max_retries: int,
        min_base_priority: float = 1.0,
        max_base_priority: float = 5.0,
        enforce_strict_bound: bool = True,
    ) -> None:
        """
        Initializes the aging queue and verifies the theoretical starvation lower bound.
        """
        if max_retries <= 1:
            raise ValueError(f"MaxRetries must be >= 2 for iterative aging (got {max_retries}).")
        if max_base_priority < min_base_priority:
            raise ValueError("max_base_priority cannot be strictly less than min_base_priority.")

        self.lambda_factor: Final[float] = float(lambda_factor)
        self.max_retries: Final[int] = int(max_retries)
        self.min_base_priority: Final[float] = float(min_base_priority)
        self.max_base_priority: Final[float] = float(max_base_priority)

        # Mathematical verification of Equation 3
        self.delta_max: Final[float] = self.max_base_priority - self.min_base_priority
        self.theoretical_minimum_lambda: Final[float] = self.delta_max / float(self.max_retries - 1)

        if enforce_strict_bound and self.lambda_factor < self.theoretical_minimum_lambda:
            msg = (
                f"[Equation 3 Violation] Configured aging factor λ={self.lambda_factor:.4f} "
                f"is strictly below the starvation-free threshold λ_min={self.theoretical_minimum_lambda:.4f} "
                f"for priority spread [{self.min_base_priority}, {self.max_base_priority}] "
                f"across MaxRetries={self.max_retries}. Starvation-freeness cannot be guaranteed."
            )
            logger.error(msg)
            raise StarvationBoundViolation(msg)

        logger.info(
            "AgingPriorityQueue initialized (λ=%.4f, λ_min=%.4f, MaxRetries=%d)",
            self.lambda_factor,
            self.theoretical_minimum_lambda,
            self.max_retries,
        )

        self._heap: list[ArbitrationEntry] = []
        self._seq_counter: Iterator[int] = itertools.count(start=0)

    def compute_theorem1_max_deferral_bound(self, max_conflict_cliques_m: int = 3) -> int:
        """
        Computes the theoretical upper bound on deferrals from Theorem 1 (Section III-A):
            D_bound = ceil(Δ_max / λ) + (M - 1) * K
        where M is the maximum number of higher-priority tasks in a conflict clique.
        """
        if self.lambda_factor <= 0:
            return self.max_retries * max_conflict_cliques_m
        catchup_steps = math.ceil(self.delta_max / self.lambda_factor)
        holding_steps = max(0, max_conflict_cliques_m - 1) * self.max_retries
        return int(catchup_steps + holding_steps)

    def push(
        self,
        patch: AgentPatch,
        base_priority: float | None = None,
        deferral_count: int | None = None,
    ) -> ArbitrationEntry:
        """
        Enqueues an agent patch proposal, dynamically computing its priority per Equation 2.
        """
        assigned_base = base_priority if base_priority is not None else self.min_base_priority
        assigned_deferrals = deferral_count if deferral_count is not None else patch.deferral_count
        seq_id = next(self._seq_counter)

        entry = ArbitrationEntry(
            patch=patch,
            base_priority=assigned_base,
            deferral_count=assigned_deferrals,
            sequence_id=seq_id,
        )
        entry.compute_priority(self.lambda_factor)

        heapq.heappush(self._heap, entry)
        logger.debug(
            "Enqueued patch %s (Agent %s): Base=%.2f, Deferrals=%d, EffectivePriority=%.4f",
            patch.patch_id,
            patch.agent_id,
            assigned_base,
            assigned_deferrals,
            entry.effective_priority,
        )
        return entry

    def pop(self) -> ArbitrationEntry:
        """
        Dequeues the patch with the highest effective priority under total order.
        """
        if not self._heap:
            raise IndexError("Cannot pop from an empty AgingPriorityQueue.")
        return heapq.heappop(self._heap)

    def peek(self) -> ArbitrationEntry | None:
        """Returns the highest-priority element without removing it."""
        return self._heap[0] if self._heap else None

    def defer_and_age(self, entry: ArbitrationEntry) -> ArbitrationEntry:
        """
        Increments the deferral counter of a conflicting patch and re-enqueues it.
        Applies Equation 2 to boost priority for the subsequent arbitration epoch.
        """
        updated_patch = AgentPatch(
            patch_id=entry.patch.patch_id,
            agent_id=entry.patch.agent_id,
            touched_ranges=entry.patch.touched_ranges,
            raw_diff=entry.patch.raw_diff,
            deferral_count=entry.deferral_count + 1,
        )
        return self.push(
            patch=updated_patch,
            base_priority=entry.base_priority,
            deferral_count=entry.deferral_count + 1,
        )

    def batch_defer_and_age(
        self, entries: Iterable[ArbitrationEntry]
    ) -> list[ArbitrationEntry]:
        """
        Atomically increments aging counts for a batch of deferred patches (ΔC_defer).
        """
        return [self.defer_and_age(e) for e in entries]

    def clear(self) -> None:
        """Flushes the priority queue."""
        self._heap.clear()

    def __len__(self) -> int:
        return len(self._heap)

    def __bool__(self) -> bool:
        return len(self._heap) > 0

    def get_snapshot_sorted(self) -> list[ArbitrationEntry]:
        """
        Returns an ordered snapshot of all entries in the queue without mutating state.
        """
        return sorted(self._heap)
