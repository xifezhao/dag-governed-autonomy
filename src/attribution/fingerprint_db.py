"""
================================================================================
Stage III: Topological State Fingerprint Database & Anti-Oscillation Engine
================================================================================

Paper Reference:
  Section III-C: Exhaustive Attribution and Oscillation Control
  Equation 9: F_t^{(i)} = Hash(Sort({sig(v) | v ∈ V_i^{(t)}}))
  Algorithm 1: Lines 12-16 (DetectCycle & Single-Point Isolation)
  "Governed Autonomy: Halting Architectural Entropy in Agentic Software 
   Workflows via Dynamic Guardrails" (SANER 2026)

Formal Invariant:
  Detects non-adjacent state transition cycles (whack-a-mole livelocks) where
  F_t^{(i)} = F_{t-m}^{(i)} for some m >= 2 and F_t^{(i)} != F_{t-1}^{(i)}
  (e.g., X -> Y -> X). Consecutive identical states (X -> X) represent incomplete
  local repairs and are bounded by the retry limit K.

Standards:
  Compliant with IEEE/ACM Artifact Evaluation Guidelines (Badge: Reusable)
"""

from __future__ import annotations

import hashlib
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Final, Sequence

from ..gating import ViolationInstance

logger = logging.getLogger("dag.attribution.fingerprint_db")


class OscillationTrapDetected(RuntimeError):
    """
    Raised when an agent produces a violation signature state identical 
    to an earlier non-adjacent generation cycle within the same task epoch.
    """
    def __init__(self, agent_id: str, recurring_hash: str, cycle_trajectory: list[str]) -> None:
        self.agent_id = agent_id
        self.recurring_hash = recurring_hash
        self.cycle_trajectory = cycle_trajectory
        trajectory_str = " -> ".join(cycle_trajectory)
        super().__init__(
            f"[Livelock Trap Detected] Agent '{agent_id}' produced recurring state hash "
            f"'{recurring_hash[:10]}...'. Cyclic Trajectory: [{trajectory_str}]."
        )


@dataclass(frozen=True, slots=True)
class FingerprintRecord:
    """
    Immutable historical snapshot of an agent's architectural violation state.
    """
    agent_id: str
    iteration: int
    state_hash: str
    violation_signatures: tuple[str, ...]
    timestamp_epoch: float = field(default_factory=time.time)

    def __repr__(self) -> str:
        sigs_preview = ", ".join(self.violation_signatures[:2])
        if len(self.violation_signatures) > 2:
            sigs_preview += ", ..."
        return (
            f"<FingerprintRecord Agent={self.agent_id} Iter={self.iteration} "
            f"Hash={self.state_hash[:8]} Sigs=[{sigs_preview}]>"
        )


class TopologicalFingerprintDB:
    """
    Operationalizes Equation 9 & Algorithm 1 DetectCycle:
        F_t^{(i)} = Hash(Sort({sig(v) | v ∈ V_i^{(t)}}))
    """

    def __init__(self, hash_algorithm: str = "sha256", scope: str = "FULL_HISTORY") -> None:
        self.hash_algorithm: Final[str] = hash_algorithm.lower()
        self.scope: Final[str] = scope.upper()

        # State Storage: agent_id -> list of FingerprintRecord (in temporal order)
        self._history: dict[str, list[FingerprintRecord]] = {}
        # Fast lookup set: agent_id -> set of state_hash strings
        self._hash_lookup: dict[str, set[str]] = {}

        logger.debug(
            "TopologicalFingerprintDB initialized (Algorithm=%s, Scope=%s)",
            self.hash_algorithm,
            self.scope,
        )

    def compute_fingerprint(self, violations: Sequence[ViolationInstance]) -> str:
        """
        Computes the canonical SHA-256 digest for an attributed violation subset.
        
        Formula (Eq. 9):
            F_t^{(i)} = Hash(Sort({sig(v) | v ∈ V_i^{(t)}}))
        """
        if not violations:
            return hashlib.sha256(b"<NO_VIOLATIONS>").hexdigest()

        unique_signatures = {v.signature for v in violations}
        sorted_signatures = sorted(unique_signatures)
        serialized_payload = "|".join(sorted_signatures).encode("utf-8")

        if self.hash_algorithm == "sha256":
            return hashlib.sha256(serialized_payload).hexdigest()
        elif self.hash_algorithm == "md5":
            return hashlib.md5(serialized_payload).hexdigest()
        else:
            raise ValueError(f"Unsupported hash algorithm: '{self.hash_algorithm}'")

    @staticmethod
    def detect_cycle(history_hashes: Sequence[str], current_hash: str) -> bool:
        """
        Implements Algorithm 1 Line 13: DetectCycle(FingerprintDB[i], F_t^{(i)}).
        Flags an oscillation cycle if F_t^{(i)} = F_{t-m}^{(i)} for some m >= 2
        with F_t^{(i)} != F_{t-1}^{(i)} (non-adjacent state cycle X -> Y -> X).
        Consecutive identical states (X -> X) represent incomplete repairs and are
        handled by the standard retry budget K.
        """
        if len(history_hashes) < 2:
            return False
        if current_hash == history_hashes[-1]:
            # X -> X is an incomplete local repair, not a multi-state oscillation cycle
            return False
        return current_hash in history_hashes[:-1]

    def record_and_check(
        self,
        agent_id: str,
        iteration: int,
        violations: Sequence[ViolationInstance],
    ) -> tuple[bool, FingerprintRecord]:
        """
        Evaluates Algorithm 1 Lines 12-16:
          1. Computes F_t^{(i)} = Hash(Sort({sig(v) | v ∈ V_i^{(t)}}))
          2. Evaluates DetectCycle(FingerprintDB[i], F_t^{(i)})
          3. Appends F_t^{(i)} to FingerprintDB[i]
          
        Returns:
            tuple[bool, FingerprintRecord]: (is_oscillating, current_record)
        """
        state_hash = self.compute_fingerprint(violations)
        signatures = tuple(sorted({v.signature for v in violations}))

        record = FingerprintRecord(
            agent_id=agent_id,
            iteration=iteration,
            state_hash=state_hash,
            violation_signatures=signatures,
        )

        if agent_id not in self._hash_lookup:
            self._hash_lookup[agent_id] = set()
            self._history[agent_id] = []

        known_hashes = self._hash_lookup[agent_id]
        agent_records = self._history[agent_id]
        prior_hashes = [r.state_hash for r in agent_records]

        is_cycle_detected = self.detect_cycle(prior_hashes, state_hash)

        if is_cycle_detected:
            trajectory = self.get_oscillation_trajectory(agent_id, state_hash)
            logger.warning(
                "Oscillation Trap Confirmed (Algorithm 1, Line 13): "
                "Agent %s entered non-adjacent state cycle %s at iteration %d. Trajectory: %s",
                agent_id,
                state_hash[:10],
                iteration,
                " -> ".join(trajectory),
            )
            self._history[agent_id].append(record)
            known_hashes.add(state_hash)
            return True, record

        known_hashes.add(state_hash)
        self._history[agent_id].append(record)

        logger.debug(
            "Enrolled state fingerprint for Agent %s (Iter %d): Hash=%s",
            agent_id,
            iteration,
            state_hash[:10],
        )
        return False, record

    def is_oscillating(self, agent_id: str, state_hash: str) -> bool:
        """Checks if state_hash forms a non-adjacent cycle in agent_id's history."""
        records = self._history.get(agent_id, [])
        return self.detect_cycle([r.state_hash for r in records], state_hash)

    def get_oscillation_trajectory(self, agent_id: str, recurring_hash: str) -> list[str]:
        """
        Reconstructs the sequence of state hashes from the initial occurrence of
        recurring_hash to its re-emergence (Cycle extraction).
        """
        records = self._history.get(agent_id, [])
        if not records:
            return [recurring_hash[:8]]

        trajectory: list[str] = []
        cycle_started = False

        for r in records:
            if r.state_hash == recurring_hash:
                cycle_started = True
            if cycle_started:
                trajectory.append(r.state_hash[:8])

        trajectory.append(recurring_hash[:8])
        return trajectory

    def get_agent_history(self, agent_id: str) -> list[FingerprintRecord]:
        """Returns the complete immutable timeline of an agent's structural iterations."""
        return list(self._history.get(agent_id, []))

    def clear(self) -> None:
        """Resets the fingerprint database across multi-agent task boundaries."""
        self._history.clear()
        self._hash_lookup.clear()

    def export_telemetry(self) -> dict[str, Any]:
        """
        Exports structured telemetry supporting empirical analysis of RQ3.
        """
        telemetry: dict[str, Any] = {
            "total_agents_tracked": len(self._history),
            "total_states_recorded": sum(len(records) for records in self._history.values()),
            "oscillating_agents_count": 0,
            "trajectories": {},
        }

        for agent_id, records in self._history.items():
            hashes = [r.state_hash for r in records]
            has_non_adjacent_cycle = False
            for idx in range(2, len(hashes)):
                if hashes[idx] != hashes[idx - 1] and hashes[idx] in hashes[: idx - 1]:
                    has_non_adjacent_cycle = True
                    break
            if has_non_adjacent_cycle:
                telemetry["oscillating_agents_count"] += 1
                telemetry["trajectories"][agent_id] = [r.state_hash[:10] for r in records]

        return telemetry


# Backward-compatible alias
CryptographicFingerprintDB = TopologicalFingerprintDB
