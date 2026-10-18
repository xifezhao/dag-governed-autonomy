"""
================================================================================
Stage I: Conflict Pre-Check & Touched Range Analyzer
================================================================================

Paper Reference:
  Section III-A: Incremental Graph Calculus and Bounded Arbitration
  "Governed Autonomy: Halting Architectural Entropy in Agentic Software 
   Workflows via Dynamic Guardrails" (SANER 2026)

Formal Implementation:
  Eq. 1: Conflict(ΔC_i, ΔC_j) <=> Touched(ΔC_i) ∩ Touched(ΔC_j) != ∅

Standards:
  Compliant with IEEE/ACM Artifact Evaluation Guidelines (Badge: Reusable)
"""

from __future__ import annotations

import re
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Final, Iterable, Sequence

logger = logging.getLogger("dag.arbitration.conflict_detector")

# Unified Diff Hunk Header Regex: matches @@ -from_line,from_count +to_line,to_count @@
HUNK_HEADER_PATTERN: Final[re.Pattern] = re.compile(
    r"^@@\s+-(?P<old_start>\d+)(?:,(?P<old_len>\d+))?\s+\+(?P<new_start>\d+)(?:,(?P<new_len>\d+))?\s+@@"
)
FILE_HEADER_PATTERN: Final[re.Pattern] = re.compile(
    r"^\+\+\+\s+[b/]*(?P<filename>\S+)"
)


@dataclass(frozen=True, slots=True)
class CodeRange:
    """
    Represents an atomic, immutable segment of code modified by an agent.
    Can operate at physical line level or hierarchical AST node level.
    """
    file_path: str
    start_line: int
    end_line: int
    ast_node_id: str | None = None

    def __post_init__(self) -> None:
        if self.start_line > self.end_line:
            raise ValueError(
                f"Invalid line range: start_line ({self.start_line}) cannot be "
                f"strictly greater than end_line ({self.end_line}) in {self.file_path}."
            )
        # Normalize relative file path separators across POSIX and Windows
        object.__setattr__(self, "file_path", Path(self.file_path).as_posix())

    def overlaps(self, other: CodeRange) -> bool:
        """
        Determines whether two code ranges intersect.
        Intersection requires identical normalized file targets and overlapping bounds.
        """
        if self.file_path != other.file_path:
            return False

        # If explicit AST Node IDs are tracked, exact matching takes precedence
        if self.ast_node_id and other.ast_node_id:
            if self.ast_node_id == other.ast_node_id:
                return True

        # Standard physical 1D interval overlap: max(start_A, start_B) <= min(end_A, end_B)
        return max(self.start_line, other.start_line) <= min(self.end_line, other.end_line)


@dataclass(frozen=True, slots=True)
class AgentPatch:
    """
    Immutable representation of an agent's proposed patch proposal (ΔC_i).
    Encapsulates touched code ranges, raw diff, and provenance.
    """
    patch_id: str
    agent_id: str
    touched_ranges: frozenset[CodeRange] = field(default_factory=frozenset)
    raw_diff: str = ""
    deferral_count: int = 0

    @property
    def touched_files(self) -> frozenset[str]:
        files = {r.file_path for r in self.touched_ranges}
        if not files and self.raw_diff:
            for line in self.raw_diff.splitlines():
                if line.startswith("+++ b/"):
                    files.add(Path(line[6:].strip()).as_posix())
                elif line.startswith("+++ ") and not line.startswith("+++ b/"):
                    files.add(Path(line[4:].strip()).as_posix())
        return frozenset(files)


@dataclass(frozen=True, slots=True)
class PatchConflict:
    """
    Structured diagnostic report emitted when Equation 1 evaluates to True.
    """
    patch_i: str
    patch_j: str
    agent_i: str
    agent_j: str
    overlapping_ranges: tuple[tuple[CodeRange, CodeRange], ...]
    conflicting_files: frozenset[str]

    def __repr__(self) -> str:
        files = ", ".join(self.conflicting_files)
        return (
            f"<PatchConflict: {self.patch_i} (Agent {self.agent_i}) <-> "
            f"{self.patch_j} (Agent {self.agent_j}) on files: [{files}]>"
        )


class ConflictDetector:
    """
    Evaluator for Equation 1:
        Conflict(ΔC_i, ΔC_j) <=> Touched(ΔC_i) ∩ Touched(ΔC_j) != ∅
    """

    def __init__(self, granularity: str = "AST_NODE", strict_syntactic_overlap: bool = True) -> None:
        self.granularity = granularity.upper()
        self.strict_syntactic_overlap = strict_syntactic_overlap
        logger.debug(
            "ConflictDetector initialized (granularity=%s, strict=%s)",
            self.granularity,
            self.strict_syntactic_overlap,
        )

    def extract_touched_ranges_from_diff(
        self, patch_id: str, agent_id: str, diff_text: str
    ) -> AgentPatch:
        """
        Parses a unified git diff and deterministically extracts the set of Touched ranges.
        
        Handles both addition of new files, deletions, and hunk-level edits.
        """
        ranges: set[CodeRange] = set()
        current_file: str | None = None

        for line in diff_text.splitlines():
            # Check for target file change (+++ b/path/to/file.py)
            file_match = FILE_HEADER_PATTERN.match(line)
            if file_match:
                raw_filename = file_match.group("filename")
                if raw_filename != "/dev/null":
                    current_file = raw_filename
                continue

            # Check for hunk header (@@ -old_start,old_len +new_start,new_len @@)
            if current_file:
                hunk_match = HUNK_HEADER_PATTERN.match(line)
                if hunk_match:
                    new_start = int(hunk_match.group("new_start"))
                    new_len = int(hunk_match.group("new_len")) if hunk_match.group("new_len") else 1
                    
                    # For zero-length additions (e.g., pure insertions), bound line range to new_start
                    end_line = new_start + max(0, new_len - 1)
                    
                    ranges.add(
                        CodeRange(
                            file_path=current_file,
                            start_line=new_start,
                            end_line=end_line,
                        )
                    )

        if not ranges and diff_text.strip():
            logger.warning(
                "Patch %s by Agent %s produced non-empty diff but extracted zero Touched ranges.",
                patch_id,
                agent_id,
            )

        return AgentPatch(
            patch_id=patch_id,
            agent_id=agent_id,
            touched_ranges=frozenset(ranges),
            raw_diff=diff_text,
        )

    def check_conflict(self, patch_a: AgentPatch, patch_b: AgentPatch) -> PatchConflict | None:
        """
        Computes pairwise conflict check between ΔC_i and ΔC_j (Eq. 1).
        
        Returns:
            PatchConflict descriptor if overlap is found, else None.
        """
        if patch_a.patch_id == patch_b.patch_id:
            return None

        # Optimization: Early exit if file sets are disjoint
        if patch_a.touched_files.isdisjoint(patch_b.touched_files):
            return None

        overlapping_pairs: list[tuple[CodeRange, CodeRange]] = []
        conflicting_files: set[str] = set()

        for range_a in patch_a.touched_ranges:
            for range_b in patch_b.touched_ranges:
                if range_a.overlaps(range_b):
                    overlapping_pairs.append((range_a, range_b))
                    conflicting_files.add(range_a.file_path)

        if overlapping_pairs:
            return PatchConflict(
                patch_i=patch_a.patch_id,
                patch_j=patch_b.patch_id,
                agent_i=patch_a.agent_id,
                agent_j=patch_b.agent_id,
                overlapping_ranges=tuple(overlapping_pairs),
                conflicting_files=frozenset(conflicting_files),
            )

        return None

    def detect_batch_conflicts(
        self, patches: Sequence[AgentPatch]
    ) -> list[PatchConflict]:
        """
        Computes the complete symmetric conflict matrix for a batch of concurrent patches.
        """
        conflicts: list[PatchConflict] = []
        n = len(patches)

        for i in range(n):
            for j in range(i + 1, n):
                conflict = self.check_conflict(patches[i], patches[j])
                if conflict:
                    conflicts.append(conflict)

        return conflicts

    def build_conflict_graph(
        self, patches: Iterable[AgentPatch]
    ) -> dict[str, set[str]]:
        """
        Builds a conflict adjacency list representation where edges represent
        Equation 1 violations between patches. Used by PatchOrthogonalizer.
        
        Returns:
            dict[patch_id, set[conflicting_patch_ids]]
        """
        patch_list = list(patches)
        graph: dict[str, set[str]] = {p.patch_id: set() for p in patch_list}

        conflicts = self.detect_batch_conflicts(patch_list)
        for c in conflicts:
            graph[c.patch_i].add(c.patch_j)
            graph[c.patch_j].add(c.patch_i)

        return graph
