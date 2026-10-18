"""
================================================================================
Baseline 3: Sequential Merging Runner (Serial Execution Control)
================================================================================

Paper Reference:
  Section IV-C: Baselines for Comparison (Sequential Merging Baseline)
  Table I: Empirical Comparison on AgTD, Pass@1, and Latency
  "Governed Autonomy: Halting Architectural Entropy in Agentic Software 
   Workflows via Dynamic Guardrails" (SANER 2026)

Mechanistic Profile:
  - Evaluates patches sequentially one-by-one rather than in concurrent batches.
  - Avoids emergent concurrent conflicts by serializing commits.
  - Key Trade-off: Incurs linear latency inflation O(n) and serial waiting times,
    proving why concurrent guardrail middleware is necessary.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final, Sequence

from src.arbitration.conflict_detector import AgentPatch
from src.gating.agtd_calculator import AgTDCalculator
from src.gating.cycle_gate import WhitelistedCycleGate
from src.gating.layer_gate import EmpiricalLayerGate
from src.graph.hpg_model import HierarchicalAttributedGraph
from src.graph.incremental_operator import IncrementalGraphOperator
from src.workflow.agent_interface import BaseAgentAdapter

logger = logging.getLogger("dag.baselines.sequential_runner")


@dataclass(frozen=True, slots=True)
class SequentialTaskResult:
    task_id: str
    status: str                         # "COMMITTED" | "REJECTED"
    iterations_executed: int
    functional_pass_at_1: bool
    agtd_delta: float
    wasted_tokens: int
    total_tokens_consumed: int
    wall_clock_latency_ms: float


class SequentialMergeRunner:
    """
    Executes multi-agent patches serially one-by-one to serve as a 
    concurrency-free sequential baseline.
    """

    def __init__(
        self,
        agent_adapter: BaseAgentAdapter,
        output_log_path: Path | str = "logs_and_artifacts/temp_sequential.jsonl",
    ) -> None:
        self.agent_adapter: Final[BaseAgentAdapter] = agent_adapter
        self.output_log_path: Final[Path] = Path(output_log_path)
        self.output_log_path.parent.mkdir(parents=True, exist_ok=True)

        self.graph_operator: Final[IncrementalGraphOperator] = IncrementalGraphOperator()
        self.agtd_calculator: Final[AgTDCalculator] = AgTDCalculator()
        self.cycle_checker: Final[WhitelistedCycleGate] = WhitelistedCycleGate()
        self.layer_checker: Final[EmpiricalLayerGate] = EmpiricalLayerGate()

    def execute_task(
        self,
        task_id: str,
        base_graph: HierarchicalAttributedGraph,
        patches: Sequence[AgentPatch],
    ) -> SequentialTaskResult:
        start_time = time.perf_counter()
        current_graph = base_graph.clone()
        total_tokens = 0
        wasted_tokens = 0
        all_passed = True
        accumulated_agtd = 0.0

        for p in patches:
            # Simulate sequential LLM generation overhead
            patch_tokens = 1450
            total_tokens += patch_tokens

            # Test patch individually against the latest committed state
            diff_res = self.graph_operator.apply_orthogonal_patches(current_graph, [p])
            candidate_g = diff_res.hypothetical_graph

            cycle_pass, _ = self.cycle_checker.evaluate(candidate_g)
            layer_pass, _ = self.layer_checker.evaluate(
                graph=candidate_g,
                added_edges=diff_res.aggregated_diff.edges_added,
                re_evaluated_edges=diff_res.aggregated_diff.re_evaluated_edges,
            )

            if cycle_pass and layer_pass:
                # Commit patch and advance base state
                current_graph = candidate_g
                profiles = self.agtd_calculator.compute_batch_agtd(
                    base_graph=current_graph,
                    hypothetical_graph=candidate_g,
                    per_patch_diffs=diff_res.per_patch_diffs,
                )
                accumulated_agtd += sum(prof.net_positive_delta for prof in profiles.values())
            else:
                # Patch rejected serially; agent re-attempts or drops
                all_passed = False
                wasted_tokens += patch_tokens

        latency_ms = (time.perf_counter() - start_time) * 1000.0

        return SequentialTaskResult(
            task_id=task_id,
            status="COMMITTED" if all_passed else "REJECTED",
            iterations_executed=len(patches),
            functional_pass_at_1=all_passed,
            agtd_delta=round(max(0.0, accumulated_agtd), 3),
            wasted_tokens=wasted_tokens,
            total_tokens_consumed=total_tokens,
            wall_clock_latency_ms=round(latency_ms, 2),
        )
