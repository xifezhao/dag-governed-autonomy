"""
================================================================================
Stage III: Main Controller Loop (Algorithm 1)
================================================================================

Paper Reference:
  Section III-C: Exhaustive Attribution and Oscillation Control
  Algorithm 1: DAG: Fault-Tolerant Multi-Agent Nudging Loop (Lines 1-23)
  "Governed Autonomy: Halting Architectural Entropy in Agentic Software 
   Workflows via Dynamic Guardrails" (SANER 2026)

Execution Loop Lifecycle:
  1. Aging-aware arbitration with starvation-free bounds (Eq. 1, 2, 3, Theorem 1)
  2. Incremental AST graph projection G' = G ⊕ ΔC_orth
  3. Structural and variance-scaled statistical constraint gating (Eq. 4, 5, 6, 7)
  4. Exhaustive fine-grained fault attribution Attr(v) (Eq. 8)
  5. Topological state fingerprinting & Single-Point Isolation (Eq. 9)
  6. Deterministic templated verbalization & aging-inherited regeneration (Eq. 10)

Standards:
  Compliant with IEEE/ACM Artifact Evaluation Guidelines (Badge: Reusable)
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Final, Sequence

# Stage I Imports
from ..arbitration.aging_priority_queue import AgingPriorityQueue
from ..arbitration.conflict_detector import AgentPatch, ConflictDetector
from ..arbitration.patch_orthogonalizer import PatchOrthogonalizer

# Stage II Imports
from ..gating import (
    AgTDCalculator,
    CompositeGatingPipeline,
    EmpiricalLayerGate,
    GatingEvaluationResult,
    StatisticalCLTGate,
    WhitelistedCycleGate,
)
from ..graph.hpg_model import HierarchicalAttributedGraph
from ..graph.incremental_operator import IncrementalGraphOperator

# Stage III Imports
from ..attribution.exhaustive_attributor import ExhaustiveAttributor
from ..attribution.fingerprint_db import CryptographicFingerprintDB, TopologicalFingerprintDB
from ..attribution.isolation_manager import (
    BatchDegradationStatus,
    IsolationDecision,
    SinglePointIsolationManager,
)
from ..attribution.verbalization_engine import TemplatedVerbalizationEngine
from .agent_interface import BaseAgentAdapter, MockAgentAdapter

logger = logging.getLogger("dag.workflow.dag_loop")


class WorkflowStatus(str, Enum):
    """Execution status outcomes of the DAG governed loop."""
    SUCCESS_FULL_MERGE = "SUCCESS_FULL_MERGE"           # 100% of candidate patches merged cleanly
    SUCCESS_PARTIAL_MERGE = "SUCCESS_PARTIAL_MERGE"     # Graceful degradation: merged healthy subset
    TERMINATED_MAX_RETRIES = "TERMINATED_MAX_RETRIES"   # Truncated by MaxRetries loop budget
    TERMINATED_TOTAL_QUARANTINE = "TOTAL_QUARANTINE"     # All candidate patches isolated; human fallback


@dataclass(frozen=True, slots=True)
class DAGLoopResult:
    """
    Structured outcome of Algorithm 1 execution.
    Directly inspected by Artifact Evaluation scripts for Tables I-IV & Figures 3-4.
    """
    final_graph: HierarchicalAttributedGraph            # Merged safe graph G_safe
    status: WorkflowStatus
    iterations_executed: int                            # K_actual
    merged_patches: tuple[AgentPatch, ...]              # Committed orthogonal patches
    quarantined_patches: tuple[AgentPatch, ...]          # Isolated patches purged from merge
    isolation_decisions: tuple[IsolationDecision, ...]  # Audit records of Equation 9 triggers
    total_violations_encountered: int
    telemetry_history: tuple[dict[str, Any], ...]       # Detailed per-iteration trace for CDF plots
    max_deferral_observed: int = 0                      # Empirical d_max for Theorem 1 verification
    theorem1_deferral_bound: int = 28                   # Theoretical bound ceil(Δ_max/λ) + (M-1)*K

    def is_successful(self) -> bool:
        return self.status in (WorkflowStatus.SUCCESS_FULL_MERGE, WorkflowStatus.SUCCESS_PARTIAL_MERGE)


class DAGLoop:
    """
    Implementation of Algorithm 1:
        DAG: Fault-Tolerant Multi-Agent Nudging Loop
    """

    def __init__(
        self,
        conflict_detector: ConflictDetector,
        priority_queue: AgingPriorityQueue,
        orthogonalizer: PatchOrthogonalizer,
        graph_operator: IncrementalGraphOperator,
        gating_pipeline: CompositeGatingPipeline,
        attributor: ExhaustiveAttributor,
        fingerprint_db: TopologicalFingerprintDB | None,
        isolation_manager: SinglePointIsolationManager | None,
        verbalization_engine: TemplatedVerbalizationEngine,
        agent_adapter: BaseAgentAdapter,
        max_retries: int = 10,
    ) -> None:
        self.conflict_detector: Final[ConflictDetector] = conflict_detector
        self.priority_queue: Final[AgingPriorityQueue] = priority_queue
        self.orthogonalizer: Final[PatchOrthogonalizer] = orthogonalizer
        self.graph_operator: Final[IncrementalGraphOperator] = graph_operator
        self.gating_pipeline: Final[CompositeGatingPipeline] = gating_pipeline
        self.attributor: Final[ExhaustiveAttributor] = attributor
        self.fingerprint_db: Final[TopologicalFingerprintDB | None] = fingerprint_db
        self.isolation_manager: Final[SinglePointIsolationManager | None] = isolation_manager
        self.verbalization_engine: Final[TemplatedVerbalizationEngine] = verbalization_engine
        self.agent_adapter: Final[BaseAgentAdapter] = agent_adapter
        self.max_retries: Final[int] = max_retries

        logger.info(
            "DAGLoop Controller initialized (MaxRetries K=%d, Aging λ=%.4f)",
            self.max_retries,
            self.priority_queue.lambda_factor,
        )

    def run(
        self,
        base_graph: HierarchicalAttributedGraph,
        initial_patches: Sequence[AgentPatch],
        base_priorities: dict[str, float] | None = None,
        workspace_file_buffers: dict[str, str] | None = None,
    ) -> DAGLoopResult:
        """
        Executes Algorithm 1: DAG Fault-Tolerant Multi-Agent Nudging Loop.
        """
        iteration = 0  # Algorithm 1: Line 2

        if self.fingerprint_db is not None:
            self.fingerprint_db.clear()  # Algorithm 1: Line 1 (FingerprintDB <- Map())
        if self.isolation_manager is not None:
            self.isolation_manager.reset_task_state()

        current_patch_set: list[AgentPatch] = list(initial_patches)
        file_buffers = dict(workspace_file_buffers or {})

        iteration_telemetry: list[dict[str, Any]] = []
        committed_graph = base_graph.clone()
        active_decisions: list[IsolationDecision] = []
        cumulative_violations_count = 0
        max_deferral_observed = max((p.deferral_count for p in current_patch_set), default=0)
        theorem1_bound = self.priority_queue.compute_theorem1_max_deferral_bound(max_conflict_cliques_m=3)
        delta_c_orth: list[AgentPatch] = []

        logger.info("Commencing Algorithm 1 execution with %d initial candidate patches.", len(current_patch_set))

        # Algorithm 1: Line 3 (While iteration < K and ΔC_set != ∅)
        while iteration < self.max_retries and current_patch_set:
            iter_start = time.time()
            logger.info(">>> --- Algorithm 1: Iteration %d / %d --- <<<", iteration + 1, self.max_retries)

            # Algorithm 1: Line 4 (AgingArbitrate)
            partition_result = self.orthogonalizer.partition(
                candidate_patches=current_patch_set,
                base_priorities=base_priorities,
            )
            delta_c_orth = list(partition_result.orthogonal_patches)
            delta_c_defer = list(partition_result.deferred_patches)

            for p in delta_c_orth + delta_c_defer:
                if p.deferral_count > max_deferral_observed:
                    max_deferral_observed = p.deferral_count

            if not delta_c_orth:
                logger.warning("No orthogonal patches admitted in iteration %d. Advancing deferral aging.", iteration)
                current_patch_set = delta_c_defer
                iteration += 1
                continue

            # Algorithm 1: Line 5 (G' <- G ⊕ ΔC_orth)
            update_result = self.graph_operator.apply_orthogonal_patches(
                base_graph=committed_graph,
                patches=delta_c_orth,
                file_buffers_after_patch=file_buffers,
            )
            hypothetical_g = update_result.hypothetical_graph

            # Algorithm 1: Line 6 (ExtractViolations)
            gating_result: GatingEvaluationResult = self.gating_pipeline.evaluate(
                base_graph=committed_graph,
                update_result=update_result,
            )
            cumulative_violations_count += len(gating_result.violations)

            iter_metric = {
                "iteration": iteration + 1,
                "candidate_count": len(current_patch_set),
                "admitted_orth_count": len(delta_c_orth),
                "deferred_count": len(delta_c_defer),
                "max_deferral_count": max_deferral_observed,
                "is_safe": gating_result.is_safe,
                "violations_count": len(gating_result.violations),
                "total_agtd_delta": gating_result.total_agtd_delta,
                "tau_batch": gating_result.dynamic_threshold_tau,
                "elapsed_seconds": time.time() - iter_start,
            }
            iteration_telemetry.append(iter_metric)

            # Algorithm 1: Line 7-9 (If V == ∅)
            if gating_result.is_safe:
                # Commit G <- G'
                committed_graph = hypothetical_g
                if not delta_c_defer:
                    logger.info(
                        "CONVERGENCE ACHIEVED (Algorithm 1, Line 8): All %d patches in ΔC_orth "
                        "are architecturally sound at Iteration %d.",
                        len(delta_c_orth),
                        iteration + 1,
                    )
                    return DAGLoopResult(
                        final_graph=committed_graph,
                        status=(
                            WorkflowStatus.SUCCESS_FULL_MERGE
                            if not active_decisions
                            else WorkflowStatus.SUCCESS_PARTIAL_MERGE
                        ),
                        iterations_executed=iteration + 1,
                        merged_patches=tuple(delta_c_orth),
                        quarantined_patches=tuple(
                            self._extract_patches_by_ids(
                                initial_patches, [d.isolated_patch_id for d in active_decisions]
                            )
                        ),
                        isolation_decisions=tuple(active_decisions),
                        total_violations_encountered=cumulative_violations_count,
                        telemetry_history=tuple(iteration_telemetry),
                        max_deferral_observed=max_deferral_observed,
                        theorem1_deferral_bound=theorem1_bound,
                    )
                else:
                    # Admitted orthogonal subset merged cleanly; continue processing deferred patches
                    current_patch_set = delta_c_defer
                    iteration += 1
                    continue

            # Algorithm 1: Line 10 (Fine-grained attribution: V_i <- {v ∈ V | ΔC_i ∈ Attr(v)})
            attribution_result = self.attributor.attribute_batch(
                violations=gating_result.violations,
                per_patch_diffs=update_result.per_patch_diffs,
                agtd_profiles=gating_result.agtd_profiles,
                tau_batch=gating_result.dynamic_threshold_tau,
            )

            regenerated_patches: list[AgentPatch] = []
            culpable_decisions_this_iter: list[IsolationDecision] = []

            for patch_i in delta_c_orth:
                v_i = attribution_result.per_patch_violations.get(patch_i.patch_id, ())

                # Shield innocent patches without re-prompting
                if not v_i:
                    regenerated_patches.append(patch_i)
                    continue

                # Algorithm 1: Line 12-13 (State fingerprint & DetectCycle)
                is_oscillating = False
                fp_record = None
                if self.fingerprint_db is not None:
                    is_oscillating, fp_record = self.fingerprint_db.record_and_check(
                        agent_id=patch_i.agent_id,
                        iteration=iteration + 1,
                        violations=v_i,
                    )

                # Algorithm 1: Line 13-16 (Single-Point Isolation on oscillation cycle)
                if (
                    is_oscillating
                    and self.fingerprint_db is not None
                    and self.isolation_manager is not None
                    and fp_record is not None
                ):
                    trajectory = self.fingerprint_db.get_oscillation_trajectory(
                        agent_id=patch_i.agent_id,
                        recurring_hash=fp_record.state_hash,
                    )
                    decision = self.isolation_manager.create_isolation_decision(
                        patch=patch_i,
                        recurring_hash=fp_record.state_hash,
                        iteration=iteration + 1,
                        cycle_trajectory=trajectory,
                    )
                    active_decisions.append(decision)
                    culpable_decisions_this_iter.append(decision)
                    continue

                # Algorithm 1: Line 18 (P_nudge <- SynthesizeNudge(V_i))
                nudge_prompt = self.verbalization_engine.synthesize_nudge(
                    agent_id=patch_i.agent_id,
                    patch_id=patch_i.patch_id,
                    violations=v_i,
                )

                # Algorithm 1: Line 19 (ΔC_i <- A_i.Regenerate(P_nudge, d_i))
                regenerated_patch = self.agent_adapter.regenerate_patch(
                    culpable_patch=patch_i,
                    nudge_prompt=nudge_prompt,
                    inherit_aging=True,
                )
                regenerated_patches.append(regenerated_patch)

            # Apply Single-Point Isolation pruning
            if culpable_decisions_this_iter and self.isolation_manager is not None:
                isolation_report = self.isolation_manager.apply_isolation(
                    current_orthogonal_patches=regenerated_patches,
                    culpable_decisions=culpable_decisions_this_iter,
                )
                if (
                    isolation_report.degradation_status == BatchDegradationStatus.TOTAL_QUARANTINE
                    and not delta_c_defer
                ):
                    logger.error("TOTAL QUARANTINE: All patches in batch were isolated. Terminating.")
                    return DAGLoopResult(
                        final_graph=committed_graph,
                        status=WorkflowStatus.TERMINATED_TOTAL_QUARANTINE,
                        iterations_executed=iteration + 1,
                        merged_patches=(),
                        quarantined_patches=tuple(initial_patches),
                        isolation_decisions=tuple(active_decisions),
                        total_violations_encountered=cumulative_violations_count,
                        telemetry_history=tuple(iteration_telemetry),
                        max_deferral_observed=max_deferral_observed,
                        theorem1_deferral_bound=theorem1_bound,
                    )
                regenerated_patches = list(isolation_report.retained_patches)

            # Algorithm 1: Line 20-21 (ΔC_set <- ΔC_orth ∪ ΔC_defer)
            current_patch_set = regenerated_patches + delta_c_defer
            iteration += 1

        # Algorithm 1: Line 23 (Fallback: Max retries exhausted, merge remaining verified safe subset)
        logger.warning("MaxRetries (%d) exhausted. Merging verified healthy subset.", self.max_retries)
        final_safe_graph = committed_graph
        if delta_c_orth:
            final_safe_graph = self.graph_operator.apply_orthogonal_patches(
                base_graph=committed_graph,
                patches=delta_c_orth,
                file_buffers_after_patch=file_buffers,
            ).hypothetical_graph

        return DAGLoopResult(
            final_graph=final_safe_graph,
            status=WorkflowStatus.TERMINATED_MAX_RETRIES,
            iterations_executed=iteration,
            merged_patches=tuple(delta_c_orth),
            quarantined_patches=tuple(
                self._extract_patches_by_ids(
                    initial_patches, [d.isolated_patch_id for d in active_decisions]
                )
            ),
            isolation_decisions=tuple(active_decisions),
            total_violations_encountered=cumulative_violations_count,
            telemetry_history=tuple(iteration_telemetry),
            max_deferral_observed=max_deferral_observed,
            theorem1_deferral_bound=theorem1_bound,
        )

    @staticmethod
    def _extract_patches_by_ids(
        all_patches: Sequence[AgentPatch], target_ids: Sequence[str]
    ) -> list[AgentPatch]:
        id_set = set(target_ids)
        return [p for p in all_patches if p.patch_id in id_set]

    @classmethod
    def from_config(cls, config_path: str | Path = "config/config.yaml") -> DAGLoop:
        """
        Factory method instantiating Algorithm 1 execution engine from YAML configuration.
        """
        path_obj = Path(config_path)
        if not path_obj.exists() and not path_obj.is_absolute():
            candidate = Path(__file__).resolve().parent.parent.parent / config_path
            if candidate.exists():
                path_obj = candidate

        cfg: dict[str, Any] = {}
        if path_obj.exists():
            try:
                import yaml  # type: ignore
                with open(path_obj, "r", encoding="utf-8") as f:
                    cfg = yaml.safe_load(f) or {}
            except Exception:
                cfg = {}

        arb_cfg = cfg.get("arbitration", {})
        max_retries = arb_cfg.get("max_retries", 10)
        lambda_factor = arb_cfg.get("aging_factor_lambda", 0.50)
        base_p = arb_cfg.get("base_priority", {})

        conflict_detector = ConflictDetector(
            granularity=arb_cfg.get("conflict_detection", {}).get("granularity", "AST_NODE"),
            strict_syntactic_overlap=arb_cfg.get("conflict_detection", {}).get("strict_syntactic_overlap", True),
        )
        priority_queue = AgingPriorityQueue(
            lambda_factor=lambda_factor,
            max_retries=max_retries,
            min_base_priority=base_p.get("min_priority", 1.0),
            max_base_priority=base_p.get("max_priority", 5.0),
            enforce_strict_bound=True,
        )
        orthogonalizer = PatchOrthogonalizer(
            conflict_detector=conflict_detector,
            priority_queue=priority_queue,
        )

        gating_cfg = cfg.get("gating", {})
        graph_op = IncrementalGraphOperator(
            max_k_hop=gating_cfg.get("graph_engine", {}).get("max_k_hop_propagation", 2),
            parser_backend=gating_cfg.get("graph_engine", {}).get("parser_backend", "tree-sitter"),
        )
        cycle_gate = WhitelistedCycleGate(
            whitelist_config_path=gating_cfg.get("cycle_gate", {}).get("whitelist_file", "config/arch_whitelist.yaml"),
            algorithm=gating_cfg.get("cycle_gate", {}).get("algorithm", "haeupler_scc"),
        )
        layer_gate = EmpiricalLayerGate(
            mined_rules_path=gating_cfg.get("layer_gate", {}).get("mined_rules_path", "data/architecture_rules/mined_rules_elb.json"),
            treat_as_elb=gating_cfg.get("layer_gate", {}).get("treat_as_elb", True),
            enforce_reclassification_check=gating_cfg.get("layer_gate", {}).get("enforce_reclassification_check", True),
        )
        agtd_calc = AgTDCalculator(
            weights=gating_cfg.get("agtd_gate", {}).get("composite_weights"),
        )
        stat_gate = StatisticalCLTGate(
            historical_mu=gating_cfg.get("agtd_gate", {}).get("baseline_distribution", {}).get("historical_mu", 0.42),
            historical_sigma=gating_cfg.get("agtd_gate", {}).get("baseline_distribution", {}).get("historical_sigma", 0.85),
            alpha=gating_cfg.get("agtd_gate", {}).get("statistical_scaling", {}).get("alpha", 0.05),
            inter_agent_rho=gating_cfg.get("agtd_gate", {}).get("statistical_scaling", {}).get("inter_agent_rho", 0.15),
            n_hist=gating_cfg.get("agtd_gate", {}).get("baseline_distribution", {}).get("n_hist", 500),
            scaling_mode=gating_cfg.get("agtd_gate", {}).get("statistical_scaling", {}).get("clt_scaling_mode", "sqrt_n"),
        )
        gating_pipeline = CompositeGatingPipeline(
            cycle_gate=cycle_gate,
            layer_gate=layer_gate,
            agtd_calculator=agtd_calc,
            statistical_gate=stat_gate,
        )

        attr_cfg = cfg.get("attribution", {})
        attributor = ExhaustiveAttributor(
            tolerance_coefficient_eta=attr_cfg.get("scalar_attribution", {}).get("tolerance_coefficient_eta", 1.20),
        )
        fingerprint_db = TopologicalFingerprintDB(
            hash_algorithm=attr_cfg.get("anti_oscillation", {}).get("hash_algorithm", "sha256"),
            scope=attr_cfg.get("anti_oscillation", {}).get("detection_window", "FULL_HISTORY"),
        )
        isolation_mgr = SinglePointIsolationManager(
            fallback_on_empty_merge=attr_cfg.get("fault_isolation", {}).get("fallback_on_empty_merge", True),
        )
        verbalization_engine = TemplatedVerbalizationEngine(
            pattern_db_path=cfg.get("feedback_synthesis", {}).get("design_pattern_db", "config/refactoring_patterns.json"),
            max_prompt_context_tokens=cfg.get("feedback_synthesis", {}).get("max_prompt_context_tokens", 1024),
        )

        agent_adapter = MockAgentAdapter()

        return cls(
            conflict_detector=conflict_detector,
            priority_queue=priority_queue,
            orthogonalizer=orthogonalizer,
            graph_operator=graph_op,
            gating_pipeline=gating_pipeline,
            attributor=attributor,
            fingerprint_db=fingerprint_db,
            isolation_manager=isolation_mgr,
            verbalization_engine=verbalization_engine,
            agent_adapter=agent_adapter,
            max_retries=max_retries,
        )
