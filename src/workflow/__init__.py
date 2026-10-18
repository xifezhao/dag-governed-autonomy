"""
================================================================================
Stage III: Main Autonomous Workflow Controller (Algorithm 1)
================================================================================

Paper Reference:
  Section III-C: Exhaustive Attribution and Semantic Decoupling
  Algorithm 1: DAG: Fault-Tolerant Multi-Agent Nudging Loop
  "Governed Autonomy: Halting Architectural Entropy in Agentic Software 
   Workflows via Dynamic Guardrails" (SANER 2026)

Formal Implementation Mapping:
--------------------------------------------------------------------------------
Component / Algorithm Step   | Formal Mechanism             | Implementation Class
--------------------------------------------------------------------------------
Line 4: AgingArbitrate       | Eq. 1, 2, 3: MWIS Aging Queue| DAGLoop & PatchOrthogonalizer
Line 5: Incremental Update ⊕ | Eq. 4: Hypothetical State G' | DAGLoop & IncrementalOperator
Line 6: ExtractViolations    | Eq. 5, 6, 7, 8: Gating Engine| DAGLoop & CompositeGating
Line 10: Attribution V_i     | Eq. 9: Bifurcated Attr(v)    | DAGLoop & ExhaustiveAttributor
Line 13: Fingerprint Check   | Eq. 10: Hash(Sort(V_i))      | DAGLoop & FingerprintDB
Line 16: Single-Point Isol.  | Drop ΔC_i (Graceful Exit)    | DAGLoop & IsolationManager
Line 18: Templated Verbalize | P_{nudge} = T(v) ∘ DB_pattern| DAGLoop & VerbalizationEngine
Line 19: Targeted Regenerate | InheritAging = True (Eq. 3)  | BaseAgentAdapter.regenerate()
--------------------------------------------------------------------------------

Licensing:
  Licensed under the Apache License, Version 2.0 (the "License").
"""

from typing import Final

# ------------------------------------------------------------------------------
# 1. Core Workflow Controller & Status Envelopes (Algorithm 1)
# ------------------------------------------------------------------------------
from .dag_loop import (
    DAGLoop,
    DAGLoopResult,
    WorkflowStatus,
)

# ------------------------------------------------------------------------------
# 2. Multi-Agent Framework Adapters & Mock Testing Tooling
# ------------------------------------------------------------------------------
from .agent_interface import (
    BaseAgentAdapter,
    MockAgentAdapter,
    AgentExecutionPayload,
)

# ------------------------------------------------------------------------------
# 3. Public Export Specifications (__all__)
# ------------------------------------------------------------------------------
__all__: Final[list[str]] = [
    # Main Workflow Loop (Algorithm 1)
    "DAGLoop",
    "DAGLoopResult",
    "WorkflowStatus",
    
    # Agent Communication Adapters
    "BaseAgentAdapter",
    "MockAgentAdapter",
    "AgentExecutionPayload",
    
    # Functional Runner Utility
    "execute_governed_workflow",
]


# ------------------------------------------------------------------------------
# 4. Functional Gateway for Zero-Configuration AE Review
# ------------------------------------------------------------------------------
def execute_governed_workflow(
    config_path: str = "config/config.yaml",
    dry_run: bool = False,
) -> DAGLoop:
    """
    Factory initializer constructing a fully wired Algorithm 1 execution engine
    from a YAML configuration file.
    
    Usage for Artifact Reviewers:
      >>> from src.workflow import execute_governed_workflow
      >>> runner = execute_governed_workflow("config/config.yaml")
      >>> assert runner.max_retries == 10
    """
    loop = DAGLoop.from_config(config_path)
    if dry_run:
        loop.logger.info("DAG governed workflow initialized in DRY-RUN mode.")
    return loop
