"""
================================================================================
DAG: Dynamic Architectural Guardrail for Autonomous Software Engineering
================================================================================

Replication Package for Paper:
  "Governed Autonomy: Halting Architectural Entropy in Agentic Software
   Workflows via Dynamic Guardrails"
Target Venue:
  SANER 2026 — Agentic AI systems and autonomous software engineering workflows track
Artifact Status:
  Compliant with IEEE/ACM Artifact Evaluation Guidelines (Badges: Reusable, Available)

Architectural Mapping to Paper Methodology (Section III):
--------------------------------------------------------------------------------
Submodule        | Paper Section | Theoretical Formalisms & Equations
--------------------------------------------------------------------------------
arbitration      | Section III-A | Conflict Pre-check (Eq. 1), Aging Queue (Eq. 2, 3, Thm 1)
graph            | Section III-A | Hierarchical Attributed Graph (HAG), AST Update ⊕
gating           | Section III-B | r_layer (Eq. 4), r_cycle (Eq. 5), VSG AgTD Gating (Eq. 6, 7)
attribution      | Section III-C | Exhaustive Attr(v) (Eq. 8), FingerprintDB (Eq. 9), Isolation
workflow         | Section III-C | Algorithm 1: Fault-Tolerant Multi-Agent Nudging Loop (Eq. 10)
--------------------------------------------------------------------------------

Licensing:
  Licensed under the Apache License, Version 2.0 (the "License").
"""

import logging
import sys
from typing import Final

__title__: Final[str] = "dag-governed-autonomy"
__version__: Final[str] = "1.0.0-saner2026-ae"
__author__: Final[str] = "Anonymous Authors (Double-Blind Review Stage)"
__license__: Final[str] = "Apache-2.0"
__docformat__: Final[str] = "restructuredtext en"

MIN_PYTHON_VERSION: Final[tuple[int, int]] = (3, 10)


def _assert_runtime_environment() -> None:
    """Validates runtime compatibility to ensure seamless Artifact Evaluation."""
    current_version = sys.version_info[:2]
    if current_version < MIN_PYTHON_VERSION:
        raise RuntimeError(
            f"[{__title__}] Incompatible Python runtime: {current_version[0]}.{current_version[1]}. "
            f"DAG framework strictly requires Python >={MIN_PYTHON_VERSION[0]}.{MIN_PYTHON_VERSION[1]} "
            f"for modern AST parsing and structural pattern matching."
        )


_assert_runtime_environment()

logger: logging.Logger = logging.getLogger("dag")
if not logger.handlers:
    _handler = logging.StreamHandler(sys.stdout)
    _handler.setFormatter(
        logging.Formatter("[%(asctime)s] [%(levelname)s] [DAG] %(message)s", datefmt="%H:%M:%S")
    )
    logger.addHandler(_handler)
    logger.setLevel(logging.INFO)

from .arbitration.conflict_detector import ConflictDetector
from .arbitration.aging_priority_queue import AgingPriorityQueue
from .arbitration.patch_orthogonalizer import PatchOrthogonalizer

from .graph.hag_model import HierarchicalAttributedGraph, HAG, HPG
from .graph.incremental_operator import IncrementalGraphOperator

from .gating.cycle_gate import WhitelistedCycleGate
from .gating.layer_gate import EmpiricalLayerGate
from .gating.agtd_calculator import AgTDCalculator
from .gating.statistical_gate import StatisticalCLTGate, VarianceScaledGate

from .attribution.exhaustive_attributor import ExhaustiveAttributor
from .attribution.fingerprint_db import TopologicalFingerprintDB, CryptographicFingerprintDB
from .attribution.isolation_manager import SinglePointIsolationManager
from .attribution.verbalization_engine import TemplatedVerbalizationEngine

from .workflow.dag_loop import DAGLoop
from .workflow.agent_interface import BaseAgentAdapter

__all__: Final[list[str]] = [
    "__version__",
    "__title__",
    "__license__",
    "logger",
    "ConflictDetector",
    "AgingPriorityQueue",
    "PatchOrthogonalizer",
    "HierarchicalAttributedGraph",
    "HAG",
    "HPG",
    "IncrementalGraphOperator",
    "WhitelistedCycleGate",
    "EmpiricalLayerGate",
    "AgTDCalculator",
    "StatisticalCLTGate",
    "VarianceScaledGate",
    "ExhaustiveAttributor",
    "TopologicalFingerprintDB",
    "CryptographicFingerprintDB",
    "SinglePointIsolationManager",
    "TemplatedVerbalizationEngine",
    "DAGLoop",
    "BaseAgentAdapter",
]


def get_framework_summary() -> dict[str, str]:
    """
    Returns an architectural mapping dictionary verifying that all mathematical
    formulations (Eq. 1 through Eq. 10 & Theorem 1) are bound to active classes.
    """
    return {
        "Eq. 1 (Conflict Pre-check)": ConflictDetector.__name__,
        "Eq. 2-3 & Thm 1 (Aging Priority & Bound)": AgingPriorityQueue.__name__,
        "Operator ⊕ (Incremental HAG Update)": IncrementalGraphOperator.__name__,
        "Eq. 4 (Empirical Layer Lower Bound)": EmpiricalLayerGate.__name__,
        "Eq. 5 (Whitelisted SCC Cycle Gate)": WhitelistedCycleGate.__name__,
        "Eq. 6 (Normalized Proportional AgTD)": AgTDCalculator.__name__,
        "Eq. 7 (Variance-Scaled Batch Gating)": VarianceScaledGate.__name__,
        "Eq. 8 (Exhaustive Edge-Level Attr)": ExhaustiveAttributor.__name__,
        "Eq. 9 (Topological State Fingerprinting)": TopologicalFingerprintDB.__name__,
        "Eq. 10 (Templated Nudge Synthesis)": TemplatedVerbalizationEngine.__name__,
        "Algorithm 1 (DAG Nudging Loop)": DAGLoop.__name__,
    }
