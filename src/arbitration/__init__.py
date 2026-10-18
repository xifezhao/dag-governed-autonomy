"""
================================================================================
Stage I: Submission & Bounded Arbitration Submodule
================================================================================

Paper Reference:
  Section III-A: Incremental Graph Calculus and Bounded Arbitration
  "Governed Autonomy: Halting Architectural Entropy in Agentic Software 
   Workflows via Dynamic Guardrails" (SANER 2026)

Theoretical Formalisms Encapsulated:
--------------------------------------------------------------------------------
Equation | Concept                        | Implementation Class
--------------------------------------------------------------------------------
Eq. 1    | Syntactic Conflict Pre-check   | ConflictDetector
Eq. 2    | Aging-Aware Priority Dispatch  | AgingPriorityQueue
Eq. 3    | Starvation-Free Lambda Bound   | AgingPriorityQueue, verify_lambda_bound()
N/A      | Orthogonal Partitioning (ΔC)   | PatchOrthogonalizer
--------------------------------------------------------------------------------

Licensing:
  Licensed under the Apache License, Version 2.0 (the "License").
"""

from typing import Final

# ------------------------------------------------------------------------------
# 1. Internal Imports from Submodule Implementations
# ------------------------------------------------------------------------------
from .conflict_detector import (
    ConflictDetector,
    CodeRange,
    PatchConflict,
)
from .aging_priority_queue import (
    AgingPriorityQueue,
    ArbitrationEntry,
    StarvationBoundViolation,
)
from .patch_orthogonalizer import (
    PatchOrthogonalizer,
    OrthogonalPartitionResult,
)

# ------------------------------------------------------------------------------
# 2. Public Export Specification (__all__)
# ------------------------------------------------------------------------------
__all__: Final[list[str]] = [
    # Eq. 1: Conflict Detection
    "ConflictDetector",
    "CodeRange",
    "PatchConflict",
    
    # Eq. 2 & 3: Aging Arbitration Queue
    "AgingPriorityQueue",
    "ArbitrationEntry",
    "StarvationBoundViolation",
    "verify_lambda_bound",
    
    # Batch Orthogonalization Pipeline
    "PatchOrthogonalizer",
    "OrthogonalPartitionResult",
]


# ------------------------------------------------------------------------------
# 3. Mathematical Verification Utility (Direct AE Reviewer Tooling)
# ------------------------------------------------------------------------------
def verify_lambda_bound(
    min_base_priority: float,
    max_base_priority: float,
    max_retries: int,
    lambda_factor: float,
) -> tuple[bool, float]:
    """
    Formally verifies compliance with Equation 3 (Starvation-Free Lower Bound).
    
    Theorem (Eq. 3):
        To guarantee that initial priority discrepancies are neutralized strictly
        within the execution budget (before MaxRetries is exhausted), the aging
        factor λ must satisfy:
            λ >= (max(BasePriority) - min(BasePriority)) / (MaxRetries - 1)
            
    Parameters:
        min_base_priority: Lower bound of base priority assigned to any agent.
        max_base_priority: Upper bound of base priority assigned to any agent.
        max_retries: Total iteration loop budget (K in Algorithm 1).
        lambda_factor: The configured aging factor λ to test.
        
    Returns:
        tuple[bool, float]: (is_valid, theoretical_minimum_lambda)
        
    Example:
        >>> is_valid, min_lambda = verify_lambda_bound(1.0, 5.0, 10, 0.50)
        >>> assert is_valid and min_lambda == pytest.approx(0.4444, rel=1e-3)
    """
    if max_retries <= 1:
        raise ValueError(f"MaxRetries must be >= 2 for iterative aging (got {max_retries}).")
    if max_base_priority < min_base_priority:
        raise ValueError("max_base_priority cannot be strictly less than min_base_priority.")
        
    delta_max: float = max_base_priority - min_base_priority
    theoretical_minimum_lambda: float = delta_max / float(max_retries - 1)
    
    is_compliant: bool = lambda_factor >= theoretical_minimum_lambda
    return is_compliant, theoretical_minimum_lambda
