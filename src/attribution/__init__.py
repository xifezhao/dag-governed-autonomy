"""
================================================================================
Stage III: Exhaustive Attribution, Anti-Oscillation & Fault Isolation
================================================================================

Paper Reference:
  Section III-C: Exhaustive Attribution and Semantic Decoupling
  Algorithm 1: Lines 9-27 (Fine-Grained Targeted Regeneration & Isolation)
  "Governed Autonomy: Halting Architectural Entropy in Agentic Software 
   Workflows via Dynamic Guardrails" (SANER 2026)

Formal Implementation Mapping:
--------------------------------------------------------------------------------
Concept                          | Equation / Formalism | Source File
--------------------------------------------------------------------------------
Bifurcated Attribution Mapping   | Eq. 9: Attr(v)       | exhaustive_attributor.py
Cryptographic State Fingerprint  | Eq. 10: F_{state}^(i)| fingerprint_db.py
Anti-Oscillation & Livelock Trap | Algorithm 1 Line 14  | fingerprint_db.py
Single-Point Fault Isolation     | Algorithm 1 Line 16  | isolation_manager.py
Deterministic Nudge Synthesis    | P_{nudge} = T ∘ DB   | verbalization_engine.py
--------------------------------------------------------------------------------

Licensing:
  Licensed under the Apache License, Version 2.0 (the "License").
"""

from typing import Final

# ------------------------------------------------------------------------------
# 1. Internal Submodule Imports
# ------------------------------------------------------------------------------
from .exhaustive_attributor import (
    AttributionResult,
    ExhaustiveAttributor,
)
from .fingerprint_db import (
    CryptographicFingerprintDB,
    FingerprintRecord,
    OscillationTrapDetected,
)
from .isolation_manager import (
    BatchDegradationStatus,
    IsolationDecision,
    SinglePointIsolationManager,
)
from .verbalization_engine import (
    RemediationPrompt,
    TemplatedVerbalizationEngine,
)

# ------------------------------------------------------------------------------
# 2. Public Export Interface (__all__)
# ------------------------------------------------------------------------------
__all__: Final[list[str]] = [
    # Eq. 9: Exhaustive Fault Attribution
    "ExhaustiveAttributor",
    "AttributionResult",
    
    # Eq. 10: Anti-Oscillation Cryptographic State Tracking
    "CryptographicFingerprintDB",
    "FingerprintRecord",
    "OscillationTrapDetected",
    
    # Single-Point Fault Isolation Protocol
    "SinglePointIsolationManager",
    "IsolationDecision",
    "BatchDegradationStatus",
    
    # Deterministic Epistemic Nudging
    "TemplatedVerbalizationEngine",
    "RemediationPrompt",
    
    # Diagnostic Helpers
    "compute_state_hash",
]


# ------------------------------------------------------------------------------
# 3. Cryptographic State Hash Utility (Direct AE Tooling)
# ------------------------------------------------------------------------------
def compute_state_hash(signatures: list[str] | set[str], algorithm: str = "sha256") -> str:
    """
    Formally evaluates Equation 10:
        F_{state} = Hash(Sort({v.signature | ΔC_i ∈ Attr(v)}))
        
    Guarantees deterministic, collision-resistant string digests for 
    detecting recurring 'whack-a-mole' oscillation trajectories.
    """
    import hashlib

    sorted_signatures = sorted(signatures)
    raw_payload = "|".join(sorted_signatures).encode("utf-8")
    
    if algorithm.lower() == "sha256":
        return hashlib.sha256(raw_payload).hexdigest()
    elif algorithm.lower() == "md5":
        return hashlib.md5(raw_payload).hexdigest()
    else:
        raise ValueError(f"Unsupported cryptographic hash algorithm: '{algorithm}'")
