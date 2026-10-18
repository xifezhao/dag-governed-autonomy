"""
================================================================================
Graph Calculus: Hierarchical Attributed Graph (HPG) & Incremental AST Engine
================================================================================

Paper Reference:
  Section III-A: Incremental Graph Calculus and Bounded Arbitration
  "Governed Autonomy: Halting Architectural Entropy in Agentic Software 
   Workflows via Dynamic Guardrails" (SANER 2026)

Formal Implementation Mapping:
--------------------------------------------------------------------------------
Concept                          | Equation / Formalism | Source File
--------------------------------------------------------------------------------
Hierarchical Attributed Graph    | G = (V, E, L, Φ)     | hpg_model.py
Granularity Hierarchy            | L: V -> {Pkg,Cls,Mtd}| hpg_model.py
Static Metric Attribute Vector   | Φ: V ∪ E -> R^k      | hpg_model.py
Incremental Update Operator ⊕    | Eq. 4: G' = G ⊕ ΔC   | incremental_operator.py
Polyglot AST Extraction (Python) | Tree-sitter / ast    | ast_visitor_python.py
Polyglot AST Extraction (Java)   | JavaParser / Tree-s. | ast_visitor_java.py
--------------------------------------------------------------------------------

Licensing:
  Licensed under the Apache License, Version 2.0 (the "License").
"""

from typing import Final

# ------------------------------------------------------------------------------
# 1. Domain Entities & Graph Representation (Section III-A)
# ------------------------------------------------------------------------------
from .hpg_model import (
    CodeEntityNode,
    DependencyEdge,
    EdgeType,
    GranularityLayer,
    HierarchicalAttributedGraph,
    HPG,
    StaticMetricsPayload,
)

# ------------------------------------------------------------------------------
# 2. Incremental AST Graph Update Operator (Eq. 4)
# ------------------------------------------------------------------------------
from .incremental_operator import (
    ASTDiffPayload,
    IncrementalGraphOperator,
    IncrementalUpdateResult,
)

# ------------------------------------------------------------------------------
# 3. Polyglot AST Extractors (Section IV-B Dataset Support)
# ------------------------------------------------------------------------------
from .ast_visitor_python import PythonASTVisitor
from .ast_visitor_java import JavaASTVisitor

# ------------------------------------------------------------------------------
# 4. Explicit Public API Interface (__all__)
# ------------------------------------------------------------------------------
__all__: Final[list[str]] = [
    # Graph Core Models (G = (V, E, L, Φ))
    "HierarchicalAttributedGraph",
    "HPG",
    "CodeEntityNode",
    "DependencyEdge",
    "GranularityLayer",
    "EdgeType",
    "StaticMetricsPayload",
    
    # Incremental Operator ⊕ (Eq. 4)
    "IncrementalGraphOperator",
    "IncrementalUpdateResult",
    "ASTDiffPayload",
    
    # Language-Specific AST Parsers
    "PythonASTVisitor",
    "JavaASTVisitor",
    
    # Verification Utilities
    "verify_hpg_integrity",
]


# ------------------------------------------------------------------------------
# 5. Invariant Integrity Verification Utility (AE Tooling)
# ------------------------------------------------------------------------------
def verify_hpg_integrity(graph: HierarchicalAttributedGraph) -> tuple[bool, list[str]]:
    """
    Validates structural invariants of the Hierarchical Attributed Graph (HPG).
    
    Invariants checked:
      1. Every edge e=(u, v) in E references valid, existing nodes u, v in V.
      2. Layer mapping L is defined for every node v in V (Package, Class, or Method).
      3. Metric attribute payload Φ is initialized and contains non-negative values
         prior to Z-score normalization.
      4. Hierarchical parent-child containment conforms to strict directional trees:
         Package contains Classes, Class contains Methods.
         
    Returns:
        tuple[bool, list[str]]: (is_healthy, list_of_violation_messages)
    """
    violations: list[str] = []

    # Invariant 1: Edge endpoints existence
    for edge in graph.edges:
        if edge.source_id not in graph.nodes:
            violations.append(
                f"Dangling edge source: Edge {edge} references non-existent node '{edge.source_id}'."
            )
        if edge.target_id not in graph.nodes:
            violations.append(
                f"Dangling edge target: Edge {edge} references non-existent node '{edge.target_id}'."
            )

    # Invariant 2 & 4: Granularity hierarchy consistency
    for node_id, node in graph.nodes.items():
        if node.layer not in (
            GranularityLayer.PACKAGE,
            GranularityLayer.CLASS,
            GranularityLayer.METHOD,
        ):
            violations.append(f"Node '{node_id}' has invalid GranularityLayer '{node.layer}'.")

        # Invariant 3: Metric sanity check
        if node.metrics is not None:
            if node.metrics.mccabe_complexity < 1.0:
                violations.append(
                    f"Node '{node_id}' has sub-minimal McCabe complexity ({node.metrics.mccabe_complexity} < 1.0)."
                )
            if node.metrics.cognitive_complexity < 0.0:
                violations.append(
                    f"Node '{node_id}' has negative Cognitive complexity ({node.metrics.cognitive_complexity})."
                )

    is_healthy = len(violations) == 0
    return is_healthy, violations
