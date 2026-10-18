"""
================================================================================
Stage II: Hierarchical Attributed Graph (HAG) Domain Model
================================================================================

Canonical module matching Section III-A notation G = (V, E, L, Φ).
Re-exports all symbols from hpg_model for unified access.
"""

from .hpg_model import (
    CodeEntityNode,
    DependencyEdge,
    EdgeType,
    GranularityLayer,
    HAG,
    HPG,
    HierarchicalAttributedGraph,
    StaticMetricsPayload,
)

__all__ = [
    "GranularityLayer",
    "EdgeType",
    "StaticMetricsPayload",
    "DependencyEdge",
    "CodeEntityNode",
    "HierarchicalAttributedGraph",
    "HAG",
    "HPG",
]
