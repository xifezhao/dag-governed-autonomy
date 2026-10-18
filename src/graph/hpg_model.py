"""
================================================================================
Stage II: Hierarchical Attributed Graph (HAG) Domain Model
================================================================================

Paper Reference:
  Section III-A: Incremental Graph Calculus and Bounded Arbitration
  "Governed Autonomy: Halting Architectural Entropy in Agentic Software 
   Workflows via Dynamic Guardrails" (SANER 2026)

Formal Implementation:
  G = (V, E, L, Φ)
  - V : Set of heterogeneous code entities (Packages, Classes, Methods)
  - E ⊆ V × V : Directed dependency edges (Invocations, Imports, Inheritances)
  - L: V -> {Package, Class, Method} : Structural granularity mapping
  - Φ: V ∪ E -> R^k : Multi-dimensional normalized static metric payloads

Standards:
  Compliant with IEEE/ACM Artifact Evaluation Guidelines (Badge: Reusable)
"""

from __future__ import annotations

import copy
import json
import logging
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Final

logger = logging.getLogger("dag.graph.hag_model")


class GranularityLayer(str, Enum):
    """
    Formal mapping L: V -> {Package, Class, Method} (Section III-A).
    """
    PACKAGE = "PACKAGE"
    CLASS = "CLASS"
    METHOD = "METHOD"


class EdgeType(str, Enum):
    """
    Taxonomy of structural and syntactic dependencies between code entities.
    """
    IMPORT = "IMPORT"               # Package/Module-level static import dependency
    INVOKE = "INVOKE"               # Direct function or method invocation
    INHERIT = "INHERIT"             # Class extension or interface implementation
    AGGREGATE = "AGGREGATE"         # Composition / Field containment relationship
    ASSOCIATE = "ASSOCIATE"         # Indirect reference or shared association
    INVOKE_PARAM = "INVOKE_PARAM"   # Passed as method parameter (e.g., Visitor pattern)


@dataclass(slots=True)
class StaticMetricsPayload:
    """
    Attribute vector Φ(v) mapping nodes to composite static metrics (Eq. 6).
    """
    mccabe_complexity: float = 1.0     # Standard control flow cyclomatic complexity
    cognitive_complexity: float = 0.0  # Nesting/readability degradation metric (SonarQube spec)
    fan_out: int = 0                   # Number of unique outgoing structural dependencies
    fan_in: int = 0                    # Number of incoming references
    loc: int = 1                       # Physical lines of code
    z_normalized_agtd: float = 0.0     # Z-score normalized metric value: \tilde{Φ}_{cplx}(v)

    def compute_composite_agtd(
        self,
        w_mccabe: float = 0.35,
        w_cognitive: float = 0.40,
        w_fanout: float = 0.25,
    ) -> float:
        """Calculates the weighted linear aggregation of metrics."""
        return (
            (w_mccabe * self.mccabe_complexity)
            + (w_cognitive * self.cognitive_complexity)
            + (w_fanout * float(self.fan_out))
        )


@dataclass(frozen=True, slots=True)
class DependencyEdge:
    """
    Directed dependency edge e = (u, v) in E.
    """
    source_id: str
    target_id: str
    edge_type: EdgeType
    weight: float = 1.0
    metadata: dict[str, Any] = field(default_factory=dict, compare=False, hash=False)

    def to_tuple(self) -> tuple[str, str, str]:
        return (self.source_id, self.target_id, self.edge_type.value)


@dataclass(slots=True)
class CodeEntityNode:
    """
    Heterogeneous entity node v in V.
    """
    node_id: str                           # Fully-qualified identifier (e.g. "org.sample.service.UserService")
    name: str                              # Short identifier name (e.g. "UserService")
    layer: GranularityLayer                # Structural granularity L(v)
    file_path: str                         # Relative file path
    parent_id: str | None = None           # Hierarchical container node (e.g. Package ID for a Class)
    architectural_tier: str = "Unknown"    # Tier tag (e.g., "Presentation", "Domain", "Infrastructure")
    metrics: StaticMetricsPayload = field(default_factory=StaticMetricsPayload)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.file_path = Path(self.file_path).as_posix()

    def __hash__(self) -> int:
        return hash(self.node_id)

    def __eq__(self, other: Any) -> bool:
        if isinstance(other, CodeEntityNode):
            return self.node_id == other.node_id
        return False


class HierarchicalAttributedGraph:
    """
    The mathematical model G = (V, E, L, Φ) providing deterministic,
    index-accelerated graph operations, SCC decomposition, and hypothetical state branching.
    """

    def __init__(self, name: str = "Codebase_HAG") -> None:
        self.name: Final[str] = name
        self.nodes: dict[str, CodeEntityNode] = {}
        self.edges: set[DependencyEdge] = set()

        # Adjacency indexes for O(1) edge traversals
        self._outgoing: dict[str, set[DependencyEdge]] = {}
        self._incoming: dict[str, set[DependencyEdge]] = {}

    def add_node(self, node: CodeEntityNode) -> None:
        """Inserts or updates an entity node v in V."""
        self.nodes[node.node_id] = node
        if node.node_id not in self._outgoing:
            self._outgoing[node.node_id] = set()
        if node.node_id not in self._incoming:
            self._incoming[node.node_id] = set()

    def remove_node(self, node_id: str) -> None:
        """
        Removes a node v from V, automatically purging all incident edges
        to strictly prevent dangling pointers.
        """
        if node_id not in self.nodes:
            return

        incident_edges = list(self._outgoing.get(node_id, set()) | self._incoming.get(node_id, set()))
        for edge in incident_edges:
            self.remove_edge(edge)

        self.nodes.pop(node_id, None)
        self._outgoing.pop(node_id, None)
        self._incoming.pop(node_id, None)

    @staticmethod
    def _infer_tier_from_id(node_id: str) -> str:
        lower = node_id.lower()
        if any(t in lower for t in ("presentation", "controller", "ui", "view", "route", "api")):
            return "Presentation"
        if any(t in lower for t in ("domain", "service", "model", "entity", "core", "usecase")):
            return "Domain"
        if any(t in lower for t in ("infrastructure", "infra", "db", "dao", "driver", "repo", "client")):
            return "Infrastructure"
        return "Unknown"

    def add_edge(self, edge: DependencyEdge) -> None:
        """
        Inserts directed edge e = (u, v) into E.
        Ensures endpoints exist in V or registers inferred placeholders.
        """
        if edge.source_id not in self.nodes:
            self.add_node(
                CodeEntityNode(
                    node_id=edge.source_id,
                    name=edge.source_id.split(".")[-1],
                    layer=GranularityLayer.CLASS,
                    file_path="<external/placeholder>",
                    architectural_tier=self._infer_tier_from_id(edge.source_id),
                )
            )

        if edge.target_id not in self.nodes:
            self.add_node(
                CodeEntityNode(
                    node_id=edge.target_id,
                    name=edge.target_id.split(".")[-1],
                    layer=GranularityLayer.CLASS,
                    file_path="<external/placeholder>",
                    architectural_tier=self._infer_tier_from_id(edge.target_id),
                )
            )

        self.edges.add(edge)
        self._outgoing[edge.source_id].add(edge)
        self._incoming[edge.target_id].add(edge)

        if self.nodes[edge.source_id].metrics is not None:
            self.nodes[edge.source_id].metrics.fan_out = len(self._outgoing[edge.source_id])
        if self.nodes[edge.target_id].metrics is not None:
            self.nodes[edge.target_id].metrics.fan_in = len(self._incoming[edge.target_id])

    def remove_edge(self, edge: DependencyEdge) -> None:
        """Purges directed edge e from E and updates adjacency indices."""
        if edge in self.edges:
            self.edges.remove(edge)
            self._outgoing[edge.source_id].discard(edge)
            self._incoming[edge.target_id].discard(edge)

            if edge.source_id in self.nodes and self.nodes[edge.source_id].metrics:
                self.nodes[edge.source_id].metrics.fan_out = len(self._outgoing[edge.source_id])
            if edge.target_id in self.nodes and self.nodes[edge.target_id].metrics:
                self.nodes[edge.target_id].metrics.fan_in = len(self._incoming[edge.target_id])

    def get_outgoing_edges(self, node_id: str) -> frozenset[DependencyEdge]:
        """Returns all edges leaving node_id in O(1)."""
        return frozenset(self._outgoing.get(node_id, set()))

    def get_incoming_edges(self, node_id: str) -> frozenset[DependencyEdge]:
        """Returns all edges targeting node_id in O(1)."""
        return frozenset(self._incoming.get(node_id, set()))

    def get_successors(self, node_id: str) -> frozenset[str]:
        """Returns direct dependency target IDs."""
        return frozenset(e.target_id for e in self._outgoing.get(node_id, set()))

    def strongly_connected_components(self) -> list[frozenset[str]]:
        """
        Computes Strongly Connected Components (SCCs) deterministically using Tarjan's
        algorithm in O(|V| + |E|) without requiring external C/NetworkX dependencies.
        """
        index_counter = 0
        indices: dict[str, int] = {}
        lowlink: dict[str, int] = {}
        stack: list[str] = []
        on_stack: set[str] = set()
        sccs: list[frozenset[str]] = []

        def strongconnect(v: str) -> None:
            nonlocal index_counter
            indices[v] = index_counter
            lowlink[v] = index_counter
            index_counter += 1
            stack.append(v)
            on_stack.add(v)

            for w in sorted(self.get_successors(v)):
                if w not in indices:
                    strongconnect(w)
                    lowlink[v] = min(lowlink[v], lowlink[w])
                elif w in on_stack:
                    lowlink[v] = min(lowlink[v], indices[w])

            if lowlink[v] == indices[v]:
                component: set[str] = set()
                while True:
                    w = stack.pop()
                    on_stack.remove(w)
                    component.add(w)
                    if w == v:
                        break
                sccs.append(frozenset(component))

        for nid in sorted(self.nodes.keys()):
            if nid not in indices:
                strongconnect(nid)

        return sccs

    def clone(self) -> HierarchicalAttributedGraph:
        """
        Creates a fast, isolated deep copy of G for hypothetical branching G'.
        """
        new_graph = HierarchicalAttributedGraph(name=f"{self.name}_hypothetical")
        for _, node in self.nodes.items():
            new_node = CodeEntityNode(
                node_id=node.node_id,
                name=node.name,
                layer=node.layer,
                file_path=node.file_path,
                parent_id=node.parent_id,
                architectural_tier=node.architectural_tier,
                metrics=copy.deepcopy(node.metrics),
                metadata=copy.deepcopy(node.metadata),
            )
            new_graph.add_node(new_node)

        for edge in self.edges:
            new_edge = DependencyEdge(
                source_id=edge.source_id,
                target_id=edge.target_id,
                edge_type=edge.edge_type,
                weight=edge.weight,
                metadata=copy.deepcopy(edge.metadata),
            )
            new_graph.add_edge(new_edge)

        return new_graph

    def to_networkx(self) -> Any:
        """
        Converts HAG to a NetworkX DiGraph when NetworkX is installed.
        """
        import networkx as nx

        G_nx = nx.DiGraph()
        for node_id, node in self.nodes.items():
            G_nx.add_node(
                node_id,
                name=node.name,
                layer=node.layer.value,
                tier=node.architectural_tier,
                metrics=asdict(node.metrics),
            )

        for edge in self.edges:
            G_nx.add_edge(
                edge.source_id,
                edge.target_id,
                edge_type=edge.edge_type.value,
                weight=edge.weight,
            )

        return G_nx

    def to_json(self) -> str:
        """Serializes HAG topology to JSON format for Artifact Evaluation storage."""
        serializable_data = {
            "name": self.name,
            "nodes": [
                {
                    "node_id": n.node_id,
                    "name": n.name,
                    "layer": n.layer.value,
                    "file_path": n.file_path,
                    "parent_id": n.parent_id,
                    "tier": n.architectural_tier,
                    "metrics": asdict(n.metrics),
                }
                for n in self.nodes.values()
            ],
            "edges": [
                {
                    "source": e.source_id,
                    "target": e.target_id,
                    "type": e.edge_type.value,
                    "weight": e.weight,
                }
                for e in self.edges
            ],
        }
        return json.dumps(serializable_data, indent=2)

    def copy(self) -> HierarchicalAttributedGraph:
        """Alias for clone() returning a deep copy of the HAG."""
        return self.clone()

    def __len__(self) -> int:
        return len(self.nodes)

    def __repr__(self) -> str:
        return f"<HAG '{self.name}': |V|={len(self.nodes)}, |E|={len(self.edges)}>"


# Formal Aliases per Paper Notation (HAG in main text; HPG retained for backward compatibility)
HAG = HierarchicalAttributedGraph
HPG = HierarchicalAttributedGraph
HAGNode = CodeEntityNode
HPGNode = CodeEntityNode
HAGEdge = DependencyEdge
HPGEdge = DependencyEdge
NodeType = GranularityLayer
