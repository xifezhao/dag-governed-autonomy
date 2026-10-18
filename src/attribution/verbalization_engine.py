"""
================================================================================
Stage III: Templated Verbalization & Deterministic Nudge Synthesis Engine
================================================================================

Paper Reference:
  Section III-C: Exhaustive Attribution and Semantic Decoupling
  Algorithm 1: Line 18 (P_{nudge} Generation via Templated Verbalization)
  "Governed Autonomy: Halting Architectural Entropy in Agentic Software 
   Workflows via Dynamic Guardrails" (SANER 2026)

Formal Specification:
  P_{nudge}^{(i)} = ⋃_{v ∈ V_i} T(v.type) ∘ <v.context, DB_{pattern}(v.type)>

Design Invariant:
  Bypasses secondary LLM hallucination by synthesizing deterministic, 
  structured markdown prompts retrieved from an architect-vetted refactoring
  knowledge base (DB_{pattern}), guaranteeing reproducible multi-agent nudging.

Standards:
  Compliant with IEEE/ACM Artifact Evaluation Guidelines (Badge: Reusable)
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Final, Mapping, Sequence

from ..gating import ViolationInstance, ViolationType

logger = logging.getLogger("dag.attribution.verbalization_engine")


@dataclass(frozen=True, slots=True)
class RemediationPrompt:
    """
    Immutable envelope encapsulating the synthesized epistemic nudge P_{nudge}^{(i)}.
    Directly injected into Agent A_i's context during Algorithm 1 Line 19.
    """
    agent_id: str
    patch_id: str
    formatted_prompt: str                  # The final structured Markdown prompt text
    violation_count: int
    suggested_patterns: tuple[str, ...]    # Retrieved design pattern identifiers
    raw_diagnostics: tuple[dict[str, Any], ...]
    estimated_token_length: int = field(init=False)

    def __post_init__(self) -> None:
        # Heuristic token approximation: ~4 characters per token
        approx_tokens = max(1, len(self.formatted_prompt) // 4)
        object.__setattr__(self, "estimated_token_length", approx_tokens)

    def __repr__(self) -> str:
        patterns_str = ", ".join(self.suggested_patterns)
        return (
            f"<RemediationPrompt Agent={self.agent_id} Patch={self.patch_id} "
            f"Violations={self.violation_count} Patterns=[{patterns_str}] "
            f"Tokens~{self.estimated_token_length}>"
        )


class TemplatedVerbalizationEngine:
    """
    Operationalizes Equation 9 / Algorithm 1 Line 18:
        Synthesizes deterministic epistemic feedback prompts by binding
        attributed violations V_i to vetted refactoring design patterns.
    """

    # Vetted design pattern knowledge base (DB_{pattern}) fallback
    DEFAULT_PATTERN_KNOWLEDGE_BASE: Final[dict[str, dict[str, str]]] = {
        ViolationType.CYCLE.value: {
            "pattern_name": "Dependency Inversion & Shared Interface Extraction",
            "gof_reference": "Gang of Four: Structural / Behavioral Decoupling",
            "actionable_guidance": (
                "Break the cyclic dependency between the coupled modules. Extract the shared "
                "data structures or method contracts into an independent interface or utility "
                "module, or invert the dependency direction using dependency injection."
            ),
        },
        ViolationType.LAYER.value: {
            "pattern_name": "Facade / Adapter / Inversion of Control",
            "gof_reference": "Clean Architecture / Hexagonal Ports & Adapters",
            "actionable_guidance": (
                "Do not directly reference internal implementation classes across distinct "
                "architectural tiers. Introduce a boundary interface in the domain tier, "
                "or route requests through an authorized service facade."
            ),
        },
        ViolationType.AGTD_SPIKE.value: {
            "pattern_name": "Extract Method & Decompose Monolithic Scope",
            "gof_reference": "Fowler's Refactoring: Dealing with Large Classes / Long Methods",
            "actionable_guidance": (
                "Your patch introduces an impermissible spike in Cognitive or McCabe complexity. "
                "Decompose long, nested conditional blocks (if-else, loops) into dedicated, "
                "well-encapsulated private helper methods to restore modular readability."
            ),
        },
    }

    def __init__(
        self,
        pattern_db_path: Path | str | None = "config/refactoring_patterns.json",
        max_prompt_context_tokens: int = 1024,
    ) -> None:
        self.pattern_db_path = Path(pattern_db_path) if pattern_db_path else None
        self.max_tokens: Final[int] = max_prompt_context_tokens
        self.pattern_db: dict[str, dict[str, str]] = {}
        self._load_pattern_database()

        logger.debug(
            "TemplatedVerbalizationEngine initialized (MaxTokens=%d)",
            self.max_tokens,
        )

    def _load_pattern_database(self) -> None:
        """Loads refactoring pattern recommendations from configuration."""
        if not self.pattern_db_path or not self.pattern_db_path.exists():
            logger.warning(
                "Pattern database not found at %s. Initializing with default vetted GoF/Fowler patterns.",
                self.pattern_db_path,
            )
            self.pattern_db = dict(self.DEFAULT_PATTERN_KNOWLEDGE_BASE)
            return

        try:
            with open(self.pattern_db_path, "r", encoding="utf-8") as f:
                self.pattern_db = json.load(f)
            logger.info("Loaded %d refactoring patterns from %s", len(self.pattern_db), self.pattern_db_path)
        except Exception as err:
            logger.error("Failed to load pattern database JSON: %s. Reverting to defaults.", err)
            self.pattern_db = dict(self.DEFAULT_PATTERN_KNOWLEDGE_BASE)

    def retrieve_pattern(self, violation_type: ViolationType | str) -> dict[str, str]:
        """Retrieves DB_{pattern}(v.type) with defensive fallback."""
        key = violation_type.value if isinstance(violation_type, ViolationType) else str(violation_type)
        return self.pattern_db.get(
            key,
            {
                "pattern_name": "General Structural Decoupling",
                "gof_reference": "Standard Object-Oriented Principles",
                "actionable_guidance": "Refactor the modified code to preserve structural encapsulation and reduce coupling.",
            },
        )

    def synthesize_nudge(
        self,
        agent_id: str,
        patch_id: str,
        violations: Sequence[ViolationInstance],
    ) -> RemediationPrompt:
        """
        Synthesizes the targeted epistemic prompt P_{nudge}^{(i)} for a culpable agent.
        
        Parameters:
            agent_id: Identifier of the culpable agent (A_i).
            patch_id: Proposal identifier (ΔC_i).
            violations: Attributed violation subset V_i = {v ∈ V | ΔC_i ∈ Attr(v)}.
            
        Returns:
            RemediationPrompt containing the bounded, deterministic feedback.
        """
        if not violations:
            return RemediationPrompt(
                agent_id=agent_id,
                patch_id=patch_id,
                formatted_prompt="No architectural violations detected. Patch is structurally sound.",
                violation_count=0,
                suggested_patterns=(),
                raw_diagnostics=(),
            )

        patterns_used: list[str] = []
        diagnostics_list: list[dict[str, Any]] = []

        # Header: Formulates the governed autonomy intervention contract
        sections: list[str] = [
            "### [DAG Guardrail Intervention: Architectural Violations Detected]",
            f"**Target Agent:** `{agent_id}` | **Target Patch:** `{patch_id}`",
            "**Notice:** Your proposed patch passed isolated unit tests, but violates system-wide "
            "architectural contracts when merged. To halt architectural entropy, the guardrail has "
            "intercepted your commit. Please refactor your implementation adhering strictly to the "
            "deterministic remediation patterns detailed below:\n",
            "---",
        ]

        # Body: Itemized structured violations mapped to DB_{pattern}
        for idx, v in enumerate(violations, start=1):
            pattern_info = self.retrieve_pattern(v.type)
            pattern_name = pattern_info.get("pattern_name", "Decoupling")
            guidance = pattern_info.get("actionable_guidance", "")
            patterns_used.append(pattern_name)
            diagnostics_list.append(v.context)

            sections.append(f"#### Violation #{idx}: `{v.type.value.upper()}` ({v.rule_id})")
            
            # Context-specific diagnostics
            if v.type == ViolationType.CYCLE:
                nodes_str = ", ".join(f"`{n}`" for n in sorted(v.nodes))
                sections.append(f"- **Involved Component Entities:** {nodes_str}")
                sections.append(f"- **Topological Defect:** Closed dependency cycle detected across system modules.")
            elif v.type == ViolationType.LAYER:
                src_tier = v.context.get("source_tier", "Unknown")
                tgt_tier = v.context.get("target_tier", "Unknown")
                sections.append(f"- **Forbidden Dependency:** Tier `{src_tier}` -> Tier `{tgt_tier}`")
                sections.append(f"- **Rule Breached:** `{v.context.get('forbidden_rule', 'Layer Boundary')}`")
            elif v.type == ViolationType.AGTD_SPIKE:
                excess = v.context.get("total_delta", 0.0)
                tau = v.context.get("tau_batch", 0.0)
                sections.append(f"- **Observed Debt Spike:** +{excess:.3f} (Permissible CLT Threshold τ={tau:.3f})")
                sections.append(f"- **Defect:** Excessive Cognitive/McCabe complexity inflation in touched methods.")

            # Remediation Guidance from DB_{pattern}
            sections.append(f"- **Recommended Pattern:** **{pattern_name}**")
            sections.append(f"- **Prescribed Action:** {guidance}\n")

        # Footer: Operational constraints on regeneration
        sections.extend([
            "---",
            "**Operational Constraints:**",
            "1. You must maintain existing public API method signatures and functional requirements.",
            "2. Do not attempt to bypass this check using reflection, dynamic imports, or string class-loading.",
            "3. Your re-generated patch will be verified against the exact same mathematical graph constraints.",
        ])

        full_prompt = "\n".join(sections)

        # Context-length bounding: truncate if exceeding safety ceiling
        approx_tokens = len(full_prompt) // 4
        if approx_tokens > self.max_tokens:
            logger.warning(
                "Remediation prompt (~%d tokens) exceeds budget (%d). Truncating context.",
                approx_tokens,
                self.max_tokens,
            )
            cutoff = self.max_tokens * 4
            full_prompt = (
                full_prompt[:cutoff] 
                + "\n\n... [Diagnostic context truncated to fit bounded token window] ..."
            )

        logger.debug(
            "Synthesized P_nudge for Agent %s (~%d tokens, %d violations)",
            agent_id,
            len(full_prompt) // 4,
            len(violations),
        )

        return RemediationPrompt(
            agent_id=agent_id,
            patch_id=patch_id,
            formatted_prompt=full_prompt,
            violation_count=len(violations),
            suggested_patterns=tuple(sorted(set(patterns_used))),
            raw_diagnostics=tuple(diagnostics_list),
        )

    def synthesize_batch(
        self,
        per_patch_violations: Mapping[str, Sequence[ViolationInstance]],
        agent_id_map: Mapping[str, str],
    ) -> dict[str, RemediationPrompt]:
        """
        Synthesizes targeted prompts across all culpable patches in a concurrent batch.
        
        Parameters:
            per_patch_violations: Mapping: patch_id -> V_i (attributed violations).
            agent_id_map: Mapping: patch_id -> agent_id.
            
        Returns:
            dict[patch_id, RemediationPrompt]
        """
        prompts: dict[str, RemediationPrompt] = {}
        for patch_id, v_list in per_patch_violations.items():
            agent_id = agent_id_map.get(patch_id, f"Agent_{patch_id}")
            prompts[patch_id] = self.synthesize_nudge(
                agent_id=agent_id,
                patch_id=patch_id,
                violations=v_list,
            )
        return prompts
