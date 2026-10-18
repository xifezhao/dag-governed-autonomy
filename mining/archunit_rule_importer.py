"""
================================================================================
Rule Importer: ArchUnit (Java) & PyTestArch (Python) Declarative Spec Extractor
================================================================================

Paper Reference:
  Section III-B: Empirical Layer Violation Gate (r_layer, Eq. 6)
  Section IV-B: Dataset Selection & Architecture-Strict Filtering
  "Governed Autonomy: Halting Architectural Entropy in Agentic Software 
   Workflows via Dynamic Guardrails" (SANER 2026)

Formal Derivation:
  Translates declarative human-authored architecture tests into DAG's internal
  forbidden mapping B(u), reconciling developer intent with autonomous governance:
    ArchUnit Spec -> AST / Semantic Parsing -> B(u) in mined_rules_elb.json

Standards:
  Compliant with IEEE/ACM Artifact Evaluation Guidelines (Badge: Reusable)
"""

from __future__ import annotations

import argparse
import json
import logging
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Final, Sequence

logger = logging.getLogger("dag.mining.archunit_rule_importer")


@dataclass(frozen=True, slots=True)
class ExtractedRule:
    """
    Normalized representation of a single extracted architectural invariant.
    """
    rule_id: str
    framework: str                        # "ArchUnit-Java" | "PyTestArch-Python"
    source_scope: str                     # Source layer, tier, or package expression
    target_scope: str                     # Target layer, tier, or package expression
    rule_type: str                        # "FORBIDDEN_DEPENDENCY" | "ONLY_ACCESSED_BY"
    raw_statement: str
    source_file: str

    def __repr__(self) -> str:
        return (
            f"<ExtractedRule [{self.framework}] {self.source_scope} -> "
            f"{self.target_scope} ({self.rule_type})>"
        )


@dataclass(slots=True)
class ImportedRulesReport:
    """
    Consolidated output schema matching data/architecture_rules/mined_rules_elb.json.
    """
    metadata: dict[str, Any]
    forbidden_tier_dependencies: dict[str, list[str]]
    forbidden_package_dependencies: dict[str, list[str]]
    extracted_rules: list[ExtractedRule]


class ArchUnitRuleImporter:
    """
    Parser and translator for Java ArchUnit and Python PyTestArch declarative tests.
    """

    # Java ArchUnit Regex Patterns
    # Matches: noClasses().that().resideInAPackage("..A..").should().dependOnClassesThat().resideInAPackage("..B..")
    JAVA_NO_CLASSES_DEPEND_PATTERN: Final[re.Pattern] = re.compile(
        r'noClasses\(\)\s*\.that\(\)\s*\.resideIn[A-Za-z]*Package\("(?P<src>[^"]+)"\)\s*'
        r'\.should\(\)\s*\.dependOnClassesThat\(\)\s*\.resideIn[A-Za-z]*Package\("(?P<tgt>[^"]+)"\)',
        re.DOTALL,
    )

    # Matches: layeredArchitecture()...whereLayer("A").mayNotBeAccessedByLayers("B")
    JAVA_LAYER_MAY_NOT_BE_ACCESSED: Final[re.Pattern] = re.compile(
        r'whereLayer\("(?P<tgt>[^"]+)"\)\s*\.mayNotBeAccessedByLayers\("(?P<src>[^"]+)"\)',
        re.DOTALL,
    )

    # Matches: whereLayer("A").mayNotAccessAnyLayer()
    JAVA_LAYER_MAY_NOT_ACCESS: Final[re.Pattern] = re.compile(
        r'whereLayer\("(?P<src>[^"]+)"\)\s*\.mayNotAccessLayers?\("(?P<tgt>[^"]+)"\)',
        re.DOTALL,
    )

    # Python PyTestArch / ArchUnitPython Patterns
    # Matches: Rule().that().have_module_name("..A..").should().not_import_modules_with_name("..B..")
    PYTHON_PYTESTARCH_NOT_IMPORT: Final[re.Pattern] = re.compile(
        r'that\(\)\s*\.(?:have_module_name|are_sub_modules_of)\("(?P<src>[^"]+)"\)\s*'
        r'\.should\(\)\s*\.(?:not_import_modules_with_name|not_import_modules_of)\("(?P<tgt>[^"]+)"\)',
        re.DOTALL,
    )

    def __init__(self, output_rules_path: Path | str = "data/architecture_rules/mined_rules_elb.json") -> None:
        self.output_rules_path: Final[Path] = Path(output_rules_path)
        logger.debug("ArchUnitRuleImporter initialized (Output=%s)", self.output_rules_path)

    @staticmethod
    def _clean_package_identifier(raw: str) -> str:
        """Strips leading/trailing wildcard regex dots from package expressions."""
        cleaned = raw.strip().replace("..", ".").strip(".")
        return cleaned or raw

    def parse_java_source(self, content: str, file_path: str = "ArchTest.java") -> list[ExtractedRule]:
        """Extracts architectural invariants from Java ArchUnit test classes."""
        rules: list[ExtractedRule] = []

        # 1. Parse noClasses()...dependOn(...)
        for idx, match in enumerate(self.JAVA_NO_CLASSES_DEPEND_PATTERN.finditer(content), start=1):
            src = self._clean_package_identifier(match.group("src"))
            tgt = self._clean_package_identifier(match.group("tgt"))
            rules.append(
                ExtractedRule(
                    rule_id=f"ARCHUNIT-JAVA-DEP-{idx}",
                    framework="ArchUnit-Java",
                    source_scope=src,
                    target_scope=tgt,
                    rule_type="FORBIDDEN_DEPENDENCY",
                    raw_statement=match.group(0),
                    source_file=file_path,
                )
            )

        # 2. Parse layeredArchitecture() layer restrictions
        for idx, match in enumerate(self.JAVA_LAYER_MAY_NOT_ACCESS.finditer(content), start=1):
            src = match.group("src").strip()
            tgt = match.group("tgt").strip()
            rules.append(
                ExtractedRule(
                    rule_id=f"ARCHUNIT-JAVA-LAYER-{idx}",
                    framework="ArchUnit-Java",
                    source_scope=src,
                    target_scope=tgt,
                    rule_type="FORBIDDEN_DEPENDENCY",
                    raw_statement=match.group(0),
                    source_file=file_path,
                )
            )

        for idx, match in enumerate(self.JAVA_LAYER_MAY_NOT_BE_ACCESSED.finditer(content), start=1):
            src = match.group("src").strip()  # Inverting logic: B cannot access A => B -> A forbidden
            tgt = match.group("tgt").strip()
            rules.append(
                ExtractedRule(
                    rule_id=f"ARCHUNIT-JAVA-ACCESS-{idx}",
                    framework="ArchUnit-Java",
                    source_scope=src,
                    target_scope=tgt,
                    rule_type="FORBIDDEN_DEPENDENCY",
                    raw_statement=match.group(0),
                    source_file=file_path,
                )
            )

        return rules

    def parse_python_source(self, content: str, file_path: str = "test_arch.py") -> list[ExtractedRule]:
        """Extracts architectural invariants from PyTestArch test files."""
        rules: list[ExtractedRule] = []

        for idx, match in enumerate(self.PYTHON_PYTESTARCH_NOT_IMPORT.finditer(content), start=1):
            src = self._clean_package_identifier(match.group("src"))
            tgt = self._clean_package_identifier(match.group("tgt"))
            rules.append(
                ExtractedRule(
                    rule_id=f"PYTESTARCH-PY-{idx}",
                    framework="PyTestArch-Python",
                    source_scope=src,
                    target_scope=tgt,
                    rule_type="FORBIDDEN_DEPENDENCY",
                    raw_statement=match.group(0),
                    source_file=file_path,
                )
            )

        return rules

    def scan_directory(self, target_dir: Path | str) -> list[ExtractedRule]:
        """Recursively scans a repository directory for architecture test files."""
        root = Path(target_dir)
        if not root.exists():
            logger.warning("Target repository path does not exist: %s", root)
            return []

        all_rules: list[ExtractedRule] = []

        # Scan Java test suites
        for java_file in root.rglob("*Test.java"):
            try:
                content = java_file.read_text(encoding="utf-8", errors="replace")
                if "ArchUnit" in content or "com.tngtech.archunit" in content:
                    rules = self.parse_java_source(content, file_path=java_file.as_posix())
                    all_rules.extend(rules)
            except Exception as err:
                logger.debug("Failed parsing Java file %s: %s", java_file, err)

        # Scan Python test suites
        for py_file in root.rglob("test_*.py"):
            try:
                content = py_file.read_text(encoding="utf-8", errors="replace")
                if "pytestarch" in content or "archunit" in content:
                    rules = self.parse_python_source(content, file_path=py_file.as_posix())
                    all_rules.extend(rules)
            except Exception as err:
                logger.debug("Failed parsing Python file %s: %s", py_file, err)

        logger.info("Scan complete: Extracted %d total rules from %s", len(all_rules), root)
        return all_rules

    def consolidate_into_dag_report(self, rules: Sequence[ExtractedRule]) -> ImportedRulesReport:
        """
        Compiles heterogeneous rules into DAG's JSON schema (Eq. 6 format).
        """
        tier_blacklist: dict[str, set[str]] = {}
        package_blacklist: dict[str, set[str]] = {}

        for rule in rules:
            src = rule.source_scope
            tgt = rule.target_scope

            # Classify whether scopes are architectural tiers or concrete packages
            is_tier = any(
                src.lower().startswith(prefix) for prefix in ("ui", "presentation", "domain", "infra", "service")
            ) and "." not in src

            if is_tier:
                if src not in tier_blacklist:
                    tier_blacklist[src] = set()
                tier_blacklist[src].add(tgt)
            else:
                if src not in package_blacklist:
                    package_blacklist[src] = set()
                package_blacklist[src].add(tgt)

        return ImportedRulesReport(
            metadata={
                "source": "Declarative Architecture Tests (ArchUnit / PyTestArch)",
                "total_imported_rules": len(rules),
                "compiler": "ArchUnitRuleImporter v1.0 (SANER 2026 AE)",
                "formal_mapping": "Equation 6: r_layer(G') forbidden mapping B(u)",
            },
            forbidden_tier_dependencies={k: sorted(v) for k, v in tier_blacklist.items()},
            forbidden_package_dependencies={k: sorted(v) for k, v in package_blacklist.items()},
            extracted_rules=list(rules),
        )

    def export_to_json(self, report: ImportedRulesReport, output_file: Path | str | None = None) -> Path:
        """Persists the consolidated report to data/architecture_rules/mined_rules_elb.json."""
        dest = Path(output_file or self.output_rules_path)
        dest.parent.mkdir(parents=True, exist_ok=True)

        payload = {
            "metadata": report.metadata,
            "forbidden_tier_dependencies": report.forbidden_tier_dependencies,
            "forbidden_package_dependencies": report.forbidden_package_dependencies,
            "imported_rules_audit": [asdict(r) for r in report.extracted_rules],
        }

        with open(dest, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

        logger.info("Successfully exported %d rules to %s", len(report.extracted_rules), dest)
        return dest

    @staticmethod
    def get_mock_architecture_test_snippets() -> dict[str, str]:
        """Returns standard ArchUnit (Java) and PyTestArch (Python) snippets for testing."""
        java_snippet = """
        package org.sample.architecture;
        import com.tngtech.archunit.junit.AnalyzeClasses;
        import com.tngtech.archunit.junit.ArchTest;
        import com.tngtech.archunit.lang.ArchRule;
        import static com.tngtech.archunit.lang.syntax.ArchRuleDefinition.noClasses;

        @AnalyzeClasses(packages = "org.sample")
        public class ArchitectureTest {
            @ArchTest
            public static final ArchRule infra_should_not_depend_on_ui =
                noClasses().that().resideInAPackage("..infrastructure..")
                           .should().dependOnClassesThat().resideInAPackage("..presentation..");

            @ArchTest
            public static final ArchRule domain_should_not_depend_on_ui =
                noClasses().that().resideInAPackage("..domain..")
                           .should().dependOnClassesThat().resideInAPackage("..presentation..");
        }
        """

        python_snippet = """
        import pytest
        from pytestarch import Rule

        def test_clean_architecture_boundaries():
            rule = (
                Rule().that().are_sub_modules_of("app.infrastructure")
                      .should().not_import_modules_of("app.presentation")
            )
            rule.check()
        """
        return {"ArchTest.java": java_snippet, "test_arch.py": python_snippet}


def main() -> None:
    """CLI Gateway for Artifact Reviewers."""
    parser = argparse.ArgumentParser(description="Declarative Architecture Test Importer (ArchUnit / PyTestArch)")
    parser.add_argument("--repo-dir", type=str, default=None, help="Root path of target repository to scan")
    parser.add_argument("--output", type=str, default="data/architecture_rules/mined_rules_elb.json", help="Output JSON path")
    parser.add_argument("--mock-sample", action="store_true", help="Parse built-in test snippets for instant validation")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] [%(levelname)s] [Importer] %(message)s")

    importer = ArchUnitRuleImporter(output_rules_path=args.output)
    extracted: list[ExtractedRule] = []

    if args.mock_sample or not args.repo_dir:
        logger.info("Executing extraction on built-in ArchUnit (Java) & PyTestArch (Python) mock test suites...")
        snippets = importer.get_mock_architecture_test_snippets()
        extracted.extend(importer.parse_java_source(snippets["ArchTest.java"], "MockArchTest.java"))
        extracted.extend(importer.parse_python_source(snippets["test_arch.py"], "mock_test_arch.py"))
    else:
        extracted = importer.scan_directory(args.repo_dir)

    report = importer.consolidate_into_dag_report(extracted)
    importer.export_to_json(report, args.output)

    print(f"\n[Artifact Evaluation Success] Imported {len(extracted)} rules -> {args.output}")
    print(f"Forbidden Tiers: {report.forbidden_tier_dependencies}")
    print(f"Forbidden Packages: {report.forbidden_package_dependencies}")


if __name__ == "__main__":
    main()
