"""
================================================================================
Evaluation Engine: Rigorous Statistical Significance & Effect Size Testing
================================================================================

Paper Reference:
  Section V-A: Answering RQ1 (Statistical Tests on AgTD & ResRate)
  Table I: Empirical Significance Testing (Wilcoxon, Mann-Whitney U, Cliff's d)
  "Governed Autonomy: Halting Architectural Entropy in Agentic Software 
   Workflows via Dynamic Guardrails" (SANER 2026)

Formal Statistical Methodology:
  - Non-parametric Paired Testing: Wilcoxon Signed-Rank Test (within-subjects)
  - Non-parametric Unpaired Testing: Mann-Whitney U Test
  - Non-parametric Effect Size: Cliff's Delta (d) with 95% Bootstrap CI
  - Multiple Hypothesis Testing Correction: Holm-Bonferroni & Benjamini-Hochberg (FDR)

Standards:
  Compliant with ACM/IEEE Artifact Evaluation Guidelines (Badge: Reusable)
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Final, Sequence

logger = logging.getLogger("dag.evaluation.statistical_tests")


def _normal_sf_two_sided(z_abs: float) -> float:
    """Two-sided p-value from standard normal Z statistic using math.erfc."""
    return max(1e-16, min(1.0, math.erfc(abs(z_abs) / math.sqrt(2.0))))


def _percentile(sorted_vals: Sequence[float], pct: float) -> float:
    """Computes linear-interpolated percentile in [0, 100] from sorted sequence."""
    if not sorted_vals:
        return 0.0
    n = len(sorted_vals)
    if n == 1:
        return float(sorted_vals[0])
    pos = (pct / 100.0) * (n - 1)
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return float(sorted_vals[lo])
    weight = pos - lo
    return float(sorted_vals[lo] * (1.0 - weight) + sorted_vals[hi] * weight)


@dataclass(frozen=True, slots=True)
class StatisticalTestReport:
    """Rigorous empirical test report matching ACM/IEEE SE reporting guidelines."""
    comparison: str
    sample_size_a: int
    sample_size_b: int
    test_type: str
    statistic: float
    p_value: float
    p_adjusted_holm: float
    p_adjusted_fdr: float
    is_significant_alpha_005: bool
    is_significant_alpha_001: bool
    cliffs_delta: float
    cliffs_ci_lower: float
    cliffs_ci_upper: float
    cliffs_magnitude: str  # "negligible" | "small" | "medium" | "large"

    def format_row(self) -> str:
        sig_marker = (
            "***"
            if self.p_adjusted_holm < 0.001
            else ("**" if self.p_adjusted_holm < 0.01 else ("*" if self.p_adjusted_holm < 0.05 else "n.s."))
        )
        return (
            f"{self.comparison:<28} | {self.test_type:<10} | {self.statistic:>9.2f} | "
            f"{self.p_adjusted_holm:>9.2e} {sig_marker:<3} | {self.cliffs_delta:>+6.3f} "
            f"[{self.cliffs_ci_lower:>+5.2f}, {self.cliffs_ci_upper:>+5.2f}] | "
            f"{self.cliffs_magnitude:<10}"
        )


class StatisticalTestEngine:
    """
    Non-parametric statistical verification suite for empirical software engineering experiments.
    Uses scipy.stats when available and provides exact/asymptotic pure-Python implementations
    so the evaluation suite runs out-of-the-box in any Python >=3.10 environment.
    """

    @staticmethod
    def compute_cliffs_delta(
        group_a: Sequence[float],
        group_b: Sequence[float],
        num_bootstrap: int = 1000,
        alpha: float = 0.05,
        random_seed: int = 42,
    ) -> tuple[float, float, float, str]:
        """
        Computes Cliff's delta non-parametric effect size with 95% Bootstrap confidence interval.
        
        Formula:
            d = ( #{a > b} - #{a < b} ) / ( |A| * |B| )
        """
        list_a = [float(x) for x in group_a]
        list_b = [float(x) for x in group_b]
        n_a, n_b = len(list_a), len(list_b)

        if n_a == 0 or n_b == 0:
            return 0.0, 0.0, 0.0, "negligible"

        more = 0
        less = 0
        for a in list_a:
            for b in list_b:
                if a > b:
                    more += 1
                elif a < b:
                    less += 1
        point_delta = float(more - less) / float(n_a * n_b)

        rng = random.Random(random_seed)
        boot_deltas: list[float] = []
        for _ in range(num_bootstrap):
            sa = [list_a[rng.randrange(n_a)] for _ in range(n_a)]
            sb = [list_b[rng.randrange(n_b)] for _ in range(n_b)]
            b_more = 0
            b_less = 0
            for a in sa:
                for b in sb:
                    if a > b:
                        b_more += 1
                    elif a < b:
                        b_less += 1
            boot_deltas.append(float(b_more - b_less) / float(n_a * n_b))

        boot_deltas.sort()
        ci_lower = _percentile(boot_deltas, 100.0 * (alpha / 2.0))
        ci_upper = _percentile(boot_deltas, 100.0 * (1.0 - alpha / 2.0))

        abs_d = abs(point_delta)
        if abs_d < 0.147:
            magnitude = "negligible"
        elif abs_d < 0.330:
            magnitude = "small"
        elif abs_d < 0.474:
            magnitude = "medium"
        else:
            magnitude = "large"

        return point_delta, ci_lower, ci_upper, magnitude

    @classmethod
    def compare_paired_samples(
        cls,
        name: str,
        group_base: Sequence[float],
        group_dag: Sequence[float],
    ) -> StatisticalTestReport:
        """
        Paired comparison on identical tasks using Wilcoxon Signed-Rank Test.
        """
        list_base = [float(x) for x in group_base]
        list_dag = [float(x) for x in group_dag]
        n = len(list_base)

        if n != len(list_dag):
            raise ValueError(f"Paired test requires equal sample lengths ({n} vs {len(list_dag)})")

        diffs = [b - d for b, d in zip(list_base, list_dag) if abs(b - d) > 1e-12]
        m = len(diffs)

        if m == 0:
            w_stat, p_val = 0.0, 1.0
        else:
            try:
                from scipy import stats as sp_stats  # type: ignore
                w_res = sp_stats.wilcoxon(list_base, list_dag, alternative="two-sided")
                w_stat, p_val = float(w_res.statistic), float(w_res.pvalue)
            except Exception:
                # Pure-Python Wilcoxon signed-rank test with tie-average ranking
                abs_diffs = sorted(enumerate(abs(x) for x in diffs), key=lambda t: t[1])
                ranks = [0.0] * m
                i = 0
                while i < m:
                    j = i
                    while j + 1 < m and abs(abs_diffs[j + 1][1] - abs_diffs[i][1]) <= 1e-12:
                        j += 1
                    avg_rank = 0.5 * (i + 1 + j + 1)
                    for k in range(i, j + 1):
                        ranks[abs_diffs[k][0]] = avg_rank
                    i = j + 1

                w_plus = sum(r for r, d in zip(ranks, diffs) if d > 0)
                w_minus = sum(r for r, d in zip(ranks, diffs) if d < 0)
                w_stat = min(w_plus, w_minus)
                mean_w = m * (m + 1) / 4.0
                var_w = m * (m + 1) * (2 * m + 1) / 24.0
                z = (abs(w_stat - mean_w) - 0.5) / math.sqrt(max(1e-9, var_w))
                p_val = _normal_sf_two_sided(z)

        delta, ci_low, ci_high, mag = cls.compute_cliffs_delta(list_base, list_dag)

        return StatisticalTestReport(
            comparison=name,
            sample_size_a=n,
            sample_size_b=n,
            test_type="Wilcoxon",
            statistic=w_stat,
            p_value=p_val,
            p_adjusted_holm=p_val,
            p_adjusted_fdr=p_val,
            is_significant_alpha_005=(p_val < 0.05),
            is_significant_alpha_001=(p_val < 0.001),
            cliffs_delta=delta,
            cliffs_ci_lower=ci_low,
            cliffs_ci_upper=ci_high,
            cliffs_magnitude=mag,
        )

    @classmethod
    def compare_independent_samples(
        cls,
        name: str,
        group_a: Sequence[float],
        group_b: Sequence[float],
    ) -> StatisticalTestReport:
        """
        Independent comparison using Mann-Whitney U Test with continuity correction.
        """
        list_a = [float(x) for x in group_a]
        list_b = [float(x) for x in group_b]
        n_a, n_b = len(list_a), len(list_b)

        try:
            from scipy import stats as sp_stats  # type: ignore
            u_res = sp_stats.mannwhitneyu(list_a, list_b, alternative="two-sided", method="auto")
            u_stat, p_val = float(u_res.statistic), float(u_res.pvalue)
        except Exception:
            u_stat = 0.0
            for a in list_a:
                for b in list_b:
                    if a > b:
                        u_stat += 1.0
                    elif abs(a - b) <= 1e-12:
                        u_stat += 0.5
            mean_u = 0.5 * n_a * n_b
            var_u = n_a * n_b * (n_a + n_b + 1) / 12.0
            z = (abs(u_stat - mean_u) - 0.5) / math.sqrt(max(1e-9, var_u))
            p_val = _normal_sf_two_sided(z)

        delta, ci_low, ci_high, mag = cls.compute_cliffs_delta(list_a, list_b)

        return StatisticalTestReport(
            comparison=name,
            sample_size_a=n_a,
            sample_size_b=n_b,
            test_type="Mann-Whitney",
            statistic=u_stat,
            p_value=p_val,
            p_adjusted_holm=p_val,
            p_adjusted_fdr=p_val,
            is_significant_alpha_005=(p_val < 0.05),
            is_significant_alpha_001=(p_val < 0.001),
            cliffs_delta=delta,
            cliffs_ci_lower=ci_low,
            cliffs_ci_upper=ci_high,
            cliffs_magnitude=mag,
        )

    @staticmethod
    def adjust_p_values_holm(p_values: Sequence[float]) -> list[float]:
        """Holm-Bonferroni step-down family-wise error rate (FWER) correction."""
        m = len(p_values)
        indexed = sorted(enumerate(p_values), key=lambda x: x[1])
        adjusted = [0.0] * m
        running_max = 0.0

        for rank, (orig_idx, p) in enumerate(indexed):
            factor = m - rank
            p_adj = min(1.0, p * factor)
            running_max = max(running_max, p_adj)
            adjusted[orig_idx] = running_max

        return adjusted

    @staticmethod
    def adjust_p_values_fdr(p_values: Sequence[float]) -> list[float]:
        """Benjamini-Hochberg false discovery rate (FDR) adjustment."""
        m = len(p_values)
        indexed = sorted(enumerate(p_values), key=lambda x: x[1])
        adjusted = [0.0] * m
        running_min = 1.0

        for rank in range(m - 1, -1, -1):
            orig_idx, p = indexed[rank]
            p_adj = min(1.0, (p * m) / float(rank + 1))
            running_min = min(running_min, p_adj)
            adjusted[orig_idx] = running_min

        return adjusted


def analyze_log_file(log_path: Path | str) -> list[StatisticalTestReport]:
    """
    Parses experiment_runs_rq1.jsonl and conducts full non-parametric statistical evaluation.
    """
    path_obj = Path(log_path)
    if not path_obj.exists():
        raise FileNotFoundError(f"Log file not found: {log_path}")

    records_by_wf: dict[str, dict[str, float]] = {}
    with open(path_obj, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            entry = json.loads(line)
            wf = entry.get("workflow", entry.get("framework"))
            tid = entry.get("task_id")
            agtd = float(entry.get("agtd_delta", entry.get("delta_td", 0.0)))
            if wf and tid:
                if wf not in records_by_wf:
                    records_by_wf[wf] = {}
                records_by_wf[wf][tid] = agtd

    wf_names = list(records_by_wf.keys())
    logger.info("Loaded workflows: %s", wf_names)

    reports: list[StatisticalTestReport] = []

    base_key = next((k for k in wf_names if "base" in k.lower()), None)
    dag_key = next((k for k in wf_names if "dag" in k.lower()), None)

    if base_key:
        for wf in wf_names:
            if wf == base_key:
                continue
            common_tasks = sorted(set(records_by_wf[base_key].keys()) & set(records_by_wf[wf].keys()))
            if common_tasks:
                base_vals = [records_by_wf[base_key][t] for t in common_tasks]
                wf_vals = [records_by_wf[wf][t] for t in common_tasks]
                rep = StatisticalTestEngine.compare_paired_samples(
                    f"{wf} vs {base_key}",
                    base_vals,
                    wf_vals,
                )
                reports.append(rep)

    if dag_key:
        naive_key = next((k for k in wf_names if "naive" in k.lower()), None)
        if naive_key and naive_key in records_by_wf:
            common_tasks = sorted(set(records_by_wf[naive_key].keys()) & set(records_by_wf[dag_key].keys()))
            if common_tasks:
                naive_vals = [records_by_wf[naive_key][t] for t in common_tasks]
                dag_vals = [records_by_wf[dag_key][t] for t in common_tasks]
                rep = StatisticalTestEngine.compare_paired_samples(
                    f"{dag_key} vs {naive_key}",
                    naive_vals,
                    dag_vals,
                )
                reports.append(rep)

    if reports:
        raw_p = [r.p_value for r in reports]
        holm_p = StatisticalTestEngine.adjust_p_values_holm(raw_p)
        fdr_p = StatisticalTestEngine.adjust_p_values_fdr(raw_p)
        adjusted_reports = []
        for i, r in enumerate(reports):
            adjusted_reports.append(
                StatisticalTestReport(
                    comparison=r.comparison,
                    sample_size_a=r.sample_size_a,
                    sample_size_b=r.sample_size_b,
                    test_type=r.test_type,
                    statistic=r.statistic,
                    p_value=r.p_value,
                    p_adjusted_holm=holm_p[i],
                    p_adjusted_fdr=fdr_p[i],
                    is_significant_alpha_005=(holm_p[i] < 0.05),
                    is_significant_alpha_001=(holm_p[i] < 0.001),
                    cliffs_delta=r.cliffs_delta,
                    cliffs_ci_lower=r.cliffs_ci_lower,
                    cliffs_ci_upper=r.cliffs_ci_upper,
                    cliffs_magnitude=r.cliffs_magnitude,
                )
            )
        reports = adjusted_reports

    return reports


def print_reports(reports: Sequence[StatisticalTestReport]) -> None:
    """Pretty prints empirical results in IEEE/ACM conference format."""
    print("\n" + "=" * 98)
    print(" STATISTICAL SIGNIFICANCE & EFFECT SIZE REPORT (NON-PARAMETRIC EVALUATION)")
    print("=" * 98)
    header = (
        f"{'Comparison':<28} | {'Test':<10} | {'Statistic':<9} | "
        f"{'p_Holm':<14} | {'Cliff\'s d [95% CI]':<23} | {'Magnitude':<10}"
    )
    print(header)
    print("-" * 98)
    for r in reports:
        print(r.format_row())
    print("-" * 98)
    print(" Significance codes:  *** p < 0.001   ** p < 0.01   * p < 0.05   n.s. Not Significant")
    print(" Multi-testing control: Holm-Bonferroni Family-Wise Error Rate (FWER) adjusted.")
    print("=" * 98 + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Statistical Significance Engine for Artifact Evaluation")
    parser.add_argument(
        "--input",
        type=str,
        default="logs_and_artifacts/experiment_runs_rq1.jsonl",
        help="Path to JSONL execution log",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    reports = analyze_log_file(args.input)
    print_reports(reports)


if __name__ == "__main__":
    main()
