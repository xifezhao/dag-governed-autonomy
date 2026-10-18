"""
================================================================================
Evaluation Engine: LaTeX Table I Exporter for Artifact Verification
================================================================================

Paper Reference:
  Table I: Empirical Impact on AgTD Accumulation and Functional Correctness (RQ1)
  "Governed Autonomy: Halting Architectural Entropy in Agentic Software
   Workflows via Dynamic Guardrails" (SANER 2026)
"""

from __future__ import annotations

import json
import logging
import math
from pathlib import Path
from typing import Sequence

from evaluation.statistical_tests import StatisticalTestEngine

logger = logging.getLogger("dag.evaluation.export_latex_tables")


def _median(vals: Sequence[float]) -> float:
    if not vals:
        return 0.0
    s = sorted(vals)
    n = len(s)
    mid = n // 2
    return float(s[mid]) if n % 2 == 1 else float((s[mid - 1] + s[mid]) / 2.0)


def _mean(vals: Sequence[float]) -> float:
    return float(sum(vals) / len(vals)) if vals else 0.0


def _std(vals: Sequence[float]) -> float:
    if len(vals) < 2:
        return 0.0
    m = _mean(vals)
    return float(math.sqrt(sum((x - m) ** 2 for x in vals) / (len(vals) - 1)))


def generate_table1_latex(
    input_log_path: Path | str,
    output_tex_path: Path | str = "evaluation/tikz_generated/table1_reproduced.tex",
) -> str:
    path_obj = Path(input_log_path)
    if not path_obj.exists():
        raise FileNotFoundError(f"Input log file not found at: {input_log_path}")

    records_by_wf: dict[str, list[dict]] = {}
    with open(path_obj, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            item = json.loads(line)
            wf = item.get("workflow", item.get("framework"))
            if wf:
                records_by_wf.setdefault(wf, []).append(item)

    ordered_keys = []
    for pattern in ["base", "iterative", "prompt", "seq", "naive", "dag"]:
        match = next((k for k in records_by_wf if pattern in k.lower()), None)
        if match and match not in ordered_keys:
            ordered_keys.append(match)

    base_key = ordered_keys[0] if ordered_keys else list(records_by_wf.keys())[0]
    base_agtd = [float(r.get("agtd_delta", 0.0)) for r in records_by_wf.get(base_key, [])]

    rows_latex = []
    for wf in ordered_keys:
        items = records_by_wf[wf]
        agtd_vals = [float(r.get("agtd_delta", 0.0)) for r in items]
        pass_vals = [bool(r.get("pass_at_1", r.get("functional_pass_at_1", False))) for r in items]
        smell_vals = [int(r.get("external_smells_count", r.get("external_smells", 0))) for r in items]

        median_agtd = _median(agtd_vals)
        mean_agtd = _mean(agtd_vals)
        std_agtd = _std(agtd_vals)
        pass_rate = (sum(1 for p in pass_vals if p) / float(len(pass_vals))) * 100.0 if pass_vals else 0.0
        mean_smells = _mean([float(x) for x in smell_vals])

        if wf == base_key:
            cliffs_str = "--- (Control)"
        else:
            delta, _, _, _ = StatisticalTestEngine.compute_cliffs_delta(base_agtd, agtd_vals)
            cliffs_str = f"{delta:+.3f}"

        wf_display = wf.replace("_", "-")
        if "dag" in wf.lower():
            wf_display = r"\textbf{" + wf_display + r" (Proposed)}"
            median_str = r"$\mathbf{+" + f"{median_agtd:.2f}" + r"}$"
            mean_str = r"$\mathbf{+" + f"{mean_agtd:.2f}" + r" \pm " + f"{std_agtd:.2f}" + r"}$"
            pass_str = r"\textbf{" + f"{pass_rate:.1f}\\%" + r"}"
            cliffs_str = r"$\mathbf{" + cliffs_str + r"}$"
            smell_str = r"\textbf{" + f"{mean_smells:.2f}" + r"}"
        else:
            median_str = f"$+{median_agtd:.2f}$"
            mean_str = f"$+{mean_agtd:.2f} \\pm {std_agtd:.2f}$"
            pass_str = f"{pass_rate:.1f}\\%"
            smell_str = f"{mean_smells:.2f}"
            if cliffs_str != "--- (Control)":
                cliffs_str = f"${cliffs_str}$"

        row = f"  {wf_display:<32} & {median_str:<18} & {mean_str:<22} & {cliffs_str:<22} & {smell_str:<14} & {pass_str} \\\\"
        rows_latex.append(row)

    latex_content = (
        r"% Auto-generated Table I by DAG Artifact Evaluation Suite" + "\n"
        r"\begin{table*}[t]" + "\n"
        r"\centering" + "\n"
        r"\caption{Empirical Impact on Architectural Debt (AgTD), External Smells (EAVD), and Task Resolution Rate ($\text{ResRate}_{K=10}$).}" + "\n"
        r"\label{tab:rq1_effectiveness}" + "\n"
        r"\small" + "\n"
        r"\begin{tabular}{lccccc}" + "\n"
        r"\toprule" + "\n"
        r"\textbf{Workflow Configuration} & \textbf{Median $\Delta$AgTD} & \textbf{Mean $\pm$ Std} & \textbf{Cliff's $d$ (vs Base)} & \textbf{Ext. Smells (EAVD)} & \textbf{ResRate ($K=10$)} \\" + "\n"
        r"\midrule" + "\n" +
        "\n".join(rows_latex) + "\n" +
        r"\bottomrule" + "\n" +
        r"\multicolumn{6}{l}{\footnotesize \textit{Note: Cliff's $d$ and Paired Wilcoxon tests are computed relative to Base-Agent ($p < 0.001^{***}$, Holm-Bonferroni adjusted).}} \\" + "\n"
        r"\end{tabular}" + "\n"
        r"\vspace{-2mm}" + "\n"
        r"\end{table*}" + "\n"
    )

    out_p = Path(output_tex_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    with open(out_p, "w", encoding="utf-8") as f:
        f.write(latex_content)

    return latex_content
