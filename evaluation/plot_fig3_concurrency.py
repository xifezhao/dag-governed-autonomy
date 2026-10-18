"""
================================================================================
Evaluation Engine: Figure 3(a) & Table II Concurrency & Attribution Visualizer
================================================================================

Paper Reference:
  Figure 3(a): Mean Wasted Token Overhead Across Concurrency Levels n in {3,4,5,6}
  Table II: Attribution Fidelity Across 160 Concurrent Batches
  Section V-B: Answering RQ2 (Concurrency & Resource Overhead)
  "Governed Autonomy: Halting Architectural Entropy in Agentic Software
   Workflows via Dynamic Guardrails" (SANER 2026)
"""

from __future__ import annotations

import argparse
import csv
import logging
from pathlib import Path

logger = logging.getLogger("dag.evaluation.plot_fig3")


def plot_figure_3(
    csv_path: Path | str = "logs_and_artifacts/token_usage_records.csv",
    output_dir: Path | str = "figures/",
) -> None:
    path_obj = Path(csv_path)
    if not path_obj.exists():
        raise FileNotFoundError(f"CSV data file not found: {csv_path}")

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, str]] = []
    with open(path_obj, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(r)

    by_n_naive: dict[int, list[float]] = {}
    by_n_dag: dict[int, list[float]] = {}
    for r in rows:
        n = int(r.get("concurrency_n", r.get("concurrency_level", 3)))
        if "wasted_tokens_naive" in r:
            by_n_naive.setdefault(n, []).append(float(r["wasted_tokens_naive"]))
            by_n_dag.setdefault(n, []).append(float(r["wasted_tokens_dag"]))
        else:
            wf = r.get("workflow", "")
            if wf == "Naive-Linter":
                by_n_naive.setdefault(n, []).append(float(r["wasted_tokens"]))
            elif wf == "DAG-Proposed":
                by_n_dag.setdefault(n, []).append(float(r["wasted_tokens"]))

    concurrency_levels = sorted(by_n_naive.keys())
    wasted_naive = [
        sum(by_n_naive[n]) / max(1, len(by_n_naive[n]))
        for n in concurrency_levels
    ]
    wasted_dag = [
        sum(by_n_dag[n]) / max(1, len(by_n_dag[n]))
        for n in concurrency_levels
    ]

    tikz_path = Path("evaluation/tikz_generated/fig3a_concurrency.tex")
    tikz_path.parent.mkdir(parents=True, exist_ok=True)
    naive_coords = " ".join(f"({n},{int(round(w))})" for n, w in zip(concurrency_levels, wasted_naive))
    dag_coords = " ".join(f"({n},{int(round(w))})" for n, w in zip(concurrency_levels, wasted_dag))
    tikz_content = (
        f"% Auto-generated Figure 3(a) PGFPlots coordinates\n"
        f"% Naive-Linter: {naive_coords}\n"
        f"% DAG-Proposed: {dag_coords}\n"
    )
    tikz_path.write_text(tikz_content, encoding="utf-8")

    try:
        import matplotlib.pyplot as plt
        import numpy as np

        fig, ax1 = plt.subplots(figsize=(6.0, 4.0), dpi=300)
        x = np.arange(len(concurrency_levels))
        width = 0.35
        ax1.bar(x - width / 2, wasted_naive, width, label="Naive-Linter", color="#C42A2A", alpha=0.85, edgecolor="black")
        ax1.bar(x + width / 2, wasted_dag, width, label="DAG (Proposed)", color="#1870A8", alpha=0.85, edgecolor="black")
        ax1.set_xlabel("Concurrency Level ($n$ Agents)")
        ax1.set_ylabel("Mean Wasted Tokens / Batch")
        ax1.set_title("(a) Computational Waste Across Concurrency")
        ax1.set_xticks(x)
        ax1.set_xticklabels([f"$n={n}$" for n in concurrency_levels])
        ax1.legend(loc="upper left")
        ax1.grid(axis="y", linestyle="--", alpha=0.5)
        plt.savefig(out_dir / "figure3_concurrency.png", bbox_inches="tight")
        plt.savefig(out_dir / "figure3_concurrency.pdf", bbox_inches="tight")
        plt.close(fig)
        print(f"[Artifact Visualizer] Figure 3(a) exported to {out_dir / 'figure3_concurrency.pdf'} and {tikz_path}")
    except ImportError:
        print(f"[Artifact Visualizer] Exported Figure 3(a) PGFPlots coordinates to {tikz_path}")
        print(f"  Naive-Linter: {naive_coords}")
        print(f"  DAG-Proposed: {dag_coords}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Render Figure 3 Concurrency Visualizations")
    parser.add_argument("--csv", type=str, default="logs_and_artifacts/token_usage_records.csv")
    parser.add_argument("--output-dir", type=str, default="figures/")
    args = parser.parse_args()
    plot_figure_3(args.csv, args.output_dir)


if __name__ == "__main__":
    main()
