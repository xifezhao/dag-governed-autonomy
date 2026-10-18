"""
================================================================================
Evaluation Engine: Figure 3(b) Oscillation & Livelock CDF Visualizer
================================================================================

Paper Reference:
  Figure 3(b): Empirical CDF of Convergence Iterations under Multi-Agent Retries
  Section V-C: Answering RQ3 (Oscillatory Livelock & Bounded Recovery)
  "Governed Autonomy: Halting Architectural Entropy in Agentic Software
   Workflows via Dynamic Guardrails" (SANER 2026)
"""

from __future__ import annotations

import argparse
import csv
import logging
from pathlib import Path

logger = logging.getLogger("dag.evaluation.plot_fig4")


def plot_figure_4(
    csv_path: Path | str = "logs_and_artifacts/cdf_iterations_data.csv",
    output_dir: Path | str = "figures/",
) -> None:
    path_obj = Path(csv_path)
    if not path_obj.exists():
        raise FileNotFoundError(f"CDF CSV data file not found: {csv_path}")

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    iterations: list[int] = []
    cdf_naive: list[float] = []
    cdf_dag: list[float] = []
    with open(path_obj, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            iterations.append(int(r["iteration"]))
            cdf_naive.append(float(r["cdf_naive_linter"]) * 100.0)
            cdf_dag.append(float(r["cdf_dag_proposed"]) * 100.0)

    tikz_path = Path("evaluation/tikz_generated/fig3b_cdf_iterations.tex")
    tikz_path.parent.mkdir(parents=True, exist_ok=True)
    naive_coords = " ".join(f"({k},{v:.0f})" for k, v in zip(iterations, cdf_naive))
    dag_coords = " ".join(f"({k},{v:.0f})" for k, v in zip(iterations, cdf_dag))
    tikz_content = (
        f"% Auto-generated Figure 3(b) PGFPlots CDF coordinates\n"
        f"% Naive-Linter: {naive_coords}\n"
        f"% DAG-Proposed: {dag_coords}\n"
    )
    tikz_path.write_text(tikz_content, encoding="utf-8")

    try:
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(6.5, 4.2), dpi=300)
        ax.step(iterations, cdf_dag, where="mid", label="DAG (Proposed: State Fingerprinting)", color="#2b8a3e", linewidth=2.5, marker="o", markersize=6)
        ax.step(iterations, cdf_naive, where="mid", label="Naive-Linter (Stateless Retries)", color="#d9534f", linewidth=2.5, marker="s", markersize=6, linestyle="--")
        ax.set_xlabel("Retry Iteration Budget ($K$)")
        ax.set_ylabel("Cumulative Convergence Probability (%)")
        ax.set_title("(b) Empirical CDF: Mitigation of Whack-a-Mole Livelocks")
        ax.set_xlim(0.5, 10.5)
        ax.set_ylim(0, 105)
        ax.set_xticks(range(1, 11))
        ax.grid(True, linestyle=":", alpha=0.6)
        ax.legend(loc="lower right", frameon=True)
        fig.savefig(out_dir / "figure4_cdf_iterations.pdf", format="pdf", bbox_inches="tight")
        fig.savefig(out_dir / "figure4_cdf_iterations.png", format="png", bbox_inches="tight")
        plt.close(fig)
        print(f"[Artifact Visualizer] Figure 3(b) exported to {out_dir / 'figure4_cdf_iterations.pdf'} and {tikz_path}")
    except ImportError:
        print(f"[Artifact Visualizer] Exported Figure 3(b) PGFPlots CDF coordinates to {tikz_path}")
        print(f"  Naive-Linter: {naive_coords}")
        print(f"  DAG-Proposed: {dag_coords}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Render Figure 3(b) CDF Iterations Visualizations")
    parser.add_argument("--csv", type=str, default="logs_and_artifacts/cdf_iterations_data.csv")
    parser.add_argument("--output-dir", type=str, default="figures/")
    args = parser.parse_args()
    plot_figure_4(args.csv, args.output_dir)


if __name__ == "__main__":
    main()
