#!/usr/bin/env bash
# ==============================================================================
# One-Click Full Reproduction Script — DAG (SANER 2026 Artifact Evaluation)
# Paper: "Governed Autonomy: Halting Architectural Entropy in Agentic Software
#         Workflows via Dynamic Guardrails"
# ==============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

if [ -z "${PYTHON_BIN:-}" ]; then
    if python3 -c "import sys" >/dev/null 2>&1; then
        PYTHON_BIN="python3"
    elif [ -x "/opt/homebrew/bin/python3" ]; then
        PYTHON_BIN="/opt/homebrew/bin/python3"
    elif [ -x "/usr/bin/python3" ]; then
        PYTHON_BIN="/usr/bin/python3"
    else
        PYTHON_BIN="python3"
    fi
fi

echo "========================================================================"
echo " [Step 1/6] Running Unit Test Suite (Equations 1-10 & Theorem 1)"
echo "========================================================================"
"${PYTHON_BIN}" -m unittest discover -s tests -v

echo ""
echo "========================================================================"
echo " [Step 2/6] Running RQ1 Effectiveness Benchmark (Table I & Held-out r_debt)"
echo "========================================================================"
"${PYTHON_BIN}" -m experiments.run_rq1_effectiveness --tasks 50

echo ""
echo "========================================================================"
echo " [Step 3/6] Running RQ2 Concurrency & Attribution Benchmark (Table II & Fig. 3a)"
echo "========================================================================"
"${PYTHON_BIN}" -m experiments.run_rq2_concurrency --batches 40

echo ""
echo "========================================================================"
echo " [Step 4/6] Running RQ3 Oscillation, SPI & Theorem 1 Verification (Fig. 3b)"
echo "========================================================================"
"${PYTHON_BIN}" -m experiments.run_rq3_oscillation --tasks 50

echo ""
echo "========================================================================"
echo " [Step 5/6] Running Component Ablation Study (Table III, 600 Executions)"
echo "========================================================================"
"${PYTHON_BIN}" -m experiments.run_ablation --tasks 100

echo ""
echo "========================================================================"
echo " [Step 6/6] Exporting LaTeX Tables & PGFPlots/TikZ Coordinates"
echo "========================================================================"
"${PYTHON_BIN}" -m evaluation.plot_fig3_concurrency
"${PYTHON_BIN}" -m evaluation.plot_fig4_cdf_iterations

echo ""
echo "========================================================================"
echo " [SUCCESS] All SANER 2026 Tables (I-III) and Figures (3a, 3b) Reproduced!"
echo "========================================================================"
