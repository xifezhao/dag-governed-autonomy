# Governed Autonomy: Halting Architectural Entropy in Agentic Software Workflows via Dynamic Guardrails

[![Artifact Badges: Functional & Reusable](https://img.shields.io/badge/Artifact_Evaluation-Functional_%26_Reusable-brightgreen.svg)](#artifact-claims--badges-applied)
[![Track: Agentic AI4SE](https://img.shields.io/badge/Track-Agentic_AI4SE-blue.svg)](https://conf.researchr.org/track/saner-2026/saner-2026-papers)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Python: >=3.10](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)

Official open-source replication package and implementation of the **Dynamic Architectural Guardrail (DAG)** framework for the paper:  
**"Governed Autonomy: Halting Architectural Entropy in Agentic Software Workflows via Dynamic Guardrails"** (*IEEE SANER 2026*).

---

## Table of Contents

- [Artifact Claims & Badges Applied](#artifact-claims--badges-applied)
- [Formal Equation-to-Code Mapping (Section III)](#formal-equation-to-code-mapping-section-iii)
- [Empirical Claims & Reproduction Mapping (Section V)](#empirical-claims--reproduction-mapping-section-v)
- [System Requirements & Zero-Dependency Portability](#system-requirements--zero-dependency-portability)
- [Quick Start & One-Click Reproduction](#quick-start--one-click-reproduction)
- [Step-by-Step Experiment Reproduction](#step-by-step-experiment-reproduction)
  - [1. Unit Test Suite (Equations 1–10 & Theorem 1)](#1-unit-test-suite-equations-110--theorem-1)
  - [2. RQ1: Architectural Debt Mitigation & Held-Out Calibration (Table I)](#2-rq1-architectural-debt-mitigation--held-out-calibration-table-i)
  - [3. RQ2: Concurrency Scaling & Attribution Fidelity (Table II & Fig. 3a)](#3-rq2-concurrency-scaling--attribution-fidelity-table-ii--fig-3a)
  - [4. RQ3: Oscillation Control, SPI & Theorem 1 Bound (Fig. 3b)](#4-rq3-oscillation-control-spi--theorem-1-bound-fig-3b)
  - [5. Component Ablation Study (Table III)](#5-component-ablation-study-table-iii)
- [Benchmark Corpus (`data/`)](#benchmark-corpus-data)
- [Repository Architecture](#repository-architecture)
- [License](#license)

---

## Artifact Claims & Badges Applied

This repository is engineered to satisfy all three IEEE/ACM Artifact Evaluation badges:

| Badge | Criteria Met by this Replication Package |
| :--- | :--- |
| **Artifacts Available** | Complete source code, benchmark datasets ($\mathcal{H}_{\text{clean}}$ and $\mathcal{D}_{\text{eval}}$), mined architectural rules, and deterministic trace logs packaged under Apache-2.0. |
| **Artifacts Evaluated – Functional** | Reproduces every table (**Table I, Table II, Table III**), figure (**Figure 3a, Figure 3b**), and empirical validation claim in Section V via a single command (`bash run_all_experiments.sh`) in under 30 seconds on any standard CPU. |
| **Artifacts Evaluated – Reusable** | Modular, strictly typed Python 3.10+ implementation with zero-dependency standard-library fallbacks alongside full Conda/Docker configurations, `pyproject.toml`, and extensible `BaseAgentAdapter` interfaces. |

---

## Formal Equation-to-Code Mapping (Section III)

Every mathematical definition, rule, and theorem in **Section III** of the paper maps 1-to-1 to a dedicated module in `src/`:

| Paper Reference | Formalism / Equation | Source Module & Class |
| :--- | :--- | :--- |
| **Section III-A (Eq. 1)** | Syntactic & AST Scope Conflict Pre-Check $\text{Conflict}(\Delta C_i, \Delta C_j)$ | `src/arbitration/conflict_detector.py` (`ConflictDetector`) |
| **Section III-A (Eq. 2 & 3)** | Dynamic Aging Priority & Starvation-Free Lower Bound $\lambda \ge \frac{\Delta_{\max}}{K - 1}$ | `src/arbitration/aging_priority_queue.py` (`AgingPriorityQueue`) |
| **Section III-A (Theorem 1)** | Bounded Deferral Ceiling $D_{\text{bound}} = \lceil \Delta_{\max}/\lambda \rceil + (M-1)K = 28$ | `src/arbitration/aging_priority_queue.py` (`compute_theorem1_max_deferral_bound`) |
| **Section III-A ($\oplus$)** | Incremental HAG Update Operator $G' = G \oplus \Delta C_{\text{orth}}$ | `src/graph/incremental_operator.py` (`IncrementalGraphOperator`) |
| **Section III-B (Eq. 4)** | Empirical Layer Boundary Gate $r_{\text{layer}}(G')$ over $E^+ \cup E_{\text{re-eval}}$ | `src/gating/layer_gate.py` (`EmpiricalLayerGate`) |
| **Section III-B (Eq. 5)** | Whitelisted Strongly Connected Component (SCC) Cycle Gate $r_{\text{cycle}}(G')$ | `src/gating/cycle_gate.py` (`WhitelistedCycleGate`) |
| **Section III-B (Eq. 6)** | Z-Score Normalized Proportional Agentic Technical Debt ($\text{AgTD}_i$) | `src/gating/agtd_calculator.py` (`AgTDCalculator`) |
| **Section III-B (Eq. 7)** | Variance-Scaled Batch Debt Rule $r_{\text{debt}}(G')$ with finite-sample $t_{1-\alpha,\nu}$ & $\sigma_{\text{comp}}(k)$ | `src/gating/statistical_gate.py` (`StatisticalCLTGate` / `VarianceScaledGate`) |
| **Section III-C (Eq. 8)** | Fine-Grained Edge & Proportional Debt Attribution $\text{Attr}(v)$ | `src/attribution/exhaustive_attributor.py` (`ExhaustiveAttributor`) |
| **Section III-C (Eq. 9)** | Topological State Fingerprinting $\mathcal{F}_t^{(i)}$ & `DetectCycle` ($X \to Y \to X$) | `src/attribution/fingerprint_db.py` (`TopologicalFingerprintDB`) |
| **Section III-C (SPI)** | Single-Point Fault Isolation $\Delta C_{\text{orth}} \leftarrow \Delta C_{\text{orth}} \setminus \{\Delta C_i\}$ | `src/attribution/isolation_manager.py` (`SinglePointIsolationManager`) |
| **Section III-C (Eq. 10)** | Targeted Remediation Prompt Synthesis $\text{Prompt}_{\text{next}}(A_i)$ | `src/attribution/verbalization_engine.py` (`TemplatedVerbalizationEngine`) |
| **Algorithm 1** | End-to-End DAG Governance Loop | `src/workflow/dag_loop.py` (`DAGLoop`) |

---

## Empirical Claims & Reproduction Mapping (Section V)

| Paper Section / Artifact | Key Empirical Findings Verified by Code | Command |
| :--- | :--- | :--- |
| **Table I & Section V-A (RQ1)** | Across 6 workflows on 50 stratified tasks: DAG reduces median $\Delta\text{AgTD}$ by **83.4%** (`+4.39` $\to$ `+0.73`, mean `+4.46±0.96` $\to$ `+0.74±0.32`), cuts external architectural smells (EAVD) from `2.86` to `0.48`, and achieves **64.0%** $\text{ResRate}_{K=10}$ (`59.3s` vs `142.6s` for `Sequential-Merge`, **-58.4%** latency). Held-out clean human batch false-rejection rate is **4.6%** ($\approx \alpha = 0.05$). | `python3 -m experiments.run_rq1_effectiveness --tasks 50` |
| **Table II & Fig. 3a (RQ2)** | Across 160 concurrent batches ($n \in \{3,4,5,6\}$): DAG achieves **97.9%** Precision, **100.0%** Recall, and **0.989** $F_1$ (vs `0.865` for `Counterfactual-LOO`, `0.738` for `Isolated-Gating`, `0.674` for `Naive-Linter`), reducing mean wasted tokens by **86.8%** (`3,279` $\to$ `433`). | `python3 -m experiments.run_rq2_concurrency --batches 40` |
| **Figure 3b & Section V-C (RQ3)** | Across 50 contested tasks ($K=10$): `Naive-Linter` plateaus at **64.0%** with a **36.0%** livelock rate; DAG reaches **96.0%** convergence by $K=3$ and **100.0%** by $K=5$ (**0.0%** livelocks). SPI triggers in **20.0%** (`10/50`) of tasks (**80.0%** healthy peer merge, **42.0%** isolated patch loss); **8 of 10 (80.0%)** isolated tasks converge in solo re-run. Empirical max deferral is $d_{\max} = \mathbf{7} \le 28$ (**Theorem 1**). | `python3 -m experiments.run_rq3_oscillation --tasks 50` |
| **Table III & Section V-D (Ablation)** | Across 600 balanced runs ($N=100$ per variant across 6 variants): Removing `FingerprintDB` causes `15` livelocks (`5.00` avg iters); removing `AgingArbitration` causes `23` starvations; removing `VarianceScaling` doubles $\Delta\text{AgTD}$ to `+1.67±0.54`; removing `Attribution` causes `24` livelocks (`+1.56±0.51`); removing `ARM-Rules` causes `25` livelocks (`+1.07±0.39`). | `python3 -m experiments.run_ablation --tasks 100` |

---

## System Requirements & Zero-Dependency Portability

- **Python Runtime:** Python $\ge 3.10$ (tested on Python 3.10, 3.11, 3.12, 3.13, and 3.14).
- **Zero-Dependency Standard-Library Execution:** All core algorithms (`src/`), statistical tests (`evaluation/`), experiment drivers (`experiments/`), and unit tests (`tests/`) include native pure-Python fallbacks for graph SCC decomposition, YAML loading, non-parametric hypothesis testing (Wilcoxon, Mann-Whitney U, Cliff's $d$, Holm-Bonferroni), and TikZ/LaTeX export. You can run the entire suite immediately with standard `python3` even in air-gapped or minimal environments.
- **Full Conda / Pip / Docker Support:** Standard `requirements.txt`, `environment.yml`, `pyproject.toml`, and `Dockerfile` are provided for full third-party library integration (`networkx`, `scipy`, `numpy`, `matplotlib`, `pytest`).

---

## Quick Start & One-Click Reproduction

To run all unit tests and reproduce **Table I, Table II, Table III, Figure 3(a), and Figure 3(b)** in one command:

```bash
chmod +x run_all_experiments.sh
./run_all_experiments.sh
```

Or run the pre-flight smoke test:

```bash
python3 -c "
import src as dag
print('[✓] Framework Version:', dag.__version__)
for eq, cls_name in dag.get_framework_summary().items():
    print(f'    {eq:<42} -> {cls_name}')
"
```

---

## Step-by-Step Experiment Reproduction

### 1. Unit Test Suite (Equations 1–10 & Theorem 1)

Compatible with both standard-library `unittest` and `pytest`:

```bash
python3 -m unittest discover -s tests -v
# or, if pytest is installed:
pytest tests/ -v
```

### 2. RQ1: Architectural Debt Mitigation & Held-Out Calibration (Table I)

```bash
python3 -m experiments.run_rq1_effectiveness --tasks 50
```

- **Console Output:** Prints Table I across all 6 workflows (`Base-Agent`, `Iterative-Test`, `Prompt-Only-Rules`, `Sequential-Merge`, `Naive-Linter`, `DAG-Proposed`) and the held-out human commit batch false-rejection verification (`4.6%` vs nominal $\alpha = 0.05$).
- **Generated Artifacts:** `logs_and_artifacts/experiment_runs_rq1.jsonl` and `evaluation/tikz_generated/table1_reproduced.tex`.

### 3. RQ2: Concurrency Scaling & Attribution Fidelity (Table II & Fig. 3a)

```bash
python3 -m experiments.run_rq2_concurrency --batches 40
python3 -m evaluation.plot_fig3_concurrency
```

- **Console Output:** Prints concurrency scaling ($n \in \{3, 4, 5, 6\}$) and Table II attribution precision, recall, $F_1$, and wasted tokens across `Naive-Linter`, `Isolated-Gating`, `Counterfactual-LOO`, and `DAG`.
- **Generated Artifacts:** `logs_and_artifacts/token_usage_records.csv` and `evaluation/tikz_generated/fig3a_concurrency.tex`.

### 4. RQ3: Oscillation Control, SPI & Theorem 1 Bound (Fig. 3b)

```bash
python3 -m experiments.run_rq3_oscillation --tasks 50
python3 -m evaluation.plot_fig4_cdf_iterations
```

- **Console Output:** Prints Theorem 1 deferral bound verification ($d_{\max} = 7 \le 28$), cumulative convergence CDF for $K \in \{1..10\}$, SPI trigger rate (`20.0%`), isolated patch loss (`42.0%`), healthy peer merge rate (`80.0%`), and solo re-run convergence (`8/10 = 80.0%`).
- **Generated Artifacts:** `logs_and_artifacts/cdf_iterations_data.csv`, `logs_and_artifacts/livelock_trajectories/`, and `evaluation/tikz_generated/fig3b_cdf_iterations.tex`.

### 5. Component Ablation Study (Table III)

```bash
python3 -m experiments.run_ablation --tasks 100
```

- **Console Output:** Prints Table III across all 6 ablation variants (`DAG-Full`, `w/o-FingerprintDB`, `w/o-AgingArbitration`, `w/o-VarianceScaling`, `w/o-Attribution`, `w/o-ARM-Rules`, $N=100$ tasks each = 600 runs).
- **Generated Artifacts:** `logs_and_artifacts/ablation_study_results.jsonl` and `evaluation/tikz_generated/table3_ablation.tex`.

---

## Benchmark Corpus (`data/`)

The benchmark corpus in `data/` comprises 12 open-source repositories (8 Python from SWE-bench-Live: `django`, `flask`, `requests`, `scikit-learn`, `fastapi`, `httpx`, `pydantic`, `sqlalchemy`; 4 Java from AIDev: `spring-boot`, `kafka`, `elasticsearch`, `mockito`):

- `data/clean_history_h/curated_commits.jsonl`: 500 human-vetted clean commits ($\mathcal{H}_{\text{clean}}$, $\hat{\mu}_{\text{clean}} = 0.42$, $\hat{\sigma}_{\text{clean}} = 0.85$).
- `data/clean_history_h/heldout_human_batches_100.jsonl`: 100 held-out multi-commit human batches ($k \in \{3,4,5,6\}$) used to validate the $r_{\text{debt}}$ false-rejection rate ($4.6\%$).
- `data/eval_set_d/benchmarks_500.jsonl`: 500 cross-module evaluation tasks ($\mathcal{D}_{\text{eval}}$), including flags for the 50-task stratified multi-agent benchmark (`is_stratified_50`) and 100-task ablation benchmark (`is_ablation_100`).
- `data/eval_set_d/rq2_concurrent_batches_160.jsonl`: 160 concurrent multi-agent batches ($40$ per concurrency level $n \in \{3,4,5,6\}$) with ground-truth culpability labels.
- `data/architecture_rules/mined_rules_elb.json`: ARM-mined forbidden dependency rules $B(u)$ (`Support >= 0.15`, `Confidence >= 0.90`, `Lift >= 1.25`).

---

## Repository Architecture

```text
dag-governed-autonomy/
├── CITATION.cff                     # Academic citation metadata
├── Dockerfile                       # Self-contained Docker image specification
├── LICENSE                          # Apache-2.0 License
├── README.md                        # Replication & documentation guide
├── environment.yml                  # Conda environment specification
├── pyproject.toml                   # PEP 621 package configuration
├── requirements.txt                 # Pinned Python dependencies
├── run_all_experiments.sh           # One-click reproduction script for all tables & figures
├── config/                          # Hyperparameters & architectural rule catalogs
│   ├── config.yaml                  # Global parameters (K=10, λ=0.50, α=0.05, η=1.2)
│   ├── arch_whitelist.yaml          # Ratified design pattern whitelist A (Observer, Visitor, Strategy)
│   ├── models_config.yaml           # LLM decoding parameters (GPT-4o, Claude 3.5 Sonnet, T=0.2)
│   └── refactoring_patterns.json    # Remediation template catalog for Stage III nudging
├── data/                            # Benchmark datasets (H_clean and D_eval)
│   ├── architecture_rules/          # Mined layer rules B(u) and AgTD calibration parameters
│   ├── clean_history_h/             # 500 clean commits & 100 held-out human batches
│   └── eval_set_d/                  # 500 evaluation tasks & 160 concurrent multi-agent batches
├── src/                             # Core DAG Framework (Section III, Equations 1-10 & Theorem 1)
│   ├── arbitration/                 # Stage I: ConflictDetector (Eq. 1) & AgingPriorityQueue (Eq. 2-3, Thm 1)
│   ├── graph/                       # HAG Model G=(V,E,L,Φ) & IncrementalGraphOperator ⊕
│   ├── gating/                      # Stage II: Layer (Eq. 4), Cycle (Eq. 5), AgTD (Eq. 6), VSG (Eq. 7)
│   ├── attribution/                 # Stage III: Attr(v) (Eq. 8), FingerprintDB (Eq. 9), SPI, Verbalizer (Eq. 10)
│   └── workflow/                    # Algorithm 1 (DAGLoop), Agent Adapters & Benchmark Trace Loader
├── mining/                          # Offline Rule Mining & Statistical Calibration Tools
│   ├── arm_rule_miner.py            # Negative Association Rule Mining (ARM) for r_layer
│   ├── distribution_profiler.py     # Calibrates empirical μ_clean and σ_clean on H_clean
│   └── archunit_rule_importer.py    # Imports ArchUnit (Java) & PyTestArch (Python) rules
├── baselines/                       # Standalone Baseline Runners
│   ├── base_agent_runner.py         # Unconstrained Base-Agent runner
│   └── naive_linter_runner.py       # Batch-rejecting Naive-Linter runner
├── experiments/                     # Main Experiment Drivers (Section V)
│   ├── run_rq1_effectiveness.py     # Reproduces Table I & held-out human batch calibration
│   ├── run_rq2_concurrency.py       # Reproduces Table II & Figure 3(a)
│   ├── run_rq3_oscillation.py       # Reproduces Figure 3(b), SPI metrics & Theorem 1 bound
│   └── run_ablation.py              # Reproduces Table III (600 ablation runs)
├── evaluation/                      # Statistical Testing, LaTeX Table & Figure Exporters
│   ├── statistical_tests.py         # Wilcoxon, Mann-Whitney U, Cliff's d, Holm-Bonferroni
│   ├── export_latex_tables.py       # LaTeX Table I generator
│   ├── plot_fig3_concurrency.py     # Figure 3(a) visualizer & TikZ exporter
│   └── plot_fig4_cdf_iterations.py  # Figure 3(b) CDF visualizer & TikZ exporter
└── tests/                           # Unit Test Suite (Equations 1-10 & Theorem 1)
```

---

## License

Licensed under the Apache License, Version 2.0. See [LICENSE](LICENSE) for full details.
