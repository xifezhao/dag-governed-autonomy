# ==============================================================================
# Artifact Evaluation Dockerfile: Dynamic Architectural Guardrail (DAG)
# Paper: Governed Autonomy: Halting Architectural Entropy in Agentic Software 
#        Workflows via Dynamic Guardrails (SANER 2026)
# Track: Agentic AI systems and autonomous software engineering workflows
# 
# Description: Self-contained container environment for reproducible evaluation
# Standards: Compliant with IEEE/ACM Artifact Evaluation Guidelines (Badges: Reusable)
# ==============================================================================

FROM python:3.10-slim-bullseye

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    DEBIAN_FRONTEND=noninteractive \
    DAG_OFFLINE_REPLAY=true \
    PYTHONPATH="/workspace/dag-governed-autonomy:${PYTHONPATH}"

# Install base system packages and build essentials
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    curl \
    graphviz \
    libgraphviz-dev \
    texlive-latex-base \
    texlive-latex-extra \
    texlive-fonts-recommended \
    && rm -rf /var/lib/apt/lists/*

# Set up working directory
WORKDIR /workspace/dag-governed-autonomy

# Copy dependency requirements
COPY requirements.txt .

# Install pinned Python dependencies
RUN pip install --upgrade pip setuptools wheel && \
    pip install -r requirements.txt

# Copy entire replication codebase
COPY . .

# Create necessary output directories
RUN mkdir -p logs_and_artifacts/execution_traces \
             logs_and_artifacts/livelock_trajectories \
             evaluation/tikz_generated \
             figures \
             data/eval_set_d \
             data/clean_history_h

# Set container default command to run test suite and smoke test
CMD ["pytest", "tests/", "-v"]
