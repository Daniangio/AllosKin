# AllosKin
AllosKin (Allostery + Kinetics) is a research project and Python pipeline for analyzing G-Protein Coupled Receptor (GPCR) activation dynamics. It moves beyond static correlation to map the causal information flow and allosteric signal networks that define functional states.

This project is based on a 3-goal experimental plan.

# Project Goals
Goal 1: Identify Static Reporters: Find individual residues ("switches") whose conformational distributions are most predictive of the global functional state (Active vs. Inactive) using Information Imbalance.
Goal 2: Find Optimal Predictive Sets (QUBO): Identify the minimal, non-redundant set of residues that collectively predicts the state of a primary switch. This is a combinatorial optimization problem solved with QUBO.
Goal 3: Map Dynamic "Orchestrated Action" (Transfer Entropy): Move from correlation to causality. Use Transfer Entropy (TE) on unbiased, time-ordered simulation data to build two distinct causal networks, $M^{\text{inactive}}$ and $M^{\text{active}}$, and identify the "triggers" and "latches" of activation by analyzing the difference network, $\Delta M$.

# Repository Structure
This repository is a monorepo containing the core scientific library, a web API, and a web frontend.
/alloskin/: The core Python library. Contains all logic for trajectory I/O, feature extraction, and the analysis components for Goals 1-3.
/backend/: A FastAPI web server that provides an HTTP API to the alloskin library.
/frontend/: A React-based web application for visualizing the results (e.g., interactive network graphs).
/config/: Default configurations, including residue selections.
/docs/: Project documentation and the original research plan./tests/: Unit and integration tests for the alloskin library.

# Quick Start
1. InstallationClone the repository and install the core library in editable mode.git clone [https://github.com/your-username/AllosKin.git](https://github.com/your-username/AllosKin.git)
cd AllosKin

# Install core library dependencies
pip install -r requirements.txt

# Install the library in editable mode
pip install -e .

# Install backend dependencies
pip install -r backend/requirements.txt
2. Running via Command Line (CLI)After installing with pip install -e ., the alloskin command will be available.alloskin goal1 \
  --active_traj /path/to/active.xtc \
  --active_topo /path/to/active.pdb \
  --inactive_traj /path/to/inactive.xtc \
  --inactive_topo /path/to/inactive.pdb \
  --config config/residue_selections.yml
3. Running with Docker (Recommended)
This is the simplest way to run the backend and frontend.# From the root AllosKin/ directory
docker-compose up --build
API will be available at http://127.0.0.1:8000/docsFrontend will be available at http://127.0.0.1:30004. Running the Web Server Manually# From the root AllosKin/ directory
uvicorn backend.main:app --reload
