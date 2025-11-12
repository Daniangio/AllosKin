"""
Defines the long-running job functions that the RQ worker
will execute.

This function runs in the 'worker' container, NOT the 'backend'.
It imports directly from the alloskin library.
"""

import time
import yaml
from typing import Dict, Any

# Import from our library modules
from alloskin.io.readers import MDTrajReader
from alloskin.features.extraction import FeatureExtractor
from alloskin.pipeline.builder import DatasetBuilder
from alloskin.analysis.static import StaticReporters
from alloskin.analysis.dynamic import TransferEntropy

# --- Health Check Job ---

def health_check_job(x: int, y: int) -> int:
    """
    A simple dummy job for the health check endpoint.
    It proves the worker can receive a job and return a result.
    """
    time.sleep(1) # Simulate a small amount of work
    return x + y

# --- Main Analysis Job ---

def run_analysis_job(
    goal: str,
    file_paths: Dict[str, str],
    config_path: str,
    params: Dict[str, Any]
) -> Dict[str, Any]:
    """
    The main analysis function that the worker will run.
    This contains the core logic originally from the CLI.
    """
    try:
        # 1. Initialization
        print(f"[JOB_RUNNER] Loading config from {config_path}")
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        residue_selections = config.get('residue_selections')
        if not residue_selections:
            raise ValueError("'residue_selections' not found in config")

        reader = MDTrajReader()
        extractor = FeatureExtractor(residue_selections)
        builder = DatasetBuilder(reader, extractor)

        paths = file_paths # for brevity
        results = {}

        # 2. Goal-Based Execution
        if goal == "static":
            print("[JOB_RUNNER] Running Goal 1")
            static_data = builder.prepare_static_analysis_data(
                paths['active_traj'], paths['active_topo'],
                paths['inactive_traj'], paths['inactive_topo']
            )
            analyzer = StaticReporters()
            results = analyzer.run(static_data)

        elif goal == "dynamic":
            print("[JOB_RUNNER] Running Goal 3")
            dynamic_data = builder.prepare_dynamic_analysis_data(
                paths['active_traj'], paths['active_topo'],
                paths['inactive_traj'], paths['inactive_topo']
            )
            analyzer = TransferEntropy()
            te_lag = params.get('te_lag', 10)
            results = analyzer.run(dynamic_data, lag=te_lag)
        
        else:
            raise ValueError(f"Unknown goal: {goal}")

        print("[JOB_RUNNER] Job complete.")
        return results

    except Exception as e:
        print(f"[JOB_RUNNER] ERROR: {e}")
        # RQ will catch this exception and mark the job as 'failed'
        raise e