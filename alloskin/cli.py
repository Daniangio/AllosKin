"""
AllosKin: Command-Line Interface (CLI)
...
"""

# --- 1. Set Environment Variables ---
# This MUST be done before any libraries (numpy, dadapy)
# are imported, to prevent multiprocessing deadlocks.
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
# --- End of change ---


# Example command:
# ...
import argparse
import yaml
import sys
# import os  <-- This is now at the top
from typing import Dict, Any

# Mock imports for a self-contained example
try:
    from alloskin.io.readers import MDAnalysisReader
    from alloskin.features.extraction import FeatureExtractor
    from alloskin.pipeline.builder import DatasetBuilder
    from alloskin.analysis.static import StaticReporters
    from alloskin.analysis.dynamic import TransferEntropy
except ImportError:
    print("Warning: 'alloskin' library modules not found. Using mock components.", file=sys.stderr)
    # Define mock classes to make the script runnable for testing
    import numpy as np
    class MockComponent:
        def __init__(self, *args, **kwargs): pass
        def run(self, *args, **kwargs):
            print(f"[Mock {self.__class__.__name__}] Running with {kwargs.get('num_workers')} workers...")
            return {'res_50': 0.5}
        def prepare_static_analysis_data(self, *args, **kwargs):
            print(f"[Mock DatasetBuilder] Preparing static data...")
            print(f"  -> Active slice: {kwargs.get('active_slice')}")
            print(f"  -> Inactive slice: {kwargs.get('inactive_slice')}")
            return ({'res_50': np.random.rand(100, 1, 6)}, np.random.randint(0, 2, 100))
        def prepare_dynamic_analysis_data(self, *args, **kwargs):
            print(f"[Mock DatasetBuilder] Preparing dynamic data...")
            print(f"  -> Active slice: {kwargs.get('active_slice')}")
            print(f"  -> Inactive slice: {kwargs.get('inactive_slice')}")
            return ({'res_50': np.random.rand(100, 1, 6)}, {'res_131': np.random.rand(100, 1, 6)})

    MDAnalysisReader = MockComponent
    FeatureExtractor = MockComponent
    DatasetBuilder = MockComponent
    StaticReporters = MockComponent
    TransferEntropy = MockComponent


def load_config(config_file: str) -> Dict[str, Any]:
    """Loads the YAML configuration file."""
    print(f"Loading configuration from {config_file}...")
    try:
        with open(config_file, 'r') as f:
            config = yaml.safe_load(f)
        if config is None:
            print("Warning: Config file is empty.")
            return {}
        return config
    except FileNotFoundError:
        print(f"Error: Configuration file not found at {config_file}", file=sys.stderr)
        sys.exit(1) # Exit if config file is missing
    except Exception as e:
        print(f"Error loading YAML configuration: {e}", file=sys.stderr)
        sys.exit(1) # Exit on YAML parsing error

def main():
    """
    Main function to parse arguments and run the analysis pipeline.
    """
    parser = argparse.ArgumentParser(description="AllosKin GPCR Analysis Pipeline")
    parser.add_argument(
        "analysis", 
        choices=["static", "qubo", "dynamic"], 
        help="The analysis goal to run."
    )
    # File path arguments
    parser.add_argument("--active_traj", required=True, help="Path to active state trajectory.")
    parser.add_argument("--active_topo", required=True, help="Path to active state topology.")
    parser.add_argument("--inactive_traj", required=True, help="Path to inactive state trajectory.")
    parser.add_argument("--inactive_topo", required=True, help="Path to inactive state topology.")
    
    # Config and parameter arguments
    parser.add_argument(
        "--config", 
        required=True, 
        help="Path to the residue_selections.yml config file."
    )
    parser.add_argument("--te_lag", type=int, default=10, help="Lag time for TE (in frames). Default: 10.")
    
    # --- New Arguments for Slicing and Multiprocessing ---
    parser.add_argument(
        "--active_slice", 
        type=str, 
        default=None, 
        help="Slice for active trajectory (e.g., '1000:5000:2')."
    )
    parser.add_argument(
        "--inactive_slice", 
        type=str, 
        default=None, 
        help="Slice for inactive trajectory (e.g., '1000:5000:2')."
    )
    parser.add_argument(
        "--num_workers", 
        type=int, 
        default=None, 
        help="Number of parallel workers for analysis. (Default: all available cores)"
    )
    # --- End New Arguments ---

    args = parser.parse_args()

    # --- 1. Initialization ---
    print("--- Initializing Pipeline ---")
    config = load_config(args.config)
    residue_selections = config.get('residue_selections')
    
    if not residue_selections or not isinstance(residue_selections, dict):
        print("Error: 'residue_selections' not found or is empty in config file.", file=sys.stderr)
        sys.exit(1)

    if not any(residue_selections.values()):
        print("Error: 'residue_selections' in config file contains no actual selections.", file=sys.stderr)
        sys.exit(1)

    reader = MDAnalysisReader()
    extractor = FeatureExtractor(residue_selections)
    builder = DatasetBuilder(reader, extractor)
    print("--- Initialization Complete ---")

    # --- 2. Analysis-Based Execution ---
    
    if args.analysis == "static":
        print("\n--- Preparing Data for Static Analysis ---")
        static_data = builder.prepare_static_analysis_data(
            args.active_traj, args.active_topo,
            args.inactive_traj, args.inactive_topo,
            active_slice=args.active_slice,
            inactive_slice=args.inactive_slice
        )
        analyzer = StaticReporters()
        results = analyzer.run(static_data, num_workers=args.num_workers)
        print("\n--- Final Static Results (Sorted by best reporter, lowest II) ---")
        print(results)

    elif args.analysis == "qubo":
        print("\n--- Preparing Data for QUBO Analysis ---")
        static_data = builder.prepare_static_analysis_data(
            args.active_traj, args.active_topo,
            args.inactive_traj, args.inactive_topo,
            active_slice=args.active_slice,
            inactive_slice=args.inactive_slice
        )
        # analyzer = QUBOSet()
        # results = analyzer.run(static_data, target_switch='res_131', num_workers=args.num_workers)
        print("--- Running Optimal Predictive Set (QUBO) ---")
        print("NOTE: QUBO analyzer is not implemented in this scaffold.")
        print("It would use the same `static_data` as the 'static' analysis.")
        results = "Not Implemented"

    elif args.analysis == "dynamic":
        print("\n--- Preparing Data for Dynamic Analysis ---")
        dynamic_data = builder.prepare_dynamic_analysis_data(
            args.active_traj, args.active_topo,
            args.inactive_traj, args.inactive_topo,
            active_slice=args.active_slice,
            inactive_slice=args.inactive_slice
        )
        print("--- Running Dynamic 'Orchestrated Action' (Transfer Entropy) ---")
        analyzer = TransferEntropy()
        results = analyzer.run(dynamic_data, lag=args.te_lag, num_workers=args.num_workers)
        print("\n--- Final Dynamic Results ---")
        print(results)

if __name__ == "__main__":
    # Create a dummy config for testing if not present
    try:
        with open("dummy_config.yml", "x") as f:
            f.write("residue_selections:\n  res_50: 'resid 50'\n")
    except FileExistsError:
        pass # File already exists, skip
        
    # Example mock run:
    # python cli.py static --active_traj a.xtc --active_topo a.pdb --inactive_traj i.xtc --inactive_topo i.pdb --config dummy_config.yml --active_slice "100:200:1" --num_workers 4
    main()