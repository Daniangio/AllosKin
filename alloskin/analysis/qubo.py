"""
Implements QUBO analysis (Goal 2) using Random Forest Regression
and PyQUBO for solving.

This module is refactored to accept a `target_selection_string`
(e.g., "resid 131 140") and the path to a real topology file
to robustly parse selections and detect overlaps.
"""

import numpy as np
import os
import concurrent.futures
import functools
from typing import Tuple, Dict, Any, List, Optional

# --- New Imports ---
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
import pyqubo
import neal
# --- MODIFICATION: Need MDAnalysis to parse selection string ---
import MDAnalysis as mda
from MDAnalysis.core.selection import SelectionError
from MDAnalysis.core.groups import AtomGroup
# --- END MODIFICATION ---

# --- Local Imports ---
try:
    from alloskin.common.types import FeatureDict
except ImportError:
    FeatureDict = Dict[str, np.ndarray]
    
from .components import AnalysisComponent


# --- Top-level Helper Function (for parallel processing) ---

def _compute_rf_regression_r2(
    X_features_3d: np.ndarray, 
    y_features_3d: np.ndarray, 
    n_samples: int, 
    cv_folds: int, 
    n_estimators: int
) -> float:
    """
    Internal helper to compute the R^2 score for a single
    regression task: X -> y.
    """
    try:
        # Reshape for sklearn: (n_samples, n_features)
        X = X_features_3d.reshape(n_samples, -1)
        y = y_features_3d.reshape(n_samples, -1)
        
        # RandomForestRegressor natively supports multi-output regression
        reg = RandomForestRegressor(
            n_estimators=n_estimators,
            n_jobs=1,       # CRITICAL: Worker process must be serial
            random_state=42
        )
        
        # Use cross-validation for a robust estimate
        scores = cross_val_score(
            reg, 
            X, 
            y, 
            cv=cv_folds, 
            scoring='r2', # Use R^2 score for regression
            n_jobs=1
        )
        
        # Return the mean R^2 score
        return np.mean(scores)

    except Exception as e:
        print(f"  Error in RF regression worker: {e}")
        return np.nan


# --- Top-level Worker 1: Relevance (R_i -> S) ---

def _compute_relevance_worker(
    item: Tuple[str, np.ndarray], 
    y_target_3d: np.ndarray, 
    n_samples: int, 
    cv_folds: int, 
    n_estimators: int
) -> Tuple[str, float]:
    """
    Worker function for parallel Relevance calculation.
    Computes R^2(R_i -> S)
    """
    res_key, X_features_3d = item
    
    r2_score = _compute_rf_regression_r2(
        X_features_3d, 
        y_target_3d, 
        n_samples, 
        cv_folds, 
        n_estimators
    )
    
    if not np.isnan(r2_score):
        print(f"  Relevance R^2( {res_key} -> S ) = {r2_score:.4f}")
    else:
         print(f"  Relevance R^2( {res_key} -> S ) = FAILED")
         
    return (res_key, r2_score)


# --- Top-level Worker 2: Redundancy (R_i <-> R_j) ---

def _compute_redundancy_worker(
    item: Tuple[str, str], 
    all_features_static: FeatureDict, 
    n_samples: int, 
    cv_folds: int, 
    n_estimators: int
) -> Tuple[str, str, float]:
    """
    Worker function for parallel Redundancy calculation.
    Computes max( R^2(R_i -> R_j), R^2(R_j -> R_i) )
    """
    key_i, key_j = item
    
    try:
        features_i = all_features_static[key_i]
        features_j = all_features_static[key_j]
        
        # R^2(R_i -> R_j)
        r2_ij = _compute_rf_regression_r2(
            features_i, features_j, n_samples, cv_folds, n_estimators
        )
        
        # R^2(R_j -> R_i)
        r2_ji = _compute_rf_regression_r2(
            features_j, features_i, n_samples, cv_folds, n_estimators
        )
        
        max_r2 = np.nanmax([0.0, r2_ij, r2_ji]) # Clamp at 0, R^2 can be negative
        
        print(f"  Redundancy R^2( {key_i} <-> {key_j} ) = {max_r2:.4f} (ij={r2_ij:.4f}, ji={r2_ji:.4f})")
        return (key_i, key_j, max_r2)

    except Exception as e:
        print(f"  Error in redundancy worker for ({key_i}, {key_j}): {e}")
        return (key_i, key_j, np.nan)


# --- Analysis Component Class ---

class QUBOSet(AnalysisComponent):
    """
    Computes the QUBO Hamiltonian components using RF Regression
    and solves for the optimal predictive set.
    """
        
    def run(self, 
            data: Tuple[FeatureDict, np.ndarray, Dict[str, str]], 
            num_workers: int = None,
            **kwargs
        ) -> Dict[str, Any]:
        """
        Computes the relevance vector (h) and redundancy matrix (J)
        and solves the QUBO problem.
        
        Args:
            data: A tuple from prepare_static_analysis_data
                  (all_features_static, labels_Y, mapping)
            num_workers (int, optional): Number of parallel processes.
            **kwargs:
                target_selection_string (str): REQUIRED. MDAnalysis selection
                    string for the target(s) S (e.g., 'resid 131 140').
                active_topo_file (str): REQUIRED. Path to the topology file
                    to parse selections against.
                lambda_redundancy (float): Penalty coefficient. Default: 1.0
                num_solutions (int): Solutions to find. Default: 5
                qubo_cv_folds (int): CV folds for RF. Default: 3
                qubo_n_estimators (int): Trees for RF. Default: 50
        """
        print("\n--- Running QUBO Hamiltonian (Random Forest) ---")
        
        # --- 1. Unpack Data and Kwargs ---
        all_features_static, _, mapping = data # labels_Y not needed
        
        max_workers = num_workers if num_workers is not None else os.cpu_count()
        print(f"Using max {max_workers or 'all'} workers for analysis.")

        target_selection_string = kwargs.get('target_selection_string')
        if not target_selection_string:
            raise ValueError("QUBO analysis requires 'target_selection_string' in kwargs")
        
        active_topo_file = kwargs.get('active_topo_file')
        if not active_topo_file:
            raise ValueError("QUBO analysis requires 'active_topo_file' in kwargs")
        
        lambda_redundancy = float(kwargs.get('lambda_redundancy', 1.0))
        num_solutions = int(kwargs.get('num_solutions', 5))
        cv_folds = int(kwargs.get('qubo_cv_folds', 3))
        n_estimators = int(kwargs.get('qubo_n_estimators', 50))
        
        print(f"Parameters: Target='{target_selection_string}', Lambda={lambda_redundancy}")
        print(f"RF Params: {cv_folds}-fold CV, {n_estimators} trees")

        if not all_features_static:
            print("  Warning: No features found to analyze.")
            return {}
        if not mapping:
            raise ValueError("QUBO analysis requires the 'mapping' dictionary, but it is missing.")

        # --- 2. Prepare Target (S) and Candidate (R_i) sets ---
        
        print(f"Loading topology {active_topo_file} to resolve selections...")
        try:
            u = mda.Universe(active_topo_file)
        except Exception as e:
            raise ValueError(f"Failed to load topology file {active_topo_file}: {e}")

        # 1. Select the target atom group (S)
        try:
            target_ag = u.select_atoms(target_selection_string)
            if target_ag.n_atoms == 0:
                raise ValueError(f"Target selection '{target_selection_string}' matched 0 atoms.")
        except SelectionError as e:
            raise ValueError(f"Invalid target selection string: '{target_selection_string}'. Error: {e}")
        
        print(f"Target selection '{target_selection_string}' resolved to {target_ag.n_atoms} atoms.")

        target_keys = set()
        candidate_keys = [] # This is the disjoint set R
        n_samples = 0
        if all_features_static:
            n_samples = next(iter(all_features_static.values())).shape[0]

        # 2. Partition all available features into Target (S) or Candidate (R)
        print("Partitioning features into Target (S) and Candidate (R) pools...")
        for key, selection_string in mapping.items():
            if key not in all_features_static:
                continue # This key wasn't in the common aligned set

            try:
                candidate_ag = u.select_atoms(selection_string)
                if candidate_ag.n_atoms == 0:
                    print(f"  Warning: Candidate '{key}' ('{selection_string}') matched 0 atoms. Skipping.")
                    continue
                
                # Check for overlap
                # We use the intersection of the *residues* to be safe
                overlap_residues = AtomGroup.intersection(target_ag, candidate_ag).residues
                
                if overlap_residues.n_residues > 0:
                    # This key belongs to the target set
                    print(f"  -> Found Target key: {key} (overlaps with S)")
                    target_keys.add(key)
                else:
                    # This key is a valid candidate
                    candidate_keys.append(key)

            except SelectionError as e:
                print(f"  Warning: Could not parse selection for candidate '{key}' ('{selection_string}'). Skipping. Error: {e}")
            except Exception as e:
                print(f"  Warning: Error processing candidate '{key}'. Skipping. Error: {e}")


        # 3. Build the final target feature vector (S)
        if not target_keys:
             raise ValueError(f"Target selection '{target_selection_string}' did not match any residues from the mapping.")
             
        print(f"Target selection resolved to feature keys: {target_keys}")

        target_features_list: List[np.ndarray] = []
        for key in sorted(list(target_keys)): # Sort for consistent concatenation
            target_features_list.append(all_features_static[key])
        
        if not target_features_list:
            # This should be caught by the check above, but as a failsafe
            raise ValueError(f"Target keys {target_keys} not found in features dictionary.")
        
        if not candidate_keys:
            print("  Warning: No candidate residues found (all residues are targets?).")
            return {}
            
        # Concatenate target features if it's a motif
        # Shape: (n_samples, 1, n_feat_S)
        target_features_S_3d = np.concatenate(target_features_list, axis=-1)
        candidate_keys.sort()
        
        print(f"Target S shape: {target_features_S_3d.shape}. Candidates: {len(candidate_keys)}.")
        
        # --- 3. Compute Relevance (h_i) in Parallel ---
        print("Computing Relevance vector (h_i = -R^2(R_i -> S))...")
        raw_relevance_scores_r2: Dict[str, float] = {}
        
        partial_relevance_worker = functools.partial(
            _compute_relevance_worker,
            y_target_3d=target_features_S_3d,
            n_samples=n_samples,
            cv_folds=cv_folds,
            n_estimators=n_estimators
        )
        
        items_to_process = [(k, all_features_static[k]) for k in candidate_keys]
        
        with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
            results = executor.map(partial_relevance_worker, items_to_process)
            for key, r2_score in results:
                raw_relevance_scores_r2[key] = r2_score

        # --- 4. Compute Redundancy (J_ij) in Parallel ---
        print("Computing Redundancy matrix (J_ij = max(R^2(R_i <-> R_j)))...")
        raw_redundancy_scores_r2: Dict[str, Dict[str, float]] = {}
        
        redundancy_jobs = []
        for i in range(len(candidate_keys)):
            for j in range(i + 1, len(candidate_keys)):
                redundancy_jobs.append((candidate_keys[i], candidate_keys[j]))
        
        print(f"Submitting {len(redundancy_jobs)} redundancy calculations...")
        
        partial_redundancy_worker = functools.partial(
            _compute_redundancy_worker,
            all_features_static=all_features_static,
            n_samples=n_samples,
            cv_folds=cv_folds,
            n_estimators=n_estimators
        )

        with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
            results = executor.map(partial_redundancy_worker, redundancy_jobs)
            for key_i, key_j, max_r2 in results:
                if np.isnan(max_r2):
                    continue
                if key_i not in raw_redundancy_scores_r2:
                    raw_redundancy_scores_r2[key_i] = {}
                raw_redundancy_scores_r2[key_i][key_j] = max_r2

        # --- 5. Build and Solve QUBO (using PyQUBO) ---
        print("Assembling Hamiltonian for PyQUBO...")
        h_i_terms: Dict[str, float] = {}
        J_ij_terms: Dict[str, Dict[str, float]] = {}
        
        # 1. Create a dictionary of pyqubo binary variables
        x_vars = {key: pyqubo.Binary(key) for key in candidate_keys}

        # 2. Build Hamiltonian terms symbolically
        relevance_term = 0.0
        redundancy_term = 0.0

        # Build linear (h_i) terms
        for key in candidate_keys:
            r2 = raw_relevance_scores_r2.get(key, np.nan)
            if np.isnan(r2):
                h_i = 1e6 # Penalize failures heavily
            else:
                h_i = -max(r2, 0.0) # h_i = -Relevance
            
            relevance_term += h_i * x_vars[key]
            h_i_terms[key] = h_i # For logging

        # Build quadratic (J_ij) terms
        for key_i, partners in raw_redundancy_scores_r2.items():
            for key_j, r2 in partners.items():
                # J_ij = lambda * Redundancy
                J_ij = lambda_redundancy * max(r2, 0.0) 
                
                # Add quadratic term
                redundancy_term += J_ij * x_vars[key_i] * x_vars[key_j]
                
                if key_i not in J_ij_terms:
                    J_ij_terms[key_i] = {}
                J_ij_terms[key_i][key_j] = J_ij # For logging
        
        # 3. Create full Hamiltonian and compile
        H = relevance_term + redundancy_term
        model = H.compile()
            
        # 4. Extract QUBO and solve with Neal's SimulatedAnnealingSampler
        Q, offset = model.to_qubo()
        
        if not Q:
            print("  Error: PyQUBO generated an empty QUBO matrix. Cannot solve.")
            return {"error": "PyQUBO matrix construction failed."}
            
        print("Solving QUBO with Simulated Annealing (neal)...")
        solutions_list = []
        try:
            sampler = neal.SimulatedAnnealingSampler()
            # Request more reads to get a good sample of low-energy states
            num_reads = max(num_solutions * 20, 100)
            response = sampler.sample_qubo(Q, num_reads=num_reads)
            
            # 5. Decode results (Parsing dimod.SampleSet directly)
            print("Decoding solutions from dimod.SampleSet...")
            
            variable_names = list(response.variables)
            unique_solutions_found = 0
            seen_solutions = set() # To track unique solutions

            for record in response.record:
                if unique_solutions_found >= num_solutions:
                    break # We have enough unique solutions

                solution_vector_array = record['sample']
                qubo_energy = record['energy']
                num_occurrences = record['num_occurrences']
                
                solution_tuple = tuple(solution_vector_array)
                
                if solution_tuple not in seen_solutions:
                    seen_solutions.add(solution_tuple)
                    unique_solutions_found += 1
                    
                    solution_vector_dict = {}
                    selected_residues = []
                    for i, var_name in enumerate(variable_names):
                        val = int(solution_vector_array[i])
                        solution_vector_dict[var_name] = val
                        if val == 1:
                            selected_residues.append(var_name)
                            
                    solutions_list.append({
                        "energy": qubo_energy + offset,
                        "selected_residues": selected_residues,
                        "num_occurrences": int(num_occurrences),
                        "solution_vector": solution_vector_dict
                    })
            
            if not solutions_list:
                print("  Warning: No solutions found in response.")

            print(f"QUBO solving complete. Found {len(solutions_list)} unique solutions.")

        except NameError:
            print("  FATAL Error: 'neal' or 'pyqubo' not found.")
            print("  Please install with: pip install pyqubo neal")
            return {"error": "Solver libraries not found. 'pip install pyqubo neal'"}
        except Exception as e:
            print(f"  FATAL Error during PyQUBO solve: {e}")
            return {"error": f"PyQUBO solve failed: {e}"}

        # --- 6. Return Final Results ---
        return {
            "analysis_type": "QUBO_RandomForest_PyQUBO",
            "parameters": {
                "target_selection_string": target_selection_string,
                "target_keys_resolved": list(target_keys),
                "lambda_redundancy": lambda_redundancy,
                "num_solutions_requested": num_solutions,
                "cv_folds": cv_folds,
                "n_estimators": n_estimators
            },
            "solutions": solutions_list,
            "hamiltonian_terms": {
                "h_i (linear)": h_i_terms,
                "J_ij (quadratic)": J_ij_terms
            },
            "raw_scores": {
                "relevance_r2": raw_relevance_scores_r2,
                "redundancy_r2_max": raw_redundancy_scores_r2
            }
        }