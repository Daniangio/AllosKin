"""
Implements QUBO analysis by computing Hamiltonian components.
"""

import numpy as np
from typing import Tuple, Dict, Any
# from dadapy.information import InformationImbalance

from alloskin.common.types import FeatureDict
from .components import AnalysisComponent

class QUBOSet(AnalysisComponent):
    """
    Computes the components for the QUBO Hamiltonian.
    H(x) = sum_i h_i * x_i + sum_ij J_ij * x_i * x_j
    
    where:
    - h_i (relevance) = -D(X_i -> S)  (Negative II from residue 'i' to target 'S')
    - J_ij (redundancy) = D(X_i -> X_j) (II between residues 'i' and 'j')
    """
    
    def run(self, data: Tuple[FeatureDict, np.ndarray], **kwargs) -> Dict[str, Any]:
        """
        Computes the relevance vector (h) and redundancy matrix (J)
        for the QUBO problem.
        
        Args:
            data: A tuple from prepare_static_analysis_data
                  (all_features_static, labels_Y)
            **kwargs:
                target_switch (str): The residue key to use as the target S.
        """
        print("\n--- Running QUBO Hamiltonian Component Calculation ---")
        all_features_static, _ = data # labels_Y are not needed here
        
        target_switch = kwargs.get('target_switch')
        if not target_switch:
            raise ValueError("QUBO analysis requires 'target_switch' in kwargs")
        
        if target_switch not in all_features_static:
            raise ValueError(f"Target switch '{target_switch}' not found in features.")

        # Get the target features S
        target_features_S = all_features_static[target_switch]
        
        relevance_h = {}
        redundancy_J = {}
        res_keys = sorted(all_features_static.keys()) # Use sorted list for consistent matrix
        
        print(f"Target switch: {target_switch}")
        print("Computing relevance vector (h) and redundancy matrix (J)...")

        # for i, res_key_i in enumerate(res_keys):
        #     print(f"  Processing {res_key_i} ({i+1}/{len(res_keys)})...")
        #     features_i = all_features_static[res_key_i]
        #     redundancy_J[res_key_i] = {}
            
        #     # 1. Compute Relevance: h_i = -D(X_i -> S)
        #     if res_key_i == target_switch:
        #         relevance_h[res_key_i] = -np.inf # Self-relevance is max
        #     else:
        #         try:
        #             ii_relevance = InformationImbalance(features_i, target_features_S)
        #             # We negate it, as QUBO minimizes. We want to *maximize* relevance.
        #             relevance_h[res_key_i] = -ii_relevance.compute_imbalance()
        #         except Exception as e:
        #             print(f"    Error (relevance) D({res_key_i} -> S): {e}")
        #             relevance_h[res_key_i] = np.nan

        #     # 2. Compute Redundancy: J_ij = D(X_i -> X_j)
        #     for j, res_key_j in enumerate(res_keys):
        #         if i >= j: # Matrix is symmetric, J_ij = J_ji
        #             continue
                
        #         features_j = all_features_static[res_key_j]
                
        #         try:
        #             ii_redundancy = InformationImbalance(features_i, features_j)
        #             # We want to *minimize* redundancy. Term is positive.
        #             j_ij = ii_redundancy.compute_imbalance()
        #             redundancy_J[res_key_i][res_key_j] = j_ij
        #         except Exception as e:
        #             print(f"    Error (redundancy) D({res_key_i} -> {res_key_j}): {e}")
        #             redundancy_J[res_key_i][res_key_j] = np.nan
        
        print("QUBO component calculation complete.")
        
        return {
            "target": target_switch,
            "relevance_vector_h": relevance_h,
            "redundancy_matrix_J": redundancy_J,
            "message": "QUBO Hamiltonian components computed. A solver is required to find the optimal set."
        }