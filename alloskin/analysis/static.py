"""
Implements Static Reporters analysis.
"""

import numpy as np
from typing import Tuple, Dict
from dadapy.metric_comparisons import MetricComparisons

# Assuming FeatureDict is defined in this module's context
# e.g., from alloskin.common.types import FeatureDict
# For this file, we only need the type hint, but the data
# structure is Dict[str, np.ndarray]
FeatureDict = Dict[str, np.ndarray]

# Assuming AnalysisComponent is a base class, e.g.:
class AnalysisComponent:
    def run(self, data, **kwargs):
        raise NotImplementedError

class StaticReporters(AnalysisComponent):
    """Implements Static Reporters analysis using Information Imbalance."""

    def run(self, data: Tuple[FeatureDict, np.ndarray], **kwargs) -> Dict[str, float]:
        """
        Runs the Information Imbalance (II) calculation.

        Args:
            data: A tuple from prepare_static_analysis_data
                  (all_features_static, labels_Y)
                  - all_features_static: Dict {res_key: (n_frames, 1, 6) array}
                  - labels_Y: (n_frames,) array

        Returns:
            A dictionary of {res_key: ii_score}, sorted by score (lowest is best).
        """
        print("\n--- Running Static Reporters (Information Imbalance) ---")
        all_features_static, labels_Y = data

        # We need a 2D array for Y for dadapy: (n_samples, 1)
        labels_Y_2d = labels_Y.reshape(-1, 1)
        
        # This dictionary will store the final scalar II scores
        scores: Dict[str, float] = {}
        
        if not all_features_static:
            print("  Warning: No features found to analyze.")
            return scores

        # --- Correct dadapy Workflow ---
        # 1. Compute the ranks for the target space (Y, the labels)
        # We only need to do this once.
        n_samples = labels_Y_2d.shape[0]
        # maxk=n_samples-1 computes all ranks
        maxk = 100 
        
        print(f"Calculating target ranks for labels (N={n_samples})...")
        try:
            mc_labels = MetricComparisons(coordinates=labels_Y_2d, maxk=maxk)
            # compute_distances() calculates distances and stores ranks
            mc_labels.compute_distances() 
            label_ranks = mc_labels.dist_indices
        except Exception as e:
            print(f"  FATAL Error: Could not compute distances for labels. Aborting. {e}")
            return scores
        # --- End Target Rank Calculation ---

        print(f"Calculating Information Imbalance (II) for {len(all_features_static)} residues...")
        for res_key, features_3d in all_features_static.items():
            if features_3d.shape[0] != n_samples:
                print(f"  Warning: Mismatch in frames for {res_key}. Skipping.")
                continue

            try:
                # features_3d shape is (n_samples, 1, 6) from extraction
                # We must reshape it to (n_samples, 6) for dadapy
                features_2d = features_3d.reshape(n_samples, -1) # Reshapes to (n_samples, 6)
                
                # 2. Instantiate MetricComparisons for the source space (X_i, the features)
                mc_features = MetricComparisons(coordinates=features_2d, maxk=maxk)

                # 3. Define which coordinates from X_i to test.
                # We want to test all of them [phi,sin(phi),psi,sin(psi),chi1,sin(chi1)]
                # as a single block.
                coord_list = [list(range(features_2d.shape[1]))] # e.g., [[0, 1, 2, 3, 4, 5]]
                
                # 4. Compute imbalance from X_i against Y's ranks
                # This method computes imbalance for each set of coords in coord_list
                # against the provided target_ranks.
                imbalances_array = mc_features.return_inf_imb_target_selected_coords(
                    target_ranks=label_ranks,
                    coord_list=coord_list
                )
                
                # The output 'imbalances_array' has shape (2, len(coord_list))
                # Row 0: Delta(Target -> Source) i.e., Delta(Y -> X_i)
                # Row 1: Delta(Source -> Target) i.e., Delta(X_i -> Y)
                # We want the second one, for the first (and only) coord set.
                delta_Xi_Y = imbalances_array[1, 0]

                scores[res_key] = delta_Xi_Y
                print(f"  II( {res_key} -> Y ) = {delta_Xi_Y:.4f}")

            except Exception as e:
                print(f"  Error computing II for {res_key}: {e}")
                scores[res_key] = np.nan

        # Sort by II score (lowest is best)
        # Filter out any nan values before sorting if necessary
        sorted_results = dict(sorted(
            scores.items(),
            key=lambda item: item[1] if not np.isnan(item[1]) else float('inf')
        ))

        print("Static Reporters analysis complete.")
        return sorted_results