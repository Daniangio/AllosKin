"""
Implements Dynamic "Orchestrated Action" (Transfer Entropy)
"""

import numpy as np
from typing import Tuple, Dict, Any, List
# import idtxl

from alloskin.common.types import FeatureDict
from .components import AnalysisComponent

class TransferEntropy(AnalysisComponent):
    """Implements dynamic causal network analysis using idtxl."""
    
    def _run_te_analysis(self, feature_dict: FeatureDict, **kwargs) -> List[Dict[str, Any]]:
        """
        Private helper to run TE on a single state's data.
        """
        
        print("  Collating features for IDTxl...")
        all_features_list = []
        variable_names = []
        
        # idtxl expects data as (N_variables, N_frames)
        # Our features are (N_frames, N_features_per_residue)
        for res_key, features in sorted(feature_dict.items()):
            if features.ndim != 2 or features.shape[0] < 10: # Basic sanity check
                print(f"    Skipping {res_key}: insufficient data")
                continue
                
            n_frames, n_features_per_res = features.shape
            
            # Transpose from (N_frames, N_features) to (N_features, N_frames)
            features_T = features.T
            all_features_list.append(features_T)
            
            # Create names for each feature dimension
            for i in range(n_features_per_res):
                variable_names.append(f"{res_key}_f{i}")
        
        if not all_features_list:
            print("  No features to collate. Aborting analysis.")
            return []
            
        # Stack into one large (N_total_variables, N_frames) array
        collated_data = np.vstack(all_features_list)
        
        print(f"  Data shape for IDTxl: {collated_data.shape}")
        
        # 2. Define settings for IDTxl
        te_lag = kwargs.get('lag', 1)
        settings = {
            'method': 'TE',
            'estimator': 'KSG',      # Kraskov-Stoegbauer-Grassberger estimator
            'k': 1,                  # k=1 is standard for KSG
            'tau_xy': te_lag,        # The lag from source to target
            'tau_z': 1,              # Lag for embedding target history
            'max_lag_sources': 5,    # Max lag to check for sources
            'min_lag_sources': 1,
            'min_var_data': 1e-10,   # For numerical stability
        }
        
        print(f"  Initializing IDTxl NetworkAnalysis with lag={te_lag}...")
        # try:
        #     net = idtxl.NetworkAnalysis(collated_data, settings, variable_names=variable_names)
            
        #     # 3. Run analysis
        #     # We run with a p-value threshold to find significant links
        #     print("  Running network analysis (this may take time)...")
        #     results = net.analyse_network(p_value=0.05)
            
        #     # 4. Parse results
        #     links = results.get_links(p_value=0.05)
            
        #     network = []
        #     for (target_name, source_name, lag), stats in links['links'].items():
        #         network.append({
        #             "source": source_name,
        #             "target": target_name,
        #             "lag": lag,
        #             "te": stats['te'],
        #             "p_value": stats['p_value']
        #         })
            
        #     print(f"  Found {len(network)} significant links.")
        #     return network
            
        # except Exception as e:
        #     print(f"  IDTxl analysis failed: {e}")
        #     raise # Re-raise to fail the job

    def run(self, data: Tuple[FeatureDict, FeatureDict], **kwargs) -> Dict[str, Any]:
        """
        Runs the Transfer Entropy (TE) calculation on both states.
        
        Args:
            data: A tuple from prepare_dynamic_analysis_data
                  (features_active, features_inactive)
            **kwargs:
                lag (int): The time lag (tau) for TE.
        """
        print("\n--- Running Dynamic 'Orchestrated Action' (Transfer Entropy) ---")
        features_active, features_inactive = data
        
        te_lag = kwargs.get('lag', 10)
        te_kwargs = {'lag': te_lag}
        
        print("Calculating TE for INACTIVE state...")
        M_inactive = self._run_te_analysis(features_inactive, **te_kwargs)
        
        print("Calculating TE for ACTIVE state...")
        M_active = self._run_te_analysis(features_active, **te_kwargs)
        
        print("Dynamic analysis complete.")
        
        return {
            "M_inactive": M_inactive,
            "M_active": M_active,
            "parameters": te_kwargs
        }