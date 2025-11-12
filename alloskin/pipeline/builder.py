"""
Implements the Strategy Pattern for data preparation.

This class enforces the critical separation between:
1. Static, time-scrambled, labeled data (Goals 1 & 2)
2. Dynamic, time-ordered, separate data (Goal 3)
"""

import numpy as np
from typing import Tuple, Dict

# Use relative imports within the package
from alloskin.io.readers import AbstractTrajectoryReader
from alloskin.features.extraction import FeatureExtractor
from alloskin.common.types import FeatureDict

class DatasetBuilder:
    """
    Handles the CRITICAL step of preparing data for different goals.
    """
    def __init__(self, reader: AbstractTrajectoryReader, extractor: FeatureExtractor):
        self.reader = reader
        self.extractor = extractor

    def prepare_static_analysis_data(
        self,
        active_traj_file: str, active_topo_file: str,
        inactive_traj_file: str, inactive_topo_file: str
    ) -> Tuple[FeatureDict, np.ndarray]:
        """
        Prepares data for GOAL 1 and GOAL 2.
        - Loads both trajectories.
        - Computes features for all residues.
        - Concatenates features into a single (time-scrambled) dataset.
        - Creates a corresponding binary state label vector Y.
        
        Returns:
            - all_features_static: Dict[res_key, np.ndarray(N_total_frames, N_features)]
            - labels_Y: np.ndarray(N_total_frames,)
        """
        print("\n--- Preparing STATIC Analysis Dataset (Goals 1 & 2) ---")
        # Load trajectories
        traj_active = self.reader.load_trajectory(active_traj_file, active_topo_file)
        traj_inactive = self.reader.load_trajectory(inactive_traj_file, inactive_topo_file)

        # Extract features
        features_active = self.extractor.extract_all_features(traj_active)
        features_inactive = self.extractor.extract_all_features(traj_inactive)

        n_frames_active = len(traj_active.trajectory)
        n_frames_inactive = len(traj_inactive.trajectory)
        n_total_frames = n_frames_active + n_frames_inactive

        # Create labels vector Y
        # Y = 1 for Active, Y = 0 for Inactive
        labels_Y = np.concatenate([
            np.ones(n_frames_active, dtype=int),
            np.zeros(n_frames_inactive, dtype=int)
        ])

        # Concatenate features
        all_features_static: FeatureDict = {}
        for res_key in self.extractor.residue_selections:
            all_features_static[res_key] = np.concatenate(
                [features_active[res_key], features_inactive[res_key]],
                axis=0
            )
            print(f"Concatenated features for {res_key}: {all_features_static[res_key].shape}")

        # 5. CRITICAL: Shuffle data
        # This is a deliberate step to enforce the "static" nature
        # of the dataset and break all time-correlations.
        # Information Imbalance is invariant to this,
        # but it prevents accidental misuse.
        print("Shuffling concatenated dataset to break time-correlations...")
        shuffle_indices = np.random.permutation(n_total_frames)
        labels_Y = labels_Y[shuffle_indices]
        for res_key in all_features_static:
            all_features_static[res_key] = all_features_static[res_key][shuffle_indices]
        
        print("Static dataset prepared and shuffled successfully.")
        return all_features_static, labels_Y

    def prepare_dynamic_analysis_data(
        self,
        active_traj_file: str, active_topo_file: str,
        inactive_traj_file: str, inactive_topo_file: str
    ) -> Tuple[FeatureDict, FeatureDict]:
        """
        Prepares data for GOAL 3.
        - Loads both trajectories.
        - Computes features for all residues.
        - Returns two SEPARATE, TIME-ORDERED feature sets.
        
        Returns:
            - features_active: Dict[res_key, np.ndarray(N_active_frames, N_features)]
            - features_inactive: Dict[res_key, np.ndarray(N_inactive_frames, N_features)]
        """
        try:
            print("\n--- Preparing DYNAMIC Analysis Dataset (Goal 3) ---")
            # Load trajectories
            print(f"Loading active trajectory: {active_traj_file}")
            traj_active = self.reader.load_trajectory(active_traj_file, active_topo_file)
            print(f"Loading inactive trajectory: {inactive_traj_file}")
            traj_inactive = self.reader.load_trajectory(inactive_traj_file, inactive_topo_file)

            # Extract features
            print("Extracting features for active state...")
            features_active = self.extractor.extract_all_features(traj_active)
            print("Extracting features for inactive state...")
            features_inactive = self.extractor.extract_all_features(traj_inactive)
            
            print("Dynamic datasets prepared. Data remains separate and time-ordered.")

            return features_active, features_inactive
        except FileNotFoundError as e:
            print(f"Trajectory file not found: {e}")
            raise
        except Exception as e:
            print(f"Error during dynamic data preparation: {e}")
            raise