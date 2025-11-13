"""
Implements the Strategy Pattern for data preparation.

This class enforces the critical separation between:
1. Static, time-scrambled, labeled data (Goals 1 & 2)
2. Dynamic, time-ordered, separate data (Goal 3)

Refactored to include:
- Trajectory slicing.
- Thread-based parallel loading and extraction of active/inactive states.
- **Corrected logic:** Slicing is now passed to the Extractor, not the Reader.
"""

import numpy as np
from typing import Tuple, Dict, Optional
import concurrent.futures

# --- Mock classes for standalone runnability ---
# (Assuming real classes are imported from alloskin.io, etc.)
class AbstractTrajectoryReader:
    def load_trajectory(self, traj_file, topo_file):
        print(f"  MockLoading: {traj_file}")
        class MockUniverse:
            def __init__(self, n_frames):
                class MockTrajectory:
                    def __init__(self, n_frames):
                        self._n_frames = n_frames
                    def __len__(self):
                        return self.total_frames()
                    def total_frames(self):
                        return 1000 # Mock total frames
                self.trajectory = MockTrajectory(1000)
        return MockUniverse(1000)

class FeatureExtractor:
    def __init__(self, residue_selections):
        self.residue_selections = residue_selections
    def extract_all_features(self, traj, slice_obj=slice(None)):
        print(f"  MockExtract: Extracting features with slice {slice_obj}")
        n_frames = self._get_sliced_length(traj, slice_obj)
        features = {}
        for key in self.residue_selections:
            n_res = 1 # Mock
            features[key] = np.random.rand(n_frames, n_res, 6)
        return features, n_frames
    
    def _get_sliced_length(self, traj, slice_obj):
        total_frames = traj.trajectory.total_frames()
        start = slice_obj.start or 0
        stop = slice_obj.stop or total_frames
        step = slice_obj.step or 1
        return len(range(start, stop, step))
        
FeatureDict = Dict[str, np.ndarray]
# --- End Mock classes ---


class DatasetBuilder:
    """
    Handles the CRITICAL step of preparing data for different goals.
    Uses a ThreadPoolExecutor to parallelize active/inactive data prep.
    """
    def __init__(self, reader: AbstractTrajectoryReader, extractor: FeatureExtractor):
        self.reader = reader
        self.extractor = extractor

    def _parse_slice(self, slice_str: Optional[str]) -> slice:
        """Converts a string 'start:stop:step' into a slice object."""
        if not slice_str:
            return slice(None) # Returns slice(None, None, None)
        try:
            parts = [int(p) if p else None for p in slice_str.split(':')]
            if len(parts) > 3:
                raise ValueError("Slice string can have at most 3 parts (start:stop:step).")
            parts.extend([None] * (3 - len(parts)))
            return slice(parts[0], parts[1], parts[2])
        except ValueError as e:
            print(f"  Error: Invalid slice string '{slice_str}'. Must be 'start:stop:step'. Using full trajectory. Error: {e}")
            return slice(None)

    def _load_and_extract(
        self, 
        traj_file: str, 
        topo_file: str, 
        slice_obj: slice
    ) -> Tuple[FeatureDict, int]:
        """
        Worker function to be run in a thread.
        Loads trajectory (I/O bound) and then extracts features (CPU bound).
        """
        try:
            # 1. Load the full universe (I/O)
            #    The reader is simple and doesn't know about slicing.
            traj = self.reader.load_trajectory(traj_file, topo_file)
            
            # 2. Extract features *using the slice* (CPU)
            #    The extractor is smart and applies the slice.
            features, n_frames = self.extractor.extract_all_features(
                traj, 
                slice_obj=slice_obj
            )
            return features, n_frames
        except Exception as e:
            print(f"  FATAL ERROR in worker thread for {traj_file}: {e}")
            return {}, 0


    def prepare_static_analysis_data(
        self,
        active_traj_file: str, active_topo_file: str,
        inactive_traj_file: str, inactive_topo_file: str,
        active_slice: Optional[str] = None,
        inactive_slice: Optional[str] = None
    ) -> Tuple[FeatureDict, np.ndarray]:
        """
        Prepares data for GOAL 1 and GOAL 2 in parallel.
        - Loads/extracts active and inactive trajectories concurrently.
        - Concatenates features into a single (time-scrambled) dataset.
        - Creates a corresponding binary state label vector Y.
        """
        print("\n--- Preparing STATIC Analysis Dataset (Goals 1 & 2) [Parallel] ---")
        
        act_slice = self._parse_slice(active_slice)
        inact_slice = self._parse_slice(inactive_slice)
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
            print("Submitting active trajectory task...")
            future_active = executor.submit(
                self._load_and_extract, 
                active_traj_file, active_topo_file, act_slice
            )
            print("Submitting inactive trajectory task...")
            future_inactive = executor.submit(
                self._load_and_extract, 
                inactive_traj_file, inactive_topo_file, inact_slice
            )
            
            print("Waiting for tasks to complete...")
            features_active, n_frames_active = future_active.result()
            features_inactive, n_frames_inactive = future_inactive.result()

        if n_frames_active == 0 or n_frames_inactive == 0:
            raise ValueError("Trajectory loading or extraction failed for one or both states (0 frames returned).")

        n_total_frames = n_frames_active + n_frames_inactive
        print(f"Tasks complete. Active sliced frames: {n_frames_active}, Inactive sliced frames: {n_frames_inactive}, Total: {n_total_frames}")

        labels_Y = np.concatenate([
            np.ones(n_frames_active, dtype=int),
            np.zeros(n_frames_inactive, dtype=int)
        ])

        all_features_static: FeatureDict = {}
        for res_key in features_active:
            if res_key not in features_inactive:
                print(f"Warning: {res_key} found in active but not inactive. Skipping.")
                continue
                
            all_features_static[res_key] = np.concatenate(
                [features_active[res_key], features_inactive[res_key]],
                axis=0
            )
            print(f"Concatenated features for {res_key}: {all_features_static[res_key].shape}")

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
        inactive_traj_file: str, inactive_topo_file: str,
        active_slice: Optional[str] = None,
        inactive_slice: Optional[str] = None
    ) -> Tuple[FeatureDict, FeatureDict]:
        """
        Prepares data for GOAL 3 in parallel.
        - Loads/extracts active and inactive trajectories concurrently.
        - Returns two SEPARATE, TIME-ORDERED feature sets.
        """
        print("\n--- Preparing DYNAMIC Analysis Dataset (Goal 3) [Parallel] ---")

        act_slice = self._parse_slice(active_slice)
        inact_slice = self._parse_slice(inactive_slice)

        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
            print("Submitting active trajectory task...")
            future_active = executor.submit(
                self._load_and_extract, 
                active_traj_file, active_topo_file, act_slice
            )
            print("Submitting inactive trajectory task...")
            future_inactive = executor.submit(
                self._load_and_extract, 
                inactive_traj_file, inactive_topo_file, inact_slice
            )
            
            print("Waiting for tasks to complete...")
            features_active, n_frames_active = future_active.result()
            features_inactive, n_frames_inactive = future_inactive.result()
            
        if n_frames_active == 0 or n_frames_inactive == 0:
            raise ValueError("Trajectory loading or extraction failed for one or both states (0 frames returned).")

        print(f"Dynamic datasets prepared. Active sliced frames: {n_frames_active}, Inactive sliced frames: {n_frames_inactive}.")
        
        return features_active, features_inactive