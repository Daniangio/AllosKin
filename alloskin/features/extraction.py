"""
Handles calculation of dihedral angles and the mandatory
sin/cos transformation.

Refactored to:
- Accept a slice_obj to perform analysis on trajectory slices.
- Use the slice parameters in the Dihedral.run() method.
- Return the number of frames processed.
"""

import MDAnalysis as mda
import MDAnalysis.analysis.dihedrals as mda_dihedrals
import numpy as np
from typing import Dict, List, Optional, Tuple

# Mock types for demonstration
TrajectoryObject = mda.Universe
FeatureDict = Dict[str, np.ndarray]

def deg2rad(deg_array: np.ndarray) -> np.ndarray:
    """Converts an array of degrees to radians."""
    return np.deg2rad(deg_array)

class FeatureExtractor:
    """
    Handles calculation of backbone (phi, psi) and sidechain (chi1)
    dihedral angles and their sin/cos transformation.
    """
    def __init__(self, residue_selections: Dict[str, str]):
        """
        Initializes with a dictionary of residue selections.
        Example: {'res_50': 'resid 50', 'res_131': 'resid 131'}
        """
        self.residue_selections = residue_selections
        print(f"FeatureExtractor initialized for {len(self.residue_selections)} residues.")

    def _transform_to_circular(self, angles_rad: np.ndarray) -> np.ndarray:
        """
        Transforms an array of angles (n_frames, n_residues, n_angles)
        into the 2D vector representation [sin(th), cos(th)].
        Output shape: (n_frames, n_residues, n_angles * 2)
        """
        sin_transformed = np.sin(angles_rad)
        cos_transformed = np.cos(angles_rad)
        
        n_frames, n_residues, n_angles = angles_rad.shape
        circular_features = np.empty((n_frames, n_residues, n_angles * 2))
        
        circular_features[..., 0::2] = sin_transformed
        circular_features[..., 1::2] = cos_transformed
        
        return circular_features

    def _get_sliced_length(self, traj: TrajectoryObject, slice_obj: slice) -> int:
        """Helper to calculate frames in a slice."""
        total_frames = len(traj.trajectory)
        # Get slice parameters, defaulting to full trajectory
        start = slice_obj.start or 0
        stop = slice_obj.stop or total_frames
        step = slice_obj.step or 1
        
        # Use range to correctly calculate the number of items
        return len(range(start, stop, step))

    def _calculate_dihedral_angle(
        self, 
        atom_groups_list: List[Optional[mda.AtomGroup]], 
        n_frames: int,
        slice_obj: slice
    ) -> np.ndarray:
        """
        Helper function to run Dihedral analysis on a list of AtomGroups
        over a specified trajectory slice.
        
        Args:
            atom_groups_list: List of AtomGroups (or None) to analyze.
            n_frames: The *sliced* number of frames, for result array shape.
            slice_obj: The slice object with start/stop/step.
            
        Returns: (n_sliced_frames, len(atom_groups_list)) array of angles.
        """
        mask = np.array([ag is not None for ag in atom_groups_list])
        angles = np.full((n_frames, len(atom_groups_list)), 0.0, dtype=np.float32)
        
        valid_atom_groups = [ag for ag in atom_groups_list if ag is not None]
        
        if not valid_atom_groups:
            return angles
            
        try:
            # Unpack slice parameters for the .run() method
            start = slice_obj.start
            stop = slice_obj.stop
            step = slice_obj.step
            
            # Run analysis only on the valid groups and the specified slice
            analysis = mda_dihedrals.Dihedral(valid_atom_groups).run(
                start=start, 
                stop=stop, 
                step=step
            )
            
            # Verify shape
            if analysis.results.angles.shape[0] != n_frames:
                print(f"  Warning: Dihedral analysis returned {analysis.results.angles.shape[0]} frames, expected {n_frames}. Check slice logic.")
                # Truncate or handle as needed; for now, we'll trust the mask
            
            angles[:, mask] = analysis.results.angles
            
        except Exception as e:
            print(f"    Warning: Dihedral calculation failed: {e}")
            
        return angles


    def extract_all_features(
        self, 
        traj: TrajectoryObject, 
        slice_obj: slice = slice(None)
    ) -> Tuple[FeatureDict, int]:
        """
        Extracts features for all defined residues on a given slice.
        
        Args:
            traj: The MDAnalysis Universe object.
            slice_obj: A slice object (e.g., slice(1000, 5000, 2)).
            
        Returns:
            - A dictionary {selection_key: feature_array}
              where each array has shape (n_sliced_frames, n_residues, 6).
            - The number of frames actually processed (n_sliced_frames).
        """
        all_features_dict: FeatureDict = {}
        # Calculate the number of frames that will be processed
        n_frames = self._get_sliced_length(traj, slice_obj)
        print(f"  Extractor: Processing {n_frames} frames from slice {slice_obj}.")

        STANDARD_PROTEIN_RESNAMES = {
            "ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS", 
            "ILE", "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP", 
            "TYR", "VAL", "HID", "HIE", "HSD", "HSE", "HSP", "CYX",
        }

        for key, sel_string in self.residue_selections.items():
            print(f"    Processing: {key} ({sel_string})...")
            
            try:
                user_residue_group = traj.select_atoms(sel_string)
                
                if len(user_residue_group.residues) == 0:
                    print(f"      Warning: Selection '{sel_string}' found no residues. Skipping.")
                    continue

                protein_residues = user_residue_group.residues[
                    np.isin(user_residue_group.residues.resnames, list(STANDARD_PROTEIN_RESNAMES))
                ]
                
                if len(protein_residues) == 0:
                    print(f"      Warning: Selection '{sel_string}' contains no standard protein residues. Skipping.")
                    continue
                
                phi_sel_list = protein_residues.residues.phi_selections()
                psi_sel_list = protein_residues.residues.psi_selections()
                chi1_sel_list = protein_residues.residues.chi1_selections()

                n_phi = len(phi_sel_list)
                n_psi = len(psi_sel_list)
                
                all_selections = phi_sel_list + psi_sel_list + chi1_sel_list
                
                if not all_selections:
                    print(f"      Warning: No valid dihedrals found for {key}. Skipping.")
                    continue

                # Calculate angles *using the slice*
                all_angles_deg = self._calculate_dihedral_angle(
                    all_selections, 
                    n_frames, 
                    slice_obj
                )
                
                phi_angles = all_angles_deg[:, :n_phi]
                psi_angles = all_angles_deg[:, n_phi : n_phi + n_psi]
                chi1_angles = all_angles_deg[:, n_phi + n_psi :]

                all_angles_deg_stacked = np.stack([phi_angles, psi_angles, chi1_angles], axis=-1)
                all_angles_rad = np.deg2rad(all_angles_deg_stacked)
                all_features_dict[key] = self._transform_to_circular(all_angles_rad)
                print(f"      Extracted features (Zeros for missing). Shape: {all_features_dict[key].shape}")

            except Exception as e:
                print(f"    FATAL ERROR processing {key} with selection '{sel_string}': {e}")
                import traceback
                traceback.print_exc()
                continue

        return all_features_dict, n_frames