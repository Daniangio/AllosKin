"""
Handles calculation of dihedral angles and the mandatory
sin/cos transformation as defined in the project plan.

This implementation uses the robust `Residue.phi_selection()`,
`Residue.psi_selection()`, and `Residue.chi1_selection()` methods
to explicitly calculate dihedrals.

Refactored to robustly handle terminal residues (N-term missing phi,
C-term missing psi) by correctly slicing the results of the
batched dihedral calculation.
"""

import MDAnalysis as mda
import MDAnalysis.analysis.dihedrals as mda_dihedrals
import numpy as np
from typing import Dict, List, Optional

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
        
        Handles np.nan inputs correctly (np.sin(np.nan) -> np.nan).
        """
        sin_transformed = np.sin(angles_rad)
        cos_transformed = np.cos(angles_rad)
        
        n_frames, n_residues, n_angles = angles_rad.shape
        circular_features = np.empty((n_frames, n_residues, n_angles * 2))
        
        # Interleave them: [sin(phi), cos(phi), sin(psi), cos(psi), ...]
        circular_features[..., 0::2] = sin_transformed
        circular_features[..., 1::2] = cos_transformed
        
        return circular_features

    def _calculate_dihedral_angle(
        self, 
        atom_groups_list: List[Optional[mda.AtomGroup]], 
        n_frames: int
    ) -> np.ndarray:
        """
        Helper function to run Dihedral analysis on a list of AtomGroups.
        
        If an item in the list is None (e.g., phi for N-term, chi1 for GLY),
        it is skipped, and its corresponding output will be np.nan.
        
        Returns: (n_frames, len(atom_groups_list)) array of angles in degrees.
        """
        # Create a mask of valid (non-None) atom groups
        mask = np.array([ag is not None for ag in atom_groups_list])
        
        # Create the final angles array, initialized to nan
        angles = np.full((n_frames, len(atom_groups_list)), np.nan)
        
        # Filter the list to only valid AtomGroups
        valid_atom_groups = [ag for ag in atom_groups_list if ag is not None]
        
        if not valid_atom_groups:
            # No valid dihedrals to calculate
            return angles
            
        try:
            # Run analysis only on the valid groups
            analysis = mda_dihedrals.Dihedral(valid_atom_groups).run()
            # Place the results back into the 'angles' array using the mask
            angles[:, mask] = analysis.results.angles
        except Exception as e:
            print(f"    Warning: Dihedral calculation failed: {e}")
            # Return the nan-filled array
            
        return angles


    def extract_all_features(self, traj: TrajectoryObject) -> FeatureDict:
        """
        Extracts features for all defined residues, ensuring a consistent
        output shape by padding missing angles (phi, psi, chi1) with NaN.
        
        Returns: A dictionary {selection_key: feature_array}
                 where each feature_array has shape (n_frames, 1, 6)
                 corresponding to [sin(phi), cos(phi), sin(psi), cos(psi), sin(chi1), cos(chi1)].
                 Values for missing angles will be np.nan.
        """
        all_features_dict: FeatureDict = {}
        n_frames = len(traj.trajectory)

        # Define standard protein residues to filter out ACE, NME, etc.
        STANDARD_PROTEIN_RESNAMES = {
            "ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS", 
            "ILE", "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP", 
            "TYR", "VAL", 
            "HID", "HIE", "HSD", "HSE", "HSP", # Common HIS protonation states
            "CYX", # Common CYS protonation states
        }

        for key, sel_string in self.residue_selections.items():
            print(f"  Processing: {key} ({sel_string})...")
            
            try:
                user_residue_group = traj.select_atoms(sel_string)
                
                if len(user_residue_group.residues) == 0:
                    print(f"    Warning: Selection '{sel_string}' found no residues. Skipping.")
                    continue

                # Filter for standard protein resnames
                protein_residues = user_residue_group.residues[
                    np.isin(user_residue_group.residues.resnames, list(STANDARD_PROTEIN_RESNAMES))
                ]
                
                if len(protein_residues) == 0:
                    print(f"    Warning: Selection '{sel_string}' contains no standard protein residues (e.g., ACE, NME). Skipping.")
                    continue
                
                # 3. Get angle selections (lists of AtomGroup or None)
                phi_sel_list = protein_residues.residues.phi_selections()
                psi_sel_list = protein_residues.residues.psi_selections()
                chi1_sel_list = protein_residues.residues.chi1_selections()

                # 4. Store lengths for robust slicing
                n_phi = len(phi_sel_list)
                n_psi = len(psi_sel_list)
                n_chi1 = len(chi1_sel_list) # n_chi1 is correct, no '1'
                
                all_selections = phi_sel_list + psi_sel_list + chi1_sel_list
                
                if not all_selections:
                    print(f"    Warning: No valid dihedrals found for {key}. Skipping.")
                    continue

                # 5. Calculate all angles in one batch
                all_angles_deg = self._calculate_dihedral_angle(all_selections, n_frames)
                
                # 6. Robustly slice the results
                phi_angles = all_angles_deg[:, :n_phi]
                psi_angles = all_angles_deg[:, n_phi : n_phi + n_psi]
                chi1_angles = all_angles_deg[:, n_phi + n_psi :]

                # 7. Consolidate, Convert, and Transform
                
                # Stack all found angles: [phi, psi, chi1]
                # Note: np.stack creates a new axis.
                # (n_frames, n_residues_in_selection) -> (n_frames, n_residues_in_selection, 3)
                # Here, n_residues_in_selection is typically 1
                all_angles_deg_stacked = np.stack([phi_angles, psi_angles, chi1_angles], axis=-1)
                
                all_angles_rad = np.deg2rad(all_angles_deg_stacked)
                
                # Apply sin/cos transformation
                # (n_frames, 1, 3) -> (n_frames, 1, 6)
                all_features_dict[key] = self._transform_to_circular(all_angles_rad)
                print(f"    Extracted features (NaNs for missing). Shape: {all_features_dict[key].shape}")

            except Exception as e:
                print(f"  FATAL ERROR processing {key} with selection '{sel_string}': {e}")
                # This could be a complex error, so it's good to see the traceback
                import traceback
                traceback.print_exc()
                continue

        return all_features_dict