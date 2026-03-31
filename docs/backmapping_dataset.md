# Backmapping Dataset

This dataset is built per `md_eval` sample.

## What it stores

- `trajectory`: `(n_frames, n_atoms, 3)` selected atom coordinates for the uploaded trajectory, restricted to the sample frames used in the assigned labels
- `atom_resids`: `(n_atoms,)` residue id for each selected atom
- `atom_names`: `(n_atoms,)` atom name for each selected atom
- `atom_residue_index`: `(n_atoms,)` residue-row index used by the cluster label arrays
- `residue_cluster_ids`: `(n_frames, n_residues)` assigned cluster id for each residue in each sample frame
- `residue_cluster_counts`: `(n_residues,)` number of clusters available for each residue
- `dihedrals`: `(n_frames, n_residues, n_dihedrals)` raw dihedral values in radians
- `dihedral_keys`: names of the dihedral dimensions, currently `phi`, `psi`, `omega`, `chi1`, `chi2`
- `residue_keys`: cluster residue keys in the same order as `residue_cluster_ids`
- `frame_indices`: original frame indices used from the uploaded trajectory
- `frame_state_ids`: source state id for each kept frame

## Webserver flow

1. Go to the system page and select a cluster.
2. In the `From MD` sample list, click the upload icon next to the target sample.
3. Upload the trajectory corresponding to that MD state.
4. Submit the job.
5. When the job finishes, the sample row shows a download action.

Current limitation:

- only `md_eval` samples tied to one concrete `state_id` are supported
- mixed-state or metastable aggregate MD samples are not supported for this dataset yet

## Phase Console

Use:

```bash
./scripts/backmapping_dataset.sh
```

The script asks for:

- cluster
- `md_eval` sample
- trajectory path

It writes `backmapping_dataset.npz` into the selected sample folder and updates that sample's metadata.
