# Metastable state discovery (webserver pipeline)

This page describes the metastable-state pipeline used by the webserver and how
the hyperparameters map to the code in `phase/analysis/vamp_pipeline.py`.

## Pipeline overview
1) Select one uploaded descriptor-ready state.
2) Load its per-residue descriptor NPZ file.
3) Flatten all residue features into a single feature matrix for that state only.
4) Standardize features and apply TICA to extract slow coordinates.
5) Cluster frames into microstates with k-means in TICA space.
6) Cluster microstate centers into metastable states and pick `k` by silhouette.
7) Save per-frame labels and representative structures for each metastable state under that state's folder.

Important: metastable discovery is never mixed across two uploaded states. If you
uploaded `Inactive` and `Active`, each one gets its own independent VAMP/TICA
run and its own derived metastable substates.

## Hyperparameters and their roles
- TICA lag (frames): `tica_lag_frames` sets the lag for the time-lagged covariance.
- TICA dims: `tica_dim` selects how many slow components are retained.
- Microstates (k-means): `n_microstates` controls the number of micro clusters.
- Metastable min/max k: `k_meta_min` and `k_meta_max` define the k range tested
  for the second clustering stage (silhouette score).
- Random seed: `random_state` controls k-means initialization.

## Notes on the descriptors
- The pipeline consumes the same per-residue dihedral features used elsewhere in
  PHASE (phi/psi/chi1 angles transformed to sin/cos).
- Features are stacked in residue order to form a single frame vector.

## Outputs
- Per-frame metastable labels stored alongside each descriptor NPZ.
- Representative PDBs for each metastable state when topology/trajectory files exist.
- Saved TICA and k-means models under `states/<state>/metastable/`.
- A per-state `metastable_metadata.json` file describing the derived substates.

## How metastable states are exposed later
- The original uploaded state remains a normal macro state.
- The discovered substates are exposed as additional derived states with kind
  `metastable`.
- They can be selected in visualization and downstream analyses, but they always
  keep a link to their parent uploaded state.

## Related docs
- [VAMP/TICA details and references](doc:vamp_tica)
- [Markov State Models (MSM)](doc:msm)
- [PCCA+ coarse graining](doc:pcca)
