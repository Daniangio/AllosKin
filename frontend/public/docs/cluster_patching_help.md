# Residue Patch Clustering

This panel lets you preview an alternative clustering method on selected residues only, without immediately changing the main cluster.

## Workflow

1. Select a cluster in Descriptor Explorer.
2. Define patch parameters and click `Create preview patch`.
3. Switch between `Original cluster` and patch variants in `Cluster variant`.
4. If the patch is good, click `Confirm swap`.
5. If not, click `Discard patch`.

`Confirm swap` replaces canonical labels for patched residues and recomputes MD cluster memberships for the cluster.

## Hyperparameters

- `Patch residues`
  - Comma-separated residue keys, for example `res_120,res_121`.
  - Only these residues are re-clustered.

- `n_clusters (optional)`
  - Target number of clusters for patched residues.
  - If blank, the current residue cluster count is reused.
  - Used when `Cluster selection mode = fixed K (maxclust)`.

- `Cluster selection mode`
  - `fixed K (maxclust)`: forces at most `K` flat clusters from the linkage tree.
  - `auto by threshold (inconsistent)`: lets the algorithm determine flat clusters using SciPy `fcluster(..., criterion="inconsistent")`.

- `Inconsistent threshold`
  - Threshold `t` used by `fcluster(..., criterion="inconsistent")`.
  - Lower values generally produce more/smaller clusters; higher values merge more.
  - Used only when `Cluster selection mode = inconsistent`.

- `Inconsistent depth`
  - `depth` argument for inconsistency calculation in SciPy `fcluster`.
  - Default is `2`.
  - Used only when `Cluster selection mode = inconsistent`.

- `Max frames (optional)`
  - Maximum number of frames used to fit the alternative clustering model.
  - If blank, all frames are used.
  - This affects fitting only. Prediction is still applied to all frames.

- `Linkage`
  - Hierarchical clustering linkage method.
  - `ward` is usually stable when Euclidean geometry is appropriate.
  - `complete` and `average` can preserve separated groups better in some cases.
  - `single` can create chaining effects and is usually less robust.

- `Covariance`
  - Covariance model used by the frozen Gaussian assignment step.
  - `full`: full covariance matrix (more flexible, more parameters).
  - `diag`: diagonal covariance only (more regularized, fewer parameters).

- `Halo percentile`
  - Controls outlier/halo detection threshold.
  - Higher values mark more points as halo (`-1`).
  - Lower values are more permissive (fewer halo labels).

## What gets stored

- Preview patch labels are stored in cluster NPZ under patch-prefixed keys.
- Patch metadata stores:
  - algorithm and parameters
  - patched residues
  - predictions and halo summary for each state/metastable condition
- Confirmed patches are recorded in `patch_history`.

## Notes

- Preview patches are non-destructive until confirmed.
- Multiple previews can coexist and be compared.
- If a patch is confirmed, downstream analyses using cluster labels should be considered dependent on the new labeling.
