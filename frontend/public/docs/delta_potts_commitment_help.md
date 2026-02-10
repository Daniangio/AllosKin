# Delta Potts Commitment (A–B)

This page evaluates **how individual residues (and residue pairs) “prefer” model A vs model B** for a set of samples.

## Inputs

- Two Potts models: **A** and **B** (typically delta models).
- A set of samples (MD-eval or Potts-sampled) providing cluster-assignment labels per frame.

## Gauge (Important)

Before comparing parameters across models we enforce the same **zero-sum gauge** on both models.
This prevents gauge artifacts from appearing as “important residues”.

## What Is Commitment?

We form the parameter differences:

- **Δh_i(a) = h_i^A(a) − h_i^B(a)**
- **ΔJ_ij(a,b) = J_ij^A(a,b) − J_ij^B(a,b)**

Then, for each sample we compute:

- **Per-residue commitment**
  - `q_i = Pr( Δh_i(X_i) < 0 )`
- **Per-edge commitment**
  - `q_ij = Pr( ΔJ_ij(X_i, X_j) < 0 )`

Interpretation:

- `q ≈ 1`: the sample tends to pick states where A is favored (negative Δ term).
- `q ≈ 0`: the sample tends to pick states where B is favored (positive Δ term).
- `q ≈ 0.5`: no strong preference (or mixed occupancy).

## Heatmaps

- Rows: selected samples.
- Columns: all residues by default; optionally you can filter to top-ranked residues/edges in the UI.
- Colors: blue (0) → white (0.5) → red (1).

## ΔE Plot

We also compute `ΔE = E_A(X) − E_B(X)` per frame and show **density** histograms with a shared binning.

- Mostly negative ΔE: frames are more A-like.
- Mostly positive ΔE: frames are more B-like.
