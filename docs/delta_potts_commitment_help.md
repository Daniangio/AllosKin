# Delta Potts Commitment (A–B)

This document mirrors the web UI help panel for the **Delta Potts Evaluation** page.

## Goal

Given two Potts models **A** and **B** (typically delta fits), and a set of samples (MD-eval or Potts-sampled),
compute interpretable per-residue and per-edge “commitment” signals, plus an overall ΔE distribution.

## Gauge (Non-Negotiable)

Before comparing parameters across two models, apply the same gauge to both (we use **zero-sum**):

- For each residue `i`: `sum_a h_i(a) = 0`
- For each edge `(i,j)`: `sum_a J_ij(a,b) = 0` and `sum_b J_ij(a,b) = 0`

This avoids manufacturing “important residues” due to gauge artifacts.

## Definitions

Let:

- `Δh_i(a) = h_i^A(a) − h_i^B(a)`
- `ΔJ_ij(a,b) = J_ij^A(a,b) − J_ij^B(a,b)`

For a sample producing per-frame labels `X`:

- Per-residue commitment: `q_i = Pr( Δh_i(X_i) < 0 )`
- Per-edge commitment: `q_ij = Pr( ΔJ_ij(X_i, X_j) < 0 )`

## Interpretation

- `q ≈ 1`: sample tends to occupy states where **A** is favored vs **B**.
- `q ≈ 0`: sample tends to occupy states where **B** is favored vs **A**.
- `q ≈ 0.5`: weak / mixed preference.

## Ranking (“Top Residues/Edges”)

We rank residues/edges by **parameter magnitude** (currently L2 norm of `Δh_i` and `ΔJ_ij`).
This ranking is sample-independent and is computed once per `(A,B,params)` analysis key.

In the UI, you can optionally filter the heatmaps to only show the top-K ranked residues/edges.

## ΔE Distributions

We compute `ΔE = E_A(X) − E_B(X)` and show density histograms using a shared binning across samples.
Negative ΔE indicates A-like frames; positive indicates B-like.
