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

## Commitment Modes (Visualization)

The analysis stores per-sample node marginals `p_i(a)` and the parameter differences `Δh_i(a)` so the UI can show alternative views:

- **Base:** `q_i = Pr(Δh_i(X_i) < 0)`
- **Centered:** `q_i(ref) = Pr(Δh_i(X_i) ≤ median_ref(Δh_i))`
  - The reference set is chosen in the UI. This helps when ensembles have mixed/discrete marginals that can make the base probability look “committed” for residues that are effectively neutral.
- **Mean:** `q_i = sigmoid(-E[Δh_i(X_i)] / scale)`
  - A smoother map based on the mean field difference per residue.

## Edge-Weighted Residue Coloring (Optional)

If enabled, the visualization blends each residue’s per-residue value with the average commitment of its incident top edges,
weighted by `|ΔJ_ij|`. This can help highlight spatially coherent patches supported by pair couplings.

## How To Interpret Centered Mode

Centered mode is designed to answer:

“Compared to a chosen reference ensemble, does this sample look more A-like or more B-like at each residue?”

Mechanically:

1. For each residue `i`, we look at the discrete set of values `Δh_i(a)` across states `a`.
2. From the reference ensemble we compute a weighted-median threshold `t_i` using the reference marginals `p_ref,i(a)`.
3. For any sample, we compute how much probability mass falls below `t_i` (A-like) versus above `t_i` (B-like).
4. Because `Δh_i(a)` is discrete, ties at the median are common. We handle ties so that the reference itself maps to ~0.5 (white) per residue by construction.

What the colors mean (centered mode):

- White (~0.5): similar to the reference at that residue.
- Red (>0.5): more A-like than the reference at that residue (shift toward smaller `Δh_i`).
- Blue (<0.5): more B-like than the reference at that residue (shift toward larger `Δh_i`).

## Interpretation

- `q ≈ 1`: sample tends to occupy states where **A** is favored vs **B**.
- `q ≈ 0`: sample tends to occupy states where **B** is favored vs **A**.
- `q ≈ 0.5`: weak / mixed preference.

## What “red even more Inactive-like than Inactive” means concretely (in your current definition)

In your delta-commitment page, per-residue commitment is built from field differences only:

Δℎ𝑖(𝑎) = ℎ𝑖𝐴(𝑎) − ℎ𝑖𝐵(𝑎)

base mode: 𝑞𝑖 = Pr(Δℎ𝑖(𝑋𝑖)<0)
centered mode: you choose a reference ensemble, compute a weighted median threshold 𝑡𝑖 from that reference, and then report how much probability mass is on the “A side” vs the “B side” of that threshold; ties are handled so the reference maps to ~0.5 (white) by construction

So if you set:

model 𝐴 = Inactive, model 𝐵 = Active,
reference = Inactive,

and you look at some other trajectory, then a residue being red in centered mode means:

Relative to the Inactive reference median at that residue, this trajectory puts more probability mass on microstates with “more A-favored” Δℎ𝑖 values.

## Ranking (“Top Residues/Edges”)

We rank residues/edges by **parameter magnitude** (currently L2 norm of `Δh_i` and `ΔJ_ij`).
This ranking is sample-independent and is computed once per `(A,B,params)` analysis key.

In the UI, you can optionally filter the heatmaps to only show the top-K ranked residues/edges.

## ΔE Distributions

We compute `ΔE = E_A(X) − E_B(X)` and show density histograms using a shared binning across samples.
Negative ΔE indicates A-like frames; positive indicates B-like.
