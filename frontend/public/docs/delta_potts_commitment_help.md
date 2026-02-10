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

## Commitment Modes (Visualization)

The analysis stores node marginals `p_i(a)` so the UI can show alternative per-residue views:

- **Base:** `Pr(Δh < 0)`
- **Centered:** `Pr(Δh ≤ median(ref))`
  - Uses a per-residue threshold computed from a selected *reference ensemble* so neutral residues appear closer to white.
- **Mean:** `sigmoid(-E[Δh]/scale)`
  - Smooth view based on the mean field difference per residue.

## Edge-Weighted Residue Coloring (Optional)

If enabled, the UI replaces each residue’s displayed value by a **blend** of:

- the per-residue commitment (chosen mode: Base / Centered / Mean-field), and
- the average commitment of its **incident top edges** (pair terms), weighted by `|ΔJ_ij|`.

This can make spatially coherent “patches” easier to see by propagating strong pair signals to nearby residues.
Note: this uses only the stored top edges (for performance), so residues with no selected incident edges will stay close to their original value.

## How To Interpret Centered Mode

Centered mode is designed to answer:

“Compared to a chosen reference ensemble, does this sample look more A-like or more B-like at each residue?”

Mechanically:

1. For each residue `i`, we look at the discrete set of values `Δh_i(a)` across states `a`.
2. From the reference ensemble we compute a **weighted median** threshold `t_i` (using the reference marginals `p_ref,i(a)`).
3. We then compute, for any sample, how much probability mass falls **below** that threshold (A-like) versus **above** it (B-like).
4. Because `Δh_i(a)` is discrete, many residues have ties at the median. We handle ties so that the **reference itself maps to ~0.5 (white)** for every residue by construction.

What the colors mean (centered mode):

- **White (~0.5)**: the sample is similar to the reference at that residue (no strong shift vs the reference).
- **Red (>0.5)**: compared to the reference, the sample shifts toward states with smaller `Δh_i` (more **A-like** at that residue).
- **Blue (<0.5)**: compared to the reference, the sample shifts toward states with larger `Δh_i` (more **B-like** at that residue).

### Example Workflow

Assume model **A** was fit on “MD Inactive” and model **B** was fit on “MD Active”.

1. Set reference = “MD Active”, and view “MD Active”.
   - You should see mostly **white** residues: this is expected because centered mode is calibrated so the reference appears neutral.
2. Keep the same reference (“MD Active”), and view “MD Inactive”.
   - Residues that turn **red** are residues whose marginals in “MD Inactive” shift toward states that are more **A-like (Inactive-like)** than the reference.
   - Residues that stay **white** do not change much between ensembles (or their changes are not aligned with `Δh_i`).
   - It is also possible for some residues to turn **blue** even if the global state is “Inactive” (real biological heterogeneity, model mismatch, or local compensation).

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
