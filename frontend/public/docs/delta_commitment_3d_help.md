# Delta Commitment (3D)

This page visualizes **per-residue commitment** for a fixed pair of Potts models (A,B) on top of a 3D structure.

## What Is Being Colored

For each residue `i`, commitment is:

`q_i = Pr(δ_i < 0)` where `δ_i(t) = h^A_i(s_{t,i}) - h^B_i(s_{t,i})`.

Interpretation:
- `q_i ~ 0`: that residue tends to favor model **B** (δ_i mostly positive).
- `q_i ~ 1`: that residue tends to favor model **A** (δ_i mostly negative).
- `q_i ~ 0.5`: ambiguous / mixed signs (partial commitment).

## Colors

The residue overlay uses a diverging palette:
- **blue**: `q → 0`
- **white**: `q ≈ 0.5`
- **red**: `q → 1`

## Scope / Current Limitation

The overlay is applied to the **top-K residues** selected by the TS-like analysis (by `|D_i|`).
Residues outside the top-K are left as the gray base cartoon.

## Practical Notes

- The selection tries to match residues by PDB residue numbering (`auth_seq_id`) when residue labels contain integers (e.g. `res_279`).
  If not available, it falls back to sequential residue indices (`label_seq_id`).
- If you don’t see colored residues, verify you loaded a structure compatible with the cluster residue indexing.

