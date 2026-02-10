# Delta Commitment (3D)

This page visualizes per-residue **commitment** on a 3D structure for a fixed pair of Potts models (A,B).

## Definition

Commitment is reported as:

`q_i = Pr(δ_i < 0)` where `δ_i(t) = h^A_i(s_{t,i}) - h^B_i(s_{t,i})`.

Interpretation:
- `q_i ≈ 0`: residue tends to favor model **B**.
- `q_i ≈ 1`: residue tends to favor model **A**.
- `q_i ≈ 0.5`: mixed / ambiguous (partial commitment).

## Coloring

The 3D overlay uses a diverging palette:
- blue: `q → 0`
- white: `q ≈ 0.5`
- red: `q → 1`

## Notes

- Currently, only the **top-K residues** (by `|D_i|`) are overlaid, to keep the visualization focused.
- Residue selection prefers PDB numbering (`auth_seq_id`) when labels include an integer (e.g. `res_279`); otherwise it falls back to sequential indices (`label_seq_id`).

