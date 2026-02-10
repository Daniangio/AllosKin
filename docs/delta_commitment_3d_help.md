# Delta Commitment (3D)

This page visualizes per-residue **commitment** on a 3D structure for a fixed pair of Potts models (A,B).

## Definition

Commitment is reported as a per-residue quantity `q_i` derived from the Potts field difference:

`q_i = Pr(δ_i < 0)` where `δ_i(t) = h^A_i(s_{t,i}) - h^B_i(s_{t,i})`.

Interpretation:
- `q_i ≈ 0`: residue tends to favor model **B**.
- `q_i ≈ 1`: residue tends to favor model **A**.
- `q_i ≈ 0.5`: mixed / ambiguous (partial commitment).

## Commitment Modes

You can switch between multiple visualization modes:

- **Base: `Pr(Δh < 0)`**
  - The direct definition above.
- **Centered: `Pr(Δh ≤ median(ref))`**
  - Computes a per-residue threshold from a *reference ensemble* (selected in the UI).
  - This is useful when ensembles have **mixed marginals** (e.g., discrete mixtures) that can make the base `Pr(Δh<0)` look “committed” even for intuitively neutral residues.
  - In centered mode, the reference set tends to appear closer to white (`≈0.5`) by construction.
- **Mean: `sigmoid(-E[Δh]/scale)`**
  - Uses the mean field difference per residue and maps it into `[0,1]` with a sigmoid for a smoother view.

### Reading Centered Mode

Centered mode is best interpreted as a **reference-normalized** view:

- Pick a reference ensemble (often an MD ensemble).
- In centered mode, the reference is calibrated to look ~white per residue.
- When you switch to another ensemble while keeping the same reference:
  - residues turning red indicate that ensemble is more A-like than the reference at those residues
  - residues turning blue indicate it is more B-like than the reference at those residues

## Coloring

The 3D overlay uses a diverging palette:
- blue: `q → 0`
- white: `q ≈ 0.5`
- red: `q → 1`

## Coupling Links (Optional)

If enabled, the page draws **line links** for the *top Potts edges* (by parameter magnitude). Link colors follow the same palette as residues.

## Edge-Weighted Residue Coloring (Optional)

If enabled, residue colors are blended with the average edge commitment of incident top edges (weighted by `|ΔJ|`).
This can make coupling-supported patches easier to see on the structure.

## Notes

- The base cartoon is gray and the commitment is applied via **overpaint** (so colors should update immediately when you change selections).
- Residue selection prefers PDB numbering (`auth_seq_id`) when labels include an integer (e.g. `res_279`); otherwise it falls back to sequential indices (`label_seq_id`).
