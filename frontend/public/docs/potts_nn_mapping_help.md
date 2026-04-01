# Potts NN Mapping

## What this analysis answers

This analysis asks a specific question:

- For each unique cluster-label pattern in a sample ensemble, what is the closest pattern already seen in a reference MD ensemble?

It does **not** compare raw coordinates. It compares sequences of residue cluster labels under a chosen Potts model and contact graph.

## Inputs

You choose:

- a `Potts model`
- a `Sample to map`
- an `MD sample`

The sample to map can be any saved sample ensemble in the cluster. The MD sample is the reference ensemble that defines what counts as "already supported by MD".

## What is compressed before comparison

The analysis first compresses both ensembles to unique label rows.

For example, if the same cluster-label sequence appears 500 times in the sample, it is evaluated once and assigned a count of 500.

This makes the run much faster and is why parallelization is over **unique rows**, not over every original frame.

## Distance definition

For each unique sample row `s`, the analysis scans all unique MD rows `m` and finds the one with minimum weighted mismatch.

The distance has three layers.

### Node distance

The node part checks residue-by-residue label disagreement, weighted by how different the Potts local fields are for the two labels.

Intuition:

- changing a residue between two labels that the model sees as almost equivalent contributes little
- changing a residue between two labels with very different field terms contributes more

### Edge distance

The edge part checks pairwise disagreement on the Potts contact edges, weighted by how different the Potts coupling terms are for the two pair states.

Intuition:

- mismatching a weak or nearly equivalent pair contributes little
- mismatching a strongly discriminative pair contributes more

### Global distance

The final nearest-neighbor distance is:

`global = beta_node * node + beta_edge * edge`

The nearest MD row is the one with the smallest global distance.

## Normalization

If `Normalize` is enabled, node and edge mismatches are scaled by the largest possible Potts gap for that residue or edge.

Interpretation:

- values closer to `0` mean the sample row is well supported by some MD row
- larger values mean the sample row is less compatible with the MD support already present in the reference ensemble

Normalization makes distances easier to compare across models and systems, but they are still model-dependent quantities.

## What each plot shows

## Weighted nearest-neighbor distance histogram

Each bar summarizes the distance distribution across unique sample rows.

If `Use unique` is enabled, the histogram is weighted by how many original frames each unique row represents.

Use it to answer:

- Is most of the sample close to the MD ensemble?
- Is there a long tail of unsupported configurations?

The dashed and dotted lines show weighted median and upper-tail reference values.

## Unique sample rows ordered by nearest distance

This plot sorts all unique sample rows from closest to farthest.

Hover shows:

- the original unique-row index
- how many frames it represents
- the nearest MD unique row
- the nearest MD representative frame
- node and edge distance parts

Use it to answer:

- Which sample motifs are most MD-like?
- Which motifs are outliers?
- Is the mismatch concentrated in a few rare rows or spread across the ensemble?

Clicking a point selects that unique sample row below.

## Threshold coverage

For each threshold `d`, the UI reports the fraction of the sample mass with nearest-neighbor distance `<= d`. These thresholds are viewer-side cutoffs; they do not affect the nearest-neighbor computation itself.

Use it as a compact coverage score.

Example:

- `d <= 0.10 : 82%` means 82% of the sample mass has a close MD neighbor under that cutoff.

## Top mean mismatched residues

This is the residue-level average mismatch across the full sample ensemble.

Large values mean that residue consistently contributes to why the sample differs from the MD reference.

Newer analyses also persist compact spread summaries:

- `std`
- `median`
- `q25`
- `q75`

These are used by the dedicated **Potts NN Mismatch Graph** page.

## Selected unique sample

This panel focuses on one unique sample row.

It shows:

- global distance to its nearest MD row
- node-only contribution
- edge-only contribution
- the nearest MD representative frame index

## Top mismatched residues for selected unique sample

This plot explains **why that one sample row** is far or close.

It shows:

- `Combined`: `(1 - alpha) * node + alpha * edge`
- `Node`: residue-local mismatch only
- `Edge`: mismatch inherited from edges touching that residue

Important:

- `alpha` here changes only the **visual blend** for residue attribution
- it does **not** recompute the nearest MD neighbor
- the nearest neighbor was already chosen using `beta_node` and `beta_edge` during analysis

## Alpha vs beta

These are different knobs.

- `beta_node`, `beta_edge`
  - used during the analysis
  - control how node and edge distances are weighted when selecting the nearest MD row
- `alpha`
  - used only in the viewer
  - controls how node and edge residue attributions are blended in the plots

## When to trust the result

This analysis is useful when you want to know whether a sampled ensemble stays within the support of a known MD ensemble in cluster space.

It is less appropriate if you need:

- exact geometric RMSD-like comparisons
- a thermodynamic free-energy statement
- a direct state classifier by itself

A low distance means "supported by an MD-neighbor under this Potts-weighted label metric", not "identical structure".
