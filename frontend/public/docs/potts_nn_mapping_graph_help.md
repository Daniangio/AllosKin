# Potts NN Mismatch Graph

This page aggregates multiple `potts_nn_mapping` analyses and renders mismatch on the Potts contact graph.

## What is aggregated

Each selected analysis already contains compact summaries:

- per-residue mean / std / median / q25 / q75
- per-edge mean / std / median / q25 / q75

These are derived from nearest-neighbor Potts mismatch against one MD target ensemble.

The graph page does **not** recompute nearest neighbors. It combines saved analysis summaries.

## Filters

You first choose a **compatible analysis group**. Analyses are grouped by:

- Potts model
- normalization flag
- `beta_node`
- `beta_edge`
- MD label mode
- invalid-row policy
- analysis format version

Only analyses from the same group should be aggregated together.

You can then filter by:

- source sampled trajectories
- target MD trajectories

## Important interpretation

Aggregating analyses over multiple target MD trajectories means:

- average mismatch across the selected targets

It does **not** mean:

- nearest-neighbor support in the union of those MD trajectories

That would require recomputing the NN analysis against the union.

## Statistics shown

For nodes and edges the page can show:

- `mean`
- `median`
- `std`
- `iqr = q75 - q25`

### Mean / std

When combining multiple analyses, mean and std are aggregated with weighted moments using the sample mass of each analysis.

### Median / q25 / q75

When combining multiple analyses, these are weighted averages of the saved per-analysis quantiles.

This is an approximation, not an exact pooled quantile.

## Graph coloring

The threshold slider maps mismatch to color saturation:

- `0` means neutral color
- `threshold` or larger means max color

Default threshold:

- `0.30`

In practice:

- lower threshold = more saturated graph
- higher threshold = only strong mismatch remains saturated

## Views

### Combined

- nodes: combined residue mismatch summary
- edges: edge mismatch summary

### Node

- nodes: node-only residue mismatch summary
- edges: neutral context only

### Edge

- edges: edge mismatch summary
- nodes: neutral context only

## Why only compact summaries are saved

Saving every per-frame or per-unique-row edge mismatch for every analysis would make visualization payloads too large.

Instead the analysis stores compact summaries that are enough to show:

- central tendency
- spread
- a robust interquartile range

This keeps the page responsive while still showing uncertainty.
